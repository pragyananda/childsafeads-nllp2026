"""Evaluate the system the way an enforcement authority would, not the way the
leaderboard does.

Three analyses the shared task asks for in prose but does not score:

1. CASCADE. Our ablation says compliance flagging works at level 1 and product
   categorisation needs level 4. The system that follows is not "run level 4 on
   everything" -- it is screen cheaply, then pay for the crawl only where it
   changes an answer. This sweeps the crawl budget from 0% to 100% and reports
   what each percentage point buys.

2. OPERATING POINTS. Macro-F1 is not an authority's currency. What determines
   the human review burden is precision at a fixed recall on the practices that
   are prohibited outright.

3. SEVERITY. The taxonomy grades flags per_se / conditional / soft_law. Macro-F1
   treats a blacklisted practice under UCPD Annex I as exactly as important as a
   soft-law HFSS case. A monitoring system should not.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, clean_st3, _g
from rules import is_short
import metrics

W = Path(__file__).resolve().parents[2] / "work"
dv = load_split("dev")
order = {x["instanceID"]: i for i, x in enumerate(dv)}
N = len(dv)

SEVERITY = {"undisclosed_advertising": "per_se", "direct_exhortation": "per_se",
            "inadequate_disclosure": "conditional", "misleading_claim": "conditional",
            "age_restricted_or_prohibited_product": "conditional",
            "hfss_food_marketing": "soft_law"}
SEV_WEIGHT = {"per_se": 3.0, "conditional": 2.0, "soft_law": 1.0}


def avg(pat, seeds=(0, 1, 2, 3, 4, 5)):
    p1 = np.zeros((N, 5)); p2 = np.zeros((N, 12)); p3 = np.zeros((N, 8)); n = 0
    for s in seeds:
        f = W / pat.format(s=s)
        if f.exists():
            z = np.load(f, allow_pickle=True)
            idx = [order[i] for i in z["ids"].tolist()]
            p1[idx] += z["p1"]; p2[idx] += z["p2"]; p3[idx] += z["p3"]; n += 1
    assert n, pat
    return p1 / n, p2 / n, p3 / n


c1, c2, c3 = avg("probs_dev_L1_ModernBERT-large_len1024_seed{s}.npz")   # cheap tier
x1, x2, x3 = avg("probs_dev_L4_ModernBERT-large_len2048_seed{s}.npz")   # crawled tier

gold3 = [x["labels"]["st3"] for x in dv]
MLB3 = MultiLabelBinarizer(classes=ST3).fit(gold3)
Y3 = MLB3.transform(gold3)


def build(p1, p2, p3_src):
    """Predictions given per-instance choices of which tier answers ST1/ST2."""
    preds = {}
    for i, x in enumerate(dv):
        s2 = [c for j, c in enumerate(ST2) if p2[i, j] >= .5] or [ST2[int(np.argmax(p2[i]))]]
        s3 = set(c for j, c in enumerate(ST3) if p3_src[i, j] >= .5)
        if _g(x, "video_context", "official_disclosure") == "true":
            s3 -= {"undisclosed_advertising"}
        if is_short(x):
            s3 = {"insufficient_context"}
        preds[x["instanceID"]] = {"st1": ST1[int(np.argmax(p1[i]))], "st2": s2,
                                  "st3": clean_st3(sorted(s3))}
    return preds


print("=" * 78)
print("1. CASCADE: what fraction of the corpus actually needs the crawl?")
print("=" * 78)
print("ST3 always answered at level 1. ST1/ST2 answered at level 1 unless the")
print("instance is escalated, in which case the crawled model answers instead.\n")

# Escalation priority: how uncertain the cheap model is about ST1/ST2.
# Margin between top-2 ST1 classes, plus distance of ST2 scores from the boundary.
st1_margin = np.sort(c1, axis=1)[:, -1] - np.sort(c1, axis=1)[:, -2]
st2_doubt = -np.abs(c2 - 0.5).min(axis=1)
uncertainty = (-st1_margin) + st2_doubt
flagged = np.array([1.0 if set(clean_st3(sorted(
    [c for j, c in enumerate(ST3) if c3[i, j] >= .5]))) != {"no_flag"} else 0.0
    for i in range(N)])
rng = np.random.default_rng(0)

policies = {
    "uncertainty of cheap tier": uncertainty,
    "compliance-flagged first": flagged + 1e-6 * uncertainty,
    "random (control)": rng.random(N),
}

print(f"{'crawl budget':>13} | " + " | ".join(f"{k:>25}" for k in policies))
print("-" * 78)
rows = {k: [] for k in policies}
for budget in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    k_esc = int(round(budget * N))
    line = f"{budget:>12.0%} | "
    cells = []
    for name, score in policies.items():
        esc = set(np.argsort(-score)[:k_esc].tolist())
        p1 = np.array([x1[i] if i in esc else c1[i] for i in range(N)])
        p2 = np.array([x2[i] if i in esc else c2[i] for i in range(N)])
        s = metrics.score(dv, build(p1, p2, c3))
        rows[name].append(s["mean"])
        cells.append(f"{s['mean']:>25.3f}")
    print(line + " | ".join(cells))

full = rows["uncertainty of cheap tier"][-1]
zero = rows["uncertainty of cheap tier"][0]
for b, v in zip([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0], rows["uncertainty of cheap tier"]):
    if v >= zero + 0.9 * (full - zero):
        print(f"\n-> {b:.0%} of the corpus crawled retains {(v - zero) / (full - zero):.0%} "
              f"of the total gain the full crawl provides ({v:.3f} vs {full:.3f}).")
        break

print("\n" + "=" * 78)
print("2. OPERATING POINTS: review burden for the per se prohibitions")
print("=" * 78)
print("UCPD Annex I practices are prohibited outright, so an authority sets a")
print("recall target and lives with the precision. Level-1 model, dev set.\n")
print(f"{'flag':<26} {'prevalence':>10} {'recall':>7} {'precision':>10} {'queue/1000':>11}")
print("-" * 78)
for flag in ["undisclosed_advertising", "direct_exhortation"]:
    j = ST3.index(flag)
    y = Y3[:, j]
    prec, rec, _ = precision_recall_curve(y, c3[:, j])
    for target in (0.80, 0.90, 0.95):
        ok = rec >= target
        p_at = prec[ok].max() if ok.any() else 0.0
        # Segments a reviewer must open per 1000 screened to hit this recall.
        queue = (y.mean() * target / p_at * 1000) if p_at > 0 else float("inf")
        print(f"{flag if target == 0.80 else '':<26} {y.mean():>9.1%} {target:>7.0%} "
              f"{p_at:>10.2f} {queue:>11.0f}")

print("\n" + "=" * 78)
print("3. SEVERITY-WEIGHTED SCORE")
print("=" * 78)
print("Macro-F1 weights all eight flags equally. Weighting by how the law")
print("operates (per_se 3, conditional 2, soft_law 1):\n")
preds_l1 = build(c1, c2, c3)
preds_l4 = build(x1, x2, x3)
for name, pr in [("ST3 from level 1", preds_l1), ("ST3 from level 4", preds_l4)]:
    P3 = MLB3.transform([pr[x["instanceID"]]["st3"] for x in dv])
    per = f1_score(Y3, P3, average=None, zero_division=0)
    plain = np.mean([per[ST3.index(f)] for f in SEVERITY])
    wts = np.array([SEV_WEIGHT[SEVERITY[f]] for f in SEVERITY])
    vals = np.array([per[ST3.index(f)] for f in SEVERITY])
    print(f"  {name}:  unweighted {plain:.3f}   severity-weighted {np.average(vals, weights=wts):.3f}")
    for f in SEVERITY:
        print(f"      {f:<38} {SEVERITY[f]:<12} F1={per[ST3.index(f)]:.2f}")
    print()
