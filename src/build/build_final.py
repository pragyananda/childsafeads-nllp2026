"""Build the final submission from components, each switched on by an explicit flag.

Nothing here is on by default that has not survived channel-grouped held-out
validation. The protocol that matters is consistency across splits, not effect
size: `no_flag` (10/10 splits) transferred to test, the ST1 logit adjustment
(7/10) did not, and the LLM hybrid (never held-out tested) did not.

Component status, test-measured where a submission spent on it:

  ST2 per-label thresholds   +0.060 +- 0.033, 30/30 splits, worst +0.011  NEVER SUBMITTED
  no_flag t=0.6              +0.021 on test                               shipped
  product gates              +0.004 ST3, never below ungated in 60 halves  not yet shipped
  ST1 class-weighted vote    dev +0.046; held-out -0.005 +- 0.088 (unresolvable)
  ST1 logit adjustment tau   dev +0.10, TEST -0.041                        rejected
  LLM exhortation hybrid     dev +0.014, TEST -0.005                       rejected
  ST3 thresholds (per-label) -0.037, 3/30 splits                           rejected
  ST3 thresholds (joint)     -0.034 +- 0.041, 3/20 splits                  rejected
  ST1 prior from ST2 set     -0.085, 0/10 splits                           rejected

Usage:
    python build_final.py --st2-thresholds work/th2_dev.npy --product-gates \
        --st1-vote --out work/SUBMIT_final_v2.jsonl
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, clean_st3, _g
from rules import is_short
import submit

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "work"

P = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
P.add_argument("--target", default="test", choices=["dev", "test"])
P.add_argument("--st1-pattern", default="probs_{t}_L4_ModernBERT-large_len2048_seed{s}_w5.npz",
               help="ST1 probability files. The _w5 runs train the ST1 head with class "
                    "weights, which is what makes `none` reachable at all.")
P.add_argument("--st1-seeds", type=int, nargs="+", default=[0, 1, 2, 4, 5])
P.add_argument("--st2-pattern", default="probs_{t}_L4_ModernBERT-large_len2048_seed{s}.npz")
P.add_argument("--st2-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
P.add_argument("--st3-pattern", default="probs_{t}_L1_ModernBERT-large_len1024_seed{s}.npz",
               help="ST3 reads transcript-only text: the product page costs it ~0.06.")
P.add_argument("--st3-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
P.add_argument("--st1-vote", action="store_true",
               help="hard vote across class-weighted ST1 models, mean probability breaking "
                    "ties, instead of argmax of the averaged probability. Averaging leaves a "
                    "1.5%%-prior class unable to win an argmax: `none` is predicted twice on "
                    "dev where fourteen exist.")
P.add_argument("--st2-thresholds", default=None,
               help="path to the per-label ST2 threshold vector from st2_thresholds.py; "
                    "omit for a fixed 0.5 cutoff")
P.add_argument("--st3-threshold", type=float, default=0.5,
               help="flat ST3 cutoff. Tuning these does not transfer, per-label or joint.")
P.add_argument("--no-flag-t", type=float, default=0.6,
               help="emit no_flag alone above this probability; 0 disables")
P.add_argument("--product-gates", action="store_true",
               help="suppress the two product-family ST3 flags when the predicted ST2 "
                    "category cannot support them")
P.add_argument("--ar-gate-categories", nargs="+",
               default=["food", "gambling", "gambling_adjacent", "health", "other"],
               help="ST2 categories that can carry age_restricted_or_prohibited_product; "
                    "covers 55/59 of the flag in train")
P.add_argument("--llm", default=None,
               help="path to an LLM prediction file whose direct_exhortation decision "
                    "replaces the encoder's. Measured at -0.005 on test; left off.")
P.add_argument("--compare-to", default=None, help="an existing submission to diff against")
P.add_argument("--out", required=True)
A = P.parse_args()

insts = load_split(A.target)
ix = {x["instanceID"]: i for i, x in enumerate(insts)}


def stack(pattern, seeds, key, ncol):
    """Per-seed probability matrices, aligned to `insts` order."""
    out, used = [], []
    for s in seeds:
        f = W / pattern.format(t=A.target, s=s)
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        p = np.zeros((len(insts), ncol))
        p[[ix[i] for i in z["ids"].tolist()]] = z[key]
        out.append(p)
        used.append(s)
    if not out:
        sys.exit(f"no probability files matched {pattern.format(t=A.target, s='*')}")
    print(f"  {key}: seeds {used} from {pattern.format(t=A.target, s='*')}")
    return out


s1 = stack(A.st1_pattern, A.st1_seeds, "p1", len(ST1))
s2 = stack(A.st2_pattern, A.st2_seeds, "p2", len(ST2))
s3 = stack(A.st3_pattern, A.st3_seeds, "p3", len(ST3))
p1, p2, p3 = sum(s1) / len(s1), sum(s2) / len(s2), sum(s3) / len(s3)

th2 = np.load(A.st2_thresholds) if A.st2_thresholds else np.full(len(ST2), 0.5)
if A.st2_thresholds:
    print(f"  ST2 thresholds: {dict(zip(ST2, th2.round(2)))}")
llm = json.load(open(A.llm)) if A.llm else None
if llm:
    print(f"  LLM exhortation override: {len(llm)} instances")

NF = ST3.index("no_flag")
UA, IA = ST3.index("undisclosed_advertising"), ST3.index("inadequate_disclosure")
AR_GATE = set(A.ar_gate_categories)

preds = {}
for i, x in enumerate(insts):
    if A.st1_vote:
        c = collections.Counter(ST1[int(np.argmax(p[i]))] for p in s1)
        top = max(c.values())
        tied = [k for k, v in c.items() if v == top]
        st1 = tied[0] if len(tied) == 1 else max(tied, key=lambda k: p1[i][ST1.index(k)])
    else:
        st1 = ST1[int(np.argmax(p1[i]))]

    # Fallback takes the largest margin over threshold, not the largest raw
    # probability: with per-label thresholds the two differ.
    st2 = [c for j, c in enumerate(ST2) if p2[i, j] >= th2[j]] \
          or [ST2[int(np.argmax(p2[i] - th2))]]

    st3 = set(c for j, c in enumerate(ST3) if p3[i, j] >= A.st3_threshold)
    if A.product_gates:
        # The flag names a property of the product, which the transcript-only
        # ST3 head never saw.
        if "food" not in st2:
            st3.discard("hfss_food_marketing")
        if not (set(st2) & AR_GATE):
            st3.discard("age_restricted_or_prohibited_product")
    if _g(x, "video_context", "official_disclosure") == "true":
        st3 -= {"undisclosed_advertising"}
    if {"undisclosed_advertising", "inadequate_disclosure"} <= st3:
        st3 -= {"inadequate_disclosure" if p3[i, UA] >= p3[i, IA] else "undisclosed_advertising"}
    if is_short(x):
        st3 = {"insufficient_context"}
    elif A.no_flag_t and p3[i, NF] >= A.no_flag_t:
        st3 = {"no_flag"}
    elif llm is not None:
        st3.discard("direct_exhortation")
        if "direct_exhortation" in llm.get(x["instanceID"], []):
            st3.add("direct_exhortation")
    preds[x["instanceID"]] = {"st1": st1, "st2": st2, "st3": clean_st3(sorted(st3))}

out = Path(A.out) if Path(A.out).is_absolute() else ROOT / A.out
submit.write(preds, out, insts)

if A.target == "dev":
    import metrics
    metrics.show("dev", metrics.score(insts, preds))

if A.compare_to:
    prev = {json.loads(l)["instanceID"]: json.loads(l)
            for l in open(ROOT / A.compare_to if not Path(A.compare_to).is_absolute()
                          else A.compare_to)}
    print(f"\nvs {A.compare_to}:")
    for k in ("st1", "st2", "st3"):
        d = sum((preds[i][k] != prev[i][k]) if k == "st1" else (set(preds[i][k]) != set(prev[i][k]))
                for i in preds if i in prev)
        print(f"  {k} changed: {d}/{len(preds)}")
    print(f"  ST2 labels/instance: {sum(len(v['st2']) for v in prev.values())/len(prev):.2f}"
          f" -> {sum(len(v['st2']) for v in preds.values())/len(preds):.2f}")
