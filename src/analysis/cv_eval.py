"""Score configurations on pooled out-of-fold predictions.

Every configuration is evaluated on the same 2,857 instances (train+dev, five
channel-grouped folds), which is 5.7x the dev set we previously selected on and
carries ~5x as many of every rare class. Per-fold scores are also reported: the
lesson from the evaluation phase is that consistency across resampled splits
predicts transfer while effect size does not, so a configuration that wins on
the pooled number but not across folds is not a candidate.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, clean_st3, _g
from rules import is_short
import metrics

W = Path(__file__).resolve().parents[2] / "work" / "cv"
pool = load_split("train") + load_split("dev")
by_id = {x["instanceID"]: x for x in pool}

groups = defaultdict(dict)          # config -> instanceID -> (p1,p2,p3)
folds = defaultdict(lambda: defaultdict(list))   # config -> fold -> ids
for f in sorted(W.glob("oof_*.npz")):
    name = f.name
    fold = int(name.split("_f")[1].split("of")[0])
    cfg = name.split("_f")[0] + name.split("of5")[1].replace(".npz", "")
    z = np.load(f, allow_pickle=True)
    for k, i in enumerate(z["ids"].tolist()):
        groups[cfg][i] = (z["p1"][k], z["p2"][k], z["p3"][k])
        folds[cfg][fold].append(i)


def decide(ids, probs, no_flag_t=0.6):
    preds = {}
    for i in ids:
        x = by_id[i]
        p1, p2, p3 = probs[i]
        st2 = [c for j, c in enumerate(ST2) if p2[j] >= .5] or [ST2[int(np.argmax(p2))]]
        st3 = set(c for j, c in enumerate(ST3) if p3[j] >= .5)
        if _g(x, "video_context", "official_disclosure") == "true":
            st3 -= {"undisclosed_advertising"}
        if is_short(x):
            st3 = {"insufficient_context"}
        elif no_flag_t and p3[ST3.index("no_flag")] >= no_flag_t:
            st3 = {"no_flag"}
        preds[i] = {"st1": ST1[int(np.argmax(p1))], "st2": st2, "st3": clean_st3(sorted(st3))}
    return preds


print(f"{'configuration':<44}{'n':>6}{'ST1':>7}{'ST2':>7}{'ST3':>7}{'mean':>8}")
print("-" * 79)
results = {}
for cfg in sorted(groups):
    ids = list(groups[cfg])
    s = metrics.score([by_id[i] for i in ids], decide(ids, groups[cfg]))
    results[cfg] = (s, ids)
    print(f"{cfg:<44}{len(ids):>6}{s['st1']:>7.3f}{s['st2']:>7.3f}{s['st3']:>7.3f}{s['mean']:>8.3f}")

print("\nPer-fold, for consistency (the signal that predicted transfer):")
print(f"{'configuration':<44}" + "".join(f"{'f'+str(k):>8}" for k in range(5)))
print("-" * 84)
perfold = {}
for cfg in sorted(groups):
    row, vals = "", []
    for k in range(5):
        ids = folds[cfg][k]
        v = metrics.score([by_id[i] for i in ids], decide(ids, groups[cfg]))["mean"]
        vals.append(v); row += f"{v:>8.3f}"
    perfold[cfg] = vals
    print(f"{cfg:<44}{row}")

base = [c for c in sorted(groups) if c.endswith("seed0")]
print("\nPaired comparisons against the matching baseline, fold by fold:")
for cfg in sorted(groups):
    if cfg.endswith("seed0"):
        continue
    stem = "L1" if "_L1_" in cfg else "L4"
    b = [c for c in base if f"_{stem}_" in c]
    if not b:
        continue
    d = [a - c for a, c in zip(perfold[cfg], perfold[b[0]])]
    print(f"  {cfg:<42} {np.mean(d):+.3f}  sd {np.std(d, ddof=1):.3f}  "
          f"{sum(x > 0 for x in d)}/5 folds")
