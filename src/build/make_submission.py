"""Build the evaluation-phase submission from cached probabilities.

Routing: ST1 and ST2 from the level-4 ensemble, ST3 from the transcript-only
ensemble. Averaged over 3 seeds per config, because run-to-run spread on a fixed
config reaches 0.089 on ST1 and selecting the best run banks that noise.

Thresholds are fitted on dev and applied to test. Dev is channel-disjoint from
test and was not trained on, so this is honest calibration -- unlike fitting on
dev and then *scoring* on dev, which inflates by ~0.05.

Both splits' probabilities must come from the same training runs, or the
thresholds describe a different model than the one being calibrated.

Usage: python make_submission.py [--target test|dev]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, clean_st3, _g
from rules import is_short
import metrics
import submit

P = argparse.ArgumentParser()
P.add_argument("--target", default="test", choices=["test", "dev"])
P.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
P.add_argument("--out", default=None)
A = P.parse_args()

W = Path(__file__).resolve().parents[2] / "work"
ST1ST2_CFG = "L4_ModernBERT-large_len2048"   # wins ST1 (0.666) and ST2 (0.753)
ST3_CFG = "L1_ModernBERT-large_len1024"      # wins ST3 (0.439), and is level 1

dev = load_split("dev")
target = load_split(A.target)

# ST1 prior correction. The ST1 head is trained with plain cross-entropy while
# ST2/ST3 get pos_weight, so nothing compensates for `none` being 1.5% of
# training data: the model predicted it twice in 504 dev instances against 14
# gold, for F1 0.250. Dividing by the class prior raised to tau restores it to
# 0.696 and ST1 macro from 0.706 to 0.824. tau=0.2 is also the value that makes
# the predicted `none` rate match the training prior (9 predictions against the
# 7.5 the prior implies), so it is not merely fitted to 14 dev instances.
TAU = 0.2
_cnt = Counter(x["labels"]["st1"] for x in load_split("train"))
PRIOR = np.maximum(np.array([_cnt[c] for c in ST1], dtype=float) / sum(_cnt.values()), 1e-6)


def adjust_st1(p1):
    return p1 / np.power(PRIOR, TAU)


def average(split, cfg, insts):
    order = {x["instanceID"]: i for i, x in enumerate(insts)}
    p1 = np.zeros((len(insts), len(ST1)))
    p2 = np.zeros((len(insts), len(ST2)))
    p3 = np.zeros((len(insts), len(ST3)))
    n = 0
    for s in A.seeds:
        f = W / f"probs_{split}_{cfg}_seed{s}.npz"
        if not f.exists():
            print(f"  !! missing {f.name}")
            continue
        z = np.load(f, allow_pickle=True)
        idx = [order[i] for i in z["ids"].tolist()]
        p1[idx] += z["p1"]; p2[idx] += z["p2"]; p3[idx] += z["p3"]
        n += 1
    if n == 0:
        raise SystemExit(f"no probability files for {split}/{cfg}")
    print(f"  {split}/{cfg}: averaged {n} seeds")
    return p1 / n, p2 / n, p3 / n


# --- fit thresholds on dev, from the same models that will predict target ---
d_a1, d_a2, d_a3 = average("dev", ST1ST2_CFG, dev)
d_b1, d_b2, d_b3 = average("dev", ST3_CFG, dev)
Y2 = MultiLabelBinarizer(classes=ST2).fit_transform([x["labels"]["st2"] for x in dev])
Y3 = MultiLabelBinarizer(classes=ST3).fit_transform([x["labels"]["st3"] for x in dev])
th2 = metrics.tune_thresholds(d_a2, Y2)   # ST2 has no exclusivity: separable
# ST3 is not separable -- no_flag and insufficient_context are exclusive of all
# other flags, so per-label tuning crushes no_flag (F1 0.217, predicted at 7.7%
# against 25.2% gold). Coordinate ascent on the real objective lifts it to 0.394
# and ST3 macro from 0.527 to 0.545.
from joint_tune import coordinate_ascent
th3, _ = coordinate_ascent(d_b3, dev, [x["labels"]["st3"] for x in dev],
                           th0=metrics.tune_thresholds(d_b3, Y3))


def build(a1, a2, b3, insts):
    preds = {}
    for i, x in enumerate(insts):
        s2 = [c for j, c in enumerate(ST2) if a2[i, j] >= th2[j]] or [ST2[int(np.argmax(a2[i]))]]
        s3 = set(c for j, c in enumerate(ST3) if b3[i, j] >= th3[j])
        if _g(x, "video_context", "official_disclosure") == "true":
            s3 -= {"undisclosed_advertising"}          # 0 of 1281 train instances co-occur
        if is_short(x):
            s3 = {"insufficient_context"}
        preds[x["instanceID"]] = {"st1": ST1[int(np.argmax(adjust_st1(a1[i])))], "st2": s2,
                                  "st3": clean_st3(sorted(s3))}
    return preds


metrics.show("routed ensemble on dev", metrics.score(dev, build(d_a1, d_a2, d_b3, dev)))

if A.target == "test":
    t_a1, t_a2, _ = average("test", ST1ST2_CFG, target)
    _, _, t_b3 = average("test", ST3_CFG, target)
    preds = build(t_a1, t_a2, t_b3, target)
else:
    preds = build(d_a1, d_a2, d_b3, dev)

out = A.out or f"../work/submission_{A.target}_routed_ens.jsonl"
submit.write(preds, Path(__file__).resolve().parents[2] / "work" / out, target)

print("  st1:", Counter(p["st1"] for p in preds.values()).most_common())
print("  st3:", Counter(l for p in preds.values() for l in p["st3"]).most_common())
