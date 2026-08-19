"""Seed ensembles per config, and the routed combination.

Averaging probabilities across seeds is preferred to selecting the best seed:
observed run-to-run spread on an identical config is 0.089 on ST1, so selection
mostly banks noise. Reported both with fixed 0.5 thresholds (honest) and with
dev-fitted thresholds (inflated, comparable to the leaderboard).
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, clean_st3, _g
from rules import is_short
import metrics

W = Path(__file__).resolve().parents[2] / "work"
dv = load_split("dev")
order = {x["instanceID"]: i for i, x in enumerate(dv)}
Y2 = MultiLabelBinarizer(classes=ST2).fit_transform([x["labels"]["st2"] for x in dv])
Y3 = MultiLabelBinarizer(classes=ST3).fit_transform([x["labels"]["st3"] for x in dv])

CONFIGS = {
    "L1@1024": [f"probs_dev_L1_ModernBERT-large_len1024_seed{s}.npz" for s in (0, 1, 2)],
    "L4@1024": [f"probs_dev_L4_ModernBERT-large_len1024_seed{s}.npz" for s in (0, 1, 2)],
    "L4@2048": [f"probs_dev_L4_ModernBERT-large_len2048_seed{s}.npz" for s in (0, 1, 2)],
}


def average(files):
    p1 = np.zeros((len(dv), len(ST1)))
    p2 = np.zeros((len(dv), len(ST2)))
    p3 = np.zeros((len(dv), len(ST3)))
    n = 0
    for f in files:
        if not (W / f).exists():
            continue
        z = np.load(W / f, allow_pickle=True)
        idx = [order[i] for i in z["ids"].tolist()]
        p1[idx] += z["p1"]; p2[idx] += z["p2"]; p3[idx] += z["p3"]
        n += 1
    return p1 / n, p2 / n, p3 / n, n


def decide(p1, p2, p3, th2, th3):
    preds = {}
    for i, x in enumerate(dv):
        s2 = [c for j, c in enumerate(ST2) if p2[i, j] >= th2[j]] or [ST2[int(np.argmax(p2[i]))]]
        s3 = set(c for j, c in enumerate(ST3) if p3[i, j] >= th3[j])
        if _g(x, "video_context", "official_disclosure") == "true":
            s3 -= {"undisclosed_advertising"}
        if is_short(x):
            s3 = {"insufficient_context"}
        preds[x["instanceID"]] = {"st1": ST1[int(np.argmax(p1[i]))], "st2": s2,
                                  "st3": clean_st3(sorted(s3))}
    return preds


P = {}
for name, files in CONFIGS.items():
    p1, p2, p3, n = average(files)
    P[name] = (p1, p2, p3)
    for mode in ("honest", "dev-tuned"):
        th2 = np.full(len(ST2), .5) if mode == "honest" else metrics.tune_thresholds(p2, Y2)
        th3 = np.full(len(ST3), .5) if mode == "honest" else metrics.tune_thresholds(p3, Y3)
        metrics.show(f"{name} x{n} [{mode}]", metrics.score(dv, decide(p1, p2, p3, th2, th3)))
    print()

# Routed: each sub-task from the config that wins it on seed means.
print("=== routed combinations (honest thresholds) ===")
best = None
for a, b, c in product(CONFIGS, repeat=3):
    p1 = P[a][0]; p2 = P[b][1]; p3 = P[c][2]
    s = metrics.score(dv, decide(p1, p2, p3, np.full(len(ST2), .5), np.full(len(ST3), .5)))
    if best is None or s["mean"] > best[0]["mean"]:
        best = (s, a, b, c)
print(f"best routing: ST1<-{best[1]}  ST2<-{best[2]}  ST3<-{best[3]}")
metrics.show("routed [honest]", best[0])
p1, p2, p3 = P[best[1]][0], P[best[2]][1], P[best[3]][2]
metrics.show("routed [dev-tuned]", metrics.score(
    dv, decide(p1, p2, p3, metrics.tune_thresholds(p2, Y2), metrics.tune_thresholds(p3, Y3))))
