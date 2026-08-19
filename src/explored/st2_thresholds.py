"""Fit per-label ST2 decision thresholds on dev, and prove they transfer.

The submitted system used a fixed 0.5 cutoff on every ST2 label, on the strength
of the threshold-transfer study in `threshold_transfer.py`. That study measured
the *mean* over all three sub-tasks and found tuned thresholds regressed. The
decomposition below shows the regression is entirely ST3's:

    ST2  +0.060 +- 0.033   wins 30/30 channel-grouped splits
    ST3  -0.037 +- 0.031   wins  3/30

so the conclusion was right for ST3 and wrong for ST2, and ST2 paid ~0.06 for it.

Why 0.5 is the wrong cutoff for ST2: the ST2 head trains under BCE with
`pos_weight = clip(neg/pos, 1, 20)`, which moves each label's decision point away
from 0.5 by an amount that differs per label (2.6 for `apps`, 20 for `toys`). At
0.5 the system emits 1.89 labels per instance against a gold mean of 1.27 --
precision 0.25 on `gambling_adjacent`, 0.26 on `toys`, 0.36 on `creator_community`.
Tuned, it emits 1.35.

Threshold rule: the centre of each label's near-optimal plateau (all thresholds
within `tol` of the best F1), not the arg-max. The arg-max lands on whichever
single grid point a handful of instances happened to favour; the plateau centre
survives resampling, and is worth +0.017 over arg-max on held-out halves.

Usage:
    python st2_thresholds.py            # fit on dev, write work/th2_dev.npy
    python st2_thresholds.py --validate # + the channel-grouped transfer study
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST2, load_split, _g

P = argparse.ArgumentParser()
P.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
P.add_argument("--tol", type=float, default=0.01,
               help="width of the near-optimal plateau whose centre becomes the threshold")
P.add_argument("--validate", action="store_true",
               help="run the channel-grouped transfer study before writing")
P.add_argument("--reps", type=int, default=30)
P.add_argument("--out", default="work/th2_dev.npy")
A = P.parse_args()

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "work"
GRID = np.arange(0.05, 0.96, 0.01)

dv = load_split("dev")
order = {x["instanceID"]: i for i, x in enumerate(dv)}
Y = MultiLabelBinarizer(classes=ST2).fit_transform([x["labels"]["st2"] for x in dv])

# Dev probabilities from the train-only models -- the same six seeds the
# evaluation submission averages. Dev is genuinely held out for these, which is
# what makes thresholds fitted here legitimate to apply to test.
p2, used = np.zeros((len(dv), len(ST2))), []
for s in A.seeds:
    f = W / f"probs_dev_L4_ModernBERT-large_len2048_seed{s}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        p2[[order[i] for i in z["ids"].tolist()]] += z["p2"]
        used.append(s)
assert used, "no dev probability files found"
p2 /= len(used)
print(f"averaged dev L4 probabilities over seeds {used}")


def macro(probs, gold, th):
    """Decision rule, identical to the one the submission applies to test."""
    D = (probs >= th).astype(int)
    for i in np.where(D.sum(1) == 0)[0]:
        D[i, int(np.argmax(probs[i] - th))] = 1   # margin-based fallback, never empty
    return f1_score(gold, D, average="macro", zero_division=0)


def fit(probs, gold, tol):
    th = np.full(probs.shape[1], 0.5)
    for j in range(probs.shape[1]):
        if not gold[:, j].sum():
            continue
        curve = np.array([f1_score(gold[:, j], (probs[:, j] >= t).astype(int),
                                   zero_division=0) for t in GRID])
        th[j] = float(np.median(GRID[curve >= curve.max() - tol]))
    return th


if A.validate:
    chans = np.array([_g(x, "channel_context", "channelID") or x["instanceID"] for x in dv])
    uniq = sorted(set(chans))
    gains = []
    for rep in range(A.reps):
        rng = np.random.default_rng(1000 + rep)
        sh = list(uniq)
        rng.shuffle(sh)
        held = set(sh[:len(sh) // 2])
        fit_i = np.array([i for i in range(len(dv)) if chans[i] in held])
        sc_i = np.array([i for i in range(len(dv)) if chans[i] not in held])
        gains.append(macro(p2[sc_i], Y[sc_i], fit(p2[fit_i], Y[fit_i], A.tol))
                     - macro(p2[sc_i], Y[sc_i], np.full(len(ST2), 0.5)))
    g = np.array(gains)
    print(f"\ntransfer over {A.reps} channel-grouped halves: {g.mean():+.4f} "
          f"+- {g.std():.4f}  worst {g.min():+.4f}  wins {int((g > 0).sum())}/{A.reps}")

th = fit(p2, Y, A.tol)
base = macro(p2, Y, np.full(len(ST2), 0.5))
print(f"\n{'label':24}{'devN':>6}{'thresh':>8}")
for j, c in enumerate(ST2):
    print(f"{c:24}{Y[:, j].sum():>6}{th[j]:>8.2f}")
print(f"\ndev ST2 macro-F1: {base:.4f} (fixed 0.5) -> {macro(p2, Y, th):.4f} (tuned, in-sample)")

D = (p2 >= th).astype(int)
for i in np.where(D.sum(1) == 0)[0]:
    D[i, int(np.argmax(p2[i] - th))] = 1
print(f"labels per instance: {(p2 >= 0.5).sum(1).mean():.2f} at 0.5 -> "
      f"{D.sum(1).mean():.2f} tuned, against {Y.sum(1).mean():.2f} gold")

np.save(ROOT / A.out, th)
print(f"\nwrote {A.out}")
