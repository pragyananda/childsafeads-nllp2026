"""Do dev-fitted thresholds transfer, or do they just fit dev's noise?

The test submission plans to fit thresholds on dev and apply them to test. That
is only sound if thresholds fitted on one channel-disjoint sample transfer to
another. This splits dev in half by channel, fits on one half, scores the other,
and compares against a fixed 0.5 cutoff on the same half.
"""
import sys
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
L4 = "probs_dev_L4_ModernBERT-large_len2048_seed{s}.npz"
L1 = "probs_dev_L1_ModernBERT-large_len1024_seed{s}.npz"


def avg(pat, seeds):
    p1 = np.zeros((len(dv), 5)); p2 = np.zeros((len(dv), 12)); p3 = np.zeros((len(dv), 8)); n = 0
    for s in seeds:
        f = W / pat.format(s=s)
        if f.exists():
            z = np.load(f, allow_pickle=True)
            idx = [order[i] for i in z["ids"].tolist()]
            p1[idx] += z["p1"]; p2[idx] += z["p2"]; p3[idx] += z["p3"]; n += 1
    return p1 / n, p2 / n, p3 / n, n


def decide(idxs, a1, a2, b3, th2, th3):
    preds = {}
    for i in idxs:
        x = dv[i]
        s2 = [c for j, c in enumerate(ST2) if a2[i, j] >= th2[j]] or [ST2[int(np.argmax(a2[i]))]]
        s3 = set(c for j, c in enumerate(ST3) if b3[i, j] >= th3[j])
        if _g(x, "video_context", "official_disclosure") == "true":
            s3 -= {"undisclosed_advertising"}
        if is_short(x):
            s3 = {"insufficient_context"}
        preds[x["instanceID"]] = {"st1": ST1[int(np.argmax(a1[i]))], "st2": s2,
                                  "st3": clean_st3(sorted(s3))}
    return preds


# Channel-grouped halves, so the two halves are disjoint the way dev/test are.
chans = sorted({_g(x, "channel_context", "channelID") or x["instanceID"] for x in dv})
rng = np.random.default_rng(0); rng.shuffle(chans)
half = set(chans[:len(chans) // 2])
A = [i for i, x in enumerate(dv) if (_g(x, "channel_context", "channelID") or x["instanceID"]) in half]
B = [i for i, x in enumerate(dv) if i not in set(A)]
print(f"fit half: {len(A)} instances | score half: {len(B)} instances")

for seeds, tag in [((0, 1, 2), "3 seeds"), ((0, 1, 2, 3, 4, 5), "6 seeds")]:
    a1, a2, a3, na = avg(L4, seeds)
    b1, b2, b3, nb = avg(L1, seeds)
    goldA = [dv[i] for i in A]
    Y2A = MultiLabelBinarizer(classes=ST2).fit_transform([x["labels"]["st2"] for x in goldA])
    Y3A = MultiLabelBinarizer(classes=ST3).fit_transform([x["labels"]["st3"] for x in goldA])
    th2 = metrics.tune_thresholds(a2[A], Y2A)
    th3 = metrics.tune_thresholds(b3[A], Y3A)
    goldB = [dv[i] for i in B]
    fixed = metrics.score(goldB, decide(B, a1, a2, b3, np.full(12, .5), np.full(8, .5)))
    tuned = metrics.score(goldB, decide(B, a1, a2, b3, th2, th3))
    metrics.show(f"{tag} fixed 0.5   -> held-out half", fixed)
    metrics.show(f"{tag} tuned on A  -> held-out half", tuned)
    print(f"    transfer gain: {tuned['mean'] - fixed['mean']:+.3f}\n")
