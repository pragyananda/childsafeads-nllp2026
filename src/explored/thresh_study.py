"""Does threshold tuning help, and does it combine with balanced class weights?

The first tuned run scored below the untuned probe, so this measures the four
combinations directly rather than assuming. Both corrections target the same
imbalance, so applying them together double-counts; and thresholds fitted on
K-fold OOF probabilities transfer badly when the fold models see much less data
than the final model. Varying folds separates those two effects.
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST2, ST3, load_split, view, domain, _g
import metrics

LEVEL = 4
train, target = load_split("train"), load_split("dev")
groups = np.array([_g(x, "channel_context", "channelID") or x["instanceID"] for x in train])
tr_txt = [view(x, LEVEL) for x in train]
te_txt = [view(x, LEVEL) for x in target]


def feats(fit_txt, fit_insts, app_txt, app_insts):
    w = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=200_000, sublinear_tf=True)
    A = w.fit_transform(fit_txt); B = w.transform(app_txt)
    d = TfidfVectorizer(analyzer=lambda s: [s] if s else [], min_df=1)
    A = sp.hstack([A, d.fit_transform([domain(x) for x in fit_insts])]).tocsr()
    B = sp.hstack([B, d.transform([domain(x) for x in app_insts])]).tocsr()
    return A, B


def fit_probs(A, Y, B, balanced):
    cw = "balanced" if balanced else None
    M = np.zeros((B.shape[0], Y.shape[1]))
    for j in range(Y.shape[1]):
        if Y[:, j].sum() < 2:
            continue
        m = LogisticRegression(max_iter=1500, C=4, class_weight=cw).fit(A, Y[:, j])
        M[:, j] = m.predict_proba(B)[:, 1]
    return M


mlb2, mlb3 = MultiLabelBinarizer(classes=ST2), MultiLabelBinarizer(classes=ST3)
Y2 = mlb2.fit_transform([x["labels"]["st2"] for x in train])
Y3 = mlb3.fit_transform([x["labels"]["st3"] for x in train])
G2 = [x["labels"]["st2"] for x in target]
G3 = [x["labels"]["st3"] for x in target]

A_full, B_full = feats(tr_txt, train, te_txt, target)
print(f"{'config':38} {'ST2':>6} {'ST3':>6}")
cache = {}
for balanced in (True, False):
    p2 = fit_probs(A_full, Y2, B_full, balanced)
    p3 = fit_probs(A_full, Y3, B_full, balanced)
    cache[balanced] = (p2, p3)
    d2 = [[c for j, c in enumerate(ST2) if p2[i, j] >= .5] or [ST2[int(np.argmax(p2[i]))]] for i in range(len(target))]
    d3 = [[c for j, c in enumerate(ST3) if p3[i, j] >= .5] or ["no_flag"] for i in range(len(target))]
    tag = "balanced" if balanced else "unweighted"
    print(f"{tag + ' + fixed 0.5':38} "
          f"{metrics._multilabel(G2, d2, ST2):6.3f} {metrics._multilabel(G3, d3, ST3):6.3f}")

for folds in (3, 5, 10):
    for balanced in (True, False):
        oof2, oof3 = np.zeros_like(Y2, float), np.zeros_like(Y3, float)
        for tr_i, va_i in GroupKFold(n_splits=folds).split(tr_txt, None, groups):
            Af, Bf = feats([tr_txt[i] for i in tr_i], [train[i] for i in tr_i],
                           [tr_txt[i] for i in va_i], [train[i] for i in va_i])
            oof2[va_i] = fit_probs(Af, Y2[tr_i], Bf, balanced)
            oof3[va_i] = fit_probs(Af, Y3[tr_i], Bf, balanced)
        th2, th3 = metrics.tune_thresholds(oof2, Y2), metrics.tune_thresholds(oof3, Y3)
        p2, p3 = cache[balanced]
        d2 = [[c for j, c in enumerate(ST2) if p2[i, j] >= th2[j]] or [ST2[int(np.argmax(p2[i] - th2))]] for i in range(len(target))]
        d3 = [[c for j, c in enumerate(ST3) if p3[i, j] >= th3[j]] or ["no_flag"] for i in range(len(target))]
        tag = ("balanced" if balanced else "unweighted") + f" + tuned({folds}f)"
        print(f"{tag:38} {metrics._multilabel(G2, d2, ST2):6.3f} {metrics._multilabel(G3, d3, ST3):6.3f}", flush=True)
