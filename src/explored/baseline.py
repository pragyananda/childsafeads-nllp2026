"""Linear baseline: TF-IDF + domain prior + macro-F1 threshold tuning.

This is the insurance system — cheap, deterministic, and good enough to be a
respectable fallback if the encoder work runs out of runway. Three design points
carry most of its score over a plain TF-IDF fit:

* a categorical feature for the outbound link's domain, because splits are
  channel-disjoint but not brand-disjoint (~2/3 of dev/test instances promote a
  domain that also appears in train);
* per-label thresholds fitted on channel-grouped out-of-fold probabilities,
  because macro-F1 weights a 5-instance flag like a 260-instance one;
* the constraint that YouTube's own paid-promotion flag rules out
  `undisclosed_advertising` (0 of 1281 such training instances carry it).

Usage: python baseline.py [--level 4] [--target dev|test]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, view, domain, clean_st3, _g
import metrics
import submit

P = argparse.ArgumentParser()
P.add_argument("--level", type=int, default=4, choices=[1, 2, 3, 4])
P.add_argument("--target", default="dev", choices=["dev", "test"])
P.add_argument("--folds", type=int, default=3)
P.add_argument("--out", default=None)
A = P.parse_args()

train = load_split("train")
target = load_split(A.target)
# During evaluation, dev labels are ours to train on: +21% data for free.
if A.target == "test":
    train = train + load_split("dev")

groups = np.array([_g(x, "channel_context", "channelID") or x["instanceID"] for x in train])
Xtr_txt = [view(x, A.level) for x in train]
Xte_txt = [view(x, A.level) for x in target]


def featurise(fit_txt, apply_txts, fit_insts, apply_insts):
    word = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=200_000, sublinear_tf=True)
    A_ = word.fit_transform(fit_txt)
    mats = [word.transform(t) for t in apply_txts]
    if A.level >= 4:  # domain is only observable once the link is resolved
        dom = TfidfVectorizer(analyzer=lambda s: [s] if s else [], min_df=1)
        A_ = sp.hstack([A_, dom.fit_transform([domain(x) for x in fit_insts])]).tocsr()
        mats = [sp.hstack([m, dom.transform([domain(x) for x in insts])]).tocsr()
                for m, insts in zip(mats, apply_insts)]
    return A_, mats


def fit_predict(Atr, ytr_1, Ytr_2, Ytr_3, Bs):
    """Return (st1 preds, st2 probs, st3 probs) for each matrix in Bs."""
    out = []
    m1 = LogisticRegression(max_iter=1500, C=4, class_weight="balanced").fit(Atr, ytr_1)
    heads = {}
    for task, Y, classes in [("st2", Ytr_2, ST2), ("st3", Ytr_3, ST3)]:
        heads[task] = []
        for j, c in enumerate(classes):
            if Y[:, j].sum() < 2:
                heads[task].append(None)
                continue
            heads[task].append(
                LogisticRegression(max_iter=1500, C=4, class_weight="balanced").fit(Atr, Y[:, j]))
    for B in Bs:
        p1 = m1.predict(B)
        probs = {}
        for task, classes in [("st2", ST2), ("st3", ST3)]:
            M = np.zeros((B.shape[0], len(classes)))
            for j, m in enumerate(heads[task]):
                if m is not None:
                    M[:, j] = m.predict_proba(B)[:, 1]
            probs[task] = M
        out.append((p1, probs["st2"], probs["st3"]))
    return out


mlb2, mlb3 = MultiLabelBinarizer(classes=ST2), MultiLabelBinarizer(classes=ST3)
Y2 = mlb2.fit_transform([x["labels"]["st2"] for x in train])
Y3 = mlb3.fit_transform([x["labels"]["st3"] for x in train])
y1 = [x["labels"]["st1"] for x in train]

# --- out-of-fold probabilities, grouped by channel to mirror the real split ---
print(f"fitting {A.folds}-fold OOF (channel-grouped) for threshold tuning ...")
oof2, oof3 = np.zeros_like(Y2, dtype=float), np.zeros_like(Y3, dtype=float)
for k, (tr_i, va_i) in enumerate(GroupKFold(n_splits=A.folds).split(Xtr_txt, y1, groups)):
    ftxt = [Xtr_txt[i] for i in tr_i]
    finst = [train[i] for i in tr_i]
    Af, (Bf,) = featurise(ftxt, [[Xtr_txt[i] for i in va_i]], finst, [[train[i] for i in va_i]])
    (_, p2, p3), = fit_predict(Af, [y1[i] for i in tr_i], Y2[tr_i], Y3[tr_i], [Bf])
    oof2[va_i], oof3[va_i] = p2, p3
    print(f"  fold {k + 1}/{A.folds} done")

th2 = metrics.tune_thresholds(oof2, Y2)
th3 = metrics.tune_thresholds(oof3, Y3)
print("ST3 thresholds:", {c: round(float(t), 2) for c, t in zip(ST3, th3)})

# --- final fit on all training data ---
Atr, (Bte,) = featurise(Xtr_txt, [Xte_txt], train, [target])
(p1, p2, p3), = fit_predict(Atr, y1, Y2, Y3, [Bte])

preds = {}
for i, x in enumerate(target):
    st2 = [c for j, c in enumerate(ST2) if p2[i, j] >= th2[j]]
    if not st2:
        st2 = [ST2[int(np.argmax(p2[i] / np.maximum(th2, 1e-6)))]]
    st3 = [c for j, c in enumerate(ST3) if p3[i, j] >= th3[j]]
    if _g(x, "video_context", "official_disclosure") == "true":
        st3 = [c for c in st3 if c != "undisclosed_advertising"]
    if "undisclosed_advertising" in st3 and "inadequate_disclosure" in st3:
        st3.remove("inadequate_disclosure" if p3[i, ST3.index("undisclosed_advertising")]
                   >= p3[i, ST3.index("inadequate_disclosure")] else "undisclosed_advertising")
    preds[x["instanceID"]] = {"st1": p1[i], "st2": st2, "st3": clean_st3(st3)}

if A.target == "dev":
    metrics.show(f"baseline L{A.level}", metrics.score(target, preds))
    print("  ST3 per-label:", {k: round(v, 2) for k, v in metrics.per_label_f1(
        [x["labels"]["st3"] for x in target], [preds[x["instanceID"]]["st3"] for x in target], ST3).items()})
    print("  ST2 per-label:", {k: round(v, 2) for k, v in metrics.per_label_f1(
        [x["labels"]["st2"] for x in target], [preds[x["instanceID"]]["st2"] for x in target], ST2).items()})

out = A.out or f"../work/submission_{A.target}_L{A.level}.jsonl"
submit.write(preds, Path(__file__).resolve().parents[2] / "work" / out, target)
