"""A properly-built classical arm for ST1/ST2, validated on the CV instrument.

Our only classical baseline was a 20-minute throwaway on day one (ST2 0.636,
untuned) and the family was dropped on that basis. A competing team reports
ST2 0.7686 from classical ML, against our fine-tuned encoder's 0.7272 -- so the
dismissal was premature, and the sub-task where we are strongest is the one a
cheaper method may win.

ST1 and ST2 are about *what the buyer receives*, which the shop page states
almost literally. That is a lexical problem: brand names, product nouns, price
formats and URL tokens. An encoder spends its capacity on composition it does
not need here, while character n-grams over the URL and page title capture the
brand identity that two thirds of instances share with training data.

Evaluated on the same channel-grouped 5-fold split over train+dev (2,857
instances) that the encoder configurations were compared on, so the two are
directly comparable and can be ensembled fold-honestly.

Usage: python classical.py --task st2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, load_split, _g, domain
import metrics

P = argparse.ArgumentParser()
P.add_argument("--task", default="st2", choices=["st1", "st2"])
P.add_argument("--folds", type=int, default=5)
P.add_argument("--save", action="store_true", help="also fit on all of train+dev and predict test")
A = P.parse_args()

W = Path(__file__).resolve().parents[2] / "work"
pool = load_split("train") + load_split("dev")
test = load_split("test")

# Same deterministic channel-grouped folds as encoder.py --cv-folds, so results
# line up with the encoder numbers instance for instance.
chans = sorted({_g(x, "channel_context", "channelID") or x["instanceID"] for x in pool})
np.random.default_rng(12345).shuffle(chans)
fold_of = {c: i % A.folds for i, c in enumerate(chans)}
fold = np.array([fold_of[_g(x, "channel_context", "channelID") or x["instanceID"]] for x in pool])


def texts(insts):
    """Three views, vectorised separately: the page, the brand, the transcript."""
    page = [f"{_g(x, 'product_page', 'page_title')} {_g(x, 'product_page', 'text')[:4000]}"
            for x in insts]
    brand = [f"{domain(x)} {_g(x, 'product_page', 'raw_url')}" for x in insts]
    spoken = [f"{_g(x, 'video_context', 'title')} {_g(x, 'transcript', 'text')[:2500]}"
              for x in insts]
    return page, brand, spoken


def build(fit_insts, apply_sets):
    """Word n-grams on prose, character n-grams on the URL (brands are substrings)."""
    fp, fb, fs = texts(fit_insts)
    vp = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_features=300_000, sublinear_tf=True)
    vb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000)
    vs = TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=200_000, sublinear_tf=True)
    A_ = sp.hstack([vp.fit_transform(fp), vb.fit_transform(fb), vs.fit_transform(fs)]).tocsr()
    out = []
    for insts in apply_sets:
        p, b, s = texts(insts)
        out.append(sp.hstack([vp.transform(p), vb.transform(b), vs.transform(s)]).tocsr())
    return A_, out


def fit_predict(Xtr, Xte, y, multilabel):
    if not multilabel:
        # Plain multinomial LR rather than a calibrated SVC: ST1's `other` class
        # has two training instances, too few for the calibrator's internal CV.
        m = LogisticRegression(max_iter=3000, C=8, class_weight="balanced").fit(Xtr, y)
        return m.predict_proba(Xte), list(m.classes_)
    Q = np.zeros((Xte.shape[0], y.shape[1]))
    for j in range(y.shape[1]):
        if y[:, j].sum() < 3:
            continue
        m = LogisticRegression(max_iter=3000, C=8, class_weight="balanced").fit(Xtr, y[:, j])
        Q[:, j] = m.predict_proba(Xte)[:, 1]
    return Q, None


classes = ST1 if A.task == "st1" else ST2
multilabel = A.task == "st2"
if multilabel:
    Y = MultiLabelBinarizer(classes=classes).fit_transform([x["labels"]["st2"] for x in pool])
else:
    Y = np.array([x["labels"]["st1"] for x in pool])

oof = np.zeros((len(pool), len(classes)))
for k in range(A.folds):
    tr_i, va_i = np.where(fold != k)[0], np.where(fold == k)[0]
    Xtr, (Xva,) = build([pool[i] for i in tr_i], [[pool[i] for i in va_i]])
    Q, cls = fit_predict(Xtr, Xva, Y[tr_i], multilabel)
    if cls is not None:                       # align single-label columns
        idx = [cls.index(c) if c in cls else None for c in classes]
        Q = np.column_stack([Q[:, i] if i is not None else np.zeros(len(va_i)) for i in idx])
    oof[va_i] = Q
    print(f"  fold {k + 1}/{A.folds} done", flush=True)

np.savez(W / f"classical_oof_{A.task}.npz", p=oof,
         ids=np.array([x["instanceID"] for x in pool]))

print(f"\n=== classical {A.task.upper()} on {len(pool)} out-of-fold instances ===")
if multilabel:
    pred = [[c for j, c in enumerate(classes) if oof[i, j] >= .5]
            or [classes[int(np.argmax(oof[i]))]] for i in range(len(pool))]
    gold = [x["labels"]["st2"] for x in pool]
    print(f"  macro-F1 {metrics._multilabel(gold, pred, classes):.4f}")
    print("  per-label:", {k: round(v, 2) for k, v in metrics.per_label_f1(gold, pred, classes).items()})
else:
    pred = [classes[int(np.argmax(oof[i]))] for i in range(len(pool))]
    gold = [x["labels"]["st1"] for x in pool]
    print(f"  macro-F1 {f1_score(gold, pred, labels=sorted(set(gold)), average='macro', zero_division=0):.4f}")

if A.save:
    Xtr, (Xte,) = build(pool, [test])
    Q, cls = fit_predict(Xtr, Xte, Y, multilabel)
    if cls is not None:
        idx = [cls.index(c) if c in cls else None for c in classes]
        Q = np.column_stack([Q[:, i] if i is not None else np.zeros(len(test)) for i in idx])
    np.savez(W / f"classical_test_{A.task}.npz", p=Q,
             ids=np.array([x["instanceID"] for x in test]))
    print(f"  wrote test predictions for {A.task}")
