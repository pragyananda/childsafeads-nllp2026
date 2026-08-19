"""Classical arm for ST3, built around the signals the encoder cannot see.

Our ST3 encoder reads level 1 -- the transcript alone -- because that is where
the ablation put it. But two flags are defined by things outside the transcript:
`inadequate_disclosure` is *"a disclosure exists but is buried in description
text"*, and `undisclosed_advertising` turns on whether a disclosure appears
anywhere at all, including the platform's own label. The encoder is blind to
both by construction.

So this arm reads transcript AND description AND the paid-promotion flag, and
adds explicit indicator features for where a disclosure appears -- the exact
cross-tab that separates the three disclosure states in training:

    official=false, unspoken, unwritten -> undisclosed  (344/583)
    official=false, unspoken, written   -> inadequate   ( 65/ 90)
    official=true                        -> compliant    (74%)

Validated on the same channel-grouped 5-fold split over train+dev used for every
encoder configuration, so the arms are directly comparable and ensemblable.

Usage: python classical_st3.py --save
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST3, load_split, _g, domain
import metrics

P = argparse.ArgumentParser()
P.add_argument("--folds", type=int, default=5)
P.add_argument("--save", action="store_true")
A = P.parse_args()

W = Path(__file__).resolve().parents[2] / "work"
pool = load_split("train") + load_split("dev")
test = load_split("test")

chans = sorted({_g(x, "channel_context", "channelID") or x["instanceID"] for x in pool})
np.random.default_rng(12345).shuffle(chans)
fold_of = {c: i % A.folds for i, c in enumerate(chans)}
fold = np.array([fold_of[_g(x, "channel_context", "channelID") or x["instanceID"]] for x in pool])

SPOKEN = re.compile(r"\b(sponsor(ed|s|ing)?\s+(by|this|today|the)|this (video|episode) is sponsored"
                    r"|thanks? to \w+ for sponsoring|paid (partnership|promotion)|sponsored by"
                    r"|our sponsor|todays? sponsor)\b", re.I)
WRITTEN = re.compile(r"(#ad\b|#sponsored|\[sponsored\]|\[ad\]|paid (promotion|partnership)"
                     r"|sponsored by|thanks? to \w+ for sponsoring|in partnership with"
                     r"|#\w*partner|advertisement|affiliate)", re.I)
EXHORT = re.compile(r"\b(go (and )?(buy|get|grab)|make sure (you|to)|don'?t (wait|miss)"
                    r"|hurry|right now|today only|limited time|you (need|have) to"
                    r"|ask your (parents|mum|mom|dad)|if you love|trust me|i promise"
                    r"|you'?ll love|please )\b", re.I)


def dense_feats(insts):
    """The signals the flags are literally defined in terms of."""
    rows = []
    for x in insts:
        t = _g(x, "transcript", "text")
        d = _g(x, "video_context", "description")
        od = _g(x, "video_context", "official_disclosure")
        nw = len(t.split())
        rows.append([
            float(od == "true"), float(od == "false"), float(od == ""),
            float(bool(SPOKEN.search(t))), float(bool(WRITTEN.search(d))),
            float(bool(EXHORT.search(t))),
            len(EXHORT.findall(t)) / 5.0,
            min(nw, 600) / 600.0, float(nw <= 10), float(nw <= 40),
            min(len(d), 4000) / 4000.0, float(len(d) < 200),
            float("http" in d.lower()), d.lower().count("http") / 20.0,
        ])
    return np.asarray(rows, dtype=np.float64)


def build(fit_insts, apply_sets):
    ftr = [_g(x, "transcript", "text")[:4000] for x in fit_insts]
    fde = [_g(x, "video_context", "description")[:2500] for x in fit_insts]
    vt = TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=200_000, sublinear_tf=True)
    vd = TfidfVectorizer(min_df=3, ngram_range=(1, 2), max_features=200_000, sublinear_tf=True)
    Xf = sp.hstack([vt.fit_transform(ftr), vd.fit_transform(fde),
                    sp.csr_matrix(dense_feats(fit_insts))]).tocsr()
    out = []
    for insts in apply_sets:
        out.append(sp.hstack([
            vt.transform([_g(x, "transcript", "text")[:4000] for x in insts]),
            vd.transform([_g(x, "video_context", "description")[:2500] for x in insts]),
            sp.csr_matrix(dense_feats(insts))]).tocsr())
    return Xf, out


Y = MultiLabelBinarizer(classes=ST3).fit_transform([x["labels"]["st3"] for x in pool])


def fit_predict(Xtr, Xte, Ytr):
    Q = np.zeros((Xte.shape[0], len(ST3)))
    for j in range(len(ST3)):
        if Ytr[:, j].sum() < 3:
            continue
        m = LogisticRegression(max_iter=3000, C=4, class_weight="balanced").fit(Xtr, Ytr[:, j])
        Q[:, j] = m.predict_proba(Xte)[:, 1]
    return Q


oof = np.zeros((len(pool), len(ST3)))
for k in range(A.folds):
    tr_i, va_i = np.where(fold != k)[0], np.where(fold == k)[0]
    Xtr, (Xva,) = build([pool[i] for i in tr_i], [[pool[i] for i in va_i]])
    oof[va_i] = fit_predict(Xtr, Xva, Y[tr_i])
    print(f"  fold {k + 1}/{A.folds} done", flush=True)

np.savez(W / "classical_oof_st3.npz", p=oof, ids=np.array([x["instanceID"] for x in pool]))
gold = [x["labels"]["st3"] for x in pool]
pred = [[c for j, c in enumerate(ST3) if oof[i, j] >= .5] or ["no_flag"] for i in range(len(pool))]
print(f"\n=== classical ST3, {len(pool)} out-of-fold instances ===")
print(f"  macro-F1 {metrics._multilabel(gold, pred, ST3):.4f}")
print("  per-flag:", {k: round(v, 3) for k, v in metrics.per_label_f1(gold, pred, ST3).items()})

if A.save:
    Xtr, (Xte,) = build(pool, [test])
    np.savez(W / "classical_test_st3.npz", p=fit_predict(Xtr, Xte, Y),
             ids=np.array([x["instanceID"] for x in test]))
    print("  wrote test predictions for st3")
