"""Local replica of the CodaBench metric, plus the threshold tuner.

The organisers report macro-F1 per sub-task and rank on their mean. One detail
is ambiguous in the task description: whether ST1 macro-averages over all five
labels or only those present in the reference. `other` has 2 train and 0 dev
instances, so the two readings differ by ~0.13 on identical predictions. We
report both and optimise against the pessimistic one (`st1_all`).
"""
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from data import ST1, ST2, ST3, ST3_FAMILY


def _multilabel(gold, pred, classes):
    mlb = MultiLabelBinarizer(classes=classes)
    G = mlb.fit_transform([[l for l in g if l in set(classes)] for g in gold])
    P = mlb.transform([[l for l in p if l in set(classes)] for p in pred])
    return f1_score(G, P, average="macro", zero_division=0)


def score(gold_insts, preds):
    """preds: {instanceID: {"st1": str, "st2": [...], "st3": [...]}}. Missing -> 0."""
    g1, p1, g2, p2, g3, p3 = [], [], [], [], [], []
    for x in gold_insts:
        pr, L = preds.get(x["instanceID"], {}), x["labels"]
        g1.append(L["st1"]); p1.append(pr.get("st1", "__MISSING__"))
        g2.append(L["st2"]); p2.append(pr.get("st2", []))
        g3.append(L["st3"]); p3.append(pr.get("st3", []))
    st1_all = f1_score(g1, p1, labels=ST1, average="macro", zero_division=0)
    st1_pres = f1_score(g1, p1, labels=sorted(set(g1)), average="macro", zero_division=0)
    st2 = _multilabel(g2, p2, ST2)
    st3 = _multilabel(g3, p3, ST3)
    fam = _multilabel([{ST3_FAMILY[l] for l in g if l in ST3_FAMILY} for g in g3],
                      [{ST3_FAMILY[l] for l in p if l in ST3_FAMILY} for p in p3],
                      ["disclosure", "content", "product"])
    # `mean` uses st1_present: a dev submission scoring 0.6185 on CodaBench
    # matched this reading to 0.6183 and the all-labels reading to 0.5710, so
    # the leaderboard macro-averages ST1 over reference-present labels only.
    return {"st1": st1_pres, "st2": st2, "st3": st3, "st3_family": fam,
            "mean": (st1_pres + st2 + st3) / 3,
            "st1_all_labels": st1_all,
            "coverage": sum(1 for x in gold_insts if x["instanceID"] in preds) / max(len(gold_insts), 1)}


def per_label_f1(gold, pred, classes):
    mlb = MultiLabelBinarizer(classes=classes)
    G = mlb.fit_transform([[l for l in g if l in set(classes)] for g in gold])
    P = mlb.transform([[l for l in p if l in set(classes)] for p in pred])
    return dict(zip(classes, f1_score(G, P, average=None, zero_division=0)))


def tune_thresholds(probs, Y, grid=None):
    """Per-label decision thresholds maximising each label's own F1.

    Macro-F1 weights a 5-instance flag exactly like a 260-instance one, so a
    global 0.5 cutoff throws away most of the rare-class score. Each column is
    independent under macro averaging, so tuning them separately is exact, not
    a heuristic.
    """
    grid = np.arange(0.05, 0.96, 0.01) if grid is None else grid
    ths = np.full(probs.shape[1], 0.5)
    for j in range(probs.shape[1]):
        if Y[:, j].sum() == 0:
            continue
        best, best_t = -1.0, 0.5
        for t in grid:
            f = f1_score(Y[:, j], (probs[:, j] >= t).astype(int), zero_division=0)
            if f > best:
                best, best_t = f, t
        ths[j] = best_t
    return ths


def show(name, s):
    print(f"{name:34} " + "  ".join(
        f"{k}={v:.3f}" for k, v in s.items() if k != "mean_optimistic"))
