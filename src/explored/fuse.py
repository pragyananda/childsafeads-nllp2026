"""Combine encoder probabilities with the rule arm and the legal constraints.

Kept separate from `encoder.py` so fusion can be re-tuned in seconds against
cached probability files instead of retraining, and so several seeds or levels
can be averaged without touching the training code.

Order matters: probabilities -> thresholds -> rule arm (which only ever adds
rare labels it is precise on) -> taxonomy constraints (which are hard, and so
run last).

Usage: python fuse.py --probs ../work/probs_dev_L4_ModernBERT-large.npz [--probs ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, clean_st3, _g
from rules import fit_domain_gazetteer, apply_gazetteer, is_short
import metrics
import submit

P = argparse.ArgumentParser()
P.add_argument("--probs", nargs="+", required=True, help="one or more .npz files to average")
P.add_argument("--target", default="dev", choices=["dev", "test"])
P.add_argument("--no-rules", action="store_true")
P.add_argument("--out", default=None)
A = P.parse_args()

train = load_split("train")
target = load_split(A.target)
if A.target == "test":
    train = train + load_split("dev")

# --- average probabilities across seeds/models, aligning on instanceID ---
order = {x["instanceID"]: i for i, x in enumerate(target)}
p1 = np.zeros((len(target), len(ST1)))
p2 = np.zeros((len(target), len(ST2)))
p3 = np.zeros((len(target), len(ST3)))
for f in A.probs:
    z = np.load(f, allow_pickle=True)
    idx = [order[i] for i in z["ids"].tolist()]
    p1[idx] += z["p1"]; p2[idx] += z["p2"]; p3[idx] += z["p3"]
p1 /= len(A.probs); p2 /= len(A.probs); p3 /= len(A.probs)

RARE3 = {"hfss_food_marketing", "age_restricted_or_prohibited_product"}
RARE2 = {"gambling_adjacent", "toys", "gambling"}
gaz3 = fit_domain_gazetteer(train, RARE3, "st3", min_count=1, min_purity=0.5)
gaz2 = fit_domain_gazetteer(train, RARE2, "st2", min_count=1, min_purity=0.5)

if A.target == "dev":
    from sklearn.preprocessing import MultiLabelBinarizer
    Y2 = MultiLabelBinarizer(classes=ST2).fit_transform([x["labels"]["st2"] for x in target])
    Y3 = MultiLabelBinarizer(classes=ST3).fit_transform([x["labels"]["st3"] for x in target])
    th2, th3 = metrics.tune_thresholds(p2, Y2), metrics.tune_thresholds(p3, Y3)
    np.savez(Path(__file__).resolve().parents[2] / "work/fuse_thresholds.npz", th2=th2, th3=th3)
else:
    z = np.load(Path(__file__).resolve().parents[2] / "work/fuse_thresholds.npz")
    th2, th3 = z["th2"], z["th3"]


def decide(i, x):
    st1 = ST1[int(np.argmax(p1[i]))]
    st2 = [c for j, c in enumerate(ST2) if p2[i, j] >= th2[j]]
    st3 = [c for j, c in enumerate(ST3) if p3[i, j] >= th3[j]]

    if not A.no_rules:
        st2 = sorted(set(st2) | apply_gazetteer(x, gaz2))
        st3 = sorted(set(st3) | apply_gazetteer(x, gaz3))
        if is_short(x):
            st3 = ["insufficient_context"]

    # Taxonomy and evidence constraints, applied last because they are hard.
    if _g(x, "video_context", "official_disclosure") == "true":
        st3 = [c for c in st3 if c != "undisclosed_advertising"]
    if "undisclosed_advertising" in st3 and "inadequate_disclosure" in st3:
        drop = ("inadequate_disclosure"
                if p3[i, ST3.index("undisclosed_advertising")] >= p3[i, ST3.index("inadequate_disclosure")]
                else "undisclosed_advertising")
        st3.remove(drop)
    if not st2:
        st2 = [ST2[int(np.argmax(p2[i]))]]
    return {"st1": st1, "st2": st2, "st3": clean_st3(st3)}


preds = {x["instanceID"]: decide(i, x) for i, x in enumerate(target)}

if A.target == "dev":
    tag = "fused" + ("(no rules)" if A.no_rules else "")
    metrics.show(tag, metrics.score(target, preds))
    print("  ST3:", {k: round(v, 2) for k, v in metrics.per_label_f1(
        [x["labels"]["st3"] for x in target], [preds[x["instanceID"]]["st3"] for x in target], ST3).items()})
    print("  ST2:", {k: round(v, 2) for k, v in metrics.per_label_f1(
        [x["labels"]["st2"] for x in target], [preds[x["instanceID"]]["st2"] for x in target], ST2).items()})

submit.write(preds, Path(__file__).resolve().parents[2] / "work" / (A.out or f"../work/fused_{A.target}.jsonl"), target)
