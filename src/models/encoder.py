"""Multi-task encoder: one backbone, three heads, one training run.

All three sub-tasks read the same text, so they share an encoder and are trained
jointly. That is why entering all three costs barely more than entering one --
and entering only one caps the leaderboard mean at 0.333, since omitted
sub-tasks score zero.

Loss: cross-entropy for ST1, BCE for ST2/ST3 with positive-class weighting, so
the rare flags that macro-F1 rewards are not drowned out by `misleading_claim`.
Thresholds are fitted afterwards on channel-grouped held-out probabilities.

Usage: python encoder.py --model answerdotai/ModernBERT-large --level 4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, view, clean_st3, _g
import metrics
import submit

P = argparse.ArgumentParser()
P.add_argument("--model", default="answerdotai/ModernBERT-large")
P.add_argument("--level", type=int, default=4, choices=[1, 2, 3, 4])
P.add_argument("--target", default="dev", choices=["dev", "test"])
P.add_argument("--maxlen", type=int, default=1024)
P.add_argument("--bs", type=int, default=4)
P.add_argument("--accum", type=int, default=4)
P.add_argument("--epochs", type=int, default=4)
P.add_argument("--lr", type=float, default=2e-5)
P.add_argument("--st1-weight", type=float, default=0.0,
               help="class-weight exponent for the ST1 cross-entropy: weight_c = "
                    "(1/freq_c)**e. The ST1 head has trained unweighted while ST2/ST3 "
                    "receive pos_weight, so `none` (1.5% of train) is almost never the "
                    "argmax -- recall 0.14. Three post-hoc corrections failed; this "
                    "fixes it in the loss instead.")
P.add_argument("--seed", type=int, default=0)
P.add_argument("--out", default=None)
P.add_argument("--tag", default="",
               help="Suffix for probability filenames. Required when varying anything the "
                    "filename does not already encode (e.g. training on train+dev), or the "
                    "run silently overwrites an earlier one's probabilities.")
P.add_argument("--holdout-frac", type=float, default=0.0,
               help="Hold out a channel-grouped slice of train, train without it, and dump "
                    "its probabilities for honest threshold fitting. Thresholds fitted on the "
                    "same data they are scored on are worth ~0.05 of pure inflation.")
P.add_argument("--cv-folds", type=int, default=0,
               help="Channel-grouped k-fold CV over train+dev. Dev alone holds 504 instances "
                    "with 14 `none` and 7 `insufficient_context`, which cannot separate a real "
                    "improvement from noise: three changes that looked good on dev moved "
                    "-0.041, -0.005 and +0.021 on the evaluation set. Pooling gives 2,857 "
                    "out-of-fold instances instead.")
P.add_argument("--cv-fold", type=int, default=0, help="Which fold to hold out (0-indexed).")
A = P.parse_args()

torch.manual_seed(A.seed)
np.random.seed(A.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(A.model)

train = load_split("train")
target = load_split(A.target)
if A.target == "test":
    train = train + load_split("dev")

if A.cv_folds:
    # Fold assignment is by channel and deterministic across folds and configs,
    # so every out-of-fold prediction set covers the same 2,857 instances and
    # different configurations remain directly comparable.
    pool = load_split("train") + load_split("dev")
    chans = sorted({_g(x, "channel_context", "channelID") or x["instanceID"] for x in pool})
    np.random.default_rng(12345).shuffle(chans)
    fold_of = {c: i % A.cv_folds for i, c in enumerate(chans)}
    key = lambda x: fold_of[_g(x, "channel_context", "channelID") or x["instanceID"]]
    train = [x for x in pool if key(x) != A.cv_fold]
    target = [x for x in pool if key(x) == A.cv_fold]
    print(f"CV fold {A.cv_fold}/{A.cv_folds}: train {len(train)}, held out {len(target)}")

holdout = []
if A.holdout_frac > 0:
    # Group by channel so the calibration slice mirrors the channel-disjoint
    # split the test set actually is; a random slice would leak channel identity
    # and produce thresholds that look good here and fail there.
    rng = np.random.default_rng(A.seed)
    chans = sorted({_g(x, "channel_context", "channelID") or x["instanceID"] for x in train})
    rng.shuffle(chans)
    held = set(chans[:max(1, int(len(chans) * A.holdout_frac))])
    keep = [x for x in train
            if (_g(x, "channel_context", "channelID") or x["instanceID"]) not in held]
    holdout = [x for x in train
               if (_g(x, "channel_context", "channelID") or x["instanceID"]) in held]
    train = keep
    print(f"holdout: {len(holdout)} instances / {len(held)} channels; training on {len(train)}")

mlb2, mlb3 = MultiLabelBinarizer(classes=ST2), MultiLabelBinarizer(classes=ST3)
Y2 = mlb2.fit_transform([x["labels"]["st2"] for x in train]).astype(np.float32)
Y3 = mlb3.fit_transform([x["labels"]["st3"] for x in train]).astype(np.float32)
y1 = np.array([ST1.index(x["labels"]["st1"]) for x in train])


class DS(Dataset):
    def __init__(self, insts, y1=None, y2=None, y3=None):
        self.t = [view(x, A.level) for x in insts]
        self.y1, self.y2, self.y3 = y1, y2, y3

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        d = {"text": self.t[i]}
        if self.y1 is not None:
            d |= {"y1": self.y1[i], "y2": self.y2[i], "y3": self.y3[i]}
        return d


def collate(batch):
    enc = tok([b["text"] for b in batch], truncation=True, max_length=A.maxlen,
              padding=True, return_tensors="pt")
    if "y1" in batch[0]:
        enc["y1"] = torch.tensor([b["y1"] for b in batch])
        enc["y2"] = torch.tensor(np.array([b["y2"] for b in batch]))
        enc["y3"] = torch.tensor(np.array([b["y3"] for b in batch]))
    return enc


class MultiTask(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.h1, self.h2, self.h3 = nn.Linear(h, len(ST1)), nn.Linear(h, len(ST2)), nn.Linear(h, len(ST3))

    def forward(self, input_ids, attention_mask, **_):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = self.drop((out * m).sum(1) / m.sum(1).clamp(min=1e-6))  # mean-pool
        return self.h1(pooled), self.h2(pooled), self.h3(pooled)


def pos_weight(Y, cap=20.0):
    pos = Y.sum(0)
    neg = len(Y) - pos
    return torch.tensor(np.clip(neg / np.maximum(pos, 1), 1.0, cap), dtype=torch.float32, device=dev)


model = MultiTask(A.model).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=A.lr, weight_decay=0.01)
dl = DataLoader(DS(train, y1, Y2, Y3), batch_size=A.bs, shuffle=True, collate_fn=collate, drop_last=True)
steps = (len(dl) // A.accum) * A.epochs
sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
if A.st1_weight:
    _cnt = np.bincount(y1, minlength=len(ST1)).astype(float)
    _w = (1.0 / np.maximum(_cnt, 1)) ** A.st1_weight
    _w = _w / _w.sum() * len(ST1)                       # mean weight 1
    _w[_cnt == 0] = 0.0                                 # unseen class contributes nothing
    print("ST1 class weights:", {c: round(float(v), 2) for c, v in zip(ST1, _w)})
    ce = nn.CrossEntropyLoss(weight=torch.tensor(_w, dtype=torch.float32, device=dev))
else:
    ce = nn.CrossEntropyLoss()
b2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight(Y2))
b3 = nn.BCEWithLogitsLoss(pos_weight=pos_weight(Y3))

# No GradScaler: loss scaling exists to stop fp16 gradients underflowing, and
# bfloat16 carries fp32's exponent range. It was harmless with ModernBERT and
# raises "Attempting to unscale FP16 gradients" on DeBERTa-v3.
model.train()
for ep in range(A.epochs):
    tot = 0.0
    for i, batch in enumerate(dl):
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.autocast(dev, dtype=torch.bfloat16):
            o1, o2, o3 = model(**batch)
            loss = (ce(o1, batch["y1"]) + b2(o2, batch["y2"]) + b3(o3, batch["y3"])) / A.accum
        loss.backward()
        if (i + 1) % A.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); sch.step()
        tot += loss.item() * A.accum
    print(f"epoch {ep + 1}/{A.epochs} loss {tot / len(dl):.4f}", flush=True)


@torch.no_grad()
def infer(insts):
    model.eval()
    p1, p2, p3 = [], [], []
    for batch in DataLoader(DS(insts), batch_size=16, collate_fn=collate):
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.autocast(dev, dtype=torch.bfloat16):
            o1, o2, o3 = model(**batch)
        p1.append(o1.float().softmax(-1).cpu().numpy())
        p2.append(o2.float().sigmoid().cpu().numpy())
        p3.append(o3.float().sigmoid().cpu().numpy())
    return np.vstack(p1), np.vstack(p2), np.vstack(p3)


if holdout:
    h1, h2, h3 = infer(holdout)
    np.savez(Path(__file__).resolve().parents[2] / f"work/calib_L{A.level}_len{A.maxlen}_seed{A.seed}{A.tag}.npz",
             p1=h1, p2=h2, p3=h3,
             y2=mlb2.transform([x["labels"]["st2"] for x in holdout]),
             y3=mlb3.transform([x["labels"]["st3"] for x in holdout]),
             ids=np.array([x["instanceID"] for x in holdout]))
    print(f"wrote calibration probabilities for {len(holdout)} held-out instances")

# Predict every split we might need from this one training run. Weights are not
# saved, so a run that only scores dev would have to be repeated in full to
# produce test predictions.
if A.target == "dev" and not A.cv_folds:
    t1, t2, t3 = infer(load_split("test"))
    np.savez(Path(__file__).resolve().parents[2] /
             f"work/probs_test_L{A.level}_{Path(A.model).name}_len{A.maxlen}_seed{A.seed}{A.tag}.npz",
             p1=t1, p2=t2, p3=t3,
             ids=np.array([x["instanceID"] for x in load_split("test")]))
    print("also wrote test probabilities from this run")

q1, q2, q3 = infer(target)

if A.cv_folds:
    out = (Path(__file__).resolve().parents[2] /
           f"work/cv/oof_L{A.level}_{Path(A.model).name}_len{A.maxlen}_ep{A.epochs}"
           f"_lr{A.lr:g}_seed{A.seed}_f{A.cv_fold}of{A.cv_folds}{A.tag}.npz")
    out.parent.mkdir(exist_ok=True)
    np.savez(out, p1=q1, p2=q2, p3=q3,
             ids=np.array([x["instanceID"] for x in target]))
    s = metrics.score(target, {x["instanceID"]: {
        "st1": ST1[int(np.argmax(q1[i]))],
        "st2": [c for j, c in enumerate(ST2) if q2[i, j] >= .5] or [ST2[int(np.argmax(q2[i]))]],
        "st3": clean_st3(sorted(c for j, c in enumerate(ST3) if q3[i, j] >= .5))}
        for i, x in enumerate(target)})
    metrics.show(f"fold {A.cv_fold} (fixed 0.5)", s)
    print(f"wrote {out.name}")
    raise SystemExit

# Name must carry every axis we vary, or a later run silently overwrites an
# earlier one's probabilities and seed ensembling averages a file with itself.
np.savez(Path(__file__).resolve().parents[2] /
         f"work/probs_{A.target}_L{A.level}_{Path(A.model).name}_len{A.maxlen}_seed{A.seed}{A.tag}.npz",
         p1=q1, p2=q2, p3=q3, ids=np.array([x["instanceID"] for x in target]))

# Thresholds: tuned on dev when developing; for the test run reuse dev-fitted ones.
if A.target == "dev":
    th2 = metrics.tune_thresholds(q2, mlb2.transform([x["labels"]["st2"] for x in target]))
    th3 = metrics.tune_thresholds(q3, mlb3.transform([x["labels"]["st3"] for x in target]))
    np.savez(Path(__file__).resolve().parents[2] / "work/thresholds.npz", th2=th2, th3=th3)
else:
    z = np.load(Path(__file__).resolve().parents[2] / "work/thresholds.npz")
    th2, th3 = z["th2"], z["th3"]

preds = {}
for i, x in enumerate(target):
    st2 = [c for j, c in enumerate(ST2) if q2[i, j] >= th2[j]] or [ST2[int(np.argmax(q2[i]))]]
    st3 = [c for j, c in enumerate(ST3) if q3[i, j] >= th3[j]]
    if _g(x, "video_context", "official_disclosure") == "true":
        st3 = [c for c in st3 if c != "undisclosed_advertising"]
    preds[x["instanceID"]] = {"st1": ST1[int(np.argmax(q1[i]))], "st2": st2, "st3": clean_st3(st3)}

if A.target == "dev":
    metrics.show(f"{Path(A.model).name} L{A.level}", metrics.score(target, preds))
    print("  ST3 per-label:", {k: round(v, 2) for k, v in metrics.per_label_f1(
        [x["labels"]["st3"] for x in target], [preds[x["instanceID"]]["st3"] for x in target], ST3).items()})

out = A.out or f"../work/enc_{A.target}_L{A.level}.jsonl"
submit.write(preds, Path(__file__).resolve().parents[2] / "work" / out, target)
