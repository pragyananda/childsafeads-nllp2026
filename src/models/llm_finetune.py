"""QLoRA fine-tune of an open-weights LLM as an ST3 multi-label classifier.

ST3 is our weakest sub-task by a wide margin (0.4913 on the evaluation set
against a field of 0.529-0.615), and the deficit localises to the two flags whose
taxonomy definitions are written *tests* rather than surface properties:
`inadequate_disclosure` (118 instances, F1 0.335) and `direct_exhortation`
(77, F1 0.343). An encoder cannot read a rulebook. Our zero-shot prompting of
this same base model reached only 0.29, but zero-shot and fine-tuned are
different propositions, and the rank-4 team reports ST3 0.587 from an
open-weights LLM.

Built as a sequence classifier rather than a generator on purpose: it emits
calibrated per-flag probabilities in the same shape as `encoder.py`, so the
result drops into the existing held-out comparison and can be averaged with the
encoder's probabilities instead of replacing them.

Loss mirrors encoder.py -- BCE with positive-class weighting -- so the rare
flags macro-F1 rewards are not drowned out by `misleading_claim` at 54%.

Usage:
    python llm_finetune.py --epochs 3 --seed 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST3, load_split, _g
import metrics

P = argparse.ArgumentParser()
P.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
P.add_argument("--epochs", type=int, default=3)
P.add_argument("--lr", type=float, default=1e-4)      # LoRA wants a larger lr than full FT
P.add_argument("--bs", type=int, default=2)
P.add_argument("--accum", type=int, default=8)
P.add_argument("--maxlen", type=int, default=1024)
P.add_argument("--rank", type=int, default=16)
P.add_argument("--seed", type=int, default=0)
P.add_argument("--tag", default="_qlora")
A = P.parse_args()

torch.manual_seed(A.seed)
np.random.seed(A.seed)
W = Path(__file__).resolve().parents[2] / "work"
train, dv, te = load_split("train"), load_split("dev"), load_split("test")

tok = AutoTokenizer.from_pretrained(A.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# The taxonomy's tests are what the encoder cannot read, so put them in the
# input rather than hoping the model recalls them.
RUBRIC = (
    "Assess this sponsored segment from a child-facing YouTube video under EU "
    "consumer law.\n"
    "undisclosed_advertising: commercial nature identified NOWHERE - not spoken, "
    "not in the description, not by the platform label.\n"
    "inadequate_disclosure: a disclosure exists but is buried in description "
    "text, in adult jargon, or otherwise not clear to a child.\n"
    "direct_exhortation: an appeal to children to buy, or to ask parents to buy. "
    "Pressure on the child, not merely an imperative verb. 'download from the "
    "link below' is an instruction, not an exhortation.\n"
    "misleading_claim: unsubstantiated claims about performance, results or "
    "price; any health, fitness or skincare claim.\n"
    "age_restricted_or_prohibited_product: alcohol, vaping, gambling, weapons.\n"
    "hfss_food_marketing: food high in fat, salt or sugar; energy drinks, "
    "confectionery, fast food.\n"
    "no_flag: appears compliant. insufficient_context: too short to assess.\n\n")


def text_of(x):
    return (RUBRIC
            + f"PAID-PROMOTION LABEL: {_g(x, 'video_context', 'official_disclosure') or 'unknown'}\n"
            + f"TITLE: {_g(x, 'video_context', 'title')}\n"
            + f"TRANSCRIPT: {_g(x, 'transcript', 'text')[:3000]}\n"
            + f"DESCRIPTION: {_g(x, 'video_context', 'description')[:1200]}")


mlb = MultiLabelBinarizer(classes=ST3)
Y = mlb.fit_transform([x["labels"]["st3"] for x in train]).astype(np.float32)


class DS(Dataset):
    def __init__(self, insts, y=None):
        self.t = [text_of(x) for x in insts]
        self.y = y

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        d = {"text": self.t[i]}
        if self.y is not None:
            d["y"] = self.y[i]
        return d


def collate(batch):
    enc = tok([b["text"] for b in batch], truncation=True, max_length=A.maxlen,
              padding=True, return_tensors="pt")
    if "y" in batch[0]:
        enc["labels"] = torch.tensor(np.array([b["y"] for b in batch]))
    return enc


dev = "cuda"
model = AutoModelForSequenceClassification.from_pretrained(
    A.model, num_labels=len(ST3), problem_type="multi_label_classification",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True),
    device_map="cuda")
model.config.pad_token_id = tok.pad_token_id
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=A.rank, lora_alpha=2 * A.rank, lora_dropout=0.05, bias="none",
    task_type="SEQ_CLS",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

pos = Y.sum(0)
pw = torch.tensor(np.clip((len(Y) - pos) / np.maximum(pos, 1), 1.0, 20.0),
                  dtype=torch.bfloat16, device=dev)
lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

dl = DataLoader(DS(train, Y), batch_size=A.bs, shuffle=True, collate_fn=collate, drop_last=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=A.lr, weight_decay=0.01)
steps = (len(dl) // A.accum) * A.epochs
sch = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

model.train()
for ep in range(A.epochs):
    tot = 0.0
    for i, b in enumerate(dl):
        b = {k: v.to(dev) for k, v in b.items()}
        y = b.pop("labels")
        out = model(**b).logits
        loss = lossf(out, y.to(out.dtype)) / A.accum
        loss.backward()
        if (i + 1) % A.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); sch.step()
        tot += loss.item() * A.accum
        if i % 200 == 0:
            print(f"  ep{ep + 1} step {i}/{len(dl)} loss {tot / max(i + 1, 1):.4f}", flush=True)
    print(f"epoch {ep + 1}/{A.epochs} loss {tot / len(dl):.4f}", flush=True)


@torch.no_grad()
def infer(insts):
    model.eval()
    out = []
    for b in DataLoader(DS(insts), batch_size=8, collate_fn=collate):
        b = {k: v.to(dev) for k, v in b.items()}
        out.append(torch.sigmoid(model(**b).logits.float()).cpu().numpy())
    return np.vstack(out)


for split, insts in (("dev", dv), ("test", te)):
    p3 = infer(insts)
    # p1/p2 are written as zeros so the file shape matches encoder.py's and the
    # existing loaders can read it without special-casing.
    np.savez(W / f"probs_{split}_L2_{Path(A.model).name}_len{A.maxlen}_seed{A.seed}{A.tag}.npz",
             p1=np.zeros((len(insts), 5)), p2=np.zeros((len(insts), 12)), p3=p3,
             ids=np.array([x["instanceID"] for x in insts]))
    print(f"wrote {split} probabilities ({len(insts)})")
    if split == "dev":
        gold = [x["labels"]["st3"] for x in insts]
        pred = [[c for j, c in enumerate(ST3) if p3[i, j] >= .5] or ["no_flag"]
                for i in range(len(insts))]
        print(f"  raw ST3 macro-F1 = {metrics._multilabel(gold, pred, ST3):.4f}")
        print("  per-flag:", {k: round(v, 3) for k, v in
                              metrics.per_label_f1(gold, pred, ST3).items()})
