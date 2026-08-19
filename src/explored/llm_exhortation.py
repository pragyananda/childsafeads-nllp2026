"""Does giving a model the written legal test help where a fine-tuned encoder cannot?

Targeted at `direct_exhortation` alone. The deep dive singled it out: the encoder
reaches 0.335 against a brand-oracle ceiling of 0.866, hand-written lexicons
score 0.238, and the taxonomy states the discriminating rule in prose --
"the pressure on the child to make the purchase happen, not the presence of an
imperative verb". An encoder trained on 2,353 examples never sees that sentence.
A prompted model can be handed it. The other weak flag, `inadequate_disclosure`,
is deliberately excluded: its labels are inconsistent across near-identical
disclosures, so no amount of legal context would recover it.

Three conditions isolate what the legal material is worth, which is the open
question the organisers posed:

  none      -- plain question, no definition
  taxonomy  -- the three-part test from labels_taxonomy.md, verbatim
  legal     -- taxonomy plus the UCPD provisions from legal_provisions.json

Scored by the log-odds of " Yes" against " No" at the next token rather than by
generating text: one forward pass per instance, and it yields a continuous score
that can be thresholded and ensembled with the encoder instead of a bare label.

Usage: python llm_exhortation.py --condition taxonomy --split dev
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import load_split, _g, DEV_DIR

P = argparse.ArgumentParser()
P.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
P.add_argument("--condition", default="taxonomy", choices=["none", "taxonomy", "legal"])
P.add_argument("--split", default="dev", choices=["dev", "test"])
P.add_argument("--limit", type=int, default=0)
P.add_argument("--bs", type=int, default=8)
P.add_argument("--max-transcript-chars", type=int, default=4000)
A = P.parse_args()

TAXONOMY_TEST = """A direct exhortation is a direct appeal to children to buy the product, or to persuade their parents or other adults to buy it for them.

COUNTS as exhortation. An explicit purchase appeal in the imperative, addressed to the audience ("go and buy this", "ask your parents to order it"). Also, wording that would otherwise be a plain instruction, where the delivery targets the young audience with personal, hyped, or pressuring language: parasocial appeals ("if you love us, please download it"), pleading or repetition, urgency aimed at the viewer, or child-directed slang and register.

DOES NOT count. Basic transactional instructions, even in the imperative: "download the app from the link below", "click the link in the description", "use my code for 15% off". Stating where or how to obtain the product, or that a discount exists, is not in itself an exhortation to buy. Friendly encouragement wrapped around an instruction ("go give it a try") also stays an instruction.

BOUNDARY. The test is the pressure on the child to make the purchase happen, not the presence of an imperative verb. Where the wording is genuinely ambiguous between an instruction and an appeal, do not flag."""


def legal_context():
    prov = json.loads((DEV_DIR / "legal_provisions.json").read_text())
    f = prov["flags"]["direct_exhortation"]
    lines = [f"Severity: {f['severity']}."]
    for inst in f.get("instruments", []):
        lines.append(f"{inst['instrument']} {', '.join(inst['provisions'])}: {inst['note']}")
    return "\n".join(lines)


def build_prompt(inst):
    t = (_g(inst, "transcript", "text") or "")[:A.max_transcript_chars]
    head = ("You are assessing sponsored segments from YouTube videos on channels with a "
            "significant audience of children and teenagers, under EU consumer law.\n\n")
    if A.condition == "none":
        rule = "Question: does this segment contain a direct exhortation to children to buy the product?"
    elif A.condition == "taxonomy":
        rule = f"Apply this test.\n\n{TAXONOMY_TEST}\n\nQuestion: does this segment contain a direct exhortation?"
    else:
        rule = (f"Apply this test.\n\n{TAXONOMY_TEST}\n\nLegal grounding.\n{legal_context()}\n\n"
                "Question: does this segment contain a direct exhortation?")
    return (f"{head}SEGMENT TRANSCRIPT:\n{t}\n\n{rule}\nAnswer with one word, Yes or No.")


tok = AutoTokenizer.from_pretrained(A.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    A.model, dtype=torch.bfloat16, device_map="cuda:0",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                           bnb_4bit_compute_dtype=torch.bfloat16,
                                           bnb_4bit_quant_type="nf4"))
model.eval()

YES = [tok.encode(s, add_special_tokens=False)[0] for s in ("Yes", " Yes", "yes")]
NO = [tok.encode(s, add_special_tokens=False)[0] for s in ("No", " No", "no")]

insts = load_split(A.split)
if A.limit:
    insts = insts[:A.limit]

scores, ids = [], []
for i in range(0, len(insts), A.bs):
    chunk = insts[i:i + A.bs]
    texts = [tok.apply_chat_template([{"role": "user", "content": build_prompt(x)}],
                                     tokenize=False, add_generation_prompt=True) for x in chunk]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda:0")
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :].float()
    lp = torch.log_softmax(logits, dim=-1)
    y = torch.logsumexp(lp[:, YES], dim=-1)
    n = torch.logsumexp(lp[:, NO], dim=-1)
    scores.extend((y - n).cpu().numpy().tolist())      # log-odds of Yes
    ids.extend(x["instanceID"] for x in chunk)
    if (i // A.bs) % 10 == 0:
        print(f"  {i + len(chunk)}/{len(insts)}", flush=True)

out = Path(__file__).resolve().parents[2] / f"work/llm_exh_{A.split}_{A.condition}.npz"
np.savez(out, score=np.array(scores), ids=np.array(ids))
print(f"wrote {out.name}")

if A.split == "dev":
    from sklearn.metrics import f1_score, roc_auc_score
    gold = np.array([1 if "direct_exhortation" in x["labels"]["st3"] else 0 for x in insts])
    s = np.array(scores)
    print(f"AUC = {roc_auc_score(gold, s):.3f}   (encoder F1 on this flag: 0.335)")
    best = max(((f1_score(gold, (s >= t).astype(int), zero_division=0), t)
                for t in np.percentile(s, np.arange(50, 100, 1))), key=lambda z: z[0])
    print(f"best F1 = {best[0]:.3f} at threshold {best[1]:.2f}   positives predicted: {(s >= best[1]).sum()} / gold {gold.sum()}")
