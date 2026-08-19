"""Does giving a model the actual legal text improve compliance flagging?

The organisers pose this as an open question and do not score it. We answer it
with a local instruct model on ST3, in three conditions that differ ONLY in how
much legal grounding the prompt carries:

  bare       flag names alone
  taxonomy   + the taxonomy's definitions, including the three-part direct
             exhortation test
  provisions + the instruments and notes from legal_provisions.json

Everything else -- model, decoding, instances, parsing -- is held constant, so
any difference is attributable to the legal context and nothing else.

The comparison target is the fine-tuned encoder at ST3 = 0.427 on the same
instances. A zero-shot 7B model is not expected to beat a model fine-tuned on
2,353 in-domain examples; the question is whether legal grounding moves it, and
whether it helps on `direct_exhortation`, whose definition is a written test an
encoder cannot read.

Usage: python llm_st3.py --condition taxonomy [--limit 100]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST3, load_split, clean_st3, _g, DEV_DIR
from rules import is_short
import metrics

P = argparse.ArgumentParser()
P.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
P.add_argument("--condition", default="taxonomy", choices=["bare", "taxonomy", "provisions"])
P.add_argument("--limit", type=int, default=0, help="0 = all dev instances")
P.add_argument("--batch", type=int, default=8)
P.add_argument("--maxnew", type=int, default=48)
P.add_argument("--split", default="dev", choices=["dev", "test"])
P.add_argument("--shots", type=int, default=0,
               help="in-context demonstrations drawn from train. Zero-shot reasons from "
                    "the legal definitions; demonstrations instead teach the annotation "
                    "policy, which is what the encoder learns from 2,353 examples.")
P.add_argument("--demos", default="knn", choices=["random", "knn"],
               help="random: one fixed set for every instance. knn: per-instance nearest "
                    "neighbours by TF-IDF over transcripts.")
A = P.parse_args()

FLAGS = [f for f in ST3 if f not in ("no_flag", "insufficient_context")]

DEFINITIONS = """\
undisclosed_advertising: The commercial nature of the segment is not identified anywhere available to the viewer: not in the spoken content, not in the description, and not via the platform's paid-promotion label.
inadequate_disclosure: A disclosure exists but fails the child-audience clarity test: buried in description text, phrased in adult jargon, or otherwise not clear and comprehensible to a child. (Mutually exclusive with undisclosed_advertising.)
direct_exhortation: A direct appeal to children to buy the product, or to persuade their parents to buy it for them.
  COUNTS: explicit purchase appeal in the imperative addressed to the audience ("go and buy this", "ask your parents to order it"); or wording that would be a plain instruction where delivery targets the young audience with personal, hyped or pressuring language -- parasocial appeals ("if you love us, please download it"), pleading, repetition, urgency aimed at the viewer, child-directed slang.
  DOES NOT COUNT: basic transactional instructions, even imperative -- "download the app from the link below", "click the link in the description", "use my code for 15% off". Stating where or how to obtain the product, or that a discount exists, is not an exhortation. Friendly encouragement around an instruction ("go give it a try") stays an instruction.
  BOUNDARY: the test is pressure on the child to make the purchase happen, not the presence of an imperative verb. If genuinely ambiguous, do not flag.
misleading_claim: Unsubstantiated or high-risk claims about product characteristics, performance, results or price. Includes any health, weight, fitness, skincare or supplement claim directed at a child audience. Identify claims of this kind; do not verify them against the world.
age_restricted_or_prohibited_product: The promoted product is age-gated: alcohol, tobacco or vaping, gambling, weapons, or similar.
hfss_food_marketing: Marketing of food high in fat, salt or sugar. Clear cases only, such as energy drinks, confectionery and fast food."""


def provisions_block():
    d = json.loads((DEV_DIR / "legal_provisions.json").read_text(encoding="utf-8"))
    out = []
    for flag, meta in d.get("flags", {}).items():
        if flag not in FLAGS:
            continue
        bits = [f"{flag} [severity: {meta.get('severity')}]"]
        for ins in meta.get("instruments", []):
            provs = ", ".join(ins.get("provisions", []))
            bits.append(f"  - {ins.get('instrument')} {provs}: {ins.get('note', '')}")
        out.append("\n".join(bits))
    return "\n".join(out)


def system_prompt(condition):
    head = ("You audit sponsored segments from YouTube videos that reach a child and teenage "
            "audience, under EU consumer law. Two facts are given and must NOT be re-assessed: "
            "the channel is child-facing, and the segment IS commercial.\n\n"
            "Decide which compliance flags apply.")
    tail = ("\n\nAnswer with a JSON array of flag strings and nothing else. Use [] if the segment "
            "appears compliant. Emit every flag that applies.\n"
            f"Valid flags: {', '.join(FLAGS)}")
    if condition == "bare":
        return head + tail
    if condition == "taxonomy":
        return head + "\n\nFLAG DEFINITIONS\n" + DEFINITIONS + tail
    return (head + "\n\nFLAG DEFINITIONS\n" + DEFINITIONS
            + "\n\nLEGAL GROUNDING (EU instruments behind each flag)\n" + provisions_block() + tail)


def user_prompt(x):
    disc = _g(x, "video_context", "official_disclosure") or "unknown"
    desc = _g(x, "video_context", "description")[:1200]
    return (f"TRANSCRIPT OF SPONSORED SEGMENT:\n{_g(x, 'transcript', 'text')[:3500]}\n\n"
            f"VIDEO TITLE: {_g(x, 'video_context', 'title')}\n"
            f"PLATFORM PAID-PROMOTION LABEL: {disc}\n"
            f"VIDEO DESCRIPTION:\n{desc}")


def parse(text, inst):
    m = re.search(r"\[.*?\]", text, re.S)
    got = []
    if m:
        try:
            got = [str(v) for v in json.loads(m.group(0))]
        except Exception:
            got = re.findall(r'"([a-z_]+)"', m.group(0))
    else:  # fall back to bare mentions when the model ignores the format
        got = [f for f in FLAGS if f in text]
    s3 = set(g for g in got if g in FLAGS)

    # Same constraint layer the encoder pipeline gets, so the comparison is of
    # the models and not of their post-processing. The model does not honour the
    # exclusivity rule on its own -- it routinely emits both disclosure flags.
    if _g(inst, "video_context", "official_disclosure") == "true":
        s3 -= {"undisclosed_advertising"}
    if {"undisclosed_advertising", "inadequate_disclosure"} <= s3:
        s3 -= {"inadequate_disclosure"}
    if is_short(inst):
        s3 = {"insufficient_context"}
    return clean_st3(sorted(s3))


dv = load_split(A.split)
if A.limit:
    dv = dv[:A.limit]

# ---- in-context demonstrations -------------------------------------------
DEMOS = {}
if A.shots:
    tr = load_split("train")
    # Only demonstrate the six substantive flags; the two housekeeping labels
    # are decided by rules, not by the model.
    def demo_answer(x):
        return json.dumps([f for f in x["labels"]["st3"] if f in FLAGS])

    def demo_user(x):
        return (f"TRANSCRIPT OF SPONSORED SEGMENT:\n{_g(x, 'transcript', 'text')[:1100]}\n\n"
                f"VIDEO TITLE: {_g(x, 'video_context', 'title')}\n"
                f"PLATFORM PAID-PROMOTION LABEL: "
                f"{_g(x, 'video_context', 'official_disclosure') or 'unknown'}")

    if A.demos == "random":
        rng = __import__("random").Random(0)
        picked = rng.sample(tr, A.shots)
        DEMOS = {x["instanceID"]: picked for x in dv}
        print(f"demonstrations: {A.shots} random, fixed for every instance")
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as _np
        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                              max_features=60000)
        TR = vec.fit_transform([_g(x, "transcript", "text") for x in tr])
        DV = vec.transform([_g(x, "transcript", "text") for x in dv])
        sim = DV @ TR.T
        for i, x in enumerate(dv):
            row = sim.getrow(i).toarray().ravel()
            DEMOS[x["instanceID"]] = [tr[j] for j in _np.argsort(-row)[:A.shots]]
        print(f"demonstrations: {A.shots} nearest neighbours per instance (TF-IDF)")

tok = AutoTokenizer.from_pretrained(A.model, padding_side="left")
# 7B in bf16 is 14.2GB of weights on a 15.5GB card, which OOMs before the KV
# cache exists. NF4 puts it near 5GB and leaves room for a 4k context.
model = AutoModelForCausalLM.from_pretrained(
    A.model, device_map="cuda",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
model.eval()

sysmsg = system_prompt(A.condition)
print(f"condition={A.condition}  system prompt: {len(tok(sysmsg)['input_ids'])} tokens  "
      f"instances={len(dv)}", flush=True)

preds, raw = {}, []
for s in range(0, len(dv), A.batch):
    chunk = dv[s:s + A.batch]
    texts = []
    for x in chunk:
        msgs = [{"role": "system", "content": sysmsg}]
        for d in DEMOS.get(x["instanceID"], []):
            msgs.append({"role": "user", "content": demo_user(d)})
            msgs.append({"role": "assistant", "content": demo_answer(d)})
        msgs.append({"role": "user", "content": user_prompt(x)})
        texts.append(tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True))
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=12288).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=A.maxnew, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    for x, o in zip(chunk, out):
        gen = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        raw.append(gen)
        preds[x["instanceID"]] = {"st3": parse(gen, x)}
    if s % (A.batch * 10) == 0:
        print(f"  {s + len(chunk)}/{len(dv)}", flush=True)

W = Path(__file__).resolve().parents[2] / "work"
(W / f"llm_{A.split}_{A.condition}{A.shots and f'_{A.demos}{A.shots}' or ''}_raw.txt").write_text("\n---\n".join(raw), encoding="utf-8")
json.dump({k: v["st3"] for k, v in preds.items()},
          (W / f"llm_{A.split}_{A.condition}{A.shots and f'_{A.demos}{A.shots}' or ''}_preds.json").open("w"), indent=1)

if A.split == "test":
    print(f"wrote {len(preds)} test predictions; no labels to score against")
    raise SystemExit

gold = [x["labels"]["st3"] for x in dv]
pred = [preds[x["instanceID"]]["st3"] for x in dv]
st3 = metrics._multilabel(gold, pred, ST3)
fam = metrics._multilabel(
    [{metrics.ST3_FAMILY[l] for l in g if l in metrics.ST3_FAMILY} for g in gold],
    [{metrics.ST3_FAMILY[l] for l in p if l in metrics.ST3_FAMILY} for p in pred],
    ["disclosure", "content", "product"])
print(f"\n=== condition={A.condition} shots={A.shots} demos={A.demos if A.shots else '-'} ===")
print(f"ST3 macro-F1 {st3:.3f}   family {fam:.3f}   (encoder baseline 0.427 / 0.645)")
print("per-flag:", {k: round(v, 2) for k, v in metrics.per_label_f1(gold, pred, ST3).items()})
