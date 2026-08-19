"""Build the ST3 arm as encoder + QLoRA-LLM, combined per flag.

The two models fail in different places, and the split is not arbitrary. The
QLoRA-tuned LLM reads the taxonomy's written tests in its prompt and dominates
the disclosure family (undisclosed_advertising 0.850 against the encoder's
0.610, inadequate_disclosure 0.482 against 0.369); it is worse on the six other
flags and 0.051 worse on the ST3 macro. The encoder is the reverse.

So the disclosure flags are averaged across both models and every other flag is
left to the encoder. Averaging all eight instead has the larger dev mean (0.541 against 0.530) and we
do not ship it: it wins 10/20 held-out splits at four times the variance
(sd 0.040 against 0.009), which is the signature of a change that will not
transfer. Restricting the blend to where the mechanism is stateable is what
turns a coin flip into 20/20.

Measured on 20 channel-grouped held-out halves of dev under the shipped
constraint layer, against the encoder-only arm: +0.029 ST3, sd 0.009, 20/20.

Usage:
    python build_st3_hybrid.py --split test --out ../work/st3_hybrid_test.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST3, load_split

P = argparse.ArgumentParser()
P.add_argument("--split", default="test", choices=["dev", "test"])
P.add_argument("--enc-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
P.add_argument("--llm-seeds", type=int, nargs="+", default=[0, 1, 2])
P.add_argument("--out", default=None)
A = P.parse_args()

W = Path(__file__).resolve().parents[2] / "work"
insts = load_split(A.split)
order = {x["instanceID"]: i for i, x in enumerate(insts)}
DISC = [ST3.index("undisclosed_advertising"), ST3.index("inadequate_disclosure")]


def avg(pat, seeds):
    p = np.zeros((len(insts), 8))
    used = []
    for s in seeds:
        f = W / pat.format(split=A.split, s=s)
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        p[[order[i] for i in z["ids"].tolist()]] += z["p3"]
        used.append(s)
    assert used, f"nothing matched {pat}"
    return p / len(used), used


enc, e_used = avg("probs_{split}_L1_ModernBERT-large_len1024_seed{s}_ep6.npz", A.enc_seeds)
llm, l_used = avg("probs_{split}_L2_Qwen2.5-7B-Instruct_len1024_seed{s}_qlora.npz", A.llm_seeds)
print(f"  encoder ep6 seeds {e_used}   QLoRA seeds {l_used}")

p3 = enc.copy()
p3[:, DISC] = (enc[:, DISC] + llm[:, DISC]) / 2

out = Path(A.out or W / f"st3_hybrid_{A.split}.npz")
np.savez(out, p1=np.zeros((len(insts), 5)), p2=np.zeros((len(insts), 12)), p3=p3,
         ids=np.array([x["instanceID"] for x in insts]))
print(f"  wrote {out.name}  ({len(insts)} instances)")
