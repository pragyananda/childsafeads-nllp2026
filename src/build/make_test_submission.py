"""Build the evaluation-phase submission.

Design, each part measured rather than assumed:

* Routed: ST1 and ST2 from the level-4 ensemble, ST3 from the transcript-only
  ensemble. ST3 gains nothing from paid data levels and loses ~0.05 from the
  full product page.
* Six seeds averaged per config. Run-to-run spread on an identical config is
  0.089 on ST1, so averaging beats selecting.
* ST3 keeps fixed 0.5 thresholds, NOT dev-fitted: on channel-grouped held-out
  halves of dev, tuned ST3 thresholds transferred at -0.037 and won 3/30 splits.
* ST2 does NOT keep them. Decomposing that same study per sub-task shows the
  regression was entirely ST3's; ST2 transferred at +0.060 and won 30/30. See
  `st2_thresholds.py` for the study and the fitted vector.
* Hard constraints last: YouTube's paid-promotion flag rules out
  `undisclosed_advertising`; a near-empty transcript means `insufficient_context`.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, clean_st3, _g
from rules import is_short
import submit

P = argparse.ArgumentParser()
P.add_argument("--tag", default="", help="'' for train-only models, '_trdev' for train+dev")
P.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
P.add_argument("--st3-tag", default=None,
               help="separate filename tag for the ST3 arm, so the two arms can be "
                    "varied independently (e.g. '_ep6' for the 6-epoch ST3 models).")
P.add_argument("--out", default=None)
P.add_argument("--st1-tau", type=float, default=0.0,
               help="post-hoc logit adjustment for ST1 class imbalance. The ST1 head "
                    "trains under unweighted cross-entropy while ST2/ST3 get pos_weight, "
                    "so rare classes are under-predicted (`none` recall 0.14). tau=0.3 "
                    "gained +0.051 ST1 on 10 channel-grouped held-out halves, 7/10 positive.")
P.add_argument("--no-flag-t", type=float, default=0.0,
               help="emit no_flag alone when its probability exceeds this. no_flag is "
                    "derived (fires only when nothing else does), so it is starved: "
                    "recall 0.16 against 127 gold. t=0.6 gained +0.031 ST3 on 10/10 "
                    "channel-grouped held-out halves, sd 0.010.")
P.add_argument("--st2-thresholds", default=None,
               help="path to the per-label ST2 threshold vector from st2_thresholds.py. "
                    "Omit for the fixed 0.5 cutoff. The ST2 head trains under BCE "
                    "pos_weight, which moves each label's decision point away from 0.5 by "
                    "a different amount, so 0.5 emits 1.80 labels per instance against a "
                    "gold mean of 1.27. Worth +0.060 on dev ST2, 30/30 held-out splits.")
P.add_argument("--product-gates", action="store_true",
               help="Suppress the two product-family ST3 flags when the predicted ST2 "
                    "category cannot support them. `hfss_food_marketing` is food "
                    "marketing: 37/40 of it sits on a `food` instance in train and 5/5 in "
                    "dev, but the ST3 head reads transcript-only text, cannot see the "
                    "product page, and fires it on fashion and creator_community segments "
                    "(18 predictions against 5 gold). +0.004 ST3 over 60 held-out halves, "
                    "and it never once scored below the ungated system.")
P.add_argument("--disclosure-tiebreak", default="higher", choices=["higher", "keep_undisc"],
               help="which flag survives when both disclosure flags fire. They are "
                    "mutually exclusive by definition and the head over-predicts both "
                    "(113 and 175 against 74 and 118 gold, firing together on 73). "
                    "Always keeping undisclosed_advertising beats keeping the higher "
                    "probability: +0.008 ST3, 19/20 held-out halves.")
P.add_argument("--inadequate-t", type=float, default=0.5,
               help="separate threshold for inadequate_disclosure, the largest remaining "
                    "ST3 deficit (118 instances at F1 0.335). 0.40 is the peak of a flat "
                    "0.35-0.45 plateau; combined with the tie-break, +0.051 ST3 over the "
                    "shipped system on 20/20 held-out halves.")
P.add_argument("--classical-st2", default=None,
               help="path to classical_test_st2.npz. Averaged 50/50 with the encoder's "
                    "ST2 probabilities. On the 2,857-instance CV instrument this is worth "
                    "+0.070 ST2 over the encoder alone, 5/5 folds -- the two families make "
                    "different errors, the encoder reading composition and the classical "
                    "arm reading brand identity through char n-grams over the URL.")
P.add_argument("--llm", default=None,
               help="path to llm_test_<condition>_preds.json; its direct_exhortation "
                    "decision replaces the encoder's (dev: 0.439 -> 0.465 on ST3)")
A = P.parse_args()

W = Path(__file__).resolve().parents[2] / "work"
test = load_split("test")
order = {x["instanceID"]: i for i, x in enumerate(test)}
SEEDS = tuple(A.seeds)
L4 = "probs_test_L4_ModernBERT-large_len2048_seed{s}" + A.tag + ".npz"
L1 = "probs_test_L1_ModernBERT-large_len1024_seed{s}" + (A.st3_tag if A.st3_tag is not None else A.tag) + ".npz"


def avg(pat):
    p1 = np.zeros((len(test), 5)); p2 = np.zeros((len(test), 12)); p3 = np.zeros((len(test), 8))
    used = []
    for s in SEEDS:
        f = W / pat.format(s=s)
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        idx = [order[i] for i in z["ids"].tolist()]
        p1[idx] += z["p1"]; p2[idx] += z["p2"]; p3[idx] += z["p3"]
        used.append(s)
    assert used, f"no probability files matched {pat}"
    print(f"  {pat.format(s='*')}: averaged seeds {used}")
    return p1 / len(used), p2 / len(used), p3 / len(used)


import json
LLM = json.load(open(A.llm)) if A.llm else None
if LLM: print(f"  LLM override loaded: {len(LLM)} instances")

a1, a2, _ = avg(L4)   # ST1, ST2
if A.st1_tau:
    import collections
    from data import load_split as _ls
    _tr = _ls("train") + (_ls("dev") if A.tag == "_trdev" else [])
    _c = collections.Counter(x["labels"]["st1"] for x in _tr)
    _prior = np.array([_c.get(c, 1) / len(_tr) for c in ST1])
    a1 = np.log(np.maximum(a1, 1e-12)) - A.st1_tau * np.log(_prior)
    print(f"  ST1 logit adjustment applied, tau={A.st1_tau}")
_, _, b3 = avg(L1)    # ST3
if A.classical_st2:
    _z = np.load(A.classical_st2, allow_pickle=True)
    _c = np.zeros_like(a2)
    _c[[order[i] for i in _z["ids"].tolist()]] = _z["p"]
    a2 = (a2 + _c) / 2
    print("  ST2: averaged 50/50 with the classical arm")

# Categories that can carry age_restricted_or_prohibited_product. Covers 55/59
# of the flag in train; the excluded categories carry 4 between them.
AR_GATE = {"food", "gambling", "gambling_adjacent", "health", "other"}

th2 = np.full(len(ST2), 0.5)
if A.st2_thresholds:
    th2 = np.load(A.st2_thresholds)
    print("  ST2 thresholds:", {c: round(float(t), 2) for c, t in zip(ST2, th2)})

preds = {}
for i, x in enumerate(test):
    # Fallback picks the largest margin over threshold, not the largest raw
    # probability: with per-label thresholds the two differ, and the margin is
    # the one the tuning was measured under.
    st2 = [c for j, c in enumerate(ST2) if a2[i, j] >= th2[j]] \
        or [ST2[int(np.argmax(a2[i] - th2))]]
    st3 = set(c for j, c in enumerate(ST3) if b3[i, j] >= 0.5)
    if b3[i, ST3.index("inadequate_disclosure")] >= A.inadequate_t:
        st3.add("inadequate_disclosure")
    else:
        st3.discard("inadequate_disclosure")
    if A.product_gates:
        # The flag names a property of the product, so the product category is
        # evidence about it that the transcript-only ST3 head never saw.
        if "food" not in st2:
            st3.discard("hfss_food_marketing")
        if not (set(st2) & AR_GATE):
            st3.discard("age_restricted_or_prohibited_product")
    if _g(x, "video_context", "official_disclosure") == "true":
        st3 -= {"undisclosed_advertising"}
    if "undisclosed_advertising" in st3 and "inadequate_disclosure" in st3:
        st3 -= {"inadequate_disclosure"} if A.disclosure_tiebreak == "keep_undisc" else \
               {"inadequate_disclosure" if b3[i, ST3.index("undisclosed_advertising")]
                >= b3[i, ST3.index("inadequate_disclosure")] else "undisclosed_advertising"}
    if is_short(x):
        st3 = {"insufficient_context"}
    elif A.no_flag_t and b3[i, ST3.index("no_flag")] >= A.no_flag_t:
        st3 = {"no_flag"}
    elif LLM is not None:
        # The encoder cannot read the three-part exhortation test; the prompted
        # model can, and beats it on this flag (0.45 vs 0.34 on dev).
        st3.discard("direct_exhortation")
        if "direct_exhortation" in LLM.get(x["instanceID"], []):
            st3.add("direct_exhortation")
    preds[x["instanceID"]] = {"st1": ST1[int(np.argmax(a1[i]))], "st2": st2,
                              "st3": clean_st3(sorted(st3))}

submit.write(preds, W / (A.out or "SUBMIT_test_routed_ensemble.jsonl"), test)

from collections import Counter
print("\nprediction distribution (sanity check against train priors):")
print("  ST1:", Counter(p["st1"] for p in preds.values()).most_common())
print("  ST2:", Counter(l for p in preds.values() for l in p["st2"]).most_common())
print("  ST3:", Counter(l for p in preds.values() for l in p["st3"]).most_common())
