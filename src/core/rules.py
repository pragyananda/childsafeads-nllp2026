"""Rule arm for the labels that are too rare to learn statistically.

Three ST3/ST2 labels score exactly 0.00 from the linear model because they carry
12-59 training examples. Two structural regularities cover most of them:

* `insufficient_context` is essentially a transcript-length property (median 2
  words, against a corpus median of 222). Bag-of-words models cannot represent
  length at all, so no amount of threshold tuning recovers this label.
* `gambling`, `age_restricted_or_prohibited_product` and `hfss_food_marketing`
  are near-deterministic given the outbound domain (mybookie.ag, gfuel.com,
  gamersupps.gg, tryfum.com ...). Splits are channel-disjoint but not
  brand-disjoint, so a gazetteer fitted on train transfers to ~2/3 of dev/test.

Both are fitted on train only and applied unchanged to dev/test.

Measured on dev (F1, linear model -> rule arm): hfss_food_marketing 0.20 -> 0.57
at precision 1.00; gambling_adjacent 0.46 -> 0.72; toys 0.24 -> 0.50;
insufficient_context 0.00 -> 0.40.

A keyword arm was tried for the two labels the gazetteer misses
(`age_restricted_or_prohibited_product`, `gambling`) and is deliberately not
included: a knowledge-based lexicon reached only P=0.09/F1=0.12 on
age_restricted, and degraded further at level 4 because product pages mention
alcohol, betting and age limits incidentally. Those two labels are left to the
encoder.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import domain, _g

SHORT_TRANSCRIPT_WORDS = 10  # train-optimal; dev agrees


def fit_domain_gazetteer(train, labels, task="st3", min_count=2, min_purity=0.5):
    """domain -> set(labels) for domains that reliably carry a rare label."""
    seen, hits = defaultdict(int), defaultdict(lambda: defaultdict(int))
    for x in train:
        d = domain(x)
        if not d:
            continue
        seen[d] += 1
        for l in x["labels"][task]:
            if l in labels:
                hits[d][l] += 1
    gaz = defaultdict(set)
    for d, per in hits.items():
        for l, c in per.items():
            if c >= min_count and c / seen[d] >= min_purity:
                gaz[d].add(l)
    return dict(gaz)


def is_short(inst):
    return len(_g(inst, "transcript", "text").split()) <= SHORT_TRANSCRIPT_WORDS


def apply_gazetteer(inst, gaz):
    return set(gaz.get(domain(inst), set()))
