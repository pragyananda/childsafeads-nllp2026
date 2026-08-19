"""Use ST2 to inform ST1.

ST1 is our weakest sub-task (0.5906 on the evaluation set, against a leader's
0.7305 from transcript alone) and ST2 is our strongest (0.7272). On train, the
ST2 label-set alone predicts ST1 with 91.2% accuracy, and the two ST1 classes we
fail on are exactly the ones ST2 localises: `none` carries ST2=other in 29 of 35
cases, `physical_services` concentrates in health.

The combination is a prior, not a fitted model:

    score(c) = log P_model(c | x)  +  lambda * log P_train(c | predicted ST2 set)

P_train comes from the training split only. The single free parameter is lambda,
and it is evaluated at FIXED values across ten channel-grouped halves of dev --
the protocol that made `no_flag` transfer and that we skipped when the ST1 logit
adjustment failed.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, _g

W = Path(__file__).resolve().parents[2] / "work"
train, dv = load_split("train"), load_split("dev")
order = {x["instanceID"]: i for i, x in enumerate(dv)}


def avg(pat, insts, seeds=(0, 1, 2, 3, 4, 5)):
    idx_of = {x["instanceID"]: i for i, x in enumerate(insts)}
    p1 = np.zeros((len(insts), 5)); p2 = np.zeros((len(insts), 12)); n = 0
    for s in seeds:
        f = W / pat.format(s=s)
        if f.exists():
            z = np.load(f, allow_pickle=True)
            ix = [idx_of[i] for i in z["ids"].tolist()]
            p1[ix] += z["p1"]; p2[ix] += z["p2"]; n += 1
    return p1 / n, p2 / n


# --- P(ST1 | ST2 set) from TRAIN only, backing off from exact set to per-label
exact = defaultdict(Counter)
per_label = defaultdict(Counter)
marginal = Counter()
for x in train:
    s2 = tuple(sorted(x["labels"]["st2"]))
    exact[s2][x["labels"]["st1"]] += 1
    for l in x["labels"]["st2"]:
        per_label[l][x["labels"]["st1"]] += 1
    marginal[x["labels"]["st1"]] += 1
MARG = np.array([marginal.get(c, 1) for c in ST1], float)
MARG /= MARG.sum()


def prior(st2_set, min_count=5):
    """P(ST1 | ST2 set): exact set if seen often enough, else average of
    per-label distributions, else the marginal."""
    key = tuple(sorted(st2_set))
    if sum(exact[key].values()) >= min_count:
        c = exact[key]
    else:
        acc = Counter()
        for l in st2_set:
            for k, v in per_label[l].items():
                acc[k] += v
        c = acc if acc else marginal
    v = np.array([c.get(k, 0) for k in ST1], float) + 0.5   # Laplace
    return v / v.sum()


def predict(p1, p2, lam):
    out = []
    for i in range(len(p1)):
        s2 = [c for j, c in enumerate(ST2) if p2[i, j] >= 0.5] or [ST2[int(np.argmax(p2[i]))]]
        sc = np.log(np.maximum(p1[i], 1e-12)) + lam * np.log(prior(s2))
        out.append(ST1[int(np.argmax(sc))])
    return out


P1, P2 = avg("probs_dev_L4_ModernBERT-large_len2048_seed{s}.npz", dv)
gold = [x["labels"]["st1"] for x in dv]

# channel-grouped halves, fixed lambda -- never tuned per split
parts = []
for sd in range(10):
    ch = sorted({_g(x, "channel_context", "channelID") or x["instanceID"] for x in dv})
    rng = np.random.default_rng(sd); rng.shuffle(ch)
    h = set(ch[:len(ch) // 2])
    parts.append([i for i, x in enumerate(dv)
                  if (_g(x, "channel_context", "channelID") or x["instanceID"]) in h])


def score(idx, lam):
    pred = predict(P1[idx], P2[idx], lam)
    g = [gold[i] for i in idx]
    return f1_score(g, pred, labels=sorted(set(g)), average="macro", zero_division=0)


base = [score(p, 0.0) for p in parts]
print(f"{'lambda':>7} {'mean ST1':>9} {'vs base':>9} {'sd':>7} {'wins':>7}")
import statistics as st
for lam in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    v = [score(p, lam) for p in parts]
    d = [a - b for a, b in zip(v, base)]
    w = "-" if lam == 0 else f"{sum(x > 0 for x in d)}/10"
    sd = 0.0 if lam == 0 else st.stdev(d)
    print(f"{lam:>7} {st.mean(v):>9.3f} {st.mean(d):>+9.3f} {sd:>7.3f} {w:>7}")

print(f"\nfull dev, per class:")
for lam in [0.0, 0.75, 1.0]:
    pred = predict(P1, P2, lam)
    per = f1_score(gold, pred, labels=sorted(set(gold)), average=None, zero_division=0)
    m = f1_score(gold, pred, labels=sorted(set(gold)), average="macro", zero_division=0)
    print(f"  lambda={lam}: ST1={m:.3f}  " +
          "  ".join(f"{c.split('_')[0]}={v:.2f}" for c, v in zip(sorted(set(gold)), per)))
