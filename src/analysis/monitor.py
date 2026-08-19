"""A monitoring pipeline, not a leaderboard submission.

The shared task asks how an authority would build a system to supervise
commercial content aimed at minors. Our ablation answers a piece of that --
compliance flagging needs only the transcript, product categorisation needs the
crawl -- but a submission file is not a system. This is the system that answer
implies.

    every segment                    cheap: transcript only
        |
        v
    [1] SCREEN  ST3 compliance model, level 1
        |
        +--> confident compliant ............. closed, no cost
        +--> uncertain ....................... ABSTAIN -> human, no crawl
        |
        v  flagged
    [2] TRIAGE  severity x confidence
        |
        v  top of queue, within crawl budget
    [3] ENRICH  crawl the outbound link, ST1/ST2 model, level 4
        |
        v
    [4] QUEUE   severity-ranked, with what-is-being-sold attached

Costs are reported in the units an authority actually budgets: crawls issued and
analyst-segments queued, not GPU-hours.

Design points, each measured rather than assumed:

* Escalation is driven by the *cheap* tier's uncertainty. Crawling the 20% of
  segments it is least sure about retains 89% of what crawling all of them buys;
  choosing at random retains 20%.
* Abstention exists because a compliance tool that cannot say "I don't know"
  routes its own errors into enforcement. `insufficient_context` is a label in
  the taxonomy; we treat low confidence as the same condition.
* Ranking is severity-weighted. Macro-F1 treats a UCPD Annex I blacklisted
  practice exactly like a soft-law HFSS case; an authority does not.

Usage:
    python monitor.py --split dev --crawl-budget 0.2 --abstain 0.15
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST1, ST2, ST3, load_split, clean_st3, _g, domain
from rules import is_short

P = argparse.ArgumentParser()
P.add_argument("--split", default="dev", choices=["dev", "test"])
P.add_argument("--crawl-budget", type=float, default=0.20,
               help="fraction of screened segments that may be enriched (crawled)")
P.add_argument("--abstain", type=float, default=0.15,
               help="fraction routed to a human as 'cannot assess' (0 disables)")
P.add_argument("--no-flag-t", type=float, default=0.6)
P.add_argument("--queue-out", default=None, help="write the review queue as JSONL")
A = P.parse_args()

W = Path(__file__).resolve().parents[2] / "work"
SEV = {"undisclosed_advertising": ("per_se", 3.0), "direct_exhortation": ("per_se", 3.0),
       "inadequate_disclosure": ("conditional", 2.0), "misleading_claim": ("conditional", 2.0),
       "age_restricted_or_prohibited_product": ("conditional", 2.0),
       "hfss_food_marketing": ("soft_law", 1.0)}

insts = load_split(A.split)
order = {x["instanceID"]: i for i, x in enumerate(insts)}
N = len(insts)


def avg(pat, seeds=(0, 1, 2, 3, 4, 5)):
    p1 = np.zeros((N, 5)); p2 = np.zeros((N, 12)); p3 = np.zeros((N, 8)); n = 0
    for s in seeds:
        f = W / pat.format(split=A.split, s=s)
        if f.exists():
            z = np.load(f, allow_pickle=True)
            idx = [order[i] for i in z["ids"].tolist()]
            p1[idx] += z["p1"]; p2[idx] += z["p2"]; p3[idx] += z["p3"]; n += 1
    assert n, f"no cached probabilities for {pat}"
    return p1 / n, p2 / n, p3 / n


# Tier 1 is the only model that sees every segment.
_, _, cheap3 = avg("probs_{split}_L1_ModernBERT-large_len1024_seed{s}.npz")
rich1, rich2, _ = avg("probs_{split}_L4_ModernBERT-large_len2048_seed{s}.npz")

# ---- [1] SCREEN -------------------------------------------------------------
NF = ST3.index("no_flag")
screened = []
for i, x in enumerate(insts):
    flags = set(c for j, c in enumerate(ST3) if cheap3[i, j] >= 0.5)
    if _g(x, "video_context", "official_disclosure") == "true":
        flags -= {"undisclosed_advertising"}
    if {"undisclosed_advertising", "inadequate_disclosure"} <= flags:
        flags -= {"inadequate_disclosure"}
    if is_short(x):
        flags = {"insufficient_context"}
    elif cheap3[i, NF] >= A.no_flag_t:
        flags = {"no_flag"}
    flags = set(clean_st3(sorted(flags)))

    real = flags & set(SEV)
    # Confidence in the screening decision: how far the deciding probabilities
    # sit from the boundary. Near 0.5 everywhere means the model is guessing.
    conf = float(np.abs(cheap3[i, [ST3.index(c) for c in SEV]] - 0.5).max() * 2)
    # Priority is the probability that ANY per se prohibition applies, as a
    # noisy-or over the two blacklisted practices -- not a severity-weighted sum
    # over all flags, which dilutes the per se signal with conditional ones and
    # ranks barely better than chance (1.3x vs 2.8x lift at k=50).
    pu = float(cheap3[i, ST3.index("undisclosed_advertising")])
    pe = float(cheap3[i, ST3.index("direct_exhortation")])
    weight = 1.0 - (1.0 - pu) * (1.0 - pe)
    screened.append({"i": i, "id": x["instanceID"], "flags": flags, "real": real,
                     "conf": conf, "weight": weight,
                     "short": is_short(x), "domain": domain(x)})

# ---- [2] TRIAGE -------------------------------------------------------------
# Abstain on the least confident screenings, whatever they were labelled.
n_abstain = int(round(A.abstain * N))
abstain_ids = {s["id"] for s in sorted(screened, key=lambda s: s["conf"])[:n_abstain]}
for s in screened:
    s["abstained"] = s["id"] in abstain_ids

flagged = [s for s in screened if s["real"] and not s["abstained"]]
compliant = [s for s in screened if not s["real"] and not s["abstained"]]
abstained = [s for s in screened if s["abstained"]]

# ---- [3] ENRICH -------------------------------------------------------------
# Crawl budget spent on the flagged queue first, hardest-to-categorise first:
# knowing *what* is being sold only matters for segments someone will act on.
budget = int(round(A.crawl_budget * N))
st1_margin = np.sort(rich1, axis=1)[:, -1] - np.sort(rich1, axis=1)[:, -2]
rank = sorted(flagged, key=lambda s: (-s["weight"], st1_margin[s["i"]]))
enriched = {s["id"] for s in rank[:budget]}

# ---- [4] QUEUE --------------------------------------------------------------
queue = []
for s in sorted(flagged, key=lambda s: -s["weight"]):
    i = s["i"]
    item = {"instanceID": s["id"], "severity_score": round(s["weight"], 3),
            "flags": sorted(s["real"]),
            "highest_severity": max((SEV[c][0] for c in s["real"]),
                                    key=lambda v: {"per_se": 3, "conditional": 2, "soft_law": 1}[v]),
            "screening_confidence": round(s["conf"], 3),
            "enriched": s["id"] in enriched}
    if s["id"] in enriched:
        item["product_type"] = ST1[int(np.argmax(rich1[i]))]
        item["product_category"] = [c for j, c in enumerate(ST2) if rich2[i, j] >= 0.5] or \
                                   [ST2[int(np.argmax(rich2[i]))]]
        item["destination"] = s["domain"]
    queue.append(item)

per_se = [q for q in queue if q["highest_severity"] == "per_se"]
print(f"""
MONITORING RUN  --  {A.split} split, {N} segments
{'=' * 68}

[1] SCREEN      level 1 (transcript only), every segment
      cost           {N} transcript forward passes, 0 crawls
      compliant      {len(compliant):>4}  ({len(compliant) / N:5.1%})  closed with no further cost
      flagged        {len(flagged):>4}  ({len(flagged) / N:5.1%})
      abstained      {len(abstained):>4}  ({len(abstained) / N:5.1%})  routed to a human, not judged

[2] TRIAGE      severity x confidence
      per se         {len(per_se):>4}  prohibited outright (UCPD Annex I)
      conditional    {len([q for q in queue if q['highest_severity'] == 'conditional']):>4}
      soft law       {len([q for q in queue if q['highest_severity'] == 'soft_law']):>4}

[3] ENRICH      level 4 (crawl), budget {A.crawl_budget:.0%}
      crawls issued  {len(enriched):>4}  ({len(enriched) / N:5.1%} of corpus)
      not crawled    {N - len(enriched):>4}  flags still actionable, product type unknown

[4] QUEUE       {len(queue)} segments for analyst review, severity-ranked
      analyst load   {len(queue) + len(abstained):>4} per {N} screened
                     = {1000 * (len(queue) + len(abstained)) / N:.0f} per 1,000 segments

WHAT THIS COSTS AT PLATFORM SCALE, per 1,000 sponsored segments
      crawls         {1000 * len(enriched) / N:.0f}
      analyst items  {1000 * (len(queue) + len(abstained)) / N:.0f}
      GPU            ~{60 * N / 3360 * 0.2 / N * 1000:.1f} min

CAVEAT. At 80% recall the per se flags run at precision 0.39
(undisclosed advertising) and 0.22 (direct exhortation), so most of this queue
is not a violation. The system triages; it does not adjudicate.
""")

# ---- does the ranking earn its keep? ----------------------------------------
# Flagging 59% of segments is a firehose, not triage: the reference itself marks
# `misleading_claim` on 54%. What makes the system usable is the ORDER of the
# queue, so measure the only thing an analyst with a fixed budget experiences.
if A.split == "dev":
    gold = {x["instanceID"]: set(x["labels"]["st3"]) for x in insts}
    per_se_gold = {i for i, g in gold.items() if g & {"undisclosed_advertising", "direct_exhortation"}}
    ranked = [s["id"] for s in sorted(screened, key=lambda s: -s["weight"])]
    rnd = list(gold); np.random.default_rng(0).shuffle(rnd)
    print(f"REVIEW BUDGET: every segment scored, analyst works down one priority list")
    print(f"  {len(per_se_gold)} of {N} segments ({len(per_se_gold)/N:.1%}) carry a true per se flag\n")
    print(f"  {'reviewed':>9} {'severity-ranked':>17} {'random order':>14} {'lift':>7}")
    for k in (25, 50, 100, 200, len(ranked)):
        k = min(k, len(ranked))
        hit = len([i for i in ranked[:k] if i in per_se_gold])
        rh = len([i for i in rnd[:k] if i in per_se_gold])
        print(f"  {k:>9} {hit:>7} ({hit/k:5.1%}) {rh:>7} ({rh/k:5.1%}) {hit/max(rh,1):>6.2f}x")
    print()
    for k in (50, 100):
        caught = len([i for i in ranked[:k] if i in per_se_gold])
        print(f"  -> top {k} of {N} ({k/N:.0%} of the corpus): {caught}/{len(per_se_gold)} "
              f"= {caught/len(per_se_gold):.0%} of all per se violations, "
              f"at {caught/k:.0%} precision")
    print()

if A.queue_out:
    with open(A.queue_out, "w", encoding="utf-8") as f:
        for q in queue:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"wrote review queue -> {A.queue_out}\n")
