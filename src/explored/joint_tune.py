"""Joint ST3 threshold search by coordinate ascent on macro-F1.

Per-label tuning treats the eight thresholds as separable. They are not:
`no_flag` and `insufficient_context` are exclusive of every other flag, so
raising one flag's recall silently destroys `no_flag`. Measured consequence of
ignoring this: `direct_exhortation` predicted at 33% against 15% gold,
`inadequate_disclosure` 39% against 23%, and `no_flag` crushed to 7.7% predicted
against 25.2% gold with F1 0.217.

Coordinate ascent optimises the actual objective -- ST3 macro-F1 after the
exclusivity rules and the rule arm have been applied -- instead of eight
independent proxies.

Fitted on dev and applied to test: dev is channel-disjoint from test and unseen
in training, so this is calibration, not leakage.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
from data import ST3, clean_st3, _g
from rules import is_short
import metrics


def decide_st3(p3, insts, th3):
    out = []
    for i, x in enumerate(insts):
        s3 = set(c for j, c in enumerate(ST3) if p3[i, j] >= th3[j])
        if _g(x, "video_context", "official_disclosure") == "true":
            s3 -= {"undisclosed_advertising"}
        if is_short(x):
            s3 = {"insufficient_context"}
        out.append(clean_st3(sorted(s3)))
    return out


def st3_macro(p3, insts, gold, th3):
    return metrics._multilabel(gold, decide_st3(p3, insts, th3), ST3)


def coordinate_ascent(p3, insts, gold, th0=None, rounds=4, grid=None):
    grid = np.arange(0.05, 0.96, 0.025) if grid is None else grid
    th = np.full(len(ST3), 0.5) if th0 is None else th0.copy()
    best = st3_macro(p3, insts, gold, th)
    for r in range(rounds):
        improved = False
        for j in range(len(ST3)):
            cur = th[j]
            for t in grid:
                th[j] = t
                s = st3_macro(p3, insts, gold, th)
                if s > best + 1e-6:
                    best, cur, improved = s, t, True
            th[j] = cur
        print(f"  round {r + 1}: ST3 macro = {best:.4f}")
        if not improved:
            break
    return th, best
