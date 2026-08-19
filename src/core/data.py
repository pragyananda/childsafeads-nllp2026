"""Loading and view construction for ChildSafeAds.

Two things the shipped starting kit gets wrong, both handled here:

1. `dev.jsonl` is not strictly line-delimited. 504 instances span 521 physical
   lines, and 6 of those lines fail `json.loads`. The official `load_data.py`
   crashes on it; naive line-parsing silently returns 503/504. We stream with
   `raw_decode` instead, which is correct for all three splits.
2. Any of the four level groups can be missing or empty on a given instance
   (40 train transcripts are empty), so every accessor tolerates absence.
"""
import json
from pathlib import Path
from urllib.parse import urlparse

ST1 = ["physical_goods", "digital_content_or_services", "physical_services", "none", "other"]
ST2 = ["toys", "food", "apps", "hardware_electronics", "fashion", "health", "education",
       "financial", "gambling", "gambling_adjacent", "creator_community", "other"]
ST3 = ["undisclosed_advertising", "inadequate_disclosure", "direct_exhortation",
       "misleading_claim", "age_restricted_or_prohibited_product", "hfss_food_marketing",
       "no_flag", "insufficient_context"]
ST3_EXCLUSIVE = {"no_flag", "insufficient_context"}
ST3_FAMILY = {"undisclosed_advertising": "disclosure", "inadequate_disclosure": "disclosure",
              "direct_exhortation": "content", "misleading_claim": "content",
              "age_restricted_or_prohibited_product": "product", "hfss_food_marketing": "product"}

ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = ROOT / "data" / "public_data_dev"
TEST_DIR = ROOT / "data" / "public_data_test"


def load(path):
    """Stream concatenated JSON objects. Tolerates objects spanning several lines."""
    s = Path(path).read_text(encoding="utf-8")
    dec, out, i, n = json.JSONDecoder(), [], 0, len(s)
    while i < n:
        while i < n and s[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, i = dec.raw_decode(s, i)
        out.append(obj)
    return out


def load_split(name):
    return load({"train": DEV_DIR / "train.jsonl", "dev": DEV_DIR / "dev.jsonl",
                 "test": TEST_DIR / "test.jsonl"}[name])


def _g(inst, group, field, default=""):
    return ((inst.get(group) or {}).get(field) or default)


def domain(inst):
    """Registrable-ish host of the resolved outbound link; '' when absent."""
    u = _g(inst, "product_page", "resolved_url") or _g(inst, "product_page", "raw_url")
    return urlparse(u).netloc.lower().removeprefix("www.") if u else ""


def view(inst, level=4, page_chars=4000, desc_chars=3000):
    """Text a system sees at a given data-access level.

    level 1 transcript only | 2 + video metadata | 3 + channel | 4 + product page.
    Field markers are explicit so a model can learn where a disclosure appeared,
    which is the distinction ST3 turns on.
    """
    parts = [f"[TRANSCRIPT] {_g(inst, 'transcript', 'text')}"]
    if level >= 2:
        parts.append(f"[TITLE] {_g(inst, 'video_context', 'title')}")
        parts.append(f"[PAID_PROMOTION_FLAG] {_g(inst, 'video_context', 'official_disclosure') or 'unknown'}")
        parts.append(f"[DESCRIPTION] {_g(inst, 'video_context', 'description')[:desc_chars]}")
    if level >= 3:
        parts.append(f"[CHANNEL] {_g(inst, 'channel_context', 'channel_name')}")
    if level >= 4:
        parts.append(f"[LINK] {domain(inst)}")
        parts.append(f"[PAGE_TITLE] {_g(inst, 'product_page', 'page_title')}")
        parts.append(f"[PAGE] {_g(inst, 'product_page', 'text')[:page_chars]}")
    return "\n".join(parts)


def clean_st3(flags):
    """Enforce the taxonomy's exclusivity rules; never emit an empty flag list."""
    f = list(dict.fromkeys(flags))
    real = [x for x in f if x not in ST3_EXCLUSIVE]
    if real:
        return real
    for x in ("no_flag", "insufficient_context"):
        if x in f:
            return [x]
    return ["no_flag"]
