"""Write, validate and zip a submission.

Validation duplicates the organisers' `check_submission.py` rules so a bad file
never reaches the upload form. This matters more than usual here: the evaluation
phase allows only three submissions in total, so a format error is expensive.
"""
import json
import zipfile
from pathlib import Path

from data import ST1, ST2, ST3, ST3_EXCLUSIVE, clean_st3


def validate(preds, target_insts=None):
    errs = []
    for iid, p in preds.items():
        if "st1" in p and p["st1"] not in ST1:
            errs.append(f"{iid}: bad st1 {p['st1']!r}")
        if "st2" in p:
            bad = set(p["st2"]) - set(ST2)
            if bad:
                errs.append(f"{iid}: bad st2 {bad}")
            if not p["st2"]:
                errs.append(f"{iid}: empty st2 scores 0 on every label")
        if "st3" in p:
            bad = set(p["st3"]) - set(ST3)
            if bad:
                errs.append(f"{iid}: bad st3 {bad}")
            if (ST3_EXCLUSIVE & set(p["st3"])) and len(p["st3"]) > 1:
                errs.append(f"{iid}: no_flag/insufficient_context must stand alone")
            if not p["st3"]:
                errs.append(f"{iid}: empty st3 scores 0 on every label")
    if target_insts is not None:
        missing = {x["instanceID"] for x in target_insts} - set(preds)
        if missing:
            errs.append(f"{len(missing)} target instances have no prediction (each scores 0)")
    return errs


def write(preds, out_path, target_insts=None, strict=True):
    """Write submission.jsonl + a same-named .zip. Returns the zip path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for p in preds.values():
        if "st3" in p:
            p["st3"] = clean_st3(p["st3"])
    errs = validate(preds, target_insts)
    if errs:
        print(f"!! {len(errs)} validation problem(s):")
        for e in errs[:10]:
            print("   -", e)
        if strict:
            raise SystemExit("refusing to write an invalid submission")
    order = [x["instanceID"] for x in target_insts] if target_insts else list(preds)
    with out_path.open("w", encoding="utf-8") as f:
        for iid in order:
            if iid in preds:
                f.write(json.dumps({"instanceID": iid, **preds[iid]}, ensure_ascii=False) + "\n")
    zip_path = out_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_path, arcname=out_path.name)
    print(f"wrote {len(preds)} predictions -> {out_path.name} + {zip_path.name}")
    return zip_path
