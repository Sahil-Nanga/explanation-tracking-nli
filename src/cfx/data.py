import json
import re

import pandas as pd

from .config import (ANNOTATIONS, LABELS, MOD_TYPES, PROCESSED, ROOT, read_jsonl,
                     sha256_file, write_jsonl)

DEV_MODS_RAW = ROOT / "result_dev_nilesh.json"
DEV_MODS = ANNOTATIONS / "dev_modifications.json"
EVAL_MODS = ANNOTATIONS / "modifications_all.csv"


def original_item(r):
    return {
        "item_id": f"{r['id']}__original",
        "orig_id": r["id"],
        "variant": "original",
        "split": r["split"],
        "premise": r["premise"],
        "hypothesis": r["hypothesis"],
        "gold_label": r["gold_label"],
        "ref_explanations": r.get("all_reference_explanations", []),
        "ref_highlights": r.get("evidence_highlights", {}),
    }


def variant_item(orig, mod_type, premise, hypothesis, gold, edited_side, span_old, span_new, reason):
    return {
        "item_id": f"{orig['id']}__{mod_type}",
        "orig_id": orig["id"],
        "variant": mod_type,
        "split": orig["split"],
        "premise": premise,
        "hypothesis": hypothesis,
        "gold_label": gold,
        "orig_gold_label": orig["gold_label"],
        "edited_side": edited_side,
        "changed_span_original": span_old,
        "changed_span_new": span_new,
        "label_reason": reason,
    }


def check_variant(item, orig, problems):
    tag = item["item_id"]
    if item["gold_label"] not in LABELS:
        problems.append(f"{tag}: bad gold label {item['gold_label']!r}")
    if (item["premise"], item["hypothesis"]) == (orig["premise"], orig["hypothesis"]):
        problems.append(f"{tag}: text identical to the original")
    same = item["gold_label"] == orig["gold_label"]
    if item["variant"] == "label_change" and same:
        problems.append(f"{tag}: label change keeps the label")
    if item["variant"] != "label_change" and not same:
        problems.append(f"{tag}: {item['variant']} changes the label")


def repair_dev_mods():
    text = DEV_MODS_RAW.read_text(encoding="utf-8")
    fixed = re.sub(r'("status":\s*"[^"]*")\s*\n\s*"":\s*\{', r'\1,\n"provenance": {', text)
    records = json.loads(fixed)
    for r in records:
        r["split"] = "dev"
        if r["modification_type"] == "irrelevant_info":
            r["modification_type"] = "irrelevant_addition"
    with open(DEV_MODS, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1, ensure_ascii=False)
    return records


def dev_variants(originals, problems):
    records = repair_dev_mods() if not DEV_MODS.exists() else json.load(open(DEV_MODS, encoding="utf-8"))
    items = []
    for r in records:
        orig = originals.get(r["original_id"])
        if orig is None:
            problems.append(f"{r['id']}: original {r['original_id']} not in dev split")
            continue
        if r["original_label"] != orig["gold_label"]:
            problems.append(f"{r['id']}: original_label differs from dev.jsonl")
        side = ("premise" if r["premise"] != orig["premise"] else "") + \
               ("hypothesis" if r["hypothesis"] != orig["hypothesis"] else "")
        span = r.get("changed_span") or {}
        item = variant_item(orig, r["modification_type"], r["premise"], r["hypothesis"], r["gold_label"],
                            side or "none", span.get("old", ""), span.get("new", ""), r.get("label_reason", ""))
        check_variant(item, orig, problems)
        items.append(item)
    return items


def eval_variants(originals, problems):
    df = pd.read_csv(EVAL_MODS, dtype=str, keep_default_na=False)
    items = []
    for r in df.to_dict("records"):
        orig = originals[r["orig_id"]]
        if (r["orig_premise"], r["orig_hypothesis"], r["orig_label"]) != \
                (orig["premise"], orig["hypothesis"], orig["gold_label"]):
            problems.append(f"{r['mod_id']}: original fields differ from eval.jsonl")
        if r["status"] != "verified":
            problems.append(f"{r['mod_id']}: status {r['status']!r}, not verified")
        item = variant_item(orig, r["mod_type"], r["new_premise"], r["new_hypothesis"], r["expected_label"],
                            r["edited_side"], r["changed_span_original"], r["changed_span_new"], r["reason"])
        item["mod_author"] = r["author"]
        item["mod_provenance"] = r["provenance"]
        check_variant(item, orig, problems)
        items.append(item)
    return items


def ordered(originals, variants):
    by_orig = {}
    for v in variants:
        by_orig.setdefault(v["orig_id"], []).append(v)
    out = []
    for oid, orig in originals.items():
        out.append(original_item(orig))
        out.extend(sorted(by_orig.get(oid, []), key=lambda v: MOD_TYPES.index(v["variant"])))
    return out


def main():
    problems = []
    dev = {r["id"]: r for r in read_jsonl(PROCESSED / "dev.jsonl")}
    ev = {r["id"]: r for r in read_jsonl(PROCESSED / "eval.jsonl")}

    dev_items = ordered(dev, dev_variants(dev, problems))
    eval_items = ordered(ev, eval_variants(ev, problems))

    selected = {i for i, r in ev.items() if r["selected_for_modification"]}
    modified = {i["orig_id"] for i in eval_items if i["variant"] != "original"}
    if modified != selected:
        problems.append(f"modified originals ({len(modified)}) differ from the selected set ({len(selected)})")
    ids = [i["item_id"] for i in dev_items + eval_items]
    if len(ids) != len(set(ids)):
        problems.append("duplicate item ids")

    write_jsonl(PROCESSED / "dev_items.jsonl", dev_items)
    write_jsonl(PROCESSED / "eval_items.jsonl", eval_items)
    manifest = {name: {"n_items": len(items), "sha256": sha256_file(PROCESSED / f"{name}_items.jsonl"),
                       "variants": pd.Series([i["variant"] for i in items]).value_counts().to_dict()}
                for name, items in [("dev", dev_items), ("eval", eval_items)]}
    with open(PROCESSED / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    for p in problems:
        print("WARN", p)
    for name, m in manifest.items():
        print(f"{name}: {m['n_items']} items {m['variants']}")
    print(f"{len(problems)} warnings")


if __name__ == "__main__":
    main()
