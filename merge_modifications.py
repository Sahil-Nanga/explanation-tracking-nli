import argparse
import json
from pathlib import Path

import pandas as pd

MOD_TYPES = ["label_change", "paraphrase", "irrelevant_addition"]
LABELS = {"entailment", "neutral", "contradiction"}

COLUMNS = [
    "mod_id", "orig_id", "mod_type",
    "orig_premise", "orig_hypothesis", "orig_label",
    "new_premise", "new_hypothesis", "expected_label",
    "edited_side", "changed_span_original", "changed_span_new", "reason",
    "author", "status", "notes", "provenance",
]


def read_csv_any(path, **kw):
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="cp1252", **kw)


def draft_row(base, mt, d, notes):
    row = dict(base)
    if mt == "irrelevant_addition":
        row.update(new_premise=f"{base['orig_premise']} {d['added']}",
                   expected_label=base["orig_label"], edited_side="premise",
                   changed_span_original="", changed_span_new=d["added"])
    elif mt == "paraphrase":
        row.update(new_premise=d["new_premise"], expected_label=base["orig_label"],
                   edited_side="premise", changed_span_original="whole sentence",
                   changed_span_new="whole sentence")
    else:
        row.update({k: d[k] for k in ("expected_label", "edited_side",
                                      "changed_span_original", "changed_span_new")})
        row["new_premise"] = d.get("new_premise", base["orig_premise"])
        row["new_hypothesis"] = d.get("new_hypothesis", base["orig_hypothesis"])
    row.update(reason=d["reason"], author="claude", status="done",
               notes=notes, provenance="claude_draft")
    return row


def validate(df, originals):
    problems = []
    selected = {i for i, r in originals.items() if r["selected_for_modification"]}
    if set(df.orig_id) != selected:
        problems.append(f"orig_id set differs from the {len(selected)} selected eval examples")
    for oid, g in df.groupby("orig_id"):
        if sorted(g.mod_type) != sorted(MOD_TYPES):
            problems.append(f"{oid}: mod types {sorted(g.mod_type)}")
    for r in df.itertuples():
        o = originals[r.orig_id]
        tag = r.mod_id
        if (r.orig_premise, r.orig_hypothesis, r.orig_label) != (o["premise"], o["hypothesis"], o["gold_label"]):
            problems.append(f"{tag}: original fields do not match eval.jsonl")
        if r.expected_label not in LABELS:
            problems.append(f"{tag}: bad expected_label {r.expected_label!r}")
        if (r.new_premise, r.new_hypothesis) == (r.orig_premise, r.orig_hypothesis):
            problems.append(f"{tag}: text unchanged")
        if r.mod_type == "label_change" and r.expected_label == r.orig_label:
            problems.append(f"{tag}: label change keeps the label")
        if r.mod_type != "label_change" and r.expected_label != r.orig_label:
            problems.append(f"{tag}: {r.mod_type} changes the label")
        if r.mod_type == "irrelevant_addition":
            side_orig = r.orig_premise if r.edited_side == "premise" else r.orig_hypothesis
            side_new = r.new_premise if r.edited_side == "premise" else r.new_hypothesis
            other_same = (r.new_hypothesis == r.orig_hypothesis) if r.edited_side == "premise" \
                else (r.new_premise == r.orig_premise)
            if not (side_new.startswith(side_orig.rstrip(".")) and other_same):
                problems.append(f"{tag}: addition does not keep the original text intact")
        if not r.reason.strip():
            problems.append(f"{tag}: missing reason")
        if r.status not in {"done", "verified"}:
            problems.append(f"{tag}: status {r.status!r}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="modifications 1(Sheet 1 - modifications).csv")
    ap.add_argument("--old", default="data/annotations/modifications.csv")
    ap.add_argument("--drafts", default="data/annotations/claude_drafts_remaining34.json")
    ap.add_argument("--eval", default="data/processed/eval.jsonl")
    ap.add_argument("--out", default="data/annotations/modifications_all.csv")
    args = ap.parse_args()

    with open(args.eval, encoding="utf-8") as f:
        originals = {r["id"]: r for r in map(json.loads, f)}

    sheet = read_csv_any(args.sheet, skiprows=1)
    sheet = sheet[sheet.mod_id.str.strip() != ""].copy()
    old = read_csv_any(args.old).set_index("mod_id")
    sheet["provenance"] = [
        old.at[m, "provenance"] if m in old.index and old.at[m, "provenance"] else
        ("human_written" if s != "todo" else "")
        for m, s in zip(sheet.mod_id, sheet.status)
    ]

    with open(args.drafts, encoding="utf-8") as f:
        drafts = {d["orig_id"]: d for d in json.load(f)}

    rows = []
    for r in sheet.to_dict("records"):
        if r["status"] == "todo":
            d = drafts.get(r["orig_id"])
            if d is None:
                raise SystemExit(f"No draft for todo row {r['mod_id']}")
            r = draft_row(r, r["mod_type"], d[r["mod_type"]], d.get("notes", ""))
        rows.append(r)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["orig_id"] = pd.Categorical(df.orig_id, categories=list(dict.fromkeys(df.orig_id)))
    df["mod_type"] = pd.Categorical(df.mod_type, categories=MOD_TYPES)
    df = df.sort_values(["orig_id", "mod_type"]).astype(str)
    for col in df.columns:
        df[col] = df[col].str.strip().str.replace("’", "'")

    problems = validate(df, originals)
    for p in problems:
        print("WARN", p)

    out = Path(args.out)
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows ({df.orig_id.nunique()} examples) -> {out}; "
          f"{(df.provenance == 'claude_draft').sum()} new drafts, {len(problems)} warnings")


if __name__ == "__main__":
    main()
