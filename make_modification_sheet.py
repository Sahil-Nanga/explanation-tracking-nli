import argparse
import json
from pathlib import Path

import pandas as pd

MOD_TYPES = ["label_change", "paraphrase", "irrelevant_addition"]

COLUMNS = [
    "mod_id", "orig_id", "mod_type",
    "orig_premise", "orig_hypothesis", "orig_label",
    "new_premise", "new_hypothesis", "expected_label",
    "edited_side", "changed_span_original", "changed_span_new", "reason",
    "author", "status", "notes",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/processed/eval.jsonl")
    ap.add_argument("--out", default="data/annotations/modifications.csv")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to overwrite edited work.")

    with open(args.eval, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    selected = [r for r in records if r["selected_for_modification"]]

    rows = []
    for r in selected:
        for mt in MOD_TYPES:
            rows.append({
                "mod_id": f"{r['id']}__{mt}",
                "orig_id": r["id"],
                "mod_type": mt,
                "orig_premise": r["premise"],
                "orig_hypothesis": r["hypothesis"],
                "orig_label": r["gold_label"],
                "new_premise": r["premise"],
                "new_hypothesis": r["hypothesis"],
                "expected_label": "" if mt == "label_change" else r["gold_label"],
                "edited_side": "", "changed_span_original": "", "changed_span_new": "",
                "reason": "", "author": "", "status": "todo", "notes": "",
            })

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
    print(f"Wrote {len(rows)} rows ({len(selected)} examples x {len(MOD_TYPES)}) -> {out}")


if __name__ == "__main__":
    main()
