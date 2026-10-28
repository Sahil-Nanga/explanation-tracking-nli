import argparse
import csv
import json
import shutil
import sys
import time
import zipfile

from make_kit import ANN, OWNED, read_csv, source


def merge(zip_path, dry_run=False):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX/")]
        kits = [n for n in names if n.rsplit("/", 1)[-1] == "kit.json"]
        if len(kits) != 1:
            sys.exit(f"{zip_path}: expected one kit.json, found {len(kits)}")
        prefix = kits[0][: -len("kit.json")]
        role = json.loads(z.read(kits[0]))["role"]
        returned = {}
        for rel in OWNED[role]:
            if prefix + rel in names:
                text = z.read(prefix + rel).decode("utf-8-sig")
                reader = csv.DictReader(text.splitlines())
                returned[rel] = list(reader)
    print(f"{zip_path}: answers from {role}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for rel, cols in OWNED[role].items():
        if rel not in returned:
            print(f"  {rel}: not in the zip, skipped")
            continue
        header, rows = read_csv(source(rel))
        key = header[0]
        by_key = {r[key]: r for r in rows}
        unknown = [r[key] for r in returned[rel] if r[key] not in by_key]
        if unknown:
            sys.exit(f"  {rel}: {len(unknown)} rows are not in the project file (e.g. {unknown[0]}); nothing merged")
        changed = cleared = 0
        for r in returned[rel]:
            target = by_key[r[key]]
            for c in cols:
                if c in r and c in target and r[c] != target[c]:
                    changed += 1
                    cleared += r[c] == "" and target[c] != ""
                    if rel == "interventions.csv" and c.endswith("_edited_sentence"):
                        kind = c.split("_")[0]
                        if target.get(f"{kind}_gold_r2") or target.get(f"{kind}_gold_final"):
                            print(f"  ! {r[key]} {kind}: sentence changed after it was labelled; check {kind}_gold_r2/final")
                    target[c] = r[c]
        print(f"  {rel}: {changed} values changed" + (f" ({cleared} cleared)" if cleared else ""))
        if changed and not dry_run:
            dest = ANN / rel
            if dest.exists():
                (ANN / "backups").mkdir(exist_ok=True)
                shutil.copy2(dest, ANN / "backups" / f"{dest.stem}-before-merge-{stamp}.csv")
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                w.writerows(rows)
            tmp.replace(dest)
    if dry_run:
        print("dry run: nothing written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    merge(args.zip, args.dry_run)


if __name__ == "__main__":
    main()
