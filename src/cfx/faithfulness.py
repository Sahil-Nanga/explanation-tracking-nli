import argparse
import csv
import random
from pathlib import Path

from . import config
from .spans import STOPWORDS, content_tokens, norm, tokens
from .stats import cohen_kappa

CSV_PATH = config.ANNOTATIONS / "interventions.csv"
EDIT_KINDS = ["cited", "uncited"]
COLUMNS = (["orig_id", "item_id", "premise", "hypothesis", "gold_label", "pred_label", "edited_side",
            "cited_span", "uncited_span"]
           + [f"{k}_{c}" for k in EDIT_KINDS for c in ("edited_sentence", "gold_r1", "gold_r2", "gold_final")]
           + ["notes"])


def uncited_candidates(sentence, other, cited_spans, length):
    words = sentence.split()
    blocked = set()
    for s in cited_spans:
        blocked |= set(content_tokens(s))
    blocked |= set(content_tokens(other))
    out = []
    for size in sorted({max(1, length - 1), length, length + 1}):
        for i in range(0, len(words) - size + 1):
            window = words[i:i + size]
            toks = tokens(" ".join(window))
            if not toks or toks[0] in STOPWORDS or toks[-1] in STOPWORDS:
                continue
            content = [t for t in toks if t not in STOPWORDS]
            if content and not (set(content) & blocked):
                out.append(" ".join(window).strip(".,;:!?"))
    return out


def select(base_run, n, seed, out=CSV_PATH):
    base_dir = config.RUNS / base_run
    manifest = config.load_json(base_dir / "manifest.json")
    items, _ = config.load_run_items(manifest)
    by_id = {it["item_id"]: it for it in items}
    modified = {it["orig_id"] for it in items if it["variant"] != "original"}
    pool = []
    for pred in config.read_jsonl(base_dir / "predictions.jsonl"):
        item = by_id[pred["item_id"]]
        if item["variant"] != "original" or pred.get("label") != item["gold_label"]:
            continue
        valid = [c for c in pred.get("span_checks") or []
                 if c["match"] in ("exact", "normalized") and content_tokens(c["span"])]
        side = "premise" if any(c["side"] == "premise" for c in valid) else "hypothesis"
        cited = [c["span"] for c in valid if c["side"] == side]
        if not cited:
            continue
        cited_span = max(cited, key=lambda s: len(content_tokens(s)))
        length = min(len(cited_span.split()), 4)
        own, other = (item["premise"], item["hypothesis"]) if side == "premise" else (item["hypothesis"], item["premise"])
        cands = uncited_candidates(own, other, cited, length)
        if not cands:
            continue
        pool.append((item["orig_id"] in modified, item, pred, side, cited_span, cands))
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool.sort(key=lambda p: p[0])
    by_label = {l: [p for p in pool if p[1]["gold_label"] == l] for l in config.LABELS}
    chosen = []
    while len(chosen) < n and any(by_label.values()):
        for l in config.LABELS:
            if by_label[l] and len(chosen) < n:
                chosen.append(by_label[l].pop(0))
    rows = []
    for _, item, pred, side, cited_span, cands in chosen:
        row = {c: "" for c in COLUMNS}
        row.update({"orig_id": item["orig_id"], "item_id": item["item_id"], "premise": item["premise"],
                    "hypothesis": item["hypothesis"], "gold_label": item["gold_label"],
                    "pred_label": pred["label"], "edited_side": side, "cited_span": cited_span,
                    "uncited_span": rng.choice(cands)})
        rows.append(row)
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; move it away first so no human edits are overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"selected {len(rows)} of {len(pool)} eligible originals -> {out}")
    if len(rows) < n:
        print(f"warning: only {len(rows)} eligible (asked for {n})")


def build(path=CSV_PATH, out=None):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    items, skipped, problems = [], 0, []
    for r in rows:
        for kind in EDIT_KINDS:
            edited = (r.get(f"{kind}_edited_sentence") or "").strip()
            gold = (r.get(f"{kind}_gold_final") or "").strip().lower()
            if not edited or not gold:
                skipped += 1
                continue
            side = r.get("edited_side") or "premise"
            if gold not in config.LABELS:
                problems.append(f"{r['orig_id']} {kind}: bad gold label {gold!r}")
                continue
            if norm(edited) == norm(r[side]):
                problems.append(f"{r['orig_id']} {kind}: edited {side} equals the original")
                continue
            span = r[f"{kind}_span"]
            if norm(span) and f" {norm(span)} " not in f" {norm(r[side])} ":
                problems.append(f"{r['orig_id']} {kind}: span not in the {side}")
            pair = {"premise": r["premise"], "hypothesis": r["hypothesis"], side: edited}
            items.append({"item_id": f"{r['orig_id']}__{kind}_edit", "orig_id": r["orig_id"],
                          "variant": f"{kind}_edit", "split": "interventions", **pair,
                          "gold_label": gold, "orig_gold_label": r["gold_label"],
                          "orig_pred_label": r["pred_label"], "gold_changed": gold != r["gold_label"],
                          "edited_side": side, "edited_span": span, "base_item_id": r["item_id"]})
    for p in problems:
        print("problem:", p)
    out = Path(out or config.ITEM_FILES["interventions"])
    config.write_jsonl(out, items)
    print(f"wrote {len(items)} intervention items -> {out} ({skipped} edits not filled in yet)")
    for kind in EDIT_KINDS:
        k = cohen_kappa([r.get(f"{kind}_gold_r1", "").strip().lower() for r in rows],
                        [r.get(f"{kind}_gold_r2", "").strip().lower() for r in rows], config.LABELS)
        print(f"gold-label agreement ({kind} edits): kappa = {round(k, 3)}")
    return items


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--base-run", default="main_m1_c3")
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=config.experiment()["seed"])
    s.add_argument("--out", default=str(CSV_PATH))
    b = sub.add_parser("build")
    b.add_argument("--csv", default=str(CSV_PATH))
    b.add_argument("--out")
    args = ap.parse_args(argv)
    if args.cmd == "select":
        select(args.base_run, args.n, args.seed, args.out)
    else:
        build(args.csv, args.out)


if __name__ == "__main__":
    main()
