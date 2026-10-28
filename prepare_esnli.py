import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SEED = 42
LABELS = ["entailment", "neutral", "contradiction"]
N_DEV, N_EVAL, N_MODIFY = 100, 300, 100


def per_label_counts(total):
    base, extra = divmod(total, len(LABELS))
    return {lab: base + (1 if i < extra else 0) for i, lab in enumerate(LABELS)}


def marked_to_spans(marked):
   
    if not isinstance(marked, str):
        return []
    spans, cur = [], []
    for tok in marked.split():
        if "*" in tok:
            cur.append(tok.replace("*", ""))
        elif cur:
            spans.append(" ".join(cur))
            cur = []
    if cur:
        spans.append(" ".join(cur))
    return spans


def n_words(text):
    return len(str(text).split())


NEG_RE = re.compile(r"\b(no|not|nobody|nothing|never|none|neither|nor|without|n't)\b|n't\b", re.I)
QUANT_RE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|some|all|every|each|"
    r"many|few|several|both|most|any|only|at least|at most)\b", re.I)


def auto_hint(premise, hypothesis):
    text = f"{premise} {hypothesis}"
    hints = []
    if NEG_RE.search(text):
        hints.append("negation?")
    if QUANT_RE.search(text):
        hints.append("quantity?")
    return ";".join(hints)


def load_split(path):
    df = pd.read_csv(path)
    required = ["pairID", "gold_label", "Sentence1", "Sentence2", "Explanation_1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}. Found: {list(df.columns)}")

    df = df.dropna(subset=["Sentence1", "Sentence2", "Explanation_1"])
    df = df[df["gold_label"].isin(LABELS)].copy()
    df["Explanation_1"] = df["Explanation_1"].astype(str)
    df = df[df["Explanation_1"].str.strip() != ""]
    return df.reset_index(drop=True)


def to_record(row, split):
    explanations = []
    evidence = {}
    for k in (1, 2, 3):
        exp_col = f"Explanation_{k}"
        if exp_col in row.index and isinstance(row[exp_col], str) and row[exp_col].strip():
            explanations.append(row[exp_col])
        evidence[f"annotator_{k}"] = {
            "premise_spans": marked_to_spans(row.get(f"Sentence1_marked_{k}")),
            "hypothesis_spans": marked_to_spans(row.get(f"Sentence2_marked_{k}")),
        }
    return {
        "id": row["pairID"],
        "split": split,
        "premise": row["Sentence1"],
        "hypothesis": row["Sentence2"],
        "gold_label": row["gold_label"],
        "reference_explanation": row["Explanation_1"],
        "all_reference_explanations": explanations,
        "evidence_highlights": evidence,
        "auto_hint": auto_hint(row["Sentence1"], row["Sentence2"]),
    }


def inspect(df, name, out_dir):
    df = df.copy()
    df["premise_len"] = df["Sentence1"].map(n_words)
    df["hyp_len"] = df["Sentence2"].map(n_words)
    df["expl_len"] = df["Explanation_1"].map(n_words)
    df["n_prem_hl"] = df["Sentence1_marked_1"].map(lambda m: sum(len(s.split()) for s in marked_to_spans(m)))
    df["n_hyp_hl"] = df["Sentence2_marked_1"].map(lambda m: sum(len(s.split()) for s in marked_to_spans(m)))

    print(f"\n=== {name}: {len(df)} usable rows ===")
    print("Label counts:\n", df["gold_label"].value_counts().to_string())
    print("Unique premises:", df["Sentence1"].nunique())
    print(df[["premise_len", "hyp_len", "expl_len", "n_prem_hl", "n_hyp_hl"]].describe().round(2).to_string())
    print("Rows with any premise highlight:", (df["n_prem_hl"] > 0).mean().round(3))
    print("Rows with any hypothesis highlight:", (df["n_hyp_hl"] > 0).mean().round(3))
    print("Highlighted words in premise, by label (mean):")
    print(df.groupby("gold_label")[["n_prem_hl", "n_hyp_hl"]].mean().round(2).to_string())

    stats_dir = Path(out_dir) / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    df[["premise_len", "hyp_len", "expl_len", "n_prem_hl", "n_hyp_hl"]].describe().to_csv(
        stats_dir / f"{name}_length_stats.csv")

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
    df["gold_label"].value_counts().reindex(LABELS).plot.bar(ax=axes[0], title="Label counts")
    df["premise_len"].plot.hist(ax=axes[1], bins=30, title="Premise length (words)")
    df["hyp_len"].plot.hist(ax=axes[2], bins=30, title="Hypothesis length (words)")
    df["expl_len"].plot.hist(ax=axes[3], bins=30, title="Explanation length (words)")
    plt.tight_layout()
    fig.savefig(stats_dir / f"{name}_overview.png", dpi=150)
    plt.close(fig)


def balanced_sample(df, total, seed, exclude_premises=None):
    pool = df.drop_duplicates(subset="Sentence1")  
    if exclude_premises:
        pool = pool[~pool["Sentence1"].isin(exclude_premises)]
    counts = per_label_counts(total)
    parts = []
    for lab, n in counts.items():
        sub = pool[pool["gold_label"] == lab]
        if len(sub) < n:
            raise ValueError(f"Not enough '{lab}' examples ({len(sub)}) for n={n}")
        parts.append(sub.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esnli_dir", required=True, help="e-SNLI 'dataset' folder containing esnli_dev.csv etc.")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    src = Path(args.esnli_dir)
    out = Path(args.out_dir)

    dev_df = load_split(src / "esnli_dev.csv")
    test_df = load_split(src / "esnli_test.csv")

    inspect(dev_df, "esnli_dev", out)
    inspect(test_df, "esnli_test", out)

    dev_sample = balanced_sample(dev_df, N_DEV, args.seed)
    eval_sample = balanced_sample(test_df, N_EVAL, args.seed)

    modify_counts = per_label_counts(N_MODIFY)
    modify_ids = set()
    for lab, n in modify_counts.items():
        ids = eval_sample[eval_sample["gold_label"] == lab].sample(n=n, random_state=args.seed)["pairID"]
        modify_ids.update(ids)

    dev_records = [to_record(r, "dev") for _, r in dev_sample.iterrows()]
    eval_records = [to_record(r, "eval") for _, r in eval_sample.iterrows()]
    for r in eval_records:
        r["selected_for_modification"] = r["id"] in modify_ids
    for r in dev_records:
        r["selected_for_modification"] = False

    write_jsonl(dev_records, out / "processed" / "dev.jsonl")
    write_jsonl(eval_records, out / "processed" / "eval.jsonl")

    for name, recs in (("dev", dev_records), ("eval", eval_records)):
        safe = [{"id": r["id"], "premise": r["premise"], "hypothesis": r["hypothesis"]} for r in recs]
        write_jsonl(safe, out / "processed" / f"{name}_inputs.jsonl")

    eval_sorted = sorted(eval_records, key=lambda r: not r["selected_for_modification"])
    tag_df = pd.DataFrame(
        [{"id": r["id"], "split": r["split"],
          "selected_for_modification": r["selected_for_modification"],
          "premise": r["premise"], "hypothesis": r["hypothesis"],
          "gold_label": r["gold_label"], "auto_hint": r["auto_hint"],
          "phenomenon": "", "notes": ""} for r in eval_sorted + dev_records]
    )
    tag_path = out / "annotations" / "phenomenon_tagging.csv"
    tag_path.parent.mkdir(parents=True, exist_ok=True)
    tag_df.to_csv(tag_path, index=False)
    print(f"Wrote tagging sheet ({len(tag_df)} rows) -> {tag_path}")

    dev_ids = {r["id"] for r in dev_records}
    eval_ids = {r["id"] for r in eval_records}
    assert not dev_ids & eval_ids, "dev and eval overlap!"
    assert len(dev_records) == N_DEV and len(eval_records) == N_EVAL
    print("\nFinal label counts")
    print("dev :", pd.Series([r["gold_label"] for r in dev_records]).value_counts().to_dict())
    print("eval:", pd.Series([r["gold_label"] for r in eval_records]).value_counts().to_dict())
    print("to modify:", pd.Series([r["gold_label"] for r in eval_records
                                   if r["selected_for_modification"]]).value_counts().to_dict())


if __name__ == "__main__":
    main()
