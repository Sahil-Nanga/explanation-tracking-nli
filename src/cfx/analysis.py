import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, stats
from .spans import content_tokens, coverage, highlight_tokens, token_f1

LOW_N = 15
VALID_MATCH = ("exact", "normalized")
TABLES = config.REPORTS / "tables"
FIGURES = config.REPORTS / "figures"


def load_runs(runs_dir=config.RUNS):
    out = {}
    for d in sorted(Path(runs_dir).iterdir()):
        if not (d / "manifest.json").exists() or not (d / "predictions.jsonl").exists():
            continue
        m = config.load_json(d / "manifest.json")
        items, _ = config.load_run_items(m)
        item_df = pd.DataFrame(items).set_index("item_id")
        df = pd.DataFrame(config.read_jsonl(d / "predictions.jsonl"))
        keep = [c for c in item_df.columns if c not in df.columns or c in ("premise", "hypothesis")]
        df = df.drop(columns=[c for c in ("premise", "hypothesis") if c in df.columns])
        df = df.join(item_df[keep], on="item_id")
        df["valid"] = df["label"].isin(config.LABELS)
        df["correct"] = (df["label"] == df["gold_label"]).astype(float)
        out[d.name] = (m, df)
    return out


def system_name(m):
    name = m.get("condition") or ""
    if m.get("run_kind") == "revise":
        name += f"+{m['method']}"
    return name


def select(runs, kind, items=None, prefix=None):
    for rid, (m, df) in runs.items():
        if m.get("run_kind") != kind:
            continue
        if items and m.get("items") != items:
            continue
        if prefix and not rid.startswith(prefix):
            continue
        yield rid, m, df


def ci_row(est, lo, hi, n):
    return {"value": est, "ci_lo": lo, "ci_hi": hi, "n": n, "low_n": n < LOW_N}


LATEX_SKIP = {"pairs_detail"}
TEX_ESCAPES = str.maketrans({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                             "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"})


def _tex(v):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.3f}"
    return str(v).translate(TEX_ESCAPES)


def to_latex(df, path):
    lines = [r"\begin{tabular}{" + "l" * len(df.columns) + "}", r"\toprule",
             " & ".join(_tex(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(_tex(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(df, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False, float_format="%.4f")
    if name not in LATEX_SKIP and not df.empty:
        (TABLES / "latex").mkdir(exist_ok=True)
        to_latex(df, TABLES / "latex" / f"{name}.tex")
    return df


def main_results(runs, reps, seed):
    rows = []
    for rid, m, df in itertools.chain(select(runs, "predict", "eval"), select(runs, "revise", "eval")):
        if rid.startswith("noise_"):
            continue
        base = {"run_id": rid, "model": m["model_key"], "system": system_name(m)}
        subsets = {"all": df, "originals": df[df.variant == "original"]}
        subsets.update({v: df[df.variant == v] for v in config.MOD_TYPES})
        for name, sub in subsets.items():
            if sub.empty:
                continue
            acc = stats.bootstrap_mean(sub.correct, sub.orig_id, reps, seed)
            r = {**base, "subset": name, "metric": "accuracy", **ci_row(*acc)}
            rows.append(r)
            if name in ("all", "originals"):
                gold, pred = sub.gold_label.to_numpy(), sub.label.fillna("none").to_numpy()
                f1 = stats.bootstrap_stat(lambda idx: stats.macro_f1(gold[idx], pred[idx], config.LABELS),
                                          len(sub), sub.orig_id.to_numpy(), min(reps, 2000), seed)
                rows.append({**base, "subset": name, "metric": "macro_f1", **ci_row(*f1)})
                rows.append({**base, "subset": name, "metric": "valid_output_rate",
                             **ci_row(*stats.bootstrap_mean(sub.valid.astype(float), sub.orig_id, reps, seed))})
        for lab in config.LABELS:
            sub = df[df.gold_label == lab]
            rows.append({**base, "subset": f"gold={lab}", "metric": "accuracy",
                         **ci_row(*stats.bootstrap_mean(sub.correct, sub.orig_id, reps, seed))})
    return save(pd.DataFrame(rows), "main_results")


def efficiency(runs):
    rows = []
    for rid, (m, df) in runs.items():
        n = max(1, m.get("n_items", len(df)))
        rows.append({"run_id": rid, "run_kind": m.get("run_kind"), "model": m["model_key"],
                     "system": system_name(m), "n_items": m.get("n_items"), "calls": m.get("calls"),
                     "calls_per_item": (m.get("calls") or 0) / n,
                     "prompt_tokens_per_item": (m.get("prompt_tokens") or 0) / n,
                     "completion_tokens_per_item": (m.get("completion_tokens") or 0) / n,
                     "format_retry_rate": df.get("format_retried", pd.Series(dtype=float)).fillna(False).mean(),
                     "wall_seconds": m.get("wall_seconds"), "cache_hits": m.get("cache_hits"),
                     "slurm_job": (m.get("slurm") or {}).get("SLURM_JOB_ID"),
                     "server_version": (m.get("server") or {}).get("server_version")})
    return save(pd.DataFrame(rows), "efficiency")


def _spans(row, side):
    v = row.get(f"{side}_spans")
    return v if isinstance(v, list) else []


def _evidence_text(row):
    spans = _spans(row, "premise") + _spans(row, "hypothesis")
    text = row.get("explanation_text")
    return " ".join(spans), (text if isinstance(text, str) else "")


def _clean_span(s):
    return " ".join(p.strip() for p in str(s or "").split(";"))


def pair_category(mod, orig_ok, var_ok, same_label):
    if mod == "label_change":
        if orig_ok and var_ok:
            return "correct_update"
        if orig_ok and same_label:
            return "insensitive"
        if orig_ok:
            return "wrong_flip"
        return "variant_only" if var_ok else "both_wrong"
    if not same_label:
        return "unstable"
    return "stable_correct" if orig_ok else "stable_wrong"


CATEGORIES = {"label_change": ["correct_update", "insensitive", "wrong_flip", "variant_only", "both_wrong"],
              "paraphrase": ["stable_correct", "stable_wrong", "unstable"],
              "irrelevant_addition": ["stable_correct", "stable_wrong", "unstable"]}


def pair_details(runs, probes, items="eval", save_as="pairs_detail"):
    rows = []
    for rid, m, df in itertools.chain(select(runs, "predict", items), select(runs, "revise", items)):
        if rid.startswith("noise_"):
            continue
        probe = probes.get(rid)
        idx = df.set_index("item_id")
        for _, var in df[df.variant != "original"].iterrows():
            orig = idx.loc[f"{var.orig_id}__original"]
            orig_ok, var_ok = orig.label == orig.gold_label, var.label == var.gold_label
            same = orig.label == var.label
            spans_v, expl_v = _evidence_text(var)
            spans_o, expl_o = _evidence_text(orig)
            new, old = _clean_span(var.get("changed_span_new")), _clean_span(var.get("changed_span_original"))
            r = {"run_id": rid, "model": m["model_key"], "system": system_name(m), "orig_id": var.orig_id,
                 "mod_type": var.variant, "orig_gold": orig.gold_label, "var_gold": var.gold_label,
                 "orig_pred": orig.label, "var_pred": var.label,
                 "category": pair_category(var.variant, orig_ok, var_ok, same),
                 "orig_explanation": expl_o, "var_explanation": expl_v,
                 "orig_spans": spans_o, "var_spans": spans_v,
                 "changed_span_original": old, "changed_span_new": new}
            evidence = f"{spans_v} {expl_v}"
            if var.variant == "label_change":
                r["addresses_change"] = max(coverage(new, evidence), coverage(old, evidence)) >= 0.5
            elif var.variant == "irrelevant_addition":
                r["cites_addition"] = bool(spans_v.strip()) and coverage(new, spans_v) >= 0.5
                r["mentions_addition"] = coverage(new, expl_v) >= 0.5
            else:
                r["explanation_token_f1"] = token_f1(content_tokens(expl_o), content_tokens(expl_v))
            if probe is not None:
                for side, item_id in (("orig", orig.name), ("var", var.item_id)):
                    p = probe.get(item_id)
                    r[f"{side}_probe_label"] = p
                    r[f"{side}_probe_consistent"] = None if p is None else p == (orig.label if side == "orig" else var.label)
            rows.append(r)
    out = pd.DataFrame(rows)
    return save(out, save_as) if save_as else out


def pairs_by_modtype(pairs, reps, seed):
    rows = []
    if pairs.empty:
        return save(pd.DataFrame(rows), "pairs_by_modtype")
    for (rid, model, system, mod), g in pairs.groupby(["run_id", "model", "system", "mod_type"]):
        for cat in CATEGORIES[mod]:
            est = stats.bootstrap_mean((g.category == cat).astype(float), g.orig_id, reps, seed)
            rows.append({"run_id": rid, "model": model, "system": system, "mod_type": mod,
                         "measure": cat, **ci_row(*est)})
        proxies = {"label_change": ["addresses_change"], "irrelevant_addition": ["cites_addition", "mentions_addition"],
                   "paraphrase": ["explanation_token_f1"]}[mod]
        for col in proxies + ["var_probe_consistent"]:
            if col in g and g[col].notna().any() and not (col == "cites_addition" and not g.var_spans.str.strip().any()):
                vals = g[col].astype(float)
                rows.append({"run_id": rid, "model": model, "system": system, "mod_type": mod,
                             "measure": col, **ci_row(*stats.bootstrap_mean(vals, g.orig_id, reps, seed))})
    return save(pd.DataFrame(rows), "pairs_by_modtype")


def span_validity(runs, reps, seed):
    rows = []
    for rid, m, df in itertools.chain(select(runs, "predict", "eval"), select(runs, "revise", "eval")):
        if rid.startswith("noise_") or "span_checks" not in df or df.span_checks.isna().all():
            continue
        checks = [c for cs in df.span_checks.dropna() for c in cs]
        base = {"run_id": rid, "model": m["model_key"], "system": system_name(m)}
        counts = pd.Series([c["match"] for c in checks]).value_counts()
        r = {**base, "n_predictions": len(df), "n_spans": len(checks),
             "spans_per_prediction": len(checks) / max(1, len(df))}
        for k in ("exact", "normalized", "fuzzy", "wrong_side", "absent"):
            r[f"share_{k}"] = counts.get(k, 0) / max(1, len(checks))
        has = df.span_checks.apply(lambda cs: bool(cs))
        all_valid = df.span_checks.apply(lambda cs: float(all(c["match"] in VALID_MATCH for c in cs)) if cs else np.nan)
        est = stats.bootstrap_mean(all_valid, df.orig_id, reps, seed)
        r.update({"share_with_spans": has.mean(), "all_spans_valid": est[0], "all_spans_valid_lo": est[1],
                  "all_spans_valid_hi": est[2]})
        orig = df[(df.variant == "original") & df.ref_highlights.notna()] if "ref_highlights" in df else df.iloc[:0]
        for side in ("premise", "hypothesis"):
            f1 = orig.apply(lambda x: token_f1([t for s in _spans(x, side) for t in content_tokens(s)],
                                               highlight_tokens(x.ref_highlights, side)), axis=1)
            est = stats.bootstrap_mean(f1.astype(float), orig.orig_id, reps, seed) if len(orig) else (np.nan,) * 3 + (0,)
            r.update({f"highlight_f1_{side}": est[0], f"highlight_f1_{side}_lo": est[1],
                      f"highlight_f1_{side}_hi": est[2]})
        rows.append(r)
    return save(pd.DataFrame(rows), "span_validity")


def comparisons(runs, reps, seed):
    main = {(m["model_key"], m["condition"]): df.set_index("item_id")
            for rid, m, df in select(runs, "predict", "eval", "main_")}
    rows = []

    def compare(family, name_a, a, name_b, b):
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        for subset, mask in (("all", np.ones(len(common), bool)), ("originals", (a.variant == "original").to_numpy())):
            d = stats.bootstrap_diff(a.correct[mask], b.correct[mask], a.orig_id[mask], reps, seed)
            r = {"family": family, "a": name_a, "b": name_b, "subset": subset, "metric": "accuracy", **d}
            if subset == "originals":
                n01, n10, p = stats.mcnemar_exact(a.correct[mask] == 1, b.correct[mask] == 1)
                r.update({"mcnemar_only_b_correct": n01, "mcnemar_only_a_correct": n10, "mcnemar_p": p})
            rows.append(r)

    models = sorted({k[0] for k in main})
    conds = config.experiment()["conditions"]
    for model in models:
        for ca, cb in itertools.combinations([c for c in conds if (model, c) in main], 2):
            compare(f"conditions:{model}", f"{model}/{cb}", main[(model, cb)], f"{model}/{ca}", main[(model, ca)])
    for ma, mb in itertools.combinations(models, 2):
        for c in conds:
            if (ma, c) in main and (mb, c) in main:
                compare(f"models:{c}", f"{ma}/{c}", main[(ma, c)], f"{mb}/{c}", main[(mb, c)])
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = np.nan
        for _, g in out.groupby(["family", "subset"]):
            out.loc[g.index, "p_holm"] = stats.holm(g.p.tolist())
    return save(out, "comparisons")


ALIGNMENT_METRICS = [
    ("label_change", "correct_update", lambda g: (g.category == "correct_update").astype(float)),
    ("label_change", "addresses_change", lambda g: g.addresses_change.astype(float)),
    ("paraphrase", "stable_correct", lambda g: (g.category == "stable_correct").astype(float)),
    ("irrelevant_addition", "stable_correct", lambda g: (g.category == "stable_correct").astype(float)),
    ("irrelevant_addition", "mentions_addition", lambda g: g.mentions_addition.astype(float)),
]


def alignment_comparisons(pairs, reps, seed):
    rows = []
    if pairs.empty:
        return save(pd.DataFrame(rows), "alignment_comparisons")
    main = pairs[pairs.run_id.str.startswith("main_")]
    conds = config.experiment()["conditions"]
    for mod, metric, fn in ALIGNMENT_METRICS:
        sub = main[main.mod_type == mod]
        vals = {}
        for (model, system), g in sub.groupby(["model", "system"]):
            if metric in ("addresses_change", "mentions_addition") and (
                    system not in config.experiment()["explanation_conditions"] or metric not in g or g[metric].isna().all()):
                continue
            vals[(model, system)] = fn(g).set_axis(g.orig_id.to_numpy())

        def compare(family, name_a, a, name_b, b):
            common = a.index.intersection(b.index)
            a, b = a.loc[common], b.loc[common]
            d = stats.bootstrap_diff(a, b, common, reps, seed)
            n01, n10, p = stats.mcnemar_exact(a == 1, b == 1)
            rows.append({"family": family, "mod_type": mod, "metric": metric, "a": name_a, "b": name_b,
                         "a_rate": a.mean(), "b_rate": b.mean(), **d,
                         "mcnemar_only_a": n10, "mcnemar_only_b": n01, "mcnemar_p": p})

        models = sorted({k[0] for k in vals})
        for model in models:
            present = [c for c in conds if (model, c) in vals]
            for ca, cb in itertools.combinations(present, 2):
                compare(f"conditions:{model}", f"{model}/{cb}", vals[(model, cb)], f"{model}/{ca}", vals[(model, ca)])
        for ma, mb in itertools.combinations(models, 2):
            for c in conds:
                if (ma, c) in vals and (mb, c) in vals:
                    compare("models", f"{ma}/{c}", vals[(ma, c)], f"{mb}/{c}", vals[(mb, c)])
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = np.nan
        for _, g in out.groupby(["family", "mod_type", "metric"]):
            out.loc[g.index, "p_holm"] = stats.holm(g.p.tolist())
    return save(out, "alignment_comparisons")


def revision(runs, probes, reps, seed):
    rows = []
    for rid, m, df in select(runs, "revise", "eval"):
        base_ok = (df.base_label == df.gold_label)
        final_ok = (df.label == df.gold_label)
        r = {"run_id": rid, "model": m["model_key"], "base": m["condition"], "method": m["method"],
             "n": len(df), "applied_rate": df.applied_revision.mean(),
             "label_changed_rate": (df.label != df.base_label).mean(),
             "wrong_to_right": int((~base_ok & final_ok).sum()), "right_to_wrong": int((base_ok & ~final_ok).sum()),
             "kept_right_rate": final_ok[base_ok].mean() if base_ok.any() else np.nan,
             "base_accuracy": base_ok.mean(), "final_accuracy": final_ok.mean()}
        r["net"] = r["wrong_to_right"] - r["right_to_wrong"]
        d = stats.bootstrap_diff(final_ok.astype(float), base_ok.astype(float), df.orig_id, reps, seed)
        r.update({"acc_diff": d["diff"], "acc_diff_lo": d["lo"], "acc_diff_hi": d["hi"], "acc_diff_p": d["p"]})
        o = df.variant == "original"
        _, _, r["mcnemar_p_originals"] = stats.mcnemar_exact(base_ok[o], final_ok[o])
        base_run = runs.get(m["base_run"])
        if base_run is not None:
            bdf = base_run[1]
            for name, frame in (("base", bdf), ("final", df)):
                v = frame.span_checks.apply(lambda cs: all(c["match"] in VALID_MATCH for c in cs) if cs else np.nan)
                r[f"{name}_all_spans_valid"] = v.astype(float).mean()
        for name, key in (("base", m["base_run"]), ("final", rid)):
            p = probes.get(key)
            if p is not None:
                labels = df.base_label if name == "base" else df.label
                agree = [p.get(i) == l for i, l in zip(df.item_id, labels) if p.get(i) is not None]
                r[f"{name}_probe_consistency"] = float(np.mean(agree)) if agree else np.nan
        r["extra_completion_tokens_per_item"] = (m.get("completion_tokens") or 0) / max(1, len(df))
        rows.append(r)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["acc_diff_p_holm"] = stats.holm(out.acc_diff_p.tolist())
    return save(out, "revision")


def load_probes(runs):
    out = {}
    for rid, m, df in select(runs, "probe"):
        out[m["base_run"]] = {i: (l if v else None) for i, l, v in zip(df.item_id, df.label, df.valid)}
    return out


def probe_table(runs, reps, seed):
    rows = []
    for rid, m, df in select(runs, "probe"):
        ok = df[df.valid & df.predicted_label.notna()]
        consistent = (ok.label == ok.predicted_label).astype(float)
        base = {"run_id": rid, "judge": m["model_key"], "base_run": m["base_run"], "base_model": m.get("base_model")}
        for name, mask in [("all", np.ones(len(ok), bool)),
                           ("prediction_correct", (ok.predicted_label == ok.gold_label).to_numpy()),
                           ("prediction_wrong", (ok.predicted_label != ok.gold_label).to_numpy())] + \
                          [(v, (ok.variant == v).to_numpy()) for v in ["original"] + config.MOD_TYPES]:
            est = stats.bootstrap_mean(consistent[mask], ok.orig_id[mask], reps, seed)
            rows.append({**base, "subset": name, "measure": "probe_consistency", **ci_row(*est)})
        est = stats.bootstrap_mean((ok.label == ok.gold_label).astype(float), ok.orig_id, reps, seed)
        rows.append({**base, "subset": "all", "measure": "probe_label_matches_gold", **ci_row(*est)})
    return save(pd.DataFrame(rows), "probe")


def faithfulness(runs, reps, seed):
    rows, omitted = [], []
    for rid, m, df in select(runs, "predict", "interventions"):
        base = runs.get(f"main_{m['model_key']}_{m['condition']}")
        if base is None:
            continue
        base_pred = base[1].set_index("item_id").label
        df = df.assign(base_pred=df.base_item_id.map(base_pred))
        df["flipped"] = (df.label != df.base_pred).astype(float)
        for (kind, changed), g in df.groupby(["variant", "gold_changed"]):
            for measure, col in (("flip_rate", g.flipped), ("accuracy", g.correct)):
                est = stats.bootstrap_mean(col, g.orig_id, reps, seed)
                rows.append({"run_id": rid, "model": m["model_key"], "system": m["condition"], "edit": kind,
                             "gold_changed": bool(changed), "measure": measure, **ci_row(*est)})
        for _, x in df[(df.variant == "uncited_edit") & df.gold_changed].iterrows():
            omitted.append({"run_id": rid, "orig_id": x.orig_id, "edited_side": x.edited_side,
                            "edited_span": x.edited_span, "premise": x.premise, "hypothesis": x.hypothesis,
                            "orig_gold": x.orig_gold_label, "new_gold": x.gold_label,
                            "base_pred": x.base_pred, "pred_after_edit": x.label})
    save(pd.DataFrame(omitted), "faithfulness_omitted_evidence")
    return save(pd.DataFrame(rows), "faithfulness")


def noise_floor(runs):
    rows = []
    for rid, m, df in select(runs, "predict", "eval", "noise_"):
        main = runs.get(f"main_{m['model_key']}_{m['condition']}")
        if main is None:
            continue
        a, b = main[1].set_index("item_id"), df.set_index("item_id")
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        rows.append({"model": m["model_key"], "condition": m["condition"], "n": len(common),
                     "label_agreement": (a.label == b.label).mean(),
                     "explanation_identical": (a.explanation_text == b.explanation_text).mean(),
                     "accuracy_main": a.correct.mean(), "accuracy_rerun": b.correct.mean()})
    return save(pd.DataFrame(rows), "noise_floor")


def phenomena(runs, pairs, reps, seed):
    path = config.ANNOTATIONS / "phenomenon_tagging.csv"
    if not path.exists():
        return None
    tags = pd.read_csv(path, dtype=str).fillna("")
    tags = tags[tags.phenomenon.str.strip() != ""]
    if tags.empty:
        print("phenomenon tags: none filled in yet, skipping that breakdown")
        return None
    long = tags.assign(phenomenon=tags.phenomenon.str.split(";")).explode("phenomenon")
    long["phenomenon"] = long.phenomenon.str.strip().str.lower()
    long = long[long.phenomenon != ""][["id", "phenomenon"]].rename(columns={"id": "orig_id"})
    rows = []
    for rid, m, df in select(runs, "predict", "eval", "main_"):
        j = df.merge(long, on="orig_id")
        for ph, g in j.groupby("phenomenon"):
            rows.append({"run_id": rid, "model": m["model_key"], "system": m["condition"], "phenomenon": ph,
                         "measure": "accuracy", **ci_row(*stats.bootstrap_mean(g.correct, g.orig_id, reps, seed))})
    if not pairs.empty:
        j = pairs.merge(long, on="orig_id")
        good = {"label_change": "correct_update", "paraphrase": "stable_correct", "irrelevant_addition": "stable_correct"}
        j["pair_success"] = (j.category == j.mod_type.map(good)).astype(float)
        for (rid, model, system, ph), g in j.groupby(["run_id", "model", "system", "phenomenon"]):
            if not rid.startswith("main_"):
                continue
            rows.append({"run_id": rid, "model": model, "system": system, "phenomenon": ph,
                         "measure": "pair_success",
                         **ci_row(*stats.bootstrap_mean(g.pair_success, g.orig_id, reps, seed))})
    return save(pd.DataFrame(rows), "phenomena")


def figures(main, pairs_tab, spans, rev):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures")
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    conds = config.experiment()["conditions"]

    acc = main[(main.metric == "accuracy") & (main.subset.isin(["originals", "all"])) & main.run_id.str.startswith("main_")]
    if not acc.empty:
        models = sorted(acc.model.unique())
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for ax, subset in zip(axes, ["originals", "all"]):
            w = 0.8 / len(models)
            for i, model in enumerate(models):
                s = acc[(acc.model == model) & (acc.subset == subset)].set_index("system").reindex(conds)
                x = np.arange(len(conds)) + i * w
                ax.bar(x, s.value, w, label=model,
                       yerr=[s.value - s.ci_lo, s.ci_hi - s.value], capsize=3)
            ax.set_xticks(np.arange(len(conds)) + w * (len(models) - 1) / 2, conds)
            ax.set_title(f"Accuracy ({subset})")
            ax.set_ylim(0, 1)
        axes[0].legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / "accuracy_by_condition.png", dpi=200)
        plt.close(fig)

    if not pairs_tab.empty:
        cats = pairs_tab[pairs_tab.measure.isin(sum(CATEGORIES.values(), []))]
        cats = cats[cats.run_id.str.startswith("main_")]
        for model, gm in cats.groupby("model"):
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
            for ax, mod in zip(axes, config.MOD_TYPES):
                g = gm[gm.mod_type == mod].pivot(index="system", columns="measure", values="value")
                g = g.reindex([c for c in conds if c in g.index])[CATEGORIES[mod]]
                bottom = np.zeros(len(g))
                for cat in CATEGORIES[mod]:
                    ax.bar(g.index, g[cat], bottom=bottom, label=cat)
                    bottom += g[cat].fillna(0).to_numpy()
                ax.set_title(mod)
                ax.legend(fontsize=7, frameon=False, loc="lower right")
            axes[0].set_ylabel("share of pairs")
            fig.suptitle(f"Counterfactual pair outcomes ({model})")
            fig.tight_layout()
            fig.savefig(FIGURES / f"pair_categories_{model}.png", dpi=200)
            plt.close(fig)

    if not spans.empty:
        s = spans.set_index("run_id")
        parts = ["share_exact", "share_normalized", "share_fuzzy", "share_wrong_side", "share_absent"]
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(s)), 3.8))
        bottom = np.zeros(len(s))
        for p in parts:
            ax.bar(s.index, s[p], bottom=bottom, label=p.replace("share_", ""))
            bottom += s[p].to_numpy()
        ax.set_ylabel("share of quoted spans")
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / "span_validity.png", dpi=200)
        plt.close(fig)

    if not rev.empty:
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(rev)), 3.6))
        names = rev.model + "/" + rev.base + "+" + rev.method
        ax.bar(names, rev.wrong_to_right, label="wrong → right")
        ax.bar(names, -rev.right_to_wrong, label="right → wrong")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("items")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIGURES / "revision_flips.png", dpi=200)
        plt.close(fig)


def fmt(r):
    return f"{r.value:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"


def summary(main, pairs_tab, spans, rev, noise, align=None):
    lines = ["# Results summary", "", "Generated by `python -m cfx.analysis`. 95% CIs from a cluster bootstrap by original example.", ""]
    acc = main[(main.metric == "accuracy") & main.subset.isin(["originals", "all"])]
    if not acc.empty:
        lines += ["## Accuracy", "", "| run | originals | all 600 |", "|---|---|---|"]
        for rid, g in acc.groupby("run_id", sort=False):
            g = g.set_index("subset")
            lines.append(f"| {rid} | {fmt(g.loc['originals'])} | {fmt(g.loc['all'])} |")
        lines.append("")
    if not pairs_tab.empty:
        good = {"label_change": "correct_update", "paraphrase": "stable_correct", "irrelevant_addition": "stable_correct"}
        lines += ["## Counterfactual pairs (share fully correct)", "",
                  "| run | label_change: correct update | paraphrase: stable & correct | irrelevant: stable & correct |",
                  "|---|---|---|---|"]
        for rid, g in pairs_tab.groupby("run_id", sort=False):
            cells = []
            for mod in config.MOD_TYPES:
                r = g[(g.mod_type == mod) & (g.measure == good[mod])]
                cells.append(fmt(r.iloc[0]) if len(r) else "–")
            lines.append(f"| {rid} | " + " | ".join(cells) + " |")
        lines.append("")
    if align is not None and not align.empty:
        sig = align[align.p_holm < 0.05]
        lines += ["## Alignment comparisons (Holm p < 0.05)", "",
                  f"{len(sig)} of {len(align)} paired comparisons in `alignment_comparisons.csv` are significant.", ""]
        if len(sig):
            lines += ["| pair type | measure | a | b | a rate | b rate | diff [CI] | p (Holm) |", "|---|---|---|---|---|---|---|---|"]
            for _, r in sig.iterrows():
                lines.append(f"| {r.mod_type} | {r.metric} | {r.a} | {r.b} | {r.a_rate:.2f} | {r.b_rate:.2f} | "
                             f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | {r.p_holm:.3g} |")
        lines.append("")
    if not spans.empty:
        lines += ["## Quoted evidence", "", "| run | spans exist (exact+normalized) | all spans valid | highlight F1 premise |",
                  "|---|---|---|---|"]
        for _, r in spans.iterrows():
            lines.append(f"| {r.run_id} | {r.share_exact + r.share_normalized:.3f} | {r.all_spans_valid:.3f} | "
                         f"{r.highlight_f1_premise:.3f} |")
        lines.append("")
    if not rev.empty:
        lines += ["## Revision", "", "| run | wrong→right | right→wrong | net | kept right | Δacc [CI] | p (Holm) |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in rev.iterrows():
            lines.append(f"| {r.run_id} | {r.wrong_to_right} | {r.right_to_wrong} | {r.net:+d} | {r.kept_right_rate:.3f} | "
                         f"{r.acc_diff:+.3f} [{r.acc_diff_lo:+.3f}, {r.acc_diff_hi:+.3f}] | {r.acc_diff_p_holm:.3g} |")
        lines.append("")
    if not noise.empty:
        lines += ["## Noise floor (same prompt, rerun)", "", noise.to_markdown(index=False) if _has_tabulate() else noise.to_string(index=False), ""]
    lines += ["Tables: `reports/tables/`. Figures: `reports/figures/`."]
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    (config.REPORTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _has_tabulate():
    try:
        import tabulate
        return True
    except ImportError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=config.experiment()["bootstrap_reps"])
    ap.add_argument("--seed", type=int, default=config.experiment()["seed"])
    ap.add_argument("--runs-dir", default=str(config.RUNS))
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    runs = load_runs(args.runs_dir)
    print(f"loaded {len(runs)} runs")
    probes = load_probes(runs)
    main_t = main_results(runs, args.reps, args.seed)
    efficiency(runs)
    pairs = pair_details(runs, probes)
    pairs_t = pairs_by_modtype(pairs, args.reps, args.seed)
    spans_t = span_validity(runs, args.reps, args.seed)
    comparisons(runs, args.reps, args.seed)
    align_t = alignment_comparisons(pairs, args.reps, args.seed)
    rev_t = revision(runs, probes, args.reps, args.seed)
    probe_table(runs, args.reps, args.seed)
    faithfulness(runs, args.reps, args.seed)
    noise_t = noise_floor(runs)
    phenomena(runs, pairs, args.reps, args.seed)
    figures(main_t, pairs_t, spans_t, rev_t)
    summary(main_t, pairs_t, spans_t, rev_t, noise_t, align_t)
    print(f"tables -> {TABLES}\nfigures -> {FIGURES}\nsummary -> {config.REPORTS / 'summary.md'}")


if __name__ == "__main__":
    main()
