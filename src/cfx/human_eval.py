import argparse
import random

import numpy as np
import pandas as pd

from . import config, stats
from .analysis import load_probes, load_runs, pair_details

DIR = config.ANNOTATIONS / "human_eval"
TABLES = config.REPORTS / "tables"

DEFAULT_ALLOCATION = {"main_m1_c1": 20, "main_m1_c2": 20, "main_m1_c3": 20, "revise_main_m1_c3_verify": 20,
                      "main_m2_c3": 20}

OUTPUT_QS = {"Q1": ["no", "partly", "yes"], "Q2": ["incorrect", "partly", "correct"],
             "Q3": ["yes", "no"], "Q4": ["none", "some", "all", "n.a."]}
PAIR_QS = {"P1": ["appropriate", "stale", "changed_when_it_shouldnt", "n.a."], "P2": ["yes", "no"]}
ORDINAL = {"Q1": ["no", "partly", "yes"], "Q2": ["incorrect", "partly", "correct"], "Q4": ["none", "some", "all"]}
FAILURE_TAGS = ["correct_pred_invalid_expl", "wrong_pred_fluent_expl", "pred_updated_expl_inconsistent",
                "unsupported_assumption", "irrelevant_sensitive", "missed_evidence_change", "other"]
RATING_COLS = ([f"orig_{q}" for q in OUTPUT_QS] + [f"var_{q}" for q in OUTPUT_QS] + list(PAIR_QS)
               + ["failure_tags", "comments"])


def _evidence(spans_text):
    return spans_text if spans_text and spans_text.strip() else "(none)"


def sample(allocation, seed, calibration=10):
    runs = load_runs()
    missing = [r for r in allocation if r not in runs]
    if missing:
        print("skipping runs that do not exist yet:", ", ".join(missing))
    allocation = {r: n for r, n in allocation.items() if r in runs}
    pairs = pair_details({r: runs[r] for r in allocation}, load_probes(runs), save_as=None)
    rng = random.Random(seed)
    used, chosen = set(), []
    for run_id, n in allocation.items():
        g = pairs[pairs.run_id == run_id]
        cells = {k: list(v.index) for k, v in g.groupby(["mod_type", "orig_gold"])}
        for k in cells:
            rng.shuffle(cells[k])
        keys, picked = sorted(cells), 0
        while picked < n and any(cells.values()):
            for k in keys:
                while cells[k] and picked < n:
                    i = cells[k].pop()
                    pid = (g.loc[i, "orig_id"], g.loc[i, "mod_type"])
                    if pid not in used:
                        used.add(pid)
                        chosen.append(i)
                        picked += 1
                        break
    sel = pairs.loc[chosen].copy()
    items = {it["item_id"]: it for it in config.load_items("eval")[0]}
    sel["pair_uid"] = [f"P{n:03d}" for n in rng.sample(range(1, 1000), len(sel))]
    sel = sel.sample(frac=1, random_state=seed)
    rows = []
    for _, p in sel.iterrows():
        o, v = items[f"{p.orig_id}__original"], items[f"{p.orig_id}__{p.mod_type}"]
        rows.append({"pair_uid": p.pair_uid, "mod_type": p.mod_type,
                     "orig_premise": o["premise"], "orig_hypothesis": o["hypothesis"],
                     "orig_label": p.orig_pred, "orig_evidence": _evidence(p.orig_spans),
                     "orig_explanation": p.orig_explanation,
                     "var_premise": v["premise"], "var_hypothesis": v["hypothesis"],
                     "var_label": p.var_pred, "var_evidence": _evidence(p.var_spans),
                     "var_explanation": p.var_explanation, **{c: "" for c in RATING_COLS}})
    DIR.mkdir(parents=True, exist_ok=True)
    for name in ("sheet_r1.csv", "sheet_r2.csv"):
        if (DIR / name).exists():
            raise SystemExit(f"{DIR / name} exists; move it away first so no ratings are overwritten")
    sheet = pd.DataFrame(rows)
    sheet.to_csv(DIR / "sheet_r1.csv", index=False)
    sheet.to_csv(DIR / "sheet_r2.csv", index=False)
    sel[["pair_uid", "run_id", "model", "system", "orig_id", "mod_type", "orig_gold", "var_gold", "orig_pred",
         "var_pred", "category"]].to_csv(DIR / "key.csv", index=False)
    print(f"sampled {len(sheet)} pairs -> {DIR}")
    print(sel.groupby(["run_id", "mod_type"]).size().unstack(fill_value=0).to_string())
    if calibration:
        _calibration(calibration, seed)


def _calibration(n, seed):
    runs = load_runs()
    dev = [r for r in runs if r.startswith("dev_") and r.endswith("_c3")]
    if not dev:
        print("no dev c3 run yet: calibration sheet skipped")
        return
    pairs = pair_details({dev[0]: runs[dev[0]]}, {}, items="dev", save_as=None)
    pairs = pairs.sample(n=min(n, len(pairs)), random_state=seed)
    items = {it["item_id"]: it for it in config.load_items("dev")[0]}
    rows = []
    for _, p in pairs.iterrows():
        o, v = items[f"{p.orig_id}__original"], items[f"{p.orig_id}__{p.mod_type}"]
        rows.append({"pair_uid": f"CAL{len(rows) + 1:02d}", "mod_type": p.mod_type,
                     "orig_premise": o["premise"], "orig_hypothesis": o["hypothesis"], "orig_label": p.orig_pred,
                     "orig_evidence": _evidence(p.orig_spans), "orig_explanation": p.orig_explanation,
                     "var_premise": v["premise"], "var_hypothesis": v["hypothesis"], "var_label": p.var_pred,
                     "var_evidence": _evidence(p.var_spans), "var_explanation": p.var_explanation,
                     **{c: "" for c in RATING_COLS}})
    pd.DataFrame(rows).to_csv(DIR / "calibration.csv", index=False)
    print(f"calibration sheet ({len(rows)} dev pairs) -> {DIR / 'calibration.csv'}")


def _read(name):
    df = pd.read_csv(DIR / name, dtype=str).fillna("")
    for c in RATING_COLS:
        if c in df and c not in ("failure_tags", "comments"):
            df[c] = df[c].str.strip().str.lower()
    return df.set_index("pair_uid")


def agreement():
    r1, r2 = _read("sheet_r1.csv"), _read("sheet_r2.csv")
    r2 = r2.loc[r1.index]
    rows = []
    questions = [f"{s}_{q}" for s in ("orig", "var") for q in OUTPUT_QS] + list(PAIR_QS)
    for col in questions:
        q = col.split("_")[-1]
        scale = OUTPUT_QS.get(q) or PAIR_QS[q]
        both = (r1[col] != "") & (r2[col] != "")
        bad = sorted({x for x in pd.concat([r1[col], r2[col]]) if x and x not in scale})
        r = {"question": col, "n_both_rated": int(both.sum()),
             "raw_agreement": float((r1[col][both] == r2[col][both]).mean()) if both.any() else np.nan,
             "kappa": stats.cohen_kappa(r1[col][both], r2[col][both], scale),
             "weighted_kappa": (stats.weighted_kappa(r1[col][both], r2[col][both], ORDINAL[q])
                                if q in ORDINAL else np.nan),
             "unknown_values": ";".join(bad)}
        rows.append(r)
    out = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "human_agreement.csv", index=False, float_format="%.3f")
    print(out.to_string(index=False))
    adj = r1.copy()
    for col in questions:
        diff = r1[col] != r2[col]
        adj.loc[diff, col] = ""
        adj[f"{col}__r1"], adj[f"{col}__r2"] = r1[col], r2[col]
    adj["failure_tags__r1"], adj["failure_tags__r2"] = r1.failure_tags, r2.failure_tags
    adj["failure_tags"] = ""
    path = DIR / "adjudicated.csv"
    if path.exists():
        path = DIR / "adjudicated.new.csv"
        print(f"adjudicated.csv exists, not overwriting; wrote {path.name} instead")
    adj.reset_index().to_csv(path, index=False)
    n_open = int(sum((adj[c] == "").sum() for c in questions))
    print(f"{n_open} disagreements left blank for the adjudicator in {path}")


def _prf(pred, gold):
    pred, gold = np.asarray(pred, bool), np.asarray(gold, bool)
    tp = (pred & gold).sum()
    p = tp / pred.sum() if pred.sum() else np.nan
    r = tp / gold.sum() if gold.sum() else np.nan
    return p, r


def results(seed):
    final = _read("adjudicated.csv")
    key = pd.read_csv(DIR / "key.csv", dtype=str).set_index("pair_uid")
    df = final.join(key.drop(columns=["mod_type"]), how="inner")
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    good = {"Q1": "yes", "Q2": "correct", "Q3": "yes", "Q4": "all", "P1": "appropriate", "P2": "yes"}
    for run_id, g in df.groupby("run_id"):
        for side in ("orig", "var"):
            for q in OUTPUT_QS:
                vals = g[f"{side}_{q}"]
                vals = vals[(vals != "") & (vals != "n.a.")]
                est = stats.bootstrap_mean((vals == good[q]).astype(float), g.loc[vals.index, "orig_id"], 2000, seed)
                rows.append({"run_id": run_id, "item": side, "question": q, "share": good[q],
                             "value": est[0], "ci_lo": est[1], "ci_hi": est[2], "n": est[3]})
        for q in PAIR_QS:
            sub = g if q == "P1" else g[g.mod_type == "irrelevant_addition"]
            vals = sub[q][(sub[q] != "") & (sub[q] != "n.a.")]
            if len(vals):
                est = stats.bootstrap_mean((vals == good[q]).astype(float), sub.loc[vals.index, "orig_id"], 2000, seed)
                rows.append({"run_id": run_id, "item": "pair", "question": q, "share": good[q],
                             "value": est[0], "ci_lo": est[1], "ci_hi": est[2], "n": est[3]})
    pd.DataFrame(rows).to_csv(TABLES / "human_results.csv", index=False, float_format="%.3f")

    tags = df.assign(tag=df.failure_tags.str.lower().str.replace(",", ";").str.split(";")).explode("tag")
    tags["tag"] = tags.tag.str.strip()
    tags = tags[tags.tag != ""]
    (tags.groupby(["run_id", "mod_type", "tag"]).size().rename("count").reset_index()
     .to_csv(TABLES / "human_failure_tags.csv", index=False))

    runs = load_runs()
    pairs = pair_details({r: runs[r] for r in df.run_id.unique() if r in runs}, load_probes(runs), save_as=None)
    pairs = pairs.merge(df.reset_index()[["pair_uid", "run_id", "orig_id", "mod_type", "var_Q1", "P1", "P2"]],
                        on=["run_id", "orig_id", "mod_type"])
    val = []
    checks = [("var_probe_consistent", "var_Q1", lambda s: s == "yes", "probe agrees with label ~ human: explanation supports label"),
              ("addresses_change", "P1", lambda s: s == "appropriate", "overlap proxy ~ human: explanation updated appropriately"),
              ("cites_addition", "P2", lambda s: s == "yes", "cites added text ~ human: treats addition as decisive"),
              ("mentions_addition", "P2", lambda s: s == "yes", "mentions added text ~ human: treats addition as decisive")]
    for proxy, human, to_bool, desc in checks:
        if proxy not in pairs:
            continue
        sub = pairs[pairs[proxy].notna() & (pairs[human] != "") & (pairs[human] != "n.a.")]
        if sub.empty:
            continue
        p_, h_ = sub[proxy].astype(bool), sub[human].map(to_bool)
        prec, rec = _prf(p_, h_)
        val.append({"proxy": proxy, "human": human, "description": desc, "n": len(sub),
                    "agreement": float((p_ == h_).mean()), "kappa": stats.cohen_kappa(p_, h_, [True, False]),
                    "precision": prec, "recall": rec})
    pd.DataFrame(val).to_csv(TABLES / "proxy_validation.csv", index=False, float_format="%.3f")
    print(f"human results -> {TABLES}/human_results.csv, human_failure_tags.csv, proxy_validation.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--allocation", help="run_id:n,run_id:n (default: %s)" %
                   ",".join(f"{k}:{v}" for k, v in DEFAULT_ALLOCATION.items()))
    s.add_argument("--calibration", type=int, default=10)
    sub.add_parser("agreement")
    sub.add_parser("results")
    ap.add_argument("--seed", type=int, default=config.experiment()["seed"])
    args = ap.parse_args(argv)
    if args.cmd == "sample":
        alloc = DEFAULT_ALLOCATION
        if args.allocation:
            alloc = {k: int(v) for k, v in (x.split(":") for x in args.allocation.split(","))}
        sample(alloc, args.seed, args.calibration)
    elif args.cmd == "agreement":
        agreement()
    else:
        results(args.seed)


if __name__ == "__main__":
    main()
