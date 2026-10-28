import argparse

from . import config
from .run import execute
from .spans import check_prediction, span_report


def base_view(pred):
    return {
        "label": pred.get("label") or "(no valid label)",
        "premise_spans": pred.get("premise_spans") or [],
        "hypothesis_spans": pred.get("hypothesis_spans") or [],
        "explanation": pred.get("explanation_text") or "(none)",
    }


def finalize(rec, base, item):
    rec["base_label"] = base.get("label")
    rec["base_explanation_text"] = base.get("explanation_text")
    rec["base_premise_spans"] = base.get("premise_spans")
    rec["base_hypothesis_spans"] = base.get("hypothesis_spans")
    rec["base_parse_status"] = base.get("parse_status")
    rec["reviser_label"] = rec.get("label")
    rec["reviser_parse_status"] = rec.get("parse_status")
    revised = rec.get("parse_status") == "ok" and rec.get("decision") == "revise"
    rec["applied_revision"] = revised
    if not revised:
        rec["label"] = base.get("label")
        rec["premise_spans"] = base.get("premise_spans") or []
        rec["hypothesis_spans"] = base.get("hypothesis_spans") or []
        rec["explanation_text"] = base.get("explanation_text")
        rec["span_checks"] = base.get("span_checks")
    else:
        rec["span_checks"] = check_prediction(rec, item["premise"], item["hypothesis"])
    rec["parse_status"] = "ok" if rec["label"] else (base.get("parse_status") or "request_error")
    return rec


def revise_run(model_key, base_run, method, run_id=None, concurrency=None):
    base_dir = config.RUNS / base_run
    base_manifest = config.load_json(base_dir / "manifest.json")
    items, path = config.load_run_items(base_manifest)
    by_id = {it["item_id"]: it for it in items}
    base_preds = config.read_jsonl(base_dir / "predictions.jsonl")
    tasks = []
    for pred in base_preds:
        item = by_id[pred["item_id"]]
        fields = {"premise": item["premise"], "hypothesis": item["hypothesis"], **base_view(pred)}
        if method == "verify":
            fields["span_report"] = span_report(pred.get("span_checks"))
        tasks.append({"item": item, "kind": method, "fields": fields,
                      "extra": {"condition": base_manifest.get("condition"), "method": method,
                                "base_run": base_run}})
    run_id = run_id or f"revise_{base_run}_{method}"
    records = execute(tasks, model_key, run_id, "revise", path,
                      {"condition": base_manifest.get("condition"), "method": method, "base_run": base_run,
                       "items": base_manifest.get("items")}, concurrency=concurrency)
    merged = [finalize(rec, pred, by_id[pred["item_id"]]) for rec, pred in zip(records, base_preds)]
    config.write_jsonl(config.RUNS / run_id / "predictions.jsonl", merged)
    applied = sum(r["applied_revision"] for r in merged)
    print(f"[{run_id}] revisions applied: {applied}/{len(merged)}")
    return merged


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--method", required=True, choices=["verify", "generic"])
    ap.add_argument("--run-id")
    ap.add_argument("--concurrency", type=int)
    args = ap.parse_args(argv)
    revise_run(args.model, args.base_run, args.method, args.run_id, args.concurrency)


if __name__ == "__main__":
    main()
