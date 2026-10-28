import argparse

from . import config
from .run import execute


def probe_run(judge_key, base_run, run_id=None, concurrency=None):
    base_dir = config.RUNS / base_run
    base_manifest = config.load_json(base_dir / "manifest.json")
    items, path = config.load_run_items(base_manifest)
    by_id = {it["item_id"]: it for it in items}
    tasks = []
    for pred in config.read_jsonl(base_dir / "predictions.jsonl"):
        expl = (pred.get("explanation_text") or "").strip()
        if not expl:
            continue
        item = by_id[pred["item_id"]]
        tasks.append({"item": item, "kind": "probe",
                      "fields": {"premise": item["premise"], "hypothesis": item["hypothesis"], "explanation": expl},
                      "extra": {"base_run": base_run, "base_model": base_manifest["model_key"],
                                "condition": base_manifest.get("condition"), "predicted_label": pred.get("label")}})
    run_id = run_id or f"probe_{base_run}"
    records = execute(tasks, judge_key, run_id, "probe", path,
                      {"base_run": base_run, "base_model": base_manifest["model_key"],
                       "condition": base_manifest.get("condition"), "items": base_manifest.get("items")},
                      concurrency=concurrency)
    ok = [r for r in records if r["parse_status"] == "ok" and r.get("predicted_label")]
    agree = sum(r["label"] == r["predicted_label"] for r in ok)
    print(f"[{run_id}] probe label = predicted label for {agree}/{len(ok)}")
    return records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--base-run", required=True)
    ap.add_argument("--run-id")
    ap.add_argument("--concurrency", type=int)
    args = ap.parse_args(argv)
    probe_run(args.judge, args.base_run, args.run_id, args.concurrency)


if __name__ == "__main__":
    main()
