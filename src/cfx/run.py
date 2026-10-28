import argparse
import difflib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from . import config
from .llm import Cache, Endpoint
from .parse import explanation_text, parse
from .prompts import messages, prompt_hashes, reminder
from .spans import check_prediction


def call_with_format_retry(endpoint, cache, model_cfg, kind, fields, max_tokens, decoding,
                           retries, refresh=False, salt=""):
    msgs = messages(kind, fields)
    attempts, usage = [], {"prompt_tokens": 0, "completion_tokens": 0}
    latency, cache_hits = 0.0, 0
    for attempt in range(retries + 1):
        key = Cache.key(model_cfg, msgs, max_tokens, decoding, salt)
        resp = None if refresh else cache.get(model_cfg, key)
        if resp is None:
            resp = endpoint.chat(model_cfg["model"], msgs, max_tokens, decoding, model_cfg.get("extra_body"))
            cache.put(model_cfg, key, resp)
        else:
            cache_hits += 1
        for k in usage:
            usage[k] += int(resp.get("usage", {}).get(k) or 0)
        latency += resp.get("latency_ms", 0.0)
        parsed = parse(kind, resp["text"])
        attempts.append({"text": resp["text"], "finish_reason": resp.get("finish_reason"),
                         "parse_status": parsed["parse_status"], "provider": resp.get("provider")})
        if parsed["parse_status"] == "ok":
            break
        msgs = msgs + [{"role": "assistant", "content": resp["text"]},
                       {"role": "user", "content": reminder(kind)}]
    parsed["format_retried"] = len(attempts) > 1
    return parsed, {"attempts": attempts, "n_calls": len(attempts), "usage": usage,
                    "latency_ms": round(latency, 1), "cache_hits": cache_hits}


def endpoint_for(model_cfg):
    if model_cfg.get("base_url"):
        key_env = model_cfg.get("api_key_env", "CFX_API_KEY")
        key = os.environ.get(key_env) or os.environ.get("CFX_API_KEY", "")
        if not key:
            raise SystemExit(f"set {key_env} to your API key for {model_cfg['base_url']}")
        return Endpoint(uds="", base_url=model_cfg["base_url"], api_key=key)
    return Endpoint()


def check_model_available(endpoint, model):
    try:
        status, data = endpoint.request("GET", "/models")
        ids = [m.get("id") for m in json.loads(data).get("data", [])] if status == 200 else []
    except (OSError, ValueError):
        return
    if ids and model not in ids:
        close = difflib.get_close_matches(model, ids, n=8, cutoff=0.3)
        raise SystemExit(f"model {model!r} is not offered at {endpoint.describe()}.\n"
                         f"closest ids: {close or ids[:20]}\nedit configs/models.json")


def execute(tasks, model_key, run_id, run_kind, items_path, extra_manifest=None,
            refresh=False, salt="", concurrency=None, log_every=100):
    exp = config.experiment()
    model_cfg = config.models()[model_key]
    endpoint, cache = endpoint_for(model_cfg), Cache()
    if not endpoint.wait_ready(timeout=float(os.environ.get("CFX_WAIT_SECONDS", 60)), interval=2):
        raise SystemExit(f"no model server answering at {endpoint.describe()} (check the URL and API key)")
    check_model_available(endpoint, model_cfg["model"])
    concurrency = concurrency or model_cfg.get("concurrency") or exp["concurrency"]
    run_dir = config.RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    def one(task):
        item, kind = task["item"], task["kind"]
        rec = {"run_id": run_id, "item_id": item["item_id"], "orig_id": item["orig_id"],
               "variant": item["variant"], "model": model_key, "kind": kind}
        rec.update(task.get("extra", {}))
        try:
            parsed, meta = call_with_format_retry(
                endpoint, cache, model_cfg, kind, task["fields"], exp["max_tokens"][kind],
                exp["decoding"], exp["format_retries"], refresh, salt)
        except Exception as e:
            rec.update({"parse_status": "request_error", "label": None, "error": str(e)})
            return rec
        rec.update(parsed)
        rec["explanation_text"] = explanation_text(parsed)
        rec["span_checks"] = check_prediction(parsed, item["premise"], item["hypothesis"])
        rec.update(meta)
        return rec

    records = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(one, t): i for i, t in enumerate(tasks)}
        for done, fut in enumerate(as_completed(futures), 1):
            records[futures[fut]] = fut.result()
            if done % log_every == 0 or done == len(tasks):
                print(f"[{run_id}] {done}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)

    config.write_jsonl(run_dir / "predictions.jsonl", records)
    statuses, providers = {}, {}
    for r in records:
        statuses[r["parse_status"]] = statuses.get(r["parse_status"], 0) + 1
        for a in r.get("attempts", []):
            if a.get("provider"):
                providers[a["provider"]] = providers.get(a["provider"], 0) + 1
    manifest = {
        "run_id": run_id, "run_kind": run_kind, "model_key": model_key, "model": model_cfg,
        "decoding": exp["decoding"], "max_tokens": exp["max_tokens"], "format_retries": exp["format_retries"],
        "cache_salt": salt, "refresh": refresh, "items_file": str(items_path),
        "items_sha256": config.sha256_file(items_path), "n_items": len(records),
        "parse_status": statuses, "providers": providers, "prompt_hashes": prompt_hashes(), "git_commit": config.git_commit(),
        "server": endpoint.server_info(), "started": started,
        "finished": datetime.now(timezone.utc).isoformat(), "wall_seconds": round(time.time() - t0, 1),
        "calls": sum(r.get("n_calls", 0) for r in records),
        "cache_hits": sum(r.get("cache_hits", 0) for r in records),
        "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in records),
        "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in records),
        "slurm": {k: v for k, v in os.environ.items()
                  if k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_GPUS_ON_NODE", "SLURM_NODELIST",
                           "CFX_ENGINE_ARGS")},
    }
    manifest.update(extra_manifest or {})
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[{run_id}] done: {statuses}", flush=True)
    return records


def run_condition(model_key, condition, items_name, run_id=None, limit=None, refresh=False, salt="",
                  concurrency=None):
    items, path = config.load_items(items_name)
    if limit:
        items = items[:limit]
    tasks = [{"item": it, "kind": condition,
              "fields": {"premise": it["premise"], "hypothesis": it["hypothesis"]},
              "extra": {"condition": condition}} for it in items]
    run_id = run_id or f"{items_name}_{model_key}_{condition}"
    return execute(tasks, model_key, run_id, "predict", path, {"condition": condition, "items": items_name},
                   refresh=refresh, salt=salt, concurrency=concurrency)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", required=True, choices=["c0", "c1", "c2", "c3", "c3n"])
    ap.add_argument("--items", default="eval", help="dev, eval, interventions, or a path")
    ap.add_argument("--run-id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--refresh", action="store_true", help="ignore cached responses")
    ap.add_argument("--salt", default="", help="separate cache namespace (e.g. for the noise-floor rerun)")
    ap.add_argument("--concurrency", type=int)
    args = ap.parse_args(argv)
    run_condition(args.model, args.condition, args.items, args.run_id, args.limit, args.refresh,
                  args.salt, args.concurrency)


if __name__ == "__main__":
    sys.exit(main())
