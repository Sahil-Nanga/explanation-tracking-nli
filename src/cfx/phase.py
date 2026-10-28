import argparse

from . import config
from .probe import probe_run
from .revise import revise_run
from .run import run_condition

PHASES = ["smoke", "dev", "main", "noise", "revise", "probe", "interventions"]


def runs_to_probe(exp):
    out = []
    for d in sorted(config.RUNS.glob("main_*")) + sorted(config.RUNS.glob("revise_main_*")):
        if not (d / "manifest.json").exists():
            continue
        if config.load_json(d / "manifest.json").get("condition") in exp["explanation_conditions"]:
            out.append(d.name)
    return out


def run_phase(phase, model_key, concurrency=None):
    exp = config.experiment()
    if phase == "smoke":
        run_condition(model_key, "c3", "dev", f"smoke_{model_key}_c3", limit=10, concurrency=concurrency)
    elif phase in ("dev", "main"):
        items = "dev" if phase == "dev" else "eval"
        for cond in exp["conditions"]:
            run_condition(model_key, cond, items, f"{phase}_{model_key}_{cond}", concurrency=concurrency)
    elif phase == "noise":
        run_condition(model_key, "c3", "eval", f"noise_{model_key}_c3", salt="noise", concurrency=concurrency)
    elif phase == "revise":
        for cond in exp["revision_bases"]:
            for method in exp["revision_methods"]:
                revise_run(model_key, f"main_{model_key}_{cond}", method, concurrency=concurrency)
    elif phase == "probe":
        for base in runs_to_probe(exp):
            probe_run(model_key, base, concurrency=concurrency)
    elif phase == "interventions":
        if not config.ITEM_FILES["interventions"].exists():
            raise SystemExit("no intervention items yet: run `python -m cfx.faithfulness build` first")
        for cond in exp["intervention_conditions"]:
            run_condition(model_key, cond, "interventions", f"int_{model_key}_{cond}", concurrency=concurrency)
    else:
        raise SystemExit(f"unknown phase {phase!r}; choose from {PHASES}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phases", required=True, help="comma-separated: " + ",".join(PHASES))
    ap.add_argument("--concurrency", type=int)
    args = ap.parse_args(argv)
    for phase in [p.strip() for p in args.phases.split(",") if p.strip()]:
        print(f"=== phase {phase} ({args.model}) ===", flush=True)
        run_phase(phase, args.model, args.concurrency)


if __name__ == "__main__":
    main()
