import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
ANNOTATIONS = DATA / "annotations"
PROMPTS = ROOT / "prompts"
CONFIGS = ROOT / "configs"
OUTPUTS = Path(os.environ.get("CFX_OUTPUTS", ROOT / "outputs"))
RUNS = OUTPUTS / "runs"
CACHE = OUTPUTS / "cache"
REPORTS = Path(os.environ.get("CFX_REPORTS", ROOT / "reports"))

LABELS = ["entailment", "neutral", "contradiction"]
MOD_TYPES = ["label_change", "paraphrase", "irrelevant_addition"]

ITEM_FILES = {
    "dev": PROCESSED / "dev_items.jsonl",
    "eval": PROCESSED / "eval_items.jsonl",
    "interventions": PROCESSED / "intervention_items.jsonl",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def models():
    return {k: v for k, v in load_json(CONFIGS / "models.json").items() if not k.startswith("_")}


def experiment():
    return load_json(CONFIGS / "experiment.json")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None


def load_items(name_or_path):
    path = ITEM_FILES.get(name_or_path, Path(name_or_path))
    return read_jsonl(path), Path(path)


def load_run_items(manifest):
    name = manifest.get("items")
    if name in ITEM_FILES:
        return load_items(name)
    path = Path(manifest["items_file"])
    return load_items(path if path.exists() else PROCESSED / path.name)
