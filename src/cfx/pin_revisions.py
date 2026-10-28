import argparse
import json

from . import config

WEIGHT_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "*.tiktoken"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--download", nargs="*", metavar="MODEL_KEY")
    args = ap.parse_args(argv)
    from huggingface_hub import HfApi, snapshot_download

    path = config.CONFIGS / "models.json"
    raw = config.load_json(path)
    api = HfApi()
    for key, cfg in config.models().items():
        if key == "fake" or cfg.get("base_url"):
            continue
        sha = api.model_info(cfg["model"], revision=cfg["revision"]).sha
        print(f"{key}: {cfg['model']} @ {cfg['revision']} -> {sha}")
        if args.write:
            raw[key]["revision"] = sha
    if args.write:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)
            f.write("\n")
        print(f"pinned revisions written to {path}")
    for key in args.download or []:
        cfg = config.load_json(path)[key]
        local = snapshot_download(cfg["model"], revision=cfg["revision"], allow_patterns=WEIGHT_PATTERNS)
        print(f"{key}: downloaded to {local}")


if __name__ == "__main__":
    main()
