import json
import re

from .config import LABELS
from .prompts import SCHEMAS

LABEL_ALIASES = {
    "entailment": "entailment", "entailed": "entailment", "entails": "entailment", "entail": "entailment",
    "neutral": "neutral", "unknown": "neutral", "undetermined": "neutral",
    "contradiction": "contradiction", "contradicts": "contradiction", "contradictory": "contradiction",
    "contradict": "contradiction",
}


def normalize_label(value):
    if not isinstance(value, str):
        return None
    v = re.sub(r"[^a-z]", "", value.lower())
    return LABEL_ALIASES.get(v)


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def _spans(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "yes", "no"}:
        return value.strip().lower() in {"true", "yes"}
    return None


def parse(kind, text):
    schema = SCHEMAS[kind]
    obj = extract_json(text or "")
    out = {"parse_status": "no_json", "key_order_ok": None, "label": None}
    if obj is None:
        return out
    present = [k for k in obj if k in schema]
    out["key_order_ok"] = present == [k for k in schema if k in obj]
    missing = [k for k in schema if k not in obj]
    label = normalize_label(obj.get("label"))
    out["label"] = label

    if "premise_spans" in schema:
        out["premise_spans"] = _spans(obj.get("premise_spans"))
        out["hypothesis_spans"] = _spans(obj.get("hypothesis_spans"))
    for key in ("explanation", "justification", "missing_information"):
        if key in schema:
            out[key] = str(obj.get(key) or "").strip()
    for key in ("spans_exist", "supports_label", "adds_unstated_facts",
                "confuses_missing_with_contradiction"):
        if key in schema:
            out[key] = _bool(obj.get(key))
    if "decision" in schema:
        d = str(obj.get("decision") or "").strip().lower()
        out["decision"] = d if d in {"keep", "revise"} else None

    if missing:
        out["parse_status"] = "missing_keys"
        out["missing_keys"] = missing
    elif label not in LABELS:
        out["parse_status"] = "bad_label"
    else:
        out["parse_status"] = "ok"
    return out


def explanation_text(pred):
    if pred.get("explanation"):
        return pred["explanation"]
    parts = [pred.get("justification") or ""]
    missing = (pred.get("missing_information") or "").strip()
    if missing and missing.lower() not in {"none", "none.", "n/a", ""}:
        parts.append(f"Missing information: {missing}")
    return " ".join(p for p in parts if p).strip()
