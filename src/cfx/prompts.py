import hashlib
import json

from .config import PROMPTS

TEMPLATES = {
    "c0": "c0_label_only.txt",
    "c1": "c1_label_first.txt",
    "c2": "c2_evidence_first.txt",
    "c3": "c3_structured.txt",
    "c3n": "c3n_structured_noquote.txt",
    "verify": "verify_revise.txt",
    "generic": "generic_selfcheck.txt",
    "probe": "probe_expl_to_label.txt",
}

SCHEMAS = {
    "c0": ["label"],
    "c1": ["label", "explanation"],
    "c2": ["premise_spans", "hypothesis_spans", "explanation", "label"],
    "c3": ["label", "premise_spans", "hypothesis_spans", "justification", "missing_information"],
    "c3n": ["label", "justification", "missing_information"],
    "verify": ["spans_exist", "supports_label", "adds_unstated_facts", "confuses_missing_with_contradiction",
               "decision", "label", "premise_spans", "hypothesis_spans", "explanation"],
    "generic": ["decision", "label", "premise_spans", "hypothesis_spans", "explanation"],
    "probe": ["label"],
}

FORMAT_REMINDER = ("Your reply could not be read. Reply again with only the JSON object, "
                   "with exactly these keys in this order: {keys}. Use one of the labels "
                   "entailment, neutral, contradiction.")


def _read(name):
    return (PROMPTS / name).read_text(encoding="utf-8")


def system_prompt():
    return _read("system.txt").strip()


def render(kind, fields):
    text = _read(TEMPLATES[kind])
    for key, value in fields.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        text = text.replace("{{" + key + "}}", value)
    if "{{" in text:
        raise ValueError(f"unfilled placeholder in {kind}: {text[text.index('{{'):][:40]}")
    return text.strip()


def messages(kind, fields):
    return [{"role": "system", "content": system_prompt()},
            {"role": "user", "content": render(kind, fields)}]


def reminder(kind):
    return FORMAT_REMINDER.format(keys=", ".join(SCHEMAS[kind]))


def prompt_hashes():
    names = ["system.txt"] + list(TEMPLATES.values())
    return {n: hashlib.sha256(_read(n).encode()).hexdigest()[:16] for n in names}
