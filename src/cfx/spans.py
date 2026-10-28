import re
from difflib import SequenceMatcher

STOPWORDS = set("""a an the is are was were be been being am of in on at to for from by with and or but
as that this these those it its his her their there here he she they them we you i our your my
while into onto over under near next some any one two three four five""".split())

FUZZY_THRESHOLD = 0.9


def norm(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text):
    return norm(text).split()


def content_tokens(text):
    return [t for t in tokens(text) if t not in STOPWORDS]


def fuzzy_ratio(span, sentence):
    s, t = norm(span), norm(sentence)
    if not s:
        return 0.0
    if s in t:
        return 1.0
    words = t.split()
    n = len(s.split())
    best = 0.0
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(0, max(1, len(words) - size + 1)):
            best = max(best, SequenceMatcher(None, s, " ".join(words[i:i + size])).ratio())
    return best


def check_span(span, own, other):
    if span.strip() and span.strip() in own:
        return "exact"
    if norm(span) and f" {norm(span)} " in f" {norm(own)} ":
        return "normalized"
    if fuzzy_ratio(span, own) >= FUZZY_THRESHOLD:
        return "fuzzy"
    if norm(span) and (f" {norm(span)} " in f" {norm(other)} " or fuzzy_ratio(span, other) >= FUZZY_THRESHOLD):
        return "wrong_side"
    return "absent"


def check_prediction(pred, premise, hypothesis):
    if "premise_spans" not in pred:
        return None
    checks = [{"side": "premise", "span": s, "match": check_span(s, premise, hypothesis)}
              for s in pred.get("premise_spans") or []]
    checks += [{"side": "hypothesis", "span": s, "match": check_span(s, hypothesis, premise)}
               for s in pred.get("hypothesis_spans") or []]
    return checks


def span_report(checks):
    if not checks:
        return "No spans were quoted."
    lines = []
    for c in checks:
        found = c["match"] == "exact"
        lines.append(f'- {c["side"]} span "{c["span"]}": {"found" if found else "NOT found"} in the {c["side"]}')
    return "\n".join(lines)


def token_f1(pred_tokens, ref_tokens):
    pred, ref = set(pred_tokens), set(ref_tokens)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = len(pred & ref)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(ref)
    return 2 * p * r / (p + r)


def highlight_tokens(highlights, side):
    out = []
    for ann in (highlights or {}).values():
        for s in ann.get(f"{side}_spans", []):
            out += content_tokens(s)
    return out


def coverage(target, text):
    t = content_tokens(target)
    if not t:
        return 0.0
    have = set(tokens(text))
    return sum(1 for w in t if w in have) / len(t)
