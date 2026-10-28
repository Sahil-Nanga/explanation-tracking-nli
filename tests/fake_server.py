import argparse
import hashlib
import json
import os
import re
import socketserver
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LABELS = ["entailment", "neutral", "contradiction"]


def answer(messages):
    user = messages[-1]["content"]
    first_user = next(m["content"] for m in messages if m["role"] == "user")
    h = int(hashlib.sha256(first_user.encode()).hexdigest(), 16)
    if len(messages) == 2 and h % 13 == 0:
        return "I believe the answer is probably entailment."
    keys_line = [l for l in first_user.splitlines() if l.startswith("{")][-1]
    keys = re.findall(r'"(\w+)":', keys_line)
    prem = re.search(r"^Premise: (.*)$", first_user, re.M).group(1)
    hyp = re.search(r"^Hypothesis: (.*)$", first_user, re.M).group(1)
    label = LABELS[h % 3]
    values = {
        "label": label,
        "premise_spans": [" ".join(prem.split()[:3])] if label != "neutral" else [],
        "hypothesis_spans": [" ".join(hyp.split()[:3])],
        "explanation": f"The premise and hypothesis suggest {label}.",
        "justification": f"The quoted spans indicate {label}.",
        "missing_information": "none" if label != "neutral" else "whether the event happens",
        "decision": "keep" if h % 4 else "revise",
        "spans_exist": True, "supports_label": True, "adds_unstated_facts": False,
        "confuses_missing_with_contradiction": False,
    }
    return json.dumps({k: values[k] for k in keys})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send(200, {"data": [{"id": "fake-model"}]})
        elif self.path == "/version":
            self._send(200, {"version": "fake-0"})
        else:
            self._send(404, {})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = answer(req["messages"])
        self._send(200, {"model": req["model"], "choices": [
            {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": sum(len(m["content"].split()) for m in req["messages"]),
                      "completion_tokens": len(text.split())}})


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def get_request(self):
        request, _ = super().get_request()
        return request, ("local", 0)


def serve(port=None, uds=None):
    if uds:
        if os.path.exists(uds):
            os.unlink(uds)
        server = UnixHTTPServer(uds, Handler)
    else:
        server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--uds")
    a = ap.parse_args()
    serve(a.port, a.uds).serve_forever()
