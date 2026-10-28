import argparse
import csv
import json
import os
import shutil
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

EDITABLE = {
    "new_premise", "new_hypothesis", "expected_label", "edited_side",
    "changed_span_original", "changed_span_new", "reason", "author",
    "status", "notes", "provenance",
}
RESTORABLE = {"orig_premise": "premise", "orig_hypothesis": "hypothesis", "orig_label": "gold_label"}
SINGLE_LINE = {"new_premise", "new_hypothesis", "changed_span_original", "changed_span_new", "reason"}
EXTRA_COLUMNS = ["provenance"]


class Store:
    def __init__(self, csv_path, eval_path):
        self.path = Path(csv_path)
        self.lock = threading.Lock()
        with open(self.path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            self.columns = list(reader.fieldnames)
            self.rows = list(reader)
        for col in EXTRA_COLUMNS:
            if col not in self.columns:
                self.columns.append(col)
                for r in self.rows:
                    r[col] = ""
        self.by_id = {r["mod_id"]: r for r in self.rows}

        self.originals = {}
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.originals[rec["id"]] = {k: rec[k] for k in ("premise", "hypothesis", "gold_label")}

        backup_dir = self.path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(self.path, backup_dir / f"{self.path.stem}-{stamp}.csv")

    def write(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(tmp, self.path)

    def update(self, mod_id, fields):
        with self.lock:
            row = self.by_id.get(mod_id)
            if row is None:
                raise ValueError(f"unknown mod_id {mod_id}")
            orig = self.originals.get(row["orig_id"])
            for key, value in fields.items():
                value = "" if value is None else str(value)
                if key in RESTORABLE:
                    if orig is None or value != orig[RESTORABLE[key]]:
                        raise ValueError(f"{key} can only be restored to the dataset value")
                elif key not in EDITABLE:
                    raise ValueError(f"field {key} is not editable")
                if key in SINGLE_LINE:
                    value = " ".join(value.split("\n")).strip()
                row[key] = value
            self.write()
            return row


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, (HERE / "index.html").read_bytes(), "text/html")
            elif self.path == "/api/data":
                with store.lock:
                    self.send(200, {"columns": store.columns, "rows": store.rows,
                                    "originals": store.originals, "csv_path": str(store.path)})
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/update":
                return self.send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                row = store.update(body["mod_id"], body.get("fields", {}))
                self.send(200, {"ok": True, "row": row})
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                self.send(400, {"ok": False, "error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(PROJECT / "data/annotations/modifications.csv"))
    ap.add_argument("--eval", default=str(PROJECT / "data/processed/eval.jsonl"))
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    store = Store(args.csv, args.eval)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Editing {store.path}\nOpen {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
