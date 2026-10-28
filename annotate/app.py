import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
KIT = json.loads((HERE / "kit.json").read_text(encoding="utf-8")) if (HERE / "kit.json").exists() else None
if KIT:
    ANN = HERE / "data"
    LABELS, OUTPUT_QS, PAIR_QS, FAILURE_TAGS = KIT["labels"], KIT["output_qs"], KIT["pair_qs"], KIT["failure_tags"]
else:
    sys.path.insert(0, str(PROJECT / "src"))
    from cfx import config
    from cfx.human_eval import FAILURE_TAGS, OUTPUT_QS, PAIR_QS
    ANN = Path(os.environ.get("CFX_ANNOTATIONS", config.ANNOTATIONS))
    LABELS = list(config.LABELS)
HE = ANN / "human_eval"
PHENOMENA = ["negation", "quantity", "action", "entity attribute", "location/scene",
             "missing information", "world knowledge", "other"]
SINGLE_LINE = {"cited_edited_sentence", "uncited_edited_sentence"}


def rating_fields():
    fields = {f"{side}_{q}": scale for side in ("orig", "var") for q, scale in OUTPUT_QS.items()}
    fields.update(PAIR_QS)
    fields["failure_tags"] = ("multi", FAILURE_TAGS)
    fields["comments"] = None
    return fields


SHEETS = {
    "interventions": (ANN / "interventions.csv", "orig_id", {
        "cited_edited_sentence": None, "uncited_edited_sentence": None,
        **{f"{k}_gold_{r}": LABELS for k in ("cited", "uncited") for r in ("r1", "r2", "final")},
        "notes": None}),
    "phenomena": (ANN / "phenomenon_tagging.csv", "id", {"phenomenon": ("multi", PHENOMENA), "notes": None}),
    "calibration_r1": (HE / "calibration_r1.csv", "pair_uid", rating_fields(), HE / "calibration.csv"),
    "calibration_r2": (HE / "calibration_r2.csv", "pair_uid", rating_fields(), HE / "calibration.csv"),
    "sheet_r1": (HE / "sheet_r1.csv", "pair_uid", rating_fields()),
    "sheet_r2": (HE / "sheet_r2.csv", "pair_uid", rating_fields()),
    "adjudicated": (HE / "adjudicated.csv", "pair_uid", rating_fields()),
}

ACTIONS = {} if KIT else {
    "build_interventions": ["-m", "cfx.faithfulness", "build"],
    "agreement": ["-m", "cfx.human_eval", "agreement"],
    "results": ["-m", "cfx.human_eval", "results"],
    "analysis": ["-m", "cfx.analysis"],
}


def export_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("kit.json", json.dumps(KIT, indent=1))
        for p in sorted(ANN.rglob("*.csv")):
            if "backups" not in p.parts:
                z.write(p, p.relative_to(ANN).as_posix())
    return buf.getvalue()


def rel(path):
    return str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else str(path)


def clean(name, spec, value):
    value = "" if value is None else str(value).strip()
    if spec is None:
        return " ".join(value.split()) if name in SINGLE_LINE else value
    if isinstance(spec, tuple):
        parts = [p.strip().lower() for p in value.replace(",", ";").split(";") if p.strip()]
        bad = [p for p in parts if p not in spec[1]]
        if bad:
            raise ValueError(f"{name}: unknown value(s) {', '.join(bad)}")
        return ";".join(dict.fromkeys(parts))
    value = value.lower()
    if value and value not in spec:
        raise ValueError(f"{name}: {value!r} is not one of {', '.join(spec)}")
    return value


class Sheet:
    def __init__(self, name, path, key, fields, template=None):
        self.name, self.path, self.key, self.fields, self.template = name, path, key, fields, template
        self.columns, self.rows, self.mtime, self.backed_up = [], [], None, False

    def load(self):
        if not self.path.exists() and self.template is not None and self.template.exists():
            shutil.copy2(self.template, self.path)
        if not self.path.exists():
            self.columns, self.rows, self.mtime = [], [], None
            return False
        mtime = self.path.stat().st_mtime_ns
        if mtime != self.mtime:
            with open(self.path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                self.columns = list(reader.fieldnames or [])
                self.rows = [{k: (v or "") for k, v in r.items()} for r in reader]
            self.mtime = mtime
            if not self.backed_up:
                backups = ANN / "backups"
                backups.mkdir(exist_ok=True)
                shutil.copy2(self.path, backups / f"{self.path.stem}-{time.strftime('%Y%m%d-%H%M%S')}.csv")
                self.backed_up = True
        return True

    def write(self):
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)
        os.replace(tmp, self.path)
        self.mtime = self.path.stat().st_mtime_ns

    def update(self, key, values):
        if not self.load():
            raise ValueError(f"{self.path.name} does not exist yet")
        row = next((r for r in self.rows if r.get(self.key) == key), None)
        if row is None:
            raise ValueError(f"unknown {self.key} {key!r}")
        for name, value in values.items():
            if name not in self.fields or name not in self.columns:
                raise ValueError(f"field {name} is not editable here")
            row[name] = clean(name, self.fields[name], value)
        self.write()
        return row


class App:
    def __init__(self):
        self.lock = threading.Lock()
        self.run_lock = threading.Lock()
        self.sheets = {n: Sheet(n, *spec) for n, spec in SHEETS.items()}

    def state(self):
        with self.lock:
            out = {}
            for n, s in self.sheets.items():
                exists = s.load()
                out[n] = {"exists": exists, "rows": len(s.rows), "path": rel(s.path)}
            return out

    def sheet(self, name):
        with self.lock:
            s = self.sheets[name]
            exists = s.load()
            fields = {k: (list(v[1]) if isinstance(v, tuple) else v) for k, v in s.fields.items()}
            multi = [k for k, v in s.fields.items() if isinstance(v, tuple)]
            return {"exists": exists, "key": s.key, "columns": s.columns, "rows": s.rows,
                    "fields": fields, "multi": multi, "path": rel(s.path)}

    def update(self, name, key, values):
        with self.lock:
            return self.sheets[name].update(key, values)

    def run(self, action):
        if not self.run_lock.acquire(blocking=False):
            raise ValueError("another step is still running")
        try:
            env = {**os.environ, "PYTHONPATH": str(PROJECT / "src")}
            p = subprocess.run([sys.executable, *ACTIONS[action]], cwd=PROJECT, env=env,
                               capture_output=True, text=True, timeout=1800)
            return {"ok": p.returncode == 0, "output": (p.stdout + p.stderr)[-20000:]}
        finally:
            self.run_lock.release()


def make_handler(app):
    meta = {"labels": LABELS, "output_qs": OUTPUT_QS, "pair_qs": PAIR_QS, "failure_tags": FAILURE_TAGS,
            "phenomena": PHENOMENA, "actions": list(ACTIONS),
            "kit": {"role": KIT["role"], "send_to": KIT.get("send_to", "")} if KIT else None}

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
            elif self.path == "/api/meta":
                self.send(200, meta)
            elif self.path == "/api/state":
                self.send(200, app.state())
            elif self.path.startswith("/api/sheet/") and self.path[11:] in SHEETS:
                self.send(200, app.sheet(self.path[11:]))
            elif self.path == "/api/export" and KIT:
                with app.lock:
                    data = export_zip()
                name = f"{KIT['role']}-answers-{time.strftime('%Y%m%d-%H%M')}.zip"
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/update" and body.get("sheet") in SHEETS:
                    row = app.update(body["sheet"], body["key"], body.get("fields", {}))
                    return self.send(200, {"ok": True, "row": row})
                if self.path == "/api/run" and body.get("action") in ACTIONS:
                    return self.send(200, app.run(body["action"]))
                self.send(404, {"ok": False, "error": "not found"})
            except (ValueError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
                self.send(400, {"ok": False, "error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    app = App()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Annotation tool: {url}")
    print("Keep this window open while you work. Close it (or press Ctrl+C) when you're done.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
