import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from cfx import config
from cfx.human_eval import FAILURE_TAGS, OUTPUT_QS, PAIR_QS, RATING_COLS

ANN = Path(os.environ.get("CFX_ANNOTATIONS", config.ANNOTATIONS))
KITS = HERE.parent / "kits"
NAMES = {"r1": "rater1", "r2": "rater2"}

OWNED = {
    "r1": {"interventions.csv": ["cited_edited_sentence", "uncited_edited_sentence", "cited_gold_r1", "uncited_gold_r1", "notes"],
           "phenomenon_tagging.csv": ["phenomenon", "notes"],
           "human_eval/calibration_r1.csv": RATING_COLS,
           "human_eval/sheet_r1.csv": RATING_COLS},
    "r2": {"interventions.csv": ["cited_gold_r2", "uncited_gold_r2"],
           "human_eval/calibration_r2.csv": RATING_COLS,
           "human_eval/sheet_r2.csv": RATING_COLS},
}
HIDDEN = {"r1": ["cited_gold_r2", "uncited_gold_r2", "cited_gold_final", "uncited_gold_final"],
          "r2": ["cited_gold_r1", "uncited_gold_r1", "cited_gold_final", "uncited_gold_final", "notes"]}


def source(rel):
    p = ANN / rel
    if not p.exists() and Path(rel).name.startswith("calibration_"):
        p = ANN / "human_eval" / "calibration.csv"
    return p


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


START_MAC = """#!/bin/bash
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
  python3 app.py
else
  echo "Python 3 is not installed. Get it from https://www.python.org/downloads/ and open this file again."
  read -r -p "Press Enter to close."
fi
"""
START_WIN = """@echo off
cd /d "%~dp0"
where py >nul 2>nul && (py -3 app.py) || (python app.py)
pause
"""
README = """ANNOTATION KIT ({name})

What you need: Python 3 (free, from https://www.python.org/downloads/). Macs usually have it already.

1. Unzip this folder anywhere (for example on your Desktop).
2. Start the website:
   - Mac: double-click "Start (Mac).command". The first time, macOS may say it can't be opened:
     right-click the file, choose Open, then Open again.
   - Windows: double-click "Start (Windows).bat".
   A small black window opens and the website opens in your browser (http://127.0.0.1:8771).
   Keep the black window open while you work.
3. Each task has a "How to do this" button with examples. Everything saves automatically,
   so you can stop at any time and continue later by starting it again.
4. To send your answers back: on the start page, click "Download my answers" and send that file to {send_to}.

If the browser doesn't open by itself, open http://127.0.0.1:8771 yourself.
If it says the address is already in use, the website is already running in another window.
The full rubric is in rubric.md.
"""


def build(role, send_to):
    name = f"annotation-kit-{NAMES[role]}"
    kit = {"role": role, "send_to": send_to, "built": time.strftime("%Y-%m-%d %H:%M"),
           "labels": list(config.LABELS), "output_qs": OUTPUT_QS, "pair_qs": PAIR_QS, "failure_tags": FAILURE_TAGS}
    KITS.mkdir(exist_ok=True)
    out = KITS / f"{name}.zip"

    def add(z, arc, data, mode=0o644):
        info = zipfile.ZipInfo(f"{name}/{arc}", time.localtime()[:6])
        info.external_attr = mode << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, data)

    with zipfile.ZipFile(out, "w") as z:
        add(z, "app.py", (HERE / "app.py").read_bytes())
        add(z, "index.html", (HERE / "index.html").read_bytes())
        add(z, "kit.json", json.dumps(kit, indent=1))
        add(z, "rubric.md", (HERE.parent / "annotation" / "rubric.md").read_bytes())
        add(z, "README.txt", README.format(name=NAMES[role], send_to=send_to or "the person who sent you this kit"))
        add(z, "Start (Mac).command", START_MAC, 0o755)
        add(z, "Start (Windows).bat", START_WIN.replace("\n", "\r\n"))
        for rel in OWNED[role]:
            src = source(rel)
            if not src.exists():
                sys.exit(f"missing {src}")
            cols, rows = read_csv(src)
            for r in rows:
                for c in HIDDEN[role]:
                    if c in r:
                        r[c] = ""
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
            add(z, f"data/{rel}", buf.getvalue())
    print(f"{out}  ({out.stat().st_size // 1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=list(OWNED))
    ap.add_argument("--send-to", default="", help="name shown to the rater as who gets their answers")
    args = ap.parse_args()
    build(args.role, args.send_to)


if __name__ == "__main__":
    main()
