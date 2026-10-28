#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${NEBIUS_API_KEY:?export NEBIUS_API_KEY first}"
PY="${PYTHON:-venv/bin/python}"
export PYTHONPATH=src
mkdir -p outputs/logs
LOG="outputs/logs/nebius-$(date +%Y%m%d-%H%M%S).log"

run() { "$PY" -m cfx.phase "$@" 2>&1 | tee -a "$LOG"; }

case "${1:-}" in
    smoke)         run --model m2 --phases smoke ;;
    all)           run --model m2 --phases dev,main,noise,revise
                   run --model m1 --phases dev,main,noise,revise
                   run --model m1 --phases probe ;;
    interventions) run --model m1 --phases interventions
                   run --model m2 --phases interventions ;;
    *)             echo "usage: $0 smoke|all|interventions"; exit 1 ;;
esac
echo "log: $LOG"
