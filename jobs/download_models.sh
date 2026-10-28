#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/env.sh
module purge
module load "$CFX_VLLM_MODULE"
mkdir -p "$HF_HOME"

KEYS=("$@")
[[ ${#KEYS[@]} -eq 0 ]] && KEYS=(roihu_m2 roihu_m1)
in_container env PYTHONPATH=src HF_HOME="$HF_HOME" python3 -m cfx.pin_revisions --write --download "${KEYS[@]}"
echo "commit configs/models.json so the pinned revisions are recorded"
