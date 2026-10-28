#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE="${CFX_REMOTE:?set CFX_REMOTE=user@roihu-gpu.csc.fi:/scratch/<project>/<user>/cfx/outputs}"
rsync -av "$REMOTE/runs/" outputs/runs/
rsync -av "$REMOTE/logs/" outputs/logs/
if [[ "${1:-}" == "--cache" ]]; then
    rsync -av "$REMOTE/cache/" outputs/cache/
fi
