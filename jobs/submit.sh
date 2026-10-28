#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/env.sh

KEY="${1:?model key (roihu_m1, roihu_m2)}"
PHASES="${2:?phases, e.g. smoke or dev,main,noise,revise}"
shift 2

PARTITION=gpumedium
TIME="$(model_field "$KEY" time_limit)"
ARGS=()
for a in "$@"; do
    if [[ "$a" == "--test" ]]; then PARTITION=gputest; TIME=00:15:00; else ARGS+=("$a"); fi
done

GPUS="$(model_field "$KEY" gpus)"
CPUS=$((72 * GPUS))
MEM="$((212 * GPUS))G"

if [[ "$CFX_ACCOUNT" == project_XXXXXXX ]]; then
    echo "set CFX_ACCOUNT in jobs/env.sh (or export it) first"; exit 1
fi
mkdir -p "$CFX_OUTPUTS/logs"
sbatch --account="$CFX_ACCOUNT" --partition="$PARTITION" --time="$TIME" \
    --gres="gpu:gh200:$GPUS" --cpus-per-task="$CPUS" --mem="$MEM" \
    --job-name="cfx-$KEY-${PHASES//,/-}" --output="$CFX_OUTPUTS/logs/%x-%j.out" \
    ${ARGS[@]+"${ARGS[@]}"} jobs/run.sbatch "$KEY" "$PHASES"
