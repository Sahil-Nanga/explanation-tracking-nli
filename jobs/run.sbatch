#!/bin/bash
#SBATCH --job-name=cfx
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

KEY="$1"
PHASES="$2"
cd "${SLURM_SUBMIT_DIR:?run this through sbatch}"
source jobs/env.sh

module purge
module load "$CFX_VLLM_MODULE"

MODEL="$(model_field "$KEY" model)"
REVISION="$(model_field "$KEY" revision)"
TP="$(model_field "$KEY" gpus)"
MAXLEN="$(model_field "$KEY" max_model_len)"
EXTRA="$(model_field "$KEY" engine_args)"

TMP="${TMPDIR:-/tmp}"
export HF_HUB_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR="$TMP/torch-cache"
export CFX_UDS="$TMP/vllm-$SLURM_JOB_ID.sock"
export CFX_ENGINE_ARGS="--revision $REVISION --tokenizer-revision $REVISION --tensor-parallel-size $TP --max-model-len $MAXLEN --seed 0 --dtype bfloat16 --generation-config vllm $EXTRA"
LOG_DIR="$CFX_OUTPUTS/logs"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/vllm-$KEY-$SLURM_JOB_ID.log"

echo "job $SLURM_JOB_ID on $(hostname): model=$MODEL rev=$REVISION tp=$TP phases=$PHASES"
in_container nvidia-smi -L || { echo "no GPU visible inside the container (try CFX_APPTAINER_FLAGS=--nv)"; exit 1; }

in_container vllm serve "$MODEL" $CFX_ENGINE_ARGS --served-model-name "$MODEL" --uds "$CFX_UDS" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
stop_server() {
    pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    rm -f "$CFX_UDS"
}
trap stop_server EXIT

echo "waiting for vLLM (log: $SERVER_LOG)"
for _ in $(seq 1 240); do
    if curl -sf --unix-socket "$CFX_UDS" http://localhost/v1/models > /dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM exited during startup:"; tail -50 "$SERVER_LOG"; exit 1
    fi
    sleep 10
done
curl -sf --unix-socket "$CFX_UDS" http://localhost/v1/models > /dev/null || { echo "vLLM not ready after 40 min"; exit 1; }
echo "vLLM ready after ${SECONDS}s"

in_container env PYTHONPATH=src CFX_UDS="$CFX_UDS" CFX_OUTPUTS="$CFX_OUTPUTS" CFX_ENGINE_ARGS="$CFX_ENGINE_ARGS" \
    CFX_WAIT_SECONDS=120 python3 -m cfx.phase --model "$KEY" --phases "$PHASES"
echo "done in ${SECONDS}s"
