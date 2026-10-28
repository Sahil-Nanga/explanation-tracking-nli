export CFX_ACCOUNT="${CFX_ACCOUNT:-project_2020210}"
export CFX_SCRATCH="${CFX_SCRATCH:-/scratch/$CFX_ACCOUNT/$USER/cfx}"
export CFX_OUTPUTS="${CFX_OUTPUTS:-$CFX_SCRATCH/outputs}"
export HF_HOME="${HF_HOME:-$CFX_SCRATCH/hf-cache}"
export CFX_VLLM_MODULE="${CFX_VLLM_MODULE:-python-vllm/0.29.0}"
export CFX_APPTAINER_FLAGS="${CFX_APPTAINER_FLAGS:-}"

in_container() {
    apptainer exec $CFX_APPTAINER_FLAGS --bind="$(csc-common-bind)" "$SIF" "$@"
}

model_field() {
    python3 -c 'import json,sys; c=json.load(open("configs/models.json"))[sys.argv[1]]; v=c.get(sys.argv[2], ""); print(" ".join(v) if isinstance(v, list) else v)' "$1" "$2"
}
