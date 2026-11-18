# Common environment for every job. Source it from the repository root: `source cluster/env.sh`
# Override any of the variables below by exporting them before submitting.

# machine-specific settings written by cluster/link_data.sh (not tracked by git)
if [[ -f cluster/local.env ]]; then
    source cluster/local.env
fi

if type module >/dev/null 2>&1; then
    module purge
    module load gcc/11.3.0 python/3.10.4
fi

# virtualenv shared with umetyu/event_segmentation (benchmarks/flow_comparison)
export EVENT_FLOW_VENV="${EVENT_FLOW_VENV:-$HOME/venvs/event-flow-torch251-cu121}"
export BENCHMARK_PYTHON="${BENCHMARK_PYTHON:-$EVENT_FLOW_VENV/bin/python}"

# datasets of the paper in HDF5 (training/, MVSEC/, ECD/, HQF/): 19.4 GB download, 94 GB on disk
export EVFLOW_DATA="${EVFLOW_DATA:-$PWD/datasets/data}"
# AffineObjects (EV-LayerSegNet): train/h5/ and test/recordings_h5/
export AFFINE_DATA="${AFFINE_DATA:-$HOME/projects/event_segmentation/dataset}"

export EVFLOW_MLRUNS="${EVFLOW_MLRUNS:-$PWD/mlruns}"

# public code of Shiba et al. (GPL-3.0, not part of this repository): see cluster/setup_shiba.sh
export EVFLOW_SHIBA_REPO="${EVFLOW_SHIBA_REPO:-$PWD/third_party/event_based_optical_flow}"
export EVFLOW_SHIBA_SCRATCH="${EVFLOW_SHIBA_SCRATCH:-${TMPDIR:-/tmp}}"
export MLFLOW_ALLOW_FILE_STORE=true
export PYTHONUNBUFFERED=1
