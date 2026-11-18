# Common environment for every job. Source it from the repository root: `source cluster/env.sh`
# Override any of the variables below by exporting them before submitting.

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
export MLFLOW_ALLOW_FILE_STORE=true
export PYTHONUNBUFFERED=1
