#!/bin/bash
# First run on Izar, from a login shell:   bash cluster/bootstrap_izar.sh [SOURCE_DIR]
# 1. links/extracts the datasets (cluster/link_data.sh)
# 2. creates or reuses the virtualenv and installs the dependencies
# 3. writes results/data_report.txt (what is really inside the data and mlruns/)
# 4. submits the smoke test, and behind it (only if it passes and the data report is clean):
#    - evaluation of the distributed weights of the paper on MVSEC, if they are in mlruns/
#    - training of the non-recurrent voxel network (the [56]-style model) on UZH-FPV, followed by
#      its evaluation on MVSEC
# The six recurrent models are NOT retrained: their distributed weights are used.
# Safe to run again: datasets and environment are reused, finished evaluations are skipped. A
# training that was already submitted is submitted again, so check `squeue -u $USER` first.
set -eo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

if ! type module >/dev/null 2>&1; then
    echo 'Run this from an Izar login shell (bash -l) so that the module command is available.' >&2
    exit 2
fi

bash cluster/link_data.sh "$@"
NO_SMOKE=1 bash cluster/setup_izar.sh
source cluster/env.sh
set -u

echo
data_ok=1
"$BENCHMARK_PYTHON" experiments/inspect_data.py || data_ok=0
echo

mkdir -p logs
smoke=$(sbatch --parsable --export=ALL cluster/smoke.sbatch)
echo "Submitted the smoke test (job $smoke)."
if [[ $data_ok -eq 0 ]]; then
    echo "The data report lists problems, so no evaluation or training was submitted. Send the report."
    exit 1
fi
after="--dependency=afterok:$smoke --kill-on-invalid-dep=yes"

if grep -q 'weights yes' results/data_report.txt; then
    # shellcheck disable=SC2086
    pretrained=$(sbatch --parsable $after --job-name=eval-pretrained \
        --export=ALL,SUITES="mvsec_dt1 mvsec_dt4",MODELS=core,REFERENCE=1 cluster/eval.sbatch)
    echo "Submitted the evaluation of the distributed weights (job $pretrained)."
else
    echo "No weights in $EVFLOW_MLRUNS: download the pretrained models of the paper (docs/README_event_flow.md,"
    echo "'Download models'), extract them so that the run folders are under $EVFLOW_MLRUNS/<experiment>/, and run"
    echo '  sbatch --export=ALL,SUITES="mvsec_dt1 mvsec_dt4",MODELS=core,REFERENCE=1 cluster/eval.sbatch'
fi

config=configs/exp/s2_evflownet-voxel9_w7500_seed0.yml
name=$(basename "$config" .yml)
# shellcheck disable=SC2086
train=$(sbatch --parsable $after --job-name="$name" --time=1-00:00:00 \
    --export=ALL,CONFIG="$config" cluster/train.sbatch)
evaluation=$(sbatch --parsable --dependency=afterok:"$train" --kill-on-invalid-dep=yes --job-name=eval-voxel9 \
    --export=ALL,SUITES="mvsec_dt1 mvsec_dt4",RUNS="$name",REFERENCE=1 cluster/eval.sbatch)
echo "Submitted the training of $name (job $train) and its evaluation (job $evaluation)."
echo "Run  bash cluster/status.sh  at any time and send its output."
