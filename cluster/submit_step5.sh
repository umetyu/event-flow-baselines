#!/bin/bash
# Step 5 (time-surface input), from the repository root:
#   bash cluster/submit_step5.sh                      # the two time-surface runs (ts5, ts3)
#   bash cluster/submit_step5.sh configs/exp/s5_*.yml # also the count / voxel5 baselines with validation
# 1. cuts the MVSEC validation chunks once (experiments/make_val_chunks.py), if they are missing
# 2. submits one training job per config (100 epochs, training + validation loss every epoch)
# 3. chains an MVSEC evaluation (dt=1 and dt=4) after each training job
set -eo pipefail
source cluster/env.sh
set -u
configs=("$@")
if [[ ${#configs[@]} -eq 0 ]]; then
    configs=(configs/exp/s5_evflownet-ts5_w1000_seed0.yml configs/exp/s5_evflownet-ts3_w1000_seed0.yml)
fi

val_dir="$EVFLOW_DATA/validation_mvsec"
if ! ls "$val_dir"/*.h5 >/dev/null 2>&1; then
    "$BENCHMARK_PYTHON" experiments/make_val_chunks.py --src "$EVFLOW_DATA/MVSEC" --dst "$val_dir"
fi
echo "validation chunks: $(ls "$val_dir"/*.h5 | wc -l) files in $val_dir"

mkdir -p logs
for config in "${configs[@]}"; do
    [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
    name="$(basename "$config" .yml)"
    train=$(sbatch --parsable --job-name="$name" --export=ALL,CONFIG="$config" cluster/train.sbatch)
    evaluation=$(sbatch --parsable --dependency=afterok:"$train" --kill-on-invalid-dep=yes \
        --job-name="eval-$name" --export=ALL,SUITES="mvsec_dt1 mvsec_dt4",RUNS="$name" cluster/eval.sbatch)
    echo "$name: training $train, evaluation $evaluation (after the training)"
done
