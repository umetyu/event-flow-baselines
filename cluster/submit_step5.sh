#!/bin/bash
# Step 5 (time-surface input), from the repository root:
#   bash cluster/submit_step5.sh                      # the two time-surface runs (ts5, ts3)
#   bash cluster/submit_step5.sh configs/exp/s5_*.yml # also the count / voxel5 baselines with validation
#   bash cluster/submit_step5.sh configs/exp/s5b_*.yml  # overfitting check: also every checkpoint on MVSEC dt=1
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

# second validation set (step 5b): MVSEC outdoor_day2, converted with tools/convert_mvsec_raw.py
if grep -q "validation_od2" "${configs[@]}"; then
    od2_dir="$EVFLOW_DATA/validation_od2"
    if ! ls "$od2_dir"/*.h5 >/dev/null 2>&1; then
        ls "$EVFLOW_DATA"/MVSEC_od2/*.h5 >/dev/null 2>&1 || {
            echo "Missing $EVFLOW_DATA/MVSEC_od2/outdoor_day2_data.h5: run tools/convert_mvsec_raw.py first" >&2; exit 2; }
        "$BENCHMARK_PYTHON" experiments/make_val_chunks.py --src "$EVFLOW_DATA/MVSEC_od2" --dst "$od2_dir" --chunks 8
    fi
    echo "validation chunks (outdoor_day2): $(ls "$od2_dir"/*.h5 | wc -l) files in $od2_dir"
fi

mkdir -p logs
for config in "${configs[@]}"; do
    [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
    name="$(basename "$config" .yml)"
    train=$(sbatch --parsable --job-name="$name" --export=ALL,CONFIG="$config" cluster/train.sbatch)
    evaluation=$(sbatch --parsable --dependency=afterok:"$train" --kill-on-invalid-dep=yes \
        --job-name="eval-$name" --export=ALL,SUITES="mvsec_dt1 mvsec_dt4",RUNS="$name" cluster/eval.sbatch)
    echo "$name: training $train, evaluation $evaluation (after the training)"
    if grep -q "^checkpoints:" "$config"; then
        ckpt=$(sbatch --parsable --dependency=afterany:"$evaluation" \
            --job-name="ckpt-$name" --export=ALL,SUITES="mvsec_dt1",RUNS="$name",CHECKPOINTS="epoch_000? epoch_00?9 val_*" cluster/eval.sbatch)
        echo "$name: checkpoint evaluation $ckpt (after the evaluation)"
        # weights chosen by the validation losses, also on dt=4
        best=$(sbatch --parsable --dependency=afterany:"$ckpt" \
            --job-name="valbest-$name" --export=ALL,SUITES="mvsec_dt4",RUNS="$name",CHECKPOINTS="val_*" \
            cluster/eval.sbatch)
        echo "$name: dt=4 evaluation of the validation-selected weights $best (after that)"
    fi
done
