#!/bin/bash
# Prints a short status to copy back: queue, data report, smoke test result, epoch times.
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
echo "== commit $(git rev-parse --short HEAD) | $(date '+%F %T')"
echo "== queue"; squeue -u "$USER" -o '%.10i %.9P %.28j %.8T %.10M %.10l %R' 2>/dev/null
echo "== data report"; cat results/data_report.txt 2>/dev/null || echo "(none)"
for log in $(ls -t logs/smoke-*.out 2>/dev/null | head -1); do
    echo "== $log"; grep -E '^(OK|FAIL|torch|All smoke|FAILED)' "$log" | tail -25
    grep -q 'All smoke tests passed' "$log" || tail -15 "$log"
done
for log in $(ls -t logs/*.out 2>/dev/null | grep -v smoke- | head -8); do
    echo "== $log"
    grep -E '^(config:|MLflow run ID)' "$log" | head -2
    grep -E '^(validation data|time surface|  input window|  tau )' "$log"
    epochs="$(grep -E '^Epoch' "$log" || true)"
    if [[ -n "$epochs" ]]; then
        { echo "$epochs" | head -3; echo '...'; echo "$epochs" | tail -2; } | awk '!seen[$0]++'
    fi
    grep -E '^(estimator:|eval |done |Optimized|FAILED|Traceback)' "$log" | tail -4
    grep -q -E '^(Epoch|eval |done )' "$log" || tail -12 "$log"
done
echo "== results"; cat results/summary.md 2>/dev/null | tail -60 || echo "(none yet)"
