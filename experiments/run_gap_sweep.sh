#!/bin/bash
# Sweep one setting and summarize how much dreaming gains over the current-belief
# baseline for each value.
#
# Usage: ./experiments/run_gap_sweep.sh <setting> <values> [hydra overrides...]
# For example: ./experiments/run_gap_sweep.sh env.domain.edge_probability 0,0.5,1
# Set SEEDS to change the seeds, for example: SEEDS=0,1 ./experiments/run_gap_sweep.sh ...
set -e
cd "$(dirname "$0")/.."
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <setting> <values> [hydra overrides...]" >&2
    exit 1
fi
LEVER="$1"
VALUES="$2"
shift 2
SEEDS="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
SWEEP_DIR="logs/gap_sweep/$(date +%Y-%m-%d_%H-%M-%S)_${LEVER}"
# Hydra varies the last override fastest, so the two approaches run back to back
# for each seed and value and see similar machine load.
python experiments/run_single_experiment.py -m \
    "$LEVER=$VALUES" seed="$SEEDS" approach=current_belief,dreaming \
    hydra.sweep.dir="$SWEEP_DIR" "$@"
python experiments/analysis/summarize_gap.py --results_dir "$SWEEP_DIR" --lever "$LEVER"
