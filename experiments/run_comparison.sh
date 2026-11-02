#!/bin/bash
# Compare approaches on the two-link arm and plot offline evaluation per cycle.
#
# Usage: ./experiments/run_comparison.sh [hydra overrides...]
# For example: ./experiments/run_comparison.sh num_cycles=10
# Set SEEDS to change the seeds, for example: SEEDS=0,1 ./experiments/run_comparison.sh
set -e
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0,1,2,3,4}"
SWEEP_DIR="logs/comparison/$(date +%Y-%m-%d_%H-%M-%S)"
python experiments/run_single_experiment.py -m approach=scratch,current_belief,dreaming \
    seed="$SEEDS" hydra.sweep.dir="$SWEEP_DIR" "$@"
python experiments/analysis/plot_cycle_results.py --results_dir "$SWEEP_DIR"
