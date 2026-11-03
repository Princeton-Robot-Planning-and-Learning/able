#!/bin/bash
# Compare approaches in the friction environment and plot offline evaluation per cycle.
#
# Usage: ./experiments/run_friction_comparison.sh [hydra overrides...]
# For example: ./experiments/run_friction_comparison.sh num_cycles=5
# Set SEEDS to change the seeds, for example: SEEDS=0,1 ./experiments/run_friction_comparison.sh
set -e
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0,1,2,3,4}"
SWEEP_DIR="logs/friction_comparison/$(date +%Y-%m-%d_%H-%M-%S)"
python experiments/run_single_experiment.py -m env=friction \
    approach=friction_scratch,friction_prior_only,friction_current_belief,friction_dreaming \
    seed="$SEEDS" hydra.sweep.dir="$SWEEP_DIR" "$@"
python experiments/analysis/plot_cycle_results.py --results_dir "$SWEEP_DIR"
