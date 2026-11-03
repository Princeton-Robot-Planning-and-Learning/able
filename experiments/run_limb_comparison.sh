#!/bin/bash
# Compare approaches in the limb environment and plot offline evaluation per cycle.
#
# Usage: ./experiments/run_limb_comparison.sh [hydra overrides...]
# For example: ./experiments/run_limb_comparison.sh num_cycles=5
# Set SEEDS to change the seeds, for example: SEEDS=0,1 ./experiments/run_limb_comparison.sh
set -e
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0,1,2,3,4}"
SWEEP_DIR="logs/limb_comparison/$(date +%Y-%m-%d_%H-%M-%S)"
python experiments/run_single_experiment.py -m env=limb \
    approach=limb_scratch,limb_prior_only,limb_current_belief,limb_dreaming \
    seed="$SEEDS" hydra.sweep.dir="$SWEEP_DIR" "$@"
python experiments/analysis/plot_cycle_results.py --results_dir "$SWEEP_DIR"
