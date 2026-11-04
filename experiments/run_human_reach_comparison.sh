#!/bin/bash
# Compare approaches in the human reach environment and plot offline evaluation per cycle.
#
# Usage: ./experiments/run_human_reach_comparison.sh [hydra overrides...]
# For example: ./experiments/run_human_reach_comparison.sh num_cycles=5
# Set SEEDS to change the seeds, for example: SEEDS=0,1 ./experiments/run_human_reach_comparison.sh
set -e
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0,1,2,3,4}"
SWEEP_DIR="logs/human_reach_comparison/$(date +%Y-%m-%d_%H-%M-%S)"
python experiments/run_single_experiment.py -m env=human_reach \
    approach=human_reach_scratch,human_reach_prior_only,human_reach_current_belief,human_reach_dreaming \
    seed="$SEEDS" hydra.sweep.dir="$SWEEP_DIR" "$@"
python experiments/analysis/plot_cycle_results.py --results_dir "$SWEEP_DIR"
