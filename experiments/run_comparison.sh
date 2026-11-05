#!/bin/bash
# Compare approaches in one environment and plot offline evaluation per cycle.
#
# Usage: ./experiments/run_comparison.sh env=<env> [hydra overrides...]
# The environment is two_link_arm, limb, friction, or human_reach (default
# two_link_arm). For example: ./experiments/run_comparison.sh env=limb num_cycles=5
# Set SEEDS or APPROACHES to change the seeds or approaches, for example:
#   SEEDS=0,1 APPROACHES=scratch,dreaming ./experiments/run_comparison.sh env=friction
# With record_videos=true, each run's stage videos are also combined into one video.
set -e
cd "$(dirname "$0")/.."
SEEDS="${SEEDS:-0,1,2,3,4}"
APPROACHES="${APPROACHES:-scratch,prior_only,current_belief,dreaming,belief_coverage}"
ENV=two_link_arm
VIDEOS=false
for arg in "$@"; do
    case "$arg" in
        env=*) ENV="${arg#env=}" ;;
        record_videos=true) VIDEOS=true ;;
    esac
done
SWEEP_DIR="logs/comparison/$ENV/$(date +%Y-%m-%d_%H-%M-%S)"
# Hydra varies the last override fastest, so the approaches run back to back for
# each seed and see similar machine load.
python experiments/run_single_experiment.py -m \
    seed="$SEEDS" approach="$APPROACHES" hydra.sweep.dir="$SWEEP_DIR" "$@"
python experiments/analysis/plot_cycle_results.py --results_dir "$SWEEP_DIR"
if [ "$VIDEOS" = true ]; then
    for run_dir in "$SWEEP_DIR"/*/; do
        if [ -d "$run_dir/videos" ]; then
            python experiments/combine_videos.py --run_dir "$run_dir"
        fi
    done
fi
