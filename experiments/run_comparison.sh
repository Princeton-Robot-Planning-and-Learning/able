#!/bin/bash
# Compare approaches in one environment and plot offline evaluation per cycle.
#
# Usage: ./experiments/run_comparison.sh env=<env> [hydra overrides...]
# The environment is two_link_arm, limb, limb_calendar, friction, human_reach, or
# friction_sites
# (default two_link_arm). For example:
#   ./experiments/run_comparison.sh env=limb num_cycles=5
# friction_sites and limb_calendar run their own approaches by default, and
# limb_calendar has eight people, seeds 0 to 7.
# Set SEEDS or APPROACHES to change the seeds or approaches, for example:
#   SEEDS=0,1 APPROACHES=scratch,dreaming ./experiments/run_comparison.sh env=friction
# Set WORKERS to give each run that many worker processes (num_workers), which
# evaluate tasks and make a sleep's dreams and training plans at once. With
# hydra/launcher=della, each job then requests that many CPUs, for example:
#   WORKERS=6 ./experiments/run_comparison.sh env=limb hydra/launcher=della
# With record_videos=true, each run's stage videos are also combined into one video.
set -e
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-1}"
ENV=two_link_arm
VIDEOS=false
for arg in "$@"; do
    case "$arg" in
        env=*) ENV="${arg#env=}" ;;
        record_videos=true) VIDEOS=true ;;
    esac
done
case "$ENV" in
    friction_sites | limb_calendar) DEFAULT_APPROACHES=scratch,current_belief,dreaming,belief_coverage,transfer_practice,deliberate_practice,one_step_greedy ;;
    *) DEFAULT_APPROACHES=scratch,prior_only,current_belief,dreaming,belief_coverage ;;
esac
case "$ENV" in
    limb_calendar) DEFAULT_SEEDS=0,1,2,3,4,5,6,7 ;;
    *) DEFAULT_SEEDS=0,1,2,3,4,5,6,7,8,9 ;;
esac
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
APPROACHES="${APPROACHES:-$DEFAULT_APPROACHES}"
SWEEP_DIR="logs/comparison/$ENV/$(date +%Y-%m-%d_%H-%M-%S)"
# Hydra varies the last override fastest, so the approaches run back to back for
# each seed and see similar machine load.
python experiments/run_single_experiment.py -m \
    seed="$SEEDS" approach="$APPROACHES" num_workers="$WORKERS" \
    hydra.sweep.dir="$SWEEP_DIR" "$@"
python experiments/analysis/plot_cycle_results.py --results_dir "$SWEEP_DIR"
if [ "$VIDEOS" = true ]; then
    for run_dir in "$SWEEP_DIR"/*/; do
        if [ -d "$run_dir/videos" ]; then
            python experiments/combine_videos.py --run_dir "$run_dir"
        fi
    done
fi
