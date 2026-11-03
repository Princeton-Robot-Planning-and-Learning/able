# ABLE

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/able/actions/workflows/ci.yml/badge.svg)

Work in progress.

## Experiments

We use [hydra](https://hydra.cc/) for experiments, with configurations in [experiments/conf](experiments/conf/). To compare four approaches on the two-link arm over five seeds and plot offline evaluation per cycle:

```
./experiments/run_comparison.sh
```

The approaches all plan with random shooting and differ in what they learn while they sleep:
- `scratch` learns nothing and plans from scratch every time.
- `prior_only` caches plans for every task under the prior in its first sleep and never learns again.
- `current_belief` caches plans for every task under its current belief, so it never anticipates belief updates.
- `dreaming` imagines the belief updates it will make before it is evaluated and caches plans for the imagined beliefs.

The results and the plot (`cycle_results.png`) go to `logs/comparison/<date>_<time>/`. Extra arguments are passed to hydra, for example `./experiments/run_comparison.sh num_cycles=10`, and `SEEDS=0,1` changes the seeds.

To run one approach with one seed:

```
python experiments/run_single_experiment.py approach=dreaming seed=0
```

Each run writes its resolved `config.yaml`, `eval_results.csv` (one row per evaluation task per cycle), and `cycle_results.csv` (one row per cycle, including sleep time and whether the belief changed between sleep and evaluation). To plot any set of runs:

```
python experiments/analysis/plot_cycle_results.py --results_dir <directory containing the runs>
```

### Comparing dreaming with the current-belief baseline

To sweep one setting and measure how much dreaming gains over `current_belief` for each value (10 seeds by default):

```
./experiments/run_gap_sweep.sh env.domain.edge_probability 0,0.25,0.5,0.75,1
```

The gains are the planning time saved per evaluation task and the increase in the rate of planning with cached plans, both overall and on revision cycles, where the belief changed between sleep and evaluation. The table (`gap_summary.md`) reports them for cycles 1–5, the second half of the run, and all cycles, along with success rates. The plot (`gap_summary.png`) shows them against the swept value. Results go to `logs/gap_sweep/<date>_<time>_<setting>/`. To summarize runs that share one setting, such as the output of `run_comparison.sh`:

```
python experiments/analysis/summarize_gap.py --results_dir <directory containing the runs>
```

### Videos

Recording is off by default. To record a video of each stage of each cycle (sleep, information gathering, task execution, and evaluation) in the run directory's `videos/` folder:

```
python experiments/run_single_experiment.py approach=dreaming num_cycles=3 record_videos=true
```

Rendering happens after each stage ends, so it does not affect planning or sleep times, but it adds a few seconds per cycle. Each cycle is about 40 seconds of video at 10 frames per second. To record only some cycles, set `video_cycles`, for example `video_cycles=[0,5,10]`. To combine a run's stage videos into one video (`videos/full_pipeline.mp4`) with a title card before each stage:

```
python experiments/combine_videos.py --run_dir <run directory>
```
