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

Each run writes its resolved `config.yaml`, `eval_results.csv` (one row per evaluation task per cycle), and `cycle_results.csv` (one row per cycle, including sleep time). To plot any set of runs:

```
python experiments/analysis/plot_cycle_results.py --results_dir <directory containing the runs>
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
