# ABLE

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/able/actions/workflows/ci.yml/badge.svg)

Work in progress.

## Experiments

We use [hydra](https://hydra.cc/) for experiments, with configurations in [experiments/conf](experiments/conf/). To compare the scratch and dreaming approaches on the two-link arm over five seeds and plot offline evaluation per cycle:

```
./experiments/run_comparison.sh
```

The results and the plot (`cycle_results.png`) go to `logs/comparison/<date>_<time>/`. Extra arguments are passed to hydra, for example `./experiments/run_comparison.sh num_cycles=10`, and `SEEDS=0,1` changes the seeds.

To run one approach with one seed:

```
python experiments/run_single_experiment.py approach=dreaming seed=0
```

Each run writes its resolved `config.yaml`, `eval_results.csv` (one row per evaluation task per cycle), and `cycle_results.csv` (one row per cycle, including sleep time). To plot any set of runs:

```
python experiments/analysis/plot_cycle_results.py --results_dir <directory containing the runs>
```
