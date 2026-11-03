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

## Limb repositioning

The limb environment needs the `limb` extra (`pip install -e ".[limb]"`). IKFast compiles on first use and needs LAPACK and BLAS (on Ubuntu, `apt install liblapack-dev libblas-dev`).

The robot holds a person's limb and moves it to a goal configuration with MPC, without passing the person's unknown range of motion. To run the robot that plans from scratch:

```
python experiments/run_single_experiment.py env=limb approach=limb_scratch
```

As on the arm, the robot asks no queries by default (`budgets.information_gathering.max_iterations=0`): one settled query nearly pins that motion's range, so every approach would learn the range at once.

`env.domain.variant` chooses the scene and limb, for example `env.domain.variant=bed-right-leg`. Planning a task takes a few minutes with the default 32 MPC rollouts per step.

The robots that learn seed MPC with a flow-matching model of the next controls, trained while they sleep on plans for sampled tasks. As on the arm, they differ only in the beliefs they train under: `limb_current_belief`, `limb_prior_only`, and `limb_dreaming`. To compare them with `limb_scratch` over five seeds:

```
./experiments/run_limb_comparison.sh
```

The limb environment evaluates 2 tasks per cycle and dreams 4 times per sleep, planning 1 training task per dream; these are `num_eval_tasks`, `budgets.sleep.max_iterations`, and `approach.num_training_tasks`.

## Friction

The friction environment needs the `friction` extra (`pip install -e ".[friction]"`).

A robot pushes a puck through a maze to a goal, over three floors (blue, tan, and green) whose friction coefficients are unknown. Each action is one push, planned with POMCPOW from a particle belief over the frictions. The puck fails the task if it passes over a hole. To run the robot that plans from scratch:

```
python experiments/run_single_experiment.py env=friction approach=friction_scratch
```

`env.domain.level` chooses how hard the maze is, from `L1` (no holes, a large goal) to `L4` (four holes, a small goal). `env.domain.sim=3d` runs the pushes in PyBullet with a TidyBot holding a paddle instead of in 2D, with the same physics. Planning a push takes several seconds with the default 1000 simulations, and a task takes up to 15 pushes. As on the arm and the limb, the robot asks no queries by default; a query probes the floor the belief is least sure of.

The robots that learn guide POMCPOW with a flow-matching model of good pushes, used for half of its proposals. While they sleep they roll out sampled tasks under a training belief and learn the well-visited pushes of the rollouts that reach the goal. As on the arm and the limb, they differ only in the beliefs they train under: `friction_current_belief`, `friction_prior_only`, and `friction_dreaming`. To compare them with `friction_scratch` over five seeds:

```
./experiments/run_friction_comparison.sh
```

The friction environment evaluates 2 tasks per cycle and dreams 4 times per sleep, rolling out 1 training task per dream; these are `num_eval_tasks`, `budgets.sleep.max_iterations`, and `approach.num_training_tasks`.

