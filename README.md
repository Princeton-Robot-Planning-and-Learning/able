# ABLE

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/able/actions/workflows/ci.yml/badge.svg)

Work in progress.

## Experiments

We use [hydra](https://hydra.cc/) for experiments, with configurations in [experiments/conf](experiments/conf/). To compare five approaches on the two-link arm over ten seeds and plot offline evaluation per cycle:

```
./experiments/run_comparison.sh
```

The approaches all plan with random shooting and differ in what they learn while they sleep:
- `scratch` learns nothing and plans from scratch every time.
- `prior_only` caches plans for every task under the prior in its first sleep and never learns again.
- `current_belief` caches plans for every task under its current belief, so it never anticipates belief updates.
- `dreaming` imagines the belief updates it will make before it is evaluated and caches plans for the imagined beliefs.
- `belief_coverage` dreams like `dreaming`, but each training belief is the farthest of four dreams from the beliefs it has already cached plans for. The distance compares the beliefs' parts of the cache key, so it spreads training over beliefs that need different plans. This adapts a belief-expansion heuristic from point-based POMDP planning ([Pineau, Gordon, and Thrun, 2006](https://jair.org/index.php/jair/article/view/10473)).

In the limb, friction, and human reach environments there is no plan cache, so `belief_coverage` compares beliefs by each hidden parameter's mean and standard deviation, in units of the prior's standard deviation, and takes the farthest of 4 dreams. Like the other approaches, it refits the model once per sleep.

The same script compares the approaches in the other environments, for example `./experiments/run_comparison.sh env=limb`. Approach names are the same in every environment: `approach=<name>` uses the shared budgets in `experiments/conf/approach/<name>.yaml` and the environment's factory in `experiments/conf/env_approach/<env>/<name>.yaml`.

The results, the plot (`cycle_results.png`) and a Markdown table of planning time, success rate, timeouts, the planner's work per task (samples, MPC rollouts, simulations per push, or stream calls, which do not depend on the machine) and sleep time (`summary_table.md`) go to `logs/comparison/<env>/<date>_<time>/`. Extra arguments are passed to hydra, for example `./experiments/run_comparison.sh num_cycles=10`. `SEEDS=0,1` changes the seeds and `APPROACHES=scratch,dreaming` the approaches. `WORKERS=4` gives each run 4 worker processes (`num_workers`), which evaluate tasks at once and, for robots that refit once per sleep with a fixed number of iterations, make the sleep's dreams and training plans at once. Each evaluation task and each dream then gets its own random generator, so results match a run without workers in distribution but not draw for draw. A run that records videos uses no workers, since the videos follow every episode. With `record_videos=true` (and, for example, `video_cycles=[0,19]`), the script also combines each run's stage videos into one video per run.

To run one approach with one seed:

```
python experiments/run_single_experiment.py approach=dreaming seed=0
```

Each run writes its resolved `config.yaml`, `eval_results.csv` (one row per evaluation task per cycle, with why planning failed in `failure` and whether a time limit caused it in `timed_out`), and `cycle_results.csv` (one row per cycle, with the wall time of each stage and of the whole cycle). Both results files are rewritten after every cycle, so a run that stops early keeps its finished cycles, and a sweep can be plotted while it runs. After each cycle, the run's log (`run_single_experiment.log`) gets one line with the cycle's success rate, mean planning time, stage times, the run's elapsed time, and an estimate of when it will finish. To plot any set of runs:

```
python experiments/analysis/plot_cycle_results.py --results_dir <directory containing the runs>
```

The plot shows raw planning time and success rate per cycle. When the runs record timeouts, timed-out tasks are left out of planning time, counted as failures, and shown in a third panel of timeout rate.

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
python experiments/run_single_experiment.py env=limb approach=scratch
```

As on the arm, the robot asks no queries by default (`budgets.information_gathering.max_iterations=0`): one settled query nearly pins that motion's range, so every approach would learn the range at once.

`env.domain.variant` chooses the scene and limb, for example `env.domain.variant=bed-right-leg`. `env.domain.min_separation` and `env.domain.max_separation` bound the joint-space distance between a task's start and goal, 0.5 to 0.8 rad in the experiments. Planning a task takes a few minutes with the default 32 MPC rollouts per step.

The robots that learn seed MPC with a flow-matching model of the next controls, trained while they sleep on plans for sampled tasks. A step the model proposes is taken if it costs at most 5% more than the best directed candidate (`approach.seed_tolerance=0.05`). As on the arm, they differ only in the beliefs they train under: `current_belief`, `prior_only`, `dreaming`, and `belief_coverage`. To compare them with `scratch` over ten seeds:

```
./experiments/run_comparison.sh env=limb
```

The limb environment evaluates 8 tasks per cycle and dreams 4 times per sleep, planning 2 training tasks per dream; these are `num_eval_tasks`, `budgets.sleep.max_iterations`, and `approach.num_training_tasks`.

## Friction

The friction environment needs the `friction` extra (`pip install -e ".[friction]"`).

A robot pushes a puck through a maze to a goal, over three floors (blue, tan, and green) whose friction coefficients are unknown. Each action is one push, planned with POMCPOW from a particle belief over the frictions. The puck fails the task if it passes over a hole. To run the robot that plans from scratch:

```
python experiments/run_single_experiment.py env=friction approach=scratch
```

`env.domain.num_holes` (0 to 9) and `env.domain.goal_radius` choose how hard the maze is, 4 and 0.3 m in the experiments. `env.domain.sim=3d` runs the pushes in PyBullet with a TidyBot holding a paddle instead of in 2D, with the same physics. Planning a push takes several seconds with the default 1000 simulations, and a task takes up to 15 pushes. As on the arm and the limb, the robot asks no queries by default; a query probes the floor the belief is least sure of.

Planning stops after 15000 POMCPOW simulations per task (`budgets.planning.max_iterations`), shared by all its pushes, which caps the planner's work the same way on any machine and under any load. That is 15 pushes of 1000 simulations, all that planning from scratch can use, so a model-guided search that finds no push and falls back to a search without the model costs the task's later pushes simulations. With a smaller `approach.num_sims` the limit does not bind. A simulation takes about 0.014 s, so this binds before the 10-minute limit (`budgets.planning.max_wall_time`), which guards against a slow machine. Running out of either counts as a timeout.

The robots that learn guide POMCPOW with a flow-matching model of good pushes, used for half of its proposals. While they sleep they roll out sampled tasks under a training belief and learn the well-visited pushes of the rollouts that reach the goal. As on the arm and the limb, they differ only in the beliefs they train under: `current_belief`, `prior_only`, `dreaming`, and `belief_coverage`. To compare them with `scratch` over ten seeds:

```
./experiments/run_comparison.sh env=friction
```

The friction environment evaluates 8 tasks per cycle and dreams 4 times per sleep, rolling out 1 training task per dream; these are `num_eval_tasks`, `budgets.sleep.max_iterations`, and `approach.num_training_tasks`.

Every push runs a fixed number of POMCPOW simulations, so friction's planning time cannot fall with learning: a better model shows up as a higher success rate instead (#130). To show the speedup as the simulations per push a robot needs to reach a given success rate, run the comparison at several budgets:

```
./experiments/run_comparison.sh env=friction approach.num_sims=100,300,1000
```

The plot against cycles then shows the largest budget, and `budget_results.png` shows each approach's success rate over the last 5 cycles against the budget, with scratch at the largest budget as a dotted line. `summary_table.md` adds each approach's success rate at every budget and the simulations per push it needs to match scratch at the largest budget, interpolated in the budget's logarithm. On Della, the smaller budgets add jobs (three times as many), but they are short.

## Human reach

The human reach environment needs the `human_reach` extra (`pip install -e ".[human_reach]"`), which builds PDDLStream's Fast Downward (it needs cmake and a C++ compiler).

A robot moves requested items from far furniture to the bedside of a person sitting up in bed, so that the person can reach them. The person's reach is an ellipsoid at their right shoulder whose semi-axes are unknown. The robot plans the whole task with PDDLStream, placing each requested item where at least 90% of its belief's particles reach it, and executes the plan one action (move, pick, or place) per step. To run the robot that plans from scratch:

```
python experiments/run_single_experiment.py env=human_reach approach=scratch
```

`env.domain.num_requested` and `env.domain.num_clutter` set the scene, 2 and 5 in the experiments, with a horizon of 60 actions. A pose is usable if 98% of the belief's particles reach it (`approach.delta=0.02`). Items can go anywhere on a surface (`approach.placement=continuous`, #137); `approach.placement=slots` instead tries only the centers of a grid of slots 11 cm apart. Under the prior, only about 6% of the bedside within the largest possible reach is usable (with slots, 2 or 3 of the 21 bedside slots), so clutter often has to be moved out of the way first, and scenes with more requested items often have no plan until the belief narrows. A uniform sample is rarely usable, so planning from scratch under the prior often fails, and a model that proposes usable placements plans far faster (#137). Planning stops after 5000 stream evaluations per task (`budgets.planning.max_iterations`), which caps the planner's work the same way on any machine and under any load. A stream evaluation takes about 0.04 s, so this usually binds before the 10-minute limit (`approach.max_time` and `budgets.planning.max_wall_time`), which guards against a search that runs long without evaluating streams. A retry without the model shares both, and planning a scene takes from seconds to minutes. A task that runs out of either counts as a timeout. A solve that grows the process's memory by more than 4 GB (`approach.max_memory_gb`) fails instead of letting the run be killed: harder scenes, such as 3 requested items, can grow PDDLStream past 9 GB in a few minutes. As in the other environments, the robot asks no queries by default; a query presents the gripper at points along a direction from the person's shoulder.

The robots that learn propose PDDLStream's placements for each item, requested or clutter, with a flow-matching model of where plans put it, trained while they sleep on plans for sampled scenes. The model's draws are the first placements an item tries, each moved onto its nearest surface, before uniform samples; with slots, the draws only rank the slots. If the model-guided try finds no plan, the robot plans again without the model; the guided try gets at most half of the time limit and of the stream calls (`approach.guided_budget_fraction=0.5`), so that a poor ranking leaves the fallback the budget to plan as scratch would (#132). As in the other environments, they differ only in the beliefs they train under: `current_belief`, `prior_only`, `dreaming`, and `belief_coverage`. To compare them with `scratch` over ten seeds:

```
./experiments/run_comparison.sh env=human_reach
```

The human reach environment evaluates 8 tasks per cycle and dreams 4 times per sleep, planning 1 training scene per dream; these are `num_eval_tasks`, `budgets.sleep.max_iterations`, and `approach.num_training_tasks`.


## Della

[Della](https://researchcomputing.princeton.edu/systems/della) is Princeton's research cluster. With `hydra/launcher=della`, each job of a Hydra multirun runs as a Slurm job on one CPU of an AMD Genoa node. All jobs run on the same node type, so their planning times are comparable.

### First-time setup

Every Della login needs Duo. To approve Duo once per connection instead of once per `ssh` or `rsync`, add these lines to `~/.ssh/config` on your laptop, replacing `<netid>` with your NetID:

```
Host della
  User <netid>
  HostName della.princeton.edu
  ProxyJump tigressgateway
  ControlMaster auto
  ControlPersist yes
  ControlPath ~/.ssh/sockets/%p-%h-%r
  ServerAliveInterval 300

Host tigressgateway
  User <netid>
  HostName tigressgateway.princeton.edu
  ControlMaster auto
  ControlPersist yes
  ControlPath ~/.ssh/sockets/%p-%h-%r
  ServerAliveInterval 300
```

Then run `mkdir -p ~/.ssh/sockets && chmod 700 ~/.ssh/sockets` and `ssh della`. The first connection asks for your NetID password and Duo, once for `tigressgateway` and once for Della. Later connections reuse it until your network drops or you run `ssh -O stop della`. Because the connection goes through `tigressgateway`, it works off campus without the VPN.

On Della, work in the lab's scratch space, `/scratch/gpfs/TSILVER/<netid>`, which is not backed up. Compute nodes have no internet access, so install everything from the login node:

```
mkdir -p /scratch/gpfs/TSILVER/$USER/.cache/uv
echo "export UV_CACHE_DIR=/scratch/gpfs/TSILVER/$USER/.cache/uv" >> ~/.bashrc
source ~/.bashrc
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /scratch/gpfs/TSILVER/$USER
git clone https://github.com/Princeton-Robot-Planning-and-Learning/able.git
cd able
uv sync --all-extras
```

The uv cache goes on scratch because `/home` allows only 2 million files. The install takes about 2 minutes, including building Fast Downward.

The limb and friction environments need IKFast, which pybullet_helpers compiles the first time they plan. Compute nodes lack the static LAPACK library it links against, so compile it once on the login node, which takes about 3 minutes. The compile script expects Ubuntu's library paths and a static BLAS library, so point it at Della's libraries and at a symlink to Della's shared BLAS library:

```
mkdir -p /scratch/gpfs/TSILVER/$USER/ikfast-libs
ln -sf /usr/lib64/libblas.so.3 /scratch/gpfs/TSILVER/$USER/ikfast-libs/libblas.a
LAPACK_DIR=/usr/lib64 LIBGFORTRAN_DIR=/usr/lib64 BLAS_DIR=/scratch/gpfs/TSILVER/$USER/ikfast-libs \
  .venv/bin/python -c "from pybullet_helpers.ikfast.load import install_ikfast_if_required; from pybullet_helpers.robots.kinova import KinovaGen3RobotiqGripperPyBulletRobot as R; install_ikfast_if_required(R.ikfast_info())"
```

### Running experiments

Add `hydra/launcher=della` to any multirun. The comparison script passes it through to Hydra:

```
cd /scratch/gpfs/TSILVER/$USER/able
tmux new -s able
source .venv/bin/activate
./experiments/run_comparison.sh env=friction hydra/launcher=della
```

With `WORKERS=6` (or `num_workers=6`), each job requests 6 CPUs for its 6 worker processes. A cycle of belief coverage in the friction environment took 2.5 times less wall time with 4 workers than with none, and planning time per task barely changed.

The script submits one job per seed and approach, waits for them to finish, and then plots. Waiting takes no compute, but the script has to stay running, which is why it runs in `tmux`. Detach with `Ctrl-b d`, and reattach later with `tmux attach -t able`. A tmux session lives on one login node, so note which one `hostname` prints. If `tmux attach` later finds no session, you are on a different login node, and `ssh della9` (or whichever node you noted) reaches the right one after another password and Duo prompt.

`squeue -u $USER` lists your jobs, and `jobstats <job id>` reports a finished job's CPU and memory use. Each job's Slurm logs are in the sweep directory's `.submitit/` folder. To see how far every job of a sweep has gotten and when it should finish, print the last line of each job's log:

```
tail -n 1 logs/comparison/<env>/<sweep>/*/run_single_experiment.log
```

The first line of each log names the job's seed and approach.

Jobs request 24 hours, `num_workers` CPUs (1 by default), and 8 GB per CPU, which puts them in Della's short QOS: up to 500 jobs and 1000 CPUs at once per user.

Shorter time limits get jobs out of the queue sooner, but a limit of an hour or less puts them in Della's test QOS, which runs only 2 jobs per user at once. To choose limits, run a short sweep (for example `num_cycles=2` with one seed), then estimate each approach's full run from its cycle times:

```
python experiments/analysis/estimate_run_times.py --results_dir <sweep> --num_cycles 20
```

It prints the slowest run's estimate per approach and a limit 1.3 times as long, never below 61 minutes. One sweep has one limit, so approaches with very different times can run as separate sweeps (`APPROACHES=...`) with `hydra.launcher.timeout_min=<limit>`. For longer runs, raise `hydra.launcher.timeout_min`; up to 4320 minutes goes to the medium QOS (400 CPUs per user) and up to 8640 minutes to vlong (325 CPUs per user). `hydra.launcher.mem_per_cpu` sets the memory per CPU.

To copy results to your laptop:

```
rsync -av della:/scratch/gpfs/TSILVER/<netid>/able/logs/ logs/
```
