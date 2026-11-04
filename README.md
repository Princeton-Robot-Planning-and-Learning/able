# ABLE

![workflow](https://github.com/Princeton-Robot-Planning-and-Learning/able/actions/workflows/ci.yml/badge.svg)

Work in progress.

## Experiments

We use [hydra](https://hydra.cc/) for experiments, with configurations in [experiments/conf](experiments/conf/). To compare six approaches on the two-link arm over five seeds and plot offline evaluation per cycle:

```
./experiments/run_comparison.sh
```

The approaches all plan with random shooting and differ in what they learn while they sleep:
- `scratch` learns nothing and plans from scratch every time.
- `prior_only` caches plans for every task under the prior in its first sleep and never learns again.
- `current_belief` caches plans for every task under its current belief, so it never anticipates belief updates.
- `dreaming` imagines the belief updates it will make before it is evaluated and caches plans for the imagined beliefs.
- `belief_coverage` dreams like `dreaming`, but each training belief is the farthest of four dreams from the beliefs it has already cached plans for. The distance compares the beliefs' parts of the cache key, so it spreads training over beliefs that need different plans. This adapts a belief-expansion heuristic from point-based POMDP planning ([Pineau, Gordon, and Thrun, 2006](https://jair.org/index.php/jair/article/view/10473)).
- `next_call` dreams 256 beliefs at the start of each sleep and groups them by their part of the cache key. Each training belief comes from the group with the largest product of its fraction of the dreams and the planning time that caching more plans for it is expected to save, estimated from the planning times of earlier training. This adapts continual computation ([Horvitz, 2001](https://www.erichorvitz.com/ccprinciples.htm)).

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

`env.domain.variant` chooses the scene and limb, for example `env.domain.variant=bed-right-leg`. `env.domain.min_separation` and `env.domain.max_separation` bound the joint-space distance between a task's start and goal, 0.5 to 0.8 rad in the experiments. Planning a task takes a few minutes with the default 32 MPC rollouts per step.

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

`env.domain.num_holes` (0 to 9) and `env.domain.goal_radius` choose how hard the maze is, 4 and 0.3 m in the experiments. `env.domain.sim=3d` runs the pushes in PyBullet with a TidyBot holding a paddle instead of in 2D, with the same physics. Planning a push takes several seconds with the default 1000 simulations, and a task takes up to 15 pushes. As on the arm and the limb, the robot asks no queries by default; a query probes the floor the belief is least sure of.

The robots that learn guide POMCPOW with a flow-matching model of good pushes, used for half of its proposals. While they sleep they roll out sampled tasks under a training belief and learn the well-visited pushes of the rollouts that reach the goal. As on the arm and the limb, they differ only in the beliefs they train under: `friction_current_belief`, `friction_prior_only`, and `friction_dreaming`. To compare them with `friction_scratch` over five seeds:

```
./experiments/run_friction_comparison.sh
```

The friction environment evaluates 2 tasks per cycle and dreams 4 times per sleep, rolling out 1 training task per dream; these are `num_eval_tasks`, `budgets.sleep.max_iterations`, and `approach.num_training_tasks`.

## Human reach

The human reach environment needs the `human_reach` extra (`pip install -e ".[human_reach]"`), which builds PDDLStream's Fast Downward (it needs cmake and a C++ compiler).

A robot moves requested items from far furniture to the bedside of a person sitting up in bed, so that the person can reach them. The person's reach is an ellipsoid at their right shoulder whose semi-axes are unknown. The robot plans the whole task with PDDLStream, placing each requested item where at least 90% of its belief's particles reach it, and executes the plan one action (move, pick, or place) per step. To run the robot that plans from scratch:

```
python experiments/run_single_experiment.py env=human_reach approach=human_reach_scratch
```

`env.domain.num_requested` and `env.domain.num_clutter` set the scene, 2 and 5 in the experiments. Under the prior, only 3 bedside spots are reachable with 90% probability, so scenes with more requested items often have no plan until the belief narrows. PDDLStream plans for up to 5 minutes (`approach.max_time`), and planning a scene takes from seconds to minutes. As in the other environments, the robot asks no queries by default; a query presents the gripper at points along a direction from the person's shoulder.

The robots that learn rank PDDLStream's placements for each requested item with a flow-matching model of where plans put it, trained while they sleep on plans for sampled scenes. As in the other environments, they differ only in the beliefs they train under: `human_reach_current_belief`, `human_reach_prior_only`, and `human_reach_dreaming`. To compare them with `human_reach_scratch` over five seeds:

```
./experiments/run_human_reach_comparison.sh
```

The human reach environment evaluates 2 tasks per cycle and dreams 4 times per sleep, planning 1 training scene per dream; these are `num_eval_tasks`, `budgets.sleep.max_iterations`, and `approach.num_training_tasks`.


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

Add `hydra/launcher=della` to any multirun. The comparison scripts pass it through to Hydra:

```
cd /scratch/gpfs/TSILVER/$USER/able
tmux new -s able
source .venv/bin/activate
./experiments/run_friction_comparison.sh hydra/launcher=della
```

The script submits one job per seed and approach, waits for them to finish, and then plots. Waiting takes no compute, but the script has to stay running, which is why it runs in `tmux`. Detach with `Ctrl-b d`, and reattach later with `tmux attach -t able`. A tmux session lives on one login node, so note which one `hostname` prints. If `tmux attach` later finds no session, you are on a different login node, and `ssh della9` (or whichever node you noted) reaches the right one after another password and Duo prompt.

`squeue -u $USER` lists your jobs, and `jobstats <job id>` reports a finished job's CPU and memory use. Each job's Slurm logs are in the sweep directory's `.submitit/` folder.

Jobs request 24 hours, 1 CPU, and 8 GB by default, which puts them in Della's short QOS: up to 500 jobs and 1000 CPUs at once per user. For longer runs, raise `hydra.launcher.timeout_min`; up to 4320 minutes goes to the medium QOS (400 CPUs per user) and up to 8640 minutes to vlong (325 CPUs per user). `hydra.launcher.mem_gb` sets the memory.

To copy results to your laptop:

```
rsync -av della:/scratch/gpfs/TSILVER/<netid>/able/logs/ logs/
```
