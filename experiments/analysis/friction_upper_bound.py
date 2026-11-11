"""Measure the most a learned push model could speed up friction planning (#130,
step 4).

For each seed (true frictions drawn from the prior), the planner solves many
training episodes with ample compute under a belief concentrated near the true
frictions, and a push model is trained on the rows of the solved ones, as the
learning robots train on their rollouts. Then the planner runs the seed's
evaluation episodes at small budgets, with and without the model. A model that
has learned the pushes for the true frictions is the best any learner could
produce, so its success at a budget bounds what learning can gain there.

Example:
    python experiments/analysis/friction_upper_bound.py --out logs/upper_bound \
        --eval_sims 50 100 300 --workers 24
"""

import argparse
import json
import os
import pickle
from multiprocessing import get_context
from pathlib import Path
from typing import Any

# One thread per process: the processes already fill the machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")  # numpy

# pylint: disable-next=wrong-import-position
import pandas as pd

# pylint: disable-next=wrong-import-position
import torch

# pylint: disable-next=wrong-import-position
from diagnose_friction import OUTCOMES, Job, Setting, run_job

# pylint: disable-next=wrong-import-position
from able.models.base import ConditionalModel, Dataset

# pylint: disable-next=wrong-import-position
from able.models.flow_matching import FlowMatchingModel

# pylint: disable-next=wrong-import-position
from able.planners.pomcpow import ProposalConfig

# Training episodes use episode indices from here, apart from evaluation ones.
TRAIN_EPISODE_OFFSET = 1000


def _init_worker() -> None:
    torch.set_num_threads(1)


def _fit(dataset: Dataset) -> ConditionalModel:
    """The push model the friction learners use, fit to the rows."""
    return FlowMatchingModel(width=128, depth=3, num_steps=2000).fit(dataset)


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _train(args: argparse.Namespace, pool: Any) -> dict[int, ConditionalModel]:
    """Solve the training episodes and fit one model per seed, or load them."""
    path = args.out / "models.pkl"
    if path.exists():
        with path.open("rb") as f:
            models: dict[int, ConditionalModel] = pickle.load(f)
        return models
    jobs = [
        Job(
            Setting("near", args.train_sims),
            seed,
            TRAIN_EPISODE_OFFSET + k,
            args.num_particles,
            args.near_spread,
            args.max_depth,
            collect_rows=True,
        )
        for seed in range(args.seeds)
        for k in range(args.train_episodes)
    ]
    rows: list[dict[str, Any]] = []
    datasets: dict[int, list[Dataset]] = {}
    for row in pool.imap_unordered(run_job, jobs):
        dataset = row.pop("dataset")
        rows.append(row)
        if dataset is not None:
            datasets.setdefault(row["seed"], []).append(dataset)
        print(f"train seed={row['seed']} ep={row['episode']}: {row['outcome']}")
    _write(args.out / "train.jsonl", rows)
    seeds = sorted(datasets)
    fitted = pool.map(_fit, [Dataset.concatenate(datasets[s]) for s in seeds])
    models = dict(zip(seeds, fitted))
    with path.open("wb") as f:
        pickle.dump(models, f)
    return models


def _evaluate(
    args: argparse.Namespace, pool: Any, models: dict[int, ConditionalModel]
) -> pd.DataFrame:
    """Run every seed's evaluation episodes at each budget, with and without its
    model."""
    jobs = [
        Job(
            Setting("near", sims),
            seed,
            episode,
            args.num_particles,
            args.near_spread,
            args.max_depth,
            push_model=model,
            proposal=None if model is None else args.proposal,
        )
        for sims in args.eval_sims
        for seed in sorted(models)
        for episode in range(args.episodes)
        for model in (None, models[seed])
    ]
    rows = []
    for job, row in zip(jobs, pool.imap(run_job, jobs)):
        del row["dataset"]
        row["learned"] = job.push_model is not None
        rows.append(row)
        print(
            f"eval sims={row['num_sims']} learned={row['learned']} "
            f"seed={row['seed']} ep={row['episode']}: {row['outcome']}"
        )
    _write(args.out / args.eval_name, rows)
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Per budget, with and without the model: the fraction of each outcome."""
    groups = rows.groupby(["num_sims", "learned"])
    table = groups["outcome"].value_counts(normalize=True).unstack(fill_value=0)
    table = table.reindex(columns=list(OUTCOMES), fill_value=0).round(2)
    table.insert(0, "episodes", groups.size())
    table["pushes"] = groups["pushes"].mean().round(1)
    return table.reset_index()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--train_episodes", type=int, default=30)
    parser.add_argument("--train_sims", type=int, default=1000)
    parser.add_argument("--eval_sims", type=int, nargs="+", default=[100])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--num_particles", type=int, default=500)
    parser.add_argument("--near_spread", type=float, default=0.03)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mix", type=float, default=0.5)
    parser.add_argument("--model_first", type=int, default=0)
    parser.add_argument("--eval_name", default="eval.jsonl")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    args.proposal = ProposalConfig(mix=args.mix, model_first=args.model_first)
    if args.summarize:
        rows = pd.read_json(args.out / args.eval_name, lines=True)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        # Spawned, not forked, with one torch thread per worker: torch training
        # hangs in pool workers otherwise.
        context = get_context("spawn")
        with context.Pool(args.workers, initializer=_init_worker) as pool:
            models = _train(args, pool)
            rows = _evaluate(args, pool, models)
    print(summarize(rows).to_string(index=False))


if __name__ == "__main__":
    _main()
