"""Utilities for loading and summarizing experiment results."""

from pathlib import Path
from typing import Sequence

import pandas as pd
from omegaconf import DictConfig, OmegaConf


def combine_results_csvs(
    directory: Path, results_filename: str, config_keys: Sequence[str] = ()
) -> pd.DataFrame:
    """Combine one results file from every run under the directory.

    A run is a directory that contains the results file and the resolved config.yaml
    written by run_single_experiment.py. Each row gets the run's environment, approach,
    and seed, and a column for each config key, such as env.domain.edge_probability. A
    key missing from a run's config gives None.
    """
    frames = []
    for config_path in sorted(directory.rglob("config.yaml")):
        results_path = config_path.parent / results_filename
        if not results_path.exists():
            continue
        cfg = OmegaConf.load(config_path)
        assert isinstance(cfg, DictConfig)
        df = pd.read_csv(results_path)
        df["env"] = cfg.env_name
        df["approach"] = cfg.approach_name
        df["seed"] = cfg.seed
        for key in config_keys:
            df[key] = OmegaConf.select(cfg, key)
        frames.append(df)
    assert frames, f"No {results_filename} files found under {directory}"
    return pd.concat(frames, ignore_index=True)


def summarize_over_seeds(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Summarize a metric for each environment, approach, and cycle.

    The metric is first averaged over the rows of each seed (for example, over
    evaluation tasks), and then the mean and standard error are taken over seeds. The
    standard error is zero when there is only one seed.
    """
    per_seed = df.groupby(["env", "approach", "seed", "cycle"], as_index=False)[
        metric
    ].mean()
    summary = per_seed.groupby(["env", "approach", "cycle"], as_index=False)[
        metric
    ].agg(mean="mean", sem="sem", num_seeds="count")
    summary["sem"] = summary["sem"].fillna(0.0)
    return summary
