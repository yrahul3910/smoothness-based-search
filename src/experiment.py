"""Utility functions for running experiments."""

import itertools
import random
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from raise_utils.data import Data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from src.data import load_data
from src.matplotlib import create_surface_data
from src.util import get_random_hyperparams

DEFAULT_HPO_SPACE = {
    "dt": {
        "criterion": ["friedman_mse", "absolute_error", "poisson", "squared_error"],
        "max_depth": list(range(1, 11)),
        "min_samples_split": [2, 3, 4, 5],
        "max_features": ["sqrt", "log2", 2, 3, 4, 5],
        "transform": ["normalize", "standardize", "minmax", "maxabs"],
    },
    "rf": {
        "n_estimators": [10, 25, 50, 75, 100],
        "criterion": ["friedman_mse", "absolute_error", "poisson", "squared_error"],
        "max_depth": list(range(1, 11)),
        "min_samples_split": [2, 3, 4, 5],
        "max_features": ["sqrt", "log2", 2, 3, 4, 5],
        "transform": ["normalize", "standardize", "minmax", "maxabs"],
    },
    "svr": {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 1],
        "degree": [2, 3, 4, 5],
        "transform": ["normalize", "standardize", "minmax", "maxabs"],
    },
}

DEFAULT_LEARNERS = {
    "dt": DecisionTreeRegressor,
    "rf": RandomForestRegressor,
}

DEFAULT_DIRS = ["./data/optimize/process/", "./data/hpo/", "./data/config/"]
DEFAULT_METRICS = [r2_score, mean_squared_error]


@dataclass
class SingleResult:
    learner: str
    config: dict[str, Any]
    score: float


type HPOSpace = dict[str, dict[str, list[Any]]]
type RawResults = dict[str, list[list[float]]]
type MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class ExperimentResult:
    """Metrics from an experiment."""

    raw_results: RawResults
    """Raw results mapping filenames to lists of scores per config."""

    mean_results: pl.DataFrame
    """Mean results per filename in a printable DataFrame."""

    runtime: float
    """Time taken to run the experiment."""

    best_results: list[SingleResult] | None = None
    """Best results per filename."""


def random_experiment(
    orig_data: Data,
    hpo_space: HPOSpace,
    learners: dict[str, type],
    metrics: list[MetricFn],
) -> list[float]:
    """Run a random search experiment over the given hyperparameter space and learners.

    This function performs a random search over the specified hyperparameter space for each learner,
    evaluating their performance using the provided metrics.

    Args:
        orig_data (Data): The original data to use for training and testing.
        hpo_space (HPOSpace): The hyperparameter space for each learner.
        learners (dict[str, type]): The learners to optimize.
        metrics (list[MetricFn]): The metrics to evaluate the learners.

    Returns:
        list[float]: The best metric scores achieved during the experiment.

    """
    best_configs = []
    best_learners = []
    num_configs = 50

    learner_choices = random.choices(list(hpo_space.keys()), k=num_configs)
    configs = [get_random_hyperparams(hpo_space[learner]) for learner in learner_choices]

    for learner_name, config in zip(learner_choices, configs, strict=False):
        data = deepcopy(orig_data)
        data.x_train = np.array(data.x_train)
        data.y_train = np.array(data.y_train)

        _transform = config.pop("transform")

        model = learners[learner_name](**config)
        model.fit(data.x_train, data.y_train)

        best_configs.append(config)
        best_learners.append(learner_name)

    best_metrics = [0.0] * len(metrics)
    for config, learner_name in zip(
        best_configs,
        best_learners,
        strict=False,
    ):
        data = deepcopy(orig_data)

        model = learners[learner_name](**config)
        model.fit(data.x_train, data.y_train)
        preds = model.predict(data.x_test)

        results = [fn(data.y_test, preds) for fn in metrics]
        best_metrics = results.copy() if results[0] > best_metrics[0] else best_metrics

    return best_metrics


def run_experiment(
    experiment_fn: Callable[[Data], list[float]] | None,
    hpo_space: HPOSpace = DEFAULT_HPO_SPACE,
    learners: dict[str, type] = DEFAULT_LEARNERS,
    dirs: list[str] = DEFAULT_DIRS,
    metrics: list[MetricFn] = DEFAULT_METRICS,
    n_repeats: int = 1,
    n_datasets: int | None = None,
) -> ExperimentResult:
    """Run an experiment to optimize hyperparameters for a set of learners on a set of datasets.

    This function makes a few assumptions about the inputs being passed:
        - Most importantly, the *first* metric in `metrics` is used as a discriminator for the optimization process.
        - It is also assumed that the data is structured as described [here](https://github.com/timm/moot).

    # Arguments:
        - experiment_fn: The function that runs the experiment. This is passed a *copy* of the data (so you can modify
          it as you wish). If `None`, a random search is performed.
        - hpo_space (dict[str, dict[str, list]]): The hyperparameter space for each learner.
        - learners (dict[str, type]): The learners to optimize.
        - dirs (list[str]): The directories to search for datasets.
        - metrics (list[Callable[[np.ndarray, np.ndarray], float]]): The metrics to evaluate the learners.
        - n_repeats (int): The number of times to repeat the experiment.
        - n_datasets (int | None): The number of datasets to use for the experiment. If `None`, all datasets are used.
          This is useful for testing to avoid long-running experiments.

    Returns:
        An `ExperimentResult` object containing the raw and processed results and the total runtime.

    """
    # Validation
    if hpo_space.keys() != learners.keys():
        msg = "hpo_space must contain all keys from learners"
        raise ValueError(msg)

    if experiment_fn is None:
        experiment_fn = partial(
            random_experiment,
            hpo_space=hpo_space,
            learners=learners,
            metrics=metrics,
        )

    files = list(itertools.chain.from_iterable(Path(d).rglob("*.csv") for d in dirs))
    full_results = {"dataset": []} | {k: [] for k in metrics}

    if n_datasets is None:
        n_datasets = len(files)

    start_time = time.time()
    for filename in files[:n_datasets]:
        data_orig = load_data(filename)
        data_orig = data_orig.select(
            (pl.all() - pl.all().min()) / (pl.all().max() - pl.all().min()),
        )

        x, y = create_surface_data(data_orig, pca=False)
        preprocessed_data = Data(
            *train_test_split(x, y, test_size=0.2, random_state=42),
        )
        full_results["dataset"].append(filename)

        for i in tqdm(range(n_repeats)):
            best_metrics = experiment_fn(deepcopy(preprocessed_data))

            for idx, metric in enumerate(metrics):
                if i == 0:
                    full_results[metric].append([best_metrics[idx] * 100.0])
                else:
                    full_results[metric][-1] += [best_metrics[idx] * 100.0]

    end_time = time.time()
    runtime = end_time - start_time

    raw_results = {k if isinstance(k, str) else k.__name__: v for k, v in full_results.items()}

    for k, v in full_results.items():
        if k != "dataset":
            full_results[k] = np.mean(v, axis=1)

    metric_names = [fn.__name__ for fn in metrics]
    df = pl.DataFrame(list(full_results.values()), schema=["dataset", *metric_names])

    processed_result = df.with_columns(
        pl.col("dataset"),
        *[pl.col(col).round(2) for col in metric_names],
    )

    return ExperimentResult(
        raw_results=raw_results,
        mean_results=processed_result,
        runtime=runtime,
    )
