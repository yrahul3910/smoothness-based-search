"""Starting point for the experiments."""

import itertools
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from raise_utils.metrics import ClassificationMetrics
from raise_utils.transforms import Transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from src.data import load_data
from src.util import (
    get_random_hyperparams,
    get_smoothness_mle_approx,
)

hpo_space = {
    "dt": {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": list(range(1, 11)),
        "min_samples_split": [2, 3, 4, 5],
        "max_features": ["sqrt", "log2", 2, 3, 4, 5],
        "transform": ["normalize", "standardize", "minmax", "maxabs"],
    },
    "rf": {
        "n_estimators": [10, 25, 50, 75, 100],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": list(range(1, 11)),
        "min_samples_split": [2, 3, 4, 5],
        "max_features": ["sqrt", "log2", 2, 3, 4, 5],
        "transform": ["normalize", "standardize", "minmax", "maxabs"],
    },
    "lr": {
        "solver": ["liblinear", "saga"],
        "penalty": ["l1", "l2"],
        "C": [0.1, 0.5, 1.0, 1.5, 2.0, 5.0],
        "transform": ["normalize", "standardize", "minmax", "maxabs"],
    },
}

learners = {
    "dt": DecisionTreeClassifier,
    "knn": KNeighborsClassifier,
    "rf": RandomForestClassifier,
    "lr": LogisticRegression,
}

dirs = ["./moot-data/optimize/process/", "./moot-data/hpo/", "./moot-data/config/"]

METRICS = ["pd", "pf", "auc", "f1"]
N_REPEATS = 1

files = itertools.chain.from_iterable(Path(d).rglob("*.csv") for d in dirs)

full_results = {"dataset": []} | {k: [] for k in METRICS}
out_file = open("moot.txt", "w")

for filename in files:
    data_orig = load_data(filename)
    full_results["dataset"].append(filename)

    for i in tqdm(range(N_REPEATS)):
        best_betas = []
        best_configs = []
        best_learners = []
        keep_configs = 5
        num_configs = 50

        learner_choices = random.choices(list(hpo_space.keys()), k=num_configs)
        configs = [
            get_random_hyperparams(hpo_space[learner]) for learner in learner_choices
        ]

        for learner_name, config in zip(learner_choices, configs, strict=False):
            data = deepcopy(data_orig)
            data.x_train = np.array(data.x_train)
            data.y_train = np.array(data.y_train)

            # transform = Transform("ros")
            # transform.apply(data)

            transform_name = config.pop("transform")
            transform = Transform(transform_name)
            transform.apply(data)

            model = learners[learner_name](**config)
            model.fit(data.x_train, data.y_train)
            smoothness = get_smoothness_mle_approx(data, model)

            if np.isinf(smoothness) or smoothness == 0.0:
                continue

            if len(best_betas) < keep_configs or smoothness > min(best_betas):
                best_betas.append(smoothness)
                best_configs.append(config)
                best_learners.append(learner_name)

                best_betas, best_configs, best_learners = zip(
                    *sorted(
                        zip(best_betas, best_configs, best_learners, strict=False),
                        reverse=False,
                        key=lambda x: x[0],
                    ),
                    strict=False,
                )
                best_betas = list(best_betas[:keep_configs])
                best_configs = list(best_configs[:keep_configs])
                best_learners = list(best_learners[:keep_configs])

        best_metrics = [0.0, 0.0, 0.0, 0.0, 0.0]
        for config, learner_name, beta in zip(
            best_configs,
            best_learners,
            best_betas,
            strict=False,
        ):
            data = deepcopy(data_orig)

            transform = Transform("ros")
            transform.apply(data)

            model = learners[learner_name](**config)
            model.fit(data.x_train, data.y_train)
            preds = model.predict(data.x_test)

            metrics = ClassificationMetrics(data.y_test, preds)
            metrics.add_metrics(METRICS)
            results = metrics.get_metrics()
            print(
                f"Config: {config} | Beta: {beta} | Metrics: {results}",
                file=out_file,
            )

            if 1.1 * results[0] + 1.05 * results[2] - results[1] > best_metrics[0]:
                best_metrics = [
                    1.1 * results[0] + 1.05 * results[2] - results[1],
                ] + results[:]

        for idx, metric in enumerate(METRICS):
            if i == 0:
                full_results[metric].append([best_metrics[idx + 1] * 100.0])
            else:
                full_results[metric][-1] += [best_metrics[idx + 1] * 100.0]

    print("---", file=out_file)

for k, v in full_results.items():
    if k != "dataset":
        full_results[k] = np.mean(v, axis=1)

print(full_results)
df = pd.DataFrame(full_results)
df.to_csv("defect.csv", float_format="%.1f")

print(df[["dataset", "pd", "pf", "auc", "f1"]].round(1))
out_file.close()
