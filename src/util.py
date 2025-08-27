"""Utility functions."""

import random

import numpy as np
from raise_utils.data import Data


def get_random_hyperparams(options: dict) -> dict:
    """Get hyperparameters from options."""
    hyperparams = {}
    for key, value in options.items():
        if isinstance(value, list):
            hyperparams[key] = random.choice(value)
        elif isinstance(value, tuple):
            hyperparams[key] = random.randint(value[0], value[1])
    return hyperparams


def get_smoothness_mle_approx(data: Data, model):
    train_size = len(data.x_train)
    subset_size = max(100, train_size // 10)
    subset_idx = np.random.choice(range(train_size), subset_size)
    subset = data.x_train[subset_idx, :]

    eps = 1e-6
    # Generate random directions for all points
    directions = np.random.normal(0, 1, subset.shape)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8

    points_plus = subset + eps * directions
    points_minus = subset - eps * directions

    # Stack all points for batch prediction: [point, point_plus, point_minus, ...]
    batch_points = np.concatenate([subset, points_plus, points_minus], axis=0)
    log_probas = model.predict_log_proba(batch_points)

    # Reshape results: first N are original, next N are plus, last N are minus
    N = subset.shape[0]
    f1 = log_probas[:N, 1]
    f2 = log_probas[N : 2 * N, 1]
    f3 = log_probas[2 * N :, 1]

    cur_smoothness = np.abs(f2 + f3 - 2 * f1) / eps**2
    smoothness = np.mean(cur_smoothness)

    return smoothness
