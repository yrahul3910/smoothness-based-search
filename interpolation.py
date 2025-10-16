"""Main file for interpolation search."""

import math
import random

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist

from src.data import load_data


def random_projection(
    x: pl.DataFrame,
    dist: np.ndarray,
) -> tuple[int, int]:
    """FASTMAP-based random projection.

    # Arguments:
        - `x` - The dataset
        - `dist` - The distance matrix

    # Returns:
        The indices of the two chosen points.
    """
    x0_idx = random.randint(0, len(x))
    x1_idx = int(np.argmax([dist[x0_idx][i] for i in range(len(x))]))
    x2_idx = int(np.argmax([dist[x1_idx][i] for i in range(len(x))]))

    return x1_idx, x2_idx


def get_closest_line(
    lines: list[tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, float, float]],
    pt: pl.DataFrame,
) -> tuple[int, np.ndarray, float, float]:
    """Return the index of the closest line, the projection of the sample on the line, and that distance.

    # Arguments:
        - `lines`: The list of lines, where each element is a tuple:
            (point1, point2, distanceVector, funcValue1, funcValue2)
            Note that `distanceVector` should be `point2 - point1`.
        - `pt`: The query point

    # Returns:
        A tuple containing:
            - The index of the closest line
            - The projection of the sample on the line
            - The linear interpolated value of the function at that point
            - The distance from `pt` to the projection
    """
    computed: list[tuple[int, np.ndarray, float, float]] = []

    for i, (x1, x2, d, y1, y2) in enumerate(lines):
        d_np = np.array(d)

        # Parameter for projection
        t = np.dot(np.array(pt - x1), d_np) / np.dot(d_np, d_np)

        # Projection point
        p = np.array(x1) + t * d_np

        # Orthogonal distance
        dist = np.linalg.norm(pt - p)

        # Estimated value
        f_numerator = y1 * np.linalg.norm(np.array(x2 - p)) + y2 * np.linalg.norm(
            np.array(p - x1),
        )
        f = float(f_numerator / np.linalg.norm(np.array(x2 - x1)))

        computed.append((i, p, f, float(dist)))

    return min(computed, key=lambda x: x[-1])


if __name__ == "__main__":
    data = load_data("./data/optimize/process/pom3c.csv")

    x_cols = [col for col in data.columns if not col.endswith(("-", "+"))]
    y_cols = [col for col in data.columns if col.endswith(("-", "+"))]

    x = data.select(*[pl.col(col) for col in x_cols])
    dist = pdist(x, metric="cityblock")

    """
    Start with a random sample of say 10 points. You can now interpolate between those. We need to follow a few rules:
        1. Ignore regions of zero gradient.
        2. Higher second derivative is preferred, because at the minimum, the second derivative is maximum.
            a. Also: the second derivative will be positive at the minimum (basic calculus).
        3. If we need to sample at a new point, we first need to check if we're within a covering ball of a point we
          have already checked; if so, use known properties (smoothness, gradient, etc.) to make an estimate. Otherwise,
          sample.
            a. The hard part here is you could be close to a line between two points, which is difficult to figure out.
    """

    y = data.select(
        pl.col(col) if col.endswith("-") else 1 - pl.col(col) for col in y_cols
    )
    y = y.map_rows(lambda row: math.sqrt(sum(val**2 for val in row)))

    N_INITIAL = 25
    chosen_idx = []
    lines: list[tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, float, float]] = (
        []
    )  # x1, x2, d, y1, y2 (d = direction vector)

    # This is 2 * N_INITIAL = 50 evaluations
    for _ in range(N_INITIAL):
        x1_idx, x2_idx = random_projection(x, dist)
        chosen_idx.extend([x1_idx, x2_idx])

        x1, x2 = x[x1_idx], x[x2_idx]
        lines.append((x1, x2, x2 - x1, float(y[x1_idx]), float(y[x2_idx])))

    N_TRIES = 100
    for _ in range(N_TRIES):
        # Pick a random point
        idx = random.randint(0, len(x))

        line_idx, proj, proj_value, dist = get_closest_line(lines, x[idx])
