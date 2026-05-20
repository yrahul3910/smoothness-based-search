"""Pool-based software-configuration optimization (Round 3, Exp 9).

The config datasets in data/optimize/config/ are lookup tables: each row is a
measured configuration with one or more performance objectives. The task is
*direct configuration optimization* — reveal up to N configs' distance-to-heaven
(d2h) and report the best (lowest) found.

This is the canonical moot/SE-config-tuning task. Unlike Rounds 1–2 (which did
HPO of a surrogate learner and scored R²), here the methods search the
configuration space itself and the metric is d2h, lower is better.

Methods (the Round-2 "8 methods" remap; peek-variants and BOHB drop out because
there is no test-set to peek at and no fidelity axis):
  random  — reveal random pool rows
  tpe     — Optuna TPE over the config space; snap each proposal to nearest pool row
  greedy  — RBF surrogate, pick lowest predicted d2h (no smoothness, ablation)
  ings    — greedy + anti-Laplacian smoothness acquisition + UCB exploration
  smas    — ings + a landscape-β controller that tunes exploration to the
            measured smoothness of the d2h surface
"""

from __future__ import annotations

import logging
import warnings

from pathlib import Path

import numpy as np
import optuna
from scipy.interpolate import RBFInterpolator
from sklearn.tree import DecisionTreeRegressor

from src.data import load_data
from src.smoothness_trees import get_tree_smoothness

optuna.logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore")

BUDGETS = [10, 20, 30, 50]
MAX_BUDGET = max(BUDGETS)
N_INIT = 5
UCB_KAPPA = 0.15
RBF_SMOOTH = 1e-2


# ── Data / d2h ──────────────────────────────────────────────────────────────────

class ConfigProblem:
    """A config-optimization instance built from one lookup-table CSV."""

    def __init__(self, name: str, X: np.ndarray, d2h: np.ndarray, n_obj: int):
        self.name = name
        self.X = X                 # normalized config vectors, shape (n_pool, d)
        self.d2h = d2h             # oracle d2h per row, shape (n_pool,)
        self.n_obj = n_obj
        self.n_pool, self.dim = X.shape
        self.oracle_min = float(d2h.min())
        self.oracle_median = float(np.median(d2h))


def load_problem(fpath: str) -> ConfigProblem:
    df = load_data(fpath)
    cols = df.columns
    y_cols = [c for c in cols if c.endswith(("-", "+"))]
    x_cols = [c for c in cols if not c.endswith(("-", "+"))]

    # Drop constant config columns (zero variance → useless / NaN on normalize).
    x_keep = [c for c in x_cols if df[c].n_unique() > 1]
    X = df.select(x_keep).to_numpy().astype(float)
    # Min-max normalize config dims to [0, 1].
    xmin, xmax = X.min(axis=0), X.max(axis=0)
    span = np.where(xmax > xmin, xmax - xmin, 1.0)
    X = (X - xmin) / span

    # Per-objective "badness" in [0,1], 0 = heaven. Flip maximize objectives.
    badness = []
    for c in y_cols:
        v = df[c].to_numpy().astype(float)
        vmin, vmax = v.min(), v.max()
        nv = (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)
        if c.endswith("+"):
            nv = 1.0 - nv
        badness.append(nv)
    B = np.vstack(badness)                     # (n_obj, n_pool)
    d2h = np.sqrt(np.mean(B ** 2, axis=0))     # mean → comparable across n_obj

    return ConfigProblem(Path(fpath).stem, X, d2h, len(y_cols))


# ── Shared helpers ───────────────────────────────────────────────────────────────

def _zscore(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return np.zeros_like(a) if s < 1e-12 else (a - a.mean()) / s


def _checkpoints(running_min_curve: list[float]) -> dict[int, float]:
    """running_min_curve[i] = best d2h after (i+1) evaluations. Sample at BUDGETS."""
    return {b: running_min_curve[min(b, len(running_min_curve)) - 1] for b in BUDGETS}


def _running_min(d2h_seq: list[float]) -> list[float]:
    out, cur = [], np.inf
    for v in d2h_seq:
        cur = min(cur, v)
        out.append(cur)
    return out


# ── Methods ──────────────────────────────────────────────────────────────────────
# Each returns dict[budget -> best d2h found within that budget].


def m_random(p: ConfigProblem, rng: np.random.Generator) -> dict[int, float]:
    idx = rng.choice(p.n_pool, size=min(MAX_BUDGET, p.n_pool), replace=False)
    return _checkpoints(_running_min([float(p.d2h[i]) for i in idx]))


def _init_reveal(p: ConfigProblem, rng: np.random.Generator):
    init = list(rng.choice(p.n_pool, size=min(N_INIT, p.n_pool), replace=False))
    revealed = set(init)
    seq = [float(p.d2h[i]) for i in init]
    return revealed, seq


def _surrogate_loop(p, rng, *, smoothness: bool, adaptive: bool) -> dict[int, float]:
    """Shared loop for greedy / ings / smas.

    smoothness=False           → greedy (argmin predicted d2h)
    smoothness=True            → ings  (anti-Laplacian + fixed UCB)
    smoothness, adaptive=True  → smas  (UCB weight scaled by landscape β)
    """
    revealed, seq = _init_reveal(p, rng)
    beta_hist: list[float] = []

    n_steps = min(MAX_BUDGET, p.n_pool) - len(revealed)
    for _ in range(n_steps):
        rev_idx = np.array(sorted(revealed))
        X_obs = p.X[rev_idx]
        y_obs = p.d2h[rev_idx]

        cand_idx = np.array([i for i in range(p.n_pool) if i not in revealed])
        if len(cand_idx) == 0:
            break
        X_cand = p.X[cand_idx]

        try:
            rbf = RBFInterpolator(X_obs, y_obs, kernel="linear", smoothing=RBF_SMOOTH)
            pred = rbf(X_cand).flatten()
        except Exception:
            pick = cand_idx[rng.integers(len(cand_idx))]
            revealed.add(int(pick)); seq.append(float(p.d2h[pick]))
            continue

        if not smoothness:
            chosen = cand_idx[int(np.argmin(pred))]
        else:
            u = -pred                                  # higher = better (low d2h)
            # Anti-Laplacian: favor smooth minima of d2h. Neighbor proxy = nearest
            # observed point's predicted u (cheap, pool-based).
            d_to_obs = np.linalg.norm(X_cand[:, None, :] - X_obs[None, :, :], axis=2)
            nn = np.argmin(d_to_obs, axis=1)
            u_neighbor = -y_obs[nn]
            al = 2.0 * u - u_neighbor
            min_dist = d_to_obs.min(axis=1)

            kappa = UCB_KAPPA
            if adaptive:
                dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=2)
                dt.fit(X_obs, y_obs)
                beta = float(get_tree_smoothness(dt))
                beta_hist.append(beta)
                if len(beta_hist) >= 3:
                    bz = (beta - np.mean(beta_hist)) / (np.std(beta_hist) + 1e-12)
                    kappa = UCB_KAPPA * float(1.0 / (1.0 + np.exp(-bz)))  # rough⇒explore
            score = _zscore(al) + kappa * _zscore(min_dist)
            chosen = cand_idx[int(np.argmax(score))]

        revealed.add(int(chosen)); seq.append(float(p.d2h[chosen]))

    return _checkpoints(_running_min(seq))


def m_greedy(p, rng): return _surrogate_loop(p, rng, smoothness=False, adaptive=False)
def m_ings(p, rng):   return _surrogate_loop(p, rng, smoothness=True, adaptive=False)
def m_smas(p, rng):   return _surrogate_loop(p, rng, smoothness=True, adaptive=True)


def m_tpe(p: ConfigProblem, rng: np.random.Generator) -> dict[int, float]:
    """TPE over the config space; snap each proposal to nearest unrevealed pool row."""
    seed = int(rng.integers(2**31 - 1))
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=N_INIT, seed=seed),
    )
    revealed: set[int] = set()
    seq: list[float] = []
    budget = min(MAX_BUDGET, p.n_pool)

    def objective(trial):
        # Suggest a point in normalized [0,1]^dim, snap to nearest unrevealed row.
        q = np.array([trial.suggest_float(f"x{j}", 0.0, 1.0) for j in range(p.dim)])
        dists = np.linalg.norm(p.X - q, axis=1)
        order = np.argsort(dists)
        pick = next((int(i) for i in order if int(i) not in revealed), int(order[0]))
        revealed.add(pick)
        d = float(p.d2h[pick])
        seq.append(d)
        return d

    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return _checkpoints(_running_min(seq))


METHODS = {
    "random": m_random,
    "tpe":    m_tpe,
    "greedy": m_greedy,
    "ings":   m_ings,
    "smas":   m_smas,
}
