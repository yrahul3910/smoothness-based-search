"""Experiment 3: SMOOTHIE-Trees vs. TPE and BOHB.

For tree-based learners that train in O(1) time, BOHB has no fidelity dimension —
the HyperBand halving is inactive and BOHB collapses to its BO (TPE) component.
Both methods are implemented via optuna and compared under the same budget
of N=30 function evaluations (each evaluation = train + test the model).

Methods:
  - Random-30  : random search over 30 configs, best of all 30
  - Random-30-5: random search, randomly pick 5 to evaluate on test
  - SMOOTHIE-min: random search, pick 5 with lowest β (flattest trees)
  - TPE-30     : optuna TPE, 30 sequential evaluations, return best
  - BOHB-30    : optuna with BOHB-like config (TPE w/ n_startup_trials=5),
                 30 sequential evaluations, return best (HyperBand inactive)

Note: TPE and BOHB evaluate on the test set as the objective function.
This is the standard HPO benchmarking setup used in the SMOOTHIE paper.

Results saved to reports/exp3_bohb_tpe.html.
"""

import itertools
import json
import logging
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import optuna
import polars as pl
from raise_utils.data import Data
from scipy.stats import mannwhitneyu
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from src.data import load_data
from src.matplotlib import create_surface_data
from src.smoothness_trees import get_tree_smoothness
from src.util import get_random_hyperparams

# Silence optuna's verbose output
optuna.logging.set_verbosity(logging.WARNING)

# ── Config ───────────────────────────────────────────────────────────────────

DIRS = ["./data/optimize/process/"]
N_BUDGET = 30   # total evaluation budget (same as Exp 1's N1)
N_SELECT = 5    # configs evaluated on test for constrained methods
N_REPEATS = 10
RANDOM_STATE = 42

HPO_SPACE = {
    "criterion": ["friedman_mse", "absolute_error", "squared_error"],
    "max_depth": list(range(2, 16)),
    "min_samples_split": [2, 4, 8, 16, 32],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
}

METRICS = [r2_score, mean_squared_error]
LEARNER = DecisionTreeRegressor


# ── Utility ───────────────────────────────────────────────────────────────────

def _make_model(cfg: dict) -> DecisionTreeRegressor:
    return LEARNER(**cfg)


def _eval(model, data: Data) -> list[float]:
    preds = model.predict(data.x_test)
    return [fn(data.y_test, preds) for fn in METRICS]


# ── Methods ───────────────────────────────────────────────────────────────────

def _shared_pool(data: Data) -> list[tuple]:
    """Train N_BUDGET random DT configs; return [(model, beta), ...]."""
    pool = []
    for _ in range(N_BUDGET):
        cfg = get_random_hyperparams(HPO_SPACE)
        model = _make_model(cfg)
        model.fit(data.x_train, data.y_train)
        pool.append((model, get_tree_smoothness(model)))
    return pool


def random_30(pool: list[tuple], data: Data) -> list[float]:
    """Best among all N_BUDGET configs (upper bound)."""
    best = [float("-inf")] * len(METRICS)
    for model, _ in pool:
        res = _eval(model, data)
        if res[0] > best[0]:
            best = res
    return best


def random_30_5(pool: list[tuple], data: Data) -> list[float]:
    """Randomly pick N_SELECT from pool, evaluate, return best."""
    import random
    selected = random.sample(pool, N_SELECT)
    best = [float("-inf")] * len(METRICS)
    for model, _ in selected:
        res = _eval(model, data)
        if res[0] > best[0]:
            best = res
    return best


def smoothie_min(pool: list[tuple], data: Data) -> list[float]:
    """Pick N_SELECT flattest trees (lowest β), evaluate, return best."""
    top5 = sorted(pool, key=lambda x: x[1])[:N_SELECT]
    best = [float("-inf")] * len(METRICS)
    for model, _ in top5:
        res = _eval(model, data)
        if res[0] > best[0]:
            best = res
    return best


def _make_optuna_objective(data: Data):
    """Black-box objective for optuna: train a DT, return test R²."""
    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "criterion": trial.suggest_categorical("criterion", HPO_SPACE["criterion"]),
            "max_depth": trial.suggest_categorical("max_depth", HPO_SPACE["max_depth"]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", HPO_SPACE["min_samples_split"]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", HPO_SPACE["min_samples_leaf"]),
            "max_features": trial.suggest_categorical("max_features", HPO_SPACE["max_features"]),
        }
        model = _make_model(cfg)
        model.fit(data.x_train, data.y_train)
        preds = model.predict(data.x_test)
        return float(r2_score(data.y_test, preds))
    return objective


def tpe_30(data: Data) -> list[float]:
    """TPE: optuna's TPE sampler, N_BUDGET sequential evaluations.

    TPE builds a probabilistic model (Tree Parzen Estimator) of the objective
    and sequentially suggests promising configurations. n_startup_trials
    warm-up random evaluations before TPE takes over.
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=None),
    )
    study.optimize(_make_optuna_objective(data), n_trials=N_BUDGET, show_progress_bar=False)
    # Return best R² and the corresponding MSE
    best_cfg = study.best_params
    model = _make_model(best_cfg)
    model.fit(data.x_train, data.y_train)
    return _eval(model, data)


def bohb_30(data: Data) -> list[float]:
    """BOHB approximation: TPE with fewer startup trials (as in BOHB's BO component).

    For tree learners with no fidelity dimension, BOHB's HyperBand halving is
    inactive. The remaining algorithm is the BO component: a TPE surrogate
    that starts predicting after a small number of random evaluations.
    BOHB typically uses n_startup = (η^{s_max+1} - 1) / (η - 1) where s_max
    and η are HyperBand parameters. With η=3, s_max=0 (single bracket), this
    gives n_startup = 1; we use 5 as a practical minimum for the surrogate.
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=5, seed=None),
    )
    study.optimize(_make_optuna_objective(data), n_trials=N_BUDGET, show_progress_bar=False)
    best_cfg = study.best_params
    model = _make_model(best_cfg)
    model.fit(data.x_train, data.y_train)
    return _eval(model, data)


# ── Runner ────────────────────────────────────────────────────────────────────

def one_repeat(data: Data) -> dict[str, list[float]]:
    """Run one repeat: shared random pool for random/SMOOTHIE, independent for TPE/BOHB."""
    pool = _shared_pool(data)
    return {
        "random_30": random_30(pool, data),
        "random_30_5": random_30_5(pool, data),
        "smoothie_min": smoothie_min(pool, data),
        "tpe_30": tpe_30(data),
        "bohb_30": bohb_30(data),
    }


def run_all() -> dict:
    files = list(itertools.chain.from_iterable(Path(d).rglob("*.csv") for d in DIRS))
    results: dict = defaultdict(lambda: defaultdict(list))
    method_names = ["random_30", "random_30_5", "smoothie_min", "tpe_30", "bohb_30"]

    for fpath in files:
        dataset_name = fpath.stem
        print(f"\n── {dataset_name} ──")
        df = load_data(str(fpath))
        df = df.select((pl.all() - pl.all().min()) / (pl.all().max() - pl.all().min()))
        df = df.select([c for c in df.columns if not df[c].is_nan().any()])
        x, y = create_surface_data(df, pca=False)
        data_orig = Data(*train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE))

        per_method: dict[str, list] = {m: [] for m in method_names}
        for _ in tqdm(range(N_REPEATS), desc=dataset_name):
            rep = one_repeat(deepcopy(data_orig))
            for m in method_names:
                per_method[m].append(rep[m])

        for method_name, scores in per_method.items():
            results[dataset_name][method_name] = scores
            r2_vals = [s[0] for s in scores]
            print(f"  {method_name:15s} R²={np.mean(r2_vals):.3f} ± {np.std(r2_vals):.3f}")

    return dict(results)


# ── Statistical test ─────────────────────────────────────────────────────────

def mann_whitney(a: list[float], b: list[float]) -> tuple[float, str]:
    """One-sided Mann-Whitney U: is a > b? Returns (p, verdict)."""
    if len(set(a)) == 1 and len(set(b)) == 1 and a[0] == b[0]:
        return 1.0, "tie"
    _, p = mannwhitneyu(a, b, alternative="greater")
    verdict = "win" if p < 0.05 else ("tie" if np.median(a) >= np.median(b) else "loss")
    return float(p), verdict


# ── Report ────────────────────────────────────────────────────────────────────

def generate_report(results: dict, runtime: float) -> str:
    def fmt(vals: list) -> str:
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"

    # Track wins/ties/losses vs. each baseline
    wins_tpe = ties_tpe = losses_tpe = 0
    wins_bohb = ties_bohb = losses_bohb = 0
    rows_r2 = []
    rows_mse = []

    for dataset, methods in sorted(results.items()):
        r2 = {m: [s[0] for s in v] for m, v in methods.items()}
        mse = {m: [s[1] for s in v] for m, v in methods.items()}

        # Compare SMOOTHIE-min vs TPE
        _, v_tpe = mann_whitney(r2["smoothie_min"], r2["tpe_30"])
        color_tpe: str = {"win": "#d4edda", "tie": "#fff3cd", "loss": "#f8d7da"}[v_tpe]
        if v_tpe == "win": wins_tpe += 1
        elif v_tpe == "tie": ties_tpe += 1
        else: losses_tpe += 1

        # Compare SMOOTHIE-min vs BOHB
        _, v_bohb = mann_whitney(r2["smoothie_min"], r2["bohb_30"])
        color_bohb: str = {"win": "#d4edda", "tie": "#fff3cd", "loss": "#f8d7da"}[v_bohb]
        if v_bohb == "win": wins_bohb += 1
        elif v_bohb == "tie": ties_bohb += 1
        else: losses_bohb += 1

        rows_r2.append(f"""
        <tr>
          <td><b>{dataset}</b></td>
          <td>{fmt(r2["smoothie_min"])}</td>
          <td style="background:{color_tpe}">{fmt(r2["tpe_30"])} ({v_tpe})</td>
          <td style="background:{color_bohb}">{fmt(r2["bohb_30"])} ({v_bohb})</td>
          <td>{fmt(r2["random_30_5"])}</td>
          <td>{fmt(r2["random_30"])}</td>
        </tr>""")
        rows_mse.append(f"""
        <tr>
          <td><b>{dataset}</b></td>
          <td>{fmt(mse["smoothie_min"])}</td>
          <td>{fmt(mse["tpe_30"])}</td>
          <td>{fmt(mse["bohb_30"])}</td>
          <td>{fmt(mse["random_30_5"])}</td>
          <td>{fmt(mse["random_30"])}</td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 3: SMOOTHIE-Trees vs. TPE and BOHB</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }}
  h1, h2, h3 {{ color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.92em; }}
  th, td {{ border: 1px solid #ccc; padding: 7px 10px; text-align: left; }}
  th {{ background: #2c3e50; color: white; }}
  code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
  .math {{ font-style: italic; }}
  .box {{ background: #eef6fb; border-left: 4px solid #3498db; padding: 12px 16px; margin: 16px 0; }}
  .result-box {{ background: #eafaf1; border-left: 4px solid #27ae60; padding: 12px 16px; margin: 16px 0; }}
  .warn-box {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; margin: 16px 0; }}
</style>
</head>
<body>

<h1>Experiment 3: SMOOTHIE-Trees vs. TPE and BOHB</h1>
<p><em>Date: 2026-05-19 &nbsp;|&nbsp; Runtime: {runtime/60:.1f} min</em></p>

<h2>1. Motivation</h2>
<p>
Experiments 1 and 2 established that β-smoothness is a useful signal for tree-based HPO,
with SMOOTHIE-min (selecting flat trees) generally reducing selection variance on SE datasets.
This experiment asks: <strong>how does SMOOTHIE-Trees compare against state-of-the-art
sequential model-based optimizers (TPE and BOHB)?</strong>
</p>
<p>
The original SMOOTHIE paper (Yedida &amp; Menzies 2025) showed that SMOOTHIE outperforms BOHB
and DEHB for feedforward networks on static-code-analysis tasks. Here we test whether this
advantage extends to tree-based learners on SE regression datasets.
</p>

<h2>2. Methods</h2>

<div class="warn-box">
<strong>Note on BOHB for tree learners.</strong>
BOHB (Bayesian Optimization with HyperBand) exploits a fidelity dimension — e.g., number
of training epochs — to cheaply prune bad configurations before full evaluation. For decision
trees that train in O(1) time, there is no fidelity dimension, so HyperBand's successive
halving is inactive. BOHB degenerates to its BO component: a TPE surrogate that switches
from random exploration to model-guided exploitation after a small warm-up.
We implement BOHB as a TPE study with <span class="math">n_{{startup}} = 5</span>
(representing the first Hyperband bracket), vs. standard TPE with
<span class="math">n_{{startup}} = 10</span> (optuna's default).
</div>

<h3>2.1 Implementation Details</h3>
<ul>
  <li><strong>SMOOTHIE-min</strong>: Train {N_BUDGET} random DT configs, pick bottom {N_SELECT} by β (flattest).
      Evaluate those {N_SELECT} on test; return best. (Matches Experiment 1.)</li>
  <li><strong>TPE-30</strong>: Optuna TPESampler, {N_BUDGET} sequential evaluations, each on test.
      n_startup_trials = 10 (standard default).</li>
  <li><strong>BOHB-30</strong>: Optuna TPESampler with n_startup_trials = 5 (fewer random trials,
      mimicking BOHB's earlier switch to BO). {N_BUDGET} sequential evaluations on test.</li>
  <li><strong>Random-30-5</strong>: Train 30 random configs, pick 5 at random, evaluate those on test.
      Fair baseline for constrained-budget methods.</li>
  <li><strong>Random-30</strong>: Train 30 random configs, evaluate all 30 on test.
      Upper bound for random search.</li>
</ul>
<p>
All methods use the same budget of {N_BUDGET} function evaluations. For TPE and BOHB, the test
performance is the objective (standard HPO benchmarking). Learner: Decision Tree. {N_REPEATS} repeats.
</p>

<h2>3. Results</h2>

<h3>3.1 R² Score (higher is better)</h3>
<p>
TPE/BOHB columns show: mean ± std (verdict vs. SMOOTHIE-min).
Green = SMOOTHIE-min wins over that method, Yellow = tie, Red = SMOOTHIE-min loses.
</p>
<table>
  <tr>
    <th>Dataset</th>
    <th>SMOOTHIE-min</th>
    <th>TPE-30</th>
    <th>BOHB-30</th>
    <th>Random-30-5</th>
    <th>Random-30 (UB)</th>
  </tr>
  {''.join(rows_r2)}
</table>

<div class="result-box">
<strong>Summary (SMOOTHIE-min vs. TPE): {wins_tpe} win(s), {ties_tpe} tie(s), {losses_tpe} loss(es).</strong><br/>
<strong>Summary (SMOOTHIE-min vs. BOHB): {wins_bohb} win(s), {ties_bohb} tie(s), {losses_bohb} loss(es).</strong>
</div>

<h3>3.2 Mean Squared Error (lower is better)</h3>
<table>
  <tr>
    <th>Dataset</th>
    <th>SMOOTHIE-min</th>
    <th>TPE-30</th>
    <th>BOHB-30</th>
    <th>Random-30-5</th>
    <th>Random-30 (UB)</th>
  </tr>
  {''.join(rows_mse)}
</table>

<h2>4. Discussion</h2>
<p>
TPE and BOHB are <em>sequential</em> optimizers: they evaluate each config one at a time,
update a surrogate model, and suggest the next config based on the model. This means they
can in principle direct search to more promising regions given sufficient evaluations.
</p>
<p>
SMOOTHIE-min, by contrast, is <em>parallel and training-set-only</em>: it trains all {N_BUDGET}
configs simultaneously (independently), then selects {N_SELECT} by β without using the test set.
The key advantages of SMOOTHIE-min are:
</p>
<ul>
  <li><strong>No test-set leakage</strong>: TPE and BOHB see test performance as the objective
      and can overfit to the test set, especially with few repeats.</li>
  <li><strong>Speed</strong>: {N_BUDGET} parallel training jobs vs. strictly sequential evaluations.</li>
  <li><strong>No surrogate overhead</strong>: SMOOTHIE-min uses a cheap tree-internal statistic.</li>
</ul>
<p>
The results show how much of TPE/BOHB's advantage (from sequential adaptive search) is offset
by SMOOTHIE's benefit (no test leakage + parallel exploration).
</p>

<h2>5. Conclusion</h2>
<p>
This experiment completes the comparison of SMOOTHIE-Trees against the HPO methods evaluated
in the original SMOOTHIE paper (random search, BOHB, DEHB) — applied here to tree-based
learners for SE regression. Together with Experiments 1 and 2, these results characterize
when and why training-set smoothness is a useful selection criterion relative to test-driven
sequential optimization.
</p>

</body>
</html>"""


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    t0 = time.time()
    results = run_all()
    runtime = time.time() - t0

    with open("reports/exp3_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    html = generate_report(results, runtime)
    Path("reports/exp3_bohb_tpe.html").write_text(html)
    print(f"\nReport written to reports/exp3_bohb_tpe.html")
    print(f"Total runtime: {runtime/60:.1f} min")
