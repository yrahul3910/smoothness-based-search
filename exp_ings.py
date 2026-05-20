"""Experiment 4: Interpolation-Guided Gradient Search (INGS).

Mathematical basis:
  1. Interpolation error bound: For a β-smooth HPO landscape f, the error of
     a linear interpolant L satisfies |f(h) - L(h)| ≤ (β/8)||h - h_nearest||².
     SE data has low β ⟹ L is a trustworthy cheap surrogate.

  2. Hessian from residuals: The center-difference Hessian at any evaluated
     point h is ∇²f(h) ≈ [f(h+δ) + f(h-δ) - 2f(h)] / δ². Since L(h) is the
     linear interpolant through h-δ and h+δ, this gives ∇²f(h) = -2r(h)/δ²
     where r(h) = f(h) - L(h) is the interpolation residual. In other words,
     points *above* the interpolant have negative curvature (local maxima).

  3. Anti-Laplacian score: Combining (1) and (2), the best candidate to
     evaluate next is the one where the RBF surrogate is highest AND where
     that surrogate is a local maximum relative to its neighbors:
         score(h) = 2·RBF(h) - mean(RBF(neighbors))
     A positive residual (f > interpolant) signals a predicted peak.

Algorithm (INGS):
  Phase 0 – Sample N_init=10 configs uniformly at random.
             Proxy: R² on a fixed 20% validation split of training data.
  Phase 1 – Iteratively:
             (a) Fit RBF interpolant on observed (encoded_config, val_R²).
             (b) Score M=40 random candidates by anti-Laplacian score.
             (c) If score range < ε (flat surrogate): explore randomly.
                 Else: evaluate the highest-scoring candidate.
  Return   – Best config by val R²; retrain on full training data; test.

Ablation:
  INGS-grad: same algorithm, score = RBF(h) (no Hessian correction).

All methods compared under the same N=30 evaluation budget.
Results saved to reports/exp4_ings.html.
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
from scipy.interpolate import RBFInterpolator
from scipy.stats import mannwhitneyu
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from src.data import load_data
from src.matplotlib import create_surface_data
from src.smoothness_trees import get_tree_smoothness
from src.util import get_random_hyperparams

optuna.logging.set_verbosity(logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────

DIRS = ["./data/optimize/process/"]
N_BUDGET = 30
N_SELECT = 5
N_INIT = 10       # INGS initialization budget
N_CANDIDATES = 40  # candidates scored per iteration
N_REPEATS = 10
RANDOM_STATE = 42
FLAT_EPS = 1e-3   # surrogate score-range threshold for flat-region escape

HPO_SPACE = {
    "criterion": ["friedman_mse", "absolute_error", "squared_error"],
    "max_depth": list(range(2, 16)),
    "min_samples_split": [2, 4, 8, 16, 32],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
}

METRICS = [r2_score, mean_squared_error]
LEARNER = DecisionTreeRegressor


# ── Utility ────────────────────────────────────────────────────────────────────

def _make_model(cfg: dict) -> DecisionTreeRegressor:
    return LEARNER(**cfg)


def _eval(model, data: Data) -> list[float]:
    preds = model.predict(data.x_test)
    return [fn(data.y_test, preds) for fn in METRICS]


def _encode(cfg: dict) -> np.ndarray:
    """Encode a discrete HPO config as a normalized float vector."""
    vec = []
    for key, options in HPO_SPACE.items():
        idx = options.index(cfg[key])
        vec.append(idx / max(1, len(options) - 1))
    return np.array(vec, dtype=float)


def _neighbor_vecs(cfg: dict) -> np.ndarray:
    """Return encoded vectors of all adjacent configs (one step per HPO axis)."""
    rows = []
    base = _encode(cfg)
    for dim_idx, (key, options) in enumerate(HPO_SPACE.items()):
        cur_idx = options.index(cfg[key])
        for delta in (-1, +1):
            neigh_idx = cur_idx + delta
            if 0 <= neigh_idx < len(options):
                v = base.copy()
                v[dim_idx] = neigh_idx / max(1, len(options) - 1)
                rows.append(v)
    return np.array(rows) if rows else np.empty((0, len(HPO_SPACE)))


# ── Shared-pool baselines ──────────────────────────────────────────────────────

def _shared_pool(data: Data) -> list[tuple]:
    pool = []
    for _ in range(N_BUDGET):
        cfg = get_random_hyperparams(HPO_SPACE)
        model = _make_model(cfg)
        model.fit(data.x_train, data.y_train)
        pool.append((model, get_tree_smoothness(model)))
    return pool


def random_30(pool, data):
    best = [float("-inf")] * len(METRICS)
    for model, _ in pool:
        res = _eval(model, data)
        if res[0] > best[0]:
            best = res
    return best


def random_30_5(pool, data):
    import random
    selected = random.sample(pool, N_SELECT)
    best = [float("-inf")] * len(METRICS)
    for model, _ in selected:
        res = _eval(model, data)
        if res[0] > best[0]:
            best = res
    return best


def smoothie_min(pool, data):
    top5 = sorted(pool, key=lambda x: x[1])[:N_SELECT]
    best = [float("-inf")] * len(METRICS)
    for model, _ in top5:
        res = _eval(model, data)
        if res[0] > best[0]:
            best = res
    return best


def smoothie_max(pool, data):
    top5 = sorted(pool, key=lambda x: x[1], reverse=True)[:N_SELECT]
    best = [float("-inf")] * len(METRICS)
    for model, _ in top5:
        res = _eval(model, data)
        if res[0] > best[0]:
            best = res
    return best


def _make_optuna_objective(data: Data):
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
        return float(r2_score(data.y_test, model.predict(data.x_test)))
    return objective


def tpe_30(data: Data) -> list[float]:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=None),
    )
    study.optimize(_make_optuna_objective(data), n_trials=N_BUDGET, show_progress_bar=False)
    model = _make_model(study.best_params)
    model.fit(data.x_train, data.y_train)
    return _eval(model, data)


def bohb_30(data: Data) -> list[float]:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=5, seed=None),
    )
    study.optimize(_make_optuna_objective(data), n_trials=N_BUDGET, show_progress_bar=False)
    model = _make_model(study.best_params)
    model.fit(data.x_train, data.y_train)
    return _eval(model, data)


# ── INGS ───────────────────────────────────────────────────────────────────────

def _ings(data: Data, use_correction: bool) -> list[float]:
    """Core INGS implementation.

    use_correction=True  → anti-Laplacian score: 2·RBF(h) - mean(RBF(neighbors))
    use_correction=False → plain gradient following: RBF(h) only
    """
    # Fixed validation split carved out of training data (no test leakage)
    x_tr, x_val, y_tr, y_val = train_test_split(
        data.x_train, data.y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    cfgs: list[dict] = []
    vecs: list[np.ndarray] = []
    proxies: list[float] = []  # val R² for each evaluated config

    # Phase 0 – random warm-up
    for _ in range(N_INIT):
        cfg = get_random_hyperparams(HPO_SPACE)
        model = _make_model(cfg)
        model.fit(x_tr, y_tr)
        p = float(r2_score(y_val, model.predict(x_val)))
        cfgs.append(cfg)
        vecs.append(_encode(cfg))
        proxies.append(p)

    # Phase 1 – interpolation-guided search
    for _ in range(N_BUDGET - N_INIT):
        X_obs = np.array(vecs)
        y_obs = np.array(proxies)

        # Fit RBF interpolant (linear kernel ↔ our theoretical linear interpolation)
        try:
            rbf = RBFInterpolator(X_obs, y_obs, kernel="linear", smoothing=1e-2)
        except Exception:
            cfg = get_random_hyperparams(HPO_SPACE)
            model = _make_model(cfg)
            model.fit(x_tr, y_tr)
            cfgs.append(cfg)
            vecs.append(_encode(cfg))
            proxies.append(float(r2_score(y_val, model.predict(x_val))))
            continue

        # Sample random candidates and score them
        candidates = [get_random_hyperparams(HPO_SPACE) for _ in range(N_CANDIDATES)]
        cand_vecs = np.array([_encode(c) for c in candidates])
        rbf_preds = rbf(cand_vecs).flatten()

        if use_correction:
            # Anti-Laplacian: score = 2·RBF(h) - mean(RBF(neighbors))
            # Positive when h is a local maximum in the surrogate (predicted peak).
            # Derivation: ∇²f(h) ≈ -2r(h)/δ² and r = f - L, so
            #   score = RBF(h) - (δ²/2)·∇²RBF(h)
            #         = RBF(h) - [RBF(h+δ)+RBF(h-δ)-2·RBF(h)]/2
            #         = 2·RBF(h) - mean_neighbor_RBF
            scores = np.empty(len(candidates))
            for i, (c, pred) in enumerate(zip(candidates, rbf_preds)):
                nv = _neighbor_vecs(c)
                if len(nv):
                    neigh_preds = rbf(nv).flatten()
                    scores[i] = 2.0 * pred - float(np.mean(neigh_preds))
                else:
                    scores[i] = pred
        else:
            scores = rbf_preds

        # Flat surrogate → explore randomly; otherwise exploit
        if float(scores.max() - scores.min()) < FLAT_EPS:
            cfg_next = get_random_hyperparams(HPO_SPACE)
        else:
            cfg_next = candidates[int(np.argmax(scores))]

        model = _make_model(cfg_next)
        model.fit(x_tr, y_tr)
        p = float(r2_score(y_val, model.predict(x_val)))
        cfgs.append(cfg_next)
        vecs.append(_encode(cfg_next))
        proxies.append(p)

    # Retrain best config (by val R²) on full training data, then evaluate on test
    best_cfg = cfgs[int(np.argmax(proxies))]
    best_model = _make_model(best_cfg)
    best_model.fit(data.x_train, data.y_train)
    return _eval(best_model, data)


def ings(data: Data) -> list[float]:
    """INGS with anti-Laplacian (Hessian-corrected) surrogate score."""
    return _ings(data, use_correction=True)


def ings_grad(data: Data) -> list[float]:
    """INGS-grad: gradient-only (no Hessian correction)."""
    return _ings(data, use_correction=False)


# ── Runner ─────────────────────────────────────────────────────────────────────

METHOD_NAMES = [
    "random_30", "random_30_5",
    "smoothie_min", "smoothie_max",
    "tpe_30", "bohb_30",
    "ings", "ings_grad",
]


def one_repeat(data: Data) -> dict[str, list[float]]:
    pool = _shared_pool(data)
    return {
        "random_30":   random_30(pool, data),
        "random_30_5": random_30_5(pool, data),
        "smoothie_min": smoothie_min(pool, data),
        "smoothie_max": smoothie_max(pool, data),
        "tpe_30":      tpe_30(data),
        "bohb_30":     bohb_30(data),
        "ings":        ings(data),
        "ings_grad":   ings_grad(data),
    }


def run_all() -> dict:
    files = list(itertools.chain.from_iterable(Path(d).rglob("*.csv") for d in DIRS))
    results: dict = defaultdict(lambda: defaultdict(list))

    for fpath in files:
        dataset_name = fpath.stem
        print(f"\n── {dataset_name} ──")
        df = load_data(str(fpath))
        df = df.select((pl.all() - pl.all().min()) / (pl.all().max() - pl.all().min()))
        df = df.select([c for c in df.columns if not df[c].is_nan().any()])
        x, y = create_surface_data(df, pca=False)
        data_orig = Data(*train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE))

        per_method: dict[str, list] = {m: [] for m in METHOD_NAMES}
        for _ in tqdm(range(N_REPEATS), desc=dataset_name):
            rep = one_repeat(deepcopy(data_orig))
            for m in METHOD_NAMES:
                per_method[m].append(rep[m])

        for method_name, scores in per_method.items():
            results[dataset_name][method_name] = scores
            r2_vals = [s[0] for s in scores]
            print(f"  {method_name:15s}  R²={np.mean(r2_vals):.3f} ± {np.std(r2_vals):.3f}")

    return dict(results)


# ── Statistical test ──────────────────────────────────────────────────────────

def mann_whitney(a: list[float], b: list[float]) -> tuple[float, str]:
    if len(set(a)) == 1 and len(set(b)) == 1 and a[0] == b[0]:
        return 1.0, "tie"
    _, p = mannwhitneyu(a, b, alternative="greater")
    verdict = "win" if p < 0.05 else ("tie" if np.median(a) >= np.median(b) else "loss")
    return float(p), verdict


# ── Report ─────────────────────────────────────────────────────────────────────

def generate_report(results: dict, runtime: float) -> str:
    def fmt(vals):
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"

    def color(v):
        return {"win": "#d4edda", "tie": "#fff3cd", "loss": "#f8d7da"}[v]

    # Win/tie/loss counters for INGS vs each reference method
    ref_methods = ["tpe_30", "bohb_30", "smoothie_min", "random_30_5"]
    counters: dict[str, dict[str, dict[str, int]]] = {
        algo: {ref: {"win": 0, "tie": 0, "loss": 0} for ref in ref_methods}
        for algo in ("ings", "ings_grad")
    }

    rows_r2 = []
    rows_ablation = []

    for dataset, methods in sorted(results.items()):
        r2 = {m: [s[0] for s in v] for m, v in methods.items()}

        # INGS vs reference methods
        verdicts: dict[str, dict[str, str]] = {}
        for algo in ("ings", "ings_grad"):
            verdicts[algo] = {}
            for ref in ref_methods:
                _, v = mann_whitney(r2[algo], r2[ref])
                verdicts[algo][ref] = v
                counters[algo][ref][v] += 1

        # Full comparison row
        rows_r2.append(f"""
        <tr>
          <td><b>{dataset}</b></td>
          <td style="background:{color(verdicts['ings']['tpe_30'])}">{fmt(r2['ings'])}</td>
          <td style="background:{color(verdicts['ings_grad']['tpe_30'])}">{fmt(r2['ings_grad'])}</td>
          <td>{fmt(r2['tpe_30'])}</td>
          <td>{fmt(r2['bohb_30'])}</td>
          <td>{fmt(r2['smoothie_min'])}</td>
          <td>{fmt(r2['smoothie_max'])}</td>
          <td>{fmt(r2['random_30_5'])}</td>
          <td>{fmt(r2['random_30'])}</td>
        </tr>""")

        # Ablation row: INGS vs INGS-grad
        _, v_ablation = mann_whitney(r2["ings"], r2["ings_grad"])
        rows_ablation.append(f"""
        <tr>
          <td><b>{dataset}</b></td>
          <td>{fmt(r2['ings'])}</td>
          <td>{fmt(r2['ings_grad'])}</td>
          <td style="background:{color(v_ablation)}">{v_ablation}</td>
        </tr>""")

    def summary_row(algo, ref):
        c = counters[algo][ref]
        return f"{c['win']}W / {c['tie']}T / {c['loss']}L"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 4: INGS — Interpolation-Guided Gradient Search</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 1050px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }}
  h1, h2, h3 {{ color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.91em; }}
  th, td {{ border: 1px solid #ccc; padding: 7px 10px; text-align: left; }}
  th {{ background: #2c3e50; color: white; }}
  code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 0.9em; }}
  .math {{ font-style: italic; }}
  .box {{ background: #eef6fb; border-left: 4px solid #3498db; padding: 12px 16px; margin: 16px 0; }}
  .result-box {{ background: #eafaf1; border-left: 4px solid #27ae60; padding: 12px 16px; margin: 16px 0; }}
  .warn-box {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; margin: 16px 0; }}
  pre {{ background: #f8f8f8; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 0.88em; }}
</style>
</head>
<body>

<h1>Experiment 4: Interpolation-Guided Gradient Search (INGS)</h1>
<p><em>Date: 2026-05-19 &nbsp;|&nbsp; Runtime: {runtime/60:.1f} min</em></p>

<h2>1. Motivation</h2>
<p>
Experiments 1–3 established that SE HPO landscapes are low-β (flat), making
β-smoothness a useful but weaker signal than direct test feedback (TPE wins on all 10 datasets).
This experiment asks: <strong>can we exploit the low-β structure more directly, by using linear
interpolation over the HPO landscape to estimate gradients and navigate toward performance peaks
without touching the test set?</strong>
</p>
<p>
The key mathematical insight is that for β-smooth functions, the interpolation error is
upper-bounded by a term proportional to β. Because SE data has low β, a linear interpolant
is a trustworthy surrogate. Furthermore, the Hessian at any point can be read off from the
interpolation residual: <span class="math">∇²f(h) ≈ −2r(h)/δ²</span>, where r(h) = f(h) − L(h).
A point above its linear interpolant has negative Hessian, i.e., is a local maximum.
</p>

<h2>2. Algorithm</h2>

<div class="box">
<strong>Anti-Laplacian scoring:</strong>
Combining the gradient estimate (high RBF prediction) and the Hessian correction (local maximum
indicator), the score for a candidate h is:
<pre>
score(h) = 2·RBF(h) − mean(RBF(neighbors of h))
</pre>
Derivation: score = RBF(h) − (δ²/2)·∇²RBF(h)
                  = RBF(h) − [RBF(h+δ) + RBF(h−δ) − 2·RBF(h)] / 2
                  = 2·RBF(h) − mean_neighbor_RBF<br/>
When score > RBF(h): the surrogate predicts h is a peak above its neighbors → evaluate it.
When the score range is flat (< ε): explore randomly to escape the plateau.
</div>

<h3>2.1 INGS Step-by-Step</h3>
<ol>
  <li><strong>Phase 0 — Initialization ({N_INIT} configs):</strong> Sample uniformly at random.
      Proxy = R² on a fixed 20% validation split of <em>training</em> data (no test leakage).</li>
  <li><strong>Phase 1 — Iterative search ({N_BUDGET - N_INIT} steps):</strong>
    <ol type="a">
      <li>Fit RBF interpolant (linear kernel) on observed (encoded config, val R²) pairs.</li>
      <li>Sample {N_CANDIDATES} random candidate configs.</li>
      <li>Compute anti-Laplacian score for each candidate.</li>
      <li>If <code>max_score − min_score &lt; {FLAT_EPS}</code> (flat surrogate): explore randomly.
          Otherwise: evaluate the highest-scoring candidate.</li>
    </ol>
  </li>
  <li><strong>Return:</strong> Best config by val R²; retrain on full training data; evaluate on test.</li>
</ol>

<h3>2.2 Ablation — INGS-grad</h3>
<p>Identical to INGS, but the score is <code>RBF(h)</code> only (no Hessian correction).
This isolates the contribution of the anti-Laplacian term.</p>

<h3>2.3 Key differences from prior methods</h3>
<table>
  <tr><th>Property</th><th>SMOOTHIE</th><th>TPE/BOHB</th><th>INGS</th></tr>
  <tr><td>Proxy signal</td><td>β-smoothness (training)</td><td>R² (test)</td><td>R² (validation split of train)</td></tr>
  <tr><td>Sequential?</td><td>No (parallel)</td><td>Yes</td><td>Yes</td></tr>
  <tr><td>Surrogate model</td><td>None</td><td>Tree Parzen Estimator</td><td>RBF interpolant</td></tr>
  <tr><td>Uses test set?</td><td>No</td><td>Yes</td><td>No</td></tr>
  <tr><td>Hessian info</td><td>No</td><td>No</td><td>Yes (anti-Laplacian)</td></tr>
</table>

<h2>3. Results</h2>

<h3>3.1 R² Score — All Methods (higher is better)</h3>
<p>INGS and INGS-grad cells are colored by their verdict vs. TPE-30.
Green = win, yellow = tie, red = loss.</p>
<table>
  <tr>
    <th>Dataset</th>
    <th>INGS</th>
    <th>INGS-grad</th>
    <th>TPE-30</th>
    <th>BOHB-30</th>
    <th>SMOOTHIE-min</th>
    <th>SMOOTHIE-max</th>
    <th>Random-30-5</th>
    <th>Random-30 (UB)</th>
  </tr>
  {''.join(rows_r2)}
</table>

<div class="result-box">
<strong>INGS vs. TPE-30:</strong> {summary_row('ings', 'tpe_30')} &nbsp;|&nbsp;
<strong>INGS vs. BOHB-30:</strong> {summary_row('ings', 'bohb_30')}<br/>
<strong>INGS vs. SMOOTHIE-min:</strong> {summary_row('ings', 'smoothie_min')} &nbsp;|&nbsp;
<strong>INGS vs. Random-30-5:</strong> {summary_row('ings', 'random_30_5')}<br/><br/>
<strong>INGS-grad vs. TPE-30:</strong> {summary_row('ings_grad', 'tpe_30')} &nbsp;|&nbsp;
<strong>INGS-grad vs. BOHB-30:</strong> {summary_row('ings_grad', 'bohb_30')}<br/>
<strong>INGS-grad vs. SMOOTHIE-min:</strong> {summary_row('ings_grad', 'smoothie_min')} &nbsp;|&nbsp;
<strong>INGS-grad vs. Random-30-5:</strong> {summary_row('ings_grad', 'random_30_5')}
</div>

<h3>3.2 Ablation: INGS vs. INGS-grad (effect of Hessian correction)</h3>
<table>
  <tr>
    <th>Dataset</th>
    <th>INGS (anti-Laplacian)</th>
    <th>INGS-grad (gradient only)</th>
    <th>Verdict (INGS &gt; INGS-grad?)</th>
  </tr>
  {''.join(rows_ablation)}
</table>

<h2>4. Discussion</h2>

<h3>4.1 INGS vs. TPE/BOHB</h3>
<p>
Both INGS and TPE see information sequentially and use a surrogate model, but their
surrogates differ fundamentally. TPE builds a probabilistic model over the test-set
objective, effectively treating HPO as test-set optimization. INGS builds a linear
interpolant over the validation-set proxy, guided by the β-smoothness bound.
</p>
<p>
The theoretical advantage of INGS over TPE: when the HPO landscape is genuinely β-smooth
(low curvature), the linear interpolant is as accurate as a GP/TPE surrogate, while
being computationally simpler and requiring no test-set access.
</p>

<h3>4.2 Ablation Interpretation</h3>
<p>
If INGS &gt; INGS-grad on a dataset, the Hessian correction helps — the landscape has
meaningful curvature that the anti-Laplacian can exploit. If INGS-grad ≥ INGS, the landscape
is smooth enough that pure gradient following is sufficient (the flat region escape fires more
often and the Hessian correction adds noise).
</p>

<h3>4.3 Relationship to the β-interpolation bound</h3>
<p>
The proxy signal used by INGS (validation R²) is richer than β-smoothness (a training-only
signal) but cheaper than sequential test-set feedback (which requires N sequential
train+test cycles). INGS fits on the spectrum between SMOOTHIE and TPE: it sees
held-out performance (validation), but from a held-out portion of training data,
not the final test set. This is a principled middle ground for low-leakage HPO.
</p>

</body>
</html>"""


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    t0 = time.time()
    results = run_all()
    runtime = time.time() - t0

    with open("reports/exp4_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    html = generate_report(results, runtime)
    Path("reports/exp4_ings.html").write_text(html)
    print(f"\nReport written to reports/exp4_ings.html")
    print(f"Total runtime: {runtime / 60:.1f} min")
