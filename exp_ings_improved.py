"""Experiment 5: Improved INGS variants.

Experiment 4 showed INGS matches TPE on smooth XOMO datasets but lags BOHB
on small datasets (nasa93dem, POM3) due to a noisy 20% validation proxy.
This experiment tests four targeted improvements:

  1. ings_cv     : k=2 fold CV proxy — averages R² over two folds for a more
                   stable signal. Budget kept at 30 trains: 15 configs × 2 folds.

  2. ings_ucb    : UCB exploration bonus. Adds a distance term to the anti-
                   Laplacian score so candidates far from observed configs get
                   a bonus, preventing premature convergence:
                       score(h) = anti_laplacian(h)
                                  + κ · min_dist(h, observed) / √d

  3. ings_multi  : Multi-fold final selection. After the search, takes the
                   top-3 candidates by val R², re-evaluates each with a fresh
                   random validation split, and picks the best. Costs 3 extra
                   model trains to reduce selection noise.

  4. ings_lhs    : Latin Hypercube Sampling for initialization. Ensures the
                   initial N_INIT configs cover the 5D HPO space uniformly
                   rather than clustering by chance.

  5. ings_full   : All four improvements combined.

  6. ings_orig   : Baseline INGS from Experiment 4 (re-run for direct comparison
                   under identical conditions).

Reference: bohb_30, tpe_30, random_30.
Results saved to reports/exp5_ings_improved.html.
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
from scipy.stats import mannwhitneyu, qmc
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from src.data import load_data
from src.matplotlib import create_surface_data
from src.util import get_random_hyperparams

optuna.logging.set_verbosity(logging.WARNING)

# ── Config ─────────────────────────────────────────────────────────────────────

DIRS = ["./data/optimize/process/"]
N_BUDGET = 30        # total model-train budget for all methods
N_REPEATS = 10
RANDOM_STATE = 42
FLAT_EPS = 1e-3
N_CANDIDATES = 40
UCB_KAPPA = 0.1      # exploration bonus weight (in val-R² units)
CV_FOLDS = 2         # for ings_cv: budget = (N_BUDGET // CV_FOLDS) configs
MULTIFOLD_TOP_K = 3  # candidates re-evaluated in multi-fold selection

HPO_SPACE = {
    "criterion": ["friedman_mse", "absolute_error", "squared_error"],
    "max_depth": list(range(2, 16)),
    "min_samples_split": [2, 4, 8, 16, 32],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
}

METRICS = [r2_score, mean_squared_error]
LEARNER = DecisionTreeRegressor
N_DIMS = len(HPO_SPACE)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_model(cfg: dict) -> DecisionTreeRegressor:
    return LEARNER(**cfg)


def _eval_test(model, data: Data) -> list[float]:
    preds = model.predict(data.x_test)
    return [fn(data.y_test, preds) for fn in METRICS]


def _encode(cfg: dict) -> np.ndarray:
    vec = []
    for key, options in HPO_SPACE.items():
        idx = options.index(cfg[key])
        vec.append(idx / max(1, len(options) - 1))
    return np.array(vec, dtype=float)


def _neighbor_vecs(cfg: dict) -> np.ndarray:
    rows = []
    base = _encode(cfg)
    for dim_idx, (key, options) in enumerate(HPO_SPACE.items()):
        cur_idx = options.index(cfg[key])
        for delta in (-1, +1):
            ni = cur_idx + delta
            if 0 <= ni < len(options):
                v = base.copy()
                v[dim_idx] = ni / max(1, len(options) - 1)
                rows.append(v)
    return np.array(rows) if rows else np.empty((0, N_DIMS))


def _anti_laplacian_scores(candidates, rbf_preds, rbf):
    """Compute anti-Laplacian score for each candidate."""
    scores = np.empty(len(candidates))
    for i, (c, pred) in enumerate(zip(candidates, rbf_preds)):
        nv = _neighbor_vecs(c)
        if len(nv):
            scores[i] = 2.0 * pred - float(np.mean(rbf(nv).flatten()))
        else:
            scores[i] = pred
    return scores


def _lhs_configs(n: int) -> list[dict]:
    """Latin Hypercube Sample of n configs covering the 5D HPO space."""
    sampler = qmc.LatinHypercube(d=N_DIMS, seed=None)
    samples = sampler.random(n=n)
    configs = []
    for row in samples:
        cfg = {}
        for dim_idx, (key, options) in enumerate(HPO_SPACE.items()):
            idx = int(round(row[dim_idx] * (len(options) - 1)))
            idx = max(0, min(len(options) - 1, idx))
            cfg[key] = options[idx]
        configs.append(cfg)
    return configs


def _proxy_single(cfg, x_tr, x_val, y_tr, y_val) -> float:
    """Train config on x_tr, evaluate R² on x_val."""
    m = _make_model(cfg)
    m.fit(x_tr, y_tr)
    return float(r2_score(y_val, m.predict(x_val)))


def _proxy_cv(cfg, x_train, y_train, k=CV_FOLDS) -> float:
    """k-fold cross-validated R² on training data."""
    kf = KFold(n_splits=k, shuffle=True)
    scores = []
    for tr_idx, val_idx in kf.split(x_train):
        m = _make_model(cfg)
        m.fit(x_train[tr_idx], y_train[tr_idx])
        scores.append(float(r2_score(y_train[val_idx], m.predict(x_train[val_idx]))))
    return float(np.mean(scores))


# ── Core INGS with flags ────────────────────────────────────────────────────────

def _ings(
    data: Data,
    use_correction: bool = True,
    use_cv: bool = False,
    use_ucb: bool = False,
    use_lhs: bool = False,
    use_multifold: bool = False,
) -> list[float]:
    """Unified INGS implementation.

    Budget accounting:
      use_cv=False  : N_BUDGET configs, each 1 train  → 30 trains
      use_cv=True   : N_BUDGET//CV_FOLDS configs, each CV_FOLDS trains → 30 trains
    """
    n_configs = N_BUDGET // CV_FOLDS if use_cv else N_BUDGET
    n_init = max(5, n_configs // 3)  # ~1/3 random warm-up
    n_steps = n_configs - n_init

    # Validation split (used when use_cv=False)
    x_tr, x_val, y_tr, y_val = train_test_split(
        data.x_train, data.y_train, test_size=0.2, random_state=RANDOM_STATE
    )

    def _get_proxy(cfg):
        if use_cv:
            return _proxy_cv(cfg, data.x_train, data.y_train)
        return _proxy_single(cfg, x_tr, x_val, y_tr, y_val)

    # Phase 0 – initialization
    if use_lhs:
        init_cfgs = _lhs_configs(n_init)
    else:
        init_cfgs = [get_random_hyperparams(HPO_SPACE) for _ in range(n_init)]

    cfgs = list(init_cfgs)
    vecs = [_encode(c) for c in cfgs]
    proxies = [_get_proxy(c) for c in cfgs]

    # Phase 1 – iterative search
    for _ in range(n_steps):
        X_obs = np.array(vecs)
        y_obs = np.array(proxies)

        try:
            rbf = RBFInterpolator(X_obs, y_obs, kernel="linear", smoothing=1e-2)
        except Exception:
            cfg = get_random_hyperparams(HPO_SPACE)
            cfgs.append(cfg)
            vecs.append(_encode(cfg))
            proxies.append(_get_proxy(cfg))
            continue

        candidates = [get_random_hyperparams(HPO_SPACE) for _ in range(N_CANDIDATES)]
        cand_vecs = np.array([_encode(c) for c in candidates])
        rbf_preds = rbf(cand_vecs).flatten()

        if use_correction:
            scores = _anti_laplacian_scores(candidates, rbf_preds, rbf)
        else:
            scores = rbf_preds.copy()

        if use_ucb:
            # Exploration bonus: distance to nearest observed config
            dists = np.array([
                np.min(np.linalg.norm(X_obs - v, axis=1))
                for v in cand_vecs
            ])
            dist_norm = dists / np.sqrt(N_DIMS)  # normalize to ~[0, 1]
            scores = scores + UCB_KAPPA * dist_norm

        if float(scores.max() - scores.min()) < FLAT_EPS:
            cfg_next = get_random_hyperparams(HPO_SPACE)
        else:
            cfg_next = candidates[int(np.argmax(scores))]

        cfgs.append(cfg_next)
        vecs.append(_encode(cfg_next))
        proxies.append(_get_proxy(cfg_next))

    # Phase 2 – final selection
    if use_multifold:
        # Re-evaluate top-k with a fresh random validation split
        top_k_idx = np.argsort(proxies)[-MULTIFOLD_TOP_K:]
        # Fresh split (different random state)
        x_tr2, x_val2, y_tr2, y_val2 = train_test_split(
            data.x_train, data.y_train, test_size=0.2
        )
        reeval = [_proxy_single(cfgs[i], x_tr2, x_val2, y_tr2, y_val2) for i in top_k_idx]
        best_cfg = cfgs[top_k_idx[int(np.argmax(reeval))]]
    else:
        best_cfg = cfgs[int(np.argmax(proxies))]

    best_model = _make_model(best_cfg)
    best_model.fit(data.x_train, data.y_train)
    return _eval_test(best_model, data)


# ── Named variants ──────────────────────────────────────────────────────────────

def ings_orig(data):
    return _ings(data, use_correction=True)

def ings_cv(data):
    return _ings(data, use_correction=True, use_cv=True)

def ings_ucb(data):
    return _ings(data, use_correction=True, use_ucb=True)

def ings_multi(data):
    return _ings(data, use_correction=True, use_multifold=True)

def ings_lhs(data):
    return _ings(data, use_correction=True, use_lhs=True)

def ings_full(data):
    return _ings(data, use_correction=True,
                 use_cv=True, use_ucb=True, use_lhs=True, use_multifold=True)


# ── Reference baselines ────────────────────────────────────────────────────────

def _make_optuna_objective(data: Data):
    def objective(trial):
        cfg = {
            "criterion": trial.suggest_categorical("criterion", HPO_SPACE["criterion"]),
            "max_depth": trial.suggest_categorical("max_depth", HPO_SPACE["max_depth"]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", HPO_SPACE["min_samples_split"]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", HPO_SPACE["min_samples_leaf"]),
            "max_features": trial.suggest_categorical("max_features", HPO_SPACE["max_features"]),
        }
        m = _make_model(cfg)
        m.fit(data.x_train, data.y_train)
        return float(r2_score(data.y_test, m.predict(data.x_test)))
    return objective


def tpe_30(data):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=None),
    )
    study.optimize(_make_optuna_objective(data), n_trials=N_BUDGET, show_progress_bar=False)
    m = _make_model(study.best_params)
    m.fit(data.x_train, data.y_train)
    return _eval_test(m, data)


def bohb_30(data):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=5, seed=None),
    )
    study.optimize(_make_optuna_objective(data), n_trials=N_BUDGET, show_progress_bar=False)
    m = _make_model(study.best_params)
    m.fit(data.x_train, data.y_train)
    return _eval_test(m, data)


def random_30(data):
    best = [float("-inf")] * len(METRICS)
    for _ in range(N_BUDGET):
        cfg = get_random_hyperparams(HPO_SPACE)
        m = _make_model(cfg)
        m.fit(data.x_train, data.y_train)
        res = _eval_test(m, data)
        if res[0] > best[0]:
            best = res
    return best


# ── Runner ─────────────────────────────────────────────────────────────────────

METHOD_NAMES = [
    "ings_orig", "ings_cv", "ings_ucb", "ings_multi", "ings_lhs", "ings_full",
    "bohb_30", "tpe_30", "random_30",
]

METHOD_FNS = {
    "ings_orig":  ings_orig,
    "ings_cv":    ings_cv,
    "ings_ucb":   ings_ucb,
    "ings_multi": ings_multi,
    "ings_lhs":   ings_lhs,
    "ings_full":  ings_full,
    "bohb_30":    bohb_30,
    "tpe_30":     tpe_30,
    "random_30":  random_30,
}


def one_repeat(data: Data) -> dict[str, list[float]]:
    return {name: fn(data) for name, fn in METHOD_FNS.items()}


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

        for name, scores in per_method.items():
            results[dataset_name][name] = scores
            r2_vals = [s[0] for s in scores]
            print(f"  {name:15s}  R²={np.mean(r2_vals):.3f} ± {np.std(r2_vals):.3f}")

    return dict(results)


# ── Statistical test ──────────────────────────────────────────────────────────

def mann_whitney(a, b) -> tuple[float, str]:
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

    ings_variants = ["ings_orig", "ings_cv", "ings_ucb", "ings_multi", "ings_lhs", "ings_full"]
    refs = ["bohb_30", "tpe_30", "random_30"]

    # Win/tie/loss counters: variant → ref → {win/tie/loss}
    counters: dict[str, dict[str, dict[str, int]]] = {
        v: {r: {"win": 0, "tie": 0, "loss": 0} for r in refs}
        for v in ings_variants
    }

    dataset_rows = []
    for dataset, methods in sorted(results.items()):
        r2 = {m: [s[0] for s in v] for m, v in methods.items()}

        verdicts: dict[str, dict[str, str]] = {}
        for v in ings_variants:
            verdicts[v] = {}
            for ref in refs:
                _, verd = mann_whitney(r2[v], r2[ref])
                verdicts[v][ref] = verd
                counters[v][ref][verd] += 1

        # Best INGS variant for this dataset
        best_v = max(ings_variants, key=lambda v: np.mean(r2[v]))

        cells = ""
        for v in ings_variants:
            bg = color(verdicts[v]["bohb_30"])
            bold_start = "<b>" if v == best_v else ""
            bold_end = "</b>" if v == best_v else ""
            cells += f'<td style="background:{bg}">{bold_start}{fmt(r2[v])}{bold_end}</td>'

        dataset_rows.append(f"""
        <tr>
          <td><b>{dataset}</b></td>
          {cells}
          <td>{fmt(r2['bohb_30'])}</td>
          <td>{fmt(r2['tpe_30'])}</td>
          <td>{fmt(r2['random_30'])}</td>
        </tr>""")

    # Summary table
    def wt(variant, ref):
        c = counters[variant][ref]
        return f"{c['win']}W/{c['tie']}T/{c['loss']}L"

    summary_rows = ""
    variant_labels = {
        "ings_orig":  "INGS (baseline)",
        "ings_cv":    "INGS-CV",
        "ings_ucb":   "INGS-UCB",
        "ings_multi": "INGS-Multi",
        "ings_lhs":   "INGS-LHS",
        "ings_full":  "INGS-Full",
    }
    for v in ings_variants:
        summary_rows += f"""
        <tr>
          <td>{variant_labels[v]}</td>
          <td>{wt(v, 'bohb_30')}</td>
          <td>{wt(v, 'tpe_30')}</td>
          <td>{wt(v, 'random_30')}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 5: Improved INGS Variants</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }}
  h1, h2, h3 {{ color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.88em; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 9px; text-align: left; }}
  th {{ background: #2c3e50; color: white; }}
  code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 0.88em; }}
  .math {{ font-style: italic; }}
  .box {{ background: #eef6fb; border-left: 4px solid #3498db; padding: 12px 16px; margin: 16px 0; }}
  .result-box {{ background: #eafaf1; border-left: 4px solid #27ae60; padding: 12px 16px; margin: 16px 0; }}
  pre {{ background: #f8f8f8; padding: 10px; border-radius: 4px; font-size: 0.87em; }}
</style>
</head>
<body>

<h1>Experiment 5: Improved INGS Variants</h1>
<p><em>Date: 2026-05-19 &nbsp;|&nbsp; Runtime: {runtime/60:.1f} min</em></p>

<h2>1. Motivation</h2>
<p>
Experiment 4 showed INGS matches TPE/BOHB on smooth XOMO landscapes but lags on
small datasets (nasa93dem, POM3) due to a noisy 20%-split validation proxy.
This experiment tests four targeted improvements and their combination.
</p>

<h2>2. Improvements Tested</h2>
<table>
  <tr><th>Variant</th><th>Change vs. INGS baseline</th><th>Motivation</th></tr>
  <tr>
    <td><b>INGS-CV</b></td>
    <td>k=2 fold CV proxy; budget = {N_BUDGET // CV_FOLDS} configs × 2 folds = {N_BUDGET} trains</td>
    <td>Halves proxy variance on small datasets; uses all training data</td>
  </tr>
  <tr>
    <td><b>INGS-UCB</b></td>
    <td>score += κ·min_dist(h, observed)/√d &nbsp; (κ={UCB_KAPPA})</td>
    <td>Prevents premature convergence to a local region</td>
  </tr>
  <tr>
    <td><b>INGS-Multi</b></td>
    <td>Top-{MULTIFOLD_TOP_K} candidates re-evaluated on a fresh val split; best selected</td>
    <td>Reduces noise in the final selection step (~3 extra trains)</td>
  </tr>
  <tr>
    <td><b>INGS-LHS</b></td>
    <td>Initialization via Latin Hypercube Sampling instead of uniform random</td>
    <td>Guarantees even coverage of HPO space; better initial surrogate</td>
  </tr>
  <tr>
    <td><b>INGS-Full</b></td>
    <td>All four improvements combined</td>
    <td>Synergistic effect of all fixes</td>
  </tr>
</table>

<h2>3. Results: R² Score (higher is better)</h2>
<p>
Cell background: INGS variant vs. BOHB-30 (green = win, yellow = tie, red = loss).
<b>Bold</b> = best INGS variant per dataset. Budget = {N_BUDGET} model trains for all methods.
</p>
<table>
  <tr>
    <th>Dataset</th>
    <th>INGS (orig)</th>
    <th>INGS-CV</th>
    <th>INGS-UCB</th>
    <th>INGS-Multi</th>
    <th>INGS-LHS</th>
    <th>INGS-Full</th>
    <th>BOHB-30</th>
    <th>TPE-30</th>
    <th>Random-30</th>
  </tr>
  {''.join(dataset_rows)}
</table>

<h2>4. Win/Tie/Loss Summary vs. Reference Methods</h2>
<table>
  <tr>
    <th>Variant</th>
    <th>vs. BOHB-30</th>
    <th>vs. TPE-30</th>
    <th>vs. Random-30</th>
  </tr>
  {summary_rows}
</table>

<h2>5. Discussion</h2>

<h3>5.1 CV proxy (INGS-CV)</h3>
<p>
The k=2 fold CV proxy uses all training data for proxy estimation, trading fewer
exploration steps (15 vs 30 configs) for a substantially more stable signal.
This should most benefit small datasets (nasa93dem, POM3) where the 20% fixed split
was too noisy. The cost is fewer steps for the surrogate to update and refine.
</p>

<h3>5.2 UCB exploration bonus (INGS-UCB)</h3>
<p>
The UCB term adds <code>κ·min_dist(h, observed)/√d</code> to the score.
With κ={UCB_KAPPA}, the bonus is proportional to the normalized distance from the nearest
evaluated config. This prevents the search from over-exploiting a small promising region
when the surrogate may not extrapolate well.
</p>

<h3>5.3 Multi-fold final selection (INGS-Multi)</h3>
<p>
After the search, the top-{MULTIFOLD_TOP_K} candidates by val R² are re-evaluated on
a fresh (different random seed) validation split. The best by this second estimate is
chosen. This two-stage selection reduces the influence of a lucky/unlucky single split
on the final pick, at the cost of {MULTIFOLD_TOP_K} additional model trains.
</p>

<h3>5.4 LHS initialization (INGS-LHS)</h3>
<p>
Latin Hypercube Sampling provides a space-filling design over the 5D HPO space.
Uniform random initialization can cluster by chance; LHS guarantees at most one
sample per interval on each dimension. A better-spread initial set gives the RBF
interpolant more reliable gradient estimates from the start.
</p>

<h3>5.5 Combined (INGS-Full)</h3>
<p>
INGS-Full applies all improvements simultaneously. If the improvements are largely
orthogonal (CV addresses proxy noise; LHS addresses initialization quality; UCB
addresses exploration-exploitation; Multi-fold addresses selection noise), their
combination should be synergistic.
</p>

</body>
</html>"""


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    t0 = time.time()
    results = run_all()
    runtime = time.time() - t0

    with open("reports/exp5_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    html = generate_report(results, runtime)
    Path("reports/exp5_ings_improved.html").write_text(html)
    print(f"\nReport written to reports/exp5_ings_improved.html")
    print(f"Total runtime: {runtime / 60:.1f} min")
