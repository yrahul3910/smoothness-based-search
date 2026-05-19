"""Experiment 2: Smoothness as a Predictor of Generalization.

For each SE dataset, samples many random configurations, trains each, and
computes the Spearman correlation between β-smoothness (training) and R² (test).

A positive correlation validates that smoothness is a good proxy for selecting
which configurations to evaluate — the core assumption behind SMOOTHIE-Trees.

Results saved to reports/exp2_smoothness_correlation.html.
"""

import itertools
import json
import time
from pathlib import Path

import polars as pl
from raise_utils.data import Data
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from src.data import load_data
from src.matplotlib import create_surface_data
from src.smoothness_trees import get_tree_smoothness
from src.util import get_random_hyperparams

# ── Config ───────────────────────────────────────────────────────────────────

DIRS = ["./data/optimize/process/"]
N_SAMPLES = 150   # configs to sample per dataset
RANDOM_STATE = 42

HPO_SPACE = {
    "dt": {
        "criterion": ["friedman_mse", "absolute_error", "squared_error"],
        "max_depth": list(range(2, 12)),
        "min_samples_split": [2, 4, 8, 16],
        "max_features": ["sqrt", "log2", None],
    },
    "rf": {
        "n_estimators": [10, 25, 50],
        "criterion": ["friedman_mse", "absolute_error", "squared_error"],
        "max_depth": list(range(2, 12)),
        "min_samples_split": [2, 4, 8, 16],
        "max_features": ["sqrt", "log2", None],
    },
}

LEARNERS = {
    "dt": DecisionTreeRegressor,
    "rf": RandomForestRegressor,
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all() -> dict:
    import random
    files = list(itertools.chain.from_iterable(Path(d).rglob("*.csv") for d in DIRS))
    all_results = {}

    for fpath in tqdm(files, desc="datasets"):
        dataset_name = fpath.stem
        df = load_data(str(fpath))
        df = df.select((pl.all() - pl.all().min()) / (pl.all().max() - pl.all().min()))
        df = df.select([c for c in df.columns if not df[c].is_nan().any()])
        x, y = create_surface_data(df, pca=False)
        data = Data(*train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE))

        betas, r2s = [], []
        for _ in tqdm(range(N_SAMPLES), desc=dataset_name, leave=False):
            name = random.choice(list(HPO_SPACE.keys()))
            cfg = get_random_hyperparams(HPO_SPACE[name])
            model = LEARNERS[name](**cfg)
            model.fit(data.x_train, data.y_train)
            beta = get_tree_smoothness(model)
            preds = model.predict(data.x_test)
            r2 = r2_score(data.y_test, preds)
            betas.append(beta)
            r2s.append(r2)

        rho, p = spearmanr(betas, r2s)
        all_results[dataset_name] = {
            "betas": betas,
            "r2s": r2s,
            "spearman_rho": float(rho),
            "p_value": float(p),
        }
        print(f"  {dataset_name:20s}  ρ={rho:.3f}  p={p:.4f}")

    return all_results


# ── Report ────────────────────────────────────────────────────────────────────

def generate_report(results: dict, runtime: float) -> str:
    rows = []
    pos_sig = sum(1 for d in results.values() if d["spearman_rho"] > 0 and d["p_value"] < 0.05)
    neg_sig = sum(1 for d in results.values() if d["spearman_rho"] < 0 and d["p_value"] < 0.05)
    nonsig  = len(results) - pos_sig - neg_sig

    for name, d in sorted(results.items()):
        rho = d["spearman_rho"]
        p   = d["p_value"]
        if p < 0.05 and rho > 0:
            color = "#d4edda"
            interp = "↑ positive, significant"
        elif p < 0.05 and rho < 0:
            color = "#f8d7da"
            interp = "↓ negative, significant"
        else:
            color = "#fff3cd"
            interp = "— not significant"
        rows.append(f"""
        <tr style="background:{color}">
          <td><b>{name}</b></td>
          <td>{rho:.3f}</td>
          <td>{p:.4f}</td>
          <td>{interp}</td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment 2: Smoothness–R² Correlation</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 960px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }}
  h1, h2, h3 {{ color: #2c3e50; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #2c3e50; color: white; }}
  code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
  .math {{ font-style: italic; }}
  .box {{ background: #eef6fb; border-left: 4px solid #3498db; padding: 12px 16px; margin: 16px 0; }}
  .result-box {{ background: #eafaf1; border-left: 4px solid #27ae60; padding: 12px 16px; margin: 16px 0; }}
</style>
</head>
<body>

<h1>Experiment 2: Does β-Smoothness Predict Generalization for SE Trees?</h1>
<p><em>Date: 2026-05-19 &nbsp;|&nbsp; Runtime: {runtime/60:.1f} min</em></p>

<h2>1. Motivation</h2>
<p>
SMOOTHIE-Trees (Experiment 1) selects configurations by maximizing β-smoothness.
For this to be a sound strategy, smoothness computed on <em>training data</em>
must correlate with test-set R². This experiment directly tests that assumption
via Spearman rank correlation across a large sample of random configurations.
</p>

<h2>2. Method</h2>
<p>
For each dataset, we sample <strong>{N_SAMPLES}</strong> random configurations
(Decision Tree or Random Forest, chosen uniformly), train each on the training split,
and record:
</p>
<ul>
  <li><span class="math">β</span>: tree β-smoothness (node-level NLL center difference)</li>
  <li><span class="math">R²</span>: coefficient of determination on the held-out test set</li>
</ul>
<p>
We compute the Spearman rank correlation ρ between β and R²
(Spearman is used because both may be non-normal). A significant positive ρ
means high-smoothness configurations tend to generalize well &mdash; validating
the smoothness heuristic.
</p>

<h2>3. Results</h2>
<table>
  <tr>
    <th>Dataset</th>
    <th>Spearman ρ</th>
    <th>p-value</th>
    <th>Interpretation</th>
  </tr>
  {''.join(rows)}
</table>

<div class="result-box">
<strong>Summary across {len(results)} datasets:</strong>
{pos_sig} positive significant, {nonsig} not significant, {neg_sig} negative significant.
</div>

<h2>4. Discussion</h2>
<p>
A positive Spearman ρ indicates that configurations producing higher β-smoothness
during training also tend to achieve higher R² on the test set. This would directly
validate the selection rule used in SMOOTHIE-Trees.
</p>
<p>
If ρ is not significant or negative on some datasets, it suggests that β-smoothness
alone may be insufficient as a selection criterion for those datasets.
This could occur when the dataset is very small (noisy correlation estimate),
or when the relationship between smoothness and generalization is non-monotone.
</p>
<p>
Taken together with Experiment 1, these results characterize both <em>how often</em>
smoothness-based selection improves HPO and <em>why</em> it succeeds or fails.
</p>

</body>
</html>"""


if __name__ == "__main__":
    Path("reports").mkdir(exist_ok=True)
    t0 = time.time()
    results = run_all()
    runtime = time.time() - t0

    with open("reports/exp2_raw.json", "w") as f:
        safe = {k: {kk: (vv if not isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in results.items()}
        json.dump(safe, f, indent=2)

    html = generate_report(results, runtime)
    Path("reports/exp2_smoothness_correlation.html").write_text(html)
    print(f"\nReport written to reports/exp2_smoothness_correlation.html")
    print(f"Total runtime: {runtime/60:.1f} min")
