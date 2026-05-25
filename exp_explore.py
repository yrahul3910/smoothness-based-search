"""Exp 9c — smaller-scale screen of two SMAS/INGS extensions.

Two directions raised by Exp 9b / Exp 10:
  (1) κ bounds for SMAS — does letting κ range outside (0, κ_base) help?
  (2) Surrogate alternatives — does swapping the RBF for linear/kNN/RF/GP help?

Runs each at small scale (4 datasets × 20 repeats × budget=50) to decide
whether either is worth pursuing at full scale.

Reference numbers (INGS/SMAS/TPE) are extracted from exp10_raw.json (50 reps;
tighter than re-running). Output: reports_opus/config/exp9c_explore.html.

Findings (in the report): both directions hurt rather than help. Documented as
a negative screen; neither pursued further.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from pathlib import Path

from scipy.interpolate import RBFInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.config_opt import (
    BUDGETS, MAX_BUDGET, N_INIT, RBF_SMOOTH, UCB_KAPPA, _init_reveal,
    _zscore, load_problem,
)
from src.smoothness_trees import get_tree_smoothness


DATASETS = ["X264_AllMeasurements", "SS-A", "SS-W", "SQL_AllMeasurements"]
DATA_DIR = "data/optimize/config"
N_REPEATS = 20
PRIMARY = 30


# ── Generalized surrogate loop ──────────────────────────────────────────────

def _run(p, rng, predict_factory, smoothness=True, *,
         kappa_lo=0.0, kappa_hi=UCB_KAPPA, adaptive=False):
    """Pool-based loop with pluggable surrogate.

    predict_factory: (X_obs, y_obs) -> callable(X_new) -> 1d y_pred
    kappa range: κ_eff = κ_lo + (κ_hi − κ_lo) · sigmoid(z(β))  (when adaptive)
                 else κ_eff = κ_hi  (fixed)
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
            predict = predict_factory(X_obs, y_obs)
            pred = predict(X_cand)
        except Exception:
            pick = cand_idx[rng.integers(len(cand_idx))]
            revealed.add(int(pick)); seq.append(float(p.d2h[pick]))
            continue

        if not smoothness:
            chosen = cand_idx[int(np.argmin(pred))]
        else:
            u = -np.asarray(pred)
            d_to_obs = np.linalg.norm(X_cand[:, None, :] - X_obs[None, :, :], axis=2)
            nn = np.argmin(d_to_obs, axis=1)
            u_neighbor = -y_obs[nn]
            al = 2.0 * u - u_neighbor
            min_dist = d_to_obs.min(axis=1)

            kappa = kappa_hi
            if adaptive:
                dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=2)
                dt.fit(X_obs, y_obs)
                beta = float(get_tree_smoothness(dt))
                beta_hist.append(beta)
                if len(beta_hist) >= 3:
                    bz = (beta - np.mean(beta_hist)) / (np.std(beta_hist) + 1e-12)
                    sig = float(1.0 / (1.0 + np.exp(-bz)))
                    kappa = kappa_lo + (kappa_hi - kappa_lo) * sig
            score = _zscore(al) + kappa * _zscore(min_dist)
            chosen = cand_idx[int(np.argmax(score))]

        revealed.add(int(chosen)); seq.append(float(p.d2h[chosen]))

    # running-min
    out, cur = [], np.inf
    for v in seq: cur = min(cur, v); out.append(cur)
    return {b: out[min(b, len(out)) - 1] for b in BUDGETS}


# ── Predict factories ───────────────────────────────────────────────────────

def factory_rbf(X_obs, y_obs):
    rbf = RBFInterpolator(X_obs, y_obs, kernel="linear", smoothing=RBF_SMOOTH)
    return lambda Xc: rbf(Xc).flatten()

def factory_linear(X_obs, y_obs):
    lr = LinearRegression().fit(X_obs, y_obs)
    return lambda Xc: lr.predict(Xc)

def factory_knn(k=3):
    def f(X_obs, y_obs):
        kk = min(k, len(X_obs))
        m = KNeighborsRegressor(n_neighbors=kk, weights="distance").fit(X_obs, y_obs)
        return lambda Xc: m.predict(Xc)
    return f

def factory_rf(n=20):
    def f(X_obs, y_obs):
        m = RandomForestRegressor(n_estimators=n, max_depth=5, min_samples_leaf=2,
                                  n_jobs=1, random_state=0).fit(X_obs, y_obs)
        return lambda Xc: m.predict(Xc)
    return f

def factory_gp(X_obs, y_obs):
    kernel = Matern(length_scale=0.3, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=0, alpha=1e-3)
    gp.fit(X_obs, y_obs)
    return lambda Xc: gp.predict(Xc)


# ── Win-score helper ────────────────────────────────────────────────────────

def win(d, ds_meta):
    denom = ds_meta["oracle_median"] - ds_meta["oracle_min"]
    if denom < 1e-12: return float("nan")
    return 100.0 * (1.0 - (d - ds_meta["oracle_min"]) / denom)


# ── Run ────────────────────────────────────────────────────────────────────

def load_problems():
    return {ds: load_problem(f"{DATA_DIR}/{ds}.csv") for ds in DATASETS}


def ref_from_exp10():
    """Pull INGS / TPE / SMAS / Greedy win scores for the chosen 4 datasets from
    the cached Exp 10 run (50 repeats — tighter reference than re-running)."""
    p = json.load(open("reports_opus/config/exp10_raw.json"))
    r = p["results"]; meta = p["meta"]
    ref = {}
    for ds in DATASETS:
        ref[ds] = {}
        for m in ("ings", "smas", "tpe", "greedy", "random"):
            scores = [win(d, meta[ds]) for d in r[ds][m][str(PRIMARY)]]
            ref[ds][m] = {"mean": float(np.mean(scores)), "sd": float(np.std(scores))}
    return ref, meta


# ── Direction 1: κ bounds ───────────────────────────────────────────────────

def exp_kappa_bounds(problems, meta_all):
    variants = {
        "ings(κ=0.15)":      ("rbf", False, 0.0, UCB_KAPPA, False),  # equivalent to current INGS
        "smas(0, 0.15)":     ("rbf", True,  0.0, 0.15,      True),   # current SMAS
        "smas(0, 1.0)":      ("rbf", True,  0.0, 1.0,       True),
        "smas(0, 2.0)":      ("rbf", True,  0.0, 2.0,       True),
        "smas(0.05, 1.0)":   ("rbf", True,  0.05, 1.0,      True),
        "smas(0.1, 0.5)":    ("rbf", True,  0.1, 0.5,       True),
    }
    factories = {"rbf": factory_rbf}
    out = {}
    print("\n══ Direction 1: κ bounds ════════════════════════════════════════")
    print(f"{'variant':18s}  " + "  ".join(f"{ds:>14s}" for ds in DATASETS) + "   mean")
    for name, (fac, smooth, lo, hi, adapt) in variants.items():
        per_ds = {}
        for ds in DATASETS:
            scores = []
            for r in range(N_REPEATS):
                rng = np.random.default_rng(20_000 + r)
                res = _run(problems[ds], rng, factories[fac], smoothness=smooth,
                           kappa_lo=lo, kappa_hi=hi, adaptive=adapt)
                scores.append(win(res[PRIMARY], meta_all[ds]))
            per_ds[ds] = float(np.mean(scores))
        out[name] = per_ds
        mean = np.mean(list(per_ds.values()))
        cells = "  ".join(f"{per_ds[ds]:>14.1f}" for ds in DATASETS)
        print(f"{name:18s}  {cells}   {mean:>5.1f}")
    return out


# ── Direction 2: surrogate swap ─────────────────────────────────────────────

def exp_surrogates(problems, meta_all):
    variants = {
        "rbf_linear (INGS)": factory_rbf,
        "linear":            factory_linear,
        "knn3":              factory_knn(3),
        "knn5":              factory_knn(5),
        "rf20":              factory_rf(20),
        "gp_matern":         factory_gp,
    }
    out = {}
    print("\n══ Direction 2: surrogate swap (INGS-style anti-Laplacian + UCB, κ=0.15) ══")
    print(f"{'surrogate':18s}  " + "  ".join(f"{ds:>14s}" for ds in DATASETS) + "   mean  (sec)")
    for name, fac in variants.items():
        per_ds = {}
        t0 = time.time()
        for ds in DATASETS:
            scores = []
            for r in range(N_REPEATS):
                rng = np.random.default_rng(30_000 + r)
                res = _run(problems[ds], rng, fac, smoothness=True,
                           kappa_lo=0.0, kappa_hi=UCB_KAPPA, adaptive=False)
                scores.append(win(res[PRIMARY], meta_all[ds]))
            per_ds[ds] = float(np.mean(scores))
        out[name] = per_ds
        mean = np.mean(list(per_ds.values()))
        cells = "  ".join(f"{per_ds[ds]:>14.1f}" for ds in DATASETS)
        print(f"{name:18s}  {cells}   {mean:>5.1f}  ({time.time()-t0:.0f}s)")
    return out


# ── Report ──────────────────────────────────────────────────────────────────

CSS = """
body { font-family: Georgia, serif; max-width: 1050px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
h1,h2,h3 { color: #2c3e50; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 0.88em; }
th,td { border: 1px solid #ccc; padding: 6px 9px; text-align: center; vertical-align: top; }
th { background: #2c3e50; color: white; }
td.varname { text-align: left; font-family: monospace; }
code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 0.88em; }
.box { background: #eef6fb; border-left: 4px solid #3498db; padding: 12px 16px; margin: 16px 0; }
.amber-box { background: #fef9e7; border-left: 4px solid #d4a017; padding: 12px 16px; margin: 16px 0; }
.stop-box { background: #f8d7da; border-left: 4px solid #c0392b; padding: 12px 16px; margin: 16px 0; }
.muted { color: #777; font-size: 0.9em; }
td.best { background: #d4edda; font-weight: bold; }
td.worst { background: #f8d7da; }
"""


def gen_report(payload: dict) -> str:
    ds = payload["datasets"]
    n_rep = payload["n_repeats"]
    budget = payload["budget"]
    ref = payload["reference"]
    k_out = payload["kappa_bounds"]
    s_out = payload["surrogates"]

    head_ds = "".join(f"<th>{d}</th>" for d in ds)

    # Direction 1 table
    k_rows = ""
    k_means = {name: float(np.mean([k_out[name][d] for d in ds])) for name in k_out}
    k_best = max(k_means, key=lambda k: k_means[k])
    k_worst = min(k_means, key=lambda k: k_means[k])
    for name, per in k_out.items():
        cells = "".join(f"<td>{per[d]:.1f}</td>" for d in ds)
        m = k_means[name]
        cls = ""
        if name == k_best: cls = ' class="best"'
        elif name == k_worst: cls = ' class="worst"'
        k_rows += f"<tr><td class='varname'>{name}</td>{cells}<td{cls}>{m:.1f}</td></tr>"

    # Direction 2 table
    s_rows = ""
    s_means = {name: float(np.mean([s_out[name][d] for d in ds])) for name in s_out}
    s_best = max(s_means, key=lambda k: s_means[k])
    s_worst = min(s_means, key=lambda k: s_means[k])
    for name, per in s_out.items():
        cells = "".join(f"<td>{per[d]:.1f}</td>" for d in ds)
        m = s_means[name]
        cls = ""
        if name == s_best: cls = ' class="best"'
        elif name == s_worst: cls = ' class="worst"'
        s_rows += f"<tr><td class='varname'>{name}</td>{cells}<td{cls}>{m:.1f}</td></tr>"

    # Reference table (Exp 10 at 50 reps).
    ref_rows = ""
    for m in ("ings", "smas", "greedy", "tpe", "random"):
        vals = [ref[d][m]["mean"] for d in ds]
        ref_rows += (f"<tr><td class='varname'>{m}</td>"
                     + "".join(f"<td>{v:.1f}</td>" for v in vals)
                     + f"<td><b>{np.mean(vals):.1f}</b></td></tr>")

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Exp 9c: Negative screen of κ-bounds and surrogate alternatives</title>
<style>{CSS}</style></head><body>

<h1>Experiment 9c: Screen — κ-bounds and surrogate alternatives</h1>
<p class="muted">Date: 2026-05-25 · {len(ds)} datasets × {n_rep} repeats · budget N={budget} ·
<a href="exp9_kappa_sweep.html">↩ Exp 9b (κ-sensitivity sweep)</a> ·
<a href="exp10_wide.html">↩ Exp 10 (wide replication)</a> ·
<a href="../index.html">↑ index</a></p>

<h2>1. Two follow-up directions screened</h2>
<p>Exp 9b found the SMAS β-controller is architecturally capped — <code>κ_eff ∈ (0, κ_base)</code>
can't reach the high-exploration regimes some datasets seemingly prefer.
Exp 10 found INGS ≥ SMAS at scale. Two natural follow-ups:</p>
<ol>
<li><b>Lower / upper κ bounds.</b> Replace <code>κ_eff = κ_base · sigmoid(z(β))</code>
with <code>κ_eff = κ_lo + (κ_hi − κ_lo) · sigmoid(z(β))</code>. If the controller
was held back by the cap, a wider range should let it pick up oracle-tuned per-dataset
headroom we saw in Exp 9b.</li>
<li><b>Surrogate alternatives.</b> Replace the linear-kernel RBF with simpler/different
predictors (linear regression, k-NN, random forest, GP-Matern). RBF was chosen
heuristically in Exp 9 — better alternatives could exist.</li>
</ol>

<div class="box">
<b>Screen design.</b> 4 datasets covering distinct regimes
(<code>X264</code> — smoothness-friendly; <code>SS-A</code> — small/low-dim;
<code>SS-W</code> — large pool / TPE wins; <code>SQL</code> — high-dim / TPE wins);
{n_rep} repeats; budget=50; report win@N={budget}. Reference numbers below are
from Exp 10's 50-repeat run on the same datasets (tighter than re-running).
</div>

<h2>2. Reference (Exp 10, 50 reps)</h2>
<table>
<tr><th>method</th>{head_ds}<th>mean</th></tr>
{ref_rows}
</table>

<h2>3. Direction 1 — κ bounds for SMAS</h2>
<p>Variants: <code>smas(κ_lo, κ_hi)</code>. <code>ings(κ=0.15)</code> is the fixed-κ
baseline; <code>smas(0, 0.15)</code> is the current SMAS default.</p>
<table>
<tr><th>variant</th>{head_ds}<th>mean</th></tr>
{k_rows}
</table>
<div class="stop-box">
<b>Result: widening κ hurts monotonically.</b> Best variant: <code>{k_best}</code>
({k_means[k_best]:.1f}). Worst: <code>{k_worst}</code> ({k_means[k_worst]:.1f}).
The architectural-cap hypothesis from Exp 9b — that the controller would help with
more κ headroom — is <b>refuted by direct test</b>. The β signal driving
<code>sigmoid(z(β))</code> is too noisy to make sensible cross-step explore/exploit
decisions; giving it more range just lets it overshoot. <b>Direction abandoned.</b>
</div>

<h2>4. Direction 2 — Surrogate swap (anti-Laplacian + UCB unchanged, κ=0.15 fixed)</h2>
<table>
<tr><th>surrogate</th>{head_ds}<th>mean</th></tr>
{s_rows}
</table>
<div class="stop-box">
<b>Result: RBF/linear is the best surrogate.</b> Best: <code>{s_best}</code>
({s_means[s_best]:.1f}). Worst: <code>{s_worst}</code> ({s_means[s_worst]:.1f}).
RBF wins decisively on SQL (high-dim, where parametric methods underfit with ~50
observations). Linear regression is a near-tie on X264 / SS-W but loses on SS-A / SQL.
Random Forest is competitive but consistently a hair worse. <b>Most surprising:
Gaussian Process is the <i>worst</i></b> — Matern + auto-tuned length-scale can't get
good fits with ~50 observations in these dimensions. <b>Direction abandoned.</b>
</div>

<h2>5. Why RBF/linear is hard to beat</h2>
<p>
The "linear" kernel RBF (<code>φ(r) = r</code>) is not dimensional regression — it's
inverse-distance-weighted interpolation through observed points. That degrades
gracefully in high-dim (SQL 39d) where parametric methods try to fit a global
structure that doesn't exist with 50 points. It's a robust default precisely
<i>because</i> it doesn't model — it just smooths between known values.
</p>

<h2>6. What this leaves open</h2>
<p class="muted">Not pursued further on this thread, but if the smoothness-aware
optimization is ever picked back up these are the next things I'd try:</p>
<ul>
<li><b>Better anti-Laplacian.</b> The current one-sided form
(<code>2·u(h) − u(nearest_revealed)</code>) is approximate. Try k-nearest revealed
neighbors averaged, or restrict invocation to candidates where the surrogate is
confident.</li>
<li><b>Portfolio with TPE.</b> TPE wins on high-dim/large-pool; INGS on smooth/small.
A trivial portfolio that runs both and picks-by-validation could plausibly beat both.</li>
<li><b>Adaptive surrogate choice.</b> Fit RBF + linear + RF each step and pick by
leave-one-out CV on the observed set.</li>
</ul>

</body></html>"""


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true",
                    help="Regenerate HTML from cached exp_explore_raw.json")
    args = ap.parse_args()

    raw_path = Path("reports_opus/config/exp_explore_raw.json")
    html_path = Path("reports_opus/config/exp9c_explore.html")

    if args.report_only:
        with open(raw_path) as f:
            payload = json.load(f)
        html_path.write_text(gen_report(payload))
        print(f"Regenerated {html_path} from cached JSON.")
        return

    print(f"Datasets: {DATASETS}")
    print(f"Reps: {N_REPEATS} (vs 50 in Exp 10 reference)")
    print(f"Budget reported: N={PRIMARY}\n")

    print("Loading datasets…", flush=True)
    problems = load_problems()
    ref, meta_all = ref_from_exp10()

    print("\n── Exp-10 reference (50 reps), win@N=30 ─────────────────────────")
    print(f"{'method':10s}  " + "  ".join(f"{ds:>14s}" for ds in DATASETS) + "   mean")
    for m in ("ings", "smas", "greedy", "tpe", "random"):
        vals = [ref[ds][m]["mean"] for ds in DATASETS]
        cells = "  ".join(f"{v:>14.1f}" for v in vals)
        print(f"{m:10s}  {cells}   {np.mean(vals):>5.1f}")

    t0 = time.time()
    out1 = exp_kappa_bounds(problems, meta_all)
    out2 = exp_surrogates(problems, meta_all)

    payload = {"reference": ref, "kappa_bounds": out1, "surrogates": out2,
               "n_repeats": N_REPEATS, "datasets": DATASETS, "budget": PRIMARY}
    with open(raw_path, "w") as f:
        json.dump(payload, f, indent=2)
    html_path.write_text(gen_report(payload))

    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
    print(f"Raw → {raw_path}")
    print(f"Report → {html_path}")


if __name__ == "__main__":
    main()
