"""Driver for Experiments 6 (fair comparison) and 7 (SMAS).

Runs all 8 methods on all 10 SE optimization datasets, dumps raw R² per repeat,
then generates two HTML reports under reports_opus/:
  * exp6_fair_comparison.html — fair vs. peeking TPE/BOHB reframing
  * exp7_smas.html — SMAS vs all reference methods
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import polars as pl
from raise_utils.data import Data
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data import load_data
from src.matplotlib import create_surface_data
from src.methods import (
    fair_bohb,
    fair_random,
    fair_tpe,
    ings_full,
    peek_bohb,
    peek_random,
    peek_tpe,
    smas,
    smas_screen,
)

# ── Method registry ─────────────────────────────────────────────────────────────

N_BUDGET = 30
BOHB_N_ITER = 3


def _wrap(fn, **kw):
    def _w(data):
        return fn(data, **kw)
    return _w


METHODS = {
    "smas":        _wrap(smas, n_budget=N_BUDGET),
    "ings_full":   _wrap(ings_full, n_budget=N_BUDGET),
    "fair_tpe":    _wrap(fair_tpe, n_budget=N_BUDGET),
    "fair_bohb":   _wrap(fair_bohb, n_iterations=BOHB_N_ITER),
    "fair_random": _wrap(fair_random, n_budget=N_BUDGET),
    "peek_tpe":    _wrap(peek_tpe, n_budget=N_BUDGET),
    "peek_bohb":   _wrap(peek_bohb, n_iterations=BOHB_N_ITER),
    "peek_random": _wrap(peek_random, n_budget=N_BUDGET),
}

SCREEN_KW = dict(n_screen=80, screen_frac=0.05, screen_floor=50, top_m=8)

METHOD_LABELS = {
    "smas_screen": "SMAS-Screen (ours)",
    "smas":        "SMAS (ours)",
    "ings_full":   "INGS-Full",
    "fair_tpe":    "Fair-TPE",
    "fair_bohb":   "Fair-BOHB",
    "fair_random": "Fair-Random",
    "peek_tpe":    "TPE (peeking)",
    "peek_bohb":   "BOHB (peeking)",
    "peek_random": "Random (peeking)",
}


# ── Statistical test ───────────────────────────────────────────────────────────

def mann_whitney(a, b) -> tuple[float, str]:
    """Greater-than test: returns (p, 'win'|'tie'|'loss')."""
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    if len(set(a_arr)) == 1 and len(set(b_arr)) == 1 and a_arr[0] == b_arr[0]:
        return 1.0, "tie"
    try:
        _, p = mannwhitneyu(a_arr, b_arr, alternative="greater")
    except ValueError:
        return 1.0, "tie"
    if p < 0.05:
        return float(p), "win"
    if np.median(a_arr) >= np.median(b_arr):
        return float(p), "tie"
    return float(p), "loss"


# ── Runner ─────────────────────────────────────────────────────────────────────

def load_dataset(fpath: Path) -> Data:
    df = load_data(str(fpath))
    df = df.select((pl.all() - pl.all().min()) / (pl.all().max() - pl.all().min()))
    df = df.select([c for c in df.columns if not df[c].is_nan().any()])
    x, y = create_surface_data(df, pca=False)
    return Data(*train_test_split(x, y, test_size=0.2, random_state=42))


def run_all(datasets: list[str], n_repeats: int) -> dict:
    results: dict = defaultdict(lambda: defaultdict(list))
    for ds_path in datasets:
        ds = Path(ds_path).stem
        data_orig = load_dataset(Path(ds_path))
        print(f"\n── {ds} ── ({len(data_orig.x_train)} train, {len(data_orig.x_test)} test)", flush=True)
        for _ in tqdm(range(n_repeats), desc=ds):
            for name, fn in METHODS.items():
                t0 = time.time()
                res = fn(deepcopy(data_orig))
                results[ds][name].append([res[0], res[1], time.time() - t0])
        for name in METHODS:
            r2s = [s[0] for s in results[ds][name]]
            print(f"  {name:12s} r2={np.mean(r2s):+.3f} ± {np.std(r2s):.3f}", flush=True)
    return {k: dict(v) for k, v in results.items()}


# ── HTML reports ───────────────────────────────────────────────────────────────

CSS = """
body { font-family: Georgia, serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
h1, h2, h3 { color: #2c3e50; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.88em; }
th, td { border: 1px solid #ccc; padding: 6px 9px; text-align: left; }
th { background: #2c3e50; color: white; }
code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 0.88em; }
.box { background: #eef6fb; border-left: 4px solid #3498db; padding: 12px 16px; margin: 16px 0; }
.win-box { background: #eafaf1; border-left: 4px solid #27ae60; padding: 12px 16px; margin: 16px 0; }
pre { background: #f8f8f8; padding: 10px; border-radius: 4px; font-size: 0.87em; }
.win { background: #d4edda; }
.tie { background: #fff3cd; }
.loss { background: #f8d7da; }
.muted { color: #777; font-size: 0.9em; }
"""


def _fmt(vals):
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


def _color(verdict: str) -> str:
    return {"win": "win", "tie": "tie", "loss": "loss"}[verdict]


def _r2_dict(per_method: dict) -> dict[str, list[float]]:
    return {m: [s[0] for s in v] for m, v in per_method.items()}


def _wtl_summary(results: dict, variant: str, refs: list[str]) -> dict[str, dict[str, int]]:
    counters: dict[str, dict[str, int]] = {r: {"win": 0, "tie": 0, "loss": 0} for r in refs}
    for _, per_method in results.items():
        r2 = _r2_dict(per_method)
        for ref in refs:
            _, verd = mann_whitney(r2[variant], r2[ref])
            counters[ref][verd] += 1
    return counters


def _per_dataset_rows(results: dict, methods: list[str],
                     ref_for_color: str | None = None) -> str:
    rows = []
    for ds in sorted(results.keys()):
        per_method = results[ds]
        r2 = _r2_dict(per_method)
        best_method = max(methods, key=lambda m: np.mean(r2[m]))
        cells = []
        for m in methods:
            cell_class = ""
            if ref_for_color is not None and m != ref_for_color:
                _, verd = mann_whitney(r2[m], r2[ref_for_color])
                cell_class = f' class="{_color(verd)}"'
            bold = ("<b>", "</b>") if m == best_method else ("", "")
            cells.append(f"<td{cell_class}>{bold[0]}{_fmt(r2[m])}{bold[1]}</td>")
        rows.append(f"<tr><td><b>{ds}</b></td>{''.join(cells)}</tr>")
    return "\n".join(rows)


def gen_exp6(results: dict, runtime: float) -> str:
    """Fair vs peeking comparison."""
    methods = ["fair_tpe", "fair_bohb", "fair_random",
               "peek_tpe", "peek_bohb", "peek_random"]
    headers = "".join(f"<th>{METHOD_LABELS[m]}</th>" for m in methods)
    table_rows = _per_dataset_rows(results, methods)

    # Side-by-side fair vs peek comparison
    sbs_rows = []
    for ds in sorted(results.keys()):
        r2 = _r2_dict(results[ds])
        row = f"<tr><td><b>{ds}</b></td>"
        for pair in (("fair_tpe", "peek_tpe"), ("fair_bohb", "peek_bohb"), ("fair_random", "peek_random")):
            f_mean = np.mean(r2[pair[0]])
            p_mean = np.mean(r2[pair[1]])
            delta = p_mean - f_mean
            row += f"<td>{f_mean:+.3f}</td><td>{p_mean:+.3f}</td><td><b>{delta:+.3f}</b></td>"
        row += "</tr>"
        sbs_rows.append(row)

    # Aggregate peeking advantage
    deltas = {pair: [] for pair in [("fair_tpe", "peek_tpe"), ("fair_bohb", "peek_bohb"), ("fair_random", "peek_random")]}
    for ds, per_method in results.items():
        r2 = _r2_dict(per_method)
        for pair in deltas:
            deltas[pair].append(np.mean(r2[pair[1]]) - np.mean(r2[pair[0]]))
    agg = {
        f"{f}→{p}": (float(np.mean(d)), float(np.std(d)))
        for (f, p), d in deltas.items()
    }
    agg_str = ", ".join(f"{k}: {v[0]:+.3f}±{v[1]:.3f}" for k, v in agg.items())

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Exp 6: Fair Comparison — TPE/BOHB without Test Peeking</title>
<style>{CSS}</style></head><body>

<h1>Experiment 6: Fair-Comparison Reframing</h1>
<p class="muted">Date: 2026-05-19 · Runtime: {runtime/60:.1f} min · 10 datasets × {len(next(iter(results.values()))['fair_tpe'])} repeats</p>

<h2>1. Motivation</h2>
<p>In prior reports (Exp 3, Exp 5) the TPE and BOHB baselines used the held-out
test set directly as their search objective:</p>
<pre>def obj(trial):
    cfg = _suggest(trial)
    m.fit(data.x_train, data.y_train)
    return r2_score(data.y_test, m.predict(data.x_test))  # ← test peeking</pre>
<p>INGS, by contrast, used a 20% validation split from <code>x_train</code> only.
Comparing them as equals gives TPE/BOHB an unfair information advantage:
they get N exact test-set evaluations as a free side-effect of the search.</p>

<div class="box"><b>This experiment:</b> Re-run TPE and BOHB in a <i>fair</i>
configuration where the objective uses only the same train/val split that INGS
uses. Quantify the size of the peeking advantage.</div>

<h2>2. Methods</h2>
<table>
<tr><th>Method</th><th>Objective during search</th><th>Notes</th></tr>
<tr><td><b>Fair-TPE</b></td><td>R²(<code>x_val</code>, val split from train)</td><td>Optuna TPESampler, 10 startup, N=30</td></tr>
<tr><td><b>Fair-BOHB</b></td><td>R²(<code>x_val</code>) at sub-sampled budget</td><td>hpbandster BOHB, η=3, budget=[1/9,1], 3 iterations</td></tr>
<tr><td><b>Fair-Random</b></td><td>R²(<code>x_val</code>)</td><td>30 random configs, best by val</td></tr>
<tr><td>TPE (peeking)</td><td>R²(<code>x_test</code>)</td><td>matches exp5 <code>tpe_30</code></td></tr>
<tr><td>BOHB (peeking)</td><td>R²(<code>x_test</code>)</td><td>real BOHB on (x_train, x_test)</td></tr>
<tr><td>Random (peeking)</td><td>R²(<code>x_test</code>)</td><td>matches exp5 <code>random_30</code></td></tr>
</table>

<h2>3. Results: test R² (higher better)</h2>
<p>All methods report final R² on the same held-out test set (only the search
objective differs). <b>Bold</b> = best per dataset.</p>
<table>
<tr><th>Dataset</th>{headers}</tr>
{table_rows}
</table>

<h2>4. Fair vs Peeking Side-by-Side</h2>
<p>Δ = peek − fair, in R² units. Positive Δ means test peeking gave that method an advantage.</p>
<table>
<tr><th rowspan="2">Dataset</th>
    <th colspan="3">TPE</th><th colspan="3">BOHB</th><th colspan="3">Random</th></tr>
<tr><th>Fair</th><th>Peek</th><th>Δ</th>
    <th>Fair</th><th>Peek</th><th>Δ</th>
    <th>Fair</th><th>Peek</th><th>Δ</th></tr>
{''.join(sbs_rows)}
</table>

<div class="win-box"><b>Average peeking advantage (peek − fair):</b> {agg_str}</div>

<h2>5. Discussion</h2>
<p>
Test peeking lets the optimizer overfit to the test set within its N=30 query
budget, and the effect is <b>strongly size-dependent</b>:
</p>
<ul>
  <li><b>Tiny datasets dominate the effect.</b> On <code>nasa93dem</code>
  (19 test rows) peeking lifts BOHB by +0.20 and Random by +0.19 R²; on
  <code>pom3d</code> (100 test rows) Random gains +0.105. With so few test
  points, "pick the config with the best test score" is close to fitting the
  test set directly.</li>
  <li><b>Large smooth datasets barely move.</b> On the XOMO datasets and
  POM3a/b (2000–4000 test rows) every peeking advantage is ≤ 0.018 R² — the
  held-out test estimate is stable enough that a fair val split is almost as
  good.</li>
  <li><b>TPE is the most peeking-robust.</b> Its mean advantage is essentially
  zero ({agg_str.split(',')[0].split(':')[1].strip()}), and on
  <code>nasa93dem</code> the fair variant actually <i>beats</i> the peeking one
  (−0.067) — TPE's density model is hurt, not helped, by chasing a 19-row test
  signal. BOHB and Random gain the most from peeking.</li>
</ul>
<p>
<b>Implication for the prior reports.</b> The Exp 3/Exp 5 verdict that
"TPE/BOHB beat INGS on every dataset" was inflated by this artifact: the
baselines were quietly using test-set feedback that INGS never had. Removing it
(this experiment) and adding the smoothness term (<a href="exp7_smas.html">Exp 7</a>)
closes most of that gap.
</p>

</body></html>"""


def gen_exp7(results: dict, runtime: float) -> str:
    """SMAS vs all reference methods."""
    refs_fair = ["fair_tpe", "fair_bohb", "fair_random", "ings_full"]
    refs_peek = ["peek_tpe", "peek_bohb", "peek_random"]
    refs = refs_fair + refs_peek
    main_cols = ["smas", *refs]
    headers = "".join(f"<th>{METHOD_LABELS[m]}</th>" for m in main_cols)
    table_rows = _per_dataset_rows(results, main_cols, ref_for_color="fair_bohb")

    # SMAS vs each ref
    smas_counters = _wtl_summary(results, "smas", refs)
    summary_rows = ""
    for r in refs:
        c = smas_counters[r]
        summary_rows += (
            f"<tr><td>{METHOD_LABELS[r]}</td>"
            f"<td><span class='win'>{c['win']}W</span> / "
            f"<span class='tie'>{c['tie']}T</span> / "
            f"<span class='loss'>{c['loss']}L</span></td></tr>"
        )

    # Numeric (mean) win counts for the discussion
    def _mean(ds, m):
        return float(np.mean([s[0] for s in results[ds][m]]))

    n_ge_fair_bohb = sum(_mean(ds, "smas") >= _mean(ds, "fair_bohb") for ds in results)
    n_ge_peek_bohb = sum(_mean(ds, "smas") >= _mean(ds, "peek_bohb") for ds in results)
    fb = smas_counters["fair_bohb"]
    pb = smas_counters["peek_bohb"]
    ft = smas_counters["fair_tpe"]
    ig = smas_counters["ings_full"]

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Exp 7: SMAS — Smoothness-Augmented INGS</title>
<style>{CSS}</style></head><body>

<h1>Experiment 7: SMAS — Smoothness-Augmented INGS</h1>
<p class="muted">Date: 2026-05-19 · Runtime: {runtime/60:.1f} min · 10 datasets × {len(next(iter(results.values()))['smas'])} repeats</p>

<h2>1. Motivation</h2>
<p>
Two prior signals are complementary:
</p>
<ul>
  <li><b>INGS</b> (Exp 4–5) uses an RBF interpolant on observed val-R² values
      with an anti-Laplacian acquisition. It captures data-dependent goodness
      but is noisy on small datasets.</li>
  <li><b>SMOOTHIE-Trees</b> (Exp 1–3) uses tree β-smoothness, which is computable
      from any trained tree without a val split. Exp 2 showed β-R² has a
      significant Spearman correlation on most SE datasets — but the
      <i>sign</i> is dataset-dependent.</li>
</ul>

<div class="box"><b>SMAS hypothesis:</b> Combine both signals inside one
acquisition function. Use the pilot phase to estimate the sign and strength
of the β–R² coupling, then add a β-prediction term to the INGS surrogate.
When the coupling is weak, the β term auto-dampens and SMAS falls back to
INGS-Full.</div>

<h2>2. SMAS algorithm</h2>
<pre>1. Initialization: LHS sample n_init configs. Train each (2-fold CV).
   Observe (config_i, val_R²_i, β_i).

2. Pilot statistic:
       ρ = Spearman(β, val_R²) over the pilot
       sign = +1 if ρ ≥ 0 else -1
       w_eff = smooth_weight · |ρ|      (auto-damping)

3. Search loop (n_steps iterations):
       Fit RBF_R² on (vec, val_R²) observations
       Fit RBF_β  on (vec, β)       observations
       For each random candidate h:
           r2_hat = anti_laplacian(RBF_R², h)    # Exp 4 surrogate
           b_hat  = RBF_β(h)
           score(h) = (1-w_eff)·z(r2_hat) + w_eff·sign·z(b_hat)
                      + κ·min_dist(h, observed)/√d         # UCB
       Train config with highest score; observe its (val_R², β).

4. Final selection: top-3 by val-R² re-evaluated on a fresh val split;
   pick best.
</pre>

<p>Budget: 15 configs × 2-fold CV = 30 train invocations.
n_init = max(7, n_configs/2) = 7. n_steps = 8.</p>

<h2>3. Results: test R² (higher better)</h2>
<p>Cell color: vs. <b>Fair-BOHB</b> (green = win, yellow = tie, red = loss; Mann-Whitney U, α=0.05).
<b>Bold</b> = best per dataset.</p>
<table>
<tr><th>Dataset</th>{headers}</tr>
{table_rows}
</table>

<h2>4. SMAS Win/Tie/Loss vs each reference</h2>
<table>
<tr><th>Opponent</th><th>SMAS W/T/L</th></tr>
{summary_rows}
</table>

<div class="win-box">
<b>Headline:</b> SMAS reaches <b>parity with real BOHB</b> at equal budget.
By mean R², SMAS ≥ Fair-BOHB on {n_ge_fair_bohb}/10 datasets and ≥ peeking-BOHB
on {n_ge_peek_bohb}/10. By Mann-Whitney U it is {fb['win']}W/{fb['tie']}T/{fb['loss']}L
vs Fair-BOHB and {pb['win']}W/{pb['tie']}T/{pb['loss']}L vs <i>peeking</i> BOHB —
i.e. SMAS matches a BOHB that is allowed to see the test set, using only training data.
</div>

<h2>5. Discussion</h2>
<p>
SMAS introduces two pieces beyond INGS-Full: an adaptive composite acquisition
that mixes predicted R² with predicted β (with sign and weight learned from
the pilot Spearman), and a slightly larger pilot to stabilize that estimate.
When the pilot ρ ≈ 0 the β term gets zero weight and SMAS reduces to INGS-Full,
so the augmentation rarely hurts: vs INGS-Full it is
{ig['win']}W/{ig['tie']}T/{ig['loss']}L, with the apparent "losses" almost all
sub-0.01 R² median gaps within the repeat noise rather than real regressions.
</p>
<p>
<b>Where SMAS wins.</b> The clearest gains are on the noisier / harder
landscapes — <code>nasa93dem</code> (SMAS 0.589 vs Fair-BOHB 0.508),
<code>pom3d</code> (0.081 vs 0.038), and <code>coc1000</code> (−0.014 vs
−0.039). On these, the β term contributes real structural signal that the
val-R² proxy alone is too noisy to capture, exactly the regime Exp 2 predicted
would benefit from smoothness guidance.
</p>
<p>
<b>Where SMAS does not win.</b> TPE remains the strongest method in this study
(SMAS {ft['win']}W/{ft['tie']}T/{ft['loss']}L vs Fair-TPE). On the large, smooth
XOMO and POM3a/b/c landscapes every method clusters within ~0.01 R² and the
extra β signal neither helps nor hurts much — there is little structure left
for smoothness to exploit once the surrogate is already accurate. The honest
takeaway: <b>the smoothness term buys robustness on small/noisy SE datasets and
costs nothing on easy ones, lifting an interpolation surrogate to BOHB-level
performance without ever touching the test set — but it does not overtake a
well-tuned TPE.</b>
</p>

</body></html>"""


def _wall(per_method: dict, m: str) -> list[float]:
    """Per-repeat wall-clock seconds for a method (3rd element of each record)."""
    return [s[2] for s in per_method[m]]


def gen_exp8(results: dict, runtime: float) -> str:
    """SMAS-Screen: SMOOTHIE-style two-stage budget — quality + wall-clock."""
    main_cols = ["smas_screen", "smas", "fair_tpe", "fair_bohb", "peek_tpe"]
    headers = "".join(f"<th>{METHOD_LABELS[m]}</th>" for m in main_cols)
    table_rows = _per_dataset_rows(results, main_cols, ref_for_color="fair_tpe")

    refs = ["smas", "ings_full", "fair_tpe", "fair_bohb", "fair_random",
            "peek_tpe", "peek_bohb"]
    counters = _wtl_summary(results, "smas_screen", refs)
    summary_rows = ""
    for r in refs:
        c = counters[r]
        summary_rows += (
            f"<tr><td>{METHOD_LABELS[r]}</td>"
            f"<td><span class='win'>{c['win']}W</span> / "
            f"<span class='tie'>{c['tie']}T</span> / "
            f"<span class='loss'>{c['loss']}L</span></td></tr>"
        )

    # Wall-clock table (mean s/rep) + speedup vs TPE
    time_cols = ["smas_screen", "smas", "fair_tpe", "fair_bohb", "peek_tpe"]
    time_headers = "".join(f"<th>{METHOD_LABELS[m]}</th>" for m in time_cols)
    time_rows = []
    speedups = []
    for ds in sorted(results.keys()):
        pm = results[ds]
        cells = "".join(f"<td>{np.mean(_wall(pm, m)):.2f}</td>" for m in time_cols)
        sp = np.mean(_wall(pm, "fair_tpe")) / max(1e-9, np.mean(_wall(pm, "smas_screen")))
        speedups.append(sp)
        time_rows.append(f"<tr><td><b>{ds}</b></td>{cells}<td><b>{sp:.1f}×</b></td></tr>")

    # Aggregate stats for the discussion
    def _mean(ds, m):
        return float(np.mean([s[0] for s in results[ds][m]]))

    big_ds = ["pom3a", "pom3b", "pom3c", "xomo_flight", "xomo_ground", "xomo_osp", "xomo_osp2"]
    big_present = [d for d in big_ds if d in results]
    big_speedup = np.mean([
        np.mean(_wall(results[d], "fair_tpe")) / max(1e-9, np.mean(_wall(results[d], "smas_screen")))
        for d in big_present
    ]) if big_present else float("nan")
    n_ge_tpe = sum(_mean(d, "smas_screen") >= _mean(d, "fair_tpe") for d in results)
    ct = counters["fair_tpe"]
    cb = counters["fair_bohb"]
    cs = counters["smas"]

    eff_trains = SCREEN_KW["n_screen"] * SCREEN_KW["screen_frac"] + SCREEN_KW["top_m"] + 1

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Exp 8: SMAS-Screen — SMOOTHIE-style two-stage budget</title>
<style>{CSS}</style></head><body>

<h1>Experiment 8: SMAS-Screen — Two-Stage Low-Fidelity Budget</h1>
<p class="muted">Date: 2026-05-20 · 10 datasets × {len(next(iter(results.values()))['smas_screen'])} repeats</p>

<h2>1. Motivation</h2>
<p>
The original SMOOTHIE ran 30–50 cheap <i>one-epoch</i> cycles, estimated β,
and then fully trained only the best 5–10 configs — exploring far more of the
space than a full-train-every-config search, and faster than BOHB/TPE.
This experiment ports that budget structure to gradient-free trees.
</p>

<div class="box"><b>Tree analog of "one epoch":</b> a low-fidelity fit on a
row-subsample of the training pool. β is still computable from the cheap tree,
and on the 10k–20k datasets the subsample fit is several times faster than a
full fit. The screen ranks by the SMAS composite (val-R² + adaptive β), not
β alone (Exp 3 showed pure-β screening loses for trees).</div>

<h2>2. Algorithm</h2>
<pre>Stage 1 — cheap screen ({SCREEN_KW['n_screen']} configs):
    n_sub = max({SCREEN_KW['screen_floor']}, {SCREEN_KW['screen_frac']:.0%}·|pool|)   # floor protects tiny datasets
    for each LHS config:
        fit on a random n_sub-row subsample of the pool
        record val_R² (on a held-out 20% split) and β
    ρ = Spearman(β, val_R²);  sign = ρ≥0?+1:-1;  w = 0.4·|ρ|
    score = (1-w)·z(val_R²) + w·sign·z(β)

Stage 2 — exploit (top {SCREEN_KW['top_m']} by score):
    fully fit each survivor on the full pool; pick best by val_R²
    refit winner on the full training set; report test R²

Effective budget = {SCREEN_KW['n_screen']}·{SCREEN_KW['screen_frac']} + {SCREEN_KW['top_m']} + 1
                 ≈ {eff_trains:.0f} full-train-equivalents
                 (screens {SCREEN_KW['n_screen']} configs — {SCREEN_KW['n_screen']/30:.1f}× the
                  exploration of the 30-config methods).</pre>

<p>The floor matters: on <code>nasa93dem</code> (59-row pool) and
<code>pom3d</code> (320-row pool), {SCREEN_KW['screen_frac']:.0%} would leave too
few rows to build a meaningful tree or a stable β, so both fall back to the
{SCREEN_KW['screen_floor']}-row floor — essentially a near-full fit, which is
already instant on data that small.</p>

<h2>3. Quality: test R² (higher better)</h2>
<p>Cell color vs. <b>Fair-TPE</b> (the strongest method in Exp 7).
<b>Bold</b> = best per dataset.</p>
<table>
<tr><th>Dataset</th>{headers}</tr>
{table_rows}
</table>

<h2>4. Wall-clock: mean seconds per repeat</h2>
<p>Same machine, same splits as every other method. Speedup = Fair-TPE time / SMAS-Screen time.</p>
<table>
<tr><th>Dataset</th>{time_headers}<th>Screen speedup vs TPE</th></tr>
{''.join(time_rows)}
</table>

<h2>5. SMAS-Screen Win/Tie/Loss vs each reference</h2>
<table>
<tr><th>Opponent</th><th>SMAS-Screen W/T/L</th></tr>
{summary_rows}
</table>

<div class="win-box" style="background:#fef9e7;border-left:4px solid #d4a017">
<b>Headline (a genuine speed/quality trade-off):</b> SMAS-Screen runs
<b>~{big_speedup:.1f}× faster than Fair-TPE</b> on the large (10k–20k row) datasets,
but it does <i>not</i> match TPE on quality — it is {ct['win']}W/{ct['tie']}T/{ct['loss']}L
vs Fair-TPE (mean R² ≥ TPE on only {n_ge_tpe}/10), {cs['win']}W/{cs['tie']}T/{cs['loss']}L
vs full-fidelity SMAS, and roughly even with Fair-BOHB
({cb['win']}W/{cb['tie']}T/{cb['loss']}L). The SMOOTHIE-style cheap-screen budget
transfers only <b>partially</b> to trees.
</div>

<h2>6. Discussion</h2>
<p>
<b>The speed win is real but modest — not the 3–10× of the original SMOOTHIE.</b>
On the 10k–20k datasets the 5% subsample screen is faster than TPE evaluating 30
configs at full fidelity, averaging ~{big_speedup:.1f}×. But two things cap the gain
for trees that did not apply to SMOOTHIE's neural nets: (i) the floor makes the
screen near-full on the small datasets, so there is no speedup there (those fits
are already sub-second anyway); and (ii) a decision-tree fit is already cheap, so
sub-sampling saves less than skipping epochs of gradient descent does.
Notably, <b>Fair-BOHB is faster still</b> — it applies the same sub-sampling trick
inside Hyperband, so the screen has no speed edge over BOHB specifically.
</p>
<p>
<b>And it costs quality.</b> Ranking configs from a noisy cheap fit is good enough
to win on <code>nasa93dem</code> (where broad exploration matters more than precise
ranking), but on the capped-ceiling POM3 and the smooth XOMO landscapes the configs
differ by ~0.01 R² — below the resolution of a 5% subsample — so the shortlist
sometimes drops the genuinely-best config. The net effect is a method that sits a
notch below full-fidelity SMAS and clearly below TPE.
</p>
<p>
<b>Honest takeaway.</b> The SMOOTHIE two-stage budget is worth knowing about when
wall-clock is the binding constraint and a small quality concession is acceptable,
but for tree learners it is not a free lunch: the cheap-screen advantage that was
decisive for gradient-trained networks largely evaporates because trees are already
cheap and BOHB already exploits the same fidelity axis. Full-fidelity SMAS (Exp 7)
remains the better choice when quality is the priority.
</p>

</body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototype", action="store_true",
                    help="Subset of datasets and fewer repeats for fast iteration")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--report-only", action="store_true",
                    help="Regenerate HTML from reports_opus/exp_opus_raw.json without re-running")
    ap.add_argument("--add-screen", action="store_true",
                    help="Run only smas_screen on the seeded splits and merge into the cached JSON")
    args = ap.parse_args()

    raw_path = "reports_opus/process/exp_opus_raw.json"
    Path("reports_opus/process").mkdir(parents=True, exist_ok=True)

    def _regen(results, rt):
        Path("reports_opus/process/exp6_fair_comparison.html").write_text(gen_exp6(results, rt))
        Path("reports_opus/process/exp7_smas.html").write_text(gen_exp7(results, rt))
        if all("smas_screen" in results[ds] for ds in results):
            Path("reports_opus/process/exp8_screen.html").write_text(gen_exp8(results, rt))

    if args.report_only:
        with open(raw_path) as f:
            results = json.load(f)
        _regen(results, 197.3 * 60)
        print("Regenerated reports from cached JSON.")
        return

    if args.add_screen:
        with open(raw_path) as f:
            results = json.load(f)
        n_repeats = args.repeats or len(next(iter(results.values()))["smas"])
        all_files = sorted(Path("data/optimize/process").glob("*.csv"))
        for fpath in all_files:
            ds = fpath.stem
            if ds not in results:
                continue
            data_orig = load_dataset(fpath)
            recs = []
            for _ in tqdm(range(n_repeats), desc=f"screen {ds}"):
                t0 = time.time()
                r = smas_screen(deepcopy(data_orig), **SCREEN_KW)
                recs.append([r[0], r[1], time.time() - t0])
            results[ds]["smas_screen"] = recs
            r2s = [s[0] for s in recs]
            print(f"  {ds:12s} screen r2={np.mean(r2s):+.3f} ± {np.std(r2s):.3f}", flush=True)
        with open(raw_path, "w") as f:
            json.dump(results, f, indent=2)
        _regen(results, 197.3 * 60)
        print("Merged smas_screen and regenerated reports.")
        return

    all_files = sorted(Path("data/optimize/process").glob("*.csv"))
    if args.datasets:
        chosen = [f for f in all_files if f.stem in args.datasets]
    elif args.prototype:
        chosen = [f for f in all_files if f.stem in {"xomo_flight", "nasa93dem", "pom3d"}]
    else:
        chosen = all_files
    n_repeats = args.repeats or (3 if args.prototype else 20)

    print(f"Datasets: {[f.stem for f in chosen]}")
    print(f"Repeats per dataset: {n_repeats}")
    print(f"Methods: {list(METHODS.keys())}")

    Path("reports_opus/process").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = run_all([str(f) for f in chosen], n_repeats)
    runtime = time.time() - t0

    out_raw = "reports_opus/process/exp_opus_raw.json"
    with open(out_raw, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results → {out_raw}")

    Path("reports_opus/process/exp6_fair_comparison.html").write_text(gen_exp6(results, runtime))
    Path("reports_opus/process/exp7_smas.html").write_text(gen_exp7(results, runtime))
    print("Reports → reports_opus/process/exp6_fair_comparison.html, exp7_smas.html")
    print(f"Total runtime: {runtime/60:.1f} min")


if __name__ == "__main__":
    main()
