"""Exp 10 — Wide replication of Exp 9 across 28 config / HPO datasets.

Same methodology as Exp 9 (pool-based active optimization, d2h metric, normalized
win score, 5 methods × 50 repeats × budgets 10/20/30/50). Wider dataset coverage:

  • 24 SS-A through SS-X from data/optimize/config/
  • Apache, X264, SQL from data/optimize/config/ (also in Exp 9; included for
    a single unified report)
  • Health-ClosedIssues combined from data/optimize/hpo/ (12 files vertically
    concatenated → ~120k row pool; disjoint sampling sweeps, not noisy replicates)

  • Scrum datasets explicitly excluded — to be run separately overnight.

Each step scores every unrevealed pool row exhaustively (no subsampling),
matching the Exp 9 setup; runtime is dominated by SS-N/W/X and Health where
the per-step distance matrix is large.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from src.config_opt import (
    BUDGETS, METHODS, load_problem, load_problem_combined,
)
from src.matplotlib import setup_matplotlib


# ── Dataset list (28) ─────────────────────────────────────────────────────────

SS_NAMES = [f"SS-{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWX"]  # 24
NAMED_CONFIG = ["Apache_AllMeasurements", "X264_AllMeasurements", "SQL_AllMeasurements"]

CONFIG_DIR = Path("data/optimize/config")
HPO_DIR = Path("data/optimize/hpo")
HEALTH_FILES = sorted(str(p) for p in HPO_DIR.glob("Health-ClosedIssues*.csv"))
HEALTH_NAME = "Health-ClosedIssues"

OUT_DIR = Path("reports_opus/config")
PRIMARY_BUDGET = 30
METHOD_ORDER = ["smas", "ings", "greedy", "tpe", "random"]
METHOD_LABELS = {
    "smas":   "SMAS",
    "ings":   "INGS",
    "greedy": "Greedy",
    "tpe":    "TPE",
    "random": "Random",
}


def load_dataset(name: str):
    if name == HEALTH_NAME:
        return load_problem_combined(HEALTH_FILES, HEALTH_NAME)
    return load_problem(str(CONFIG_DIR / f"{name}.csv"))


# ── Runner ──────────────────────────────────────────────────────────────────

def run(datasets: list[str], n_repeats: int) -> dict:
    results: dict = {}
    meta: dict = {}
    for ds in datasets:
        t0_ds = time.time()
        p = load_dataset(ds)
        meta[ds] = {"pool": p.n_pool, "dim": p.dim, "n_obj": p.n_obj,
                    "oracle_min": p.oracle_min, "oracle_median": p.oracle_median}
        print(f"\n── {ds}  pool={p.n_pool} dim={p.dim} nobj={p.n_obj} "
              f"d*={p.oracle_min:.4f} d0={p.oracle_median:.4f}", flush=True)
        per = {m: {b: [] for b in BUDGETS} for m in METHODS}
        for r in range(n_repeats):
            for mi, (m, fn) in enumerate(METHODS.items()):
                rng = np.random.default_rng(10_000 * r + 17 * mi + 1)
                res = fn(p, rng)
                for b in BUDGETS:
                    per[m][b].append(res[b])
        results[ds] = per
        for m in METHOD_ORDER:
            scores = [_win(d, p.oracle_min, p.oracle_median) for d in per[m][PRIMARY_BUDGET]]
            print(f"  {m:7s} d2h@N{PRIMARY_BUDGET}={np.mean(per[m][PRIMARY_BUDGET]):.4f}  "
                  f"win={np.mean(scores):.1f}", flush=True)
        print(f"  ({time.time()-t0_ds:.1f}s)", flush=True)
    return {"results": results, "meta": meta, "n_repeats": n_repeats}


# ── Metrics ──────────────────────────────────────────────────────────────────

def _win(d_best: float, d_star: float, d_zero: float) -> float:
    denom = d_zero - d_star
    if denom < 1e-12:
        return float("nan")
    return 100.0 * (1.0 - (d_best - d_star) / denom)


def mw_lower(a, b) -> str:
    a, b = np.asarray(a), np.asarray(b)
    if len(set(a)) == 1 and len(set(b)) == 1 and a[0] == b[0]:
        return "tie"
    try:
        _, p = mannwhitneyu(a, b, alternative="less")
    except ValueError:
        return "tie"
    if p < 0.05:
        return "win"
    return "tie" if np.median(a) <= np.median(b) else "loss"


# ── Plots ────────────────────────────────────────────────────────────────────

def make_plots(payload: dict) -> None:
    setup_matplotlib()
    from matplotlib import pyplot as plt
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"smas": "#c0392b", "ings": "#e67e22", "greedy": "#7f8c8d",
              "tpe": "#2980b9", "random": "#95a5a6"}
    for ds, per in payload["results"].items():
        m_oracle = payload["meta"][ds]["oracle_min"]
        m_med = payload["meta"][ds]["oracle_median"]
        fig, ax = plt.subplots(figsize=(4.4, 3.0))
        for m in METHOD_ORDER:
            wins = [[_win(d, m_oracle, m_med) for d in per[m][b]] for b in BUDGETS]
            means = [np.mean(w) for w in wins]
            sems = [np.std(w) / np.sqrt(len(w)) for w in wins]
            ax.errorbar(BUDGETS, means, yerr=sems, marker="o", ms=3, capsize=2,
                        label=METHOD_LABELS[m], color=colors[m],
                        lw=2.0 if m in ("smas", "ings") else 1.0,
                        ls="-" if m in ("smas", "ings", "tpe") else "--")
        ax.axhline(100, color="green", ls=":", lw=1, label="oracle")
        ax.axhline(0, color="black", ls=":", lw=1)
        ax.set_xlabel("budget")
        ax.set_ylabel("win score (↑)")
        ax.set_title(ds, fontsize=10)
        ax.legend(fontsize=6, framealpha=0.9, loc="lower right")
        ax.set_xticks(BUDGETS)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"exp10_{ds}.png", dpi=105)
        plt.close(fig)

    # Headline aggregate plot — mean win across all datasets vs budget.
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for m in METHOD_ORDER:
        per_b = []
        for b in BUDGETS:
            ds_means = []
            for ds, per in payload["results"].items():
                mo = payload["meta"][ds]["oracle_min"]
                me = payload["meta"][ds]["oracle_median"]
                ds_means.append(np.mean([_win(d, mo, me) for d in per[m][b]]))
            per_b.append(ds_means)
        means = [np.mean(x) for x in per_b]
        sems = [np.std(x) / np.sqrt(len(x)) for x in per_b]
        ax.errorbar(BUDGETS, means, yerr=sems, marker="o", ms=5, capsize=3,
                    label=METHOD_LABELS[m], color=colors[m],
                    lw=2.3 if m in ("smas", "ings") else 1.4,
                    ls="-" if m in ("smas", "ings", "tpe") else "--")
    ax.axhline(100, color="green", ls=":", lw=1, label="oracle")
    ax.set_xlabel("evaluation budget")
    ax.set_ylabel("mean win score across datasets (↑)")
    ax.set_title(f"Aggregate convergence across {len(payload['results'])} datasets")
    ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
    ax.set_xticks(BUDGETS)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp10_aggregate.png", dpi=120)
    plt.close(fig)


# ── Report ───────────────────────────────────────────────────────────────────

CSS = """
body { font-family: Georgia, serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
h1, h2, h3 { color: #2c3e50; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 0.84em; }
th, td { border: 1px solid #ccc; padding: 5px 8px; text-align: center; vertical-align: top; }
th { background: #2c3e50; color: white; }
td.dsname { text-align: left; font-weight: bold; white-space: nowrap; }
code { background: #f4f4f4; padding: 2px 4px; border-radius: 3px; font-size: 0.88em; }
.box { background: #eef6fb; border-left: 4px solid #3498db; padding: 12px 16px; margin: 16px 0; }
.win-box { background: #eafaf1; border-left: 4px solid #27ae60; padding: 12px 16px; margin: 16px 0; }
.amber-box { background: #fef9e7; border-left: 4px solid #d4a017; padding: 12px 16px; margin: 16px 0; }
.muted { color: #777; font-size: 0.88em; }
.win { background: #d4edda; } .tie { background: #fff3cd; } .loss { background: #f8d7da; }
.best { font-weight: bold; }
img.curve { width: 31%; margin: 3px; vertical-align: top; border: 1px solid #eee; }
img.agg { max-width: 100%; border: 1px solid #ccc; }
"""


def gen_report(payload: dict, runtime: float) -> str:
    results, meta, n_rep = payload["results"], payload["meta"], payload["n_repeats"]
    ds_list = list(results.keys())

    def win_at(m: str, b: int, ds: str) -> list[float]:
        mo = meta[ds]["oracle_min"]; me = meta[ds]["oracle_median"]
        return [_win(d, mo, me) for d in results[ds][m][b]]

    def mean_win(m: str, b: int) -> float:
        return float(np.mean([np.mean(win_at(m, b, ds)) for ds in ds_list]))

    # Per-dataset win-score table at PRIMARY_BUDGET (colored vs TPE).
    rows = []
    for ds in ds_list:
        per_ds = {m: float(np.mean(win_at(m, PRIMARY_BUDGET, ds))) for m in METHOD_ORDER}
        best_m = max(METHOD_ORDER, key=lambda mm: per_ds[mm])
        cells = ""
        for m in METHOD_ORDER:
            cls_parts = []
            if m != "tpe":
                a = np.array(win_at(m, PRIMARY_BUDGET, ds))
                t = np.array(win_at("tpe", PRIMARY_BUDGET, ds))
                if np.allclose(a, t):
                    verd = "tie"
                else:
                    try:
                        _, pv = mannwhitneyu(a, t, alternative="greater")
                        if pv < 0.05: verd = "win"
                        elif np.median(a) >= np.median(t): verd = "tie"
                        else: verd = "loss"
                    except Exception:
                        verd = "tie"
                cls_parts.append(verd)
            if m == best_m: cls_parts.append("best")
            cls_attr = f' class="{" ".join(cls_parts)}"' if cls_parts else ""
            cells += f"<td{cls_attr}>{per_ds[m]:.1f}</td>"
        rows.append(
            f"<tr><td class='dsname'>{ds}</td>"
            f"<td class='muted'>{meta[ds]['pool']}×{meta[ds]['dim']}</td>"
            f"{cells}</tr>")

    head = "".join(f"<th>{METHOD_LABELS[m]}</th>" for m in METHOD_ORDER)
    win_table = (f"<table><tr><th>Dataset</th><th>pool×dim</th>{head}</tr>"
                 + "".join(rows) + "</table>")

    # Mean across datasets per method × budget.
    agg_rows = ""
    for m in METHOD_ORDER:
        cells = "".join(f"<td>{mean_win(m, b):.1f}</td>" for b in BUDGETS)
        agg_rows += f"<tr><td>{METHOD_LABELS[m]}</td>{cells}</tr>"
    agg_table = (f"<table><tr><th>Method</th>"
                 + "".join(f"<th>N={b}</th>" for b in BUDGETS)
                 + f"</tr>{agg_rows}</table>")

    # Precompute headline figures for prose.
    smas_vs_tpe_wtl = {}
    ings_vs_tpe_wtl = {}
    for b in BUDGETS:
        for v, store in (("smas", smas_vs_tpe_wtl), ("ings", ings_vs_tpe_wtl)):
            w = t = l = 0
            for ds in ds_list:
                a = np.array(win_at(v, b, ds))
                tt = np.array(win_at("tpe", b, ds))
                if np.allclose(a, tt): t += 1; continue
                try:
                    _, pv = mannwhitneyu(a, tt, alternative="greater")
                except Exception:
                    t += 1; continue
                if pv < 0.05: w += 1
                elif np.median(a) >= np.median(tt): t += 1
                else: l += 1
            store[b] = f"{w}W/{t}T/{l}L"

    smas_paired_p = {}
    ings_paired_p = {}
    for b in BUDGETS:
        for v, store in (("smas", smas_paired_p), ("ings", ings_paired_p)):
            a = np.array([np.mean(win_at(v, b, ds)) for ds in ds_list])
            tt = np.array([np.mean(win_at("tpe", b, ds)) for ds in ds_list])
            try:
                _, pv = wilcoxon(a, tt, alternative="greater")
            except Exception:
                pv = float("nan")
            store[b] = pv

    # Paired Wilcoxon (1-sided, greater) on per-dataset mean win scores vs TPE.
    paired_rows = ""
    for m in ("smas", "ings", "greedy", "random"):
        cells = ""
        for b in BUDGETS:
            a = np.array([np.mean(win_at(m, b, ds)) for ds in ds_list])
            t = np.array([np.mean(win_at("tpe", b, ds)) for ds in ds_list])
            try:
                _, pv = wilcoxon(a, t, alternative="greater")
            except Exception:
                pv = float("nan")
            sig = " <b>win</b>" if pv < 0.05 else ""
            cells += f"<td>p={pv:.3f}{sig}</td>"
        paired_rows += f"<tr><td>{METHOD_LABELS[m]} vs TPE</td>{cells}</tr>"
    paired_table = (f"<table><tr><th>Comparison</th>"
                    + "".join(f"<th>N={b}</th>" for b in BUDGETS)
                    + f"</tr>{paired_rows}</table>")

    # W/T/L counts: SMAS/INGS vs TPE on per-dataset win scores at each budget.
    wtl_rows = ""
    for v in ("smas", "ings"):
        for ref in ("tpe", "greedy", "random"):
            cells = ""
            for b in BUDGETS:
                w = t = l = 0
                for ds in ds_list:
                    # Higher win is better — use Mann-Whitney "greater" semantics
                    a = np.array(win_at(v, b, ds))
                    r = np.array(win_at(ref, b, ds))
                    if np.allclose(a, r):
                        t += 1; continue
                    try:
                        _, pv = mannwhitneyu(a, r, alternative="greater")
                    except Exception:
                        t += 1; continue
                    if pv < 0.05: w += 1
                    elif np.median(a) >= np.median(r): t += 1
                    else: l += 1
                cells += f"<td>{w}W/{t}T/{l}L</td>"
            wtl_rows += f"<tr><td>{METHOD_LABELS[v]} vs {METHOD_LABELS[ref]}</td>{cells}</tr>"
    wtl_table = (f"<table><tr><th>Comparison</th>"
                 + "".join(f"<th>N={b}</th>" for b in BUDGETS)
                 + f"</tr>{wtl_rows}</table>")

    # Convergence: aggregate plot + per-dataset grid.
    per_ds_imgs = "".join(f'<img class="curve" src="exp10_{ds}.png" alt="{ds}">' for ds in ds_list)

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Exp 10: Wide replication of smoothness-guided config optimization</title>
<style>{CSS}</style></head><body>

<h1>Experiment 10: Wide replication across {len(ds_list)} datasets</h1>
<p class="muted">Date: 2026-05-25 · {len(ds_list)} datasets × {n_rep} repeats × budgets {BUDGETS} · runtime {runtime/60:.1f} min ·
<a href="exp9_config_opt.html">↩ Exp 9 (8 datasets)</a> · <a href="../index.html">↑ index</a></p>

<h2>1. Setup</h2>
<p>
Same methodology as <a href="exp9_config_opt.html">Exp 9</a>: pool-based active
optimization on lookup tables; metric = best d2h found within an evaluation
budget; normalized win score 100·(1 − (d2h(best) − d*)/(d₀ − d*)) where d* and d₀
are the min and median d2h over the <b>full pool</b>; 5 methods (Random, TPE,
Greedy surrogate, INGS, SMAS); budgets 10/20/30/50; 50 repeats.
</p>
<div class="box">
<b>Datasets ({len(ds_list)}):</b>
24 SS-A through SS-X + Apache, X264, SQL from <code>data/optimize/config/</code>;
Health-ClosedIssues concatenated from 12 files in <code>data/optimize/hpo/</code>
(~120k row pool of disjoint sampling sweeps).
Scrum1k/10k/100k explicitly excluded; to be run separately.
</div>
<p class="muted">Each step scores every unrevealed pool row exhaustively (no candidate subsampling), matching Exp 9.</p>

<div class="amber-box">
<b>Headline — the Exp 9 result moderates with wider coverage.</b> Across the
{len(ds_list)}-dataset set, both INGS and SMAS clearly beat the no-smoothness
Greedy ablation (the smoothness signal is real) and outperform TPE on aggregate
mean win score by ~5 points at N={PRIMARY_BUDGET}: <b>INGS {mean_win('ings', PRIMARY_BUDGET):.1f}</b>,
<b>SMAS {mean_win('smas', PRIMARY_BUDGET):.1f}</b> vs TPE {mean_win('tpe', PRIMARY_BUDGET):.1f},
Greedy {mean_win('greedy', PRIMARY_BUDGET):.1f}, Random {mean_win('random', PRIMARY_BUDGET):.1f}.
But the per-dataset W/T/L is now nearly even rather than dominant:
SMAS vs TPE = {smas_vs_tpe_wtl[PRIMARY_BUDGET]}, INGS vs TPE = {ings_vs_tpe_wtl[PRIMARY_BUDGET]}
at N={PRIMARY_BUDGET}. The only paired Wilcoxon that crosses α=0.05 is
<b>INGS vs TPE at N=20 (p={ings_paired_p[20]:.3f})</b>; SMAS never reaches
significance (smallest p={min(smas_paired_p.values()):.3f}). The "SMAS beats TPE
5W/1T/2L at N=30" headline from Exp 9 (8 datasets) does <i>not</i> generalize to
the wider set — it was driven by a favorable subset.
</div>

<h2>2. Aggregate convergence (mean win score across {len(ds_list)} datasets)</h2>
<p><img class="agg" src="exp10_aggregate.png" alt="aggregate convergence"></p>

<h2>3. Win score at N={PRIMARY_BUDGET} per dataset</h2>
<p>Cell color: vs <b>TPE</b> (green = significantly higher / win, yellow = tie, red = lower).
<b>Bold</b> = best per dataset. Win = 100·(1 − (d2h(best) − d*)/(d₀ − d*)); higher is better.</p>
{win_table}

<h2>4. Mean win score across all {len(ds_list)} datasets, per method × budget</h2>
{agg_table}

<h2>5. Paired Wilcoxon (one-sided, method &gt; TPE) on per-dataset win-score means</h2>
{paired_table}

<h2>6. Per-dataset Mann-Whitney W/T/L on win scores</h2>
{wtl_table}

<h2>7. Per-dataset convergence curves</h2>
<p class="muted">y = mean win score across 50 repeats, SEM error bars. Dotted green = oracle (100). Each plot ~32% width; full-size by clicking.</p>
<div>{per_ds_imgs}</div>

<h2>8. Discussion</h2>
<p>
<b>The smoothness signal is real but the TPE-beating claim weakens with wider coverage.</b>
Both INGS and SMAS comfortably beat the no-smoothness Greedy ablation at every budget on
aggregate, confirming the smoothness machinery (RBF + anti-Laplacian + UCB) carries
real information beyond a naïve surrogate. But where Exp 9 reported SMAS beating TPE
5W/1T/2L at N=30 on 8 datasets, on the 28-dataset set the same comparison is
{smas_vs_tpe_wtl[PRIMARY_BUDGET]} — a near-tie. INGS does slightly better
({ings_vs_tpe_wtl[PRIMARY_BUDGET]}) and is the only method to reach a significant paired
Wilcoxon vs TPE at any budget (N=20, p={ings_paired_p[20]:.3f}). The honest reading: the
Exp-9 dataset selection was favorable; the smoothness-aware methods are
<i>competitive with</i>, not dominant over, TPE in this regime.
</p>
<p>
<b>INGS ≥ SMAS holds at scale.</b> Across both N and breadth of datasets, INGS edges out
SMAS slightly (mean win at N={PRIMARY_BUDGET}: INGS {mean_win('ings', PRIMARY_BUDGET):.1f} vs SMAS
{mean_win('smas', PRIMARY_BUDGET):.1f}; INGS is the only one significant vs TPE).
This reinforces Exp 9b's finding that the SMAS β-controller doesn't earn its complexity —
on a 3.5× larger evaluation set, the simpler INGS is at least as good.
</p>
<p>
<b>Where TPE wins.</b> Looking at the per-dataset table, TPE beats the surrogate methods
most clearly on (a) the largest pools (SS-X at 86k, SS-W at 66k, Health at 120k), and
(b) the highest dimensions (SQL at 39 dims, Health at multiple objectives). Both are
regimes where the RBF surrogate over a small set of evaluated points predicts poorly:
either because the pool is too vast to cover with 50 samples, or because the input space
is too high-dim for an RBF in only ~50 points to interpolate meaningfully. TPE's KDE,
which doesn't try to model the response surface globally, scales better here.
</p>
<p>
<b>Where INGS/SMAS win.</b> The clearest gains are on the smaller, lower-dim datasets
(X264, several SS-* in the 3–8 dim, 200–7000 pool range) — exactly the regime where the
RBF can interpolate well from a few dozen observations. This is consistent with the
paper's smoothness hypothesis: the smoothness-aware acquisition exploits a property
(low landscape curvature) that's most visible when the surrogate has a fighting chance
of being accurate.
</p>
<p>
<b>Random is competitive.</b> The mean win for Random matches or exceeds TPE at every
budget on this set (N={PRIMARY_BUDGET}: Random {mean_win('random', PRIMARY_BUDGET):.1f},
TPE {mean_win('tpe', PRIMARY_BUDGET):.1f}). This isn't a TPE bug — it's a well-known
property of pool-based config tuning that random is a strong baseline, particularly
when the snap-to-nearest-row mapping disrupts TPE's density model.
</p>

</body></html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prototype", action="store_true",
                    help="Run only on 3 small datasets for a quick smoke check")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "exp10_raw.json"

    if args.report_only:
        with open(raw_path) as f:
            payload = json.load(f)
        payload["results"] = {
            ds: {m: {int(b): v for b, v in bv.items()} for m, bv in mv.items()}
            for ds, mv in payload["results"].items()
        }
        make_plots(payload)
        (OUT_DIR / "exp10_wide.html").write_text(gen_report(payload, payload.get("runtime", 0.0)))
        print("Regenerated Exp 10 report from cached JSON.")
        return

    if args.prototype:
        datasets = ["SS-A", "SS-D", "Apache_AllMeasurements"]
    else:
        datasets = SS_NAMES + NAMED_CONFIG + [HEALTH_NAME]
    n_repeats = args.repeats or (3 if args.prototype else 50)

    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Repeats: {n_repeats}")

    t0 = time.time()
    payload = run(datasets, n_repeats)
    payload["runtime"] = time.time() - t0

    with open(raw_path, "w") as f:
        json.dump(payload, f, indent=2)
    make_plots(payload)
    (OUT_DIR / "exp10_wide.html").write_text(gen_report(payload, payload["runtime"]))
    print(f"\nReport → {OUT_DIR}/exp10_wide.html  ({payload['runtime']/60:.1f} min)")


if __name__ == "__main__":
    main()
