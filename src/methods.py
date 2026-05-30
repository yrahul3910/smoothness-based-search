"""HPO methods for Experiments 6 and 7.

Provides:
  * fair_tpe / peek_tpe : TPE via Optuna (val proxy vs test proxy)
  * fair_bohb / peek_bohb : real BOHB via hpbandster with subsample fidelity
  * smas                : Smoothness-Augmented INGS (val proxy)
  * ings_full           : INGS-Full baseline (val proxy) — copied from exp_ings_improved
  * random_search       : Random search baseline (val proxy)
  * peek_random         : Random search picking by test R² (matches Exp5 random_30)
"""

from __future__ import annotations

import logging
import os
import warnings
from copy import deepcopy
from typing import Any

import numpy as np
import optuna
from raise_utils.data import Data
from scipy.interpolate import RBFInterpolator
from scipy.stats import qmc, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.smoothness_trees import get_tree_smoothness
from src.util import get_random_hyperparams

optuna.logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

# ── Shared HPO config ──────────────────────────────────────────────────────────

HPO_SPACE: dict[str, list[Any]] = {
    "criterion": ["friedman_mse", "absolute_error", "squared_error"],
    "max_depth": list(range(2, 16)),
    "min_samples_split": [2, 4, 8, 16, 32],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
}

N_BUDGET_DEFAULT = 30
N_DIMS = len(HPO_SPACE)
METRICS = [r2_score, mean_squared_error]
FLAT_EPS = 1e-3
N_CANDIDATES = 40
UCB_KAPPA = 0.1
CV_FOLDS = 2
MULTIFOLD_TOP_K = 3


def _make_model(cfg: dict) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(**cfg)


def _eval_test(model, data: Data) -> list[float]:
    preds = model.predict(data.x_test)
    return [float(fn(data.y_test, preds)) for fn in METRICS]


def _train_eval_test(cfg: dict, data: Data) -> list[float]:
    m = _make_model(cfg)
    m.fit(data.x_train, data.y_train)
    return _eval_test(m, data)


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


def _lhs_configs(n: int) -> list[dict]:
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


def _proxy_single(cfg, x_tr, x_val, y_tr, y_val) -> tuple[float, float]:
    """Train on x_tr, return (val R², β-smoothness of the trained tree)."""
    m = _make_model(cfg)
    m.fit(x_tr, y_tr)
    r2 = float(r2_score(y_val, m.predict(x_val)))
    beta = float(get_tree_smoothness(m))
    return r2, beta


def _proxy_cv(cfg, x_train, y_train, k=CV_FOLDS) -> tuple[float, float]:
    kf = KFold(n_splits=k, shuffle=True)
    r2s, betas = [], []
    for tr_idx, val_idx in kf.split(x_train):
        m = _make_model(cfg)
        m.fit(x_train[tr_idx], y_train[tr_idx])
        r2s.append(float(r2_score(y_train[val_idx], m.predict(x_train[val_idx]))))
        betas.append(float(get_tree_smoothness(m)))
    return float(np.mean(r2s)), float(np.mean(betas))


def _zscore(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


# ── INGS-Full baseline (carried over from exp5) ─────────────────────────────────


def ings_full(data: Data, n_budget: int = N_BUDGET_DEFAULT) -> list[float]:
    n_configs = n_budget // CV_FOLDS
    n_init = max(5, n_configs // 3)
    n_steps = n_configs - n_init

    cfgs = _lhs_configs(n_init)
    vecs = [_encode(c) for c in cfgs]
    proxies = [_proxy_cv(c, data.x_train, data.y_train)[0] for c in cfgs]

    for _ in range(n_steps):
        X_obs = np.array(vecs)
        y_obs = np.array(proxies)
        try:
            rbf = RBFInterpolator(X_obs, y_obs, kernel="linear", smoothing=1e-2)
        except Exception:
            cfg = get_random_hyperparams(HPO_SPACE)
            cfgs.append(cfg)
            vecs.append(_encode(cfg))
            proxies.append(_proxy_cv(cfg, data.x_train, data.y_train)[0])
            continue

        candidates = [get_random_hyperparams(HPO_SPACE) for _ in range(N_CANDIDATES)]
        cand_vecs = np.array([_encode(c) for c in candidates])
        rbf_preds = rbf(cand_vecs).flatten()

        # Anti-Laplacian
        scores = np.empty(len(candidates))
        for i, (c, pred) in enumerate(zip(candidates, rbf_preds, strict=False)):
            nv = _neighbor_vecs(c)
            if len(nv):
                scores[i] = 2.0 * pred - float(np.mean(rbf(nv).flatten()))
            else:
                scores[i] = pred

        # UCB
        dists = np.array([np.min(np.linalg.norm(X_obs - v, axis=1)) for v in cand_vecs])
        scores = scores + UCB_KAPPA * dists / np.sqrt(N_DIMS)

        if float(scores.max() - scores.min()) < FLAT_EPS:
            cfg_next = get_random_hyperparams(HPO_SPACE)
        else:
            cfg_next = candidates[int(np.argmax(scores))]
        cfgs.append(cfg_next)
        vecs.append(_encode(cfg_next))
        proxies.append(_proxy_cv(cfg_next, data.x_train, data.y_train)[0])

    # Multi-fold final selection
    top_k_idx = np.argsort(proxies)[-MULTIFOLD_TOP_K:]
    x_tr2, x_val2, y_tr2, y_val2 = train_test_split(
        data.x_train, data.y_train, test_size=0.2
    )
    reeval = [_proxy_single(cfgs[i], x_tr2, x_val2, y_tr2, y_val2)[0] for i in top_k_idx]
    best_cfg = cfgs[top_k_idx[int(np.argmax(reeval))]]
    return _train_eval_test(best_cfg, data)


# ── SMAS — Smoothness-Augmented INGS ───────────────────────────────────────────


def smas(
    data: Data,
    n_budget: int = N_BUDGET_DEFAULT,
    smooth_weight: float = 0.4,
    log_sign: list | None = None,
) -> list[float]:
    """Smoothness-Augmented INGS.

    Acquisition combines (z-scored) val-R² prediction and (z-scored) β prediction
    with adaptive sign and weight learned from the pilot:

        sign = +1 if Spearman(β, R²) ≥ 0 else -1   (which direction helps?)
        w_eff = smooth_weight · |ρ|                 (how strongly does β help?)
        score = (1 - w_eff) · z(R²_pred) + w_eff · sign · z(β_pred) + κ·exploration

    When the β-R² correlation is weak (|ρ|→0), the β term auto-dampens and SMAS
    falls back to INGS-Full behavior. When strong, β takes over more of the
    acquisition.
    """
    n_configs = n_budget // CV_FOLDS  # 2-fold CV proxy
    # Larger pilot than ings_full so Spearman is stable: ~half the budget.
    n_init = max(7, n_configs // 2)
    n_steps = n_configs - n_init

    cfgs = _lhs_configs(n_init)
    vecs = [_encode(c) for c in cfgs]
    r2s_b: list[tuple[float, float]] = [
        _proxy_cv(c, data.x_train, data.y_train) for c in cfgs
    ]
    proxies = [v[0] for v in r2s_b]
    betas = [v[1] for v in r2s_b]

    # ── Pilot: choose smoothness sign + adaptive weight from Spearman(β, R²) ───
    rho, _ = spearmanr(betas, proxies)
    if np.isnan(rho):
        rho = 0.0
    sign = +1.0 if rho >= 0 else -1.0
    eff_weight = smooth_weight * float(min(1.0, abs(rho)))
    if log_sign is not None:
        log_sign.append((sign, float(rho), eff_weight))

    for _ in range(n_steps):
        X_obs = np.array(vecs)
        y_obs = np.array(proxies)
        b_obs = np.array(betas)

        try:
            rbf_r2 = RBFInterpolator(X_obs, y_obs, kernel="linear", smoothing=1e-2)
            rbf_b = RBFInterpolator(X_obs, b_obs, kernel="linear", smoothing=1e-2)
        except Exception:
            cfg = get_random_hyperparams(HPO_SPACE)
            r2v, bv = _proxy_cv(cfg, data.x_train, data.y_train)
            cfgs.append(cfg)
            vecs.append(_encode(cfg))
            proxies.append(r2v)
            betas.append(bv)
            continue

        candidates = [get_random_hyperparams(HPO_SPACE) for _ in range(N_CANDIDATES)]
        cand_vecs = np.array([_encode(c) for c in candidates])
        r2_pred = rbf_r2(cand_vecs).flatten()
        b_pred = rbf_b(cand_vecs).flatten()

        # Anti-Laplacian correction on R² surface (preserved from INGS)
        r2_corrected = np.empty(len(candidates))
        for i, (c, pred) in enumerate(zip(candidates, r2_pred, strict=False)):
            nv = _neighbor_vecs(c)
            if len(nv):
                r2_corrected[i] = 2.0 * pred - float(np.mean(rbf_r2(nv).flatten()))
            else:
                r2_corrected[i] = pred

        # Composite acquisition: z-scored sum with adaptive weight
        r2_z = _zscore(r2_corrected)
        b_z = _zscore(b_pred)
        scores = (1.0 - eff_weight) * r2_z + eff_weight * sign * b_z

        # UCB exploration
        dists = np.array([np.min(np.linalg.norm(X_obs - v, axis=1)) for v in cand_vecs])
        scores = scores + UCB_KAPPA * dists / np.sqrt(N_DIMS)

        if float(scores.max() - scores.min()) < FLAT_EPS:
            cfg_next = get_random_hyperparams(HPO_SPACE)
        else:
            cfg_next = candidates[int(np.argmax(scores))]

        r2v, bv = _proxy_cv(cfg_next, data.x_train, data.y_train)
        cfgs.append(cfg_next)
        vecs.append(_encode(cfg_next))
        proxies.append(r2v)
        betas.append(bv)

    # Multi-fold final selection (fresh split)
    top_k_idx = np.argsort(proxies)[-MULTIFOLD_TOP_K:]
    x_tr2, x_val2, y_tr2, y_val2 = train_test_split(
        data.x_train, data.y_train, test_size=0.2
    )
    reeval = [
        _proxy_single(cfgs[i], x_tr2, x_val2, y_tr2, y_val2)[0] for i in top_k_idx
    ]
    best_cfg = cfgs[top_k_idx[int(np.argmax(reeval))]]
    return _train_eval_test(best_cfg, data)


# ── SMAS-Screen — SMOOTHIE-style two-stage low-fidelity screen ──────────────────


def smas_screen(
    data: Data,
    n_screen: int = 80,
    screen_frac: float = 0.10,
    screen_floor: int = 50,
    top_m: int = 8,
    smooth_weight: float = 0.4,
    log: dict | None = None,
) -> list[float]:
    """SMOOTHIE-style budget: cheap low-fidelity screen over many configs, then
    fully train only the top-m.

    Maps the original SMOOTHIE recipe (30–50 one-epoch cycles → β estimate →
    fully train best 5–10) to gradient-free trees. The "one-epoch" proxy is a
    low-fidelity fit on a row-subsample of the training pool; β is computable
    from that cheap tree. Ranking uses the SMAS composite (val-R² + adaptive β),
    since pure-β screening loses for trees (Exp 3).

    Budget (effective full-trains):
        n_screen · (n_sub / |pool|)   cheap screens
      + top_m                          full exploit trains
      + 1                              final refit on full train
    """
    x_pool, x_val, y_pool, y_val = train_test_split(
        data.x_train, data.y_train, test_size=0.2
    )
    n_pool = len(x_pool)
    n_sub = min(n_pool, max(screen_floor, int(screen_frac * n_pool)))

    # ── Stage 1: cheap low-fidelity screen ─────────────────────────────────────
    cfgs = _lhs_configs(n_screen)
    val_r2s, betas = [], []
    for cfg in cfgs:
        idx = np.random.choice(n_pool, n_sub, replace=False)
        m = _make_model(cfg)
        m.fit(x_pool[idx], y_pool[idx])
        val_r2s.append(float(r2_score(y_val, m.predict(x_val))))
        betas.append(float(get_tree_smoothness(m)))

    # Pilot direction + adaptive weight (same scheme as smas)
    rho, _ = spearmanr(betas, val_r2s)
    if np.isnan(rho):
        rho = 0.0
    sign = +1.0 if rho >= 0 else -1.0
    eff_weight = smooth_weight * float(min(1.0, abs(rho)))

    scores = (1.0 - eff_weight) * _zscore(np.array(val_r2s)) \
        + eff_weight * sign * _zscore(np.array(betas))
    top_idx = np.argsort(scores)[-top_m:]

    # ── Stage 2: full-fidelity train of survivors; pick best on val ────────────
    best_cfg, best_v = cfgs[int(top_idx[-1])], -np.inf
    for i in top_idx:
        m = _make_model(cfgs[i])
        m.fit(x_pool, y_pool)
        v = float(r2_score(y_val, m.predict(x_val)))
        if v > best_v:
            best_v, best_cfg = v, cfgs[i]

    if log is not None:
        log.setdefault("n_sub", []).append(n_sub)
        log.setdefault("n_pool", []).append(n_pool)
        log.setdefault("rho", []).append(float(rho))
        log.setdefault("eff_trains", []).append(
            n_screen * (n_sub / n_pool) + top_m + 1
        )

    # Final refit on full training set
    return _train_eval_test(best_cfg, data)


# ── Random search baselines ────────────────────────────────────────────────────


def fair_random(data: Data, n_budget: int = N_BUDGET_DEFAULT) -> list[float]:
    """Random search picking by val R² (no test peek during search)."""
    x_tr, x_val, y_tr, y_val = train_test_split(
        data.x_train, data.y_train, test_size=0.2
    )
    best_cfg, best_val = None, -np.inf
    for _ in range(n_budget):
        cfg = get_random_hyperparams(HPO_SPACE)
        m = _make_model(cfg)
        m.fit(x_tr, y_tr)
        v = float(r2_score(y_val, m.predict(x_val)))
        if v > best_val:
            best_val, best_cfg = v, cfg
    return _train_eval_test(best_cfg, data)


def peek_random(data: Data, n_budget: int = N_BUDGET_DEFAULT) -> list[float]:
    """Random search picking by test R² (matches exp5 random_30 — test peeking)."""
    best = [float("-inf"), float("inf")]
    for _ in range(n_budget):
        cfg = get_random_hyperparams(HPO_SPACE)
        m = _make_model(cfg)
        m.fit(data.x_train, data.y_train)
        res = _eval_test(m, data)
        if res[0] > best[0]:
            best = res
    return best


# ── TPE baselines (Optuna) ──────────────────────────────────────────────────────


def _suggest(trial):
    return {
        "criterion": trial.suggest_categorical("criterion", HPO_SPACE["criterion"]),
        "max_depth": trial.suggest_categorical("max_depth", HPO_SPACE["max_depth"]),
        "min_samples_split": trial.suggest_categorical(
            "min_samples_split", HPO_SPACE["min_samples_split"]
        ),
        "min_samples_leaf": trial.suggest_categorical(
            "min_samples_leaf", HPO_SPACE["min_samples_leaf"]
        ),
        "max_features": trial.suggest_categorical(
            "max_features", HPO_SPACE["max_features"]
        ),
    }


def fair_tpe(
    data: Data, n_budget: int = N_BUDGET_DEFAULT, n_startup: int = 10
) -> list[float]:
    """TPE on (x_train_split, x_val). No test access during search."""
    x_tr, x_val, y_tr, y_val = train_test_split(
        data.x_train, data.y_train, test_size=0.2
    )

    def obj(trial):
        cfg = _suggest(trial)
        m = _make_model(cfg)
        m.fit(x_tr, y_tr)
        return float(r2_score(y_val, m.predict(x_val)))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup, seed=None),
    )
    study.optimize(obj, n_trials=n_budget, show_progress_bar=False)
    return _train_eval_test(study.best_params, data)


def peek_tpe(data: Data, n_budget: int = N_BUDGET_DEFAULT) -> list[float]:
    """TPE objective = test R². Test peeking (matches exp5 tpe_30)."""
    def obj(trial):
        cfg = _suggest(trial)
        m = _make_model(cfg)
        m.fit(data.x_train, data.y_train)
        return float(r2_score(data.y_test, m.predict(data.x_test)))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=None),
    )
    study.optimize(obj, n_trials=n_budget, show_progress_bar=False)
    return _train_eval_test(study.best_params, data)


# ── Real BOHB via hpbandster ───────────────────────────────────────────────────
# Imported lazily to avoid hpbandster's startup cost on every import.

_BOHB_LOGGERS_SILENCED = False


def _register_numpy_with_serpent():
    """Pyro4/serpent (used by hpbandster) can't serialize numpy scalars by default.
    ConfigSpace returns numpy.str_/int_/float_ for sampled configs, which crashes the
    worker dispatch. Register replacers that fall back to plain Python scalars."""
    import serpent

    def _np_to_python(obj, serializer, out, level):
        if isinstance(obj, np.integer):
            serializer._serialize(int(obj), out, level)
        elif isinstance(obj, np.floating):
            serializer._serialize(float(obj), out, level)
        else:
            serializer._serialize(str(obj), out, level)

    for tp in (np.str_, np.bytes_, np.bool_, np.integer, np.floating, np.ndarray):
        try:
            serpent.register_class(tp, _np_to_python)
        except Exception:
            pass


def _silence_hpbandster_logs():
    global _BOHB_LOGGERS_SILENCED
    if _BOHB_LOGGERS_SILENCED:
        return
    _register_numpy_with_serpent()
    for name in (
        "hpbandster",
        "hpbandster.optimizers.bohb",
        "hpbandster.core.master",
        "hpbandster.core.dispatcher",
        "hpbandster.core.nameserver",
        "hpbandster.optimizers.iterations.successivehalving",
        "Pyro4",
        "Pyro4.core",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    _BOHB_LOGGERS_SILENCED = True


def _bohb_config_space():
    import ConfigSpace as CS
    import ConfigSpace.hyperparameters as CSH

    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(
        CSH.CategoricalHyperparameter("criterion", HPO_SPACE["criterion"])
    )
    cs.add_hyperparameter(
        CSH.UniformIntegerHyperparameter("max_depth", lower=2, upper=15)
    )
    cs.add_hyperparameter(
        CSH.CategoricalHyperparameter(
            "min_samples_split", [str(v) for v in HPO_SPACE["min_samples_split"]]
        )
    )
    cs.add_hyperparameter(
        CSH.CategoricalHyperparameter(
            "min_samples_leaf", [str(v) for v in HPO_SPACE["min_samples_leaf"]]
        )
    )
    cs.add_hyperparameter(
        CSH.CategoricalHyperparameter("max_features", ["sqrt", "log2", "none"])
    )
    return cs


def _decode_bohb_config(cfg: dict) -> dict:
    out = dict(cfg)
    out["min_samples_split"] = int(out["min_samples_split"])
    out["min_samples_leaf"] = int(out["min_samples_leaf"])
    if out["max_features"] == "none":
        out["max_features"] = None
    return out


def _bohb_run(
    x_tr: np.ndarray,
    x_eval: np.ndarray,
    y_tr: np.ndarray,
    y_eval: np.ndarray,
    n_iterations: int,
    min_budget: float = 1.0 / 9.0,
    max_budget: float = 1.0,
    eta: int = 3,
) -> tuple[dict, int]:
    """Run BOHB. Returns (best config, total compute calls)."""
    _silence_hpbandster_logs()
    import hpbandster.core.nameserver as hpns
    from hpbandster.core.worker import Worker
    from hpbandster.optimizers import BOHB

    run_id = f"bohb_{os.getpid()}_{np.random.randint(1_000_000)}"

    class TreeWorker(Worker):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.n_calls = 0

        def compute(self, config, budget, **kwargs):
            self.n_calls += 1
            cfg = _decode_bohb_config(config)
            n = max(2, int(len(x_tr) * float(budget)))
            idx = np.random.choice(len(x_tr), n, replace=False)
            m = DecisionTreeRegressor(**cfg)
            m.fit(x_tr[idx], y_tr[idx])
            r2 = float(r2_score(y_eval, m.predict(x_eval)))
            return {"loss": -r2, "info": {}}

    ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=0)
    ns_host, ns_port = ns.start()
    worker = TreeWorker(
        nameserver=ns_host, nameserver_port=ns_port, run_id=run_id
    )
    worker.run(background=True)

    bohb = BOHB(
        configspace=_bohb_config_space(),
        run_id=run_id,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=min_budget,
        max_budget=max_budget,
        eta=eta,
    )
    try:
        res = bohb.run(n_iterations=n_iterations, min_n_workers=1)
    finally:
        bohb.shutdown(shutdown_workers=True)
        ns.shutdown()

    inc_id = res.get_incumbent_id()
    id2cfg = res.get_id2config_mapping()
    if inc_id is None or inc_id not in id2cfg:
        # fallback: pick best logged run
        runs = res.get_all_runs()
        if not runs:
            return _decode_bohb_config(
                {k: v[0] if isinstance(v, list) else v for k, v in HPO_SPACE.items()}
            ), worker.n_calls
        best = min(runs, key=lambda r: r.loss if r.loss is not None else 1e9)
        inc_id = best.config_id
    best_cfg = _decode_bohb_config(id2cfg[inc_id]["config"])
    return best_cfg, worker.n_calls


def fair_bohb(
    data: Data, n_budget: int = N_BUDGET_DEFAULT, n_iterations: int = 3
) -> list[float]:
    """BOHB on (x_train_split, x_val) — no test access during search."""
    x_tr, x_val, y_tr, y_val = train_test_split(
        data.x_train, data.y_train, test_size=0.2
    )
    best_cfg, _ = _bohb_run(
        x_tr, x_val, y_tr, y_val, n_iterations=n_iterations
    )
    return _train_eval_test(best_cfg, data)


def peek_bohb(
    data: Data, n_budget: int = N_BUDGET_DEFAULT, n_iterations: int = 3
) -> list[float]:
    """BOHB on (x_train, x_test) — test peeking, comparable to exp5 bohb_30."""
    best_cfg, _ = _bohb_run(
        data.x_train,
        data.x_test,
        data.y_train,
        data.y_test,
        n_iterations=n_iterations,
    )
    return _train_eval_test(best_cfg, data)
