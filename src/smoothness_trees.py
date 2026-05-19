"""Smoothness computation for tree-based learners.

For trees, smoothness is computed via a center-difference approximation of the
negative log-likelihood (NLL) across internal nodes, following the formulation
proposed in Yedida & Menzies (2025) Section IX:

    β(node) = |ℓ(left) + ℓ(right) - 2·ℓ(parent)|
    ℓ(node)  = n_samples_in_node × impurity   (weighted MSE for regression)

This is a discrete analog of ∇²E ≈ (E(x+h) + E(x-h) - 2E(x)) / h²,
where the "step" h corresponds to one level of the tree.
"""

import numpy as np

TREE_LEAF = -1


def _single_tree_smoothness(tree_) -> float:
    """β-smoothness for one sklearn tree_ object (the underlying C struct)."""
    betas = []
    for i in range(tree_.node_count):
        left = tree_.children_left[i]
        if left == TREE_LEAF:
            continue
        right = tree_.children_right[i]
        wH_parent = tree_.n_node_samples[i] * tree_.impurity[i]
        wH_left = tree_.n_node_samples[left] * tree_.impurity[left]
        wH_right = tree_.n_node_samples[right] * tree_.impurity[right]
        betas.append(abs(wH_left + wH_right - 2 * wH_parent))
    return float(np.mean(betas)) if betas else 0.0


def get_tree_smoothness(model) -> float:
    """β-smoothness for a DecisionTreeRegressor or RandomForestRegressor."""
    if hasattr(model, "estimators_"):
        return float(np.mean([_single_tree_smoothness(t.tree_) for t in model.estimators_]))
    return _single_tree_smoothness(model.tree_)
