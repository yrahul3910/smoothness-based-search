"""HPO using smoothness."""

from raise_utils.data import Data
from scipy.spatial import KDTree
from sklearn.metrics import mean_squared_error, r2_score

from src.experiment import DEFAULT_HPO_SPACE, HPOSpace, MetricFn, run_experiment
from src.util import get_random_hyperparams

dirs = ["./data/optimize/process/", "./data/hpo/", "./data/config/"]

METRICS = [r2_score, mean_squared_error]
N_REPEATS = 1
HPO_SPACE = DEFAULT_HPO_SPACE


def interpolation_search(
    data: Data,
    hpo_space: HPOSpace,
    learners: dict[str, type],
    metrics: list[MetricFn],
) -> list[float]:
    """Run an interpolation-based hyper-parameter search.

    Args:
        data (Data): The original data to use for training and testing.
        hpo_space (HPOSpace): The hyperparameter space for each learner.
        learners (dict[str, type]): The learners to optimize.
        metrics (list[MetricFn]): The metrics to evaluate the learners.

    Returns:
        list[float]: The best metric scores achieved during the experiment.

    """
    BUDGET = 50
    THRESHOLD = 0.05
    INITIAL_POINTS = 10

    used_budget = 0

    sampled_points = [get_random_hyperparams(HPO_SPACE) for _ in range(INITIAL_POINTS)]

    best_metrics = [float("-inf")] * len(metrics)
    best_config = None

    while used_budget < BUDGET:
        # Find a point worth exploring
        while True:
            point = get_random_hyperparams(hpo_space)
            # TODO 3: Compute beta here, maybe create a dataclass to hold values

            tree = KDTree(data.x_train, leafsize=5)
            cur_dist, cur_idx = tree.query(point)
            max_diff = cur_dist**2 / 8.0 * beta

            # TODO 4 (Optional): Also use descent lemma to compute `max_diff`.

            if max_diff >= THRESHOLD:
                break

        # TODO 2: Actually use the budget
        model = learners[learner_name](**point)
        model.fit(data.x_train, data.y_train)
        preds = model.predict(data.x_test)

        results = [fn(data.y_test, preds) for fn in metrics]

        if results > best_metrics:
            best_metrics = results.copy()
            best_config = point

        sampled_points.append(point)
        used_budget += 1

    return sampled_points


experiment_results = run_experiment(
    experiment_fn=None,
    hpo_space=HPO_SPACE,
    dirs=dirs,
    n_repeats=N_REPEATS,
    n_datasets=1,
)
print(experiment_results.mean_results)
