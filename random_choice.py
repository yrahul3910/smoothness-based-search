"""HPO using random search."""

from sklearn.metrics import mean_squared_error, r2_score

from src.experiment import run_experiment

dirs = ["./data/optimize/process/", "./data/hpo/", "./data/config/"]

METRICS = [r2_score, mean_squared_error]
N_REPEATS = 1

experiment_results = run_experiment(
    experiment_fn=None,
    dirs=dirs,
    n_repeats=N_REPEATS,
    n_datasets=1,
)
print(experiment_results.mean_results)
