import numpy as np
import polars as pl
from scipy.interpolate import griddata


def load_data(file_path: str) -> pl.DataFrame:
    """Load data from a CSV file."""
    # infer_schema_length=0 forces polars to load every column as a string,
    # avoiding "inferred i64 but value 1.00E+05 doesn't fit" errors on the
    # SS/Health files. We then cast string→float below.
    df = pl.read_csv(file_path, infer_schema_length=0)
    # Strip whitespace from column names (some CSVs use "col1, col2" with spaces)
    df = df.rename({c: c.strip() for c in df.columns})
    # Try to cast string columns to float; drop those that genuinely contain text
    for col in df.select(pl.col(pl.Utf8)).columns:
        try:
            df = df.with_columns(pl.col(col).str.strip_chars().cast(pl.Float64))
        except Exception:
            df = df.drop(col)
    return df


def jitter_data_1d(
    x_pca: pl.DataFrame,
    z_values: np.ndarray,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Return the difference between the original and perturbed z values."""
    epsilon = 0.05
    x_perturbed = x_pca + epsilon

    # Use interpolation to estimate f(x + epsilon)
    # Interpolate to get f(x + epsilon) values at perturbed points
    z_perturbed = griddata(
        x_pca,
        z_values,
        x_perturbed,
        method="linear",
        fill_value=0,
    )

    # Compute the difference: f(x + epsilon) - f(x)
    return z_perturbed - z_values


def jitter_data_2d(
    x_pca: pl.DataFrame,
    z_values: np.ndarray,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Return an approximation of the local Hessian."""
    epsilon = 0.05
    x_perturbed_plus = x_pca + epsilon
    x_perturbed_minus = x_pca - epsilon

    z_perturbed_plus = griddata(
        x_pca,
        z_values,
        x_perturbed_plus,
        method="cubic",
        fill_value=0,
    )
    z_perturbed_minus = griddata(
        x_pca,
        z_values,
        x_perturbed_minus,
        method="cubic",
        fill_value=0,
    )

    return z_perturbed_minus + z_perturbed_plus - 2 * z_values


def approximated_local_function(
    x_pca: pl.DataFrame,
    z_values: np.ndarray,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Return the difference between the original and perturbed z values."""
    epsilon = 0.05
    x_perturbed_plus = x_pca + epsilon
    x_perturbed_minus = x_pca - epsilon

    z_perturbed_plus = griddata(
        x_pca,
        z_values,
        x_perturbed_plus,
        method="cubic",
        fill_value=0,
    )
    z_perturbed_minus = griddata(
        x_pca,
        z_values,
        x_perturbed_minus,
        method="cubic",
        fill_value=0,
    )

    return z_perturbed_minus + z_perturbed_plus - 2 * z_values
