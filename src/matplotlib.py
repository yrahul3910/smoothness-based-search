"""Functions to interact with matplotlib."""

import math
import os

import numpy as np
import polars as pl
from scipy.interpolate import griddata
from sklearn.decomposition import PCA

import matplotlib as mpl
from matplotlib import pyplot as plt


def setup_matplotlib() -> bool:
    """Set up matplotlib backend based on environment."""
    if os.environ.get("DISPLAY") is None:
        mpl.use("Agg")
        return False
    try:
        mpl.use("TkAgg")
    except ImportError:
        mpl.use("Agg")
        return False
    else:
        return True


def create_surface_data(data: pl.DataFrame):
    """Load and process data for surface plotting."""
    columns = data.columns
    x_cols = [col for col in columns if not col.endswith(("-", "+"))]
    y_cols = [col for col in columns if col.endswith(("-", "+"))]

    # PCA transformation
    x_pca = PCA(n_components=2).fit_transform(data.select(x_cols))
    y = data.select(
        pl.col(col) if col.endswith("-") else 1 - pl.col(col) for col in y_cols
    )
    y = y.map_rows(lambda row: math.sqrt(sum(val**2 for val in row)))

    return x_pca, y.to_numpy().ravel()


def plot_smooth_surface(
    x_pca: np.ndarray,
    z_values: np.ndarray,
    filename: str,
    title: str,
) -> None:
    """Create a smooth interpolated surface plot."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    x_points = x_pca[:, 0]
    y_points = x_pca[:, 1]

    xi = np.linspace(x_points.min(), x_points.max(), 80)
    yi = np.linspace(y_points.min(), y_points.max(), 80)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata(
        (x_points, y_points),
        z_values,
        (Xi, Yi),
        method="cubic",
        fill_value=0,
    )

    surf = ax.plot_surface(
        Xi,
        Yi,
        Zi,
        cmap="viridis",
        alpha=0.8,
        linewidth=0,
        antialiased=True,
        shade=True,
    )
    ax.contour(Xi, Yi, Zi, zdir="z", offset=Zi.min(), cmap="viridis", alpha=0.4)
    fig.colorbar(surf, label="Loss Value", shrink=0.5, aspect=5)
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.set_zlabel("Loss Value")
    ax.set_title(title)
    ax.view_init(elev=25, azim=45)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Smooth surface plot saved as '{filename}'")
    plt.close()


def plot_filled_contour_surface(x_pca, z_values, filename="plots/contour_surface.png"):
    """Create a filled contour surface plot."""
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    x_points = x_pca[:, 0]
    y_points = x_pca[:, 1]
    xi = np.linspace(x_points.min(), x_points.max(), 60)
    yi = np.linspace(y_points.min(), y_points.max(), 60)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(
        (x_points, y_points),
        z_values,
        (Xi, Yi),
        method="cubic",
        fill_value=0,
    )
    surf = ax1.plot_surface(Xi, Yi, Zi, cmap="plasma", alpha=0.7, antialiased=True)
    ax1.set_xlabel("First Principal Component")
    ax1.set_ylabel("Second Principal Component")
    ax1.set_zlabel("Loss Value")
    ax1.set_title("3D Surface")
    ax1.view_init(elev=35, azim=45)
    contourf = ax2.contourf(Xi, Yi, Zi, levels=20, cmap="plasma", alpha=0.8)
    contour_lines = ax2.contour(
        Xi,
        Yi,
        Zi,
        levels=20,
        colors="black",
        alpha=0.3,
        linewidths=0.5,
    )
    ax2.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")
    ax2.set_xlabel("First Principal Component")
    ax2.set_ylabel("Second Principal Component")
    ax2.set_title("2D Contour Map")
    ax2.set_aspect("equal")
    fig.colorbar(surf, ax=ax1, label="Loss Value", shrink=0.5, aspect=5)
    fig.colorbar(contourf, ax=ax2, label="Loss Value", shrink=0.8, aspect=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Combined surface and contour plot saved as '{filename}'")
    plt.close()


def plot_gradient_surface(x_pca, z_values, filename="gradient_surface.png"):
    """Create a surface plot with gradient visualization."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    x_points = x_pca[:, 0]
    y_points = x_pca[:, 1]
    xi = np.linspace(x_points.min(), x_points.max(), 50)
    yi = np.linspace(y_points.min(), y_points.max(), 50)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(
        (x_points, y_points),
        z_values,
        (Xi, Yi),
        method="cubic",
        fill_value=0,
    )
    dy, dx = np.gradient(Zi)
    surf = ax.plot_surface(
        Xi,
        Yi,
        Zi,
        facecolors=plt.cm.coolwarm(np.sqrt(dx**2 + dy**2)),
        alpha=0.8,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax.scatter(
        x_points[::100],
        y_points[::100],
        z_values[::100],
        c="red",
        s=20,
        alpha=0.6,
        label="Data Points",
    )
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.set_zlabel("Loss Value")
    ax.set_title("Gradient-Colored Surface with Data Points")
    ax.view_init(elev=30, azim=45)
    ax.legend()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Gradient surface plot saved as '{filename}'")
    plt.close()
