"""Loss surface visualization."""

import numpy as np
from scipy.interpolate import griddata

from matplotlib import pyplot as plt
from src.matplotlib import create_surface_data, setup_matplotlib


def main() -> None:
    """Generate and save a 3D surface plot of loss data with contours."""
    interactive = setup_matplotlib()
    x_pca, z_values = create_surface_data()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    x_points = x_pca[:, 0]
    y_points = x_pca[:, 1]
    z_points = z_values

    xi = np.linspace(x_points.min(), x_points.max(), 50)
    yi = np.linspace(y_points.min(), y_points.max(), 50)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    zi_grid = griddata(
        (x_points, y_points),
        z_points,
        (xi_grid, yi_grid),
        method="cubic",
        fill_value=0,
    )

    surf = ax.plot_surface(
        xi_grid,
        yi_grid,
        zi_grid,
        cmap="coolwarm",
        alpha=0.7,
        linewidth=0,
        antialiased=True,
    )
    ax.contour(
        xi_grid,
        yi_grid,
        zi_grid,
        zdir="z",
        offset=zi_grid.min(),
        cmap="coolwarm",
        alpha=0.6,
    )
    fig.colorbar(surf, label="Loss Value", shrink=0.5, aspect=5)
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.set_zlabel("Loss Value")
    ax.set_title("3D Surface Plot of Loss Data with Contours")
    ax.view_init(elev=30, azim=45)
    plt.savefig("plots/loss_visualization.png", dpi=300, bbox_inches="tight")
    print("Plot saved as 'loss_visualization.png'")
    if interactive:
        plt.show()
    else:
        print("Running in non-interactive mode - plot saved to file only")


if __name__ == "__main__":
    main()
