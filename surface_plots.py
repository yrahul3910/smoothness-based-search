"""Surface plot visualization."""

from pathlib import Path

from src.data import jitter_data_2d, load_data
from src.matplotlib import (
    create_surface_data,
    plot_filled_contour_surface,
    plot_gradient_surface,
    plot_smooth_surface,
    setup_matplotlib,
)


def main() -> None:
    """Generate all surface plot variations."""
    setup_matplotlib()

    print("Loading and processing data...")
    filename = "pom3a.csv"
    data = load_data("./data/optimize/process/" + filename)
    x_pca, z_values = create_surface_data(data)

    z_jittered = jitter_data_2d(x_pca, z_values)

    print("Creating surface plots...")
    data_name = filename.split(".")[0]

    if not (plot_path := Path(f"./plots/{data_name}")).exists():
        Path(plot_path).mkdir(parents=True)

    # Create raw plots
    plot_smooth_surface(
        x_pca,
        z_values,
        filename=f"{plot_path}/surface_raw.png",
        title=f"{data_name} Surface Plot",
    )
    plot_filled_contour_surface(
        x_pca,
        z_values,
        filename=f"{plot_path}/contour_raw.png",
    )
    plot_gradient_surface(x_pca, z_values)

    # Create jittered plots
    plot_smooth_surface(
        x_pca,
        z_jittered,
        filename=f"{plot_path}/surface_jittered_2d.png",
        title=f"{data_name} Jittered Surface Plot",
    )
    plot_filled_contour_surface(
        x_pca,
        z_jittered,
        filename=f"{plot_path}/contour_jittered_2d.png",
    )
    plot_gradient_surface(x_pca, z_jittered)

    print("\nAll surface plots have been generated!")


if __name__ == "__main__":
    main()
