"""
Utility functions for handling plots in tests.
"""

import os
import matplotlib.pyplot as plt


def save_or_show_plot(filename, dpi=150, bbox_inches="tight", show_plots=None):
    """
    Save plot to tests/plots directory or show it based on environment variable.

    Parameters
    ----------
    filename : str
        Name of the file to save (without path)
    dpi : int, default=150
        DPI for saved image
    bbox_inches : str, default='tight'
        Bbox setting for saved image
    show_plots : bool, optional
        Override environment variable. If None, uses SHOW_PLOTS env var.
    """
    if show_plots is None:
        show_plots = os.getenv("SHOW_PLOTS", "0") == "1"

    if show_plots:
        plt.show()
    else:
        # Create plots directory if it doesn't exist
        current_dir = os.path.dirname(__file__)
        plots_dir = os.path.join(current_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plot_path = os.path.join(plots_dir, filename)
        plt.savefig(plot_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Plot saved to: {plot_path}")

    plt.close()  # Always close the figure to free memory
