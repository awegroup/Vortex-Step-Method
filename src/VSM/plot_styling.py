import matplotlib.pyplot as plt
from typing import Optional
from cycler import cycler

PALETTE = {
    "Black": "#000000",
    "Orange": "#E69F00",
    "Sky Blue": "#56B4E9",
    "Bluish Green": "#009E73",
    "Yellow": "#F0E442",
    "Blue": "#0072B2",
    "Vermillion": "#D55E00",
    "Reddish Purple": "#CC79A7",
}


def set_plot_style():
    """
    Set the default style for plots using LaTeX and custom color palette.

    Tips:
    - If you specify colors, they will still be used.
    - If you want to change the axis margins:
        1. try with ax.xlim and ax.ylim
        2. try by changing the 'axes.autolimit_mode' parameter to data
    - more?
    """

    # Define the color palette as a list of colors
    color_cycle = [
        PALETTE["Black"],
        PALETTE["Orange"],
        PALETTE["Sky Blue"],
        PALETTE["Bluish Green"],
        PALETTE["Yellow"],
        PALETTE["Blue"],
        PALETTE["Vermillion"],
        PALETTE["Reddish Purple"],
    ]

    # Apply Seaborn style and custom settings
    # plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            ## Axes settings
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#C5C5C5",
            "axes.labelcolor": "black",
            "axes.autolimit_mode": "round_numbers",
            "axes.xmargin": 0,  # Remove extra margin
            "axes.ymargin": 0,  # Remove extra margin
            ## Grid settings
            "axes.grid": True,
            "axes.grid.axis": "both",
            "grid.alpha": 0.5,
            "grid.color": "#C5C5C5",
            "grid.linestyle": "-",
            "grid.linewidth": 1.0,
            ## Line settings
            "lines.linewidth": 1,
            "lines.markersize": 6,
            # "lines.color": "grey",,
            "figure.titlesize": 15,
            "pgf.texsystem": "pdflatex",  # Use pdflatex
            "pgf.rcfonts": False,
            "figure.figsize": (15, 5),  # Default figure size
            "axes.prop_cycle": cycler(
                "color", color_cycle
            ),  # Set the custom color cycle
            ## tick settings
            "xtick.color": "#C5C5C5",
            "ytick.color": "#C5C5C5",
            "xtick.labelcolor": "black",
            "ytick.labelcolor": "black",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.top": True,  # Show ticks on both sides
            "xtick.bottom": True,
            "ytick.left": True,
            "ytick.right": True,
            "xtick.direction": "in",  # Direction for x-axis ticks
            "ytick.direction": "in",  # Direction for y-axis ticks
            ## legend settings
            "legend.fontsize": 15,
        }
    )


def plot_on_ax(
    ax,
    x,
    y,
    label: str,
    color: str = None,
    linestyle: str = "-",
    marker: str = None,
    markersize: int = None,
    is_with_grid: bool = True,
    is_return_ax: bool = False,
    x_label: str = "X-axis",
    y_label: str = "Y-axis",
    is_with_x_label: bool = True,
    is_with_y_label: bool = True,
    is_with_x_ticks: bool = True,
    is_with_y_ticks: bool = True,
    title: str = None,
):
    """Plot data on a given axis."""

    # turning off the ticks
    if not is_with_x_ticks:
        ax.tick_params(labelbottom=False)
    if not is_with_y_ticks:
        ax.tick_params(left=False, right=False, labelleft=False)

    if is_with_grid:
        ax.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
        )
    else:
        ax.grid(False)

    if color is None:
        if marker is None:
            if markersize is None:
                ax.plot(x, y, label=label, linestyle=linestyle)
            else:
                ax.plot(x, y, label=label, linestyle=linestyle, markersize=markersize)
        else:
            ax.plot(x, y, label=label, linestyle=linestyle, marker=marker)
    else:
        if marker is None:
            ax.plot(x, y, label=label, color=color, linestyle=linestyle)
        else:
            if markersize is None:
                ax.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                )
            else:
                ax.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=markersize,
                )
    if title is not None:
        ax.set_title(title)

    if is_with_x_label:
        ax.set_xlabel(x_label)
    if is_with_y_label:
        ax.set_ylabel(y_label)
    if is_return_ax:
        return ax
