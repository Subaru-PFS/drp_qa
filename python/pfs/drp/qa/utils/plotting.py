from typing import Iterable, Optional

import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

div_palette = plt.cm.RdBu_r.with_extremes(over="magenta", under="cyan", bad="lime")
detector_palette = {"b": "tab:blue", "r": "tab:red", "n": "goldenrod", "m": "tab:pink"}
description_palette = {
    "Trace": "black",
    "ArI": "tab:blue",
    "CdI": "tab:orange",
    "HgI": "tab:green",
    "KrI": "tab:red",
    "NeI": "tab:purple",
    "XeI": "tab:brown",
    "O2": "tab:pink",
    "OH": "tab:gray",
    "OI": "tab:olive",
    "NaI": "tab:cyan",
}
spectrograph_plot_markers = {1: "s", 2: "o", 3: "X", 4: "P"}


def scatterplot_with_outliers(
    data: pd.DataFrame,
    X: str,
    Y: str,
    hue: str = "status_name",
    ymin: float = -0.1,
    ymax: float = 0.1,
    palette: Optional[dict] = None,
    ax: Optional[Axes] = None,
    refline: Optional[Iterable[float]] = None,
    vertical: bool = False,
    rasterized: bool = False,
    showUnusedOutliers: bool = False,
) -> Axes:
    """Make a scatterplot with outliers marked.

    The plot can be rendered vertically, but you should still use the `X` and
    `Y` parameters as if it were horizontal.

    Parameters
    ----------
    data : `pandas.DataFrame`
        The data.
    X : `str`
        The x column.
    Y : `str`
        The y column.
    hue : `str`, optional
        The hue column. Default is ``'status_name'``.
    ymin : `float`, optional
        The minimum y value. Default is -0.1.
    ymax : `float`, optional
        The maximum y value. Default is 0.1.
    palette : `dict`, optional
        The palette. Default is ``None``.
    ax : `matplotlib.axes.Axes`, optional
        The axes. Default is ``None``.
    refline : `float`, optional
        Reference lines to plot. Default is ``None``.
    vertical : `bool`, optional
        Is the plot vertical? Default is ``False``.
    rasterized : `bool`, optional
        Rasterize the plot? Default is ``False``.
    showUnusedOutliers : `bool`, optional
        If unused datapoints should be included in plot. Default is ``False``.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        A scatter plot with the outliers marked.
    """
    # Main plot.
    sb.scatterplot(
        data=data,
        x=X,
        y=Y,
        hue=hue,
        hue_order=["isReserved", "isUsed"] if hue == "status" else None,
        s=20,
        ec="k",
        style="isOutlier",
        markers={True: "X", False: "."},
        zorder=100,
        palette=palette,
        rasterized=rasterized,
        ax=ax,
    )

    if showUnusedOutliers is True:
        # Positive outliers.
        pos = data.query(f"{X if vertical else Y} >= @ymax").copy()
        pos[X if vertical else Y] = ymax
        marker = "X" if vertical is True else "X"
        sb.scatterplot(
            data=pos,
            x=X,
            y=Y,
            hue=hue,
            palette=palette,
            legend=False,
            marker=marker,
            ec="k",
            lw=0.5,
            s=50,
            alpha=0.5,
            clip_on=False,
            zorder=100,
            ax=ax,
        )

        # Negative outliers.
        neg = data.query(f"{X if vertical else Y} <= @ymin").copy()
        neg[X if vertical else Y] = ymin
        marker = "X" if vertical is True else "X"
        sb.scatterplot(
            data=neg,
            x=X,
            y=Y,
            hue=hue,
            palette=palette,
            legend=False,
            marker=marker,
            ec="k",
            lw=0.5,
            s=50,
            alpha=0.5,
            clip_on=False,
            zorder=100,
            ax=ax,
        )

    # Reference line.
    if isinstance(refline, (float, int)):
        if vertical:
            ax.axvline(refline, color="k", ls="--", alpha=0.5, zorder=-100)
        else:
            ax.axhline(refline, color="k", ls="--", alpha=0.5, zorder=-100)

    if vertical is True:
        ax.set_xlim(ymin, ymax)
    else:
        ax.set_ylim(ymin, ymax)

    ax.grid(True, alpha=0.15)

    return ax
