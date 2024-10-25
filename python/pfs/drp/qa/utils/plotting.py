from typing import Iterable, Optional

import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

div_palette = plt.cm.RdBu_r.with_extremes(over="magenta", under="cyan", bad="lime")
detector_palette = {"b": "tab:blue", "r": "tab:red", "n": "tab:orange", "m": "tab:pink"}
description_palette = {
    "Trace": "#F664AF",
    "ArI": "tab:orange",
    "CdI,HgI": "tab:purple",
    "HgI": "tab:purple",
    "KrI": "tab:brown",
    "NeI": "tab:pink",
    "XeI": "tab:olive",
    "O2,OH": "tab:blue",
    "OH": "tab:blue",
    "OI": "tab:blue",
    "NaI,OI": "tab:blue",
}
spectrograph_plot_markers = {1: "s", 2: "o", 3: "X", 4: "P"}


def plot_detector_soften(detector_stats: pd.DataFrame) -> Figure:
    """Plot the soften values.

    The soften value is the pixel value that is added to the spatial and wavelength
    values so that chi^2/dof = 1.

    Parameters
    ----------
    detector_stats : `pandas.DataFrame`
        The detector statistics.

    Returns
    -------
    fig : `Figure`
        The soften plot.
    """
    plot_data = detector_stats.melt(id_vars=["ccd", "status_type", "description"])

    plot_data.loc[
        plot_data.query('variable.str.contains("spatial")').index, "metric"
    ] = "spatial"
    plot_data.loc[
        plot_data.query('variable.str.contains("wavelength")').index, "metric"
    ] = "wavelength"

    fg = sb.catplot(
        data=plot_data.dropna()
        .query(
            'description != "all" and variable.str.contains("soften") and status_type == "RESERVED"'
        )
        .sort_values(by=["ccd"]),
        row="metric",
        x="ccd",
        y="value",
        hue="description",
        height=2,
        aspect=4,
        palette="Set1",
        ec="k",
        linewidth=0.5,
        legend=False,
    )
    for ax in fg.figure.axes:
        ax.grid(alpha=0.25)

    fg.figure.legend(
        *fg.figure.axes[0].get_legend_handles_labels(), shadow=True, fontsize="small"
    )
    fg.figure.set_tight_layout("inches")

    return fg.figure


def plot_detector_medians(detector_stats: pd.DataFrame) -> Figure:
    """Plot the median values.

    A plot of the median values for the RESERVED spatial and wavelength data for
    the detector as a whole.

    Parameters
    ----------
    detector_stats : `pandas.DataFrame`
        The detector statistics.

    Returns
    -------
    fig : `Figure`
        The median plot.
    """
    plot_data = detector_stats.query(
        'description == "all" and status_type=="RESERVED"'
    ).filter(regex="ccd|median|soften|weighted")
    plot_data["arm"] = plot_data.ccd.str[0]

    fig, axes = plt.subplots(nrows=2, sharex=True, layout="constrained")
    fig.set_size_inches(12, 6)

    for ax, metric in zip(axes, ["spatial", "wavelength"]):
        for ccd, row in plot_data.groupby("ccd"):
            ax.errorbar(
                x=row.ccd,
                y=row[f"{metric}.median"],
                yerr=row[f"{metric}.weightedRms"],
                c=detector_palette[row.arm[0]],
                ls="",
                lw=1.5,
                capsize=2,
                zorder=-100,
            )

        sb.scatterplot(
            data=plot_data.fillna(0),
            x="ccd",
            y=f"{metric}.median",
            hue="arm",
            palette=detector_palette,
            size=f"{metric}.softenFit",
            size_norm=(0, 0.5),
            legend=False,
            ax=ax,
        )

        ax.grid(alpha=0.15)
        ax.set_title(metric)
        ax.set_ylim(-0.1, 0.1)
        ax.set_ylabel("pixel")
        ax.axhline(-0.1, c="g", ls="--", alpha=0.35)
        ax.axhline(0.1, c="g", ls="--", alpha=0.35)
        ax.axhline(0.0, c="k", ls="--", alpha=0.35, zorder=-100)

    return fig


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
