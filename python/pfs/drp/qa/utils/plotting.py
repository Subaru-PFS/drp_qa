import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from matplotlib.figure import Figure

div_palette = plt.cm.RdBu_r.with_extremes(over="magenta", under="cyan", bad="lime")
detector_palette = {"b": "tab:blue", "r": "tab:red", "n": "tab:orange", "m": "tab:pink"}
spectrograph_plot_markers = {1: "s", 2: "o", 3: "X", 4: "P"}
description_palette = {
    "ArI": "tab:green",
    "CdI,HgI": "tab:purple",
    "KrI": "tab:brown",
    "NeI": "tab:pink",
    "Trace": "tab:cyan",
    "XeI": "tab:olive",
    "O2,OH": "tab:blue",
}


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
