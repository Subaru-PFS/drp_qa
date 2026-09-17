"""Combined (cross-visit, cross-detector) detector-map residual plots.

Takes DataFrames, returns `matplotlib.figure.Figure` objects, and imports no
Butler and no task class. `reportFigures` yields the pages of the combined
report in order; assembling them into a multi-page PDF is the task's job, since
that is the part that needs the Butler storage class.

See ``doc/qa-rebuild-plan.md`` section 1.5.
"""

import itertools
from collections.abc import Iterator
from itertools import product
from typing import Any

import pandas as pd
import seaborn as sb
from matplotlib.figure import Figure

from pfs.drp.qa.plotting.dmResiduals import plot_detectormap_residuals
from pfs.drp.qa.plotting.palettes import description_palette, detector_palette

__all__ = [
    "plot_detector_summary",
    "plot_detector_summary_per_desc",
    "plot_title",
    "plot_visits",
    "reportFigures",
]

#: Columns of ``dmQaResidualData`` the per-detector residual pages are drawn from.
REPORT_PLOT_COLUMNS = [
    "fiberId",
    "wavelength",
    "x",
    "xErr",
    "y",
    "yErr",
    "isTrace",
    "isLine",
    "xResid",
    "yResid",
    "xResidOutlier",
    "yResidOutlier",
    "isUsed",
    "isReserved",
    "status",
    "visit",
]


def reportFigures(
    residual_stats: pd.DataFrame,
    residual_data: pd.DataFrame,
    detectorMaps: dict[str, Any],
    run_name: str,
    log: Any,
) -> Iterator[Figure]:
    """Yield the pages of the combined residual report, in order.

    The caller decides what to do with them. `DetectorMapCombinedResidualsTask`
    appends each to a `MultipagePdfFigure`; a dashboard or an on-demand report
    can render them one at a time.

    Parameters
    ----------
    residual_stats : `pandas.DataFrame`
        Concatenated ``dmQaResidualStats``.
    residual_data : `pandas.DataFrame`
        Concatenated ``dmQaResidualData``.
    detectorMaps : `dict` [`str`, `DetectorMap`]
        Detector maps keyed by CCD name (``"b1"``, ``"r3"``, ...). Only the
        geometry is used; see
        `pfs.drp.qa.plotting.dmResiduals.DetectorGeometry`.
    run_name : `str`
        The collection the statistics came from, shown on the title page.
    log : `Any`
        A logger, used to report the detectors that could not be drawn.

    Yields
    ------
    `matplotlib.figure.Figure`
        Title page, two detector summaries, then two pages per CCD.
    """
    reserved_stats = residual_stats.query("status_type == 'RESERVED'").copy()

    yield plot_title(run_name)

    log.info("Making detector summary plots")
    yield plot_detector_summary(reserved_stats)
    yield plot_detector_summary_per_desc(reserved_stats)

    for ccd, visit_stats in residual_stats.groupby("ccd", observed=False):
        log.info(f"Making plots for {ccd}")
        try:
            arm = ccd[0]
            spec = int(ccd[1])
            plot_data = residual_data.query(f"arm == '{arm}' and spectrograph == {spec}")

            # For a combined report we want the mean across visits.
            grouped = plot_data[REPORT_PLOT_COLUMNS].groupby(["status", "isLine", "fiberId", "y"])
            plot_data = grouped.mean().reset_index()

            residFig = plot_detectormap_residuals(plot_data, visit_stats, detectorMaps[str(ccd)])
            residFig.suptitle(f"DetectorMap Residuals - Median of all visits - {ccd}", weight="bold")

            descFig = plot_visits(visit_stats.query('status_type == "RESERVED"'), palette=description_palette)
            descFig.suptitle(f"{descFig.get_suptitle()} - {ccd}")
        except KeyError:
            log.warning(f"DetectorMap not found for {ccd}. Skipping.")
        except Exception as e:
            log.warning(f"Error plotting for {ccd}: {e}")
            continue
        else:
            yield residFig
            yield descFig


def plot_detector_summary(stats: pd.DataFrame) -> Figure:
    plot_data_spatial = (
        stats.query("description == 'Trace'")
        .filter(regex="ccd|spatial.(median|weighted|soften)")
        .groupby("ccd", observed=False)
        .mean()
    )
    plot_data_spatial.columns = [c.replace("spatial.", "") for c in plot_data_spatial.columns]
    plot_data_wavelength = (
        stats.query("description != 'Trace'")
        .filter(regex="ccd|wavelength.(median|weighted|soften)")
        .groupby("ccd", observed=False)
        .mean()
    )
    plot_data_wavelength.columns = [c.replace("wavelength.", "") for c in plot_data_wavelength.columns]

    fig = Figure(figsize=(11, 8), layout="constrained")
    spatial_plot_ax = fig.add_subplot(221)
    wavelength_plot_ax = fig.add_subplot(222, sharey=spatial_plot_ax)
    spatial_table_ax = fig.add_subplot(223, sharex=spatial_plot_ax)
    wavelength_table_ax = fig.add_subplot(224, sharex=wavelength_plot_ax, sharey=spatial_table_ax)

    formatter = "{:5.04f}".format

    # Plot the spatial median and weightedRms and show table below.
    for ccd, row in plot_data_spatial.iterrows():
        spatial_plot_ax.errorbar(
            x=ccd,
            y=row["median"],
            yerr=row["weightedRms"],
            markersize=max(row["softenFit"].mean() * 100, 1),
            marker="o",
            mec="k",
            linewidth=2,
            capsize=2,
            color=detector_palette[ccd[0]],
        )

    spatial_table = pd.plotting.table(spatial_table_ax, plot_data_spatial.map(formatter), loc="center")
    spatial_table.set_fontsize(11)
    spatial_table_ax.set_title("Spatial median and weightedRms error (quartz only)")

    for ccd, row in plot_data_wavelength.iterrows():
        wavelength_plot_ax.errorbar(
            x=ccd,
            y=row["median"],
            yerr=row["weightedRms"],
            markersize=max(row["softenFit"].mean() * 100, 1),
            marker="o",
            markeredgecolor="k",
            linewidth=2,
            capsize=2,
            color=detector_palette[ccd[0]],
        )

    wavelength_table = pd.plotting.table(
        wavelength_table_ax, plot_data_wavelength.map(formatter), loc="center"
    )
    wavelength_table.set_fontsize(11)
    wavelength_table_ax.set_title("Wavelength median and weightedRms")

    # Set the titles and labels
    spatial_plot_ax.axhline(0, c="k", ls="--", alpha=0.3)
    spatial_plot_ax.set_title("Spatial median and weightedRms error (quartz only)")
    spatial_plot_ax.set_ylabel("Median (pixel)")

    wavelength_plot_ax.axhline(0, c="k", ls="--", alpha=0.3)
    wavelength_plot_ax.set_title("Wavelength median and weightedRms")

    spatial_plot_ax.grid(True, color="k", linestyle="--", alpha=0.15)
    wavelength_plot_ax.grid(True, color="k", linestyle="--", alpha=0.15)
    spatial_table_ax.set_axis_off()
    wavelength_table_ax.set_axis_off()

    fig.suptitle("DetectorMap Residuals Summary", y=1.05)

    return fig


def plot_detector_summary_per_desc(stats: pd.DataFrame) -> Figure:
    plot_data = (
        stats.set_index(["ccd", "description"])
        .filter(regex="median|weighted|soften")
        .melt(ignore_index=False)
        .reset_index()
    )
    plot_data.loc[
        plot_data.query('description != "Trace" and variable.str.startswith("spatial")').index, "value"
    ] = pd.NA

    col_order = [
        f"{b}.{a}" for a, b in product(["median", "weightedRms", "softenFit"], ["spatial", "wavelength"])
    ]

    fg = sb.catplot(
        data=plot_data.dropna(),
        x="ccd",
        y="value",
        col="variable",
        col_wrap=2,
        col_order=col_order,
        hue="description",
        palette=description_palette,
        kind="box",
        sharey=False,
        height=3,
        aspect=2.5,
        flierprops={"marker": ".", "ms": 2},
    )
    fg.figure.suptitle("DetectorMap Residuals by description", y=1)
    for i, ax in enumerate(fg.figure.axes):
        ax.set_ylabel("Median residual (pixel)")
        if i == 0:
            # Should be spatial median.
            ax.set_ylim(-0.001, 0.001)
        elif i == 1:
            # Should be wavelength median.
            ax.set_ylim(-0.05, 0.05)
        else:
            # Last plots are the rms and soften.
            ax.set_ylim(0.0, 0.5)

        ax.axhline(0, c="k", ls="--", alpha=0.3)

        # Get the x-axis tick labels and their positions
        xticks = ax.get_xticks()

        # Shade alternate backgrounds
        for j in range(0, len(xticks), 2):
            ax.fill_between(
                [xticks[j] - 0.5, xticks[j] + 0.5],
                0,
                1,
                transform=ax.get_xaxis_transform(),
                color="lightgray",
                alpha=0.5,
            )

    return fg.figure


def plot_visits(
    plotData: pd.DataFrame,
    palette: dict | list | None = None,
    spatialRange: float = 0.1,
    wavelengthRange: float = 0.1,
    fig: Figure | None = None,
) -> Figure:
    """Plot the visit statistics.

    Parameters
    ----------
    plotData : `pandas.DataFrame`
        The data.
    palette : `dict`, optional
        The palette to use for the arcline descriptions. Keys are the descriptions
        and values are the colors. Default is ``None``.
    spatialRange : `float`, optional
        The range for the spatial data. Default is 0.1.
    wavelengthRange : `float`, optional
        The range for the wavelength data. Default is 0.1.
    fig : `Figure`, optional
        The figure. Default is ``None``.

    Returns
    -------
    fig : `Figure`
        The visit statistics plot.

    """
    plotData = plotData.sort_values(by=["visit"]).copy()
    fig = fig or Figure(layout="constrained")
    fig.set_size_inches(11, 8)
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122, sharex=ax0, sharey=ax0)

    plotData["visit_idx"] = plotData.visit.rank(method="first")

    palette = palette or description_palette

    for ax, metric in zip([ax0, ax1], ["spatial", "wavelength"], strict=False):
        metricData = plotData.copy()
        if metric == "spatial":
            metricData = metricData.query("description == 'Trace'")
        else:
            metricData = metricData.query("description != 'Trace'")

        for desc, grp in metricData.groupby("description"):
            grpPlotData = grp.copy()
            ax.errorbar(
                y=grpPlotData["visit_idx"],
                x=grpPlotData[f"{metric}.median"],
                xerr=grpPlotData[f"{metric}.weightedRms"],
                marker="o",
                ms=10,
                elinewidth=2,
                capsize=4,
                mec="w",
                ls="",
                color=palette.get(desc, "black"),
                label=desc,
                zorder=110,
            )

        # Mark the median and 1-sigma range across the visits.
        summary_stats = metricData.filter(regex="median|weighted").median().to_dict()
        upper_range = summary_stats[f"{metric}.median"] + summary_stats[f"{metric}.weightedRms"]
        lower_range = summary_stats[f"{metric}.median"] - summary_stats[f"{metric}.weightedRms"]

        ax.axvline(summary_stats[f"{metric}.median"], c="k", ls="--", zorder=101)
        ax.axvline(upper_range, c="g", ls="--", zorder=101)
        ax.axvline(lower_range, c="g", ls="--", zorder=101)
        ax.set_title(
            f"{metric.upper()}: "
            f"median={summary_stats[f'{metric}.median']:5.04f} "
            f"rms={summary_stats[f'{metric}.weightedRms']:5.04f}"
        )

        ax.grid(which="major", color="k", axis="y", zorder=-100)
        ax.axvline(0, c="k", ls="-", alpha=0.75)
        ax.set_title(f"{metric}")
        ax.set_xlabel("pix")

        leg = ax.legend(loc="upper right", shadow=True)
        leg.set_zorder(1000)

        if spatialRange is not None and metric == "spatial":
            ax.set_xlim(-spatialRange, spatialRange)
        if wavelengthRange is not None and metric == "wavelength":
            ax.set_xlim(-wavelengthRange, wavelengthRange)

    # Only label a visit the first time it's seen.
    labeled_ticks = {}
    all_ticks = []
    for _idx, row in plotData.reset_index().iterrows():
        if row.visit not in labeled_ticks.values():
            all_ticks.append(f"{row.visit}")
            labeled_ticks[row.visit_idx] = row.visit

    # Create a striped background to offset the visits.
    ax0.set_yticks(list(labeled_ticks.keys()), list(labeled_ticks.values()), fontsize="xx-small")
    for i, (y0, y1) in enumerate(itertools.pairwise(list(labeled_ticks.keys()))):
        ax0.axhspan(y0, y1, color="whitesmoke" if i % 2 == 0 else "ivory", alpha=0.5)
        ax1.axhspan(y0, y1, color="whitesmoke" if i % 2 == 0 else "ivory", alpha=0.5)

    ax0.set_ylabel("Visit")
    ax0.invert_yaxis()

    fig.suptitle("RESERVED median and 1-sigma weighted errors", fontsize="small")

    return fig


def plot_title(run_name: str) -> Figure:
    """Plot a title page for the combined report."""
    fig = Figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.text(0.5, 0.5, "DetectorMap Residuals Summary", ha="center", va="center", fontsize="large")
    ax.text(0.5, 0.35, f"{run_name}", ha="center", va="center")
    return fig
