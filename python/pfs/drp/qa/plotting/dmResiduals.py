"""Detector-map residual plots.

Takes DataFrames, returns `matplotlib.figure.Figure` objects, and imports no
Butler and no task class -- which is what lets the dashboard, an on-demand
static report and the notebooks share one implementation, and what makes these
functions testable at all. See ``doc/qa-rebuild-plan.md`` section 1.5.

``pfs.drp.qa.dmResiduals`` re-exports both plotting functions, so existing
imports keep working.
"""

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from pfs.drp.qa.metrics.fitStats import FitStats
from pfs.drp.qa.plotting.palettes import div_palette, scatterplot_with_outliers
from pfs.drp.qa.utils.math import getWeightedRMS

__all__ = ["DetectorGeometry", "plot_detectormap_residuals", "plot_residual"]


@dataclass(frozen=True)
class DetectorGeometry:
    """The detector extent and wavelength range a residual plot is drawn over.

    This is everything the plots need from a `pfs.drp.stella.DetectorMap`.
    Taking the values rather than the object is what keeps this module free of
    the stack, and it is what lets a test build a figure from a synthetic frame.

    Attributes
    ----------
    width : `int`
        Detector width in pixels.
    height : `int`
        Detector height in pixels.
    fiberIdMin : `int`
        Lowest fiberId on the detector.
    fiberIdMax : `int`
        Highest fiberId on the detector.
    wavelengthMin : `float`
        Lowest wavelength covered.
    wavelengthMax : `float`
        Highest wavelength covered.
    """

    width: int = 4096
    height: int = 4176
    fiberIdMin: int | None = None
    fiberIdMax: int | None = None
    wavelengthMin: float | None = None
    wavelengthMax: float | None = None

    @classmethod
    def fromDetectorMap(cls, detectorMap: Any) -> "DetectorGeometry":
        """Read the geometry off a `pfs.drp.stella.DetectorMap`.

        Parameters
        ----------
        detectorMap : `DetectorMap`
            The detector map. Duck-typed: only ``getBBox()``, ``fiberId`` and
            ``metadata`` are used, so this module still imports nothing from
            the stack.

        Returns
        -------
        `DetectorGeometry`
            The extracted geometry.
        """
        bbox = detectorMap.getBBox()
        return cls(
            width=bbox.width,
            height=bbox.height,
            fiberIdMin=detectorMap.fiberId.min(),
            fiberIdMax=detectorMap.fiberId.max(),
            wavelengthMin=detectorMap.metadata["WAV-MIN"],
            wavelengthMax=detectorMap.metadata["WAV-MAX"],
        )

    @classmethod
    def coerce(cls, value: "DetectorGeometry | Any") -> "DetectorGeometry":
        """Accept either a `DetectorGeometry` or a `DetectorMap`.

        Parameters
        ----------
        value : `DetectorGeometry` or `DetectorMap`
            The geometry, or the detector map to read it from.

        Returns
        -------
        `DetectorGeometry`
            The geometry.
        """
        return value if isinstance(value, cls) else cls.fromDetectorMap(value)


def plot_detectormap_residuals(
    arc_data: pd.DataFrame,
    visit_stats: pd.DataFrame,
    detectorMap: "DetectorGeometry | Any",
    useSigmaRange: bool = False,
    spatialRange: float = 0.1,
    wavelengthRange: float = 0.1,
    binWavelength: float = 0.1,
):
    """Make a plot of the residuals.

    Parameters
    ----------
    arc_data : `pandas.DataFrame`
        The arc data.
    visit_stats : `pandas.DataFrame`
        The visit statistics.
    detectorMap : `DetectorGeometry` or `DetectorMap`
        The detector extent and wavelength range the plot is drawn over. A
        `pfs.drp.stella.DetectorMap` is accepted and reduced to a
        `DetectorGeometry`, so existing callers are unaffected; passing a
        `DetectorGeometry` is what lets this function be exercised without the
        stack.
    useSigmaRange : `bool`
        Use the sigma range? Default is ``False``.
    spatialRange : `float`, optional
        The range for the spatial data. Default is 0.1.
    wavelengthRange : `float`, optional
        The range for the wavelength data. Default is 0.1.
    binWavelength : `float`, optional
        The value by which to bin the wavelength. If None, no binning.
    """
    if useSigmaRange is True:
        spatialRange = None
        wavelengthRange = None

    geometry = DetectorGeometry.coerce(detectorMap)
    dmWidth = geometry.width
    dmHeight = geometry.height
    fiberIdMin = geometry.fiberIdMin
    fiberIdMax = geometry.fiberIdMax
    wavelengthMin = geometry.wavelengthMin
    wavelengthMax = geometry.wavelengthMax

    # One big fig.
    main_fig = Figure(layout="constrained", figsize=(12, 8), dpi=150)

    # Split top fig into wo columns.
    (x_fig, y_fig) = main_fig.subfigures(1, 2, wspace=0)

    try:
        for sub_fig, column in zip([x_fig, y_fig], ["xResid", "yResid"], strict=False):
            if column == "xResid":
                plot_stats = visit_stats.query("description == 'Trace'")
            else:
                plot_stats = visit_stats.query("description != 'Trace'")

            try:
                plot_residual(
                    arc_data,
                    plot_stats,
                    column=column,
                    dataRange=spatialRange if column == "xResid" else wavelengthRange,
                    binWavelength=binWavelength,
                    sigmaLines=(1.0,),
                    dmWidth=dmWidth,
                    dmHeight=dmHeight,
                    fiberIdMin=fiberIdMin,
                    fiberIdMax=fiberIdMax,
                    wavelengthMin=wavelengthMin,
                    wavelengthMax=wavelengthMax,
                    fig=sub_fig,
                )
                sub_fig.suptitle(f"{column}", fontsize="small", fontweight="bold")
            except Exception:
                continue

        return main_fig
    except ValueError as e:
        print(e)
        return None


def plot_residual(
    data: pd.DataFrame,
    visit_stats: pd.DataFrame,
    column: str = "xResid",
    dataRange: float | None = None,
    sigmaRange: int = 2.5,
    sigmaLines: Iterable[float] | None = None,
    goodRange: float | None = None,
    binWavelength: float | None = None,
    useDMLayout: bool = True,
    dmWidth: int = 4096,
    dmHeight: int = 4176,
    fiberIdMin: int | None = None,
    fiberIdMax: int | None = None,
    wavelengthMin: float | None = None,
    wavelengthMax: float | None = None,
    fig: Figure | None = None,
) -> Figure:
    """Plot the 1D and 2D residuals on a single figure.

    Parameters
    ----------
    data : `pandas.DataFrame`
        The data.
    visit_stats : `pandas.DataFrame`
        The visit statistics.
    column : `str`, optional
        The column to use. Default is ``'xResid'``.
    dataRange : `float`, optional
        The range for the data. Default is ``None``.
    sigmaRange : `int`, optional
        The sigma range. Default is 2.5.
    sigmaLines : `tuple`, optional
        The sigma lines to plot. If None, use [1.0, 2.5].
    goodRange : `float`, optional
        Used for showing an "acceptable" range.
    binWavelength : `float`, optional
        The value by which to bin the wavelength. If None, no binning.
    useDMLayout : `bool`, optional
        Use the detector map layout? Default is ``True``.
    dmWidth : `int`, optional
        The detector map width. Default is 4096.
    dmHeight : `int`, optional
        The detector map height. Default is 4176.
    fiberIdMin : `int`, optional
        The minimum fiberId. Default is ``None``.
    fiberIdMax : `int`, optional
        The maximum fiberId. Default is ``None``.
    wavelengthMin : `float`, optional
        The minimum wavelength. Default is ``None``.
    wavelengthMax : `float`, optional
        The maximum wavelength. Default is ``None``.
    fig : `Figure`, optional
        The figure. Default is ``None``.

    Returns
    -------
    fig : `Figure`
        A summary plot of the 1D and 2D residuals.
    """
    # Wavelength residual
    if sigmaLines is None:
        sigmaLines = (1.0, 2.5)

    data["bin"] = 1
    bin_wl = False
    if isinstance(binWavelength, (int, float)) and binWavelength > 0:
        bins = np.arange(data.wavelength.min() - 1, data.wavelength.max() + 1, binWavelength)
        s_cut, bins = pd.cut(data.wavelength, bins=bins, retbins=True, labels=False)
        data["bin"] = pd.Categorical(s_cut)
        bin_wl = True

    plotData = data.melt(
        id_vars=[
            "fiberId",
            "wavelength",
            "x",
            "xErr",
            "y",
            "yErr",
            "isTrace",
            "isLine",
            "bin",
            column,
            f"{column}Outlier",
        ],
        value_vars=["isUsed", "isReserved"],
        var_name="status",
    ).query("value == True")
    plotData.rename(columns={f"{column}Outlier": "isOutlier"}, inplace=True)

    units = "pix"
    if column.startswith("y"):
        plotData = plotData.query("isTrace == False").copy()
    else:
        plotData = plotData.query("isTrace == True").copy()

    reserved_data = plotData.query('status == "isReserved" and isOutlier == False')
    if len(reserved_data) == 0:
        raise ValueError("No data")

    fig = fig or Figure(layout="constrained")

    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    # Upper row
    # Fiber residual
    fiber_avg = (
        plotData.groupby(["fiberId", "status", "isOutlier"])
        .apply(
            lambda rows: (
                len(rows),
                rows[column].median(),
                getWeightedRMS(rows[column], rows[f"{column[0]}Err"]),
            )
        )
        .reset_index()
        .rename(columns={0: "vals"})
    )
    fiber_avg = fiber_avg.join(
        pd.DataFrame(fiber_avg.vals.to_list(), columns=["count", "median", "weightedRms"])
    ).drop(columns=["vals"])

    fiber_avg.sort_values(["fiberId", "status"], inplace=True)

    pal = dict(zip(sorted(fiber_avg.status.unique()), plt.cm.tab10.colors, strict=False))
    pal_colors = [pal[x] for x in fiber_avg.status]

    # Just the errors, no markers
    goodFibersAvg = fiber_avg.query("isOutlier == False")
    ax0.errorbar(
        goodFibersAvg.fiberId,
        goodFibersAvg["median"],
        goodFibersAvg.weightedRms,
        ls="",
        ecolor=pal_colors,
        alpha=0.5,
    )

    # Get the reserved and used stats.
    fit_stats_reserved = FitStats.from_dataframe(visit_stats.query("status_type == 'RESERVED'"))
    fit_stats_used = FitStats.from_dataframe(visit_stats.query("status_type == 'USED'"))

    which_data = "spatial" if column.startswith("x") else "wavelength"
    fit_stats_reserved = getattr(fit_stats_reserved, which_data)
    fit_stats_used = getattr(fit_stats_used, which_data)

    weightedRms = fit_stats_reserved.weightedRms
    numFibers = fit_stats_reserved.num_fibers

    # Use sigma range if no range given.
    if dataRange is None and sigmaRange is not None:
        dataRange = weightedRms * sigmaRange

    # Scatterplot with outliers marked.
    ax0 = scatterplot_with_outliers(
        goodFibersAvg,
        "fiberId",
        "median",
        hue="status",
        ymin=-dataRange,
        ymax=dataRange,
        palette=pal,
        ax=ax0,
        refline=[0],
        rasterized=True,
    )

    def drawRefLines(ax, goodRange, sigmaRange, isVertical=False):
        method = "axhline" if isVertical is False else "axvline"
        refLine = getattr(ax, method)
        # Good sigmas
        if goodRange is not None:
            for i, lim in enumerate(goodRange):
                refLine(lim, c="g", ls="-.", alpha=0.75, label="Good limits")
                if i == 0:
                    ax.text(
                        fiber_avg.fiberId.min(),
                        1.5 * lim,
                        f"±1.0σ={abs(lim):.4f}",
                        c="g",
                        ha="right",
                        clip_on=True,
                        weight="bold",
                        zorder=100,
                        bbox={"boxstyle": "round", "ec": "k", "fc": "wheat", "alpha": 0.75},
                    )

        if sigmaLines is not None:
            for sigmaLine in sigmaLines:
                for i, sigmaMultiplier in enumerate([sigmaLine, -1 * sigmaLine]):
                    lim = sigmaMultiplier * weightedRms
                    refLine(
                        lim,
                        c=pal["isReserved"],
                        ls="--",
                        alpha=0.75,
                        label=f"{lim} * sigma",
                    )
                    if i == 0:
                        ax.text(
                            fiber_avg.fiberId.min(),
                            1.5 * lim,
                            f"±{sigmaMultiplier}σ={abs(lim):.4f}",
                            c=pal["isReserved"],
                            ha="right",
                            clip_on=True,
                            weight="bold",
                            zorder=100,
                            bbox={"boxstyle": "round", "ec": "k", "fc": "wheat", "alpha": 0.75},
                        )

    drawRefLines(ax0, goodRange, sigmaRange)

    ax0.legend(
        loc="lower right",
        shadow=True,
        prop={"family": "monospace", "weight": "bold"},
        bbox_to_anchor=(1.2, 0),
    )

    fiber_outliers = goodFibersAvg.query(f'status=="isReserved" and abs(median) >= {weightedRms}')
    num_sig_outliers = fiber_outliers.fiberId.count()
    fiber_big_outliers = fiber_outliers.query(f"abs(median) >= {sigmaRange * weightedRms}")
    num_siglimit_outliers = fiber_big_outliers.fiberId.count()
    ax0.text(
        0.01,
        0.0,
        f"Number of fibers: {numFibers} "
        f"Number of outliers: "
        f"1σ: {num_sig_outliers} "
        f"{sigmaRange}σ: {num_siglimit_outliers}",
        transform=ax0.transAxes,
        bbox={"boxstyle": "round", "ec": "k", "fc": "wheat"},
        fontsize="small",
        zorder=100,
    )

    if fiberIdMin is not None and fiberIdMax is not None:
        ax0.set_xlim(fiberIdMin, fiberIdMax)

    if useDMLayout is True:
        # Reverse the fiber order to match the xy-pixel layout
        ax0.set_xlim(*list(reversed(ax0.get_xlim())))

    ax0.set_ylabel(f"Δ {units}")
    ax0.xaxis.set_label_position("top")
    ax0.set_xlabel("")
    ax0.xaxis.tick_top()
    ax0.set_title(
        f"Median {which_data} residual and 1-sigma weighted error by fiberId",
        weight="bold",
        fontsize="small",
    )
    legend = ax0.get_legend()
    legend.set_title("")
    legend.set_visible(False)

    ax1.text(
        -0.01,
        0.0,
        f"RESERVED:\n{fit_stats_reserved}\nUSED:\n{fit_stats_used}",
        transform=ax1.transAxes,
        fontfamily="monospace",
        fontsize="small",
        fontweight="bold",
        bbox={"boxstyle": "round", "alpha": 0.5, "facecolor": "white"},
    )
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Lower row
    # 2d residual
    norm = colors.Normalize(vmin=-dataRange, vmax=dataRange)
    if useDMLayout:
        X = "x"
        Y = "y"
    else:
        X = "fiberId"
        Y = "wavelength"

    for isLine, rows in reserved_data.groupby("isLine", observed=False):
        im = ax2.scatter(
            rows[X],
            rows[Y],
            c=rows[column],
            norm=norm,
            cmap=div_palette,
            s=2,
            marker="d" if isLine else ".",
            zorder=100 if isLine else 0,
            rasterized=True,
        )

    fig.colorbar(
        im,
        ax=ax2,
        orientation="horizontal",
        extend="both",
        fraction=0.02,
        aspect=75,
        pad=0.01,
    )

    ax2.set_xlim(0, dmWidth)
    ax2.set_ylim(0, dmHeight)
    ax2.set_ylabel(Y)
    ax2.set_xlabel(X)
    ax2.set_title(f"2D residual of RESERVED {which_data} data", weight="bold", fontsize="small")

    if bin_wl is True:
        binned_data = plotData.dropna(subset=["wavelength", column]).groupby(
            ["bin", "status", "isOutlier"], observed=False
        )[["wavelength", column]]
        # NB: this used to read `.agg("median", robustRms)`. A string aggregator
        # takes no function argument -- robustRms was swallowed as `numeric_only`
        # and never called -- so the median is what was always computed. Dropping
        # the argument changes no output and removes the last stack import from
        # the plotting path.
        plotData = binned_data.agg("median").reset_index().sort_values("status")

    ax3 = scatterplot_with_outliers(
        plotData.query("isOutlier == False"),
        column,
        "wavelength",
        hue="status",
        ymin=-dataRange,
        ymax=dataRange,
        palette=pal,
        ax=ax3,
        refline=[0.0],
        vertical=True,
        rasterized=True,
    )
    # Skip missing wavelength legend.
    with contextlib.suppress(AttributeError):
        ax3.get_legend().set_visible(False)

    drawRefLines(ax3, goodRange, sigmaRange, isVertical=True)

    ax3.set_ylim(wavelengthMin, wavelengthMax)

    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.set_xlabel(f"Δ {units}")
    ax_title = f"{which_data.title()} residual\nby wavelength"
    if bin_wl:
        ax_title += f" binsize={binWavelength} {units}"
    ax3.set_title(ax_title, weight="bold", fontsize="small")

    return fig
