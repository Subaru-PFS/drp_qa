from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Iterable, Optional

import lsstDebug
import numpy as np
import pandas as pd
import seaborn as sb
from astropy.stats import sigma_clip
from lsst.daf.persistence import NoResults
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, Task
from matplotlib import colors, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from pfs.drp.stella import ArcLineSet, DetectorMap, ReferenceLineStatus
from pfs.drp.stella.utils.math import robustRms
from pfs.utils.fiberids import FiberIds
from scipy.optimize import bisect

from pfs.drp.qa.utils.math import getChi2, getWeightedRMS
from pfs.drp.qa.utils.plotting import description_palette, div_palette


class PlotResidualConfig(Config):
    """Configuration for PlotResidualTask"""

    combineVisits = Field(dtype=bool, default=False, doc="Combine all visits for processing.")
    makeResidualPlots = Field(dtype=bool, default=True, doc="Generate a residual plot for each dataId.")
    useSigmaRange = Field(dtype=bool, default=False, doc="Use ±2.5 sigma as range")
    xrange = Field(dtype=float, default=0.1, doc="Range of the residual (X center) in a plot in pix.")
    wrange = Field(dtype=float, default=0.1, doc="Range of the residual (wavelength) in a plot in pix.")
    binWavelength = Field(dtype=float, default=0.1, doc="Wavelength bin for residual plot.")


class PlotResidualTask(Task):
    """Task for QA of detectorMap."""

    ConfigClass = PlotResidualConfig
    _DefaultName = "plotResidual"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, detectorName, arcLinesSet: ArcLineSet, detectorMaps: DetectorMap, dataIds) -> Struct:
        """QA of adjustDetectorMap by plotting the fitting residual.

        Parameters
        ----------
        detectorName : `str`
            The detector name, e.g. ``"r2"``.
        arcLinesSet : `ArcLineSet`
            Emission line measurements by adjustDetectorMap.
        detectorMaps : `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        dataIds : `list`
            List of dataIds.

        Returns
        -------
        dmQaResidualPlot : `MultipagePdfFigure`
            1D and 2D plots of the residual between the detectormap and the arclines for a given visit.
        dmQaCombinedResidualPlot : `MultipagePdfFigure`
            1D and 2D plots of the residual between the detectormap and the arclines for the entire detector.
        dmQaResidualStats : `pd.DataFrame`
            Statistics of the residual analysis.
        dmQaDetectorStats : `pd.DataFrame`
            Statistics of the residual analysis.
        """
        arc_data, visit_stats, detector_stats = get_residual_info(arcLinesSet, detectorMaps, dataIds)

        run = dataIds[0]["run"]

        results = Struct()
        if arc_data is not None and len(arc_data) and visit_stats is not None and len(visit_stats):
            if self.config.makeResidualPlots is True:
                arm = str(detectorName[-2])
                spectrograph = int(detectorName[-1])
                residFig = plot_detectormap_residuals(
                    arc_data,
                    visit_stats,
                    arm,
                    spectrograph,
                    useSigmaRange=self.config.useSigmaRange,
                    xrange=self.config.xrange,
                    wrange=self.config.wrange,
                    binWavelength=self.config.binWavelength,
                )

                suptitle = f"DetectorMap Residuals {detectorName}\n{run}"
                if self.config.combineVisits is True:
                    suptitle = f"Combined {suptitle}"

                residFig.suptitle(suptitle, weight="bold")
                if self.config.combineVisits is True:
                    results = Struct(
                        dmQaCombinedResidualPlot=[residFig],
                        dmQaDetectorStats=[detector_stats],
                    )
                else:
                    results = Struct(
                        dmQaResidualPlot=[residFig],
                        dmQaResidualStats=[visit_stats],
                    )

        return results


@dataclass
class FitStat:
    median: float
    robustRms: float
    weightedRms: float
    softenFit: float
    dof: float
    num_fibers: int
    num_lines: int

    def __str__(self):
        return f"""median  = {self.median:> 7.05f}
rms     = {self.weightedRms:> 7.05f}
soften  = {self.softenFit:> 7.05f}
fibers  = {self.num_fibers:>8d}
lines   = {self.num_lines:>8d}
"""


@dataclass
class FitStats:
    dof: int
    chi2X: float
    chi2Y: float
    spatial: FitStat
    wavelength: FitStat

    def to_dict(self):
        """Output as dict."""
        return dict(
            dof=self.dof,
            chi2X=self.chi2X,
            chi2Y=self.chi2Y,
            spatial=self.spatial.__dict__,
            wavelength=self.wavelength.__dict__,
        )


def plot_detectormap_residuals(
    arc_data: pd.DataFrame,
    visit_stats: pd.DataFrame,
    arm: str,
    spectrograph: int,
    useSigmaRange: bool = False,
    xrange: float = 0.1,
    wrange: float = 0.1,
    binWavelength: float = 0.1,
):
    """Make a plot of the residuals.

    Parameters
    ----------
    arc_data : `pandas.DataFrame`
        The arc data.
    visit_stats : `pandas.DataFrame`
        The visit statistics.
    arm : `str`
        The arm.
    spectrograph : `int`
        The spectrograph.
    useSigmaRange : `bool`
        Use the sigma range? Default is ``False``.
    xrange : `float`, optional
        The range for the spatial data. Default is 0.1.
    wrange : `float`, optional
        The range for the wavelength data. Default is 0.1.
    binWavelength : `float`, optional
        The value by which to bin the wavelength. If None, no binning.
    """
    if useSigmaRange is True:
        xrange = None
        wrange = None

    ccd = f"{arm}{spectrograph}"

    # Get just the reserved visits for this ccd.
    visit_stats = visit_stats.query('status_type == "RESERVED" and ccd == @ccd').sort_values("visit").copy()

    try:
        visit_stat = visit_stats.iloc[0]
        dmWidth = visit_stat.detector_width
        dmHeight = visit_stat.detector_height
        fiberIdMin = visit_stat.fiberId_min
        fiberIdMax = visit_stat.fiberId_max
        wavelengthMin = visit_stat.wavelength_min
        wavelengthMax = visit_stat.wavelength_max
    except IndexError:
        dmWidth = None
        dmHeight = None
        fiberIdMin = None
        fiberIdMax = None
        wavelengthMin = None
        wavelengthMax = None

    # One big fig.
    main_fig = Figure(layout="constrained", figsize=(12, 10), dpi=150)

    # Split into two rows.
    (top_fig, bottom_fig) = main_fig.subfigures(2, 1, wspace=0, height_ratios=[5, 1.5])

    # Split top fig into wo columns.
    (x_fig, y_fig) = top_fig.subfigures(1, 2, wspace=0)

    try:
        pd0 = arc_data.query(f'arm == "{arm}" and spectrograph == {spectrograph}').copy()

        for sub_fig, column in zip([x_fig, y_fig], ["xResid", "yResid"]):
            try:
                plot_residual(
                    pd0,
                    column=column,
                    xrange=xrange,
                    wrange=wrange,
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
                sub_fig.suptitle(
                    f"{arm}{spectrograph}\n{column}",
                    fontsize="small",
                    fontweight="bold",
                )
            except Exception as e:
                print(f"Problem plotting residual {e}")

        visit_fig = plot_visits(visit_stats, description_palette, fig=bottom_fig)
        for ax in visit_fig.axes:
            ax.set_xlim(-0.3, 0.3)
        visit_fig.suptitle(f"RESERVED median and 1-sigma weighted error per visit {ccd=}")

        return main_fig
    except ValueError as e:
        print(e)
        return None


def plot_residual(
    data: pd.DataFrame,
    column: str = "xResid",
    xrange: float = None,
    wrange: float = None,
    sigmaRange: int = 2.5,
    sigmaLines: Optional[Iterable[float]] = None,
    goodRange: float = None,
    binWavelength: Optional[float] = None,
    useDMLayout: bool = True,
    dmWidth: int = 4096,
    dmHeight: int = 4176,
    fiberIdMin: Optional[int] = None,
    fiberIdMax: Optional[int] = None,
    wavelengthMin: Optional[float] = None,
    wavelengthMax: Optional[float] = None,
    fig: Optional[Figure] = None,
) -> Figure:
    """Plot the 1D and 2D residuals on a single figure.

    Parameters
    ----------
    data : `pandas.DataFrame`
        The data.
    column : `str`, optional
        The column to use. Default is ``'xResid'``.
    xrange : `float`, optional
        The range for the spatial data.
    wrange : `float`, optional
        The range for the wavelength data.
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
    which_data = "spatial"
    if column.startswith("y"):
        plotData = plotData.query("isTrace == False").copy()
        which_data = "wavelength"

    reserved_data = plotData.query('status == "isReserved" and isOutlier == False')
    if len(reserved_data) == 0:
        raise ValueError("No data")

    # Get summary statistics.
    fit_stats_all = get_fit_stats(data.query(f"isReserved == True and {column}Outlier == False"))
    fit_stats = getattr(fit_stats_all, which_data)
    fit_stats_used = getattr(get_fit_stats(data.query("isUsed == True")), which_data)

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

    pal = dict(zip(sorted(fiber_avg.status.unique()), plt.cm.tab10.colors))
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

    # Use sigma range if no range given.
    if xrange is None and sigmaRange is not None:
        xrange = fit_stats.weightedRms * sigmaRange

    # Scatterplot with outliers marked.
    ax0 = scatterplot_with_outliers(
        goodFibersAvg,
        "fiberId",
        "median",
        hue="status",
        ymin=-xrange,
        ymax=xrange,
        palette=pal,
        ax=ax0,
        refline=0,
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
                        bbox=dict(boxstyle="round", ec="k", fc="wheat", alpha=0.75),
                    )

        if sigmaLines is not None:
            for sigmaLine in sigmaLines:
                for i, sigmaMultiplier in enumerate([sigmaLine, -1 * sigmaLine]):
                    lim = sigmaMultiplier * fit_stats.weightedRms
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
                            bbox=dict(boxstyle="round", ec="k", fc="wheat", alpha=0.75),
                        )

    drawRefLines(ax0, goodRange, sigmaRange)

    ax0.legend(
        loc="lower right",
        shadow=True,
        prop=dict(family="monospace", weight="bold"),
        bbox_to_anchor=(1.2, 0),
    )

    fiber_outliers = goodFibersAvg.query(f'status=="isReserved" and abs(median) >= {fit_stats.weightedRms}')
    num_sig_outliers = fiber_outliers.fiberId.count()
    fiber_big_outliers = fiber_outliers.query(f"abs(median) >= {sigmaRange * fit_stats.weightedRms}")
    num_siglimit_outliers = fiber_big_outliers.fiberId.count()
    ax0.text(
        0.01,
        0.0,
        f"Number of fibers: {fit_stats.num_fibers} "
        f"Number of outliers: "
        f"1σ: {num_sig_outliers} "
        f"{sigmaRange}σ: {num_siglimit_outliers}",
        transform=ax0.transAxes,
        bbox=dict(boxstyle="round", ec="k", fc="wheat"),
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

    # Summary stats area is blank.
    ax1.text(
        -0.01,
        0.0,
        f"RESERVED:\n{fit_stats}\nUSED:\n{fit_stats_used}",
        transform=ax1.transAxes,
        fontfamily="monospace",
        fontsize="small",
        fontweight="bold",
        bbox=dict(boxstyle="round", alpha=0.5, facecolor="white"),
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
    norm = colors.Normalize(vmin=-xrange, vmax=xrange)
    if useDMLayout:
        X = "x"
        Y = "y"
    else:
        X = "fiberId"
        Y = "wavelength"

    im = ax2.scatter(
        reserved_data[X],
        reserved_data[Y],
        c=reserved_data[column],
        norm=norm,
        cmap=div_palette,
        s=2,
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

    # Use sigma range if no range given.
    if wrange is None and sigmaRange is not None:
        wrange = fit_stats.weightedRms * sigmaRange

    if bin_wl is True:
        binned_data = plotData.dropna(subset=["wavelength", column]).groupby(["bin", "status", "isOutlier"])[
            ["wavelength", column]
        ]
        plotData = binned_data.agg("median", robustRms).reset_index().sort_values("status")

    ax3 = scatterplot_with_outliers(
        plotData.query("isOutlier == False"),
        column,
        "wavelength",
        hue="status",
        ymin=-wrange,
        ymax=wrange,
        palette=pal,
        ax=ax3,
        refline=0.0,
        vertical=True,
        rasterized=True if not bin_wl else False,
    )
    try:
        ax3.get_legend().set_visible(False)
    except AttributeError:
        # Skip missing wavelength legend.
        pass

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


def plot_visits(
    plotData: pd.DataFrame,
    palette: Optional[dict] = None,
    fig: Optional[Figure] = None,
) -> Figure:
    """Plot the visit statistics.

    Parameters
    ----------
    plotData : `pandas.DataFrame`
        The data.
    palette : `dict`, optional
        The palette to use for the arcline descriptions. Keys are the descriptions
        and values are the colors. Default is ``None``.
    fig : `Figure`, optional
        The figure. Default is ``None``.

    Returns
    -------
    fig : `Figure`
        The visit statistics plot.

    """
    plotData = plotData.copy()
    fig = fig or Figure(layout="constrained")
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122, sharex=ax0, sharey=ax0)

    plotData["visit_idx"] = plotData.visit.rank(method="first")

    for ax, metric in zip([ax0, ax1], ["spatial", "wavelength"]):
        for desc, grp in plotData.groupby("description"):
            grp.plot.scatter(
                y="visit_idx",
                x=f"{metric}.median",
                xerr=f"{metric}.weightedRms",
                marker="o",
                color=palette.get(desc, "red") if palette is not None else None,
                label=desc,
                ax=ax,
            )

        ax.grid(alpha=0.2)
        ax.axvline(0, c="k", ls="--", alpha=0.5)
        ax.set_title(f"{metric}")
        ax.set_xlabel("pix")

    visit_label = [f"{row.visit}" for idx, row in plotData.iterrows()]
    ax0.set_yticks(plotData.visit_idx, visit_label, fontsize="xx-small")
    ax0.set_ylabel("Visit")
    ax0.invert_yaxis()

    fig.suptitle("RESERVED median and 1-sigma weighted errors")

    return fig


def load_and_mask_data(
    arcLines: ArcLineSet,
    detectorMap: DetectorMap,
    dropNaColumns: bool = True,
    addFiberInfo: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Cleans and masks the data. Adds fiberInfo if requested.

    The arcline data includes basic statistics, such as the median and sigma of the residuals.

    This method is called on init.

    Parameters
    ----------
    arcLines : `ArcLineSet`
        The arc lines.
    detectorMap : `DetectorMap`
        The detector map.
    dropNaColumns : `bool`, optional
        Drop columns where all values are NaN. Default is True.
    addFiberInfo : `bool`, optional
        Add fiber information to the dataframe. Default is True.

    Returns
    -------
    arcData : `pandas.DataFrame`
    """

    # Get dataframe for arc lines and add detectorMap information, then calculate residuals.
    arcData = scrub_data(arcLines, detectorMap, dropNaColumns=dropNaColumns, **kwargs)

    # Mark the sigma-clipped outliers for each relevant group.
    def maskOutliers(grp):
        grp["xResidOutlier"] = sigma_clip(grp.xResid).mask
        grp["yResidOutlier"] = sigma_clip(grp.yResid).mask
        return grp

    arcData = arcData.groupby(["status_type", "isLine"]).apply(maskOutliers)
    arcData.reset_index(drop=True, inplace=True)

    if addFiberInfo is True:
        mtp_df = pd.DataFrame(
            FiberIds().fiberIdToMTP(detectorMap.fiberId), columns=["mtpId", "mtpHoles", "cobraId"]
        )
        mtp_df.index = detectorMap.fiberId
        mtp_df.index.name = "fiberId"
        arcData = arcData.merge(mtp_df.reset_index(), on="fiberId")

    return arcData


def scrub_data(
    arcLines: ArcLineSet,
    detectorMap: DetectorMap,
    dropNaColumns: bool = False,
    removeFlagged: bool = True,
    onlyReservedAndUsed: bool = True,
) -> pd.DataFrame:
    """Gets a copy of the arcline data, with some columns added.

    Parameters
    ----------
    arcLines : `ArcLineSet`
        The arc lines.
    detectorMap : `DetectorMap`
        The detector map.
    dropNaColumns : `bool`, optional
        Drop columns where all values are NaN. Default is True.
    removeFlagged : `bool`, optional
        Remove rows with ``flag=True``? Default is True.
    onlyReservedAndUsed : `bool`, optional
        Only include rows with status RESERVED or USED? Default is True.

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """
    isTrace = arcLines.description == "Trace"
    isLine = ~isTrace

    fitPosition = np.full((len(arcLines), 2), np.nan, dtype=float)

    if isLine.any():
        fitPosition[isLine] = detectorMap.findPoint(arcLines.fiberId[isLine], arcLines.wavelength[isLine])
    if isTrace.any():
        fitPosition[isTrace, 0] = detectorMap.getXCenter(arcLines.fiberId[isTrace], arcLines.y[isTrace])
        fitPosition[isTrace, 1] = np.nan

    arcLines.data["isTrace"] = isTrace
    arcLines.data["isLine"] = isLine
    arcLines.data["xModel"] = fitPosition[:, 0]
    arcLines.data["yModel"] = fitPosition[:, 1]

    arcLines.data["xResid"] = arcLines.data.x - arcLines.data.xModel
    arcLines.data["yResid"] = arcLines.data.y - arcLines.data.yModel

    # Copy the dataframe from the arcline set.
    arc_data = arcLines.data.copy()

    # Convert nm to pixels.
    arc_data["dispersion"] = detectorMap.getDispersion(arcLines.fiberId, arcLines.wavelength)

    if removeFlagged:
        arc_data = arc_data.query("flag == False").copy()

    # Get USED and RESERVED status.
    is_reserved = (arc_data.status & ReferenceLineStatus.DETECTORMAP_RESERVED) != 0
    is_used = (arc_data.status & ReferenceLineStatus.DETECTORMAP_USED) != 0

    # Make one-hot columns for status_names.
    arc_data.loc[:, "isUsed"] = is_used
    arc_data.loc[:, "isReserved"] = is_reserved
    arc_data.loc[arc_data.isReserved, "status_type"] = "RESERVED"
    arc_data.loc[arc_data.isUsed, "status_type"] = "USED"

    # Filter to only the RESERVED and USED data.
    if onlyReservedAndUsed is True:
        arc_data = arc_data[is_used | is_reserved]

    # Drop empty rows.
    if dropNaColumns:
        arc_data = arc_data.dropna(axis=0, how="all")

        # Drop rows without enough info in position.
        arc_data = arc_data.dropna(subset=["x", "y"])

    # Change some of the dtypes explicitly.
    try:
        arc_data.y = arc_data.y.astype(np.float64)
    except AttributeError:
        pass

    # Replace inf with nans.
    arc_data = arc_data.replace([np.inf, -np.inf], np.nan)

    # Get full status names. (the .name attribute doesn't work properly so need the str instance)
    arc_data["status_name"] = arc_data.status.map(lambda x: str(ReferenceLineStatus(x)).split(".")[-1])
    arc_data["status_name"] = arc_data["status_name"].astype("category")
    arc_data.status_name = arc_data.status_name.cat.remove_unused_categories()

    return arc_data


def get_fit_stats(
    arc_data: pd.DataFrame,
    xSoften: float = 0.0,
    ySoften: float = 0.0,
    numParams: int = 0,
    maxSoften: float = 1.0,
    sigmaClipOnly: bool = True,
) -> FitStats:
    """Get the fit stats.

    Parameters
    ----------
    arc_data : `pandas.DataFrame`
        The arc data.
    xSoften : `float`, optional
        The softening parameter for the x residuals. Default is 0.0.
    ySoften : `float`, optional
        The softening parameter for the y residuals. Default is 0.0.
    numParams : `int`, optional
        The number of parameters in the model. Default is 0.
    maxSoften : `float`, optional
        The maximum value for the softening parameter. Default is 1.0.
    sigmaClipOnly : `bool`, optional
        Only include non-outliers in the fit stats. Default is True.

    Returns
    -------
    fitStats : `FitStats`
    """
    if sigmaClipOnly is True:
        arc_data = arc_data.query("xResidOutlier == False")

    traces = arc_data.query("isTrace == True").copy()
    lines = arc_data.query("isLine == True").dropna(subset=["yResid"]).copy()

    xNum = len(arc_data)
    try:
        yNum = lines.isLine.value_counts()[True]
    except KeyError:
        yNum = 0

    xWeightedRms = getWeightedRMS(arc_data.xResid, arc_data.xErr, soften=xSoften)
    yWeightedRms = getWeightedRMS(lines.yResid, lines.yErr, soften=ySoften)

    def doRobust(x):
        try:
            return robustRms(x.dropna())
        except (IndexError, ValueError):
            return np.nan

    xRobustRms = doRobust(arc_data.xResid)
    yRobustRms = doRobust(lines.yResid)

    chi2X = getChi2(arc_data.xResid, arc_data.xErr, xSoften)
    chi2Y = getChi2(lines.yResid, lines.yErr, ySoften)

    xDof = xNum - numParams / 2
    yDof = yNum - numParams / 2
    dof = xDof + yDof

    def getSoften(resid, err, dof, soften=0):
        if len(resid) == 0:
            return 0
        with np.errstate(invalid="ignore"):
            return (getChi2(resid, err, soften) / dof) - 1

    f_x = partial(getSoften, arc_data.xResid, arc_data.xErr, xDof)
    f_y = partial(getSoften, lines.yResid, lines.yErr, yDof)

    if f_x(0) < 0:
        xSoftFit = 0.0
    elif f_x(maxSoften) > 0:
        xSoftFit = np.nan
    else:
        xSoftFit = bisect(f_x, 0, maxSoften)

    if f_y(0) < 0:
        ySoftFit = 0.0
    elif f_y(maxSoften) > 0:
        ySoftFit = np.nan
    else:
        ySoftFit = bisect(f_y, 0, maxSoften)

    xFibers = len(traces.fiberId.unique())
    yFibers = len(lines.fiberId.unique())

    xFitStat = FitStat(arc_data.xResid.median(), xRobustRms, xWeightedRms, xSoftFit, xDof, xFibers, xNum)
    yFitStat = FitStat(lines.yResid.median(), yRobustRms, yWeightedRms, ySoftFit, yDof, yFibers, yNum)

    return FitStats(dof, chi2X, chi2Y, xFitStat, yFitStat)


def get_residual_info(
    arcLinesSet: Iterable[ArcLineSet], detectorMaps: Iterable[DetectorMap], dataIds: Iterable[dict]
) -> tuple:
    """Get the stats for the residual between the arclines and the detectormap.

    Parameters
    ----------
    arcLinesSet : `Iterable[ArcLineSet]`
        The arc lines.
    detectorMaps : `Iterable[DetectorMap]`
        The detector maps.
    dataIds : `Iterable[dict]`
        The data IDs.

    Returns
    -------
    all_arc_data : `pandas.DataFrame`
    all_visit_stats : `pandas.DataFrame`
    all_detector_stats : `pandas.DataFrame`
    """
    all_data = list()
    visit_stats = list()
    detector_stats = list()

    all_arc_data = None
    all_visit_stats = None
    all_detector_stats = None

    for arcLines, detectorMap, dataId in zip(arcLinesSet, detectorMaps, dataIds):
        try:
            arc_data = load_and_mask_data(arcLines, detectorMap)

            if len(arc_data) == 0:
                print(f"No data for {dataId}")
                continue

            visit = dataId["visit"]
            arm = dataId["arm"]
            spectrograph = dataId["spectrograph"]
            ccd = f"{arm}{spectrograph}"
            arc_data["visit"] = visit
            arc_data["arm"] = arm
            arc_data["spectrograph"] = spectrograph

            all_data.append(arc_data)

            descriptions = sorted(list(arc_data.description.unique()))
            with suppress(ValueError):
                if len(descriptions) > 1:
                    descriptions.remove("Trace")

            dmap_bbox = detectorMap.getBBox()
            fiberIdMin = detectorMap.fiberId.min()
            fiberIdMax = detectorMap.fiberId.max()
            wavelengthMin = int(arcLines.wavelength.min())
            wavelengthMax = int(arcLines.wavelength.max())

            for idx, rows in arc_data.groupby("status_type"):
                visit_stat = pd.json_normalize(get_fit_stats(rows).to_dict())
                visit_stat["visit"] = visit
                visit_stat["arm"] = arm
                visit_stat["spectrograph"] = spectrograph
                visit_stat["ccd"] = ccd
                visit_stat["status_type"] = idx
                visit_stat["description"] = ",".join(descriptions)
                visit_stat["detector_width"] = dmap_bbox.width
                visit_stat["detector_height"] = dmap_bbox.height
                visit_stat["fiberId_min"] = fiberIdMin
                visit_stat["fiberId_max"] = fiberIdMax
                visit_stat["wavelength_min"] = wavelengthMin
                visit_stat["wavelength_max"] = wavelengthMax
                visit_stats.append(visit_stat)
        except NoResults:
            print(f"No results found for {dataId}")
        except Exception as e:
            print(e)

    if len(visit_stats):
        all_visit_stats = pd.concat(visit_stats)

    if len(all_data):
        all_arc_data = pd.concat(all_data)

        # Get the stats for the whole detector by status type.
        for status_type, rows in all_arc_data.groupby(["status_type"]):
            try:
                detectorStats = pd.json_normalize(get_fit_stats(rows).to_dict())
                arm = rows["arm"].iloc[0]
                spectrograph = rows["spectrograph"].iloc[0]
                ccd = f"{arm}{spectrograph}"
                detectorStats["ccd"] = ccd
                detectorStats["status_type"] = status_type
                detectorStats["description"] = "all"
                detector_stats.append(detectorStats)
            except Exception as e:
                print(status_type, e)

        # Get stats for each description type.
        for (status_type, desc), rows in all_arc_data.groupby(["status_type", "description"]):
            try:
                detectorStats = pd.json_normalize(get_fit_stats(rows).to_dict())
                arm = rows["arm"].iloc[0]
                spectrograph = rows["spectrograph"].iloc[0]
                ccd = f"{arm}{spectrograph}"
                detectorStats["ccd"] = ccd
                detectorStats["status_type"] = status_type
                detectorStats["description"] = desc
                detector_stats.append(detectorStats)
            except Exception as e:
                print(status_type, desc, e)

        all_detector_stats = pd.concat(detector_stats)

    return all_arc_data, all_visit_stats, all_detector_stats
