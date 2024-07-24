from typing import Optional, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from pfs.drp.qa.utils.helpers import getFitStats
from pfs.drp.qa.utils.math import getWeightedRMS
from pfs.drp.stella.utils.math import robustRms

div_palette = plt.cm.RdBu_r.with_extremes(over="magenta", under="cyan", bad="lime")
detector_palette = {"b": "tab:blue", "r": "tab:red", "n": "tab:orange", "m": "tab:pink"}

description_palette = {
    "ArI": "tab:green",
    "CdI,HgI": "tab:purple",
    "KrI": "tab:brown",
    "NeI": "tab:pink",
    "Trace": "tab:cyan",
    "XeI": "tab:olive",
    "O2,OH": "tab:blue",
}


def makePlot(
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
                plotResidual(
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
                sub_fig.suptitle(f"{arm}{spectrograph}\n{column}", fontsize="small", fontweight="bold")
            except Exception as e:
                print(f"Problem plotting residual {e}")

        visit_fig = plotVisits(
            visit_stats.query('status_type == "RESERVED" and ccd == @ccd').sort_values(by="visit").copy(),
            description_palette,
            fig=bottom_fig,
        )
        for ax in visit_fig.axes:
            ax.set_xlim(-0.3, 0.3)
        visit_fig.suptitle(f"RESERVED median and 1-sigma weighted error per visit {ccd=}")

        return main_fig
    except ValueError as e:
        print(e)
        return None


def plotResidual(
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
    fit_stats_all = getFitStats(data.query(f"isReserved == True and {column}Outlier == False"))
    fit_stats = getattr(fit_stats_all, which_data)
    fit_stats_used = getattr(getFitStats(data.query("isUsed == True")), which_data)

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
    ax0 = scatterplotWithOutliers(
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
                    refLine(lim, c=pal["isReserved"], ls="--", alpha=0.75, label=f"{lim} * sigma")
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
        loc="lower right", shadow=True, prop=dict(family="monospace", weight="bold"), bbox_to_anchor=(1.2, 0)
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
        f"Median {which_data} residual and 1-sigma weighted error by fiberId", weight="bold", fontsize="small"
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
    fig.colorbar(im, ax=ax2, orientation="horizontal", extend="both", fraction=0.02, aspect=75, pad=0.01)

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

    ax3 = scatterplotWithOutliers(
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


def scatterplotWithOutliers(
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

    The plot can be rendered vertically but you should still use the `X` and
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


def plotVisits(
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


def plotDetectorSoften(detector_stats: pd.DataFrame) -> Figure:
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

    plot_data.loc[plot_data.query('variable.str.contains("spatial")').index, "metric"] = "spatial"
    plot_data.loc[plot_data.query('variable.str.contains("wavelength")').index, "metric"] = "wavelength"

    fg = sb.catplot(
        data=plot_data.dropna()
        .query('description != "all" and variable.str.contains("soften") and status_type == "RESERVED"')
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

    fg.figure.legend(*fg.figure.axes[0].get_legend_handles_labels(), shadow=True, fontsize="small")
    fg.figure.set_tight_layout("inches")

    return fg.figure


def plotDetectorMedians(detector_stats: pd.DataFrame) -> Figure:
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
    plot_data = detector_stats.query('description == "all" and status_type=="RESERVED"').filter(
        regex="ccd|median|soften|weighted"
    )
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
