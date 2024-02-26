import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import iqr
import pandas as pd
import seaborn as sb

from matplotlib import colors
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

from pfs.drp.stella.utils import addPfsCursor
from pfs.drp.stella import ArcLineSet, DetectorMap, ReferenceLineStatus

div_palette = plt.cm.RdBu_r.with_extremes(over='magenta', under='cyan', bad='lime')


def iqr_sigma(x) -> float:
    """Calculate the sigma of the interquartile range as a robust estimate of the std.

    Note: This will ignore NaNs.

    Parameters
    ----------
    x : `numpy.ndarray`
        The data.


    Returns
    -------
    sigma : `float`
        The sigma of the interquartile range.
    """
    return iqr(x, nan_policy='omit') / 1.349


def plotResiduals1D(arcLines: ArcLineSet,
                    detectorMap: DetectorMap,
                    arcData: pd.DataFrame,
                    showAllRange: bool = False,
                    xrange: float = 0.2,
                    wrange: float = 0.03,
                    pointSize: float = 0.2,
                    quivLength: float = 0.2
                    ) -> plt.Figure:
    """Plot the residuals as a function of wavelength and fiberId.

    Parameters
    ----------
    arcLines : `ArcLineSet`
        The arc lines.
    detectorMap : `DetectorMap`
        The detector map.
    arcData : `pd.DataFrame`
        The arc data.
    showAllRange : `bool`
        Show the full range of residuals. Default is ``False``.
    xrange : `float`
        The range of the x-axis. Default is 0.2.
    wrange : `float`
        The range of the y-axis. Default is 0.03.
    pointSize : `float`
        The size of the points. Default is 0.2.
    quivLength : `float`
        The length of the quiver. Default is 0.2.

    Returns
    -------
    fig1 : `plt.Figure`
    """
    fmin, fmax = np.amin(arcLines.fiberId), np.amax(arcLines.fiberId)
    dmapUsed = (arcLines.status & ReferenceLineStatus.DETECTORMAP_USED) != 0
    dmapReserved = (arcLines.status & ReferenceLineStatus.DETECTORMAP_RESERVED) != 0

    measured = (
            np.logical_not(np.isnan(arcLines.flux))
            & np.logical_not(np.isnan(arcLines.x))
            & np.logical_not(np.isnan(arcLines.y))
            & np.logical_not(np.isnan(arcLines.xErr))
            & np.logical_not(np.isnan(arcLines.yErr))
            & np.logical_not(np.isnan(arcLines.fluxErr))
    )

    flist = []
    for f in range(fmin, fmax + 1):
        notNan_f = (arcLines.fiberId == f) & measured
        if np.sum(notNan_f) > 0:
            flist.append(f)

    arcLinesMeasured = arcLines[measured]
    residualX = arcLinesMeasured.x - detectorMap.getXCenter(arcLinesMeasured.fiberId,
                                                            arcLinesMeasured.y.astype(np.float64))
    residualW = arcLinesMeasured.wavelength - detectorMap.findWavelength(
        fiberId=arcLinesMeasured.fiberId, row=arcLinesMeasured.y.astype(np.float64)
    )
    minw = np.amin(
        detectorMap.findWavelength(fiberId=arcLinesMeasured.fiberId, row=arcLinesMeasured.y.astype(np.float64)))
    maxw = np.amax(
        detectorMap.findWavelength(fiberId=arcLinesMeasured.fiberId, row=arcLinesMeasured.y.astype(np.float64)))
    bufw = (maxw - minw) * 0.02

    dmUsedMeasured = dmapUsed[measured]
    dmReservedMeasured = dmapReserved[measured]
    if showAllRange:
        residualXMax = max(np.amax(residualX[dmUsedMeasured]), np.amax(residualX[dmReservedMeasured]))
        residualXMin = min(np.amin(residualX[dmUsedMeasured]), np.amin(residualX[dmReservedMeasured]))
        residualWMax = max(np.amax(residualW[dmUsedMeasured]), np.amax(residualW[dmReservedMeasured]))
        residualWMin = min(np.amin(residualW[dmUsedMeasured]), np.amin(residualW[dmReservedMeasured]))
        yxmax = xrange * 3 if residualXMax < xrange * 3 else residualXMax * 1.05
        yxmin = -xrange * 3 if residualXMin > -xrange * 3 else residualXMin * 1.05
        ywmax = wrange * 3 if residualWMax < wrange * 3 else residualWMax * 1.05
        ywmin = -wrange * 3 if residualWMin > -wrange * 3 else residualWMin * 1.05
    else:
        yxmax = xrange * 3
        yxmin = -xrange * 3
        ywmax = wrange * 3
        ywmin = -wrange * 3
        largeX = residualX > yxmax
        smallX = residualX < yxmin
        largeW = residualW > ywmax
        smallW = residualW < ywmin

    # Set up the figure and the axes
    fig1 = plt.figure()
    ax1 = [
        plt.axes([0.08, 0.08, 0.37, 0.36]),
        plt.axes([0.46, 0.08, 0.07, 0.36]),
        plt.axes([0.08, 0.54, 0.37, 0.36]),
        plt.axes([0.46, 0.54, 0.07, 0.36]),
        plt.axes([0.58, 0.08, 0.37, 0.36]),
        plt.axes([0.58, 0.54, 0.37, 0.36]),
    ]
    bl_ax = ax1[0]
    bl_hist_ax = ax1[1]
    tl_ax = ax1[2]
    tl_hist_ax = ax1[3]
    br_ax = ax1[4]
    tr_ax = ax1[5]

    # X center residual of 'used' data.
    bl_ax.scatter(
        arcLinesMeasured.wavelength[dmUsedMeasured],
        residualX[dmUsedMeasured],
        s=pointSize,
        c="b",
        label="DETECTORMAP_USED\n(median:{:.2e}, sigma:{:.2e})".format(
            np.median(residualX[dmUsedMeasured]), iqr(residualX[dmUsedMeasured]) / 1.349
        ),
    )
    # Show full range on X center plot if requested.
    if not showAllRange:
        if np.sum(largeX) + np.sum(smallX) > 0:
            bl_ax.quiver(arcLinesMeasured.wavelength[dmUsedMeasured & largeX],
                         np.zeros(np.sum(
                             dmUsedMeasured & largeX)) + yxmax - xrange * quivLength, 0,
                         xrange * quivLength,
                         label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                             yxmax, np.sum(dmUsedMeasured & largeX) / np.sum(dmUsedMeasured) * 100
                         ),
                         color="b",
                         angles="xy",
                         scale_units="xy",
                         scale=2,
                         )
            bl_ax.quiver(
                arcLinesMeasured.wavelength[dmUsedMeasured & smallX],
                np.zeros(np.sum(dmUsedMeasured & smallX)) + yxmin + xrange * quivLength, 0,
                -xrange * quivLength,
                color="b",
                angles="xy",
                scale_units="xy",
                scale=2,
            )

    # X center residual of 'reserved' data.
    bl_ax.scatter(
        arcLinesMeasured.wavelength[dmReservedMeasured],
        residualX[dmReservedMeasured],
        s=pointSize,
        c="r",
        label="DETECTORMAP_RESERVED\n(median:{:.2e}, sigma:{:.2e})".format(
            np.median(residualX[dmReservedMeasured]), iqr(residualX[dmReservedMeasured]) / 1.349
        ),
    )
    # Show full range on X center plot if requested.
    if not showAllRange:
        if np.sum(largeX) + np.sum(smallX) > 0:
            bl_ax.quiver(arcLinesMeasured.wavelength[dmReservedMeasured & largeX],
                         np.zeros(np.sum(
                             dmReservedMeasured & largeX)) + yxmax - xrange * quivLength,
                         0,
                         xrange * quivLength,
                         label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                             yxmax,
                             (np.sum(dmReservedMeasured & largeX) + np.sum(dmReservedMeasured & smallX))
                             / np.sum(dmReservedMeasured)
                             * 100,
                         ),
                         color="r",
                         angles="xy",
                         scale_units="xy",
                         scale=2,
                         )
            bl_ax.quiver(
                arcLinesMeasured.wavelength[dmReservedMeasured & smallX],
                np.zeros(np.sum(dmReservedMeasured & smallX)) + yxmin + xrange * quivLength,
                0,
                -xrange * quivLength,
                color="r",
                angles="xy",
                scale_units="xy",
                scale=2,
            )

    # X center residual histogram of 'used'.
    bl_hist_ax.hist(
        residualX[dmUsedMeasured],
        color="b",
        range=(-xrange * 3, xrange * 3),
        bins=35,
        orientation="horizontal",
    )
    # X center residual histogram 'reserved'.
    bl_hist_ax.hist(
        residualX[dmReservedMeasured],
        color="r",
        range=(-xrange * 3, xrange * 3),
        bins=35,
        orientation="horizontal",
    )

    # Wavelength residual of 'used' data.
    tl_ax.scatter(
        arcLinesMeasured.wavelength[dmUsedMeasured & (arcLinesMeasured.description != "Trace")],
        residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")],
        s=pointSize,
        c="b",
        label="DETECTORMAP_USED\n(median:{:.2e}, sigma:{:.2e})".format(
            np.median(residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")]),
            iqr(residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")]) / 1.349,
        ),
    )
    if not showAllRange:
        if np.sum(largeW) + np.sum(smallW) > 0:
            tl_ax.quiver(
                arcLinesMeasured.wavelength[dmUsedMeasured & largeW],
                np.zeros(np.sum(dmUsedMeasured & largeW)) + ywmax - wrange * quivLength, 0,
                wrange * quivLength,
                label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                    ywmax,
                    (np.sum(dmUsedMeasured & largeW) + np.sum(dmUsedMeasured & smallW))
                    / np.sum(dmUsedMeasured)
                    * 100,
                ),
                color="b",
                angles="xy",
                scale_units="xy",
                scale=2,
            )
            tl_ax.quiver(
                arcLinesMeasured.wavelength[dmUsedMeasured & smallW],
                np.zeros(np.sum(dmUsedMeasured & smallW)) + ywmin + wrange * quivLength, 0,
                wrange * quivLength,
                color="b",
                angles="xy",
                scale_units="xy",
                scale=2,
            )

    # Wavelength residual of 'reserved' data.
    tl_ax.scatter(
        arcLinesMeasured.wavelength[dmReservedMeasured & (arcLinesMeasured.description != "Trace")],
        residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")],
        s=pointSize,
        c="r",
        label="DETECTORMAP_RESERVED\n(median:{:.2e}, sigma:{:.2e})".format(
            np.median(residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")]),
            iqr(residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")]) / 1.349,
        ),
    )
    if not showAllRange:
        if np.sum(largeW) + np.sum(smallW) > 0:
            tl_ax.quiver(
                arcLinesMeasured.wavelength[dmReservedMeasured & largeW],
                np.zeros(np.sum(dmReservedMeasured & largeW)) + ywmax - wrange * quivLength,
                0,
                wrange * quivLength,
                label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                    ywmax,
                    (np.sum(dmReservedMeasured & largeW) + np.sum(dmReservedMeasured & smallW))
                    / np.sum(dmReservedMeasured)
                    * 100,
                ),
                color="r",
                angles="xy",
                scale_units="xy",
                scale=2,
            )
            tl_ax.quiver(
                arcLinesMeasured.wavelength[dmReservedMeasured & smallW],
                np.zeros(np.sum(dmReservedMeasured & smallW)) + ywmin + wrange * quivLength,
                0,
                -wrange * quivLength,
                color="r",
                angles="xy",
                scale_units="xy",
                scale=2,
            )

    # Wavelength residual histogram of 'used'.
    tl_hist_ax.hist(
        residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")],
        color="b",
        range=(-wrange * 3, wrange * 3),
        bins=35,
        orientation="horizontal",
    )
    # Wavelength residual histogram of 'reserved'.
    tl_hist_ax.hist(
        residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")],
        color="r",
        range=(-wrange * 3, wrange * 3),
        bins=35,
        orientation="horizontal",
    )

    # X center residual fiber errors.
    plot_data = arcData.query('isReserved == True')[['fiberId', 'dx', 'status_name']]
    plot_data = plot_data.groupby('fiberId').dx.agg(['median', iqr_sigma]).reset_index()
    label = f'USED\n'
    label += f'median={plot_data["median"].median():>13.03e}\n'
    label += f'sigma  ={plot_data["iqr_sigma"].median():>13.03e}'
    br_ax.errorbar(
        plot_data.fiberId,
        plot_data['median'],
        plot_data['iqr_sigma'],
        ls='',
        marker='.',
        mec='k',
        label=label
    )

    # Wavelength residual fiber errors.
    plot_data = arcData.query('isTrace == False and isReserved == True')[
        ['fiberId', 'dy_nm', 'status_name']]
    plot_data = plot_data.groupby('fiberId').dy_nm.agg(['median', iqr_sigma]).reset_index()
    label = f'USED\n'
    label += f'median={plot_data["median"].median():>13.03e}\n'
    label += f'sigma  ={plot_data["iqr_sigma"].median():>13.03e}'
    tr_ax.errorbar(
        plot_data.fiberId,
        plot_data['median'],
        plot_data['iqr_sigma'],
        ls='',
        marker='.',
        mec='k',
        label=label
    )

    bl_ax.legend(fontsize='small', shadow=True)
    bl_ax.set_ylabel("X residual (pix)")
    bl_ax.set_xlabel("Wavelength (nm)")
    bl_ax.set_xlim(minw - bufw, maxw + bufw)
    bl_ax.set_ylim(yxmin, yxmax)
    bl_ax.set_title("X center residual (unit=pix)")

    bl_hist_ax.set_ylim(yxmin, yxmax)
    bl_hist_ax.set_yticklabels([])

    tl_ax.legend(fontsize='small', shadow=True)
    tl_ax.set_ylabel("Wavelength residual (nm)")
    tl_ax.set_xlabel("Wavelength (nm)")
    tl_ax.set_xlim(minw - bufw, maxw + bufw)
    tl_ax.set_ylim(ywmin, ywmax)
    tl_ax.set_title("Wavelength residual (unit=nm)")

    tl_hist_ax.set_ylim(ywmin, ywmax)
    tl_hist_ax.set_yticklabels([])

    br_ax.set_xlabel("fiberId")
    br_ax.set_ylim(yxmin, yxmax)
    br_ax.set_title("X center residual of each fiber\n(point=median, errbar=1sigma scatter, unit=pix)")
    tr_ax.set_xlabel("fiberId")
    tr_ax.set_ylim(ywmin, ywmax)
    tr_ax.set_title("Wavelength residual of each fiber\n(point=median, errbar=1sigma scatter, unit=nm)")

    return fig1


def plotResiduals2D(arcData: pd.DataFrame,
                    detectorMap: DetectorMap = None,
                    reservedOnly: bool = True,
                    positionCol='dx', wavelengthCol='dy',
                    showWavelength=False,
                    hexBin=False, gridsize=250,
                    plotKws: dict = None,
                    title: str = None,
                    addCursor: bool = False,
                    showLabels: bool = True
                    ) -> Figure:
    """ Plot residuals as a 2D histogram.

    Parameters
    ----------
    arcData : `pd.DataFrame`
        The arc data.
    detectorMap : `DetectorMap`, optional
        The detector map. Default is ``None``.
    reservedOnly : `bool`, optional
        Show only reserved data? Default is ``True``.
    positionCol : `str`, optional
        The column to use for the position. Default is ``'dx'``.
    wavelengthCol : `str`, optional
        The column to use for the wavelength. Default is ``'dy'``.
    showWavelength : `bool`, optional
        Show the wavelength? Default is ``False``.
    hexBin : `bool`, optional
        Use hexbin? Default is ``False``.
    gridsize : `int`, optional
        The gridsize. Default is 250.
    plotKws : `dict`, optional
        The plot keywords. Default is ``None``.
    title : `str`, optional
        The title. Default is ``None``.
    addCursor : `bool`, optional
        Add cursor? Default is ``False``.
    showLabels : `bool`, optional
        Show labels? Default is ``True``.

    Returns
    -------
    fig : `Figure`
    """
    plotKws = plotKws or dict()
    plotKws.setdefault('cmap', div_palette)

    arc_data = arcData

    if reservedOnly is True:
        arc_data = arc_data.query('status_name.str.contains("RESERVED")')

    # Don't use trace for wavelength.
    if showWavelength:
        arc_data = arc_data.query('isTrace == False')

    width = None if detectorMap is None else detectorMap.getBBox().width
    height = None if detectorMap is None else detectorMap.getBBox().height

    # ncols = 2 if showWavelength else 1
    ncols = 1

    fig = Figure()
    ax = fig.add_subplot()

    if showWavelength:
        suptitle = f'{wavelengthCol} {"(pixel)" if wavelengthCol == "dy" else ""}'
        plot_col = arc_data[wavelengthCol]
    else:
        suptitle = f'{positionCol} (pixel)'
        plot_col = arc_data[positionCol]

    norm = colors.Normalize(vmin=plotKws.pop('vmin', None), vmax=plotKws.pop('vmax', None))

    if hexBin:
        im = ax.hexbin(arc_data.x, arc_data.y, plot_col, norm=norm, gridsize=gridsize, **plotKws)
    else:
        im = ax.scatter(arc_data.x, arc_data.y, c=plot_col, s=1, norm=norm, **plotKws)

    stats_string = f'median={plot_col.median():.03e} sigma={iqr_sigma(plot_col):.03e}'

    # Add colorbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.05)
    fig.colorbar(im, ax=ax, cax=cax, orientation='vertical', shrink=0.6, extend='both', label=suptitle)

    ax.set_title(f"{stats_string} \n 2D residuals {suptitle}")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')

    fig.suptitle(title, y=0.975)

    ax.set_xlabel('x position')
    ax.set_ylabel('y position')

    if showLabels is False:
        ax.tick_params('x', bottom=False, labelbottom=False)
        ax.tick_params('y', left=False, labelleft=False)

    if addCursor is True and detectorMap is not None:
        ax.format_coord = addPfsCursor(None, detectorMap)

    return fig


def plotResidual(data, column='dx', use_dm_layout=True, vmin=None, vmax=None, binWavelength=None) -> Figure:
    """Plot the 1D and 2D residuals on a single figure.

    Parameters
    ----------
    data : `pandas.DataFrame`
        The data.
    column : `str`, optional
        The column to use. Default is ``'dx'``.
    use_dm_layout : `bool`, optional
        Use the detector map layout? Default is ``True``.
    vmin : `float`, optional
        The minimum value. Default is ``None``.
    vmax : `float`, optional
        The maximum value. Default is ``None``.

    Returns
    -------
    fig : `Figure`
    """
    # Wavelength residual
    data['bin'] = 1
    bin_wl = False
    if isinstance(binWavelength, (int, float)):
        bins = np.arange(data.wavelength.min() - 1, data.wavelength.max() + 1, binWavelength)
        s_cut, bins = pd.cut(data.wavelength, bins=bins, retbins=True, labels=False)
        data['bin'] = pd.Categorical(s_cut)
        bin_wl = True
        
    plot_data = data.melt(
        id_vars=['fiberId', 'wavelength', 'x', 'y', 'isTrace', 'bin', column],
        value_vars=['isUsed', 'isReserved'],
        var_name='status').query('value == True')

    if column.startswith('dy'):
        plot_data = plot_data.query('isTrace == False').copy()

    reserved_data = plot_data.query('status == "isReserved"')
    if len(reserved_data) == 0:
        return None

    spatial_avg = plot_data.groupby(['fiberId', 'status'])[column].agg(['median', iqr_sigma, 'count']).reset_index()
    stats_df = plot_data.groupby('status')[column].agg(['median', iqr_sigma])

    pal = dict(zip(spatial_avg.status.unique(), plt.cm.tab10.colors))
    pal_colors = [pal[x] for x in spatial_avg.status]

    if column == 'dy_nm':
        units = 'nm'
        vmin = vmin or -0.1
        vmax = vmax or 0.1
    else:
        units = 'pix'
        vmin = vmin or -0.6
        vmax = vmax or 0.6

    fig = plt.figure(figsize=(10, 10), layout='constrained')

    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    # Upper row
    # Fiber residual
    # Just the errors, no markers
    ax0.errorbar(spatial_avg.fiberId, spatial_avg['median'], spatial_avg.iqr_sigma,
                 ls='', ecolor=pal_colors, alpha=0.5,
                 )

    # Scatterplot with outliers marked.
    ax0 = scatterplotWithOutliers(
        spatial_avg,
        'fiberId',
        'median',
        hue='status',
        ymin=vmin,
        ymax=vmax,
        palette=pal,
        ax=ax0,
        refline=0,
    )
    ax0.legend(loc='lower right', shadow=True, prop=dict(family='monospace', weight='bold'), bbox_to_anchor=(1.2, 0))

    if use_dm_layout is True:
        # Reverse the fiber order to match the xy-pixel layout
        ax0.set_xlim(list(reversed(ax0.get_xlim())))

    ax0.set_ylabel(f'Δ {units}')
    ax0.xaxis.set_label_position('top')
    ax0.set_xlabel('')
    ax0.xaxis.tick_top()
    ax0.set_title(f'Median fiber residual and 1-sigma error', weight='bold', fontsize='small')
    legend = ax0.get_legend()
    legend.set_title('')
    legend.set_visible(False)

    # Summary stats area is blank.
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Lower row
    # 2d residual
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    if use_dm_layout:
        X = 'x'
        Y = 'y'
    else:
        X = 'fiberId'
        Y = 'wavelength'

    im = ax2.scatter(reserved_data[X],
                     reserved_data[Y],
                     c=reserved_data[column],
                     norm=norm,
                     cmap=div_palette,
                     s=4
                     )
    fig.colorbar(im, ax=ax2, orientation='horizontal', extend='both', fraction=0.02, aspect=75, pad=0.01)
    ax2.set_ylabel(Y)
    ax2.set_xlabel(X)
    resid_stats = f'{reserved_data[column].median():.06f} {iqr_sigma(reserved_data[column]):.06f}'
    ax2.set_title(f'2D residual of RESERVED', weight='bold', fontsize='small')
        
    if bin_wl is True:
        plot_data = plot_data.groupby(['bin', 'status'])[['wavelength', column]].agg('median', iqr_sigma).dropna().reset_index().sort_values('status')
    
    ax3 = scatterplotWithOutliers(
        plot_data,
        column,
        'wavelength',
        hue='status',
        ymin=vmin,
        ymax=vmax,
        palette=pal,
        ax=ax3,
        refline=0.,
        vertical=True,
        rasterized=True,
    )
    try:
        ax3.get_legend().set_visible(False)
    except AttributeError:
        # Skip missing wavelength legend.
        pass

    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    ax3.set_xlabel(f'Δ {units}')
    ax_title = f'Residual by {"binned" if bin_wl else ""} wavelength'
    if bin_wl:
        ax_title += f'\nbinsize={binWavelength} nm'
    ax3.set_title(ax_title, weight='bold', fontsize='small')

    fig.suptitle(f'DetectorMap Residuals', weight='bold')

    # Make a legend with stats.
    reserved_stats_str = f'RESERVED\n{stats_df.loc["isReserved"].to_string()}'
    used_stats_str = f'USED\n{stats_df.loc["isUsed"].to_string()}'
    handles, labels = ax0.get_legend_handles_labels()
    handles = handles[:2]  # Remove the outliers markers
    labels = [reserved_stats_str, used_stats_str]
    fig.legend(handles=handles,
               labels=labels,
               labelspacing=1,
               prop=dict(family='monospace', weight='bold', size='small'),
               loc='upper right',
               bbox_to_anchor=(0.97, 0.885),
               title=f'Overall Stats ({units})',
               title_fontproperties=dict(weight='bold')
               )

    return fig


def scatterplotWithOutliers(data, X, Y, hue='status_name',
                            ymin=-0.1, ymax=0.1, palette=None,
                            ax=None, refline=None, vertical=False,
                            rasterized=False,
                            ) -> Axes:
    """Make a scatterplot with outliers marked.

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
        The reference line. Default is ``None``.
    vertical : `bool`, optional
        Is the plot vertical? Default is ``False``.
    rasterized : `bool`, optional
        Rasterize the plot? Default is ``False``.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
    """
    ax = sb.scatterplot(
        data=data,
        x=X,
        y=Y,
        hue=hue,
        s=20,
        ec='k',
        zorder=100,
        palette=palette,
        rasterized=rasterized,
        ax=ax
    )

    pos = data.query(f'{X if vertical else Y} >= @ymax').copy()
    pos[X if vertical else Y] = ymax
    neg = data.query(f'{X if vertical else Y} <= @ymin').copy()
    neg[X if vertical else Y] = ymin

    marker = 'v'
    if vertical is True:
        marker = '<'

    sb.scatterplot(data=pos, x=X, y=Y, hue=hue, palette=palette,
                   legend=False,
                   marker=marker, ec='k', lw=0.5, s=100,
                   clip_on=False, zorder=100, ax=ax
                   )

    marker = '^'
    if vertical is True:
        marker = '>'
    sb.scatterplot(data=neg, x=X, y=Y, hue=hue, palette=palette,
                   legend=False,
                   marker=marker, ec='k', lw=0.5, s=100,
                   clip_on=False, zorder=100, ax=ax
                   )

    # Reference line.
    if isinstance(refline, (float, int)):
        if vertical:
            ax.axvline(refline, color='k', ls='--', alpha=0.5, zorder=-100)
        else:
            ax.axhline(refline, color='k', ls='--', alpha=0.5, zorder=-100)

    if vertical is True:
        ax.set_xlim(ymin, ymax)
    else:
        ax.set_ylim(ymin, ymax)

    return ax
