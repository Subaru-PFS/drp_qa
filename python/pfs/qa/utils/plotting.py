import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr
import pandas as pd

from matplotlib import colors
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pfs.drp.stella.utils import addPfsCursor
from pfs.drp.stella import ArcLineSet, DetectorMap, ReferenceLineStatus

div_palette = plt.cm.RdBu_r.with_extremes(over='magenta', under='cyan', bad='lime')


def iqr_sigma(x):
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

    Parameters:
        arcLines: The arc lines.
        detectorMap: The detector map.

    Returns:
        The figure.
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
    positionCol: `str`
        The column to plot for the position residuals. Default is 'dx'.
    wavelengthCol: `str`
        The column to plot for the wavelength residuals. Default is 'dy'.
    showWavelength: `bool`
        Show the y-axis. Default is ``True``.
    hexBin: `bool`
        Use hexbin plot. Default is ``True``.
    gridsize: `int`
        Grid size for hexbin plot. Default is 250.
    plotKws: `dict`
        Arguments passed to plotting function.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
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
