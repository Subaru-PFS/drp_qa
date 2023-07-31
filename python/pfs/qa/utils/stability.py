from typing import Dict, Any

import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pfs.drp.stella import ReferenceLineStatus, ArcLineSet, DetectorMap, PfsArm
from pfs.drp.stella.utils import addPfsCursor
from scipy.stats import iqr


def getArclineData(als: ArcLineSet,
                   dropNa: bool = True,
                   removeFlagged: bool = True,
                   oneHotStatus: bool = False,
                   includeTrace: bool = True,
                   statusTypes=None
                   ) -> pd.DataFrame:
    """Gets a copy of the arcline data, with some columns added.

    Parameters
    ----------
    als : `pfs.drp.stella.arcLine.ArcLineSet`
        Arc lines.
    dropNa : `bool`, optional
        Drop rows with NaN values? Default is True.
    removeFlagged : `bool`, optional
        Remove rows with ``flag=True``? Default is True.
    oneHotStatus : `bool`, optional
        Add one-hot columns for the status? Default is False.
    includeTrace : `bool`, optional
        Include rows with ``Trace=True``? Default is False.
    statusTypes : `list` of `pfs.drp.stella.ReferenceLineStatus`, optional
        Status types to include. Default is ``[DETECTORMAP_RESERVED, DETECTORMAP_USED]``.

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """
    if statusTypes is None:
        statusTypes = [ReferenceLineStatus.DETECTORMAP_RESERVED,
                       ReferenceLineStatus.DETECTORMAP_USED]
    arc_data = als.data.copy()

    if dropNa:
        arc_data = arc_data.dropna()

    if removeFlagged:
        arc_data = arc_data.query('flag == False')

    if len(statusTypes):
        arc_data = arc_data.query(' or '.join(f'status == {s}' for s in statusTypes))

    arc_data = arc_data.copy()

    # Get status names.
    arc_data['status_name'] = arc_data.status.map({v: k for k, v in ReferenceLineStatus.__members__.items()})
    arc_data['status_name'] = arc_data['status_name'].astype('category')

    # Get one hot for status.
    if oneHotStatus:
        arc_data = arc_data.merge(pd.get_dummies(arc_data.status_name), left_index=True, right_index=True)

    # Make a one-hot for the Trace.
    try:
        arc_data['Trace'] = arc_data.description.str.get_dummies()['Trace']
    except KeyError:
        arc_data['Trace'] = False

    if includeTrace is False:
        arc_data = arc_data.query(f'Trace == False').copy()

    return arc_data


def addTraceLambdaToArclines(arc_data: pd.DataFrame,
                             detectorMap: DetectorMap) -> pd.DataFrame:
    """Adds detector map trace position and wavelength to arcline data.

    Parameters
    ----------
    arc_data : `pandas.DataFrame`
        Arc line data.
    detectorMap : `pfs.drp.stella.DoubleDetectorMap.DoubleDetectorMap`
        Detector map.

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """
    dispersion = detectorMap.getDispersion(arc_data.fiberId.values, arc_data.wavelength.values)

    arc_data['lam'] = detectorMap.findWavelength(arc_data.fiberId.values, arc_data.y.values)
    arc_data['lamErr'] = (arc_data.yErr / arc_data.y) * (arc_data.lam / dispersion)
    arc_data['lamErr_nm'] = arc_data.lamErr * dispersion
    points = detectorMap.findPoint(arc_data.fiberId.values, arc_data.wavelength.values)
    arc_data['tracePosX'] = points[:, 0]
    arc_data['tracePosY'] = points[:, 1]

    return arc_data


def addResidualsToArclines(arc_data: pd.DataFrame, fitYTo: str = 'y') -> pd.DataFrame:
    """Adds residuals to arcline data.

    This will calculate residuals for the X-center position and wavelength.

    Adds the following columns to the arcline data:

    - ``dx``: X-center position minus trace position (from DetectorMap `findPoint`).
    - ``dxResidualLinear``: Linear fit to ``dx``.
    - ``dxResidualMean``: Mean fit to ``dx``.
    - ``dxResidualMedian``: Median fit to ``dx``.
    - ``dy``: Y-center position minus trace position (from DetectorMap `findPoint`).
    - ``dyResidualLinear``: Linear fit to ``dy``.
    - ``dyResidualMean``: Mean fit to ``dy``.
    - ``dyResidualMedian``: Median fit to ``dy``.
    - ``dy_nm``: Wavelength minus trace wavelength.
    - ``dy_nmResidualLinear``: Linear fit to ``dy_nm``.
    - ``dy_nmResidualMean``: Mean fit to ``dy_nm``.
    - ``dy_nmResidualMedian``: Median fit to ``dy_nm``.
    - ``centroidErr``: Hypotenuse of ``xErr`` and ``yErr``.
    - ``detectorMapErr``: Hypotenuse of ``dx`` and ``dy``.


    Parameters
    ----------
    arc_data : `pandas.DataFrame`
        Arc line data.
    fitYTo : `str`, optional
        Column to fit Y to. Default is ``y``, could also be ``wavelength``.

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """

    arc_data['dx'] = arc_data.x - arc_data.tracePosX
    calculateResiduals(arc_data, 'dx', fitYTo=fitYTo)

    arc_data['dy'] = arc_data.y - arc_data.tracePosY
    calculateResiduals(arc_data, 'dy', fitYTo=fitYTo)

    arc_data['dy_nm'] = (arc_data.wavelength - arc_data.lam)
    calculateResiduals(arc_data, 'dy_nm', fitYTo=fitYTo)

    arc_data['centroidErr'] = np.hypot(arc_data.xErr, arc_data.yErr)
    arc_data['detectorMapErr'] = np.hypot(arc_data.dx, arc_data.dy)

    return arc_data


def calculateResiduals(arc_data: pd.DataFrame, targetCol: str, fitXTo: str = 'fiberId',
                       fitYTo: str = 'wavelength') -> pd.DataFrame:
    """Calculates residuals.

    This will calculate residuals for the X-center position and wavelength.

    Adds the following columns to the arcline data:

    - ``dx``: X-center position minus trace position (from DetectorMap `findPoint`).
    - ``dxResidualLinear``: Linear fit to ``dx``.
    - ``dxResidualMean``: Mean fit to ``dx``.
    - ``dxResidualMedian``: Median fit to ``dx``.

    Parameters
    ----------
    arc_data : `pandas.DataFrame`
        Arc line data.
    targetCol : `str`
        Column to calculate residuals for.
    fitXTo : `str`, optional
        Column to fit X to. Default is ``fiberId``, could also be ``x``.
    fitYTo : `str`, optional
        Column to fit Y to. Default is ``wavelength``, could also be ``y``.

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """
    # Linear fit
    a = arc_data[fitXTo]
    b = arc_data[fitYTo]

    X = np.vstack([np.ones_like(a), a, b]).T
    Y = arc_data[targetCol]

    c0, c1, c2 = np.linalg.lstsq(X, Y, rcond=None)[0]

    fit = c0 + (c1 * a) + (c2 * b)
    arc_data[f'{targetCol}Fit'] = fit
    arc_data[f'{targetCol}ResidualLinear'] = Y - fit

    # Mean and median fits
    arc_data[f'{targetCol}ResidualMean'] = Y - Y.mean()
    arc_data[f'{targetCol}ResidualMedian'] = Y - Y.median()

    return arc_data


def plotArcResiduals(arc_data: pd.DataFrame,
                     title: str = "",
                     fitType: str = "mean",
                     lamErrMax: float = 0) -> Figure:
    """
    Plot arc line residuals.

    Parameters
    ----------
    arc_data: `pandas.DataFrame`
        Arc line data.
    title: `str`
        Title for plot.
    fitType: `str`
        Type of fit to use for residuals. Default is ``mean``, could also be ``median`` or ``linear``.
    lamErrMax: `float`
        Maximum wavelength error to include in plot. Default is 0.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
    """
    totalFibers = len(arc_data.fiberId.unique())

    groups = ['DETECTORMAP_USED', 'DETECTORMAP_RESERVED']

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=False, gridspec_kw=dict(hspace=0.3))

    for group in groups:

        plot_data = arc_data.query(f'status == {ReferenceLineStatus[group]}').copy()
        plot_data.status_name = plot_data.status_name.cat.remove_unused_categories()

        if lamErrMax > 0:
            plot_data = plot_data.query(f'lamErr < {lamErrMax}').copy()

        position_col = f'dxResidual{fitType.title()}'
        wavelength_col = f'dy_nmResidual{fitType.title()}'

        # Wavelength
        wavelength_data = plot_data.query(f'Trace == False')
        label = f"{group}\n" \
                f"{fitType}={wavelength_data.dy_nm.median():.02e} " \
                f"σ={iqr(wavelength_data.dy_nm) / 1.349:.3e} pix"
        sb.scatterplot(data=wavelength_data, x='wavelength', y=wavelength_col, marker='.', ax=ax0, label=label)
        ax0.axhline(0, color='k', zorder=-1)
        ax0.set_xlabel('wavelength (nm)')
        ax0.set_ylabel('Wavelength residual (nm)')
        ax0.set_title(f"Wavelength residual")
        # ax0.set_ylim(-.1, .1)

        # X-center
        label = f"{group}\n{fitType}={plot_data.dx.median():.02e} σ={iqr(plot_data.dx) / 1.349:.3e} pix "
        sb.scatterplot(data=plot_data, x='wavelength', y=position_col, marker='.', ax=ax1, label=label)
        ax1.axhline(0, color='k', zorder=-1)
        ax1.set_xlabel('row (pixel)')
        ax1.set_ylabel('Position residual (pix)')
        ax1.set_title('Position residual')
        # ax1.set_ylim(-.8, .8)

        fig.suptitle(f"{title}\n "
                     f"$\sigma_\lambda$ < {lamErrMax} = {len(plot_data.fiberId.unique()):.0f} / {totalFibers} lines")

    return fig


def plotArcResiduals1D(plot_data: pd.DataFrame, col: str = 'dx', title: str = '', fitType: str = 'mean',
                       fig: Figure = None) -> Figure:
    """
    Plot 1D arc line residuals.

    Parameters
    ----------
    plot_data: `pandas.DataFrame`
        Arc line data.
    col: `str`
        Column to plot.
    title: `str`
        Title for plot.
    fitType: `str`
        Type of fit to use for residuals. Default is ``mean``, could also be ``median`` or ``linear``.
    fig: `matplotlib.figure.Figure`
        Figure to plot on. Default is to create a new figure.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
    """
    fig = fig or plt.figure()

    gs = GridSpec(1, 3, width_ratios=[3, 1, 3])

    residual_col = f'{col}Residual{fitType.title()}'

    used = plot_data.query(f'status == {ReferenceLineStatus.DETECTORMAP_USED}').copy()
    reserved = plot_data.query(f'status == {ReferenceLineStatus.DETECTORMAP_RESERVED}').copy()

    used.status_name = used.status_name.cat.remove_unused_categories()
    reserved.status_name = reserved.status_name.cat.remove_unused_categories()

    used_label = f"{'used': >15s} " \
                 f"{fitType}={used[residual_col].median():+.02e} " \
                 f"σ={iqr(used[residual_col]) / 1.349:+.3e}"
    reserved_label = f"{'reserved': >15s} " \
                     f"{fitType}={reserved[residual_col].median():+.02e} " \
                     f"σ={iqr(reserved[residual_col]) / 1.349:+.3e}"

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    ax0.axhline(0, color='k', zorder=-1)
    ax2.axhline(0, color='k', zorder=-1)

    for color, in_trace in zip(['b', 'r'], [True, False]):
        marker = 'x' if in_trace else '.'
        sb.scatterplot(data=used.query(f'Trace == {in_trace}'), x='wavelength', y=residual_col, color='b',
                       marker=marker, ax=ax0, label=f'{used_label}')
        sb.scatterplot(data=reserved.query(f'Trace == {in_trace}'), x='wavelength', y=residual_col, color='r',
                       marker=marker, ax=ax0, label=f'{reserved_label}')

        sb.histplot(data=used.query(f'Trace == {in_trace}'), y=residual_col, bins=100, color='b', ax=ax1, label='used')
        sb.histplot(data=reserved.query(f'Trace == {in_trace}'), y=residual_col, bins=100, color='r', ax=ax1,
                    label='reserved')

        used_median = used.query(f'Trace == {in_trace}').groupby('fiberId').median().reset_index()
        sb.scatterplot(data=used_median, x='fiberId', y=residual_col, ax=ax2, marker=marker,
                       label=f'Trace == {in_trace}')

    fig.suptitle(f"{title}\n{col}")

    return fig


def plotResidualsQuiver(arc_data: pd.DataFrame, detectorMap: DetectorMap,
                        title: str = '', cmap: str = 'coolwarm',
                        arrowSize: float = 0.1, arrowScale: int = 100,
                        maxCentroidErr: float = 0.1, maxDetectorMapErr: float = 1, limitMax: bool = False) -> Figure:
    """
    Plot residuals as a quiver plot.

    Parameters
    ----------
    arc_data: `pandas.DataFrame`
        Arc line data.
    detectorMap: `DetectorMap`
        Detector map.
    title: `str`
        Title for plot.
    cmap: `str`
        Colormap to use. Default is ``coolwarm``.
    arrowSize: `float`
        Size of arrows. Default is 0.1.
    arrowScale: `int`
        Scale of arrows. Default is 100.
    maxCentroidErr: `float`
        Maximum centroid error to plot. Default is 0.1.
    maxDetectorMapErr: `float`
        Maximum detector map error to plot. Default is 1.
    limitMax: `bool`
        Limit the maximum centroid and detector map errors. Default is ``False``.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
    """
    fig, ax0 = plt.subplots(1, 1)

    if limitMax:
        limited_arc_data = arc_data.query(f'centroidErr < {maxCentroidErr} and detectorMapErr < {maxDetectorMapErr}')
        len_diff = len(arc_data) - len(limited_arc_data)
        print(f'Entries outside {maxCentroidErr=} and {maxDetectorMapErr=}: {len_diff} of {len(arc_data)}')
        arc_data = limited_arc_data

    C = np.hypot(arc_data.dx, arc_data.dy)
    im = ax0.quiver(arc_data.x, arc_data.y, arc_data.dx, arc_data.dy,
                    C, cmap=cmap,
                    # alpha=0.75,
                    angles='xy', scale_units='xy', scale=(arrowSize * arrowScale) / detectorMap.getBBox().getHeight()
                    )
    ax0.quiverkey(im, 0.1, 1.075, arrowSize, label=f"{arrowSize} pixels")
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size='3%', pad=0.02)
    fig.colorbar(im, ax=ax0, cax=cax, orientation='vertical', shrink=0.6)

    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    ax0.format_coord = addPfsCursor(None, detectorMap)
    ax0.set_aspect('equal')

    bbox = detectorMap.getBBox()
    ax0.set_xlim(bbox.getMinX(), bbox.getMaxX())
    ax0.set_ylim(bbox.getMinY(), bbox.getMaxY())

    fig.suptitle(f"{title}\nResidual quiver")

    return fig


def plotArcResiduals2D(arc_data, detectorMap, title="",
                       maxCentroidErr=0.1, maxDetectorMapErr=1,
                       limitMax=False, showY=True,
                       percentiles=None,
                       hexBin=False, gridsize=100,
                       cmap='coolwarm') -> Figure:
    """ Plot residuals as a 2D histogram.

    Parameters
    ----------
    arc_data: `pandas.DataFrame`
        Arc line data.
    detectorMap: `DetectorMap`
        Detector map.
    title: `str`
        Title for plot.
    maxCentroidErr: `float`
        Maximum centroid error to plot. Default is 0.1.
    maxDetectorMapErr: `float`
        Maximum detector map error to plot. Default is 1.
    limitMax: `bool`
        Limit the maximum centroid and detector map errors. Default is ``False``.
    showY: `bool`
        Show the y-axis. Default is ``True``.
    percentiles: `list`
        Percentiles to use for the color scale. Default is ``[25, 75]``.
    hexBin: `bool`
        Use hexbin plot. Default is ``False``.
    gridsize: `int`
        Grid size for hexbin plot. Default is 100.
    cmap: `str`
        Colormap to use. Default is ``coolwarm``.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
    """
    if percentiles is None:
        percentiles = [25, 75]

    if limitMax:
        limited_arc_data = arc_data.query(f'centroidErr < {maxCentroidErr} and detectorMapErr < {maxDetectorMapErr}')
        len_diff = len(arc_data) - len(limited_arc_data)
        print(f'Entries outside {maxCentroidErr=} and {maxDetectorMapErr=}: {len_diff} of {len(arc_data)}')
        arc_data = limited_arc_data

    ncols = 2 if showY else 1

    bbox = detectorMap.getBBox()

    fig, axes = plt.subplots(1, ncols, sharex=True, sharey=True, constrained_layout=True)

    def _make_subplot(ax, data, subtitle=''):
        vmin, vmax = np.percentile(data, percentiles)
        # vmax = data.std()
        # vmin = -vmax
        # vmin = data.min()
        # vmax = data.max()
        norm = colors.Normalize(vmin, vmax)

        if hexBin:
            im = ax.hexbin(arc_data.x, arc_data.y, data, norm=norm, cmap=cmap, gridsize=gridsize)
        else:
            im = ax.scatter(arc_data.x, arc_data.y, c=data, s=3, norm=norm, cmap=cmap)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.format_coord = addPfsCursor(None, detectorMap)
        ax.set_aspect('equal')
        ax.set_title(f"{subtitle}")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.02)
        fig.colorbar(im, ax=ax, cax=cax, orientation='vertical', shrink=0.6)

        ax.set_xlim(bbox.getMinX(), bbox.getMaxX())
        ax.set_ylim(bbox.getMinY(), bbox.getMaxY())

    _make_subplot(axes[0], arc_data.dxResidualMedian, subtitle='dx mean [pixel]')

    if showY:
        _make_subplot(axes[1], arc_data.dyResidualMedian, subtitle='dy mean [nm]')

    plt.suptitle(f"Residual {title}")

    return fig


def getStatistics(arc_data: pd.DataFrame, pfsArm: PfsArm) -> Dict[str, Any]:
    dmapUsed = arc_data.query(f'status == {ReferenceLineStatus.DETECTORMAP_USED}')
    dmapReserved = arc_data.query(f'status == {ReferenceLineStatus.DETECTORMAP_RESERVED}')

    statistics = {
        "fiberId": arc_data.fiberId.unique(),
        "MedianXusedAll": np.nanmedian(dmapUsed.dx),
        "MedianXreservedAll": np.nanmedian(dmapReserved.dx),
        "MedianWusedAll": np.nanmedian(dmapUsed.query('Trace == False').dy),
        "MedianWreservedAll": np.nanmedian(dmapReserved.query('Trace == False').dy),
        "SigmaXusedAll": iqr(dmapUsed.dx) / 1.349,
        "SigmaXreservedAll": iqr(dmapReserved.dx) / 1.349,
        "SigmaWusedAll": iqr(dmapUsed.query('Trace == False').dy) / 1.349,
        "SigmaWreservedAll": iqr(dmapReserved.query('Trace == False').dy) / 1.349
    }
    dictkeys = [
        "N_Xused",
        "N_Xreserved",
        "N_Wused",
        "N_Wreserved",
        "Sigma_Xused",
        "Sigma_Xreserved",
        "Sigma_Wused",
        "Sigma_Wreserved",
        "Median_Xused",
        "Median_Xreserved",
        "Median_Wused",
        "Median_Wreserved",
        "pfsArmFluxMedian",
    ]
    for k in dictkeys:
        statistics[k] = np.array([])
    for f in arc_data.fiberId.unique():
        dmapUsedFiber = dmapUsed.query(f'fiberId == {f}')
        dmapUsedFiberNoTrace = dmapUsedFiber.query('Trace == False')
        dmapReservedFiber = dmapReserved.query(f'fiberId == {f}')
        dmapReservedFiberNoTrace = dmapReservedFiber.query('Trace == False')

        dictvalues = [
            len(dmapUsedFiber),
            len(dmapReservedFiber),
            len(dmapUsedFiberNoTrace),
            len(dmapReservedFiberNoTrace),
            iqr(dmapUsedFiber.dx) / 1.349,
            iqr(dmapReservedFiber.dx) / 1.349,
            iqr(dmapUsedFiberNoTrace.dy) / 1.349,
            iqr(dmapReservedFiberNoTrace.dy) / 1.349,
            dmapUsedFiber.dx.median(),
            dmapReservedFiber.dx.median(),
            dmapUsedFiberNoTrace.dy.median(),
            dmapReservedFiberNoTrace.dy.median(),
            pd.DataFrame(pfsArm.flux[pfsArm.fiberId == f].T).median().values[0],
        ]
        for k, v in zip(dictkeys, dictvalues):
            statistics[k] = np.append(statistics[k], v)

    return statistics
