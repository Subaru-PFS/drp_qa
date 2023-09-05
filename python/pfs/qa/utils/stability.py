from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pfs.drp.stella import ReferenceLineStatus, ArcLineSet, DetectorMap, PfsArm
from scipy.stats import iqr


def getArclineData(als: ArcLineSet,
                   dropNa: bool = False,
                   dropColumns: Optional[list] = None,
                   removeFlagged: bool = False,
                   oneHotStatus: bool = False,
                   removeTrace: bool = False,
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
    removeTrace : `bool`, optional
        Remove rows with ``Trace==True``? Default is False.
    statusTypes : `list` of `pfs.drp.stella.ReferenceLineStatus`, optional
        Status types to include. Default is ``[DETECTORMAP_RESERVED, DETECTORMAP_USED]``.
    ignoreLines : `list` of

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """
    if statusTypes is None:
        statusTypes = [ReferenceLineStatus.DETECTORMAP_RESERVED,
                       ReferenceLineStatus.DETECTORMAP_USED]
    arc_data = als.data.copy()

    if dropColumns is not None:
        arc_data = arc_data.drop(columns=dropColumns)

    if removeFlagged:
        arc_data = arc_data.query('flag == False')

    if len(statusTypes):
        arc_data = arc_data.query(' or '.join(f'status == {s}' for s in statusTypes))

    if dropNa:
        arc_data = arc_data.dropna()

    arc_data = arc_data.copy()

    # Change some of the dtypes explicitly.
    arc_data.y = arc_data.y.astype(np.float64)

    # Get status names.
    arc_data['status_name'] = arc_data.status.map(lambda x: str(ReferenceLineStatus(x)).split('.')[-1])
    arc_data['status_name'] = arc_data['status_name'].astype('category')

    # Clean up categories.
    ignore_lines = [
        'REJECTED',
        'NOT_VISIBLE',
    ]

    # Ignore bad line categories.
    for ignore in ignore_lines:
        arc_data = arc_data[~arc_data.status_name.str.contains(ignore)]

    arc_data.status_name = arc_data.status_name.cat.remove_unused_categories()

    # Get one hot for status.
    if oneHotStatus:
        arc_data = arc_data.merge(pd.get_dummies(arc_data.status_name), left_index=True, right_index=True)

    # Make a one-hot for the Trace.
    try:
        arc_data['Trace'] = arc_data.description.str.get_dummies()['Trace']
    except KeyError:
        arc_data['Trace'] = False

    if removeTrace is True:
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
    # Get the wavelength according to the detectormap for fiberId.
    arc_data['lam'] = detectorMap.findWavelength(arc_data.fiberId.to_numpy(), arc_data.y.to_numpy())
    arc_data['lamErr'] = arc_data.yErr * arc_data.lam / arc_data.y

    # Convert nm to pixels.
    dispersion = detectorMap.getDispersion(arc_data.fiberId.to_numpy(), arc_data.wavelength.to_numpy())
    arc_data['lam_pix'] = arc_data.lam / dispersion
    arc_data['lamErr_pix'] = arc_data.lamErr / dispersion

    # Get the trace positions according to the detectormap.
    points = detectorMap.findPoint(arc_data.fiberId.to_numpy(), arc_data.wavelength.to_numpy())
    arc_data['tracePosX'] = points[:, 0]
    arc_data['tracePosY'] = points[:, 1]

    # Get `observed - expected`.
    arc_data['dx'] = arc_data.tracePosX - arc_data.x
    arc_data['dy_nm'] = arc_data.lam - arc_data.wavelength
    arc_data['dy'] = arc_data.dy_nm / dispersion

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
    calculateResiduals(arc_data, 'dx', fitYTo=fitYTo)
    calculateResiduals(arc_data, 'dy', fitYTo=fitYTo)

    arc_data['centroidErr'] = np.hypot(arc_data.xErr, arc_data.yErr)
    arc_data['detectorMapErr'] = np.hypot(arc_data.dx, arc_data.dy)

    return arc_data


def calculateResiduals(arc_data: pd.DataFrame, targetCol: str, fitXTo: str = 'fiberId',
                       fitYTo: str = 'y') -> pd.DataFrame:
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


def plotResidualsQuiver(arc_data: pd.DataFrame, title: str = '', arrowScale: float = 0.01, usePixels=True,
                        width: int = None, height: int = None) -> Figure:
    """
    Plot residuals as a quiver plot.

    Parameters
    ----------
    arc_data: `pandas.DataFrame`
        Arc line data.
    title: `str`
        Title for plot.
    arrowScale: `float`
        Scale for quiver arrows.
    usePixels: `bool`
        If wavelength should be plotted in pixels, default True.
    width: `int`
        Width of the detectormap.
    height: `int`
        Height of the detectormap.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
    """
    fig, ax0 = plt.subplots(1, 1)

    wavelength_col = arc_data.dy if usePixels is True else arc_data.dy_nm

    C = np.hypot(arc_data.dx, wavelength_col)
    im = ax0.quiver(arc_data.tracePosX, arc_data.tracePosY, arc_data.dx, wavelength_col, C,
                    norm=colors.Normalize(vmin=0, vmax=1),
                    angles='xy', scale_units='xy', scale=arrowScale, units='width',
                    cmap='coolwarm')
    ax0.quiverkey(im, 0.1, 1., arrowScale, label=f'{arrowScale=}')

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size='3%', pad=0.02)
    fig.colorbar(im, ax=ax0, cax=cax, orientation='vertical', shrink=0.6)
    ax0.set_aspect('equal')
    fig.suptitle(f"Residual quiver\n{title}")

    ax0.set_xlim(0, width)
    ax0.set_ylim(0, height)

    return fig


def plotArcResiduals2D(arc_data,
                       positionCol='dx', wavelengthCol='dy',
                       showWavelength=True,
                       hexBin=False, gridsize=100,
                       width: int = None, height: int = None) -> Figure:
    """ Plot residuals as a 2D histogram.

    Parameters
    ----------
    arc_data: `pandas.DataFrame`
        Arc line data.
    positionCol: `str`
        The column to plot for the position residuals. Default is 'dx'.
    wavelengthCol: `str`
        The column to plot for the wavelength residuals. Default is 'dy'.
    showWavelength: `bool`
        Show the y-axis. Default is ``True``.
    hexBin: `bool`
        Use hexbin plot. Default is ``False``.
    gridsize: `int`
        Grid size for hexbin plot. Default is 100.
    width: `int`
        Width of the detectormap.
    height: `int`
        Height of the detectormap.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
    """
    ncols = 2 if showWavelength else 1

    fig, axes = plt.subplots(1, ncols, sharex=True, sharey=True)

    def _make_subplot(ax, data, subtitle=''):
        vmin, vmax = -1., 1
        norm = colors.Normalize(vmin, vmax)

        if hexBin:
            im = ax.hexbin(arc_data.x, arc_data.y, data, norm=norm, gridsize=gridsize)
        else:
            im = ax.scatter(arc_data.x, arc_data.y, c=data, s=3, norm=norm)

        ax.set_aspect('equal')
        ax.set_title(f"{subtitle}")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.02)
        fig.colorbar(im, ax=ax, cax=cax, orientation='vertical', shrink=0.6)

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)

    if showWavelength:
        _make_subplot(axes[0], arc_data[positionCol], subtitle=f'{positionCol} [pixel]')
        _make_subplot(axes[1], arc_data[wavelengthCol], subtitle=f'{wavelengthCol} [pixel]')
    else:
        _make_subplot(axes, arc_data[positionCol], subtitle='dx [pixel]')

    return fig


def getStats(arc_data, dataId, hd5_fn=None):
    def sigma(x):
        return iqr(x) / 1.349

    # Get aggregate stats.
    do_agg = ['count', 'mean', 'median', 'std', sigma]

    # For entire detector.
    ccd_stats = arc_data.groupby(['status_name']).agg({'dx': do_agg, 'dy': do_agg})
    ccd_stats = ccd_stats.reset_index().melt(id_vars=['status_name'], var_name=['col', 'metric'])
    ccd_stats.insert(0, 'visit', dataId['visit'])
    ccd_stats.insert(1, 'ccd', dataId['arm'] + str(dataId['spectrograph']))

    # Per fiber.
    fiber_stats = arc_data.groupby(['fiberId', 'status_name']).agg({'dx': do_agg, 'dy': do_agg})
    fiber_stats = fiber_stats.reset_index().melt(id_vars=['fiberId', 'status_name'], var_name=['col', 'metric'])
    fiber_stats.insert(0, 'visit', dataId['visit'])
    fiber_stats.insert(1, 'ccd', dataId['arm'] + str(dataId['spectrograph']))

    # # By wavelength (?).
    # wavelength_stats = arc_data.groupby(['wavelength', 'status_name']).agg({'dx': do_agg, 'dy': do_agg})
    # wavelength_stats = wavelength_stats.query('')
    # wavelength_stats = wavelength_stats.reset_index().melt(id_vars=['wavelength', 'status_name'], var_name=['col', 'metric'])
    # wavelength_stats.insert(0, 'visit', dataId['visit'])
    # wavelength_stats.insert(1, 'ccd', dataId['arm'] + str(dataId['spectrograph']))

    if hd5_fn is not None:
        ccd_stats.to_hdf(hd5_fn, key='ccd', format='table', append=True, index=False)
        fiber_stats.to_hdf(hd5_fn, key='fiber', format='table', append=True, index=False)
        # wavelength_stats.to_hdf(hd5_fn, key='wavelength', format='table', append=True, index=False)
    else:
        return ccd_stats, fiber_stats  # , wavelength_stats


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
