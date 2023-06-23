import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pfs.drp.stella import ReferenceLineStatus
from pfs.drp.stella.utils import addPfsCursor
from scipy.stats import iqr


def calculateResiduals(arc_data, col, fitXTo='fiberId', fitYTo='wavelength'):
    # Linear fit
    a = arc_data[fitXTo]
    b = arc_data[fitYTo]

    X = np.vstack([np.ones_like(a), a, b]).T
    Y = arc_data[col]

    c0, c1, c2 = np.linalg.lstsq(X, Y, rcond=None)[0]

    fit = c0 + (c1 * a) + (c2 * b)

    arc_data[f'{col}ResidualLinear'] = Y - fit
    arc_data[f'{col}Fit'] = fit

    # Mean and median fits
    arc_data[f'{col}ResidualMean'] = Y - Y.mean()
    arc_data[f'{col}ResidualMedian'] = Y - Y.median()


def addDetectorMaptoArclines(als, detectorMap,
                             dropNa=True,
                             removeFlagged=True,
                             oneHotStatus=False,
                             includeTrace=False,
                             statusTypes=[ReferenceLineStatus.DETECTORMAP_RESERVED,
                                          ReferenceLineStatus.DETECTORMAP_USED]
                             ):
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

    # Get one hot for status
    if oneHotStatus:
        arc_data = arc_data.merge(pd.get_dummies(arc_data.status_name), left_index=True, right_index=True)

    # Make a one hot for the Trace.
    arc_data['Trace'] = arc_data.description.str.get_dummies()['Trace']
    if includeTrace is False:
        arc_data = arc_data.query(f'Trace == False').copy()

    dispersion = detectorMap.getDispersion(arc_data.fiberId.values, arc_data.wavelength.values)

    arc_data['lam'] = detectorMap.findWavelength(arc_data.fiberId.values, arc_data.y.values)
    arc_data['lamErr'] = (arc_data.yErr / arc_data.y) * (arc_data.lam / dispersion)
    arc_data['lamErr_nm'] = arc_data.lamErr * dispersion
    points = detectorMap.findPoint(arc_data.fiberId.values, arc_data.wavelength.values)
    arc_data['tracePosX'] = points[:, 0]
    arc_data['tracePosY'] = points[:, 1]

    arc_data['dx'] = arc_data.x - arc_data.tracePosX
    calculateResiduals(arc_data, 'dx', fitYTo='y')

    arc_data['dy'] = arc_data.y - arc_data.tracePosY
    calculateResiduals(arc_data, 'dy', fitYTo='y')

    arc_data['dy_nm'] = (arc_data.wavelength - arc_data.lam)
    calculateResiduals(arc_data, 'dy_nm', fitYTo='y')

    arc_data['centroidErr'] = np.hypot(arc_data.xErr, arc_data.yErr)
    arc_data['detectorMapErr'] = np.hypot(arc_data.dx, arc_data.dy)

    return arc_data


def plotArcResiduals(arc_lines_data,
                     title="",
                     fitType="mean",
                     lamErrMax=0,
                     nsigma=0):
    """

    fitType: "linear", "mean", "median" "per fiber"
    usePixels: report wavelength residuals in pixels, not nm
    """
    totalFibers = len(arc_lines_data.fiberId.unique())

    groups = ['DETECTORMAP_USED', 'DETECTORMAP_RESERVED']

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=False, gridspec_kw=dict(hspace=0.3))

    for group in groups:

        plot_data = arc_lines_data.query(f'status == {ReferenceLineStatus[group]}').copy()
        plot_data.status_name = plot_data.status_name.cat.remove_unused_categories()

        if lamErrMax > 0:
            plot_data = plot_data.query(f'lamErr < {lamErrMax}').copy()

        position_col = f'dxResidual{fitType.title()}'
        wavelength_col = f'dy_nmResidual{fitType.title()}'

        # Wavelength
        wavelength_data = plot_data.query(f'Trace == False')
        label = f"{group}\n{fitType}={wavelength_data.dy_nm.median():.02e} σ={iqr(wavelength_data.dy_nm) / 1.349:.3e} pix"
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

        fig.suptitle(
            f"{title}\n $\sigma_\lambda$ < {lamErrMax} = {len(plot_data.fiberId.unique()):.0f} / {totalFibers} lines")

    return fig


def plotArcResiduals1D(plot_data, col='dx', title='', fitType='mean', fig=None):
    fig = fig or plt.figure()

    gs = GridSpec(1, 3, width_ratios=[3, 1, 3])

    residual_col = f'{col}Residual{fitType.title()}'

    used = plot_data.query(f'status == {ReferenceLineStatus.DETECTORMAP_USED}').copy()
    reserved = plot_data.query(f'status == {ReferenceLineStatus.DETECTORMAP_RESERVED}').copy()

    used.status_name = used.status_name.cat.remove_unused_categories()
    reserved.status_name = reserved.status_name.cat.remove_unused_categories()

    used_label = f"{'used': >15s} {fitType}={used[residual_col].median():+.02e} σ={iqr(used[residual_col]) / 1.349:+.3e}"
    reserved_label = f"{'reserved': >15s} {fitType}={reserved[residual_col].median():+.02e} σ={iqr(reserved[residual_col]) / 1.349:+.3e}"

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

    fig.suptitle(f"{title} {col}")

    return fig


def residualsQuiverPlot(arc_data, detectorMap,
                        title='', cmap='coolwarm',
                        arrowSize=0.1, arrowScale=100,
                        maxCentroidErr=0.1, maxDetectorMapErr=1, limitMax=False):
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

    fig.suptitle(f"Residual quiver {title}")

    return fig


def plotArcResiduals2D(arc_data, detectorMap, title="", fitType="mean",
                       maxCentroidErr=0.1, maxDetectorMapErr=1,
                       limitMax=False, showY=True,
                       vmin=None, vmax=None, percentiles=[25, 75],
                       hexBin=False, gridsize=100, linewidths=None,
                       cmap='coolwarm'):
    """
    arrowSize: characteristic arrow length in pixels
    """
    if limitMax:
        limited_arc_data = arc_data.query(f'centroidErr < {maxCentroidErr} and detectorMapErr < {maxDetectorMapErr}')
        len_diff = len(arc_data) - len(limited_arc_data)
        print(f'Entries outside {maxCentroidErr=} and {maxDetectorMapErr=}: {len_diff} of {len(arc_data)}')
        arc_data = limited_arc_data

    ncols = 2 if showY else 1

    bbox = detectorMap.getBBox()

    fig, axes = plt.subplots(1, ncols, sharex=True, sharey=True, constrained_layout=True)

    def _make_subplot(ax, data, title=''):
        vmin, vmax = np.percentile(data, percentiles)
        # vmax = data.std()
        # vmin = -vmax
        # vmin = data.min()
        # vmax = data.max()
        print(f'Normalization: {vmin=} {vmax=}')
        norm = colors.Normalize(vmin, vmax)

        if hexBin:
            im = ax.hexbin(arc_data.x, arc_data.y, data, norm=norm, cmap=cmap, gridsize=gridsize)
        else:
            im = ax.scatter(arc_data.x, arc_data.y, c=data, s=3, norm=norm, cmap=cmap)

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.format_coord = addPfsCursor(None, detectorMap)
        ax.set_aspect('equal')
        ax.set_title(f"{title}")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.02)
        fig.colorbar(im, ax=ax, cax=cax, orientation='vertical', shrink=0.6)

        ax.set_xlim(bbox.getMinX(), bbox.getMaxX())
        ax.set_ylim(bbox.getMinY(), bbox.getMaxY())

    _make_subplot(axes[0], arc_data.dxResidualMean, title='dx mean [pixel]')

    if showY:
        _make_subplot(axes[1], arc_data.dyResidualMean, title='dy mean [nm]')

    plt.suptitle(f"Residual {title}")

    return fig, arc_data
