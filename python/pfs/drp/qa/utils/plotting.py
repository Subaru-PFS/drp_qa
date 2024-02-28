import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sb

from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from pfs.drp.qa.utils.helpers import iqr_sigma, getFitStats

div_palette = plt.cm.RdBu_r.with_extremes(over='magenta', under='cyan', bad='lime')


def plotResidual(
        data: pd.DataFrame,
        column: str = 'xResid',
        xrange : float = None,
        wrange : float = None,
        sigmaRange : int = 2.5,
        sigmaLines : list = [1.0, 2.5],
        goodRange: float = None,
        binWavelength : float = None,
        useDMLayout: bool = True,
        dmWidth: int = 4096,
        dmHeight: int = 4176) -> Figure:
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
    goodRange : `float`, optional
        Used for showing an "acceptable" range.
    binWavelength : `float`, optional
        The value by which to bin the wavelength. If None, no binning.
    useDMLayout : `bool`, optional
        Use the detector map layout? Default is ``True``.

    Returns
    -------
    fig : `Figure`
        A summary plot of the 1D and 2D residuals.
    """
    # Wavelength residual
    data['bin'] = 1
    bin_wl = False
    if isinstance(binWavelength, (int, float)) and binWavelength > 0:
        bins = np.arange(data.wavelength.min() - 1, data.wavelength.max() + 1, binWavelength)
        s_cut, bins = pd.cut(data.wavelength, bins=bins, retbins=True, labels=False)
        data['bin'] = pd.Categorical(s_cut)
        bin_wl = True

    plotData = data.melt(
        id_vars=[
            'fiberId',
            'wavelength',
            'x',
            'y',
            'isTrace',
            'isLine',
            'bin',
            column,
            f'{column}Outlier'
        ],
        value_vars=['isUsed', 'isReserved'],
        var_name='status').query('value == True')
    plotData.rename(columns={f'{column}Outlier': 'isOutlier'}, inplace=True)

    units = 'pix'
    which_data = 'spatial'
    if column.startswith('y'):
        plotData = plotData.query('isTrace == False').copy()
        which_data = 'wavelength'

    reserved_data = plotData.query('status == "isReserved" and isOutlier == False')
    if len(reserved_data) == 0:
        raise ValueError('No data')

    # Get summary statistics.
    fit_stats_all = getFitStats(data.query('isReserved == True'))
    fit_stats = getattr(fit_stats_all, which_data)
    fit_stats_used = getattr(getFitStats(data.query('isUsed == True')), which_data)

    fig = Figure(layout='constrained')
    fig.set_size_inches(10, 10)

    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    # Upper row
    # Fiber residual
    fiber_avg = plotData.groupby([
        'fiberId',
        'status',
        'isOutlier'
    ])[column].agg(['median', iqr_sigma, 'count']).reset_index()

    fiber_avg.sort_values(['fiberId', 'status'], inplace=True)

    pal = dict(zip(sorted(fiber_avg.status.unique()), plt.cm.tab10.colors))
    pal_colors = [pal[x] for x in fiber_avg.status]

    # Just the errors, no markers
    goodFibersAvg = fiber_avg.query('isOutlier == False')
    ax0.errorbar(goodFibersAvg.fiberId, goodFibersAvg['median'], goodFibersAvg.iqr_sigma,
                 ls='', ecolor=pal_colors, alpha=0.5)

    # Use sigma range if no range given.
    if xrange is None and sigmaRange is not None:
        xrange = fit_stats.weightedRms * sigmaRange

    # Scatterplot with outliers marked.
    ax0 = scatterplotWithOutliers(
        goodFibersAvg,
        'fiberId',
        'median',
        hue='status',
        ymin=-xrange,
        ymax=xrange,
        palette=pal,
        ax=ax0,
        refline=0,
    )

    def drawRefLines(ax, goodRange, sigmaRange, isVertical=False):
        method = 'axhline' if isVertical is False else 'axvline'
        refLine = getattr(ax, method)
        # Good sigmas
        if goodRange is not None:
            for i, lim in enumerate(goodRange):
                refLine(lim, c='g', ls='-.', alpha=0.75, label='Good limits')
                if i == 0:
                    ax.text(
                        fiber_avg.fiberId.min() + 85,
                        1.5 * lim,
                        f'±1.0σ={abs(lim):.4f}',
                        c='g',
                        clip_on=True,
                        weight='bold',
                        zorder=100,
                        bbox=dict(boxstyle='round', ec='k', fc='wheat', alpha=0.75),
                    )

        if sigmaLines is not None:
            for sigmaLine in sigmaLines:
                for i, sigmaMultiplier in enumerate([sigmaLine, -1 * sigmaLine]):
                    lim = sigmaMultiplier * fit_stats.weightedRms
                    refLine(lim, c=pal['isReserved'], ls='--', alpha=0.75, label=f'{lim} * sigma')
                    if i == 0:
                        ax.text(
                            fiber_avg.fiberId.min() + 85,
                            1.5 * lim,
                            f'±{sigmaMultiplier}σ={abs(lim):.4f}',
                            c=pal['isReserved'],
                            clip_on=True,
                            weight='bold',
                            zorder=100,
                            bbox=dict(boxstyle='round', ec='k', fc='wheat', alpha=0.75),
                        )

    drawRefLines(ax0, goodRange, sigmaRange)

    ax0.legend(loc='lower right',
               shadow=True,
               prop=dict(family='monospace', weight='bold'), bbox_to_anchor=(1.2, 0)
               )

    fiber_outliers = goodFibersAvg.query(f'status=="isReserved" and abs(median) >= {fit_stats.weightedRms}')
    num_sig_outliers = fiber_outliers.fiberId.count()
    fiber_big_outliers = fiber_outliers.query(f'abs(median) >= {sigmaRange * fit_stats.weightedRms}')
    num_siglimit_outliers = fiber_big_outliers.fiberId.count()
    ax0.text(
        0.02, 0.07,
        f'Number of fibers: {fit_stats.num_fibers}\n'
        f'Number of outliers:\n'
        f'>=1σ = {num_sig_outliers}\n'
        f'>={sigmaRange}σ = {num_siglimit_outliers}',
        transform=ax0.transAxes,
        bbox=dict(boxstyle='round', ec='k', fc='wheat', alpha=0.75),
        fontsize='small', zorder=100
    )

    if useDMLayout is True:
        # Reverse the fiber order to match the xy-pixel layout
        ax0.set_xlim(list(reversed(ax0.get_xlim())))

    ax0.set_ylabel(f'Δ {units}')
    ax0.xaxis.set_label_position('top')
    ax0.set_xlabel('')
    ax0.xaxis.tick_top()
    ax0.set_title(
        f'Median {which_data} residual and 1-sigma weighted error by fiberId',
        weight='bold',
        fontsize='small'
    )
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
    norm = colors.Normalize(vmin=-xrange, vmax=xrange)
    if useDMLayout:
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
                     s=4,
                     )
    fig.colorbar(im, ax=ax2, orientation='horizontal', extend='both', fraction=0.02, aspect=75, pad=0.01)

    ax2.set_xlim(0, dmWidth)
    ax2.set_ylim(0, dmHeight)
    ax2.set_ylabel(Y)
    ax2.set_xlabel(X)
    ax2.set_title(f'2D residual of RESERVED {which_data} data', weight='bold', fontsize='small')

    # Use sigma range if no range given.
    if wrange is None and sigmaRange is not None:
        wrange = fit_stats.weightedRms * sigmaRange

    if bin_wl is True:
        binned_data = plotData.groupby(['bin', 'status', 'isOutlier'])[['wavelength', column]]
        plotData = binned_data.agg('median', iqr_sigma).dropna().reset_index().sort_values('status')

    ax3 = scatterplotWithOutliers(
        plotData,
        column,
        'wavelength',
        hue='status',
        ymin=-wrange,
        ymax=wrange,
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

    drawRefLines(ax3, goodRange, sigmaRange, isVertical=True)

    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    ax3.set_xlabel(f'Δ {units}')
    ax_title = f'{which_data.title()} residual by {"binned" if bin_wl else ""} wavelength'
    if bin_wl:
        ax_title += f'\nbinsize={binWavelength} {units}'
    ax3.set_title(ax_title, weight='bold', fontsize='small')

    fig.suptitle('DetectorMap Residuals', weight='bold')

    # Make a legend with stats.
    handles, labels = ax0.get_legend_handles_labels()
    handles = handles[1:3]  # Remove the outliers markers
    labels = [
        f'RESERVED:\n{fit_stats}',
        f'USED:\n{fit_stats_used}'
    ]
    legend = fig.legend(handles=handles,
                        labels=labels,
                        labelspacing=0.15,
                        prop=dict(family='monospace', weight='bold', size='small'),
                        loc='upper right',
                        bbox_to_anchor=(0.97, 0.95),
                        title=f'{which_data.title()} Stats ({units})',
                        title_fontproperties=dict(weight='bold')
                        )

    return fig


def scatterplotWithOutliers(data, X, Y, hue='status_name',
                            ymin=-0.1, ymax=0.1, palette=None,
                            ax=None, refline=None, vertical=False,
                            rasterized=False,
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
        The reference line. Default is ``None``.
    vertical : `bool`, optional
        Is the plot vertical? Default is ``False``.
    rasterized : `bool`, optional
        Rasterize the plot? Default is ``False``.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        A scatter plot with the outliers marked.
    """
    # Main plot.
    ax = sb.scatterplot(
        data=data,
        x=X,
        y=Y,
        hue=hue,
        hue_order=['isReserved', 'isUsed'] if hue == 'status' else None,
        s=20,
        ec='k',
        style='isOutlier',
        markers={True: 'X', False: '.'},
        zorder=100,
        palette=palette,
        rasterized=rasterized,
        ax=ax
    )

    # Positive outliers.
    pos = data.query(f'{X if vertical else Y} >= @ymax').copy()
    pos[X if vertical else Y] = ymax
    marker = 'X' if vertical is True else 'X'
    sb.scatterplot(data=pos, x=X, y=Y, hue=hue, palette=palette, legend=False,
                   marker=marker, ec='k', lw=0.5, s=50,
                   clip_on=False, zorder=100, ax=ax,
                   )

    # Negative outliers.
    neg = data.query(f'{X if vertical else Y} <= @ymin').copy()
    neg[X if vertical else Y] = ymin
    marker = 'X' if vertical is True else 'X'
    sb.scatterplot(data=neg, x=X, y=Y, hue=hue, palette=palette, legend=False,
                   marker=marker, ec='k', lw=0.5, s=50,
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

    ax.grid(True, alpha=0.15)

    return ax


def plotVisits(plotData, detectorStats, desc_pal=None):
    fig = Figure(layout='constrained')
    fig.set_size_inches(8, 4)
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122, sharex=ax0, sharey=ax0)

    plotData['visit_idx'] = plotData.visit.rank(method='first')

    for ax, metric in zip([ax0, ax1], ['spatial', 'wavelength']):
        for desc, grp in plotData.groupby('description'):
            grp.plot.scatter(
                y='visit_idx',
                x=f'{metric}.median',
                xerr=f'{metric}.weightedRms',
                marker='o',
                color=desc_pal[desc] if desc_pal is not None else None,
                label=desc,
                ax=ax,
            )
        ax.legend().set_visible(False)

        ax.axvline(detectorStats[f'{metric}.weightedRms'], ls='--', c='g', alpha=0.5, label='±1σ detector')
        ax.axvline(detectorStats[f'{metric}.weightedRms'] * -1, ls='--', c='g', alpha=0.5)
        ax.grid(alpha=0.2)
        ax.axvline(0, c='k', ls='--', alpha=0.5)
        ax.set_title(f'{metric}')
        ax.set_xlabel('pix')

    visit_label = [f'{row.visit}-{row.ccd}' for idx, row in plotData.iterrows()]
    ax1.set_yticks(plotData.visit_idx, visit_label, rotation=90)
    ax0.set_ylabel('Visit')

    fig.legend(*ax0.get_legend_handles_labels(), shadow=True, bbox_to_anchor=(1.2, 1))
    fig.suptitle('RESERVED median and 1-sigma weighted errors')

    return fig
