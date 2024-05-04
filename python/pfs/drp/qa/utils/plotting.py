import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from pfs.drp.qa.utils.helpers import iqr_sigma, getFitStats, getWeightedRMS

div_palette = plt.cm.RdBu_r.with_extremes(over='magenta', under='cyan', bad='lime')
detector_palette = {
    'b': 'tab:blue',
    'r': 'tab:red',
    'n': 'tab:orange',
    'm': 'tab:pink'
}


def makePlot(
        arc_data,
        visit_stats,
        arm,
        spectrograph,
        useSigmaRange=False,
        xrange=0.1,
        wrange=0.1,
        binWavelength=0.1
):
    if useSigmaRange is True:
        xrange = None
        wrange = None

    ccd = f'{arm}{spectrograph}'

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
    main_fig = Figure(layout='constrained', figsize=(14, 10))

    # Split into two rows.
    (top_fig, bottom_fig) = main_fig.subfigures(2, 1, wspace=0, height_ratios=[5, 1.5])
    top_fig.suptitle(
        f'DetectorMap Residuals\n'
        f'{arm}{spectrograph}\n',
        weight='bold',
        fontsize='small'
    )

    # Split top fig into wo columns.
    (x_fig, y_fig) = top_fig.subfigures(1, 2, wspace=0)

    try:
        pd0 = arc_data.query(f'arm == "{arm}" and spectrograph == {spectrograph}').copy()

        for sub_fig, column in zip([x_fig, y_fig], ['xResid', 'yResid']):
            try:
                plotResidual(
                    pd0,
                    column=column,
                    xrange=xrange,
                    wrange=wrange,
                    binWavelength=binWavelength,
                    # goodRange=goodLimits[column],
                    sigmaLines=[1.],
                    dmWidth=dmWidth,
                    dmHeight=dmHeight,
                    fiberIdMin=fiberIdMin,
                    fiberIdMax=fiberIdMax,
                    wavelengthMin=wavelengthMin,
                    wavelengthMax=wavelengthMax,
                    fig=sub_fig,
                )
                sub_fig.suptitle(f'{arm}{spectrograph}\n{column}', fontsize='small', fontweight='bold')
            except Exception as e:
                print(f'Problem plotting residual {e}')

        desc_pal = {desc: plt.cm.tab10(i) for i, desc in enumerate(sorted(visit_stats.description.unique()))}

        visit_fig = plotVisits(
            visit_stats.query(
                'status_type == "RESERVED"'
                ' and ccd == @ccd'
                # f' and visit {"" if calib_inputs_only else "not"} in @visit'
            ).sort_values(by='visit').copy(),
            desc_pal,
            showLegend=True,
            fig=bottom_fig
        )
        for ax in visit_fig.axes:
            ax.set_xlim(-0.3, 0.3)
        visit_fig.suptitle(f'RESERVED median and 1-sigma weighted error per visit {ccd=}')

        return main_fig
    except ValueError as e:
        print(e)
        return None


def plotResidual(
        data: pd.DataFrame,
        column: str = 'xResid',
        xrange: float = None,
        wrange: float = None,
        sigmaRange: int = 2.5,
        sigmaLines: list = [1.0, 2.5],
        goodRange: float = None,
        binWavelength: float = None,
        useDMLayout: bool = True,
        dmWidth: int = 4096,
        dmHeight: int = 4176,
        fiberIdMin: int = None,
        fiberIdMax: int = None,
        wavelengthMin: float = None,
        wavelengthMax: float = None,
        fig: Figure = None
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
            'xErr',
            'y',
            'yErr',
            'isTrace',
            'isLine',
            'bin',
            column,
            f'{column}Outlier'
        ],
        value_vars=['isUsed', 'isReserved'],
        var_name='status'
    ).query('value == True')
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
    fit_stats_all = getFitStats(data.query(f'isReserved == True and {column}Outlier == False'))
    fit_stats = getattr(fit_stats_all, which_data)
    fit_stats_used = getattr(getFitStats(data.query('isUsed == True')), which_data)

    fig = fig or Figure(layout='constrained')

    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    # Upper row
    # Fiber residual
    fiber_avg = plotData.groupby(
        [
            'fiberId',
            'status',
            'isOutlier'
        ]
    ).apply(
        lambda rows:
        (
            len(rows),
            rows[column].median(),
            getWeightedRMS(rows[column], rows[f'{column[0]}Err'])
        )
    ).reset_index().rename(columns={0: 'vals'})
    fiber_avg = fiber_avg.join(
        pd.DataFrame(fiber_avg.vals.to_list(), columns=['count', 'median', 'weightedRms'])
    ).drop(columns=['vals'])

    fiber_avg.sort_values(['fiberId', 'status'], inplace=True)

    pal = dict(zip(sorted(fiber_avg.status.unique()), plt.cm.tab10.colors))
    pal_colors = [pal[x] for x in fiber_avg.status]

    # Just the errors, no markers
    goodFibersAvg = fiber_avg.query('isOutlier == False')
    ax0.errorbar(
        goodFibersAvg.fiberId, goodFibersAvg['median'], goodFibersAvg.weightedRms,
        ls='', ecolor=pal_colors, alpha=0.5
    )

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
                        fiber_avg.fiberId.min(),
                        1.5 * lim,
                        f'±1.0σ={abs(lim):.4f}',
                        c='g',
                        ha='right',
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
                            fiber_avg.fiberId.min(),
                            1.5 * lim,
                            f'±{sigmaMultiplier}σ={abs(lim):.4f}',
                            c=pal['isReserved'],
                            ha='right',
                            clip_on=True,
                            weight='bold',
                            zorder=100,
                            bbox=dict(boxstyle='round', ec='k', fc='wheat', alpha=0.75),
                        )

    drawRefLines(ax0, goodRange, sigmaRange)

    ax0.legend(
        loc='lower right',
        shadow=True,
        prop=dict(family='monospace', weight='bold'), bbox_to_anchor=(1.2, 0)
    )

    fiber_outliers = goodFibersAvg.query(f'status=="isReserved" and abs(median) >= {fit_stats.weightedRms}')
    num_sig_outliers = fiber_outliers.fiberId.count()
    fiber_big_outliers = fiber_outliers.query(f'abs(median) >= {sigmaRange * fit_stats.weightedRms}')
    num_siglimit_outliers = fiber_big_outliers.fiberId.count()
    ax0.text(
        0.01, 0.0,
        f'Number of fibers: {fit_stats.num_fibers} '
        f'Number of outliers: '
        f'1σ: {num_sig_outliers} '
        f'{sigmaRange}σ: {num_siglimit_outliers}',
        transform=ax0.transAxes,
        bbox=dict(boxstyle='round', ec='k', fc='wheat'),
        fontsize='small', zorder=100
    )

    if fiberIdMin is not None and fiberIdMax is not None:
        ax0.set_xlim(fiberIdMin, fiberIdMax)

    if useDMLayout is True:
        # Reverse the fiber order to match the xy-pixel layout
        ax0.set_xlim(*list(reversed(ax0.get_xlim())))

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
    ax1.text(
        -0.01,
        0.0,
        f'RESERVED:\n{fit_stats}\nUSED:\n{fit_stats_used}',
        transform=ax1.transAxes,
        fontfamily='monospace',
        fontsize='small',
        fontweight='bold',
        bbox=dict(boxstyle='round', alpha=0.5, facecolor='white')
    )
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

    im = ax2.scatter(
        reserved_data[X],
        reserved_data[Y],
        c=reserved_data[column],
        norm=norm,
        cmap=div_palette,
        s=2,
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
        plotData.query('isOutlier == False'),
        column,
        'wavelength',
        hue='status',
        ymin=-wrange,
        ymax=wrange,
        palette=pal,
        ax=ax3,
        refline=0.,
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

    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    ax3.set_xlabel(f'Δ {units}')
    ax_title = f'{which_data.title()} residual\nby wavelength'
    if bin_wl:
        ax_title += f' binsize={binWavelength} {units}'
    ax3.set_title(ax_title, weight='bold', fontsize='small')

    fig.suptitle('DetectorMap Residuals', weight='bold')

    return fig


def scatterplotWithOutliers(
        data, X, Y, hue='status_name',
        ymin=-0.1, ymax=0.1, palette=None,
        ax=None, refline=None, vertical=False,
        rasterized=False, showUnusedOutliers=False,
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

    if showUnusedOutliers is True:
        # Positive outliers.
        pos = data.query(f'{X if vertical else Y} >= @ymax').copy()
        pos[X if vertical else Y] = ymax
        marker = 'X' if vertical is True else 'X'
        sb.scatterplot(
            data=pos, x=X, y=Y, hue=hue, palette=palette, legend=False,
            marker=marker, ec='k', lw=0.5, s=50, alpha=0.5,
            clip_on=False, zorder=100, ax=ax,
        )

        # Negative outliers.
        neg = data.query(f'{X if vertical else Y} <= @ymin').copy()
        neg[X if vertical else Y] = ymin
        marker = 'X' if vertical is True else 'X'
        sb.scatterplot(
            data=neg, x=X, y=Y, hue=hue, palette=palette, legend=False,
            marker=marker, ec='k', lw=0.5, s=50, alpha=0.5,
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


def plotVisits(plotData, desc_pal=None, fig=None, showLegend=False):
    plotData = plotData.copy()
    fig = fig or Figure(layout='constrained')
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
        # ax.legend().set_visible(False)

        ax.grid(alpha=0.2)
        ax.axvline(0, c='k', ls='--', alpha=0.5)
        ax.set_title(f'{metric}')
        ax.set_xlabel('pix')

    visit_label = [f'{row.visit}' for idx, row in plotData.iterrows()]
    ax0.set_yticks(plotData.visit_idx, visit_label, fontsize='xx-small')
    ax0.set_ylabel('Visit')
    ax0.invert_yaxis()

    # if showLegend:
    #     fig.legend(*ax0.get_legend_handles_labels(), shadow=True)
    fig.suptitle('RESERVED median and 1-sigma weighted errors')

    return fig


def plotDetectorSoften(detector_stats):
    plot_data = detector_stats.melt(id_vars=['ccd', 'status_type', 'description'])

    plot_data.loc[plot_data.query('variable.str.contains("spatial")').index, 'metric'] = 'spatial'
    plot_data.loc[plot_data.query('variable.str.contains("wavelength")').index, 'metric'] = 'wavelength'

    fg = sb.catplot(
        data=plot_data.dropna().query(
            'description != "all" and variable.str.contains("soften") and status_type == "RESERVED"'
        ).sort_values(by=['ccd']),
        row='metric',
        x='ccd',
        y='value',
        hue='description',
        height=2,
        aspect=4,
        palette='Set1',
        ec='k',
        linewidth=0.5,
        legend=False
    )
    for ax in fg.figure.axes:
        ax.grid(alpha=0.25)
        # ax.set_ylim(0, 1)
    fg.figure.legend(*fg.figure.axes[0].get_legend_handles_labels(), shadow=True, fontsize='small')
    fg.figure.set_tight_layout('inches')

    return fg.figure


def plotDetectorMedians(detector_stats):
    plot_data = detector_stats.query('description == "all" and status_type=="RESERVED"').filter(
        regex='ccd|median|soften|weighted'
    )
    plot_data['arm'] = plot_data.ccd.str[0]

    fig, axes = plt.subplots(nrows=2, sharex=True, layout='constrained')
    fig.set_size_inches(12, 6)

    for ax, metric in zip(axes, ['spatial', 'wavelength']):
        for ccd, row in plot_data.groupby('ccd'):
            ax.errorbar(
                x=row.ccd,
                y=row[f'{metric}.median'],
                yerr=row[f'{metric}.weightedRms'],
                # ms=row[f'{metric}.softenFit'][0] * 100,
                c=detector_palette[row.arm[0]],
                ls='',
                lw=1.5,
                capsize=2,
                zorder=-100,
                # marker='o',
            )

        sb.scatterplot(
            data=plot_data,
            x='ccd',
            y=f'{metric}.median',
            hue='arm',
            palette=detector_palette,
            size=f'{metric}.softenFit',
            size_norm=(0, 0.5),
            legend=False,
            ax=ax
        )
        # ax.legend(title='softenFit (pix)', loc='upper right', bbox_to_anchor=(1.25, 1))
        ax.grid(alpha=0.15)
        ax.set_title(metric)
        ax.set_ylim(-0.1, 0.1)
        ax.set_ylabel('pixel')
        ax.axhline(-0.1, c='g', ls='--', alpha=0.35)
        ax.axhline(0.1, c='g', ls='--', alpha=0.35)
        ax.axhline(0., c='k', ls='--', alpha=0.35, zorder=-100)

    return fig
