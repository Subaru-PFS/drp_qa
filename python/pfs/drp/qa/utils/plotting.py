import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import iqr
import pandas as pd
import seaborn as sb

from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

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

    num_fibers = len(data.fiberId.unique())

    plot_data = data.melt(
        id_vars=['fiberId', 'wavelength', 'x', 'y', 'isTrace', 'bin', column],
        value_vars=['isUsed', 'isReserved'],
        var_name='status').query('value == True')

    if column.startswith('dy'):
        plot_data = plot_data.query('isTrace == False').copy()

    reserved_data = plot_data.query('status == "isReserved"')
    if len(reserved_data) == 0:
        raise ValueError('No data')

    # Get summary statistics.
    stats_df = plot_data.groupby('status')[column].agg(['median', iqr_sigma])
    spatial_avg = plot_data.groupby(
        ['fiberId', 'status'])[column].agg(['median', iqr_sigma, 'count']).reset_index()

    reserved_sigma = stats_df.loc['isReserved'].iqr_sigma
    sigmaLimit = 2.5 * reserved_sigma
    vmin = -sigmaLimit if vmin is None else vmin
    vmax = sigmaLimit if vmax is None else vmax

    pal = dict(zip(spatial_avg.status.unique(), plt.cm.tab10.colors))
    pal_colors = [pal[x] for x in spatial_avg.status]

    units = 'nm' if column == 'dy_nm' else 'pix'

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
    # Show sigmas.
    ax0.axhline(-sigmaLimit, c='r', ls='--', alpha=0.35, label='±2.5 * sigma')
    ax0.axhline(sigmaLimit, c='r', ls='--', alpha=0.35)
    ax0.text(spatial_avg.fiberId.min() + 65, 3 * reserved_sigma, f'±2.5σ={sigmaLimit:.3f}', c='r',
             clip_on=True)
    ax0.legend(
        loc='lower right',
        shadow=True,
        prop=dict(family='monospace', weight='bold'), bbox_to_anchor=(1.2, 0)
    )
    num_outliers = spatial_avg.query('abs(median) >= 2.5 * @reserved_sigma').fiberId.count()
    ax0.text(0.02, 0.07,
             f'Number of fibers: {num_fibers}; Number of outliers (2.5σ={sigmaLimit:.3f}): {num_outliers}',
             transform=ax0.transAxes,
             bbox=dict(boxstyle='round', ec='k', fc='wheat', alpha=0.75),
             fontsize='small'
             )

    if use_dm_layout is True:
        # Reverse the fiber order to match the xy-pixel layout
        ax0.set_xlim(list(reversed(ax0.get_xlim())))

    ax0.set_ylabel(f'Δ {units}')
    ax0.xaxis.set_label_position('top')
    ax0.set_xlabel('')
    ax0.xaxis.tick_top()
    ax0.set_title('Median fiber residual and 1-sigma error', weight='bold', fontsize='small')
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
    ax2.set_title('2D residual of RESERVED', weight='bold', fontsize='small')

    if bin_wl is True:
        binned_data = plot_data.groupby(['bin', 'status'])[['wavelength', column]]
        plot_data = binned_data.agg('median', iqr_sigma).dropna().reset_index().sort_values('status')

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
    ax3.axvline(-sigmaLimit, c='r', ls='--', alpha=0.35, label='-2.5 * sigma')
    ax3.axvline(sigmaLimit, c='r', ls='--', alpha=0.35)
    num_outliers = plot_data.query(f'abs({column}) >= @sigmaLimit').wavelength.count()
    ax3.text(0.05, 0.02,
             f'Number of lines: {len(plot_data)}\nNumber of outliers: {num_outliers}',
             bbox=dict(boxstyle='round', ec='k', fc='wheat', alpha=0.75),
             transform=ax3.transAxes,
             fontsize='small', zorder=100
             )

    ax3.yaxis.set_label_position('right')
    ax3.yaxis.tick_right()
    ax3.set_xlabel(f'Δ {units}')
    ax_title = f'Residual by {"binned" if bin_wl else ""} wavelength'
    if bin_wl:
        ax_title += f'\nbinsize={binWavelength} nm'
    ax3.set_title(ax_title, weight='bold', fontsize='small')

    fig.suptitle('DetectorMap Residuals', weight='bold')

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
        s=20,
        ec='k',
        marker='.',
        zorder=100,
        palette=palette,
        rasterized=rasterized,
        ax=ax
    )

    # Positive outliers.
    pos = data.query(f'{X if vertical else Y} >= @ymax').copy()
    pos[X if vertical else Y] = ymax
    marker = '<' if vertical is True else 'v'
    sb.scatterplot(data=pos, x=X, y=Y, hue=hue, palette=palette, legend=False,
                   marker=marker, ec='k', lw=0.5, s=100,
                   clip_on=False, zorder=100, ax=ax,
                   )

    # Negative outliers.
    neg = data.query(f'{X if vertical else Y} <= @ymin').copy()
    neg[X if vertical else Y] = ymin
    marker = '>' if vertical is True else '^'
    sb.scatterplot(data=neg, x=X, y=Y, hue=hue, palette=palette, legend=False,
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

    ax.grid(True, alpha=0.15)

    return ax
