import numpy as np
import pfs.drp.qa.skySubtraction.plot as skySubtractionQaPlot
import scipy.stats
from pfs.drp.qa.skySubtraction.skySubtractionQa import arm_colors, getStdev, buildReference, rolling
from pfs.drp.qa.skySubtraction.skySubtractionQa import splitSpectraIntoReferenceAndTest


def summarizeSpectrograph(hold,
                          spectrograph,
                          arms=('b', 'r', 'n'),
                          colors=None,
                          fontsize=25,
                          xlim=(-10, 10),
                          alpha=0.2):
    """
    Summarize spectrograph sky subtraction residuals using chi distributions.

    This function generates a summary plot of sky-subtracted residuals (`chi` values)
    across different arms of the spectrograph, comparing mean, median, standard deviation,
    and interquartile range (IQR) statistics.

    Parameters
    ----------
    hold : `dict`
        Dictionary containing sky-subtraction residuals for different spectrograph arms.
    spectrograph : `int`
        Spectrograph number for labeling the plots.
    arms : `tuple` of `str`, optional
        List of arms to include in the analysis (default: ('b', 'r', 'n')).
    colors : `list` of `str`, optional
        Colors for each arm in the plot. Defaults to predefined `arm_colors` if None.
    fontsize : `int`, optional
        Font size for labels and titles (default: 25).
    xlim : `tuple` of `float`, optional
        X-axis limits for the histograms (default: (-10, 10)).
    alpha : `float`, optional
        Transparency level for histogram layers (default: 0.2).

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure object containing the summary plots.
    ax_dict : `dict`
        Dictionary of subplot axes for further customization.

    Notes
    -----
    - Generates chi distribution histograms per arm.
    - Compares mean, median, standard deviation, and IQR across arms.
    - Uses `skySubtractionQaPlot` for visualization.
    """

    if colors is None:
        colors = arm_colors  # Default to predefined colors

    all_axs = ['ABC', 'DEF', 'GHI'][:len(arms)]
    axt = '\n'.join(all_axs)
    fig, ax_dict = skySubtractionQaPlot.get_mosaic(axt, figsize=(20, 10))

    # Iterate over arms and generate histograms
    for color, arm, axs in zip(colors, arms, all_axs):
        h = hold[(spectrograph, arm)]
        big_chi = []  # Store all chi values for overall distribution
        layers = []
        means = []
        stdev = []

        # Process each fiber
        for fib in h.keys():
            chi = h[fib]['chi']
            chiPoisson = h[fib]['chiPoisson']
            # std = h[fib]['std']

            # Add histogram layer for each fiber
            layers.append(
                skySubtractionQaPlot.Layer('hist', chi, color=color, alpha=alpha,
                                           linewidth=2, density=True, rnge=xlim, bins=30)
            )

            # Add histogram layer for each fiber
            layers.append(
                skySubtractionQaPlot.Layer('hist', chiPoisson, color='k', alpha=0.1,
                                           linewidth=2, density=True, rnge=xlim, bins=30)
            )

            # Compute statistical metrics
            means.append([np.mean(chi), np.median(chi)])
            stdev.append([np.std(chi), getStdev(chi, useIQR=True)])
            big_chi.extend(chi)

        # Convert lists to NumPy arrays for easier processing
        means = np.array(means)
        stdev = np.array(stdev)

        # Add combined chi distribution (all fibers) with a distinctive color
        layers.append(
            skySubtractionQaPlot.Layer('hist', big_chi, color='magenta',
                                       alpha=1, linewidth=6, density=True, rnge=xlim, bins=30)
        )

        # Plot chi distribution
        skySubtractionQaPlot.make_plot(
            layers, ax_dict[axs[0]], xlim=xlim,
            xlabel=r'$\chi$' if arm == arms[-1] else None,
            ylabel=f'Arm: {arm}\nPDF', fontsize=fontsize
        )

        # Labels for statistics
        labels = [['Mean', 'Median'], ['Stdev', 'IQR Stdev']]
        rnge_options = [(-3, 3), (0, 3)]  # Range for mean/median and stdev/IQR plots

        # Iterate over mean/median and stdev/IQR plots
        for j, x, ax, rnge in zip(range(2), [means, stdev], axs[1:], rnge_options):
            other = [skySubtractionQaPlot.Layer('vert', 0 if j == 0 else 1, linestyle='--', zorder=10)]

            # Generate the histogram layers for statistical metrics
            hist_layers = [
                skySubtractionQaPlot.Layer('hist', x[:, i], color=color,
                                           alpha=[1, 0.5][i], density=True,
                                           rnge=rnge, bins=30, linewidth=4,
                                           histtype=['step', 'stepfilled'][i],
                                           label=labels[j][i]
                                           ) for i in range(2)
            ]

            # Plot statistical metrics
            skySubtractionQaPlot.make_plot(
                other + hist_layers, ax_dict[ax], xlim=rnge, legend='A' in axs,
                loc='upper right', fontsize=fontsize,
                title=f'Spectrograph: {spectrograph}' if ((arm == arms[0]) and (j == 0)) else None,
                xlabel=[r'Mean/Median $\chi$', r'Stdev/$\sigma_\mathrm{IQR}$ $\chi$'][j]
                if arm == arms[-1] else None
            )

        # Remove x-axis labels for non-bottom plots
        if arm != arms[-1]:
            for ax in axs:
                ax_dict[ax].axes.xaxis.set_ticklabels([])

        # Remove y-axis labels for all plots
        for ax in axs:
            ax_dict[ax].axes.yaxis.set_ticklabels([])

    return fig, ax_dict


def plot_1d_spectrograph(holdAsDict, plotId, arms, fontsize=22, xlim=(-5, 5)):
    """
    Generate 1D plots summarizing spectrograph data, including a Gaussian reference.

    Parameters
    ----------
    holdAsDict : `dict`
        Dictionary containing spectrograph data.
    plotId : `dict`
        Dictionary containing plot metadata (`visit`, `spectrograph`, `block`).
    arms : `list` of `str`
        List of spectral arms (e.g., ['b', 'r', 'n']).

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The generated figure.
    ax_dict : `dict`
        Dictionary of axes corresponding to the plotted elements.
    """
    visit, spectrograph, block = plotId['visit'], plotId['spectrograph'], plotId['block']

    fontsize = 22
    xlim = [-5, 5]
    all_axs = ['ABC', 'DEF', 'GHI'][:len(arms)]
    all_labels = ['Blue arm\n', 'Red arm\n', 'NIR arm\n'][:len(arms)]
    ax0 = [ax[0] for ax in all_axs]

    # Generate spectrograph summary plots
    fig, ax_dict = summarizeSpectrograph(holdAsDict,
                                         spectrograph=spectrograph,
                                         arms=arms,
                                         fontsize=fontsize,
                                         xlim=xlim,
                                         alpha=0.5)

    # Generate Gaussian distribution
    xp = np.linspace(-6, 6, 1000)
    yp = scipy.stats.norm.pdf(xp, loc=0, scale=1)

    # Update axis labels and add Gaussian reference
    for ax, arm in zip(ax0, all_labels):
        ax_dict[ax].set_ylabel(arm, fontsize=fontsize)
        ax_dict[ax].plot(xp, yp, color='k', linewidth=4, linestyle='--')

    # Set title
    ax_dict['B'].set_title(f'visit={visit}; SM{spectrograph}; blocksize={block}', fontsize=fontsize)

    # Add legend
    ax_dict['A'].plot([], [], color=arm_colors[0], label='DRP')
    ax_dict['A'].plot([], [], color='magenta', label='Combined DRP')
    ax_dict['A'].plot([], [], color='k', label='Using Poisson errors')

    ax_dict['A'].legend(fontsize=fontsize * 0.6, loc='upper left')

    return fig, ax_dict


def plot_2d_spectrograph(hold, plotId, arms, binsize=10):
    """
    Generate a 2D spectrograph plot showing sky subtraction residuals.

    This function visualizes the chi values and sky subtraction performance
    across fibers and wavelengths.

    Parameters
    ----------
    hold : `dict`
        Dictionary containing spectrograph data.
    plotId : `dict`
        Dictionary containing plot metadata (`visit`, `spectrograph`, `block`).
    arms : `list` of `str`
        List of spectral arms (e.g., ['b', 'r', 'n']).
    binsize : `int`, optional
        Size of wavelength bins for rolling median (default: 10).

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The generated figure.
    ax_dict : `dict`
        Dictionary of axes corresponding to the plotted elements.

    Notes
    -----
    - Uses `buildReference` to construct reference spectra for chi and sky_chi.
    - Uses `rolling` to smooth data along the wavelength axis.
    - Uses `pcolormesh` to create a 2D heatmap for visualization.
    """
    visit, spectrograph, block = plotId['visit'], plotId['spectrograph'], plotId['block']

    # Copy and remove pfsConfig to avoid unnecessary data
    specs = hold.copy()
    specs.pop('pfsConfig', None)

    # Define axis layout for the number of arms
    axt = 'ABC'[:len(arms)]
    fig, ax_dict = skySubtractionQaPlot.get_mosaic(axt, figsize=(15, 5))

    # Loop through each spectral arm
    for i, arm in enumerate(arms):
        ax = axt[i]
        skySpectra = specs[(spectrograph, arm)]

        # Build reference spectra
        references = buildReference(skySpectra, func=None, model='chi')
        # references_none = buildReference(skySpectra, func=None, model='sky_chi')

        # Extract data
        x, y = references
        xb, yb, eb = rolling(x, y[0], binsize)

        # Initialize 2D array for fiber-based spectral residuals
        # Initialize 2D array for fiber-based spectral residuals
        z = np.ones((len(xb), len(y)))

        for i, yi in enumerate(y):
            xb, yb, eb = rolling(x, yi, binsize)
            z[:, i] = yb

        # Create mesh grid for plotting
        X, Y = np.meshgrid(np.arange(len(y)), xb)

        # Plot 2D colormap of residuals
        sc = ax_dict[ax].pcolormesh(X, Y, z, vmin=-1, vmax=1, cmap='bwr')

        # Configure plot labels
        skySubtractionQaPlot.make_plot([],
                                       ax_dict[ax],
                                       xlabel='Fiber Index',
                                       ylabel='Wavelength [nm]' if ax == 'A' else None,
                                       title=f'Arm: {arm}')

    # Add colorbar and overall title
    fig.colorbar(sc, ax=ax_dict['A'], location='left')
    fig.suptitle(f'visit={visit}; SM{spectrograph}; blocksize={block}', fontsize=22)

    return fig, ax_dict


def plot_outlier_summary(hold, holdAsDict, plotId, arms):
    """
    Generate a summary plot highlighting outliers in sky subtraction residuals.

    This function visualizes spectral regions where the absolute chi values exceed
    predefined thresholds (5 and 15) and provides a sky model reference plot.

    Parameters
    ----------
    hold : `dict`
        Dictionary containing spectrograph data.
    holdAsDict : `dict`
        Dictionary of processed fiber data with fiber-specific residuals.
    plotId : `dict`
        Dictionary containing plot metadata (`visit`, `spectrograph`, `block`).
    arms : `list` of `str`
        List of spectral arms to be processed (e.g., ['b', 'r', 'n']).

    Returns
    -------
    figs : `list` of `matplotlib.figure.Figure`
        List of generated figures for each arm.
    ax_dicts : `list` of `dict`
        List of dictionaries containing axis handles for each figure.

    Notes
    -----
    - Uses `buildReference` to generate a median sky spectrum.
    - Highlights outlier chi values with thresholds at 5 and 15.
    - Uses `scatter` to visualize outliers in wavelength space.
    """
    visit, spectrograph, block = plotId['visit'], plotId['spectrograph'], plotId['block']

    # Copy and remove pfsConfig to avoid unnecessary data
    specs = hold.copy()
    specs.pop('pfsConfig', None)

    figs, ax_dicts = [], []

    # Loop through each spectral arm
    for i, arm in enumerate(arms):
        skySpectra = specs[(spectrograph, arm)]

        # Create figure layout
        fig, ax_dict = skySubtractionQaPlot.get_mosaic(
            """
            AAB
            AAB
            """, figsize=(6, 4))

        # Retrieve fiber data
        fibers = holdAsDict[(spectrograph, arm)]

        # Compute sky reference spectrum
        wve_sky, flx_sky = buildReference(skySpectra, func=np.nanmedian, model='sky')

        # Loop over fibers and plot outliers
        for fiberId, fiber in fibers.items():
            wve, _, chi = fiber['wave'], fiber['flux'], fiber['chi']
            absChi = np.abs(chi)

            # Define outlier conditions
            C1 = (absChi > 5) & (absChi < 15)
            C2 = (absChi > 15)

            # Plot scatter points for outliers
            for C, color in zip([C1, C2], ['steelblue', 'navy']):
                sc = ax_dict['A'].scatter(
                    [fiberId] * len(wve[C]), wve[C], c=absChi[C], vmin=5, vmax=15, cmap='viridis'
                )

        # Adjust plot limits
        ax_dict['A'].set_xlim(min(fibers.keys()) - 10, max(fibers.keys()) + 10)

        # Plot the sky spectrum
        ax_dict['B'].plot(flx_sky, wve_sky)

        # Add colorbar
        fig.colorbar(sc, ax=ax_dict['A'], location='top')

        # Add title
        fig.suptitle(f'visit={visit}; SM{spectrograph}; Arm {arm}; blocksize={block}')

        # Store figure and axis dictionary
        figs.append(fig)
        ax_dicts.append(ax_dict)

    return figs, ax_dicts


def plot_vs_sky_brightness(hold, plotId, arms):
    """
    Generate plots comparing sky brightness with spectral residuals.

    This function visualizes the relationship between median residual flux
    and sky brightness percentile, as well as how residuals change with wavelength.

    Parameters
    ----------
    hold : `dict`
        Dictionary containing spectrograph data.
    plotId : `dict`
        Dictionary containing plot metadata (`visit`, `spectrograph`, `block`).
    arms : `list` of `str`
        List of spectral arms to be processed (e.g., ['b', 'r', 'n']).

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Generated figure containing subplots.
    ax_dict : `dict`
        Dictionary containing axis handles.

    Notes
    -----
    - Uses `splitSpectraIntoReferenceAndTest` to separate reference and test spectra.
    - Compares sky brightness and residuals across different wavelengths.
    - Uses `rolling` to compute binned statistics of residuals versus sky brightness percentile.
    """
    visit, spectrograph, block = plotId['visit'], plotId['spectrograph'], plotId['block']

    # Define panel layout dynamically based on arms
    axt = [t[:len(arms)] for t in ['ABC', 'DEF']]
    axt = '\n'.join(axt)
    panel_labels = [''.join([a[i] for a in axt.split('\n')]) for i in range(len(arms))]

    # Create figure layout
    fig, ax_dict = skySubtractionQaPlot.get_mosaic(axt, figsize=(int(5 * len(arms)), 10))

    # Copy and remove pfsConfig to avoid unnecessary data
    specs = hold.copy()
    specs.pop('pfsConfig', None)

    # Loop through each spectral arm
    for i, arm in enumerate(arms):
        skySpectra = specs[(spectrograph, arm)]

        # Split into reference and test spectra
        referenceSpectra, testSpectra = splitSpectraIntoReferenceAndTest(skySpectra)

        # Compute reference and test statistics
        references_sky = buildReference(referenceSpectra, func=np.nanmedian, model='none')
        references_flx = buildReference(testSpectra, func=np.median, model='residuals')
        # references_err = buildReference(testSpectra, func='quadrature', model='variance')
        references_chi_median = buildReference(testSpectra, func=np.median, model='chi')

        color = arm_colors[i]
        col = panel_labels[i]

        # Interpolate sky brightness onto residual wavelength grid
        wve_sky, sky = references_sky
        wve, flx = references_flx
        sky = np.interp(wve, wve_sky, sky)

        chi = references_chi_median[1]

        # Compute ranked percentile of sky brightness
        ranked = np.argsort(np.argsort(sky))
        ranked = 100 * ranked / len(ranked)

        # Bin residuals based on sky brightness percentiles
        yb, xb, eb = rolling(ranked, chi, 10)

        # Scatter plot of residual flux vs wavelength
        ax_dict[col[0]].scatter(wve, flx, s=1, color=color, rasterized=True, alpha=0.7)
        ax_dict[col[0]].plot(wve, sky / 100, color='k', linewidth=1, alpha=0.6, label='1% sky')

        # Scatter plot of residuals vs sky brightness percentile
        ax_dict[col[1]].scatter(chi, ranked, s=1, color=color, rasterized=True, alpha=0.7)
        ax_dict[col[1]].errorbar(xb, yb, xerr=eb, color='k', linewidth=3)

        # Set axis limits
        ax_dict[col[0]].set_ylim([-100, 100])
        ax_dict[col[1]].set_xlim([-0.5, 0.5])

        # Set axis labels
        ax_dict[col[0]].set_xlabel('Wavelength [nm]')
        ax_dict[col[0]].set_ylabel('Median Counts')

        ax_dict[col[1]].set_xlabel(r'Median $\chi$')
        ax_dict[col[1]].set_ylabel('Sky Counts Percentile')

        # Add reference lines
        ax_dict[col[1]].axvline(0, linestyle='--', color='k')
        ax_dict[col[0]].axhline(0, linestyle='--', color='k')

    # Add legend and title
    ax_dict['A'].legend()
    fig.suptitle(f'visit={visit}; SM{spectrograph}; blocksize={block}')

    return fig, ax_dict
