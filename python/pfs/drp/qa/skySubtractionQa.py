from dataclasses import dataclass
from numbers import Number
from typing import Callable, Iterable, List

import numpy as np
import scipy.stats
from astropy.nddata import NDDataArray
from lsst.pex.config import Field, ListField
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)
from lsst.pipe.base.connectionTypes import (
    Input as InputConnection,
    Output as OutputConnection,
)
from matplotlib import pylab as plt
from matplotlib.axes import Axes
from pfs.drp.stella import PfsArm, PfsConfig
from pfs.drp.stella.fitFocalPlane import FitBlockedOversampledSplineConfig, FitBlockedOversampledSplineTask
from pfs.drp.stella.selectFibers import SelectFibersTask
from pfs.drp.stella.subtractSky1d import subtractSky1d

from pfs.drp.qa.storageClasses import MultipagePdfFigure

arm_colors = ["steelblue", "firebrick", "darkgoldenrod"]
plot_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
]


class SkyArmSubtractionConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Connections for SkySubtractionTask"""

    pfsArm = InputConnection(
        name="pfsArm",
        doc="PfsArm data",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    pfsConfig = InputConnection(
        name="pfsConfig",
        doc="PfsConfig data",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
    )

    mergeArms_config = InputConnection(
        name="mergeArms_config",
        doc="Configuration for merging arms",
        storageClass="Config",
        dimensions=(),
    )

    skySubtraction_mergedSpectra = OutputConnection(
        name="skySubtraction_mergedSpectra",
        doc="Merged spectra after sky subtraction",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class SkyArmSubtractionConfig(PipelineTaskConfig, pipelineConnections=SkyArmSubtractionConnections):
    """Configuration for SkySubtractionTask"""

    blockSize = Field(dtype=int, default=None, optional=True, doc="Block size for sky model fitting.")
    rejIterations = Field(dtype=int, default=None, optional=True, doc="Number of rejection iterations.")
    rejThreshold = Field(dtype=float, default=None, optional=True, doc="Rejection threshold.")
    oversample = Field(dtype=float, default=None, optional=True, doc="Oversampling factor.")
    mask = ListField(dtype=str, default=None, optional=True, doc="Mask types.")


class SkyArmSubtractionTask(PipelineTask):
    """Task for QA of sky subtraction for a single PfsArm"""

    ConfigClass = SkyArmSubtractionConfig
    _DefaultName = "skyArmSubtraction"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        mergeArms_config = butlerQC.get(inputRefs.mergeArms_config)
        # Get default sky model configuration
        fitSkyModelConfig = FitBlockedOversampledSplineConfig()
        defaultConfig = mergeArms_config.fitSkyModel.toDict()

        # Update only if user provides values
        if self.config.blockSize is not None:
            defaultConfig["blockSize"] = self.config.blockSize
        if self.config.rejIterations is not None:
            defaultConfig["rejIterations"] = self.config.rejIterations
        if self.config.rejThreshold is not None:
            defaultConfig["rejThreshold"] = self.config.rejThreshold
        if self.config.oversample is not None:
            defaultConfig["oversample"] = self.config.oversample
        if self.config.mask is not None:
            defaultConfig["mask"] = self.config.mask

        fitSkyModelConfig.update(**defaultConfig)
        self.log.info("Using sky model configuration: %s", fitSkyModelConfig)

        inputs = butlerQC.get(inputRefs)
        inputs["fitSkyModelConfig"] = fitSkyModelConfig

        try:
            # Perform the actual processing.
            outputs = self.run(**inputs)
        except ValueError as e:
            self.log.error(e)
        else:
            # Store the results if valid.
            butlerQC.put(outputs, outputRefs)

    def run(
        self,
        pfsArm: PfsArm,
        pfsConfig: PfsConfig,
        fitSkyModelConfig: FitBlockedOversampledSplineConfig,
        **kwargs,
    ) -> Struct:
        """Perform QA on sky subtraction.

        This function performs sky subtraction multiple times,
        leaving out one sky fiber at a time and computing the residuals.

        Parameters
        ----------
        pfsArm : `pfs.drp.stella.PfsArm`
            The input PfsArm data.
        pfsConfig : `pfs.drp.stella.PfsConfig`
            The input PfsConfig data.
        fitSkyModelConfig : `pfs.drp.stella.FitBlockedOversampledSplineConfig`
            The configuration for fitting the sky model.

        Returns
        -------
        Struct
            A struct containing the plots.
        """
        spectrograph = pfsArm.identity.spectrograph
        visit = pfsArm.identity.visit
        arm = pfsArm.identity.arm

        self.log.info(f"Starting sky subtraction qa for v{visit}{arm}{spectrograph}")

        # Select sky fibers.
        selectSky = SelectFibersTask()
        selectSky.config.targetType = ("SKY",)  # Selecting only sky fibers
        skyConfig = selectSky.run(pfsConfig.select(fiberId=pfsArm.fiberId))

        # Create a sky model using the given configuration.
        fitSkyModel = FitBlockedOversampledSplineTask(config=fitSkyModelConfig)

        # Perform sky subtraction on PFS spectra while excluding a specific fiber.
        # This loop selects sky fibers, fits a sky model, subtracts it,
        # and returns only the spectrum of the excluded fiber.
        spectras = list()
        for excludeFiberId in skyConfig.fiberId:
            skyConfig0 = skyConfig[skyConfig.fiberId != excludeFiberId]
            skySpectra = pfsArm.select(pfsConfig, fiberId=skyConfig0.fiberId)

            if len(skySpectra) == 0:
                raise ValueError("No sky spectra to use for sky subtraction.")

            # Fit sky model using the given configuration
            sky1d = fitSkyModel.run(skySpectra, skyConfig0)

            # Apply sky subtraction to the full spectra.
            subtractSky1d(pfsArm, pfsConfig, sky1d)

            spectras.append(pfsArm[pfsArm.fiberId == excludeFiberId])

        try:
            merged_spectra = PfsArm.fromMerge(spectras)
            merged_spectra.metadata["blockSize"] = fitSkyModelConfig.blockSize
            return Struct(skySubtraction_mergedSpectra=merged_spectra)
        except Exception as e:
            self.log.error(f"Failed to merge spectra: {e}")
            return Struct()


class SkySubtractionConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "spectrograph"),
):
    """Connections for SkySubtractionTask"""

    skySubtraction_mergedSpectra = InputConnection(
        name="skySubtraction_mergedSpectra",
        doc="Merged spectra after sky subtraction",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    skySubtractionQaPlot = OutputConnection(
        name="skySubtractionQaPlot",
        doc="Sky Subtraction Plots: 1d, 2d, outliers, and vs sky brightness",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument", "visit", "spectrograph"),
    )


class SkySubtractionConfig(PipelineTaskConfig, pipelineConnections=SkySubtractionConnections):
    """Configuration for SkySubtractionTask"""


class SkySubtractionQaTask(PipelineTask):
    """Task for QA of skySubtraction"""

    ConfigClass = SkySubtractionConfig
    _DefaultName = "skySubtraction"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        inputs = butlerQC.get(inputRefs)

        try:
            # Perform the actual processing.
            outputs = self.run(**inputs)
        except ValueError as e:
            self.log.error(e)
        else:
            # Store the results if valid.
            butlerQC.put(outputs, outputRefs)

    def run(self, skySubtraction_mergedSpectra: Iterable[PfsArm], **kwargs) -> Struct:
        """Perform QA on sky subtraction.

        Parameters
        ----------
        skySubtraction_mergedSpectra : `Iterable[pfs.drp.stella.PfsArm]`
            The input PfsArm data.

        Returns
        -------
        Struct
            A struct containing the plots.
        """
        spectras = dict()
        arms = list()
        blockSize = None
        for pfsArm in skySubtraction_mergedSpectra:
            spectrograph = pfsArm.identity.spectrograph
            arm = pfsArm.identity.arm
            visit = pfsArm.identity.visit
            spectras[(spectrograph, arm)] = pfsArm
            arms.append(arm)
            if blockSize is None:
                blockSize = pfsArm.metadata["blockSize"]
            elif blockSize != pfsArm.metadata["blockSize"]:
                raise ValueError("Block size mismatch between arms.")

        spectraDict = convertToDict(spectras)
        plotId = dict(visit=visit, arm=arm, spectrograph=spectrograph, block=blockSize)
        arms = list(set(arms))

        self.log.info(f"Plotting 1D spectra for arms {arms}.")
        fig_1d, _ = plot_1d_spectrograph(spectraDict, plotId, arms)

        self.log.info(f"Plotting 2D spectra for arms {arms}.")
        fig_2d, _ = plot_2d_spectrograph(spectras, plotId, arms)

        self.log.info(f"Plotting outlier summary for arms {arms}.")
        fig_outlier, ax_dicts = plot_outlier_summary(spectras, spectraDict, plotId, arms)

        self.log.info(f"Plotting vs sky brightness for arms {arms}.")
        fig_sky_brightness, _ = plot_vs_sky_brightness(spectras, plotId, arms)

        pdf = MultipagePdfFigure()
        pdf.append(fig_1d)
        pdf.append(fig_2d)
        for fig in fig_outlier:
            pdf.append(fig)
        pdf.append(fig_sky_brightness)

        return Struct(skySubtractionQaPlot=pdf)


def convertToDict(spectras: dict):
    """
    Convert spectral data into a structured dictionary format.

    This function processes PFS spectral data and organizes it into a nested dictionary,
    where each entry corresponds to a spectrograph arm and fiber ID, storing relevant spectral
    properties such as wavelength, flux, standard deviation, sky background, and chi values.

    - Uses `extractFiber` to extract spectral information for each fiber.
    - Each spectrograph arm has its own dictionary containing fiber-specific data.
    - If `pfsConfig` is not found in `hold`, fiber positions (`xy`, `ra_dec`) will be omitted.

    Parameters
    ----------
    spectras : `dict`
        Dictionary containing spectral data indexed by `(spectrograph, arm)`.
        Expected to contain a `pfsConfig` key for positional metadata.

    Returns
    -------
    ret : `dict`
        Nested dictionary with structure:
        ```
        {
            (spectrograph, arm): {
                fiberId: {
                    'wave'  : numpy.ndarray,  # Wavelength values
                    'flux'  : numpy.ndarray,  # Flux values
                    'std'   : numpy.ndarray,  # Standard deviation of flux
                    'sky'   : numpy.ndarray,  # Sky background values
                    'chi'   : numpy.ndarray,  # Chi values (flux/std)
                    'xy'    : tuple(float, float),  # PFI nominal position (x, y)
                    'ra_dec': tuple(float, float)   # Sky coordinates (RA, Dec)
                },
                ...
            },
            ...
        }
        ```
    """
    spectras = spectras.copy()  # Prevent modification of the original input
    pfsConfig = spectras.pop("pfsConfig", None)  # Extract pfsConfig metadata if available

    ret = {}  # Dictionary to store processed data

    # Iterate over spectrograph-arm combinations
    for (spectrograph, arm), spectra in spectras.items():
        ret[(spectrograph, arm)] = {}

        # Process each fiber
        for iFib, fiberId in enumerate(spectra.fiberId):
            # Extract spectral data
            wave, flux, std, sky, chi, stdPoisson, chiPoisson, C = extractFiber(
                spectra, fiberId=fiberId, finite=True
            )

            # Initialize fiber entry
            ret[(spectrograph, arm)][fiberId] = {
                "wave": wave,
                "flux": flux,
                "std": std,
                "sky": sky,
                "chi": chi,
                "stdPoisson": stdPoisson,
                "chiPoisson": chiPoisson,
            }

            # Add positional metadata if `pfsConfig` is available
            if pfsConfig:
                ret[(spectrograph, arm)][fiberId].update(
                    {
                        "xy": tuple(pfsConfig.pfiNominal[iFib]),  # PFI nominal position (x, y)
                        "ra_dec": (pfsConfig.ra[iFib], pfsConfig.dec[iFib]),  # Sky coordinates (RA, Dec)
                    }
                )

    return ret


def summarizeSpectrograph(
    hold: dict,
    spectrograph: int,
    arms: Iterable[str] = ("b", "r", "n"),
    fontsize: int = 25,
    xlim: tuple[int, int] = (-10, 10),
    alpha: float = 0.2,
):
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
    fontsize : `int`, optional
        Font size for labels and titles (default: 25).
    xlim : `tuple` of `int`, optional
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
    all_axs = ["ABC", "DEF", "GHI"][: len(arms)]
    axt = "\n".join(all_axs)
    fig, ax_dict = get_mosaic(axt, figsize=(20, 10))

    # Iterate over arms and generate histograms.
    for color, arm, axs in zip(plot_colors, arms, all_axs):
        h = hold[(spectrograph, arm)]
        big_chi = []  # Store all chi values for overall distribution
        layers = []
        means = []
        stdev = []

        # Process each fiber.
        for fib in h.keys():
            chi = h[fib]["chi"]
            chiPoisson = h[fib]["chiPoisson"]

            # Add a histogram layer for each fiber.
            layers.append(
                Layer("hist", chi, color=color, alpha=alpha, linewidth=2, density=True, rnge=xlim, bins=30)
            )

            # Add a histogram layer for each fiber.
            layers.append(
                Layer("hist", chiPoisson, color="k", alpha=0.1, linewidth=2, density=True, rnge=xlim, bins=30)
            )

            # Compute statistical metrics
            means.append([np.mean(chi), np.median(chi)])
            stdev.append([np.std(chi), getStddev(chi, useIQR=True)])
            big_chi.extend(chi)

        # Convert lists to NumPy arrays for easier processing
        means = np.array(means)
        stdev = np.array(stdev)

        # Add combined chi distribution (all fibers) with a distinctive color
        layers.append(
            Layer("hist", big_chi, color="magenta", alpha=1, linewidth=6, density=True, rnge=xlim, bins=30)
        )

        # Plot chi distribution
        make_plot(
            layers,
            ax_dict[axs[0]],
            xlim=xlim,
            xlabel=r"$\chi$" if arm == arms[-1] else None,
            ylabel=f"Arm: {arm}\nPDF",
            fontsize=fontsize,
        )

        # Labels for statistics
        labels = [["Mean", "Median"], ["Stdev", "IQR Stdev"]]
        rnge_options = [(-3, 3), (0, 3)]  # Range for mean/median and stdev/IQR plots

        # Iterate over mean/median and stdev/IQR plots
        for j, x, ax, rnge in zip(range(2), [means, stdev], axs[1:], rnge_options):
            other = [Layer("vert", X=0 if j == 0 else 1, linestyle="--", zorder=10)]

            # Generate the histogram layers for statistical metrics
            hist_layers = [
                Layer(
                    "hist",
                    X=x[:, i],
                    color=color,
                    alpha=[1, 0.5][i],
                    density=True,
                    rnge=rnge,
                    bins=30,
                    linewidth=4,
                    histtype=["step", "stepfilled"][i],
                    label=labels[j][i],
                )
                for i in range(2)
            ]

            # Plot statistical metrics
            make_plot(
                other + hist_layers,
                ax_dict[ax],
                xlim=rnge,
                legend="A" in axs,
                loc="upper right",
                fontsize=fontsize,
                title=f"Spectrograph: {spectrograph}" if ((arm == arms[0]) and (j == 0)) else None,
                xlabel=(
                    [r"Mean/Median $\chi$", r"Stdev/$\sigma_\mathrm{IQR}$ $\chi$"][j]
                    if arm == arms[-1]
                    else None
                ),
            )

        # Remove x-axis labels for non-bottom plots
        if arm != arms[-1]:
            for ax in axs:
                ax_dict[ax].axes.xaxis.set_ticklabels([])

        # Remove y-axis labels for all plots
        for ax in axs:
            ax_dict[ax].axes.yaxis.set_ticklabels([])

    return fig, ax_dict


def plot_1d_spectrograph(
    spectraDict: dict,
    plotId: dict,
    arms: List[str],
    fontsize: int = 22,
    xlim: tuple[int, int] = (-5, 5),
):
    """
    Generate 1D plots summarizing spectrograph data, including a Gaussian reference.

    Parameters
    ----------
    spectraDict : `dict`
        Dictionary containing spectrograph data.
    plotId : `dict`
        Dictionary containing plot metadata (`visit`, `spectrograph`, `block`).
    arms : `list` of `str`
        List of spectral arms (e.g., ['b', 'r', 'n']).
    fontsize : `int`, optional
        Font size for labels and titles (default: 22).
    xlim : `tuple` of `int`, optional
        X-axis limits for the plots (default: (-5, 5)).

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The generated figure.
    ax_dict : `dict`
        Dictionary of axes corresponding to the plotted elements.
    """
    visit, spectrograph, block = plotId["visit"], plotId["spectrograph"], plotId["block"]

    all_axs = ["ABC", "DEF", "GHI"][: len(arms)]
    all_labels = ["Blue arm\n", "Red arm\n", "NIR arm\n"][: len(arms)]
    ax0 = [ax[0] for ax in all_axs]

    # Generate spectrograph summary plots
    fig, ax_dict = summarizeSpectrograph(
        spectraDict, spectrograph=spectrograph, arms=arms, fontsize=fontsize, xlim=xlim, alpha=0.5
    )

    # Generate Gaussian distribution
    xp = np.linspace(-6, 6, 1000)
    yp = scipy.stats.norm.pdf(xp, loc=0, scale=1)

    # Update axis labels and add Gaussian reference
    for ax, arm in zip(ax0, all_labels):
        ax_dict[ax].set_ylabel(arm, fontsize=fontsize)
        ax_dict[ax].plot(xp, yp, color="k", linewidth=4, linestyle="--")

    # Set title
    ax_dict["B"].set_title(f"visit={visit}; SM{spectrograph}; blocksize={block}", fontsize=fontsize)

    # Add legend
    ax_dict["A"].plot([], [], color=arm_colors[0], label="DRP")
    ax_dict["A"].plot([], [], color="magenta", label="Combined DRP")
    ax_dict["A"].plot([], [], color="k", label="Using Poisson errors")

    ax_dict["A"].legend(fontsize=fontsize * 0.6, loc="upper left")

    return fig, ax_dict


def plot_2d_spectrograph(spectras: dict, plotId: dict, arms: List[str], binsize: int | None = 10):
    """
    Generate a 2D spectrograph plot showing sky subtraction residuals.

    This function visualizes the chi values and sky subtraction performance
    across fibers and wavelengths.

    Parameters
    ----------
    spectras : `dict`
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
    visit, spectrograph, block = plotId["visit"], plotId["spectrograph"], plotId["block"]

    # Copy and remove pfsConfig to avoid unnecessary data.
    specs = spectras.copy()
    specs.pop("pfsConfig", None)

    # Define axis layout for the number of arms.
    axt = "ABC"[: len(arms)]
    fig, ax_dict = get_mosaic(axt, figsize=(15, 5))

    # Loop through each spectral arm.
    for i, arm in enumerate(arms):
        ax = axt[i]
        skySpectra = specs[(spectrograph, arm)]

        # Build reference spectra.
        references = buildReference(skySpectra, func=None, model="chi")

        # Extract data.
        x, y = references
        xb, yb, eb = rolling(x, y[0], binsize)

        # Initialize 2D array for fiber-based spectral residuals.
        z = np.ones((len(xb), len(y)))

        for i, yi in enumerate(y):
            xb, yb, eb = rolling(x, yi, binsize)
            z[:, i] = yb

        # Create a mesh grid for plotting.
        X, Y = np.meshgrid(np.arange(len(y)), xb)

        # Plot 2D colormap of residuals.
        sc = ax_dict[ax].pcolormesh(X, Y, z, vmin=-1, vmax=1, cmap="bwr")

        # Configure plot labels.
        make_plot(
            [],
            ax_dict[ax],
            xlabel="Fiber Index",
            ylabel="Wavelength [nm]" if ax == "A" else None,
            title=f"Arm: {arm}",
        )

    # Add colorbar and overall title.
    fig.colorbar(sc, ax=ax_dict["A"], location="left")
    fig.suptitle(f"visit={visit}; SM{spectrograph}; blocksize={block}", fontsize=22)

    return fig, ax_dict


def plot_outlier_summary(spectras: dict, spectraDict: dict, plotId: dict, arms: List[str]):
    """
    Generate a summary plot highlighting outliers in sky subtraction residuals.

    This function visualizes spectral regions where the absolute chi values exceed
    predefined thresholds (5 and 15) and provides a sky model reference plot.

    Parameters
    ----------
    spectras : `dict`
        Dictionary containing spectrograph data.
    spectraDict : `dict`
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
    visit, spectrograph, block = plotId["visit"], plotId["spectrograph"], plotId["block"]

    # Copy and remove pfsConfig to avoid unnecessary data.
    specs = spectras.copy()
    specs.pop("pfsConfig", None)

    figs, ax_dicts = [], []

    # Loop through each spectral arm.
    for i, arm in enumerate(arms):
        skySpectra = specs[(spectrograph, arm)]

        # Create a figure layout.
        fig, ax_dict = get_mosaic(
            """
            AAB
            AAB
            """,
            figsize=(6, 4),
        )

        # Retrieve fiber data.
        fibers = spectraDict[(spectrograph, arm)]

        # Compute sky reference spectrum.
        wve_sky, flx_sky = buildReference(skySpectra, func=np.nanmedian, model="sky")

        # Loop over fibers and plot outliers.
        for fiberId, fiber in fibers.items():
            wve, _, chi = fiber["wave"], fiber["flux"], fiber["chi"]
            absChi = np.abs(chi)

            # Define outlier conditions.
            C1 = (absChi > 5) & (absChi < 15)
            C2 = absChi > 15

            # Plot scatter points for outliers.
            for C, color in zip([C1, C2], ["steelblue", "navy"]):
                sc = ax_dict["A"].scatter(
                    [fiberId] * len(wve[C]), wve[C], c=absChi[C], vmin=5, vmax=15, cmap="viridis"
                )

        # Adjust plot limits.
        ax_dict["A"].set_xlim(min(fibers.keys()) - 10, max(fibers.keys()) + 10)

        # Plot the sky spectrum.
        ax_dict["B"].plot(flx_sky, wve_sky)

        # Add colorbar.
        fig.colorbar(sc, ax=ax_dict["A"], location="top")

        # Add title.
        fig.suptitle(f"visit={visit}; SM{spectrograph}; Arm {arm}; blocksize={block}")

        # Store figure and axis dictionary.
        figs.append(fig)
        ax_dicts.append(ax_dict)

    return figs, ax_dicts


def plot_vs_sky_brightness(spectras: dict, plotId: dict, arms: List[str]):
    """
    Generate plots comparing sky brightness with spectral residuals.

    This function visualizes the relationship between median residual flux
    and sky brightness percentile, as well as how residuals change with wavelength.

    Parameters
    ----------
    spectras : `dict`
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
    visit, spectrograph, block = plotId["visit"], plotId["spectrograph"], plotId["block"]

    # Define panel layout dynamically based on arms.
    axt = [t[: len(arms)] for t in ["ABC", "DEF"]]
    axt = "\n".join(axt)
    panel_labels = ["".join([a[i] for a in axt.split("\n")]) for i in range(len(arms))]

    # Create a figure layout.
    fig, ax_dict = get_mosaic(axt, figsize=(int(5 * len(arms)), 10))

    # Copy and remove pfsConfig to avoid unnecessary data.
    specs = spectras.copy()
    specs.pop("pfsConfig", None)

    # Loop through each spectral arm.
    # TODO (wtgee) this should be moved out of the plotting code.
    for i, arm in enumerate(arms):
        skySpectra = specs[(spectrograph, arm)]

        # Split into reference and test spectra.
        referenceSpectra, testSpectra = splitSpectraIntoReferenceAndTest(skySpectra)

        # Compute reference and test statistics.
        references_sky = buildReference(referenceSpectra, func=np.nanmedian, model="none")
        references_flx = buildReference(testSpectra, func=np.median, model="residuals")
        # references_err = buildReference(testSpectra, func='quadrature', model='variance')
        references_chi_median = buildReference(testSpectra, func=np.median, model="chi")

        color = arm_colors[i]
        col = panel_labels[i]

        # Interpolate sky brightness onto a residual wavelength grid.
        wve_sky, sky = references_sky
        wve, flx = references_flx
        sky = np.interp(wve, wve_sky, sky)

        chi = references_chi_median[1]

        # Compute ranked percentile of sky brightness.
        ranked = np.argsort(np.argsort(sky))
        ranked = 100 * ranked / len(ranked)

        # Bin residuals based on sky brightness percentiles.
        yb, xb, eb = rolling(ranked, chi, 10)

        # Scatter plot of residual flux vs wavelength.
        ax_dict[col[0]].scatter(wve, flx, s=1, color=color, rasterized=True, alpha=0.7)
        ax_dict[col[0]].plot(wve, sky / 100, color="k", linewidth=1, alpha=0.6, label="1% sky")

        # Scatter plot of residuals vs sky brightness percentile.
        ax_dict[col[1]].scatter(chi, ranked, s=1, color=color, rasterized=True, alpha=0.7)
        ax_dict[col[1]].errorbar(xb, yb, xerr=eb, color="k", linewidth=3)

        # Set axis limits.
        ax_dict[col[0]].set_ylim(-100, 100)
        ax_dict[col[1]].set_xlim(-0.5, 0.5)

        # Set axis labels.
        ax_dict[col[0]].set_xlabel("Wavelength [nm]")
        ax_dict[col[0]].set_ylabel("Median Counts")

        ax_dict[col[1]].set_xlabel(r"Median $\chi$")
        ax_dict[col[1]].set_ylabel("Sky Counts Percentile")

        # Add reference lines.
        ax_dict[col[1]].axvline(0, linestyle="--", color="k")
        ax_dict[col[0]].axhline(0, linestyle="--", color="k")

    # Add legend and title.
    ax_dict["A"].legend()
    fig.suptitle(f"visit={visit}; SM{spectrograph}; blocksize={block}")

    return fig, ax_dict


def rolling(x: NDDataArray, y: NDDataArray, sep: int):
    """
    Compute a rolling statistic over a dataset, binning data points into segments of size `sep`.

    Parameters
    ----------
    x : `numpy.ndarray`
        The independent variable (e.g., wavelength, time).
    y : `numpy.ndarray`
        The dependent variable (e.g., flux, intensity).
    sep : `int`
        The bin width for segmenting `x`.

    Returns
    -------
    xw : `numpy.ndarray`
        The center points of the bins.
    yw : `numpy.ndarray`
        The median of `y` values in each bin.
    ew : `numpy.ndarray`
        The computed standard deviation (or IQR-based metric) for `y` in each bin.

    Notes
    -----
    - The function slides over `x` in steps of `sep`, computing statistics for each window.
    - The rolling statistic is defined as:
        - `yw`: Median of `y` in the bin.
        - `ew`: Standard deviation (or IQR-based deviation if `get_stdev_func` is IQR-based).
    """
    x0 = x.min()
    xw, yw, ew = [], [], []

    while x0 + sep / 2 < x.max():
        # Define the bin mask
        C = (x > x0) & (x <= x0 + sep)

        if np.any(C):  # Ensure there are valid points in the bin
            xw.append(x0 + sep / 2)  # Bin center
            yw.append(np.median(y[C]))  # Median value of y in the bin
            ew.append(getStddev(y[C]))  # Custom standard deviation function

        x0 += sep  # Move to the next bin

    return np.array(xw), np.array(yw), np.array(ew)


def getStddev(x: NDDataArray, axis: int = 0, useIQR: bool = True):
    """
    Compute the standard deviation of an array using either the interquartile range (IQR)
    or the standard deviation method.

    Parameters
    ----------
    x : `numpy.ndarray`
        Input array for which the standard deviation is computed.
    axis : `int`, optional
        Axis along which the standard deviation is computed (default: 0).
    useIQR : `bool`, optional
        If `True`, computes a robust estimate of standard deviation using the IQR method.
        If `False`, computes the standard deviation using `np.std` (default: True).

    Returns
    -------
    stddev : `float` or `numpy.ndarray`
        Estimated standard deviation along the specified axis.

    Notes
    -----
    - The IQR-based estimator uses: `alpha = 0.741 * (Q3 - Q1)`, where:
      - Q1 is the first quartile (25th percentile)
      - Q3 is the third quartile (75th percentile)
      - 0.741 is a conversion factor to approximate standard deviation for a normal distribution.
    - If `useIQR=False`, it falls back to the standard deviation computed via `np.std`.

    """
    if useIQR:
        q1, q3 = np.nanpercentile(x, [25, 75], axis=axis)
        alpha = 0.741 * (q3 - q1)
        return alpha
    else:
        return np.nanstd(x, axis=axis)


def buildReference(spectra: PfsArm, func: Callable = np.mean, model: str = "residuals"):
    """
    Build a reference spectrum by aggregating spectral data from multiple fibers.
    The reference spectrum is constructed by applying a specified aggregation function
    (e.g., mean, median) to the selected model across all fibers.

    Parameters
    ----------
    spectra : `pfs.datamodel.PfsArm`
        PFS spectra object containing spectral data for multiple fibers.
    func : `callable` or `str`, optional
        Function used to aggregate the spectra (default: `numpy.mean`).
        - If 'quadrature', computes the quadrature sum (sqrt of sum of squares).
    model : `str`, optional
        Specifies which data to use when building the reference spectrum.
        Options:
        - 'none'        : Total flux (sky + object flux).
        - 'sky'         : Sky model.
        - 'chi'         : Chi (flux / standard deviation).
        - 'chi_poisson' : Poissonian chi (flux / sqrt(sky + flux)).
        - 'residuals'   : Residual flux (default).
        - 'variance'    : Variance.
        - 'sky_chi'     : (sky + flux) / standard deviation.

    Returns
    -------
    wave_ref : `numpy.ndarray`
        Reference wavelength array.
    sky_ref : `numpy.ndarray`
        Aggregated reference spectrum based on the chosen model.

    Notes
    -----
    - The function extracts and aligns spectra from all fibers to a common wavelength grid.
    - Uses interpolation to map each fiber's spectrum onto the reference wavelength array.
    - Applies the selected aggregation function (`func`) to compute the final reference spectrum.
    """
    # Containers for spectral data.
    x, y = [], []

    # Process each fiber.
    for fiberId in spectra.fiberId:
        wave, flux, std, sky, chi, stdPoisson, chiPoisson, C = extractFiber(
            spectra, fiberId=fiberId, finite=True
        )

        # Select model to build reference spectrum.
        if model == "none":
            y.append(sky + flux)
        elif model == "sky":
            y.append(sky)
        elif model == "chi":
            y.append(chi)
        elif model == "chi_poisson":
            y.append(chiPoisson)
        elif model == "residuals":
            y.append(flux)
        elif model == "variance":
            y.append(std**2)
        elif model == "sky_chi":
            y.append((sky + flux) / std)
        else:
            raise ValueError(
                "Unsupported model. Choose from [residuals, chi, sky, variance, none, chi_poisson, sky_chi]"
            )

        x.append(wave)

    # Choose the longest wavelength grid as the reference.
    wave_ref = x[np.argmax([len(xi) for xi in x])]

    # Interpolate all spectra to the reference wavelength grid.
    sky_ref = [np.interp(wave_ref, wave, sky) for wave, sky in zip(x, y)]

    # Apply the aggregation function to compute final reference spectrum.
    if func:
        if func == "quadrature":
            sky_ref = np.sqrt(np.sum(np.array(sky_ref) ** 2, axis=0))
        else:
            sky_ref = func(np.array(sky_ref), axis=0)

    return wave_ref, sky_ref


def splitSpectraIntoReferenceAndTest(spectra: PfsArm, referenceFraction: float = 0.1):
    """
    Randomly split spectra into reference (10%) and test (90%) subsets.

    This function randomly selects a subset of spectra corresponding to the given `referenceFraction`
    (default: 10%) and assigns the remaining spectra to the test subset.

    Parameters
    ----------
    spectra : `pfs.datamodel.PfsArm`
        The PFS spectra object containing spectral data for multiple fibers.
    referenceFraction : `float`, optional
        The fraction of spectra to include in the reference subset (default: 0.1).
        Must be between 0 and 1.

    Returns
    -------
    referenceSpectra : `pfs.datamodel.PfsArm`
        A subset containing `referenceFraction` of the spectra, used as the reference.
    testSpectra : `pfs.datamodel.PfsArm`
        The remaining `1 - referenceFraction` of the spectra, used for testing.

    Notes
    -----
    - The function shuffles fiber IDs before splitting to ensure randomness.
    - Uses `np.isin` to efficiently filter the spectra based on fiber ID.
    - The sum of spectra in both subsets equals the original spectra.
    """
    if not (0 < referenceFraction < 1):
        raise ValueError("referenceFraction must be between 0 and 1.")

    # Randomly shuffle fiber IDs.
    shuffled_fiber_ids = np.random.choice(spectra.fiberId, size=len(spectra), replace=False)

    # Split at the reference fraction point.
    split_idx = int(len(shuffled_fiber_ids) * referenceFraction)
    reference_fiber_ids = shuffled_fiber_ids[:split_idx]
    test_fiber_ids = shuffled_fiber_ids[split_idx:]

    # Filter spectra based on selected fiber IDs,
    referenceSpectra = spectra[np.isin(spectra.fiberId, reference_fiber_ids)]
    testSpectra = spectra[np.isin(spectra.fiberId, test_fiber_ids)]

    return referenceSpectra, testSpectra


def extractFiber(spectra: PfsArm, fiberId: int, finite: bool = True):
    """
    Extract relevant spectral data for a specific fiber.

    This function retrieves wavelength, flux, sky background, variance, and
    computes standard deviation and chi values for the given fiber.

    Parameters
    ----------
    spectra : `pfs.datamodel.PfsArm`
        PFS spectra object containing spectral data.
    fiberId : `int`
        The fiber ID for which the data needs to be extracted.
    finite : `bool`, optional
        If True, only finite values (non-NaN, positive sky, and positive variance) are returned.

    Returns
    -------
    wave : `numpy.ndarray`
        Wavelength array for the selected fiber.
    flux : `numpy.ndarray`
        Flux values for the selected fiber.
    std : `numpy.ndarray`
        Standard deviation (sqrt of variance) for the selected fiber.
    sky : `numpy.ndarray`
        Sky background values for the selected fiber.
    chi : `numpy.ndarray`
        Chi values computed as flux / sqrt(variance).
    C : `numpy.ndarray`
        Boolean mask indicating valid (finite) data points.

    Notes
    -----
    - The function applies filtering to ensure valid data: sky > 0 and variance > 0.
    - If `finite` is True, only valid data points are returned.
    """
    # Identify the index of the fiber in the spectra.
    j = spectra.fiberId == fiberId

    # Retrieve corresponding arrays.
    wave = spectra.wavelength[j][0]
    flux = spectra.flux[j][0]
    sky = spectra.sky[j][0]
    var = spectra.variance[j][0]

    # Define a validity mask: data must be finite, with positive sky & variance.
    C = np.isfinite(flux) & (sky > 0) & (var > 0)
    # adding masked values.
    C = np.logical_and(C, spectra.mask[j][0] == 0)

    # Compute standard deviation (sqrt of variance), setting invalid values to NaN.
    std = np.ones_like(var) * np.nan
    std[C] = np.sqrt(var[C])

    # Compute chi (flux divided by standard deviation), handling invalid cases.
    chi = np.ones_like(var) * np.nan
    chi[C] = flux[C] / std[C]

    # calculating poisson error only.
    stdPoisson = np.ones_like(var) * np.nan
    stdPoisson[C] = np.sqrt(np.abs(flux[C] + sky[C]))

    # calculating chi using poisson error only.
    chiPoisson = np.ones_like(var) * np.nan
    chiPoisson[C] = flux[C] / stdPoisson[C]

    # Return only finite values if requested.
    if finite:
        wave, flux, std, sky, chi = wave[C], flux[C], std[C], sky[C], chi[C]
        stdPoisson, chiPoisson = stdPoisson[C], chiPoisson[C]

    return wave, flux, std, sky, chi, stdPoisson, chiPoisson, C


@dataclass
class Layer:
    version: str = "scatter"
    X: Number | List[Number] | None = None
    Y: Number | List[Number] | None = None
    Z: Number | List[Number] | None = None
    XERR: List[Number] | None = None
    YERR: List[Number] | None = None
    W: List[Number] | None = None
    STR: str | None = None
    rnge: tuple[Number, Number] | None = None
    label: str | None = None
    zlabel: str | None = None
    capsize: float | None = None
    color: str = "k"
    shape: str = "o"
    alpha: float = 1.0
    zorder: float = 1.0
    size: float | None = None
    linestyle: str = "-"
    linewidth: int = 3
    bar: bool | None = None
    cumulative: bool = False
    bins: str | int = "auto"
    density: bool = False
    contours: bool | None = None
    smooth: int | None = None
    bold: bool = False
    orientation: str = "vertical"
    step: str | None = None
    histtype: str = "step"
    vmin: float | None = None
    vmax: float | None = None


def make_plot(
    layers: Iterable[Layer],
    ax: Axes | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    fontsize: int = 20,
    title: str = None,
    xlim: tuple[int, int] | None = None,
    ylim: tuple[int, int] | None = None,
    legend: bool = False,
    square: bool = False,
    figsize=(8, 5),
    frameon=False,
    xticks=None,
    yticks=None,
    xreverse=False,
    yreverse=False,
    xrotation=None,
    yrotation=None,
    xlog=False,
    ylog=False,
    loc="best",
    ncol=1,
):
    """Add the plot layer.

    Generates a customizable plot by adding layers and applying various visual
    settings. This function allows for detailed configuration of plot appearance,
    including axis labels, font sizes, ticks, log scales, legends, titles, and
    more. It also supports optional axis reversal and aspect ratio adjustments.

    Parameters:
        layers (Iterable[Layer]): A collection of plot layers to be added to the
            plot. Each layer represents an individual component of the plot.
        ax (Axes | None, optional): An optional Axes instance where the plot will
            be rendered. If not provided, a new Axes will be created.
        xlabel (str | None, optional): Label for the x-axis. If None, no label
            will be added.
        ylabel (str | None, optional): Label for the y-axis. If None, no label
            will be added.
        fontsize (int, optional): Font size to be used for plot labels, ticks, and
            title. Default is 20.
        title (str | None, optional): Title of the plot. If None, no title will be
            added.
        xlim (tuple[int, int] | None, optional): Limits for the x-axis as a tuple
            of two integers. If None, no limits are set.
        ylim (tuple[int, int] | None, optional): Limits for the y-axis as a tuple
            of two integers. If None, no limits are set.
        legend (bool, optional): Whether to display a legend for the plot. Default
            is False.
        square (bool, optional): Whether the axes should be adjusted to have equal
            aspect ratio. Default is False.
        figsize (tuple, optional): Size of the figure in inches. Default is (8, 5).
        frameon (bool, optional): Whether a frame should be displayed around the
            legend. Default is False.
        xticks (optional): Custom values and labels for x-axis ticks. Can be None
            or a tuple of two lists (tick positions and tick labels).
        yticks (optional): Custom values and labels for y-axis ticks. Can be None
            or a tuple of two lists (tick positions and tick labels).
        xreverse (bool, optional): Whether to invert the x-axis. Default is False.
        yreverse (bool, optional): Whether to invert the y-axis. Default is False.
        xrotation (optional): Rotation angle for x-axis tick labels. If None, no
            rotation is applied.
        yrotation (optional): Rotation angle for y-axis tick labels. If None, no
            rotation is applied.
        xlog (bool, optional): Whether to use a logarithmic scale for the x-axis.
            Default is False.
        ylog (bool, optional): Whether to use a logarithmic scale for the y-axis.
            Default is False.
        loc (str, optional): Location for the legend on the plot. Default is
            "best".
        ncol (int, optional): Number of columns in the legend. Default is 1.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(figsize)

    # Add the plotting layers.
    for layer in layers:
        layer.fontsize = fontsize
        add_layer(ax, layer)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.tick_params(axis="both", which="major", labelsize=fontsize, length=10, width=2)
    ax.tick_params(axis="both", which="minor", labelsize=fontsize, length=3, width=1)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    if legend:
        ax.legend(fontsize=fontsize, frameon=frameon, loc=loc, ncol=ncol)

    if title:
        ax.set_title(title, fontsize=fontsize)

    ax.ticklabel_format(useOffset=False)

    if ylog:
        ax.set_yscale("log")

    if xlog:
        ax.set_xscale("log")

    if xticks:
        if xticks == "None":
            ax.set_xticklabels([], fontsize=fontsize)
        else:
            ax.set_xticks(xticks[0])
            ax.set_xticklabels(xticks[1], fontsize=fontsize, rotation=xrotation)

            ax.tick_params(axis="x", which="major", length=10, width=2)
            ax.tick_params(axis="x", which="minor", labelsize=0, length=5, width=1)

    if yticks:
        if yticks == "None":
            ax.set_yticklabels([], fontsize=fontsize)
        else:
            ax.set_yticks(yticks[0])
            ax.set_yticklabels(yticks[1], fontsize=fontsize, rotation=yrotation)
            ax.tick_params(axis="y", which="major", length=10, width=2)
            ax.tick_params(axis="y", which="minor", labelsize=0, length=5, width=1)

    if yreverse:
        ax.invert_yaxis()
    if xreverse:
        ax.invert_xaxis()

    if type(ax) is None:
        fig.tight_layout()

    if square:
        ax.set_aspect("equal", adjustable="box")


def get_mosaic(mosaic="A", figsize=(10, 10)):
    """Create a figure with a specified layout using matplotlib's subplot_mosaic."""
    mosaic = mosaic
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    ax_dict = fig.subplot_mosaic(mosaic)
    return fig, ax_dict


def add_layer(ax: Axes, layer: Layer):
    """Adds a layer to the plot based on the specified version.

    The bulk of the plotting options and work is done in this function.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The axes to which the layer will be added.
    layer : `Layer`
        The layer object containing the data and properties for the plot.
    """
    if layer.version == "line":
        ax.plot(
            layer.X,
            layer.Y,
            color=layer.color,
            linestyle=layer.linestyle,
            linewidth=layer.linewidth,
            label=layer.label,
            alpha=layer.alpha,
            zorder=layer.zorder,
        )

    elif layer.version == "scatter":
        if not layer.rnge:
            rnge = [None, None]
        else:
            rnge = layer.rnge
        if isinstance(layer.Z, (np.ndarray, list)):

            if layer.bold:
                p1 = ax.scatter(
                    layer.X,
                    layer.Y,
                    c=layer.Z,
                    cmap="coolwarm",
                    label=layer.label,
                    marker=layer.shape,
                    vmin=rnge[0],
                    vmax=rnge[1],
                    alpha=layer.alpha,
                    edgecolor="k",
                    linewidth=layer.linewidth,
                    s=layer.size,
                )
            else:
                p1 = ax.scatter(
                    layer.X,
                    layer.Y,
                    c=layer.Z,
                    cmap=layer.color,
                    label=layer.label,
                    s=layer.size,
                    marker=layer.shape,
                    vmin=rnge[0],
                    vmax=rnge[1],
                    alpha=layer.alpha,
                )
            if layer.bar:
                c1 = plt.colorbar(p1, ax=ax, orientation="vertical")
                if layer.zlabel:
                    c1.set_label(layer.zlabel, fontsize=layer.fontsize)
                c1.ax.tick_params(labelsize=layer.fontsize)
        else:
            ax.scatter(
                layer.X,
                layer.Y,
                label=layer.label,
                color=layer.color,
                alpha=layer.alpha,
                marker=layer.shape,
                s=layer.size,
            )

    elif layer.version == "hist":

        ax.hist(
            layer.X,
            bins=layer.bins,
            zorder=layer.zorder,
            density=layer.density,
            weights=layer.W,
            color=layer.color,
            alpha=layer.alpha,
            range=layer.rnge,
            label=layer.label,
            linestyle=layer.linestyle,
            linewidth=layer.linewidth,
            histtype=layer.histtype,
            cumulative=layer.cumulative,
            orientation=layer.orientation,
        )

    elif layer.version == "bar":

        ax.bar(
            layer.X,
            height=layer.Y,
            width=layer.W,
            color="none",
            alpha=layer.alpha,
            edgecolor=layer.color,
            label=layer.label,
            linestyle=layer.linestyle,
            align="center",
            linewidth=4,
        )

    elif layer.version == "vert":

        ax.axvline(
            layer.X,
            linestyle=layer.linestyle,
            color=layer.color,
            linewidth=layer.linewidth,
            label=layer.label,
            alpha=layer.alpha,
            zorder=layer.zorder,
        )

    elif layer.version == "horiz":

        ax.axhline(
            layer.X,
            linestyle=layer.linestyle,
            color=layer.color,
            linewidth=layer.linewidth,
            label=layer.label,
            alpha=layer.alpha,
            zorder=layer.zorder,
        )
    elif layer.version == "arrow":

        ax.arrow(
            layer.X[0],
            layer.X[1],
            layer.X[2],
            layer.X[3],
            color=layer.color,
            width=layer.linewidth,
            label=layer.label,
            alpha=layer.alpha,
            zorder=layer.zorder,
        )

    elif layer.version == "fill":

        ax.fill_between(
            layer.X,
            layer.Y[0],
            layer.Y[1],
            color=layer.color,
            alpha=layer.alpha,
            step=layer.step,
            label=layer.label,
        )

    elif layer.version == "step":

        ax.step(
            layer.X,
            layer.Y,
            color=layer.color,
            linewidth=layer.linewidth,
            alpha=layer.alpha,
            label=layer.label,
            linestyle=layer.linestyle,
            where="post",
        )

    elif layer.version == "errorbar":

        ax.errorbar(
            layer.X,
            layer.Y,
            xerr=layer.XERR,
            yerr=layer.YERR,
            zorder=layer.zorder,
            color=layer.color,
            alpha=layer.alpha,
            markersize=layer.size,
            label=layer.label,
            linestyle=layer.linestyle,
            capsize=layer.capsize,
            marker=layer.shape,
        )

    elif layer.version == "text":
        ax.text(layer.X, layer.Y, layer.STR, fontsize=layer.fontsize, color=layer.color, alpha=layer.alpha)

    elif layer.version == "hist2d":
        if not layer.rnge:
            rnge = [[min(layer.X), max(layer.X)], [min(layer.Y), max(layer.Y)]]
        else:
            rnge = layer.rnge
        H, xbins, ybins = np.histogram2d(layer.X, layer.Y, weights=layer.W, bins=layer.bins, range=rnge)

        H = np.rot90(H)
        H = np.flipud(H)

        X, Y = np.meshgrid(xbins[:-1], ybins[:-1])

        if layer.smooth is None:
            from scipy.signal import wiener

            H = wiener(H, mysize=layer.smooth)

        H = H / np.sum(H)
        Hmask = np.ma.masked_where(H == 0, H)

        cmin = 1e-4
        cmax = 1.0
        if not layer.vmin:
            vmin = cmin * np.max(Hmask)
        else:
            vmin = layer.vmin

        if not layer.vmax:
            vmax = cmax * np.max(Hmask)
        else:
            vmax = layer.vmax

        # norm = LogNorm(vmin,vmax),
        p1 = ax.pcolormesh(
            X,
            Y,
            (Hmask),
            cmap=layer.color,
            vmin=vmin,
            vmax=vmax,
            linewidth=0.0,
            shading="auto",
            alpha=layer.alpha,
            edgecolors=None,
        )
        p1.set_edgecolor("none")

        if layer.bar:
            c1 = plt.colorbar(p1, ax=ax, orientation="vertical")
            if layer.zlabel:
                c1.set_label(layer.zlabel, fontsize=layer.fontsize)
            c1.ax.tick_params(labelsize=layer.fontsize)
