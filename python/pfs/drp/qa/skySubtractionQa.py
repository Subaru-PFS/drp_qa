import contextlib
from dataclasses import dataclass
from numbers import Number
from typing import Callable, Iterable, List

import matplotlib
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sb
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
from matplotlib.axes import Axes
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import DataFrame
from pfs.drp.stella import PfsArm, PfsConfig
from pfs.drp.stella.fitFocalPlane import FitBlockedOversampledSplineConfig, FitBlockedOversampledSplineTask
from pfs.drp.stella.selectFibers import SelectFibersTask
from pfs.drp.stella.subtractSky1d import subtractSky1d
from pfs.drp.stella.utils.math import robustRms

from pfs.drp.qa.storageClasses import MultipagePdfFigure
from pfs.drp.qa.utils.plotting import detector_palette, div_palette

matplotlib.rcParams["font.size"] = 8

mpl.rcParams["xtick.major.size"] = 9
mpl.rcParams["xtick.major.width"] = 1.2
mpl.rcParams["xtick.minor.size"] = 4
mpl.rcParams["xtick.minor.width"] = 1.2

mpl.rcParams["ytick.major.size"] = 9
mpl.rcParams["ytick.major.width"] = 1.2
mpl.rcParams["ytick.minor.size"] = 4
mpl.rcParams["ytick.minor.width"] = 1.2

mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"

mpl.rcParams["ytick.right"] = True

mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.minor.visible"] = True

mpl.rcParams["xtick.top"] = True


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

        if hasattr(mergeArms_config, "fitSkyModel"):
            # Update only if user provides values
            if self.config.blockSize is not None:
                fitSkyModelConfig.blockSize = self.config.blockSize
            if self.config.rejIterations is not None:
                fitSkyModelConfig.rejIterations = self.config.rejIterations
            if self.config.rejThreshold is not None:
                fitSkyModelConfig.rejThreshold = self.config.rejThreshold
            if self.config.oversample is not None:
                fitSkyModelConfig.oversample = self.config.oversample
            if self.config.mask is not None:
                fitSkyModelConfig.mask = self.config.mask

        self.log.info(f"Using sky model configuration: {fitSkyModelConfig}")

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

    skySubtractionFiberStats = OutputConnection(
        name="skySubtractionFiberStats",
        doc="Sky Subtraction Fiber Statistics",
        storageClass="DataFrame",
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

        run_name = inputRefs.skySubtraction_mergedSpectra[0].run

        inputs = butlerQC.get(inputRefs)
        inputs["run_name"] = run_name

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
        skySubtraction_mergedSpectra: Iterable[PfsArm],
        make_pdf: bool = True,
        run_name: str = None,
        **kwargs,
    ) -> Struct:
        """Perform QA on sky subtraction.

        Parameters
        ----------
        skySubtraction_mergedSpectra : `Iterable[pfs.drp.stella.PfsArm]`
            The input PfsArm data.
        make_pdf : `bool`, optional
            If True, generate a PDF with the plots. Default is True,
            otherwise return all plot figures.
        run_name : `str`, optional
            The name of the run, used for logging and titles.

        Returns
        -------
        Struct
            A struct containing the plots if `store_results` is True else None.
        """
        spectras, spectraFibers, stats = getSpectraData(skySubtraction_mergedSpectra)
        arms = [arm for (_, arm) in spectras.keys()]
        identity = list(skySubtraction_mergedSpectra)[0].identity
        visit = identity.visit
        spectrograph = identity.spectrograph

        self.log.info(f"Plotting 1D spectra for arms {arms}.")
        fig_1d = plot_1d_spectrograph(spectraFibers, stats)
        fig_1d.suptitle(
            f"Sky Subtraction QA - {visit} SM{spectrograph}\n" f"Chi (flux / std) of Sky Fibers\n{run_name}",
            fontsize=16,
        )

        self.log.info(f"Plotting 2D spectra for arms {arms}.")
        fig_2d = plot_2d_chi(spectraFibers)

        self.log.info(f"Plotting outlier summary for arms {arms}.")
        fig_outlier = plot_outlier_summary(spectras, spectraFibers)

        self.log.info(f"Plotting sky reference for arms {arms}.")
        fig_sky_ref = plot_sky_reference(spectras)

        self.log.info(f"Plotting vs sky brightness for arms {arms}.")
        fig_sky_brightness_median = plot_vs_sky_brightness(spectras)

        fig_sky_brightness = plot_vs_sky_brightness_all(spectraFibers)

        results = Struct(
            skySubtractionFiberStats=stats,
        )

        if make_pdf:
            # Create a PDF with all the plots.
            pdf = MultipagePdfFigure()
            pdf.append(fig_1d)
            pdf.append(fig_sky_brightness_median)
            pdf.append(fig_sky_brightness)
            pdf.append(fig_2d)
            pdf.append(fig_outlier)
            pdf.append(fig_sky_ref)

            results.skySubtractionQaPlot = pdf
        else:
            # Store the figures in the results struct.
            results.skySubtractionQaPlot = {
                "1d": fig_1d,
                "2d": fig_2d,
                "outlier": fig_outlier,
                "sky_ref": fig_sky_ref,
                "sky_brightness": fig_sky_brightness_median,
                "sky_brightness_all": fig_sky_brightness,
            }

        return results


def getSpectraData(skySubtraction_mergedSpectra: Iterable[PfsArm]) -> tuple[dict, dict, DataFrame]:
    """Extract spectra data from merged spectra.

    This function processes merged spectra data and extracts relevant information
    for each arm. It organizes the data into a structured dictionary format,
    where each entry corresponds to a spectrograph arm and fiber ID, storing
    relevant spectral properties such as wavelength, flux, standard deviation,
    sky background, and chi values.
    - Uses `extractFibers` to extract spectral information for each fiber.
    - Each spectrograph arm has its own dictionary containing fiber-specific data.

    Parameters
    ----------
    skySubtraction_mergedSpectra : `Iterable[pfs.drp.stella.PfsArm]`
        Iterable of merged spectra after sky subtraction.

    Returns
    -------
    spectras : `dict`
        Dictionary containing spectral data indexed by `(spectrograph, arm)`.
        Each entry contains a nested dictionary with fiber IDs as keys and spectral properties
        (wavelength, flux, standard deviation, sky background, chi values) as values.
    spectraFibers : `dict`
        Dictionary containing fiber data indexed by `(spectrograph, arm)`.
    stats : `pandas.DataFrame`
        DataFrame containing statistics for each arm.
    """

    spectras = dict()
    arms = list()
    blockSize = None
    for subtracted_pfsArm in skySubtraction_mergedSpectra:
        spectrograph = subtracted_pfsArm.identity.spectrograph
        arm = subtracted_pfsArm.identity.arm
        spectras[(spectrograph, arm)] = subtracted_pfsArm
        arms.append(arm)
        if blockSize is None:
            blockSize = subtracted_pfsArm.metadata["blockSize"]
        elif blockSize != subtracted_pfsArm.metadata["blockSize"]:
            raise ValueError("Block size mismatch between arms.")

    # Extract the information for the fibers.
    spectraFibers = extractFibers(spectras)

    # Get the stats for each arm.
    stats = getSpectraStats(spectraFibers)

    return spectras, spectraFibers, stats


def getSpectraStats(spectraFibers: dict) -> DataFrame:
    """Compute statistics for each arm in the spectraFibers dictionary.

    This function calculates the mean, median, standard deviation, and interquartile range
    (IQR) of the chi values for each arm in the provided spectraFibers dictionary.

    Parameters
    ----------
    spectraFibers : `dict`
        Dictionary containing fiber data indexed by `(spectrograph, arm)`.


    """
    df = getFiberData(spectraFibers)

    stats = (
        df.groupby(["arm", "spectrograph", "fiberId"], observed=False)
        .chi.agg(["mean", "median", "std", robustRms])
        .reset_index()
    )
    stats.rename(
        columns={
            "mean": "fiberChiMean",
            "median": "fiberChiMedian",
            "std": "fiberChiStd",
            "robustRms": "fiberChiIQR",
        },
        inplace=True,
    )

    return stats


def getFiberData(spectraFibers: dict) -> DataFrame:
    """Convert spectraFibers dictionary into a DataFrame.

    This function takes a nested dictionary structure containing spectral data for different
    spectrograph arms and fibers, and flattens it into a pandas DataFrame. Each row in the
    DataFrame corresponds to a specific fiber, with columns for wavelength, flux, standard
    deviation, sky background, chi values, and additional metadata such as fiber ID, arm,
    and spectrograph.

    Parameters
    ----------
    spectraFibers : `dict`
        Dictionary containing spectral data indexed by `(spectrograph, arm)`.
        Each entry contains a nested dictionary with fiber IDs as keys and spectral properties
        (wavelength, flux, std, sky, chi) as values.

    Returns
    -------
    df : `pandas.DataFrame`
        DataFrame containing the flattened spectral data.
    """
    dfs = list()
    for spec, arm in spectraFibers.keys():
        for fiberId, data in spectraFibers[(spec, arm)].items():
            df = pd.DataFrame(data)
            df["fiberId"] = fiberId
            df["arm"] = str(arm)
            df["spectrograph"] = spec
            dfs.append(df)
    df = pd.concat(dfs)
    df.fiberId = df.fiberId.astype("category")
    df.arm = df.arm.astype("category")
    df.spectrograph = df.spectrograph.astype("category")
    return df


def extractFibers(spectras: dict):
    """Extract fiber data from PFS spectral data.

    Convert spectral data into a structured dictionary format.

    This function processes PFS spectral data and organizes it into a nested dictionary,
    where each entry corresponds to a spectrograph arm and fiber ID, storing relevant spectral
    properties such as wavelength, flux, standard deviation, sky background, and chi values.

    - Uses `extractFiberInfo` to extract spectral information for each fiber.
    - Each spectrograph arm has its own dictionary containing fiber-specific data.

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
                    'std_poisson': numpy.ndarray,  # Poisson standard deviation of flux
                    'chi_poisson': numpy.ndarray,  # Chi values using Poisson errors
                },
                ...
            },
            ...
        }
        ```
    """
    spectraDict = {}  # Dictionary to store processed data

    # Iterate over spectrograph-arm combinations
    for (spectrograph, arm), spectra in spectras.items():
        spectraDict[(spectrograph, arm)] = {}

        # Process each fiber
        for iFib, fiberId in enumerate(spectra.fiberId):
            # Extract spectral data
            wave, flux, std, sky, chi, std_poisson, chi_poisson, C = extractFiberInfo(
                spectra, fiberId=fiberId, finite=True
            )

            # Initialize fiber entry
            spectraDict[(spectrograph, arm)][fiberId] = {
                "wave": wave,
                "flux": flux,
                "std": std,
                "sky": sky,
                "chi": chi,
                "std_poisson": std_poisson,
                "chi_poisson": chi_poisson,
            }

    return spectraDict


def summarizeSpectrograph(
    spectraFibers: dict,
    stats: DataFrame,
    xlim: tuple[int, int] = (-10, 10),
):
    """
    Summarize spectrograph sky subtraction residuals using chi distributions.

    This function generates a summary plot of sky-subtracted residuals (`chi` values)
    across different arms of the spectrograph, comparing mean, median, standard deviation,
    and interquartile range (IQR) statistics provided by the `stats` DataFrame.

    Parameters
    ----------
    spectraFibers : `dict`
        Dictionary containing sky-subtraction residuals for different spectrograph arms.
    stats : `DataFrame`
        DataFrame containing statistics for each arm.
    spectrograph : `int`
        Spectrograph number for labeling the plots.
    arms : `tuple` of `str`, optional
        List of arms to include in the analysis (default: ('b', 'r', 'n')).
    xlim : `tuple` of `int`, optional
        X-axis limits for the histograms (default: (-10, 10)).

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
    arms = [arm for (_, arm) in spectraFibers.keys()]

    all_axs = {arm: [f"{arm}_HIST", f"{arm}_AVG", f"{arm}_ERR"] for arm in "brmn" if arm in arms}
    fig, ax_dict = get_mosaic(all_axs.values(), figsize=(10, 5), sharex=True)

    # Iterate over arms and generate histograms.
    # for plot_color, arm, axs in zip(plot_colors, arms, all_axs):
    for arm in ["b", "r", "m", "n"]:
        for spectrograph in [1, 2, 3, 4]:
            spec_key = (spectrograph, arm)
            if spec_key not in spectraFibers:
                continue

            fibers = spectraFibers[spec_key]
            layers = []
            big_chi = []  # Store all chi values for overall distribution
            plot_color = detector_palette[arm]

            # Process each fiber.
            for fib in fibers.keys():
                chi = fibers[fib]["chi"]
                chi_poisson = fibers[fib]["chi_poisson"]

                # DRP chi distribution per fiber.
                layers.append(
                    PlotLayer("hist", chi, color=plot_color, alpha=0.5, linewidth=2, rnge=xlim, bins=30)
                )

                # Poisson chi distribution per fiber.
                layers.append(
                    PlotLayer("hist", chi_poisson, color="k", alpha=0.1, linewidth=2, rnge=xlim, bins=30)
                )

                # Compute statistical metrics.
                big_chi.extend(chi)

            # Add combined chi distribution (all fibers) with a distinctive color.
            layers.append(
                PlotLayer("hist", big_chi, color="magenta", alpha=1, linewidth=6, rnge=xlim, bins=30)
            )

            # Plot chi distribution.
            make_plot(
                layers,
                ax_dict[all_axs[arm][0]],
                xlim=xlim,
                ylabel=f"Arm: {arm}\nPDF",
            )

            # Labels for statistics
            labels = [["Mean", "Median"], ["Stddev", "IQR Stddev"]]
            rnge_options = [(-3, 3), (-3, 3)]  # Range for mean/median and stddev/IQR plots

            means = stats.loc[stats.spectrograph == spectrograph, ["fiberChiMean", "fiberChiMedian"]].values
            stdev = stats.loc[stats.spectrograph == spectrograph, ["fiberChiStd", "fiberChiIQR"]].values

            # Iterate over mean/median and stdev/IQR plots
            for j, x, ax, rnge in zip(range(2), [means, stdev], all_axs[arm][1:], rnge_options):
                ref_line = [PlotLayer("vert", X=0 if j == 0 else 1, linestyle="--")]

                # Generate the histogram layers for statistical metrics
                hist_layers = [
                    PlotLayer(
                        "hist",
                        X=x[:, i],
                        color=plot_color,
                        alpha=[1, 0.5][i],
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
                    ref_line + hist_layers,
                    ax_dict[ax],
                    xlim=rnge,
                    legend="A" in all_axs[arm],
                    loc="upper right",
                )

    # Set axis labels and titles
    for arm in arms:
        for ax_name in ["HIST", "AVG", "ERR"]:
            ax_dict[f"{arm}_{ax_name}"].set_yticks([])

            if arm == "n":
                ax_dict[f"{arm}_{ax_name}"].set_xlabel(r"$\chi$")

    return fig, ax_dict


def plot_1d_spectrograph(
    spectraFibers: dict,
    stats: DataFrame,
    xlim: tuple[int, int] = (-5, 5),
) -> Figure:
    """
    Generate 1D plots summarizing spectrograph data, including a Gaussian reference.

    Parameters
    ----------
    spectraFibers : `dict`
        Dictionary containing spectrograph data.
    stats : `DataFrame`
        DataFrame containing statistics for each arm.
    xlim : `tuple` of `int`, optional
        X-axis limits for the plots (default: (-5, 5)).

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The generated figure.
    """
    label_lookup = {"b": "Blue", "r": "Red", "n": "NIR", "m": "Medium"}

    # Generate spectrograph summary plots.
    fig, ax_dict = summarizeSpectrograph(spectraFibers, stats, xlim=xlim)

    # Generate Gaussian distribution.
    xp = np.linspace(-6, 6, 1000)
    yp = scipy.stats.norm.pdf(xp, loc=0, scale=1)

    # Update axis labels and add Gaussian reference.
    for specKey in spectraFibers.keys():
        spectrograph, arm = specKey
        ax = ax_dict[f"{arm}_HIST"]
        ax.set_ylabel(f"{label_lookup[arm]} arm")
        ax.plot(xp, yp, color="k", linewidth=4, linestyle="--")

    # Set title.
    ax_dict["b_HIST"].set_title("Chi Histogram")
    ax_dict["b_AVG"].set_title("Mean and Median Chi")
    ax_dict["b_ERR"].set_title("Stddev and IQR Chi")

    # Add legend.
    ax_dict["b_HIST"].plot([], [], color=detector_palette["b"], label="DRP")
    ax_dict["b_HIST"].plot([], [], color="magenta", label="Combined DRP")
    ax_dict["b_HIST"].plot([], [], color="k", label="Using Poisson errors")

    ax_dict["b_HIST"].legend(loc="upper left")

    return fig


def plot_2d_chi(
    spectraFibers: dict,
    plotCol: str = "chi",
    wave_lims: tuple[float, float] = None,
    vlims: tuple[float, float] = None,
    aggfunc: str | Callable = "mean",
) -> Figure:
    """
    Generate a 2D plot of chi values for sky subtraction residuals.

    This function visualizes the chi values across different fibers and wavelengths,
    providing insights into the performance of sky subtraction.

    Parameters
    ----------
    spectraFibers : `dict`
        Dictionary containing spectrograph data.
    plotCol : `str`, optional
        Column name to plot (default: 'chi').
    wave_lims : `tuple` of `float`, optional
        Wavelength limits for the plot (default: None, uses min and max).
    vlims : `tuple` of `float`, optional
        Color scale limits for the plot (default: None, (-3, 3)).
    aggfunc : `str` or `Callable`, optional
        Aggregation function for chi values (default: "mean").
        Can be a string or a callable function.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The generated figure.
    """
    fd0 = getFiberData(spectraFibers)
    fd0["wave_row"] = fd0.wave.astype("int")

    allArms = ["b", "r", "m", "n"]
    arms = fd0.arm.unique()
    availableArms = [arm for arm in allArms if arm in arms]

    dp = div_palette.copy()
    dp.set_bad(color="k", alpha=0.0)

    vmin = vlims[0] if vlims is not None else -3
    vmax = vlims[1] if vlims is not None else 3

    fig, ax_dict = get_mosaic("".join(availableArms), figsize=(10, 5), sharey=True)

    plotData = dict()
    for i, arm in enumerate(availableArms):
        ax = ax_dict[arm]
        rows = fd0.query(f"arm == '{arm}'")

        arm_fd0 = rows.pivot_table(
            index=["fiberId"], columns="wave_row", values=plotCol, observed=False, aggfunc=aggfunc
        )

        wave_min = wave_lims[0] if wave_lims is not None else arm_fd0.columns.min()
        wave_max = wave_lims[1] if wave_lims is not None else arm_fd0.columns.max()

        wave_range = np.arange(wave_min, wave_max + 1, dtype="int")
        arm_fd0 = arm_fd0.T.reindex(wave_range).T

        # Plot each arm separately
        plotData[arm] = arm_fd0
        ax.set_title(f"{arm} arm")

        cax = None
        cbar_kws = dict()
        cbar = False
        if i == len(availableArms) - 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            cbar_kws = dict(orientation="vertical", extend="both", ticklocation="left")
            cbar = True

        sb.heatmap(
            arm_fd0,
            cmap=dp,
            ax=ax,
            center=0,
            vmin=vmin,
            vmax=vmax,
            robust=True,
            cbar=cbar,
            cbar_ax=cax,
            cbar_kws=cbar_kws,
        )
        ax.set(xlabel="", ylabel="")
        ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)
        ax.spines["top"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["right"].set_visible(True)

        if i == 0:
            ax.set_ylabel("Fiber ID")
        if i == 1:
            ax.set_xlabel("Wavelength [nm]")

    fig.suptitle(f"{aggfunc.title()} {plotCol} values per nm")

    return fig


def plot_outlier_summary(spectras: dict, spectraFibers: dict, thresholds=None) -> Figure:
    """
    Generate a summary plot highlighting outliers in sky subtraction residuals.

    This function visualizes spectral regions where the absolute chi values exceed
    predefined thresholds (5 and 15) and provides a sky model reference plot.

    Notes
    -----
        - Uses `buildReference` to generate a median sky spectrum.
        - Highlights outlier chi values with thresholds at 5 and 15.
        - Uses `scatter` to visualize outliers in wavelength space.

    Parameters
    ----------
    spectras : `dict`
        Dictionary containing spectrograph data.
    spectraFibers : `dict`
        Dictionary containing fiber data.
    thresholds : `list` of `float`, optional
        List of thresholds for outlier detection (default: [3, 10]).

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The generated figure containing the outlier summary plot.
    """
    if thresholds is None:
        thresholds = [3, 10]

    threshold_low = thresholds[0]
    threshold_high = thresholds[1]

    df = getFiberData(spectraFibers)

    df["chi_value"] = df.chi.map(
        lambda x: (
            f"< {threshold_low}"
            if abs(x) < threshold_low
            else (
                f"> {threshold_high}"
                if abs(x) > threshold_high
                else f"{threshold_low} < x < {threshold_high}"
            )
        )
    )

    fig, ax = get_mosaic([["SKY"], ["CHI"]], sharex=True)
    fig.set_size_inches(10, 4)

    sb.scatterplot(
        data=df.query(f"abs(chi) >= {threshold_low}"),
        x="wave",
        y="fiberId",
        hue="chi",
        hue_norm=SymLogNorm(threshold_low),
        palette=div_palette,
        size="chi_value",
        size_order=[
            f"> {threshold_high}",
            f"{threshold_low} < x < {threshold_high}",
        ],
        ax=ax["CHI"],
        legend=False,
    )
    ax["CHI"].set_ylabel("Fiber ID")
    ax["CHI"].set_xlabel("Wavelength [nm]")
    ax["CHI"].set_title("Sky fibers chi outliers")
    ax["CHI"].grid(True, alpha=0.25)

    for skySpectra in spectras.values():
        arm = skySpectra.identity.arm
        sky_wavelength, sky_flux = buildReference(skySpectra, func=np.nanmedian, model="sky")
        ax["SKY"].plot(sky_wavelength, sky_flux, color=detector_palette[arm], label=arm)

    ax["SKY"].set_yscale("log")
    ax["SKY"].set_title("Median sky flux")
    ax["SKY"].grid(True, alpha=0.25)

    return fig


def plot_sky_reference(spectras: dict) -> Figure:
    """Plots the sky reference spectrum and flux residuals.

    Parameters
    ----------
    spectras : `dict`
        Dictionary containing spectrograph data.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Generated figure.
    """
    # Create a figure layout.
    fig = Figure(figsize=(10, 3), layout="constrained")
    ax = fig.add_subplot(111)

    # Loop through each spectral arm in a specific order.
    i = 0
    for arm in ["b", "r", "m", "n"]:
        for spectrograph in [1, 2, 3, 4]:
            spec_key = (spectrograph, arm)
            if spec_key not in spectras:
                continue

            i += 1

            skySpectra = spectras[spec_key]
            # Split into reference and test spectra.
            referenceSpectra, testSpectra = splitSpectraIntoReferenceAndTest(skySpectra)

            # Compute reference and test statistics.
            references_sky = buildReference(referenceSpectra, func=np.nanmedian, model="none")
            residual_flux = buildReference(testSpectra, func=np.median, model="residuals")

            arm_color = detector_palette[arm]

            # Interpolate sky brightness onto a residual wavelength grid.
            sky_wave_ref, sky_flux = references_sky
            resid_wave_ref, resid_flux = residual_flux

            sky_flux = np.interp(resid_wave_ref, sky_wave_ref, sky_flux)

            # Scatter plot of residual flux vs wavelength.
            ax.scatter(resid_wave_ref, resid_flux, s=3, color=arm_color, rasterized=True, alpha=0.9)
            # Plot 1% sky brightness reference.
            ax.plot(
                resid_wave_ref, sky_flux / 100, color="k", linewidth=1, alpha=0.3, label="1% sky", zorder=-100
            )

    # Set axis limits.
    ax.set_ylim(-100, 100)

    # Set axis labels.
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Median Sky Flux Counts")

    # Add reference lines.
    ax.axhline(0, linestyle="--", color="k")

    fig.suptitle("Residual flux and 1% sky spectra reference")

    return fig


def plot_vs_sky_brightness_all(spectraFibers, method="median", binsize=10) -> Figure:
    """Plots all data."""
    fd0 = getFiberData(spectraFibers)
    fd0["totalFlux"] = fd0.eval("flux + sky")
    fd0["wave_rank"] = fd0.groupby(["arm", "fiberId"], observed=False).wave.rank()

    fd0["ranked"] = fd0.groupby("fiberId", observed=False).totalFlux.rank(pct=True) * 100
    fd0["ranked_bin"] = pd.cut(
        fd0.ranked, binsize, labels=np.arange(start=0, stop=100 + binsize, step=(100 + binsize) // binsize)
    )

    fig, ax_dict = get_mosaic(
        [["CHI_0", "CHI_1", "CHI_2"], ["CHI_POISSON_0", "CHI_POISSON_1", "CHI_POISSON_2"]],
        figsize=(10, 6),
        sharex=True,
        sharey=True,
    )

    for row, row_name in enumerate(["chi", "chi_poisson"]):
        i = 0
        for arm in ["b", "r", "m", "n"]:
            for spectrograph in [1, 2, 3, 4]:
                fd1 = fd0.query(f"arm == '{arm}' and spectrograph == {spectrograph}", engine="python")
                if len(fd1) == 0:
                    continue

                ax_name = f"{row_name.upper()}_{i}"
                if i == 1:
                    ax_dict[ax_name].set_title(f"{row_name}")
                ax = ax_dict[ax_name]

                plotPercentile(fd1, ax, arm, method=method, column=row_name)
                ax.set_xlim(-3, 3)
                i += 1

    # Set title and labels.
    fig.suptitle("Fiber chis by sky brightness percentile")
    ax_dict["CHI_0"].set_ylabel("Sky Brightness Percentile")
    ax_dict["CHI_POISSON_0"].set_ylabel("Sky Brightness Percentile")
    ax_dict["CHI_POISSON_0"].set_xlabel(r"$\chi$")
    ax_dict["CHI_POISSON_1"].set_xlabel(r"$\chi$")
    ax_dict["CHI_POISSON_2"].set_xlabel(r"$\chi$")

    return fig


def plot_vs_sky_brightness(spectras: dict, method="median") -> Figure:
    """
    Generate plots comparing sky brightness with spectral residuals.

    This function visualizes the relationship between median residual flux
    and sky brightness percentile, as well as how residuals change with wavelength.

    Notes
    -----
    - Uses `splitSpectraIntoReferenceAndTest` to separate reference and test spectra.
    - Compares sky brightness and residuals across different wavelengths.
    - Uses `rolling` to compute binned statistics of residuals versus sky brightness percentile.

    Parameters
    ----------
    spectras : `dict`
        Dictionary containing spectrograph data.
    method : `str`, optional
        Method to compute the median residual flux (default: "median").
        Can be "median", "mean", or any other valid aggregation function.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Generated figure containing subplots.
    """
    # Create a figure layout.
    fig, ax_dict = get_mosaic(
        [["CHI_0", "CHI_1", "CHI_2"], ["CHI_POISSON_0", "CHI_POISSON_1", "CHI_POISSON_2"]],
        figsize=(10, 6),
        sharex=True,
        sharey=True,
    )

    # Loop through each spectral arm.
    for row, col in enumerate(["chi", "chi_poisson"]):
        i = 0
        for arm in ["b", "r", "m", "n"]:
            for spectrograph in [1, 2, 3, 4]:
                spec_key = (spectrograph, arm)
                if spec_key not in spectras:
                    continue

                ax_name = f"{col.upper()}_{i}"
                if i == 1:
                    ax_dict[ax_name].set_title(f"{col}")

                skySpectra = spectras[spec_key]
                rank0 = getSkyPercentile(skySpectra, column=col)
                plotPercentile(rank0, ax_dict[ax_name], skySpectra.identity.arm, method=method, column=col, rasterized=False)

                i += 1

    # Set title and labels.
    fig.suptitle("Median chis by sky brightness percentile")
    ax_dict["CHI_0"].set_ylabel("Sky Brightness Percentile")
    ax_dict["CHI_POISSON_0"].set_ylabel("Sky Brightness Percentile")
    ax_dict["CHI_POISSON_0"].set_xlabel(r"$\chi$")
    ax_dict["CHI_POISSON_1"].set_xlabel(r"$\chi$")
    ax_dict["CHI_POISSON_2"].set_xlabel(r"$\chi$")

    return fig


def getSkyPercentile(skySpectra: PfsArm, column: str = "chi", binsize: int = 10) -> DataFrame:
    """Compute the sky brightness percentile for a given spectral arm.

    This function processes the sky spectra to compute the percentile of sky brightness
    based on the specified column (e.g., 'chi' or 'chi_poisson'). It splits the spectra
    into reference and test sets, computes the median sky brightness, and ranks the
    sky brightness values into bins.

    Parameters
    ----------
    skySpectra : `pfs.datamodel.PfsArm`
        PFS spectra object containing spectral data for a specific arm.
    column : `str`, optional
        Column to compute the percentile for (default: 'chi').
        Can be 'chi' or 'chi_poisson'.
    binsize : `int`, optional
        Size of the bins for ranking (default: 10).


    Returns
    -------
    rank0 : `pandas.DataFrame`
        DataFrame containing the ranked sky brightness percentiles and corresponding chi values.
        Columns include 'ranked', the specified column (e.g., 'chi'), and 'ranked_bin'.
    """
    referenceSpectra, testSpectra = splitSpectraIntoReferenceAndTest(skySpectra)

    # Compute reference and test statistics.
    references_sky = buildReference(referenceSpectra, func=np.nanmedian, model="none")
    references_chi_median = buildReference(testSpectra, func=np.median, model=column)

    sky_wave_ref, sky_flux = references_sky
    chi_wave_ref, chi = references_chi_median

    # Interpolate sky brightness onto a chi wavelength grid.
    sky_flux = np.interp(chi_wave_ref, sky_wave_ref, sky_flux)

    # Compute ranked percentile of sky brightness.
    ranked = pd.Series(sky_flux).rank(pct=True).values * 100

    bright_bin = pd.cut(
        ranked, binsize, labels=np.arange(start=0, stop=100 + binsize, step=(100 + binsize) // binsize)
    )

    rank0 = pd.DataFrame({"ranked": ranked, column: chi, "ranked_bin": bright_bin})

    return rank0


def plotPercentile(
    data: DataFrame, ax: Axes, arm: str, column: str = "chi", method: str | Callable = "median", rasterized: bool = True
):
    """Plot percentile data against a specified column.

    This function generates a scatter plot of percentile data against a specified column
    (e.g., 'chi' or 'chi_poisson') for a given arm of the spectrograph. It plots the median values
    and robust RMS of the specified column against the ranked bins, providing a visual representation
    of the relationship between the percentile of sky brightness and the spectral residuals.

    Parameters
    ----------
    data : `pandas.DataFrame`
        DataFrame containing the ranked sky brightness percentiles and corresponding chi values.
        Must contain columns 'ranked', the specified column (e.g., 'chi'), and 'ranked_bin'.
    ax : `matplotlib.axes.Axes`
        Axes object to plot the data on.
    arm : `str`
        Spectrograph arm identifier (e.g., 'b', 'r', 'm', 'n'), used for color coding.
    column : `str`, optional
        Column to plot against the ranked bins (default: 'chi').
    method : `str` or `Callable`, optional
        Method to compute the median values (default: 'median').
        Can be a string (e.g., 'mean', 'median') or a callable function.
    rasterized : `bool`, optional
        Whether to rasterize the scatter plot for performance (default: True).
    """
    rank0_grp = data.groupby("ranked_bin", observed=False)[column].agg([method, robustRms]).reset_index()

    X = rank0_grp[method]
    Xerr = rank0_grp.robustRms
    Y = rank0_grp.ranked_bin

    plot_color = detector_palette[arm]

    ax.errorbar(x=X, xerr=Xerr, y=Y, marker="o", mfc=plot_color, c="k", linewidth=2, capsize=5, capthick=3)

    ax.scatter(data[column], data.ranked, c=plot_color, marker="o", alpha=0.25, zorder=-100, s=4, rasterized=rasterized)

    ax.axvline(0, c="k", ls="--")

    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel(r"$\chi$")
    ax.tick_params(axis="both", which="both", bottom=False, right=False, top=False, left=False)
    ax.grid()


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
            ew.append(robustRms(y[C]))  # Custom standard deviation function

        x0 += sep  # Move to the next bin

    return np.array(xw), np.array(yw), np.array(ew)


def buildReference(spectra: PfsArm, func: Callable | None = np.mean, model: str = "residuals"):
    """
    Build a reference spectrum by aggregating spectral data from multiple fibers.
    The reference spectrum is constructed by applying a specified aggregation function
    (e.g., mean, median) to the selected model across all fibers.

    Notes
    -----
    - The function extracts and aligns spectra from all fibers to a common wavelength grid.
    - Uses interpolation to map each fiber's spectrum onto the reference wavelength array.
    - Applies the selected aggregation function (`func`) to compute the final reference spectrum.

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
    """
    # Containers for spectral data.
    x, y = [], []

    # Process each fiber.
    for fiberId in spectra.fiberId:
        wave, flux, std, sky, chi, stdPoisson, chi_poisson, C = extractFiberInfo(
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
            y.append(chi_poisson)
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

    # Apply the aggregation function to compute the final reference spectrum.
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


def extractFiberInfo(spectra: PfsArm, fiberId: int, finite: bool = True):
    """
    Extract relevant spectral data for a specific fiber.

    This function retrieves wavelength, flux, sky background, variance, and
    computes standard deviation and chi values for the given fiber.

    Notes
    -----
    - The function applies filtering to ensure valid data: sky > 0 and variance > 0.
    - If `finite` is True, only valid data points are returned.

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
class PlotLayer:
    version: str = "scatter"
    X: Number | List[Number] | None = None
    W: List[Number] | None = None
    rnge: tuple[Number, Number] | None = None
    label: str | None = None
    color: str = "k"
    alpha: float = 1.0
    linestyle: str = "-"
    linewidth: int = 3
    bins: str | int = "auto"
    histtype: str = "step"


def make_plot(
    layers: Iterable[PlotLayer],
    ax: Axes | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    fontsize: int = 12,
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
        layers (Iterable[PlotLayer]): A collection of plot layers to be added to the
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
        fig = Figure(layout="constrained", figsize=figsize)
        ax = fig.add_subplot(111)

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

    with contextlib.suppress(AttributeError):
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


def get_mosaic(mosaic="A", figsize=(10, 10), **kwargs):
    """Create a figure with a specified layout using matplotlib's subplot_mosaic."""
    fig = Figure(layout="constrained", figsize=figsize)
    ax_dict = fig.subplot_mosaic(mosaic, **kwargs)
    return fig, ax_dict


def add_layer(ax: Axes, layer: PlotLayer):
    """Adds a layer to the plot based on the specified version.

    The bulk of the plotting options and work is done in this function.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The axes to which the layer will be added.
    layer : `PlotLayer`
        The layer object containing the data and properties for the plot.
    """
    if layer.version == "hist":
        ax.hist(
            layer.X,
            bins=30,
            density=True,
            color=layer.color,
            range=layer.rnge,
            label=layer.label,
            linestyle=layer.linestyle,
            linewidth=layer.linewidth,
            histtype=layer.histtype,
            alpha=layer.alpha,
        )
    elif layer.version == "vert":
        ax.axvline(
            layer.X,
            linestyle=layer.linestyle,
            color=layer.color,
            linewidth=layer.linewidth,
            label=layer.label,
            alpha=layer.alpha,
            zorder=10,
        )
