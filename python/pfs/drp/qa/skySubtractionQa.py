from lsst.afw.image import VisitInfo
from lsst.pex.config import Field
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
from pfs.drp.stella import PfsArm
from pfs.drp.stella.fitFocalPlane import FitBlockedOversampledSplineConfig, FitBlockedOversampledSplineTask
from pfs.drp.stella.selectFibers import SelectFibersTask
from pfs.drp.stella.subtractSky1d import subtractSky1d

from pfs.drp.qa.skySubtraction.summaryPlots import (
    plot_1d_spectrograph,
    plot_2d_spectrograph,
    plot_outlier_summary,
    plot_vs_sky_brightness,
)


class SkySubtractionConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Connections for SkySubtractionTask"""

    pfsArm = InputConnection(
        name="pfsArm",
        doc="PfsArm data",
        storageClass="Spectra",
        dimensions=("instrument", "visit", "spectrograph"),
        multiple=True,
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
        dimensions=("instrument",),
    )

    skySubtraction1DPlot = OutputConnection(
        name="skySubtraction1DPlot",
        doc="Sky Subtraction 1D Plot",
        storageClass="Plot",
        dimensions=("instrument", "visit", "spectrograph"),
    )

    skySubtraction2DPlot = OutputConnection(
        name="skySubtraction2DPlot",
        doc="Sky Subtraction 2D Plot",
        storageClass="Plot",
        dimensions=("instrument", "visit", "spectrograph"),
    )

    skySubtractionOutlierPlot = OutputConnection(
        name="skySubtractionOutlierPlot",
        doc="Sky Subtraction Outlier Plot",
        storageClass="Plot",
        dimensions=("instrument", "visit", "spectrograph"),
    )

    skySubtractionSkyBrightnessPlot = OutputConnection(
        name="skySubtractionSkyBrightnessPlot",
        doc="Sky Subtraction Sky Brightness Plot",
        storageClass="Plot",
        dimensions=("instrument", "visit", "spectrograph"),
    )


class SkySubtractionConfig(PipelineTaskConfig, pipelineConnections=SkySubtractionConnections):
    """Configuration for SkySubtractionTask"""

    blockSize = Field(dtype=int, default=20, doc="Block size for sky model fitting.")
    rejIterations = Field(dtype=int, default=5, doc="Number of rejection iterations.")
    rejThreshold = Field(dtype=float, default=3.0, doc="Rejection threshold.")
    mask = Field(dtype=list, default=["NO_DATA"], doc="Mask types.")
    oversample = Field(dtype=int, default=2, doc="Oversampling factor.")


class SkySubtractionTask(PipelineTask):
    """Task for QA of skySubtraction"""

    ConfigClass = SkySubtractionConfig
    _DefaultName = "skySubtraction"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        mergeArms_config = inputRefs.mergeArms_config
        # Get default sky model configuration
        fitSkyModelConfig = FitBlockedOversampledSplineConfig()
        defaultConfig = mergeArms_config.fitSkyModel.toDict()

        # Update only if user provides values
        # defaultConfig["blockSize"] = self.config.blockSize
        # defaultConfig["rejIterations"] = self.config.rejIterations
        # defaultConfig["rejThreshold"] = self.config.rejThreshold
        # defaultConfig["mask"] = ["NO_DATA", "BAD_FLAT", "BAD_FIBERNORMS", "SUSPECT"]
        # defaultConfig["oversample"] = self.config.oversample

        fitSkyModelConfig.update(**defaultConfig)

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

    def run(self, pfsArm, pfsConfig, fitSkyModelConfig, **kwargs) -> Struct:
        """Perform QA on sky subtraction.

        Parameters
        ----------
        pfsArm : `pfs.drp.stella.ArcLineSet`
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
        # Select sky fibers.
        selectSky = SelectFibersTask()
        selectSky.config.targetType = ("SKY",)  # Selecting only sky fibers
        skyConfig = selectSky.run(pfsConfig.select(fiberId=pfsArm.fiberId))

        # Remove the excluded fiber from the sky selection
        spectras = list()
        for excludeFiberId in skyConfig.fiberId:
            skyConfig0 = skyConfig[skyConfig.fiberId != excludeFiberId]
            skySpectra = pfsArm.select(pfsConfig, fiberId=skyConfig0.fiberId)

            if len(skySpectra) == 0:
                raise ValueError("No sky spectra to use for sky subtraction.")

            # Fit sky model using the given configuration
            fitSkyModel = FitBlockedOversampledSplineTask(config=fitSkyModelConfig)
            sky1d = fitSkyModel.run(skySpectra, skyConfig0)

            # Apply sky subtraction to the full spectra.
            subtractSky1d(pfsArm, pfsConfig, sky1d)

            spectras.append(pfsArm[pfsArm.fiberId == excludeFiberId])

        mergedSpectra = PfsArm.fromMerge(spectras)

        hold = None
        holdAsDict = None
        plotId = None
        arms = kwargs.get("arms", ["b", "r", "n"])

        fig_1d, _ = plot_1d_spectrograph(holdAsDict, plotId, arms)
        fig_2d, _ = plot_2d_spectrograph(hold, plotId, arms)
        fig_outlier, ax_dicts = plot_outlier_summary(hold, holdAsDict, plotId, arms)
        fig_sky_brightness, _ = plot_vs_sky_brightness(hold, plotId, arms)

        return Struct(
            skySubtraction1DPlot=fig_1d,
            skySubtraction2DPlot=fig_2d,
            skySubtractionOutlierPlot=fig_outlier,
            skySubtractionSkyBrightnessPlot=fig_sky_brightness,
        )
