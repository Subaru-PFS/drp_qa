from lsst.afw.image import VisitInfo
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
from pfs.drp.stella import PfsArm
from pfs.drp.stella.fitFocalPlane import FitBlockedOversampledSplineConfig, FitBlockedOversampledSplineTask
from pfs.drp.stella.selectFibers import SelectFibersTask
from pfs.drp.stella.subtractSky1d import subtractSky1d

from pfs.drp.qa.skySubtraction.skySubtractionQa import convertToDict
from pfs.drp.qa.skySubtraction.summaryPlots import (
    plot_1d_spectrograph,
    plot_2d_spectrograph,
    plot_outlier_summary,
    plot_vs_sky_brightness,
)
from pfs.drp.qa.storageClasses import MultipagePdfFigure


class SkySubtractionConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "spectrograph"),
):
    """Connections for SkySubtractionTask"""

    pfsArms = InputConnection(
        name="pfsArm",
        doc="PfsArm data",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
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
        dimensions=(),
    )

    skySubtractionQaPlot = OutputConnection(
        name="skySubtractionQaPlot",
        doc="Sky Subtraction Plots: 1d, 2d, outliers, and vs sky brightness",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument", "visit", "spectrograph"),
    )


class SkySubtractionConfig(PipelineTaskConfig, pipelineConnections=SkySubtractionConnections):
    """Configuration for SkySubtractionTask"""

    blockSize = Field(dtype=int, default=None, optional=True, doc="Block size for sky model fitting.")
    rejIterations = Field(dtype=int, default=None, optional=True, doc="Number of rejection iterations.")
    rejThreshold = Field(dtype=float, default=None, optional=True, doc="Rejection threshold.")
    oversample = Field(dtype=float, default=None, optional=True, doc="Oversampling factor.")
    mask = ListField(dtype=str, default=None, optional=True, doc="Mask types.")


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

    def run(self, pfsArms, pfsConfig, fitSkyModelConfig, **kwargs) -> Struct:
        """Perform QA on sky subtraction.

        Parameters
        ----------
        pfsArms : `pfs.drp.stella.ArcLineSet`
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
        hold = dict()
        arms = list()
        for pfsArm in pfsArms:
            spectrograph = pfsArm.identity.spectrograph
            visit = pfsArm.identity.visit
            arm = pfsArm.identity.arm
            arms.append(arm)

            self.log.info(f"Starting sky subtraction qa for v{visit}{arm}{spectrograph}")

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

            hold[(spectrograph, arm)] = PfsArm.fromMerge(spectras)

        holdAsDict = convertToDict(hold)
        plotId = dict(visit=visit, arm=arm, spectrograph=spectrograph, block=self.config.blockSize)
        arms = list(set(arms))

        self.log.info(f"Plotting 1D spectra for arms {arms}.")
        fig_1d, _ = plot_1d_spectrograph(holdAsDict, plotId, arms)

        self.log.info(f"Plotting 2D spectra for arms {arms}.")
        fig_2d, _ = plot_2d_spectrograph(hold, plotId, arms)

        self.log.info(f"Plotting outlier summary for arms {arms}.")
        fig_outlier, ax_dicts = plot_outlier_summary(hold, holdAsDict, plotId, arms)

        self.log.info(f"Plotting vs sky brightness for arms {arms}.")
        fig_sky_brightness, _ = plot_vs_sky_brightness(hold, plotId, arms)

        pdf = MultipagePdfFigure()
        pdf.append(fig_1d)
        pdf.append(fig_2d)
        for fig in fig_outlier:
            pdf.append(fig)
        pdf.append(fig_sky_brightness)

        return Struct(skySubtractionQaPlot=pdf)
