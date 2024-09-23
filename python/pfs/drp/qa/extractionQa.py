import lsstDebug
from lsst.afw.image import ExposureF
from lsst.daf.persistence import ButlerDataRef
from lsst.pex.config import ConfigurableField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base.connectionTypes import (
    Input as InputConnection,
    Output as OutputConnection,
    PrerequisiteInput as PrerequisiteConnection,
)
from pfs.datamodel import PfsConfig
from pfs.drp.stella import DetectorMap, FiberProfileSet, PfsArm

from pfs.drp.qa.tasks.fiberExtraction import ExtractionQaTask


class ExtractionQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "exposure", "detector"),
):
    """Connections for ExtractionQaTask"""

    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
    )
    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Position and shape of fibers",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    detectorMap = InputConnection(
        name="detectorMap_used",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "exposure", "detector"),
    )
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra from arm",
        storageClass="PfsArm",
        dimensions=("instrument", "exposure", "detector"),
    )
    calexp = InputConnection(
        name="calexp",
        doc="Calibrated exposure, optionally sky-subtracted",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    extQaStats = OutputConnection(
        name="extQaStats",
        doc="Summary plots. Results of the residual analysis of extraction are plotted.",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument", "exposure", "detector"),
    )
    extQaImage = OutputConnection(
        name="extQaImage",
        doc="Detail plots. Calexp, residual, and chi images and the comparison of the calexp"
        "profile and fiberProfiles are plotted for some fibers with bad extraction quality.",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument", "exposure", "detector"),
    )
    extQaImage_pickle = OutputConnection(
        name="extQaImage_pickle",
        doc="Statistics of the residual analysis.",
        storageClass="QaDict",
        dimensions=("instrument", "exposure", "detector"),
    )


class ExtractionQaConfig(PipelineTaskConfig, pipelineConnections=ExtractionQaConnections):
    """Configuration for ExtractionQaTask"""

    fiberExtractionQa = ConfigurableField(target=ExtractionQaTask, doc="Plot the fiber extraction QA.")


class ExtractionQaTask(PipelineTask):
    """Task for QA of extraction"""

    ConfigClass = ExtractionQaConfig
    _DefaultName = "extractionQa"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def runDataRef(self, dataRef: ButlerDataRef) -> None:
        """Calls ``self.run()``

        Parameters
        ----------
        dataRef : `ButlerDataRef`
            Data reference for merged spectrum.
        """
        calexp = dataRef.get("calexp")
        fiberProfiles = dataRef.get("fiberProfiles")
        detectorMap = dataRef.get("detectorMap_used")
        pfsArm = dataRef.get("pfsArm")
        pfsConfig = dataRef.get("pfsConfig")

        outputs = self.run(calexp, fiberProfiles, detectorMap, pfsArm, pfsConfig)
        for datasetType, data in outputs.getDict().items():
            if data is not None:
                dataRef.put(data, datasetType=datasetType)

    def run(
        self,
        calexp: ExposureF,
        fiberProfiles: FiberProfileSet,
        detectorMap: DetectorMap,
        pfsArm: PfsArm,
        pfsConfig: PfsConfig,
    ) -> Struct:
        """QA of extraction by analyzing the residual image.

        Parameters
        ----------
        calexp : `ExposureF`
            Exposure data
        fiberProfiles : `FiberProfileSet`
            Profiles of each fiber.
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y
        pfsArm : `PfsArm`
            Extracted spectra from arm.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers.

        Returns
        -------
        extQaStats : `MultipagePdfFigure`
            Summary plots.
            Results of the residual analysis of extraction are plotted.
        extQaImage : `MultipagePdfFigure`
            Detail plots.
            Calexp, residual, and chi images and the comparison of the calexp
            profile and fiberProfiles are plotted for some fibers with bad
            extraction quality.
        extQaImage_pickle : `QaDict`
            Data to be pickled.
            Statistics of the residual analysis.
        """
        return self.fiberExtractionQa.run(calexp, fiberProfiles, detectorMap, pfsArm, pfsConfig)
