from typing import Dict, Tuple

import lsstDebug
from lsst.daf.persistence import ButlerDataRef, NoResults
from lsst.pex.config import ConfigurableField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    TaskRunner,
)
from lsst.pipe.base.connectionTypes import (
    Output as OutputConnection,
    PrerequisiteInput as PrerequisiteConnection,
)
from pfs.datamodel import PfsConfig, PfsSingle, TargetType

from pfs.drp.qa.tasks.fluxCalibration import FluxCalibrationTask


class FluxCalQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "exposure", "detector"),
):
    """Connections for fluxCalQaTask"""

    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
    )
    pfsSingle = PrerequisiteConnection(
        name="pfsSingle",
        doc="Flux-calibrated, single epoch spectrum",
        storageClass="PfsSingle",
        dimensions=("instrument", "exposure"),
    )
    fluxCalStats = OutputConnection(
        name="fluxCalStats",
        doc="Statistics of the flux calibration analysis.",
        storageClass="pandas.core.frame.DataFrame",
        dimensions=("instrument", "exposure", "detector"),
    )
    fluxCalMagDiffPlot = OutputConnection(
        name="fluxCalMagDiffPlot",
        doc="Plot of the flux calibration magnitude difference.",
        storageClass="matplotlib.figure.Figure",
        dimensions=("instrument", "exposure", "detector"),
    )


class FluxCalQaConfig(PipelineTaskConfig, pipelineConnections=FluxCalQaConnections):
    """Configuration for fluxCalQaTask"""

    fluxCal = ConfigurableField(target=FluxCalibrationTask, doc="Flux Calibration Task.")


class FluxCalQaRunner(TaskRunner):
    """Runner for FluxCalQaTask"""

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Produce list of targets for DetectorMapQaTask.

        The visits and detector are processed as group, one per visit.
        """
        groups = dict()
        for ref in parsedCmd.id.refList:
            visit = ref.dataId["visit"]
            groups[visit] = ref

        processGroups = [((key, group), kwargs) for key, group in groups.items()]

        return processGroups


class FluxCalQaTask(PipelineTask):
    """Task for generating fluxCalibration QA plots."""

    ConfigClass = FluxCalQaConfig
    _DefaultName = "fluxCalQa"
    RunnerClass = FluxCalQaRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fluxCal")
        self.debugInfo = lsstDebug.Info(__name__)

    def runDataRef(self, dataRefList: Tuple[str, ButlerDataRef]) -> Struct:
        """Calls ``self.run()``

        Parameters
        ----------
        dataRefList : tuple
            A tuple of the visit and the butler data reference.
        """
        visit = dataRefList[0]
        dataRef = dataRefList[1]

        # Get the pfsConfig and pfsSingles for the FLUXSTD targets.
        pfsConfig = dataRef.get("pfsConfig")

        pfsSingles = dict()
        for fiberId in pfsConfig.select(targetType=TargetType.FLUXSTD).fiberId:
            try:
                pfsSingles[fiberId] = dataRef.get("pfsSingle", **pfsConfig.getIdentity(fiberId))
            except NoResults:
                self.log.warn(f"No PfsSingle found for fiberId {fiberId}")
                continue

        if not pfsSingles:
            self.log.warn(f"No FLUXSTD targets found for visit {visit}")
            return Struct()

        outputs = self.run(pfsConfig, pfsSingles)
        for datasetType, data in outputs.getDict().items():
            self.log.info(f"Writing {datasetType} for visit {visit}")
            dataRef.put(data, datasetType)

        return outputs

    def run(self, pfsConfig: PfsConfig, pfsSingles: Dict[str, PfsSingle]) -> Struct:
        """QA plots for flux calibration.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration.
        pfsSingles : dict of `pfs.datamodel.PfsSingle`
            Flux-calibrated, single epoch spectra.

        Returns
        -------
        outputs : `Struct`
            QA outputs.
        """
        return self.fluxCal.run(pfsConfig, pfsSingles)

    def _getMetadataName(self):
        return None
