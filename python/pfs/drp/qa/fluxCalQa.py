from typing import Dict, Tuple

import lsstDebug
from lsst.daf.persistence import ButlerDataRef, NoResults
from lsst.pex.config import Field
from lsst.pipe.base import (
    CmdLineTask,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    TaskRunner,
)
from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
from lsst.pipe.base.connectionTypes import (
    Output as OutputConnection,
    PrerequisiteInput as PrerequisiteConnection,
)
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from pfs.datamodel import PfsConfig, PfsSingle, TargetType


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

    filterSet = Field(dtype=str, default="ps1", doc="Filter set to use, e.g. 'ps1'")
    includeFakeJ = Field(dtype=bool, default=False, doc="Include the fake narrow J filter")
    diffFilter = Field(dtype=str, default="g_ps1", doc="Filter to use for the color magnitude difference")


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


class FluxCalQaTask(CmdLineTask, PipelineTask):
    """Task for generating fluxCalibration QA plots."""

    ConfigClass = FluxCalQaConfig
    _DefaultName = "fluxCalQa"
    RunnerClass = FluxCalQaRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def runQuantum(
        self,
        butler: ButlerQuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with butler I/O

        Parameters
        ----------
        butler : `ButlerQuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        inputs = butler.get(inputRefs)
        outputs = self.run(**inputs)
        butler.put(outputs, outputRefs)

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
        return self.fluxCalQa(pfsConfig, pfsSingles)

    def _getMetadataName(self):
        """Get the name of the metadata dataset type, or `None` if metadata is
        not to be persisted.

        Notes
        -----
        The name may depend on the config; that is why this is not a class
        method.
        """
        return None
