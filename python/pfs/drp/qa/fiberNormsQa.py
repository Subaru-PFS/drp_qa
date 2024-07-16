import lsstDebug
from lsst.daf.persistence import ButlerDataRef
from lsst.pex.config import Field
from lsst.pipe.base import (
    CmdLineTask,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    TaskRunner,
)
from lsst.pipe.base.connectionTypes import Input as InputConnection
from pfs.drp.stella import PfsConfig, PfsFiberNorms


class FiberNormQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "arm"),
):
    """Connections for fiberNormQaTask"""

    fiberNorms = InputConnection(
        name="fiberNorms_meas",
        doc="Measured fiber normalisations",
        storageClass="PfsFiberNorms",
        dimensions=("instrument", "arm"),
    )
    pfsConfig = InputConnection(
        name="pfsConfig",
        doc="Top-end configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
    )


class FiberNormsQaRunner(TaskRunner):
    """Runner for fiberNormQaTask"""

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """We operate on sets of arms."""
        groups = dict()
        for ref in parsedCmd.id.refList:
            arm = ref.dataId["arm"]
            groups[arm] = ref

        return [(groups[arm], kwargs) for arm in groups.keys()]


class FiberNormQaConfig(PipelineTaskConfig, pipelineConnections=FiberNormQaConnections):
    """Configuration for fiberNormQaTask"""

    plotLower = Field(dtype=float, default=2.5, doc="Lower bound for plot (standard deviations from median)")
    plotUpper = Field(dtype=float, default=2.5, doc="Upper bound for plot (standard deviations from median)")


class FiberNormQaTask(CmdLineTask, PipelineTask):
    """Task for QA of fiberNorms"""

    ConfigClass = FiberNormQaConfig
    _DefaultName = "fiberNormsQa"
    RunnerClass = FiberNormsQaRunner

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
        self.log.info(f"Running fiberNormsQaTask on {dataRef.dataId}")
        fiberNorms = dataRef.get("fiberNorms_meas")
        pfsConfig = dataRef.get("pfsConfig")

        outputs = self.run(fiberNorms, pfsConfig)
        for datasetType, data in outputs.getDict().items():
            if data is not None:
                dataRef.put(data, datasetType=datasetType)

    def run(self, fiberNorms: PfsFiberNorms, pfsConfig: PfsConfig) -> Struct:
        """QA of fiberNorms.

        Parameters
        ----------
        fiberNorms : `PfsFiberNorms`
            Measured fiber normalisations.
        pfsConfig : `PfsConfig`
            Top-end configuration.

        Returns
        -------
        outputs : `Struct`
            QA outputs.
        """
        self.log.info(f"Plotting fiber norms QA for {fiberNorms.identity}")

        fig, axes = fiberNorms.plot(pfsConfig, lower=self.config.plotLower, upper=self.config.plotUpper)
        visitList = fiberNorms.identity.visit0
        arm = fiberNorms.identity.arm
        axes.set_title(f"Fiber normalization for {arm=}\nvisits: {visitList}")

        return Struct(fiberNorms_plot=fig)

    def _getMetadataName(self):
        """Get the name of the metadata dataset type, or `None` if metadata is
        not to be persisted.

        Notes
        -----
        The name may depend on the config; that is why this is not a class
        method.
        """
        return None
