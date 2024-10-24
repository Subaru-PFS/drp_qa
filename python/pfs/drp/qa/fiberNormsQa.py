from lsst.pex.config import Field
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
from pfs.drp.stella import PfsConfig, PfsFiberNorms


class FiberNormQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm"),
):
    """Connections for fiberNormQaTask"""

    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
    )
    fiberNorms = InputConnection(
        name="fiberNorms",
        doc="Measured fiber normalisations",
        storageClass="PfsFiberNorms",
        dimensions=("instrument", "visit", "arm"),
    )
    fiberNormsPlot = OutputConnection(
        name="fiberNormsPlot",
        doc="FiberNorms QA plot",
        storageClass="Plot",
        dimensions=("instrument", "visit", "arm"),
    )


class FiberNormQaConfig(PipelineTaskConfig, pipelineConnections=FiberNormQaConnections):
    """Configuration for fiberNormQaTask"""

    plotLower = Field(dtype=float, default=2.5, doc="Lower bound for plot (standard deviations from median)")
    plotUpper = Field(dtype=float, default=2.5, doc="Upper bound for plot (standard deviations from median)")


class FiberNormQaTask(PipelineTask):
    """Task for QA of fiberNorms"""

    ConfigClass = FiberNormQaConfig
    _DefaultName = "fiberNormsQa"

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

        upper_bounds = self.config.plotUpper
        lower_bounds = self.config.plotLower
        fig, axes = fiberNorms.plot(pfsConfig, lower=lower_bounds, upper=upper_bounds)
        visitList = fiberNorms.identity.visit0
        arm = fiberNorms.identity.arm
        axes.set_title(f"Fiber normalization for {arm=}\nvisits: {visitList}")

        return Struct(fiberNormsPlot=fig)
