from typing import Dict, Iterable

import lsstDebug
from lsst.pex.config import Field
from lsst.pipe.base import (
    CmdLineTask,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base.connectionTypes import Input as InputConnection, Output as OutputConnection
from pfs.drp.stella import ArcLineSet, DetectorMap


class DetectorMapQaConnections(
    PipelineTaskConnections,
    dimensions=("exposure", "detector", "physical_filter"),
):
    """Connections for DetectorMapQaTask"""

    detectorMap = InputConnection(
        name="detectorMap_used",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("exposure", "detector"),
        multiple=True,
    )
    arcLines = InputConnection(
        name="arcLines",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("exposure", "detector"),
        multiple=True,
    )
    dmQaResidualPlot = OutputConnection(
        name="dmQaResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for a given visit.",
        storageClass="MultipagePdfFigure",
        dimensions=("exposure", "detector"),
        multiple=True,
    )
    dmQaCombinedResidualPlot = OutputConnection(
        name="dmQaCombinedResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for the entire detector.",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument",),
        multiple=True,
    )
    dmQaResidualStats = OutputConnection(
        name="dmQaResidualStats",
        doc="Statistics of the residual analysis for the visit.",
        storageClass="pandas.core.frame.DataFrame",
        dimensions=("exposure", "detector"),
        multiple=True,
    )
    dmQaDetectorStats = OutputConnection(
        name="dmQaDetectorStats",
        doc="Statistics of the residual analysis for the entire detector.",
        storageClass="pandas.core.frame.DataFrame",
        dimensions=("detector",),
        multiple=True,
    )


class DetectorMapQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapQaConnections):
    """Configuration for DetectorMapQaTask"""

    saveOutput = Field(doc="Save the results via butler", dtype=bool, default=True)


class DetectorMapQaTask(CmdLineTask, PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapQaConfig
    _DefaultName = "detectorMapQa"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("plotResidual")
        self.debugInfo = lsstDebug.Info(__name__)

    def run(
        self,
        groupName: str,
        arclineSet: Iterable[ArcLineSet],
        detectorMaps: Iterable[DetectorMap],
        dataIds: Iterable[Dict],
    ) -> Struct:
        """Generate detectorMapQa plots.

        Parameters
        ----------
        groupName : `str`
            Group name, either the visit or the detector.
        arclineSet : iterable of `ArcLineSet`
            Emission line measurements by adjustDetectorMap.
        detectorMaps : iterable of `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        dataIds : iterable of `dict`
            List of dataIds.

        Returns
        -------
        Struct
            Output data products. See `DetectorMapQaConnections`.
        """
        # List all the objects we have received.
        self.log.info(f"Processing {len(arclineSet)} ArcLineSets and {len(detectorMaps)} DetectorMaps")

    def _getMetadataName(self):
        return None
