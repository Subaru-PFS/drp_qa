from typing import Iterable

import pandas as pd
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
)
from pandas import DataFrame


class DetectorMapCombinedQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "arm", "spectrograph"),
):
    """Connections for DetectorMapCombinedQaTask"""

    dmQaResidualStats = InputConnection(
        name="dmQaResidualStats",
        doc="DM QA residual statistics",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
        multiple=True,
    )

    dmQaCombinedResidualPlot = OutputConnection(
        name="dmQaCombinedResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for the entire detector.",
        storageClass="Plot",
        dimensions=(
            "instrument",
            "arm",
            "spectrograph",
        ),
    )

    dmQaDetectorStats = OutputConnection(
        name="dmQaDetectorStats",
        doc="Statistics of the residual analysis for the entire detector.",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "arm",
            "spectrograph",
        ),
    )


class DetectorMapCombinedQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapCombinedQaConnections):
    """Configuration for DetectorMapCombinedQaTask"""

    useSigmaRange = Field(dtype=bool, default=False, doc="Use ±2.5 sigma as range")


class DetectorMapCombinedQaTask(PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapCombinedQaConfig
    _DefaultName = "dmCombinedResiduals"

    def run(self, dmQaResidualStats: Iterable[DataFrame]) -> Struct:
        """Create detector level stats and plots.

        Parameters
        ----------
        dmQaResidualStats : Iterable[DataFrame]
            A an iterable of DataFrames containing DM QA residual statistics. These
            are combined into a single DataFrame for processing.

        Returns
        -------
        dmQaCombinedResidualPlot : `MultipagePdfFigure`
            1D and 2D plots of the residual between the detectormap and the arclines for the entire detector.
        dmQaDetectorStats : `pd.DataFrame`
            Statistics of the residual analysis.
        """
        stats = pd.concat(dmQaResidualStats)
        print(f"DM QA stats: {len(stats)}")
