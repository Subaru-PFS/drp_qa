from typing import Dict, Iterable

from lsst.pex.config import ConfigurableField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, QuantumContext, Struct
from lsst.pipe.base.connectionTypes import Input as InputConnection, Output as OutputConnection
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from pfs.drp.stella import ArcLineSet, DetectorMap

from pfs.drp.qa.tasks.detectorMapResiduals import PlotResidualTask


class DetectorMapQaConnections(
    PipelineTaskConnections,
    dimensions=(
        "instrument",
        "exposure",
        "arm",
        "spectrograph",
    ),
):
    """Connections for DetectorMapQaTask"""

    detectorMap = InputConnection(
        name="detectorMap_calib",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=(
            "instrument",
            "arm",
            "spectrograph",
        ),
        isCalibration=True,
    )
    arcLines = InputConnection(
        name="lines",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
    )
    dmQaResidualPlot = OutputConnection(
        name="dmQaResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for a given visit.",
        storageClass="MultipagePdfFigure",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
    )
    dmQaResidualStats = OutputConnection(
        name="dmQaResidualStats",
        doc="Statistics of the residual analysis for the visit.",
        storageClass="pandas.core.frame.DataFrame",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
    )
    # dmQaCombinedResidualPlot = OutputConnection(
    #     name="dmQaCombinedResidualPlot",
    #     doc="The 1D and 2D residual plots of the detectormap with the arclines for the entire detector.",
    #     storageClass="MultipagePdfFigure",
    #     dimensions=(
    #         "instrument",
    #         "arm",
    #         "spectrograph",
    #     ),
    # )

    # dmQaDetectorStats = OutputConnection(
    #     name="dmQaDetectorStats",
    #     doc="Statistics of the residual analysis for the entire detector.",
    #     storageClass="pandas.core.frame.DataFrame",
    #     dimensions=(
    #         "instrument",
    #         "arm",
    #         "spectrograph",
    #     ),
    # )


class DetectorMapQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapQaConnections):
    """Configuration for DetectorMapQaTask"""

    plotResidual = ConfigurableField(
        target=PlotResidualTask,
        doc="Plot the residual of the detectormap with the arclines.",
    )


class DetectorMapQaTask(PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapQaConfig
    _DefaultName = "detectorMapQa"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("plotResidual")

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        inputs = butler.get(inputRefs)
        dataId = inputRefs.exposure.dataId

        # Run the task
        print(f"Running {self.name} for {dataId}")
        outputs = self.run(**inputs, dataId=dataId)
        butler.put(outputs, outputRefs)
        return outputs

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
        # self.plotResidual.run(arclineSet, detectorMaps, dataIds)
