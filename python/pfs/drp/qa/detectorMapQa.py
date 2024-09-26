from typing import Iterable

from lsst.pex.config import ConfigurableField
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)
from lsst.pipe.base.connectionTypes import Input as InputConnection, Output as OutputConnection
from pfs.drp.stella import ArcLineSet, DetectorMap

from pfs.drp.qa.tasks.detectorMapResiduals import PlotResidualTask


class DetectorMapQaConnections(
    PipelineTaskConnections,
    dimensions=(
        "instrument",
        "arm",
        "spectrograph",
    ),
):
    """Connections for DetectorMapQaTask"""

    def __init__(self, *, config=None):
        print(self.detectorMaps.dimensions)
        if config.plotResidual.combineVisits:
            print("Combining all exposures into one.")
            self.detectorMaps.dimensions = ("instrument", "arm", "spectrograph")

    detectorMaps = InputConnection(
        name="detectorMap_calib",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
        isCalibration=True,
        multiple=True,
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
        multiple=True,
    )
    dmQaResidualStats = OutputConnection(
        name="dmQaResidualStats",
        doc="Statistics of the DM residual analysis.",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
        multiple=True,
    )

    dmQaResidualPlot = OutputConnection(
        name="dmQaResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for a given visit.",
        storageClass="Plot",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
        multiple=True,
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
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        # If we are combining all the dataIds into one.
        data_ids = []
        for ref in inputRefs.arcLines:
            if "exposure" in ref.dataId.full:
                data_id = {k: v for k, v in ref.dataId.full.items()}
                data_id["visit"] = data_id["exposure"]
                data_id["run"] = ref.run
                data_ids.append(data_id)

        inputs = butlerQC.get(inputRefs)
        # There should only be one detectorMap input ref, so get it's detector name.
        inputs["detectorName"] = "{arm}{spectrograph}".format(**inputRefs.detectorMaps[0].dataId)
        inputs["dataIds"] = data_ids

        # Perform the actual processing.
        outputs = self.run(**inputs)

        # Store the results.
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        detectorName: str,
        arcLines: Iterable[ArcLineSet],
        detectorMaps: Iterable[DetectorMap],
        dataIds: Iterable[dict],
    ) -> Struct:
        """Generate detectorMapQa plots.

        Parameters
        ----------
        detectorName : str
            Name of the detector.
        arcLines : iterable of `ArcLineSet`
            Emission line measurements by adjustDetectorMap.
        detectorMaps : iterable of `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        dataIds : iterable of `dict`
            Data IDs for the input data.

        Returns
        -------
        Struct
            Output data products. See `DetectorMapQaConnections`.
        """
        # List all the objects we have received.
        self.log.info(
            f"Processing {detectorName=} {len(arcLines)} ArcLineSets and {len(detectorMaps)} DetectorMaps"
        )
        outputs = self.plotResidual.run(detectorName, arcLines, detectorMaps, dataIds)

        return outputs

        # If processing every exposure individually.
        # stats = list()
        # plots = list()
        # for data_id, lines in zip(dataIds, arcLines):
        #     self.log.info(f"Processing dataId {data_id}")
        #     detector_name = "{arm}{spectrograph}".format(**data_id)
        #
        #     output = self.plotResidual.run(detector_name, [lines], detectorMaps, [data_id])
        #     stats.append(output.dmQaResidualStats)
        #     plots.append(output.dmQaResidualPlot)

        # return Struct(
        #     dmQaResidualStats=stats,
        #     dmQaResidualPlot=plots,
        # )
