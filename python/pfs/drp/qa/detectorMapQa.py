from typing import Collection, Dict, Iterable, Mapping, Tuple

from lsst.daf.butler import DataCoordinate, DatasetRef
from lsst.pex.config import Field
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
    BaseInput,
    Input as InputConnection,
    Output,
    Output as OutputConnection,
)
from pfs.drp.stella import ArcLineSet, DetectorMap

from pfs.drp.qa.tasks.detectorMapResiduals import get_residual_info, plot_detectormap_residuals


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
        self.log.info(f"DetectorMapQaConnections.__init__ {config=}")
        # Remove the unused connections depending on the configuration.
        if config.combineVisits:
            del self.dmQaResidualPlot
            del self.dmQaResidualStats
        else:
            del self.dmQaCombinedResidualPlot
            del self.dmQaDetectorStats

    def adjustQuantum(
        self,
        inputs: Dict[str, Tuple[BaseInput, Collection[DatasetRef]]],
        outputs: Dict[str, Tuple[Output, Collection[DatasetRef]]],
        label: str,
        data_id: DataCoordinate,
    ) -> Tuple[
        Mapping[str, Tuple[BaseInput, Collection[DatasetRef]]],
        Mapping[str, Tuple[Output, Collection[DatasetRef]]],
    ]:
        """Adjust the connections for a single quantum.

        We duplicate the detectorMap for each arcline.

        Parameters
        ----------
        inputs : dict
            The inputs for the quantum.
        outputs : dict
            The outputs for the quantum.
        label : str
            The label for the quantum.
        data_id : DataCoordinate
            The data ID for the quantum.

        Returns
        -------
        dict
            The adjusted inputs for the quantum.
        dict
            The adjusted outputs for the quantum.
        """

        adjusted_inputs = inputs.copy()
        adjusted_outputs = outputs.copy()

        # Duplicate the detectorMap for each arcline.
        adjusted_inputs["detectorMaps"] = (
            inputs["detectorMaps"][0],
            inputs["detectorMaps"][1] * len(inputs["arcLines"][1]),
        )
        inputs["detectorMaps"] = adjusted_inputs["detectorMaps"]

        super().adjustQuantum(inputs, outputs, label, data_id)
        return adjusted_inputs, adjusted_outputs

    detectorMaps = InputConnection(
        name="detectorMap_calib",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=(
            "instrument",
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


class DetectorMapQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapQaConnections):
    """Configuration for DetectorMapQaTask"""

    combineVisits = Field(dtype=bool, default=False, doc="Combine all visits for processing.")
    makeResidualPlots = Field(dtype=bool, default=True, doc="Generate a residual plot for each dataId.")
    useSigmaRange = Field(dtype=bool, default=False, doc="Use ±2.5 sigma as range")
    spatialRange = Field(
        dtype=float, default=0.1, doc="Spatial range for the residual plot, implies useSigmaRange is False."
    )
    wavelengthRange = Field(
        dtype=float,
        default=0.1,
        doc="Wavelegnth range for the residual plot, implies useSigmaRange is False.",
    )
    binWavelength = Field(dtype=float, default=0.1, doc="Wavelength bin for residual plot.")


class DetectorMapQaTask(PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapQaConfig
    _DefaultName = "detectorMapQa"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        # Get the dataIds for help with plotting.
        data_ids = []
        for ref in inputRefs.arcLines:
            if "exposure" in ref.dataId.full:
                data_id = {k: v for k, v in ref.dataId.full.items()}
                data_id["visit"] = data_id["exposure"]
                data_id["run"] = ref.run
                data_ids.append(data_id)

        inputs = butlerQC.get(inputRefs)

        inputs["dataIds"] = data_ids
        # There should only be one detectorMap unique input ref, so get from the first.
        inputs["detectorName"] = "{arm}{spectrograph}".format(**data_ids[0])

        # Perform the actual processing.
        outputs = self.run(**inputs)

        # Store the results.
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        detectorName,
        arcLines: Iterable[ArcLineSet],
        detectorMaps: Iterable[DetectorMap],
        dataIds: Iterable[dict],
    ) -> Struct:
        """QA of adjustDetectorMap by plotting the fitting residual.

        Parameters
        ----------
        detectorName : `str`
            The detector name, e.g. ``"r2"``.
        arcLines : `ArcLineSet`
            Emission line measurements by adjustDetectorMap.
        detectorMaps : `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        dataIds : `list`
            List of dataIds.

        Returns
        -------
        dmQaResidualPlot : `MultipagePdfFigure`
            1D and 2D plots of the residual between the detectormap and the arclines for a given visit.
        dmQaCombinedResidualPlot : `MultipagePdfFigure`
            1D and 2D plots of the residual between the detectormap and the arclines for the entire detector.
        dmQaResidualStats : `pd.DataFrame`
            Statistics of the residual analysis.
        dmQaDetectorStats : `pd.DataFrame`
            Statistics of the residual analysis.
        """
        arc_data, visit_stats, detector_stats = get_residual_info(arcLines, detectorMaps, dataIds)

        run = dataIds[0]["run"]

        results = Struct()
        if arc_data is not None and len(arc_data) and visit_stats is not None and len(visit_stats):
            if self.config.makeResidualPlots is True:
                arm = str(detectorName[-2])
                spectrograph = int(detectorName[-1])
                residFig = plot_detectormap_residuals(
                    arc_data,
                    visit_stats,
                    arm,
                    spectrograph,
                    useSigmaRange=self.config.useSigmaRange,
                    xrange=self.config.xrange,
                    wrange=self.config.wrange,
                    binWavelength=self.config.binWavelength,
                )

                suptitle = f"DetectorMap Residuals {detectorName}\n{run}"
                if self.config.combineVisits is True:
                    suptitle = f"Combined {suptitle}"

                residFig.suptitle(suptitle, weight="bold")
                if self.config.combineVisits is True:
                    results = Struct(
                        dmQaCombinedResidualPlot=residFig,
                        dmQaDetectorStats=detector_stats,
                    )
                else:
                    results = Struct(
                        dmQaResidualPlot=[residFig],
                        dmQaResidualStats=[visit_stats],
                    )

        return results
