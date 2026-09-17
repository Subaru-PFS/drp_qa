import itertools
from collections.abc import Iterable

import pandas as pd
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
    Input as InputConnection,
)
from lsst.pipe.base.connectionTypes import (
    Output as OutputConnection,
)

from pfs.drp.qa.plotting.dmCombined import (
    plot_detector_summary,
    plot_detector_summary_per_desc,
    plot_title,
    plot_visits,
    reportFigures,
)
from pfs.drp.qa.storageClasses import MultipagePdfFigure
from pfs.drp.stella import DetectorMap

# Re-exported for backwards compatibility: the plotting functions moved to
# `pfs.drp.qa.plotting.dmCombined`, where they can be tested without the stack.
# Prefer importing from there in new code.
__all__ = [
    "DetectorMapCombinedResidualsTask",
    "make_report",
    "plot_detector_summary",
    "plot_detector_summary_per_desc",
    "plot_title",
    "plot_visits",
]


class DetectorMapCombinedResidualsConnections(
    PipelineTaskConnections,
    dimensions=("instrument",),
):
    """Connections for DetectorMapCombinedQaTask."""

    detectorMaps = InputConnection(
        name="detectorMap",
        doc="Adjusted detector mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=(
            "instrument",
            "visit",
            "arm",
            "spectrograph",
        ),
        multiple=True,
    )

    dmQaResidualData = InputConnection(
        name="dmQaResidualData",
        doc="DM QA residual data for plotting",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "visit",
            "arm",
            "spectrograph",
        ),
        multiple=True,
    )

    dmQaResidualStats = InputConnection(
        name="dmQaResidualStats",
        doc="DM QA residual statistics",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "visit",
            "arm",
            "spectrograph",
        ),
        multiple=True,
    )

    dmQaCombinedResidualPlot = OutputConnection(
        name="dmQaCombinedResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for all detectors.",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument",),
    )

    dmQaDetectorStats = OutputConnection(
        name="dmQaDetectorStats",
        doc="Statistics of the residual analysis for all detectors.",
        storageClass="DataFrame",
        dimensions=("instrument",),
    )


class DetectorMapCombinedResidualsConfig(
    PipelineTaskConfig, pipelineConnections=DetectorMapCombinedResidualsConnections
):
    """Configuration for DetectorMapCombinedQaTask."""

    useSigmaRange = Field(dtype=bool, default=False, doc="Use ±2.5 sigma as range")


class DetectorMapCombinedResidualsTask(PipelineTask):
    """Task for QA of detectorMap."""

    ConfigClass = DetectorMapCombinedResidualsConfig
    _DefaultName = "dmCombinedResiduals"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        run_name = inputRefs.dmQaResidualStats[0].run

        inputs = butlerQC.get(inputRefs)
        inputs["run_name"] = run_name

        # Perform the actual processing.
        outputs = self.run(**inputs)

        # Store the results.
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        detectorMaps: Iterable[DetectorMap],
        dmQaResidualData: Iterable[pd.DataFrame],
        dmQaResidualStats: Iterable[pd.DataFrame],
        run_name: str,
    ) -> Struct:
        """Create detector level residual_stats and plots.

        Parameters
        ----------
        detectorMaps : Iterable[DetectorMap]
            An iterable of detector maps. Used for plotting metadata.
        dmQaResidualData : Iterable[DataFrame]
            An iterable of DataFrames containing DM QA residual data. These
            are combined into a single DataFrame for processing.
        dmQaResidualStats : Iterable[DataFrame]
            An iterable of DataFrames containing DM QA residual statistics. These
            are combined into a single DataFrame for processing.
        run_name : str
            The name of the collection that was used for the residual_stats.

        Returns
        -------
        dmQaCombinedResidualPlot : `MultipagePdfFigure`
            1D and 2D plots of the residual between the detectormap and the arclines for the entire detector.
        dmQaDetectorStats : `pd.DataFrame`
            Statistics of the residual analysis.
        """
        # Put the DetectorMaps in a dict by CCD.
        self.log.debug(f"Visits: { {dm.getVisitInfo().id for dm in detectorMaps} }")

        # Small helper to use while https://pfspipe.ipmu.jp/jira/browse/PIPE2D-1423
        def get_ccd(dm: DetectorMap) -> str:
            return "".join([x.split("=")[1] for x in dm.metadata["CALIB_ID"].split(" ")[:2]])

        detectorMaps = {get_ccd(detectorMap): detectorMap for detectorMap in detectorMaps}
        self.log.debug(f"DetectorMap CCDs: {detectorMaps.keys()}")

        residual_data = pd.concat(dmQaResidualData)
        residual_stats = pd.concat(dmQaResidualStats)
        residual_stats.sort_values(by=["visit", "arm", "spectrograph", "description"], inplace=True)

        # Put the CCD column in a wavelength sorted order.
        residual_stats.ccd = residual_stats.ccd.astype("category")
        spec_order = [1, 2, 3, 4]
        arm_order = ["b", "r", "m", "n"]
        detector_order = [f"{arm}{spec}" for arm, spec in itertools.product(arm_order, spec_order)]
        detector_order = [d for d in detector_order if d in residual_stats.ccd.cat.categories]
        residual_stats.ccd = residual_stats.ccd.cat.reorder_categories(detector_order, ordered=True)

        self.log.info("Making combined report")
        pdf = make_report(residual_stats, residual_data, detectorMaps, run_name=run_name, log=self.log)

        return Struct(dmQaCombinedResidualPlot=pdf, dmQaDetectorStats=residual_stats)


def make_report(
    residual_stats: pd.DataFrame,
    residual_data: pd.DataFrame,
    detectorMaps: dict[str, DetectorMap],
    run_name: str,
    log: object,
) -> MultipagePdfFigure:
    """Assemble the combined residual report as a multi-page PDF.

    The pages come from `pfs.drp.qa.plotting.dmCombined.reportFigures`; the only
    thing that happens here is binding them to the Butler storage class, which
    is why this function stays with the task and the drawing does not.

    Parameters
    ----------
    residual_stats : `pandas.DataFrame`
        Concatenated ``dmQaResidualStats``.
    residual_data : `pandas.DataFrame`
        Concatenated ``dmQaResidualData``.
    detectorMaps : `dict` [`str`, `DetectorMap`]
        Detector maps keyed by CCD name.
    run_name : `str`
        The collection the statistics came from.
    log : `object`
        A logger.

    Returns
    -------
    `MultipagePdfFigure`
        The report. Each figure is saved at its own dpi, which
        `plot_detectormap_residuals` already sets to 150.
    """
    pdf = MultipagePdfFigure()
    for figure in reportFigures(residual_stats, residual_data, detectorMaps, run_name, log):
        pdf.append(figure)
    return pdf
