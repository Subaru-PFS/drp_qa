from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, Task
from pfs.drp.qa.utils import helpers, plotting
from pfs.drp.stella import ArcLineSet, DetectorMap


class PlotResidualConfig(Config):
    """Configuration for PlotResidualTask"""

    combineVisits = Field(dtype=bool, default=False, doc="Combine all visits for processing.")
    makeResidualPlots = Field(dtype=bool, default=True, doc="Generate a residual plot for each dataId.")
    useSigmaRange = Field(dtype=bool, default=False, doc="Use ±2.5 sigma as range")
    xrange = Field(dtype=float, default=0.1, doc="Range of the residual (X center) in a plot in pix.")
    wrange = Field(dtype=float, default=0.1, doc="Range of the residual (wavelength) in a plot in pix.")
    binWavelength = Field(dtype=float, default=0.1, doc="Wavelength bin for residual plot.")


class PlotResidualTask(Task):
    """Task for QA of detectorMap."""

    ConfigClass = PlotResidualConfig
    _DefaultName = "plotResidual"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, groupName, arcLinesSet: ArcLineSet, detectorMaps: DetectorMap, dataIds) -> Struct:
        """QA of adjustDetectorMap by plotting the fitting residual.

        Parameters
        ----------
        groupName : `str`
            Group name, either the visit or the detector.
        arcLinesSet : `ArcLineSet`
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

        arc_data, visit_stats, detector_stats = helpers.getStats(arcLinesSet, detectorMaps, dataIds)

        results = Struct()
        if arc_data is not None and len(arc_data) and visit_stats is not None and len(visit_stats):
            if self.config.makeResidualPlots is True:
                arm = str(groupName[-2])
                spectrograph = int(groupName[-1])
                residFig = plotting.makePlot(
                    arc_data,
                    visit_stats,
                    arm,
                    spectrograph,
                    useSigmaRange=self.config.useSigmaRange,
                    xrange=self.config.xrange,
                    wrange=self.config.wrange,
                    binWavelength=self.config.binWavelength,
                )

                suptitle = f"DetectorMap Residuals\n{groupName}"
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
                        dmQaResidualPlot=residFig,
                        dmQaResidualStats=visit_stats,
                    )

        return results
