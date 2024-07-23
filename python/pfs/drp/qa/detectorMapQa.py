from collections import defaultdict
from typing import Iterable

import lsstDebug
import matplotlib.pyplot as plt
import numpy as np
from lsst.daf.persistence.butlerExceptions import NoResults
from lsst.pex.config import Field, ConfigurableField, Config
from lsst.pipe.base import (
    ArgumentParser,
    CmdLineTask,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    Task,
    TaskRunner,
)
from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from pfs.drp.qa.utils import helpers
from pfs.drp.qa.utils import plotting
from pfs.drp.stella import ArcLineSet, DetectorMap, PfsArm
from scipy.stats import iqr


class PlotResidualConfig(Config):
    """Configuration for PlotResidualTask"""

    combineVisits = Field(
        dtype=bool, default=False, doc="If all given visits should be combined and compared."
    )
    makeResidualPlots = Field(dtype=bool, default=True, doc="Generate a residual plot for each dataId.")
    useSigmaRange = Field(dtype=bool, default=False, doc="Use ±2.5 sigma as range")
    xrange = Field(dtype=float, default=0.1, doc="Range of the residual (X center) in a plot in pix.")
    wrange = Field(dtype=float, default=0.1, doc="Range of the residual (wavelength) in a plot in pix.")
    binWavelength = Field(
        dtype=float, default=0.1, doc="Bin the wavelength in the by-wavelength residual plot."
    )


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
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y
        arcLines : `ArcLineSet`
            Emission line measurements by adjustDetectorMap.
        pfsArm : `PfsArm`
            Extracted spectra from arm.

        Returns
        -------
        dmQaResidualImage : `MultipagePdfFigure`
            Detail plots.
            1D and 2D plots of the residual between the detectormap and the arclines.
        dmQaResidualStats : `QaDict`
            Data to be pickled.
            Statistics of the residual analysis.
        """

        arc_data, visit_stats, detector_stats = helpers.getStats(arcLinesSet, detectorMaps, dataIds)

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
                residFig.suptitle(f"DetectorMap Residuals\n{groupName}", weight="bold")

                return Struct(
                    dmQaResidualImage=residFig,
                    dmQaResidualStats=visit_stats,
                    dmQaDetectorStats=detector_stats,
                )
            else:
                return Struct(dmQaResidualStats=visit_stats)

    @classmethod
    def _makeArgumentParser(cls) -> ArgumentParser:
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(
            name="--id", datasetType="arcLines", level="Visit", help="data IDs, e.g. --id exp=12345"
        )
        return parser


class OverlapRegionLinesConfig(Config):
    """Configuration for OverlapRegionLinesTask"""

    pass


class OverlapRegionLinesTask(Task):
    """Task for QA of detectorMap"""

    ConfigClass = OverlapRegionLinesConfig
    _DefaultName = "overlapRegionLines"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(
        self, detectorMap: Iterable[DetectorMap], arcLines: Iterable[ArcLineSet], pfsArm: Iterable[PfsArm]
    ) -> Struct:
        """QA of adjustDetectorMap by plotting the wavelength difference of sky lines detected in multiple
        arms.

        Parameters
        ----------
        detectorMap : iterable of `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        arcLines : iterable of `ArcLineSet`
            Emission line measurements by adjustDetectorMap.
        pfsArm : iterable of `PfsArm`
            Extracted spectra from arm.

        Returns
        -------
        None

        Outputs
        -------
        plot : `overlapLines-{:06}-{}{}{}.png`
        """

        visit = pfsArm[0].identity.visit
        arm = [pp.identity.arm for pp in pfsArm]
        spectrograph = pfsArm[0].identity.spectrograph
        fmin, fmax = [np.amin(aa.fiberId) for aa in arcLines], [np.amax(aa.fiberId) for aa in arcLines]

        measured = [
            np.logical_not(np.isnan(aa.flux))
            & np.logical_not(np.isnan(aa.x))
            & np.logical_not(np.isnan(aa.y))
            & np.logical_not(np.isnan(aa.xErr))
            & np.logical_not(np.isnan(aa.yErr))
            & np.logical_not(np.isnan(aa.fluxErr))
            for aa in arcLines
        ]

        flist = []
        for aa in range(len(arcLines)):
            flist.append([])
            for f in range(fmin[aa], fmax[aa] + 1):
                notNan_f = (arcLines[aa].fiberId == f) & measured[aa]
                if np.sum(notNan_f) > 0:
                    flist[aa].append(f)
            self.log.info(f"Fiber number ({arm[aa]}{spectrograph}): {len(flist[aa])}")
            self.log.info(f"Measured line ({arm[aa]}{spectrograph}): {np.sum(measured[aa])}")

        plt.figure()
        fcommon = set(flist[0]) | set(flist[1])
        fibers = {}
        difference = {}
        wcommon = []
        goodLines = [
            "630.20",
            "636.55",
            "937.69",
            "937.85",
            "942.23",
            "947.94",
            "952.20",
            "957.00",
            "970.19",
            "972.25",
            "979.21",
            "979.38",
            "980.24",
        ]
        for f in fcommon:
            b0 = (arcLines[0].fiberId == f) & measured[0]
            b1 = (arcLines[1].fiberId == f) & measured[1]
            wav0 = set(arcLines[0][b0].wavelength)
            wav1 = set(arcLines[1][b1].wavelength)
            wav = list(wav0 & wav1)
            if len(wav) > 0:
                wav.sort()
                for w in wav:
                    if "{:.2f}".format(w) in goodLines:
                        if w not in wcommon:
                            wcommon.append(w)
                            fibers[w] = []
                            difference[w] = []
                        y = [aa[(aa.fiberId == f) & (aa.wavelength == w)].y[0] for aa in arcLines]
                        fibers[w].append(f)
                        diff_wl0 = detectorMap[0].findWavelength(fiberId=f, row=y[0])
                        diff_wl1 = detectorMap[1].findWavelength(fiberId=f, row=y[1])
                        difference[w].append(diff_wl0 - diff_wl1)
        plt.figure()
        wcommon.sort()
        for w in wcommon:
            self.log.info(
                f"{w} nm ({len(fibers[w])} fibers, "
                f"median={np.median(difference[w]):.1e} nm, "
                f"1sigma={iqr(difference[w]) / 1.349:.3f} nm)"
            )
            plt.scatter(
                fibers[w],
                difference[w],
                s=3,
                label=f"{w} nm ({len(fibers[w])} fibers, "
                f"median={np.median(difference[w]):.1e} nm, "
                f"1sigma={iqr(difference[w]) / 1.349:.3f} nm)",
            )
        plt.legend(fontsize=7)
        plt.xlabel("fiberId")
        plt.ylabel(f"Wavelength difference ({arm[0]}-{arm[1]}) [nm]")
        plt.savefig(f"overlapLines-{visit:06}-{arm[0]}{arm[1]}{spectrograph}.png")
        plt.close()

        return Struct()

    @classmethod
    def _makeArgumentParser(cls) -> ArgumentParser:
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(
            name="--id", datasetType="arcLines", level="Visit", help="data IDs, e.g. --id exp=12345"
        )
        return parser


# (Gen3) If this task is ("instrument", "exposure", "detector") declare to be executed for each combination.
class DetectorMapQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "exposure"),
):
    """Connections for DetectorMapQaTask"""

    detectorMap = InputConnection(
        name="detectorMap_used",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "exposure", "detector"),
    )
    arcLines = InputConnection(
        name="arcLines",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra from arm",
        storageClass="PfsArm",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    dmQaResidualImage = OutputConnection(
        name="dmQaResidualImage",
        doc="Detail plots. The 1D and 2D residual plots of the detectormap with the arclines.",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument", "exposure", "detector"),
    )
    dmQaResidualStats = OutputConnection(
        name="dmQaResidualStats",
        doc="Statistics of the residual analysis.",
        storageClass="pandas.core.frame.DataFrame",
        dimensions=("instrument", "exposure", "detector"),
    )


class DetectorMapQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapQaConnections):
    """Configuration for DetectorMapQaTask"""

    checkOverlap = Field(dtype=bool, default=False, doc="If the overlapRegionLines should be checked.")

    plotResidual = ConfigurableField(target=PlotResidualTask, doc="Plot the detector map residual.")
    overlapRegionLines = ConfigurableField(
        target=OverlapRegionLinesTask,
        doc="Plot the wavelength difference of the sky lines commonly detected in multiple arms.",
    )


class DetectorMapQaRunner(TaskRunner):
    """Runner for DetectorMapQaTask"""

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Produce list of targets for DetectorMapQaTask.

        The visits and detector are processed as part of a group, with group
        membership consisting of:

            * If combineVisits, group all visits by detector.
            * If checkOverlap, group all detectors by visit.
            * Otherwise, group per visit per detector (i.e. don't group).
        """
        combineVisits = parsedCmd.config.plotResidual.combineVisits
        checkOverlap = parsedCmd.config.checkOverlap

        groups = defaultdict(list)
        for ref in parsedCmd.id.refList:
            visit = ref.dataId["visit"]
            spectrograph = ref.dataId["spectrograph"]
            arm = ref.dataId["arm"]
            ccd = f"{arm}{spectrograph}"

            if combineVisits is True:
                groups[ccd].append(ref)
            elif checkOverlap is True:
                groups[visit].append(ref)
            else:
                groups[f"{visit}-{ccd}"].append(ref)

        processGroups = [((key, group), kwargs) for key, group in groups.items()]

        return processGroups


class DetectorMapQaTask(CmdLineTask, PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapQaConfig
    _DefaultName = "detectorMapQa"
    RunnerClass = DetectorMapQaRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("plotResidual")
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

    def runDataRef(self, expSpecRefList) -> Struct:
        """Calls ``self.run()``

        Parameters
        ----------
        expSpecRefList : iterable of iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for each sensor, grouped by spectrograph.

        Returns
        -------
        None
        """
        groupName = expSpecRefList[0]
        groupDataRefs = expSpecRefList[1]

        self.log.info(f"Starting processing for {groupName=} with {len(groupDataRefs)} dataIds")

        arcLinesSet = list()
        detectorMaps = list()
        dataIds = list()
        rerun_name = None
        calib_dir = None
        for dataRef in groupDataRefs:
            if rerun_name is None:
                # TODO fix this one day with Gen3
                rerun_name = dataRef.butlerSubset.butler._repos.inputs()[0].repoArgs.root
                calib_dir = dataRef.butlerSubset.butler._repos.inputs()[0].repoArgs.mapperArgs["calibRoot"]

            try:
                detectorMap = dataRef.get("detectorMap_used")
                arcLines = dataRef.get("arcLines")

                arcLinesSet.append(arcLines)
                detectorMaps.append(detectorMap)
                dataIds.append(dataRef.dataId)
            except NoResults:
                self.log.info(f"No results for {dataRef}")
            except Exception as e:
                self.log.error(e)

        # Run the task and get the outputs.
        outputs = self.run(groupName, arcLinesSet, detectorMaps, dataIds)
        if outputs is not None:
            for dataRef in groupDataRefs:
                for datasetType, data in outputs.getDict().items():
                    dataIdStr = "v{visit}-{arm}{spectrograph}".format(**dataRef.dataId)

                    self.log.info(f"Saving {datasetType} for {dataIdStr}")
                    dataRef.put(data, datasetType=datasetType)

    def run(self, *args, **kwargs) -> Struct:
        """Generate detectorMapQa plots.

        Parameters
        ----------
        detectorMap: `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        arcLines : `ArcLineSet`
            Emission line measurements by adjustDetectorMap.
        pfsArm : `PfsArm`
            Extracted spectra from arm.

        Returns
        -------
        None
        """
        return self.plotResidual.run(*args, **kwargs)

    def _getMetadataName(self):
        return None
