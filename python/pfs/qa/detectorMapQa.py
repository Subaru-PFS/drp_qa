import pickle
from collections import defaultdict
from typing import Iterable, Any, Dict

import lsstDebug
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
from lsst.pipe.base.connectionTypes import Input as InputConnection
from pfs.drp.stella import ArcLineSet, DetectorMap, PfsArm, ReferenceLineStatus
from scipy.stats import iqr
from .utils import stability


class PlotResidualConfig(Config):
    """Configuration for PlotResidualTask"""

    showAllRange = Field(dtype=bool, default=False, doc="Show all data points in a plot?")
    xrange = Field(dtype=float, default=0.2, doc="Range of the residual (X center) in a plot in pix.")
    wrange = Field(dtype=float, default=0.03, doc="Range of the residual (wavelength) in a plot in nm.")
    pointSize = Field(dtype=float, default=0.2, doc="Point size in plots.")
    quivLength = Field(dtype=float, default=0.2, doc="Quiver length in plots")


class PlotResidualTask(Task):
    """Task for QA of detectorMap."""

    ConfigClass = PlotResidualConfig
    _DefaultName = "plotResidual"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, detectorMap: DetectorMap, arcLines: ArcLineSet, pfsArm: PfsArm) -> Struct:
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
        None

        Outputs
        -------
        1d plot : `dmapQAPlot-{:06}-{}{}.png`
            The residuals of Xcenter and wavelength are plotted.
        2d plots : `dmapQAPlot2Dused-{:06}-{}{}.png`, `dmapQAPlot2Dreserved-{:06}-{}{}.png`
            The residuals of Xcenter and wavelength are plotted in the array format.
        pickle data : `dmapQAStats-{:06}-{}{}.pickle`
            The fiberId, number of lines (detectormap_used, detectormap_reserved), medians and sigmas of the
            fitting residuals are stored in the dict format.
        """

        visit = pfsArm.identity.visit
        arm = pfsArm.identity.arm
        spectrograph = pfsArm.identity.spectrograph

        arc_data = stability.getArclineData(arcLines)
        self.log.info(f"Number of fibers: {len(arc_data.fiberId.unique())}")
        self.log.info(f"Number of Measured lines: {len(arc_data)}")

        # Get dataframe for arc lines and add detectorMap information, then calculate residuals.
        arc_data = stability.getArclineData(arcLines)
        arc_data = stability.addTraceLambdaToArclines(arc_data, detectorMap)
        arc_data = stability.addResidualsToArclines(arc_data)

        # Get our statistics and write them to a pickle file.
        statistics = self.getStatistics(arc_data, pfsArm)
        with open(f"dmapQAStats-{visit:06}-{arm}{spectrograph}.pickle", "wb") as f:
            pickle.dump(statistics, f)

        fig1 = self.plotResiduals1D(arcLines, detectorMap, statistics)
        fig2, fig3 = self.plotResiduals2D(arc_data, detectorMap)

        fig1.suptitle(f"Detector map residual ({visit=}, {arm=}, {spectrograph=})")
        fig2.suptitle(f"DETECTORMAP_USED residual ({visit=}, {arm=}, {spectrograph=})")
        fig3.suptitle(f"DETECTORMAP_RESERVED residual ({visit=}, {arm=}, {spectrograph=})")

        fig1.savefig(f'dmapQAPlot-{visit:06}-{arm}{spectrograph}.png', format="png")
        fig2.savefig(f'dmapQAPlot2Dused-{visit:06}-{arm}{spectrograph}.png', format="png")
        fig3.savefig(f'dmapQAPlot2Dreserved-{visit:06}-{arm}{spectrograph}.png', format="png")

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

        # There is no output in this template, so I can't give an example write the output here
        return Struct()

    def plotResiduals1D(self, arcLines: ArcLineSet, detectorMap: DetectorMap, statistics: Dict[str, Any]) -> plt.Figure:
        """Plot the residuals as a function of wavelength and fiberId.

        Parameters:
            arcLines: The arc lines.
            detectorMap: The detector map.
            statistics: The statistics.

        Returns:
            The figure.
        """
        fmin, fmax = np.amin(arcLines.fiberId), np.amax(arcLines.fiberId)
        dmapUsed = (arcLines.status & ReferenceLineStatus.DETECTORMAP_USED) != 0
        dmapResearved = (arcLines.status & ReferenceLineStatus.DETECTORMAP_RESERVED) != 0

        measured = (
                np.logical_not(np.isnan(arcLines.flux))
                & np.logical_not(np.isnan(arcLines.x))
                & np.logical_not(np.isnan(arcLines.y))
                & np.logical_not(np.isnan(arcLines.xErr))
                & np.logical_not(np.isnan(arcLines.yErr))
                & np.logical_not(np.isnan(arcLines.fluxErr))
        )

        flist = []
        for f in range(fmin, fmax + 1):
            notNan_f = (arcLines.fiberId == f) & measured
            if np.sum(notNan_f) > 0:
                flist.append(f)

        arcLinesMeasured = arcLines[measured]
        residualX = arcLinesMeasured.x - detectorMap.getXCenter(arcLinesMeasured.fiberId, arcLinesMeasured.y)
        residualW = arcLinesMeasured.wavelength - detectorMap.findWavelength(
            fiberId=arcLinesMeasured.fiberId, row=arcLinesMeasured.y
        )
        minw = np.amin(detectorMap.findWavelength(fiberId=arcLinesMeasured.fiberId, row=arcLinesMeasured.y))
        maxw = np.amax(detectorMap.findWavelength(fiberId=arcLinesMeasured.fiberId, row=arcLinesMeasured.y))
        bufw = (maxw - minw) * 0.02

        dmUsedMeasured = dmapUsed[measured]
        dmReservedMeasured = dmapResearved[measured]
        if self.config.showAllRange:
            residualXMax = max(np.amax(residualX[dmUsedMeasured]), np.amax(residualX[dmReservedMeasured]))
            residualXMin = min(np.amin(residualX[dmUsedMeasured]), np.amin(residualX[dmReservedMeasured]))
            residualWMax = max(np.amax(residualW[dmUsedMeasured]), np.amax(residualW[dmReservedMeasured]))
            residualWMin = min(np.amin(residualW[dmUsedMeasured]), np.amin(residualW[dmReservedMeasured]))
            yxmax = self.config.xrange * 3 if residualXMax < self.config.xrange * 3 else residualXMax * 1.05
            yxmin = -self.config.xrange * 3 if residualXMin > -self.config.xrange * 3 else residualXMin * 1.05
            ywmax = self.config.wrange * 3 if residualWMax < self.config.wrange * 3 else residualWMax * 1.05
            ywmin = -self.config.wrange * 3 if residualWMin > -self.config.wrange * 3 else residualWMin * 1.05
        else:
            yxmax = self.config.xrange * 3
            yxmin = -self.config.xrange * 3
            ywmax = self.config.wrange * 3
            ywmin = -self.config.wrange * 3
            largeX = residualX > yxmax
            smallX = residualX < yxmin
            largeW = residualW > ywmax
            smallW = residualW < ywmin

        fig1 = plt.figure(figsize=(12, 8))

        ax1 = [
            plt.axes([0.08, 0.08, 0.37, 0.36]),
            plt.axes([0.46, 0.08, 0.07, 0.36]),
            plt.axes([0.08, 0.54, 0.37, 0.36]),
            plt.axes([0.46, 0.54, 0.07, 0.36]),
            plt.axes([0.58, 0.08, 0.37, 0.36]),
            plt.axes([0.58, 0.54, 0.37, 0.36]),
        ]

        ax1[0].scatter(
            arcLinesMeasured.wavelength[dmUsedMeasured],
            residualX[dmUsedMeasured],
            s=self.config.pointSize,
            c="b",
            label="DETECTORMAP_USED\n(median:{:.2e}, sigma:{:.2e})".format(
                np.median(residualX[dmUsedMeasured]), iqr(residualX[dmUsedMeasured]) / 1.349
            ),
        )
        if not self.config.showAllRange:
            if np.sum(largeX) + np.sum(smallX) > 0:
                ax1[0].quiver(arcLinesMeasured.wavelength[dmUsedMeasured & largeX],
                              np.zeros(np.sum(
                                  dmUsedMeasured & largeX)) + yxmax - self.config.xrange * self.config.quivLength, 0,
                              self.config.xrange * self.config.quivLength,
                              label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                                  yxmax, np.sum(dmUsedMeasured & largeX) / np.sum(dmUsedMeasured) * 100
                              ),
                              color="b",
                              angles="xy",
                              scale_units="xy",
                              scale=2,
                              )
                ax1[0].quiver(
                    arcLinesMeasured.wavelength[dmUsedMeasured & smallX],
                    np.zeros(np.sum(dmUsedMeasured & smallX)) + yxmin + self.config.xrange * self.config.quivLength, 0,
                    -self.config.xrange * self.config.quivLength,
                    color="b",
                    angles="xy",
                    scale_units="xy",
                    scale=2,
                )
        ax1[0].scatter(
            arcLinesMeasured.wavelength[dmReservedMeasured],
            residualX[dmReservedMeasured],
            s=self.config.pointSize,
            c="r",
            label="DETECTORMAP_RESERVED\n(median:{:.2e}, sigma:{:.2e})".format(
                np.median(residualX[dmReservedMeasured]), iqr(residualX[dmReservedMeasured]) / 1.349
            ),
        )
        if not self.config.showAllRange:
            if np.sum(largeX) + np.sum(smallX) > 0:
                ax1[0].quiver(arcLinesMeasured.wavelength[dmReservedMeasured & largeX],
                              np.zeros(np.sum(
                                  dmReservedMeasured & largeX)) + yxmax - self.config.xrange * self.config.quivLength,
                              0,
                              self.config.xrange * self.config.quivLength,
                              label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                                  yxmax,
                                  (np.sum(dmReservedMeasured & largeX) + np.sum(dmReservedMeasured & smallX))
                                  / np.sum(dmReservedMeasured)
                                  * 100,
                              ),
                              color="r",
                              angles="xy",
                              scale_units="xy",
                              scale=2,
                              )
                ax1[0].quiver(
                    arcLinesMeasured.wavelength[dmReservedMeasured & smallX],
                    np.zeros(np.sum(dmReservedMeasured & smallX)) + yxmin + self.config.xrange * self.config.quivLength,
                    0,
                    -self.config.xrange * self.config.quivLength,
                    color="r",
                    angles="xy",
                    scale_units="xy",
                    scale=2,
                )
        ax1[1].hist(
            residualX[dmUsedMeasured],
            color="b",
            range=(-self.config.xrange * 3, self.config.xrange * 3),
            bins=35,
            orientation="horizontal",
        )
        ax1[1].hist(
            residualX[dmReservedMeasured],
            color="r",
            range=(-self.config.xrange * 3, self.config.xrange * 3),
            bins=35,
            orientation="horizontal",
        )
        ax1[2].scatter(
            arcLinesMeasured.wavelength[dmUsedMeasured & (arcLinesMeasured.description != "Trace")],
            residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")],
            s=self.config.pointSize,
            c="b",
            label="DETECTORMAP_USED\n(median:{:.2e}, sigma:{:.2e})".format(
                np.median(residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")]),
                iqr(residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")]) / 1.349,
            ),
        )
        if not self.config.showAllRange:
            if np.sum(largeW) + np.sum(smallW) > 0:
                ax1[2].quiver(
                    arcLinesMeasured.wavelength[dmUsedMeasured & largeW],
                    np.zeros(np.sum(dmUsedMeasured & largeW)) + ywmax - self.config.wrange * self.config.quivLength, 0,
                    self.config.wrange * self.config.quivLength,
                    label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                        ywmax,
                        (np.sum(dmUsedMeasured & largeW) + np.sum(dmUsedMeasured & smallW))
                        / np.sum(dmUsedMeasured)
                        * 100,
                    ),
                    color="b",
                    angles="xy",
                    scale_units="xy",
                    scale=2,
                )
                ax1[2].quiver(
                    arcLinesMeasured.wavelength[dmUsedMeasured & smallW],
                    np.zeros(np.sum(dmUsedMeasured & smallW)) + ywmin + self.config.wrange * self.config.quivLength, 0,
                    self.config.wrange * self.config.quivLength,
                    color="b",
                    angles="xy",
                    scale_units="xy",
                    scale=2,
                )
        ax1[2].scatter(
            arcLinesMeasured.wavelength[dmReservedMeasured & (arcLinesMeasured.description != "Trace")],
            residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")],
            s=self.config.pointSize,
            c="r",
            label="DETECTORMAP_RESERVED\n(median:{:.2e}, sigma:{:.2e})".format(
                np.median(residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")]),
                iqr(residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")]) / 1.349,
            ),
        )
        if not self.config.showAllRange:
            if np.sum(largeW) + np.sum(smallW) > 0:
                ax1[2].quiver(
                    arcLinesMeasured.wavelength[dmReservedMeasured & largeW],
                    np.zeros(np.sum(dmReservedMeasured & largeW)) + ywmax - self.config.wrange * self.config.quivLength,
                    0,
                    self.config.wrange * self.config.quivLength,
                    label="Greater than {:.2f} in absolute value ({:.1e}%)".format(
                        ywmax,
                        (np.sum(dmReservedMeasured & largeW) + np.sum(dmReservedMeasured & smallW))
                        / np.sum(dmReservedMeasured)
                        * 100,
                    ),
                    color="r",
                    angles="xy",
                    scale_units="xy",
                    scale=2,
                )
                ax1[2].quiver(
                    arcLinesMeasured.wavelength[dmReservedMeasured & smallW],
                    np.zeros(np.sum(dmReservedMeasured & smallW)) + ywmin + self.config.wrange * self.config.quivLength,
                    0,
                    -self.config.wrange * self.config.quivLength,
                    color="r",
                    angles="xy",
                    scale_units="xy",
                    scale=2,
                )
        ax1[3].hist(
            residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")],
            color="b",
            range=(-self.config.wrange * 3, self.config.wrange * 3),
            bins=35,
            orientation="horizontal",
        )
        ax1[3].hist(
            residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")],
            color="r",
            range=(-self.config.wrange * 3, self.config.wrange * 3),
            bins=35,
            orientation="horizontal",
        )
        ax1[4].errorbar(
            statistics["fiberId"],
            statistics["Median_Xused"],
            yerr=statistics["Sigma_Xused"],
            fmt="bo",
            markersize=2,
        )
        ax1[5].errorbar(
            statistics["fiberId"],
            statistics["Median_Wused"],
            yerr=statistics["Sigma_Wused"],
            fmt="bo",
            markersize=2,
        )
        ax1[0].legend(fontsize=8)
        ax1[0].set_ylabel("X residual (pix)")
        ax1[0].set_xlabel("Wavelength (nm)")
        ax1[0].set_xlim(minw - bufw, maxw + bufw)
        ax1[0].set_ylim(yxmin, yxmax)
        ax1[0].set_title("X center residual (unit=pix)")
        ax1[1].set_ylim(yxmin, yxmax)
        ax1[1].set_yticklabels([])
        ax1[2].legend(fontsize=8)
        ax1[2].set_ylabel("Wavelength residual (nm)")
        ax1[2].set_xlabel("Wavelength (nm)")
        ax1[2].set_xlim(minw - bufw, maxw + bufw)
        ax1[2].set_ylim(ywmin, ywmax)
        ax1[2].set_title("Wavelength residual (unit=nm)")
        ax1[3].set_ylim(ywmin, ywmax)
        ax1[3].set_yticklabels([])
        ax1[4].set_xlabel("fiberId")
        ax1[4].set_ylim(yxmin, yxmax)
        ax1[4].set_title("X center residual of each fiber\n(point=median, errbar=1sigma scatter, unit=pix)")
        ax1[5].set_xlabel("fiberId")
        ax1[5].set_ylim(ywmin, ywmax)
        ax1[5].set_title("Wavelength residual of each fiber\n(point=median, errbar=1sigma scatter, unit=nm)")

        return fig1

    def plotResiduals2D(self, arc_data: pd.DataFrame, detectorMap: DetectorMap):

        used_data = arc_data.query(f'status == "{ReferenceLineStatus.DETECTORMAP_USED}"')
        fig2 = stability.plotArcResiduals2D(used_data, detectorMap)

        reserved_data = arc_data.query(f'status == "{ReferenceLineStatus.DETECTORMAP_RESERVED}"')
        fig3 = stability.plotArcResiduals2D(reserved_data, detectorMap)

        return fig2, fig3

    def getStatistics(self, arc_data: pd.DataFrame, pfsArm: PfsArm) -> Dict[str, Any]:
        dmapUsed = arc_data.query(f'status == {ReferenceLineStatus.DETECTORMAP_USED}')
        dmapReserved = arc_data.query(f'status == {ReferenceLineStatus.DETECTORMAP_RESERVED}')

        statistics = {
            "fiberId": arc_data.fiberId.unique(),
            "MedianXusedAll": dmapUsed.dx.median(),
            "MedianXreservedAll": dmapReserved.dx.median(),
            "MedianWusedAll": dmapUsed.query('Trace == False').dy.median(),
            "MedianWreservedAll": dmapReserved.query('Trace == False').dy.median(),
            "SigmaXusedAll": iqr(dmapUsed.dx) / 1.349,
            "SigmaXreservedAll": iqr(dmapReserved.dx) / 1.349,
            "SigmaWusedAll": iqr(dmapUsed.query('Trace == False').dy) / 1.349,
            "SigmaWreservedAll": iqr(dmapReserved.query('Trace == False').dy) / 1.349
        }
        dictkeys = [
            "N_Xused",
            "N_Xreserved",
            "N_Wused",
            "N_Wreserved",
            "Sigma_Xused",
            "Sigma_Xreserved",
            "Sigma_Wused",
            "Sigma_Wreserved",
            "Median_Xused",
            "Median_Xreserved",
            "Median_Wused",
            "Median_Wreserved",
            "pfsArmFluxMedian",
        ]
        for k in dictkeys:
            statistics[k] = np.array([])
        for f in arc_data.fiberId.unique():
            dmapUsedFiber = dmapUsed.query(f'fiberId == {f}')
            dmapUsedFiberNoTrace = dmapUsedFiber.query('Trace == False')
            dmapReservedFiber = dmapReserved.query(f'fiberId == {f}')
            dmapReservedFiberNoTrace = dmapReservedFiber.query('Trace == False')

            dictvalues = [
                len(dmapUsedFiber),
                len(dmapReservedFiber),
                len(dmapUsedFiberNoTrace),
                len(dmapReservedFiberNoTrace),
                iqr(dmapUsedFiber.dx) / 1.349,
                iqr(dmapReservedFiber.dx) / 1.349,
                iqr(dmapUsedFiberNoTrace.dy) / 1.349,
                iqr(dmapReservedFiberNoTrace.dy) / 1.349,
                np.median(dmapUsedFiber.dx),
                np.median(dmapReservedFiber.dx),
                np.median(dmapUsedFiberNoTrace.dy),
                np.median(dmapReservedFiberNoTrace.dy),
                np.median(pfsArm.flux[pfsArm.fiberId == f]),
            ]
            for k, v in zip(dictkeys, dictvalues):
                statistics[k] = np.append(statistics[k], v)

        return statistics

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
            self, detectorMap: Iterable[DetectorMap], arcLines: Iterable[ArcLineSet],
            pfsArm: Iterable[PfsArm]
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
            self.log.info("Fiber number ({}{}): {}".format(arm[aa], spectrograph, len(flist[aa])))
            self.log.info("Measured line ({}{}): {}".format(arm[aa], spectrograph, np.sum(measured[aa])))

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
                        difference[w].append(
                            detectorMap[0].findWavelength(fiberId=f, row=y[0]) - detectorMap[1].findWavelength(
                                fiberId=f, row=y[1])
                        )
        plt.figure()
        wcommon.sort()
        for w in wcommon:
            self.log.info(
                "{} nm ({} fibers, median={:.1e} nm, 1sigma={:.3f} nm)".format(
                    w, len(fibers[w]),
                    np.median(difference[w]),
                    iqr(difference[w]) / 1.349)
            )
            plt.scatter(fibers[w], difference[w], s=3,
                        label="{} nm ({} fibers, median={:.1e} nm, 1sigma={:.3f} nm)".format(w, len(fibers[w]),
                                                                                             np.median(difference[w]),
                                                                                             iqr(difference[
                                                                                                     w]) / 1.349),
                        )
        plt.legend(fontsize=7)
        plt.xlabel("fiberId")
        plt.ylabel("Wavelength difference ({}-{}) [nm]".format(arm[0], arm[1]))
        plt.savefig("overlapLines-{:06}-{}{}{}.png".format(visit, arm[0], arm[1], spectrograph))
        plt.close()

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


class DetectorMapQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapQaConnections):
    """Configuration for DetectorMapQaTask"""

    plotResidual = ConfigurableField(target=PlotResidualTask, doc="Plot the detector map residual.")
    overlapRegionLines = ConfigurableField(
        target=OverlapRegionLinesTask,
        doc="Plot the wavelength difference of the sky lines commonly detected in multiple arms."
    )


class DetectorMapQaRunner(TaskRunner):
    """Runner for DetectorMapQaTask"""

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Produce list of targets for DetectorMapQaTask

        We want to operate on all data within a single exposure at once.
        """
        exposures = defaultdict(lambda: defaultdict(list))
        for ref in parsedCmd.id.refList:
            visit = ref.dataId["visit"]
            spectrograph = ref.dataId["spectrograph"]
            exposures[visit][spectrograph].append(ref)
        return [(list(specs.values()), kwargs) for specs in exposures.values()]


class DetectorMapQaTask(CmdLineTask, PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapQaConfig
    _DefaultName = "detectorMapQa"
    RunnerClass = DetectorMapQaRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("plotResidual")
        self.makeSubtask("overlapRegionLines")
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

        detectorMapList = [
            [dataRef.get("detectorMap_used") for dataRef in specRefList] for specRefList in
            expSpecRefList
        ]
        arcLinesList = [
            [dataRef.get("arcLines") for dataRef in specRefList] for specRefList in expSpecRefList
        ]
        pfsArmList = [[dataRef.get("pfsArm") for dataRef in specRefList] for specRefList in expSpecRefList]
        return self.run(detectorMapList, arcLinesList, pfsArmList)

    def run(
            self,
            detectorMapList: Iterable[DetectorMap],
            arcLinesList: Iterable[ArcLineSet],
            pfsArmList: Iterable[PfsArm],
    ) -> Struct:
        """Generate detectorMapQa plots: 1) Residual of the adjustDetectorMap fitting, 2) Wavelength
        difference of the lines detected in multiple arms.

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
        """

        for detectorMap, arcLines, pfsArm in zip(detectorMapList, arcLinesList, pfsArmList):
            for dd, aa, pp in zip(detectorMap, arcLines, pfsArm):
                self.plotResidual.run(dd, aa, pp)
            arm = np.array([pp.identity.arm for pp in pfsArm])
            brId = np.logical_or(arm == "b", arm == "r")
            rnId = np.logical_or(arm == "r", arm == "n")
            if np.sum(brId) == 2:
                detectorMapBR, arcLinesBR, pfsArmBR = [], [], []
                for i in range(brId.size):
                    if brId[i]:
                        detectorMapBR.append(detectorMap[i])
                        arcLinesBR.append(arcLines[i])
                        pfsArmBR.append(pfsArm[i])
                self.overlapRegionLines.run(detectorMapBR, arcLinesBR, pfsArmBR)
            if np.sum(rnId) == 2:
                detectorMapRN, arcLinesRN, pfsArmRN = [], [], []
                for i in range(rnId.size):
                    if rnId[i]:
                        detectorMapRN.append(detectorMap[i])
                        arcLinesRN.append(arcLines[i])
                        pfsArmRN.append(pfsArm[i])
                self.overlapRegionLines.run(detectorMapRN, arcLinesRN, pfsArmRN)

    def _getMetadataName(self):
        return None
