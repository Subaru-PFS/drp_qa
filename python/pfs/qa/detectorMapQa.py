import pickle
from collections import defaultdict
from typing import Iterable

import lsstDebug
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
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
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from pfs.drp.stella import ArcLineSet, DetectorMap, PfsArm, ReferenceLineStatus
from scipy.stats import iqr


class PlotResidualConfig(Config):
    """Configuration for DetectorMapQaTask"""

    numKnots = Field(dtype=int, default=30, doc="Number of spline knots")


class PlotResidualTask(Task):
    """Task for QA of detectorMap."""

    ConfigClass = PlotResidualConfig
    _DefaultName = "plotResidual"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, detectorMap: DetectorMap, arcLines: Iterable[ArcLineSet], pfsArm: PfsArm) -> Struct:
        """QA of adjustDetectorMap.

        This script produces the plots and pickle files storing the fitting residual data of the adjustDetectorMap.

        Parameters
        ----------
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y
        arcLines: `Iterable[ArcLineSet]`
            Emission line measurements
        pfsArm: `PfsArm`

        Returns:
        ----------
        None

        Outputs:
        ----------
        1d plot: 'dmapQAPlot-{:06}-{}{}.png'
            The residuals of Xcenter and wavelength are plotted.
        2d plots: "dmapQAPlot2Dused-{:06}-{}{}.png", "dmapQAPlot2Dreserved-{:06}-{}{}.png"
            The residuals of Xcenter and wavelength are plotted in the array format.
        pickle data: "dmapQAStats-{:06}-{}{}.pickle"
            The fiberId, number of lines (detectormap_used, detectormap_reserved), medians and sigmas of the
            fitting residuals are stored in the dict format.
        """

        visit = pfsArm.identity.visit
        arm = pfsArm.identity.arm
        spectrograph = pfsArm.identity.spectrograph
        fmin, fmax = np.amin(arcLines.fiberId), np.amax(arcLines.fiberId)

        dmapUsed = (arcLines.status & ReferenceLineStatus.DETECTORMAP_USED) != 0
        dmapResearved = (arcLines.status & ReferenceLineStatus.DETECTORMAP_RESERVED) != 0

        measured = np.logical_not(np.isnan(arcLines.flux)) & np.logical_not(np.isnan(arcLines.x)) & np.logical_not(
            np.isnan(arcLines.y)) & np.logical_not(np.isnan(arcLines.xErr)) & np.logical_not(
            np.isnan(arcLines.yErr)) & np.logical_not(np.isnan(arcLines.fluxErr))

        flist = []
        for f in range(fmin, fmax + 1):
            notNan_f = (arcLines.fiberId == f) & measured
            if np.sum(notNan_f) > 0:
                flist.append(f)

        self.log.info("Fiber number: {}".format(len(flist)))
        self.log.info("Measured line: {}".format(np.sum(measured)))

        fig1 = plt.figure(figsize=(12, 10))
        ax1 = [plt.axes([0.08, 0.08, 0.37, 0.36]),
               plt.axes([0.46, 0.08, 0.07, 0.36]),
               plt.axes([0.08, 0.54, 0.37, 0.36]),
               plt.axes([0.46, 0.54, 0.07, 0.36]),
               plt.axes([0.58, 0.08, 0.37, 0.36]),
               plt.axes([0.58, 0.54, 0.37, 0.36])]
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
        fig3, ax3 = plt.subplots(1, 2, figsize=(12, 5))

        xrange = 0.2
        wrange = 0.03
        ps = 0.2
        quivLeng = 0.2
        showAllRange = False

        arcLinesMeasured = arcLines[measured]
        residualX = arcLinesMeasured.x - detectorMap.getXCenter(arcLinesMeasured.fiberId, arcLinesMeasured.y)
        residualW = arcLinesMeasured.wavelength - detectorMap.findWavelength(fiberId=arcLinesMeasured.fiberId,
                                                                             row=arcLinesMeasured.y)
        minw = np.amin(detectorMap.findWavelength(fiberId=arcLinesMeasured.fiberId, row=arcLinesMeasured.y))
        maxw = np.amax(detectorMap.findWavelength(fiberId=arcLinesMeasured.fiberId, row=arcLinesMeasured.y))
        bufw = (maxw - minw) * 0.02

        dmUsedMeasured = dmapUsed[measured]
        dmReservedMeasured = dmapResearved[measured]
        if showAllRange:  # self.config.showAllRange:
            residualXMax = max(np.amax(residualX[dmUsedMeasured]), np.amax(residualX[dmReservedMeasured]))
            residualXMin = min(np.amin(residualX[dmUsedMeasured]), np.amin(residualX[dmReservedMeasured]))
            residualWMax = max(np.amax(residualW[dmUsedMeasured]), np.amax(residualW[dmReservedMeasured]))
            residualWMin = min(np.amin(residualW[dmUsedMeasured]), np.amin(residualW[dmReservedMeasured]))
            yxmax = xrange * 3 if residualXMax < xrange * 3 else residualXMax * 1.05
            yxmin = -xrange * 3 if residualXMin > -xrange * 3 else residualXMin * 1.05
            ywmax = wrange * 3 if residualWMax < wrange * 3 else residualWMax * 1.05
            ywmin = -wrange * 3 if residualWMin > -wrange * 3 else residualWMin * 1.05
        else:
            yxmax = xrange * 3
            yxmin = -xrange * 3
            ywmax = wrange * 3
            ywmin = -wrange * 3
            largeX = residualX > yxmax
            smallX = residualX < yxmin
            largeW = residualW > ywmax
            smallW = residualW < ywmin

        statistics = {}
        dictkeys = ["N_Xused", "N_Xreserved", "N_Wused", "N_Wreserved", "Sigma_Xused", "Sigma_Xreserved",
                    "Sigma_Wused", "Sigma_Wreserved", "Median_Xused", "Median_Xreserved", "Median_Wused",
                    "Median_Wreserved", "pfsArmFluxMedian"]
        statistics["fiberId"] = np.array(flist)
        statistics["MedianXusedAll"] = np.median(residualX[dmUsedMeasured])
        statistics["MedianXreservedAll"] = np.median(residualX[dmReservedMeasured])
        statistics["MedianWusedAll"] = np.median(residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")])
        statistics["MedianWreservedAll"] = np.median(
            residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")])
        statistics["SigmaXusedAll"] = iqr(residualX[dmUsedMeasured]) / 1.349
        statistics["SigmaXreservedAll"] = iqr(residualX[dmReservedMeasured]) / 1.349
        statistics["SigmaWusedAll"] = iqr(residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")]) / 1.349
        statistics["SigmaWreservedAll"] = iqr(
            residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")]) / 1.349
        for k in dictkeys:
            statistics[k] = np.array([])
        for f in flist:
            xu = dmUsedMeasured & (arcLinesMeasured.fiberId == f)
            xr = dmReservedMeasured & (arcLinesMeasured.fiberId == f)
            wu = dmUsedMeasured & (arcLinesMeasured.fiberId == f) & (arcLinesMeasured.description != "Trace")
            wr = dmReservedMeasured & (arcLinesMeasured.fiberId == f) & (arcLinesMeasured.description != "Trace")
            dictvalues = [np.sum(xu), np.sum(xr), np.sum(wu), np.sum(wr), iqr(residualX[xu]) / 1.349,
                          iqr(residualX[xr]) / 1.349, iqr(residualW[wu]) / 1.349,
                          iqr(residualW[wr]) / 1.349, np.median(residualX[xu]), np.median(residualX[xr]),
                          np.median(residualW[wu]), np.median(residualW[wr])]
            dictvalues.append(np.median(pfsArm.flux[pfsArm.fiberId == f]))
            for k, v in zip(dictkeys, dictvalues):
                statistics[k] = np.append(statistics[k], v)
        statsFile = open("dmapQAStats-{:06}-{}{}.pickle".format(visit, arm, spectrograph), 'wb')
        pickle.dump(statistics, statsFile)
        statsFile.close()

        ax1[0].scatter(arcLinesMeasured.wavelength[dmUsedMeasured], residualX[dmUsedMeasured], s=ps, c="b",
                       label="DETECTORMAP_USED\n(median:{:.2e}, sigma:{:.2e})".format(
                           np.median(residualX[dmUsedMeasured]), iqr(residualX[dmUsedMeasured]) / 1.349))
        if not showAllRange:  # self.config.showAllRange:
            if np.sum(largeX) + np.sum(smallX) > 0:
                ax1[0].quiver(arcLinesMeasured.wavelength[dmUsedMeasured & largeX],
                              np.zeros(np.sum(dmUsedMeasured & largeX)) + yxmax - xrange * quivLeng, 0,
                              xrange * quivLeng,
                              label="Greater than {:.2f} in absolute value ({:.1e}%)".format(yxmax, np.sum(
                                  dmUsedMeasured & largeX) / np.sum(dmUsedMeasured) * 100), color="b", angles='xy',
                              scale_units='xy', scale=2)
                ax1[0].quiver(arcLinesMeasured.wavelength[dmUsedMeasured & smallX],
                              np.zeros(np.sum(dmUsedMeasured & smallX)) + yxmin + xrange * quivLeng, 0,
                              -xrange * quivLeng,
                              color="b", angles='xy', scale_units='xy', scale=2)
        ax1[0].scatter(arcLinesMeasured.wavelength[dmReservedMeasured], residualX[dmReservedMeasured], s=ps, c="r",
                       label="DETECTORMAP_RESERVED\n(median:{:.2e}, sigma:{:.2e})".format(
                           np.median(residualX[dmReservedMeasured]), iqr(residualX[dmReservedMeasured]) / 1.349))
        if not showAllRange:  # self.config.showAllRange:
            if np.sum(largeX) + np.sum(smallX) > 0:
                ax1[0].quiver(arcLinesMeasured.wavelength[dmReservedMeasured & largeX],
                              np.zeros(np.sum(dmReservedMeasured & largeX)) + yxmax - xrange * quivLeng, 0,
                              xrange * quivLeng,
                              label="Greater than {:.2f} in absolute value ({:.1e}%)".format(yxmax, (
                                      np.sum(dmReservedMeasured & largeX) + np.sum(
                                  dmReservedMeasured & smallX)) / np.sum(dmReservedMeasured) * 100), color="r",
                              angles='xy', scale_units='xy', scale=2)
                ax1[0].quiver(arcLinesMeasured.wavelength[dmReservedMeasured & smallX],
                              np.zeros(np.sum(dmReservedMeasured & smallX)) + yxmin + xrange * quivLeng, 0,
                              -xrange * quivLeng,
                              color="r", angles='xy',
                              scale_units='xy', scale=2)
        ax1[1].hist(residualX[dmUsedMeasured], color="b", range=(-xrange * 3, xrange * 3), bins=35,
                    orientation="horizontal")
        ax1[1].hist(residualX[dmReservedMeasured], color="r", range=(-xrange * 3, xrange * 3), bins=35,
                    orientation="horizontal")
        ax1[2].scatter(arcLinesMeasured.wavelength[dmUsedMeasured & (arcLinesMeasured.description != "Trace")],
                       residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")], s=ps, c="b",
                       label="DETECTORMAP_USED\n(median:{:.2e}, sigma:{:.2e})".format(
                           np.median(residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")]),
                           iqr(residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")]) / 1.349))
        if not showAllRange:  # self.config.showAllRange:
            if np.sum(largeW) + np.sum(smallW) > 0:
                ax1[2].quiver(arcLinesMeasured.wavelength[dmUsedMeasured & largeW],
                              np.zeros(np.sum(dmUsedMeasured & largeW)) + ywmax - wrange * quivLeng, 0,
                              wrange * quivLeng,
                              label="Greater than {:.2f} in absolute value ({:.1e}%)".format(ywmax, (np.sum(
                                  dmUsedMeasured & largeW) + np.sum(
                                  dmUsedMeasured & smallW)) / np.sum(dmUsedMeasured) * 100), color="b", angles='xy',
                              scale_units='xy', scale=2)
                ax1[2].quiver(arcLinesMeasured.wavelength[dmUsedMeasured & smallW],
                              np.zeros(np.sum(dmUsedMeasured & smallW)) + ywmin + wrange * quivLeng, 0,
                              wrange * quivLeng,
                              color="b", angles='xy', scale_units='xy', scale=2)
        ax1[2].scatter(arcLinesMeasured.wavelength[dmReservedMeasured & (arcLinesMeasured.description != "Trace")],
                       residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")], s=ps, c="r",
                       label="DETECTORMAP_RESERVED\n(median:{:.2e}, sigma:{:.2e})".format(
                           np.median(residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")]),
                           iqr(residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")]) / 1.349))
        if not showAllRange:  # self.config.showAllRange:
            if np.sum(largeW) + np.sum(smallW) > 0:
                ax1[2].quiver(arcLinesMeasured.wavelength[dmReservedMeasured & largeW],
                              np.zeros(np.sum(dmReservedMeasured & largeW)) + ywmax - wrange * quivLeng, 0,
                              wrange * quivLeng,
                              label="Greater than {:.2f} in absolute value ({:.1e}%)".format(ywmax, (
                                      np.sum(dmReservedMeasured & largeW) + np.sum(
                                  dmReservedMeasured & smallW)) / np.sum(dmReservedMeasured) * 100), color="r",
                              angles='xy', scale_units='xy', scale=2)
                ax1[2].quiver(arcLinesMeasured.wavelength[dmReservedMeasured & smallW],
                              np.zeros(np.sum(dmReservedMeasured & smallW)) + ywmin + wrange * quivLeng, 0,
                              -wrange * quivLeng,
                              color="r", angles='xy', scale_units='xy', scale=2)
        ax1[3].hist(residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")], color="b",
                    range=(-wrange * 3, wrange * 3), bins=35, orientation="horizontal")
        ax1[3].hist(residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")], color="r",
                    range=(-wrange * 3, wrange * 3), bins=35, orientation="horizontal")
        ax1[4].errorbar(statistics["fiberId"], statistics["Median_Xused"], yerr=statistics["Sigma_Xused"], fmt="bo",
                        markersize=2)
        ax1[5].errorbar(statistics["fiberId"], statistics["Median_Wused"], yerr=statistics["Sigma_Wused"], fmt="bo",
                        markersize=2)

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

        img1 = ax2[0].scatter(arcLinesMeasured.x[dmUsedMeasured], arcLinesMeasured.y[dmUsedMeasured], s=ps,
                              c=residualX[dmUsedMeasured], vmin=-xrange, vmax=xrange,
                              cmap=cm.coolwarm)
        img2 = ax2[1].scatter(arcLinesMeasured.x[dmUsedMeasured & (arcLinesMeasured.description != "Trace")],
                              arcLinesMeasured.y[dmUsedMeasured & (arcLinesMeasured.description != "Trace")], s=ps,
                              c=residualW[dmUsedMeasured & (arcLinesMeasured.description != "Trace")], vmin=-wrange,
                              vmax=wrange,
                              cmap=cm.coolwarm)
        img3 = ax3[0].scatter(arcLinesMeasured.x[dmReservedMeasured], arcLinesMeasured.y[dmReservedMeasured], s=ps,
                              c=residualX[dmReservedMeasured], vmin=-xrange, vmax=xrange, cmap=cm.coolwarm)
        img4 = ax3[1].scatter(arcLinesMeasured.x[dmReservedMeasured & (arcLinesMeasured.description != "Trace")],
                              arcLinesMeasured.y[dmReservedMeasured & (arcLinesMeasured.description != "Trace")], s=ps,
                              c=residualW[dmReservedMeasured & (arcLinesMeasured.description != "Trace")], vmin=-wrange,
                              vmax=wrange, cmap=cm.coolwarm)

        cbar1 = fig2.colorbar(img1, ax=ax2[0], aspect=50, pad=0.08, shrink=1, orientation='vertical')
        cbar2 = fig2.colorbar(img2, ax=ax2[1], aspect=50, pad=0.08, shrink=1, orientation='vertical')
        cbar3 = fig2.colorbar(img3, ax=ax3[0], aspect=50, pad=0.08, shrink=1, orientation='vertical')
        cbar4 = fig2.colorbar(img4, ax=ax3[1], aspect=50, pad=0.08, shrink=1, orientation='vertical')

        ax2[0].set_title("X center, unit=pix")
        ax2[1].set_title("Wavelength, unit=nm")
        ax2[0].set_xlim(0, 4096)
        ax2[1].set_xlim(0, 4096)
        ax2[0].set_ylim(0, 4176)
        ax2[1].set_ylim(0, 4176)
        ax3[0].set_title("Detector map residual (X center, unit=pix)")
        ax3[1].set_title("Detector map residual (wavelength, unit=nm)")
        ax3[0].set_xlim(0, 4096)
        ax3[1].set_xlim(0, 4096)
        ax3[0].set_ylim(0, 4176)
        ax3[1].set_ylim(0, 4176)

        fig1.suptitle("Detector map residual (visit={}, arm={}, spectrograph={})".format(visit, arm, spectrograph))
        fig2.suptitle("DECTORMAP_USED residual (visit={}, arm={}, spectrograph={})".format(visit, arm, spectrograph))
        fig3.suptitle(
            "DECTORMAP_RESERVED residual (visit={}, arm={}, spectrograph={})".format(visit, arm, spectrograph))

        fig1.savefig("dmapQAPlot-{:06}-{}{}.png".format(visit, arm, spectrograph), format="png")
        fig1.clf()
        fig2.savefig("dmapQAPlot2Dused-{:06}-{}{}.png".format(visit, arm, spectrograph), format="png")
        fig2.clf()
        fig3.savefig("dmapQAPlot2Dreserved-{:06}-{}{}.png".format(visit, arm, spectrograph), format="png")
        fig3.clf()

        # There is no output in this template, so I can't give an example write the output here
        return Struct()

    @classmethod
    def _makeArgumentParser(cls) -> ArgumentParser:
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(
            name="--id", datasetType="arcLines", level="Visit", help="data IDs, e.g. --id exp=12345"
        )
        return parser


class OverlapRegionLinesConfig(Config):
    """Configuration for DetectorMapQaTask"""

    numKnots = Field(dtype=int, default=30, doc="Number of spline knots")


class OverlapRegionLinesTask(Task):
    """Task for QA of detectorMap"""

    ConfigClass = OverlapRegionLinesConfig
    _DefaultName = "overlapRegionLines"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, detectorMap: Iterable[DetectorMap], arcLines: Iterable[ArcLineSet],
            pfsArm: Iterable[PfsArm]) -> Struct:
        """QA of adjustDetectorMap.

        This script produces the plots and pickle files storing the fitting residual data of the adjustDetectorMap.

        Parameters
        ----------
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y
        arcLines: `Iterable[ArcLineSet]`
            Emission line measurements
        pfsArm: `PfsArm`

        Returns:
        ----------
        None

        Outputs:
        ----------
        1d plot: 'dmapQAPlot-{:06}-{}{}.png'
            The residuals of Xcenter and wavelength are plotted.
        2d plots: "dmapQAPlot2Dused-{:06}-{}{}.png", "dmapQAPlot2Dreserved-{:06}-{}{}.png"
            The residuals of Xcenter and wavelength are plotted in the array format.
        pickle data: "dmapQAStats-{:06}-{}{}.pickle"
            The fiberId, number of lines (detectormap_used, detectormap_reserved), medians and sigmas of the
            fitting residuals are stored in the dict format.
        """
        visit = pfsArm[0].identity.visit
        arm = [pp.identity.arm for pp in pfsArm]
        spectrograph = pfsArm[0].identity.spectrograph
        fmin, fmax = [np.amin(aa.fiberId) for aa in arcLines], [np.amax(aa.fiberId) for aa in arcLines]

        dmapUsed = [(aa.status & ReferenceLineStatus.DETECTORMAP_USED) != 0 for aa in arcLines]
        dmapResearved = [(aa.status & ReferenceLineStatus.DETECTORMAP_RESERVED) != 0 for aa in arcLines]
        measured = [np.logical_not(np.isnan(aa.flux)) & np.logical_not(np.isnan(aa.x)) & np.logical_not(
            np.isnan(aa.y)) & np.logical_not(np.isnan(aa.xErr)) & np.logical_not(
            np.isnan(aa.yErr)) & np.logical_not(np.isnan(aa.fluxErr)) for aa in arcLines]

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
        OIlines = ['630.20', '636.55']
        # pp = PdfPages("pfsArms-{:06}-{}{}{}.pdf".format(visit, arm[0], arm[1], spectrograph))
        for f in fcommon:
            b0 = (arcLines[0].fiberId == f) & measured[0]  # & (dmapUsed[0] | dmapResearved[0])
            b1 = (arcLines[1].fiberId == f) & measured[1]  # & (dmapUsed[1] | dmapResearved[1])
            wav0 = set(arcLines[0][b0].wavelength)
            wav1 = set(arcLines[1][b1].wavelength)
            wav = list(wav0 & wav1)
            # plt.plot(pfsArm[0][pfsArm[0].fiberId == f].wavelength[0], pfsArm[0][pfsArm[0].fiberId == f].flux[0],
            #          "b")
            # plt.plot(pfsArm[1][pfsArm[1].fiberId == f].wavelength[0], pfsArm[1][pfsArm[1].fiberId == f].flux[0],
            #          "r")
            if len(wav) > 0:
                wav.sort()
                for w in wav:
                    if "{:.2f}".format(w) in OIlines:
                        if (not w in wcommon):
                            wcommon.append(w)
                            fibers[w] = []
                            difference[w] = []
                        y = [aa[(aa.fiberId == f) & (aa.wavelength == w)].y[0] for aa in arcLines]
                        fibers[w].append(f)
                        difference[w].append(
                            detectorMap[0].findWavelength(fiberId=f, row=y[0]) - detectorMap[1].findWavelength(
                                fiberId=f, row=y[1]))
                        plt.plot([w, w], [0, 5000], 'k')
                        # b0w = b0 & (arcLines[0].wavelength == w)
                        # b1w = b1 & (arcLines[1].wavelength == w)
                        # print(f, w, '{:.1f} ({:.1e}) {}/{} {:.2f}({:.2f})/{:.2f}({:.2f})'.format(
                        #     w, difference[w][-1], arcLines[0][b0w].status[0], arcLines[1][b1w].status[0],
                        #     arcLines[0][b0w].flux[0], arcLines[0][b0w].fluxErr[0], arcLines[1][b1w].flux[0],
                        #     arcLines[1][b1w].fluxErr[0]))
                        # plt.text(w,1000, '{:.1f} ({:.1e}) {}/{} {:.2f}({:.2f})/{:.2f}({:.2f})'.format(
                        #     w, difference[w][-1], arcLines[0][b0w].status[0], arcLines[1][b1w].status[0],
                        #     arcLines[0][b0w].flux[0], arcLines[0][b0w].fluxErr[0], arcLines[1][b1w].flux[0],
                        #     arcLines[1][b1w].fluxErr[0]), fontsize=6, rotation=90, va='bottom', ha='left')

            # plt.title("fiberId={}".format(f))
            # plt.xlabel("Wavelength [nm]")
            # plt.ylabel("Flux")
            # plt.xlim(620, 660)
            # plt.savefig(pp, format="pdf")
            # plt.clf()
        # pp.close()
        plt.figure()
        wcommon.sort()
        for w in wcommon:
            print('{}: {} nm ({} fibers, median={:.1e} nm, 1sigma={:.3f} nm)'.format(f, w, len(fibers[w]),
                                                                                     np.median(difference[w]),
                                                                                     iqr(difference[w]) / 1.349))
            plt.scatter(fibers[w], difference[w], s=3,
                        label='{} nm ({} fibers, median={:.1e} nm, 1sigma={:.3f} nm)'.format(w, len(fibers[w]),
                                                                                             np.median(difference[w]),
                                                                                             iqr(difference[
                                                                                                     w]) / 1.349))
        plt.legend(fontsize=7)
        plt.xlabel('fiberId')
        plt.ylabel('Wavelength difference (b-r) [nm]')
        plt.savefig("overlapLines-{:06}-{}{}{}.png".format(visit, arm[0], arm[1], spectrograph))

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

    # Please list other inputs and outputs
    # in drp_stella/python/pfs/drp/stella/pipelines/
    # It's OK if you copy and paste from scripts


class DetectorMapQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapQaConnections):
    """Configuration for DetectorMapQaTask"""
    plotResidual = ConfigurableField(target=PlotResidualTask, doc="Plot the detector map residual.")
    overlapRegionLines = ConfigurableField(target=OverlapRegionLinesTask, doc="Plot the detector map residual.")

    nanikanoParam = Field(
        dtype=float,
        default=1.0,
        doc="Parameter for a certain use",  # "Expand y range to show all data points.",
    )


class DetectorMapQaRunner(TaskRunner):
    """Runner for DetectorMapQaTask"""

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Produce list of targets for MergeArmsTask

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

    # (Gen3)
    # this is called when you run a pipeline with the pipetask command
    # It's normal to let run() do all the non-interface work
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

    # (Gen2)
    # On Gen2 this will be the interface.
    # Let run() do all the non-interface work.
    def runDataRef(self, expSpecRefList) -> Struct:
        """Calls ``self.run()``

        Parameters
        ----------
        expSpecRefList : iterable of iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for each sensor, grouped by spectrograph.

        Returns
        -------
        name : `type`
            Description.
        """

        # detectorMap = dataRef.get("detectorMap_used")
        # arcLines = dataRef.get("arcLines")
        # pfsArm = dataRef.get("pfsArm")
        detectorMapList = [[dataRef.get("detectorMap_used") for dataRef in specRefList] for specRefList in
                           expSpecRefList]
        arcLinesList = [[dataRef.get("arcLines") for dataRef in specRefList] for specRefList in expSpecRefList]
        pfsArmList = [[dataRef.get("pfsArm") for dataRef in specRefList] for specRefList in expSpecRefList]
        return self.run(detectorMapList, arcLinesList, pfsArmList)

    def run(self, detectorMapList: DetectorMap, arcLinesList: Iterable[ArcLineSet], pfsArmList: PfsArm) -> Struct:
        for detectorMap, arcLines, pfsArm in zip(detectorMapList, arcLinesList, pfsArmList):
            for dd, aa, pp in zip(detectorMap, arcLines, pfsArm):
                self.plotResidual.run(dd, aa, pp)
            # self.overlapRegionLines.run(detectorMap, arcLines, pfsArm)

    def _getMetadataName(self):
        return None
