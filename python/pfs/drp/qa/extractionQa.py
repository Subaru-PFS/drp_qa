import dataclasses
import itertools
import math
from typing import Dict

import lsst.afw.display as afwDisplay
import lsstDebug
import matplotlib
import numpy as np
from lsst.afw.image import ExposureF, ImageF, MaskedImageF
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
    Output as OutputConnection,
    PrerequisiteInput as PrerequisiteConnection,
)
from matplotlib import pyplot as plt
from pfs.datamodel import FiberStatus, PfsConfig, TargetType
from pfs.drp.stella import DetectorMap, FiberProfileSet, PfsArm, SpectrumSet
from pfs.drp.stella.utils import addPfsCursor
from scipy.optimize import curve_fit
from scipy.stats import iqr

from pfs.drp.qa.storageClasses import MultipagePdfFigure, QaDict
from pfs.drp.qa.utils.math import gaussianFixedWidth, gaussian_func


@dataclasses.dataclass
class StatsPerFiber:
    """Statistics for a fiber.

    Parameters
    ----------
    chi2 : `float`
        XXXXX
    im_ave : `float`
        XXXXX
    x_ave : `float`
        XXXXX
    chi_f : `np.ndarray` of `float`, shape ``(N, M)``
        XXXXX
    mask_f : `np.ndarray` of `int`, shape ``(N, M)``
        XXXXX
    """

    chi2: float
    im_ave: float
    x_ave: float
    chi_f: np.ndarray
    mask_f: np.ndarray


class ExtractionQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Connections for ExtractionQaTask"""

    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
    )
    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Position and shape of fibers",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    detectorMap = InputConnection(
        name="detectorMap",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra from arm",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    postISRCCD = InputConnection(
        name="postISRCCD",
        doc="Calibrated visit, optionally sky-subtracted",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    extQaStats = OutputConnection(
        name="extQaStats",
        doc="Summary plots. Results of the residual analysis of extraction are plotted.",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    extQaImage = OutputConnection(
        name="extQaImage",
        doc="Detail plots. postISRCCD, residual, and chi images and the comparison of the postISRCCD"
        "profile and fiberProfiles are plotted for some fibers with bad extraction quality.",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    extQaImage_pickle = OutputConnection(
        name="extQaImage_pickle",
        doc="Statistics of the residual analysis.",
        storageClass="QaDict",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class ExtractionQaConfig(PipelineTaskConfig, pipelineConnections=ExtractionQaConnections):
    """Configuration for ExtractionQaTask"""

    fixWidth = Field(dtype=bool, default=False, doc="Fix the widths during Gaussian fitting")
    rowNum = Field(dtype=int, default=200, doc="Number of rows picked up for profile analysis")
    thresError = Field(dtype=float, default=0.1, doc="Threshold of the fitting error")
    thresChi = Field(dtype=float, default=1.5, doc="Threshold for chi standard deviation")
    fiberWidth = Field(dtype=int, default=3, doc="Half width of a fiber region (pix)")
    fitWidth = Field(dtype=int, default=3, doc="Half width  of a fitting region (pix)")
    plotWidth = Field(dtype=int, default=15, doc="Half width  of plot (pix)")
    plotFiberNum = Field(dtype=int, default=20, doc="Maximum fiber number of detailed plots")
    figureDpi = Field(dtype=int, default=72, doc="resolution of plot for residual")


class ExtractionQaTask(PipelineTask):
    """Task for QA of extraction"""

    ConfigClass = ExtractionQaConfig
    _DefaultName = "extractionQa"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        inputs = butlerQC.get(inputRefs)

        try:
            # Perform the actual processing.
            outputs = self.run(**inputs)
        except ValueError as e:
            self.log.error(e)
        else:
            butlerQC.put(outputs, outputRefs)

    def run(
        self,
        postISRCCD: ExposureF,
        fiberProfiles: FiberProfileSet,
        detectorMap: DetectorMap,
        pfsArm: PfsArm,
        pfsConfig: PfsConfig,
    ) -> Struct:
        """QA of extraction by analyzing the residual image.

        Parameters
        ----------
        postISRCCD : `ExposureF`
            Exposure data
        fiberProfiles : `FiberProfileSet`
            Profiles of each fiber.
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y
        pfsArm : `PfsArm`
            Extracted spectra from arm.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers.

        Returns
        -------
        extQaStats : `MultipagePdfFigure`
            Summary plots.
            Results of the residual analysis of extraction are plotted.
        extQaImage : `MultipagePdfFigure`
            Detail plots.
            postISRCCD, residual, and chi images and the comparison of the postISRCCD
            profile and fiberProfiles are plotted for some fibers with bad
            extraction quality.
        extQaImage_pickle : `QaDict`
            Data to be pickled.
            Statistics of the residual analysis.
        """
        self.log.info("Extraction QA on %s", pfsArm.identity)

        matplotlib.use("agg")
        afwDisplay.setDefaultBackend("matplotlib")

        visit = pfsArm.identity.visit
        arm = pfsArm.identity.arm
        spectrograph = pfsArm.identity.spectrograph
        dataId = dict(visit=visit, spectrograph=spectrograph, arm=arm)

        spectra = SpectrumSet.fromPfsArm(pfsArm)
        traces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap)
        image = spectra.makeImage(postISRCCD.getDimensions(), traces)

        subtracted = postISRCCD.clone()
        subtracted.image -= image
        divided = subtracted.clone()
        divided.image /= postISRCCD.image
        chiimage = subtracted.clone()
        chiimage.image.array /= np.sqrt(postISRCCD.variance.array)

        msk = (pfsConfig.spectrograph == dataId["spectrograph"]) * (pfsConfig.fiberStatus == FiberStatus.GOOD)
        fiberIds = pfsConfig[msk].fiberId
        targetMask = {t: pfsConfig[msk].targetType == t for t in TargetType}

        data = chiimage.maskedImage
        xa = []
        chiSquare = []
        chiAve = []
        chiMedian = []
        chiStd = []
        chiPeak = []
        pfsArmAve = []
        chiAveSpec = []

        for fiberId in fiberIds:
            stats = self.getStatsPerFiber(data, detectorMap, fiberId, xwin=self.config.fiberWidth)
            stats.chi_f[stats.mask_f != 0] = math.nan
            chiSquare.append(stats.chi2)
            chiAve.append(np.average(stats.chi_f[stats.mask_f == 0]))
            chiMedian.append(np.median(stats.chi_f[stats.mask_f == 0]))
            chiStd.append(iqr(stats.chi_f[stats.mask_f == 0]) / 1.349)
            chiPeak.append(
                np.average(
                    stats.chi_f[:, self.config.fiberWidth][stats.mask_f[:, self.config.fiberWidth] == 0]
                )
            )
            pfsArmAve.append(np.average(pfsArm[pfsArm.fiberId == fiberId].flux[0]))
            chiAveSpec.append(np.average(stats.chi_f, axis=1))
            xa.append(stats.x_ave)

        ymin = 0
        ymax = data.getDimensions()[1]
        yo = np.arange(ymin, ymax).astype(np.float64)

        plotNum = min(fiberIds.size, self.config.plotFiberNum)
        thresPlot = max(self.config.thresChi, sorted(chiStd)[-plotNum])
        ys = np.arange(ymin, ymax, (ymax - ymin) / self.config.rowNum).astype("int32")
        xarray = []
        yarray = []
        idarray = []
        dxarray = []
        dwarray = []
        qaStatsPdf: MultipagePdfFigure | None = None
        if any(np.array(chiStd) > self.config.thresChi):
            qaStatsPdf = MultipagePdfFigure()
            for i, fiberId in enumerate(fiberIds):
                # Note: In the script version, this line is
                # Note: if chiStd[i] >= thresChi or pfsArmAve[i] > thresPfsArm:
                if chiStd[i] >= self.config.thresChi:
                    xo = detectorMap.getXCenter(fiberId, yo)
                    xint = xo.astype("int32")
                    centerdif = []
                    widthdif = []
                    ydif = []
                    failNum = 0
                    for j in range(self.config.rowNum):
                        yssub = ys[j]
                        xssub = xint[yssub]
                        xcoordNarrow = np.arange(
                            max(xssub - self.config.fitWidth, 0),
                            min(xssub + self.config.fitWidth + 1, data.getDimensions()[0]),
                        )
                        pfsArmCutNarrow = image.array[
                            yssub, xssub - self.config.fitWidth : xssub + self.config.fitWidth + 1
                        ]
                        calExpCutNarrow = postISRCCD.image.array[
                            yssub, xssub - self.config.fitWidth : xssub + self.config.fitWidth + 1
                        ]
                        fitresult = self.fitGaussiansToPfsArmCutAndCalExpCut(
                            xcoordNarrow,
                            pfsArmCutNarrow,
                            calExpCutNarrow,
                            xo[yssub],
                        )
                        if fitresult.ok:
                            pfsArmCenter = fitresult.pfsArmCenter
                            pfsArmWidth = fitresult.pfsArmWidth
                            calExpCenter = fitresult.calExpCenter
                            calExpWidth = fitresult.calExpWidth
                            centerdif.append(pfsArmCenter - calExpCenter)
                            ydif.append(yssub)
                            widthdif.append(2 * (pfsArmWidth - calExpWidth) / (pfsArmWidth + calExpWidth))
                            xarray.append(xssub)
                            yarray.append(yssub)
                            idarray.append(fiberId)
                        else:
                            failNum += 1

                    if chiStd[i] >= thresPlot:
                        self.drawStats(
                            qaStatsPdf,
                            dataId,
                            image,
                            postISRCCD,
                            subtracted,
                            chiimage,
                            fiberId,
                            xo,
                            yo,
                            xa[i],
                            chiAveSpec[i],
                            centerdif,
                            widthdif,
                            ydif,
                        )

                    dxarray += centerdif
                    dwarray += widthdif

        pfsArmAve = np.array(pfsArmAve)
        chiSquare = np.array(chiSquare)
        chiAve = np.array(chiAve)
        chiMedian = np.array(chiMedian)
        chiStd = np.array(chiStd)
        chiPeak = np.array(chiPeak)
        xa = np.array(xa)

        qaStats = QaDict(
            {
                "fiberIds": fiberIds,
                "xa": xa,
                "pfsArmAve": pfsArmAve,
                "targetMask": targetMask,
                "chiSquare": chiSquare,
                "chiAve": chiAve,
                "chiMedian": chiMedian,
                "chiStd": chiStd,
                "chiPeak": chiPeak,
                "Xarray": xarray,
                "Yarray": yarray,
                "dx": dxarray,
                "dsigma": dwarray,
                "fiberIDarray": idarray,
            }
        )

        qaImagePdf = self.makeImagePdf(qaStats, dataId, detectorMap, chiimage)

        if qaStatsPdf is None:
            raise ValueError("qaStatsPdf is empty.")

        return Struct(
            extQaStats=qaStatsPdf,
            extQaImage=qaImagePdf,
            extQaImage_pickle=qaStats,
        )

    def drawStats(
        self,
        qaStatsPdf: MultipagePdfFigure,
        dataId: dict,
        image: ImageF,
        postISRCCD: ExposureF,
        subtracted: ExposureF,
        chiimage: ExposureF,
        fiberId: int,
        xo: np.ndarray,
        yo: np.ndarray,
        xa: float,
        chiAveSpec: np.ndarray,
        centerdif: np.ndarray,
        widthdif: np.ndarray,
        ydif: np.ndarray,
    ) -> None:
        """Draw figures on new pages of ``qaStatsPdf``.

        Parameters
        ----------
        qaStatsPdf : `MultipagePdfFigure`
            PDF to which to append new pages.
        dataId : `dict`
            Data ID. Required keys are: "visit", "arm", "spectrograph".
        image : `ImageF`
            XXXXX
        postISRCCD : `ExposureF`
            XXXXX
        subtracted : `ExposureF`
            XXXXX
        chiimage : `ExposureF`
            XXXXX
        fiberId : `int`
            XXXXX
        xo : `np.ndarray`
            XXXXX
        yo : `np.ndarray`
            XXXXX
        xa : `float`
            XXXXX
        chiAveSpec : `np.ndarray`
            XXXXX
        centerdif : `np.ndarray`
            XXXXX
        widthdif : `np.ndarray`
            XXXXX
        ydif : `np.ndarray`
            XXXXX
        """
        ymin = 0
        ymax = chiimage.getDimensions()[1]

        fig, ax = plt.subplots(1, 6, figsize=(12, 7))
        plt.subplots_adjust(wspace=0.5)
        plt.sca(ax[0])
        disp = afwDisplay.Display(fig)
        disp.scale("asinh", "zscale", Q=1)
        disp.setMaskPlaneColor("REFLINE", afwDisplay.IGNORE)
        disp.mtv(postISRCCD[int(xa) - 10 : int(xa) + 11, :])
        ax[0].plot(xo, yo, "r", alpha=0.8)
        ax[0].set_xlim(xa - 10, xa + 10)
        ax[0].set_aspect("auto")
        ax[0].set_ylabel("Y (pix)")
        ax[0].set_title("postISRCCD")

        plt.sca(ax[1])
        disp = afwDisplay.Display(fig)
        disp.setMaskPlaneColor("REFLINE", afwDisplay.BLACK)
        disp.setImageColormap("coolwarm")
        disp.scale("asinh", "zscale", Q=1)
        disp.mtv(subtracted[int(xa) - 10 : int(xa) + 11, :])
        ax[1].plot(xo, yo, "k", alpha=0.8)
        ax[1].set_xlim(xa - 10, xa + 10)
        ax[1].set_aspect("auto")
        ax[1].tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
        ax[1].set_xlabel("X (pix)")
        ax[1].set_title("Residual")

        plt.sca(ax[2])
        disp = afwDisplay.Display(fig)
        disp.scale("linear", -5, 5, Q=1)
        disp.setMaskPlaneColor("REFLINE", afwDisplay.BLACK)
        disp.setImageColormap("coolwarm")
        disp.mtv(chiimage[int(xa) - 10 : int(xa) + 11, :])
        ax[2].plot(xo, yo, "k", alpha=0.8)
        ax[2].set_xlim(xa - 10, xa + 10)
        ax[2].set_aspect("auto")
        ax[2].tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
        ax[2].set_title("Chi")

        ax[3].scatter(chiAveSpec, yo, s=3)
        ax[3].set_ylim(ymin, ymax)
        ax[3].plot([0, 0], [ymin, ymax], "0.8")
        ax[3].set_xlabel("Chi at each row")
        ax[3].tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
        ax[3].set_title("Chi")

        ax[4].scatter(centerdif, ydif, s=3)
        ax[4].set_ylim(ymin, ymax)
        ax[4].plot([0, 0], [ymin, ymax], "0.8")
        ax[4].set_xlabel("dx")
        ax[4].tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
        ax[4].set_title("Peak center diff.")

        ax[5].scatter(widthdif, ydif, s=3)
        ax[5].set_ylim(ymin, ymax)
        ax[5].plot([0, 0], [ymin, ymax], "0.8")
        ax[5].set_xlabel("d$\\sigma$/$\\sigma$")
        ax[5].tick_params(labelbottom=True, labelleft=False, labelright=True, labeltop=False)
        ax[5].set_title("Width diff.")

        fig.suptitle(
            "visit={:d} arm={:s} spectrograph={:d}\nf={:d}, X={:.1f}".format(
                dataId["visit"], dataId["arm"], dataId["spectrograph"], fiberId, xa
            ),
            fontsize=12,
        )

        qaStatsPdf.append(fig)
        plt.close(fig)

        numPanels = 3
        ysplot = np.arange(ymin, ymax, (ymax - ymin) / (numPanels * numPanels + 1)).astype("int32")
        xint = xo.astype("int32")

        fig, ax = plt.subplots(numPanels, numPanels, figsize=(12, 7))
        for j in range(numPanels):
            for k in range(numPanels):
                yssub = ysplot[k + j * numPanels + 1]
                xssub = xint[yssub]
                xcoord = np.arange(
                    max(xssub - self.config.plotWidth, 0),
                    min(xssub + self.config.plotWidth + 1, chiimage.getDimensions()[0]),
                )
                pfsArmCut = image.array[
                    yssub, xssub - self.config.plotWidth : xssub + self.config.plotWidth + 1
                ]
                calExpCut = postISRCCD.image.array[
                    yssub, xssub - self.config.plotWidth : xssub + self.config.plotWidth + 1
                ]
                xcoordNarrow = np.arange(
                    max(xssub - self.config.fitWidth, 0),
                    min(xssub + self.config.fitWidth + 1, chiimage.getDimensions()[0]),
                )
                pfsArmCutNarrow = image.array[
                    yssub, xssub - self.config.fitWidth : xssub + self.config.fitWidth + 1
                ]
                calExpCutNarrow = postISRCCD.image.array[
                    yssub, xssub - self.config.fitWidth : xssub + self.config.fitWidth + 1
                ]

                fitresult = self.fitGaussiansToPfsArmCutAndCalExpCut(
                    xcoordNarrow,
                    pfsArmCutNarrow,
                    calExpCutNarrow,
                    xo[yssub],
                )
                pfsArmCenter = fitresult.pfsArmCenter
                pfsArmWidth = fitresult.pfsArmWidth
                calExpCenter = fitresult.calExpCenter
                calExpWidth = fitresult.calExpWidth

                ax[j][k].plot(xcoord, np.zeros(xcoord.shape), "k--")
                ax[j][k].step(
                    xcoord,
                    pfsArmCut,
                    label="pfsArm\n(x={:.2f}, $\\sigma$={:.2f})".format(pfsArmCenter, pfsArmWidth),
                    color="b",
                )
                ax[j][k].step(
                    xcoord,
                    calExpCut,
                    label="postISRCCD\n(x={:.2f}, $\\sigma$={:.2f})".format(calExpCenter, calExpWidth),
                    color="k",
                )
                ax[j][k].step(
                    xcoord,
                    subtracted.image.array[
                        yssub,
                        xssub - self.config.plotWidth : xssub + self.config.plotWidth + 1,
                    ]
                    * 5,
                    label="Residual*5",
                    color="r",
                )
                ypeak = np.amax(
                    image.array[yssub, xssub - self.config.fitWidth : xssub + self.config.fitWidth + 1]
                )
                ax[j][k].plot(
                    [xo[yssub], xo[yssub]],
                    [-ypeak / 10 * 3, ypeak * 1.5],
                    "b--",
                    label="Trace",
                )
                ax[j][k].set_ylim(-ypeak / 10 * 3, ypeak * 1.5)
                ax[j][k].set_title(
                    "Y={} (dx={:.1e}, d$\\sigma$={:.1e})".format(
                        yssub, calExpCenter - pfsArmCenter, calExpWidth - pfsArmWidth
                    ),
                    fontsize=8,
                )
                ax[j][k].legend(fontsize=4)
                labelbottom = False if j != 3 else True
                ax[j][k].tick_params(
                    labelbottom=labelbottom, labelleft=False, labelright=False, labeltop=False
                )
        fig.suptitle(
            "visit={:d} arm={:s} spectrograph={:d}\nf={:d}, X={:.1f}".format(
                dataId["visit"], dataId["arm"], dataId["spectrograph"], fiberId, xa
            ),
            fontsize=12,
        )
        qaStatsPdf.append(fig)
        plt.close(fig)

    def makeImagePdf(
        self, qaStats: QaDict, dataId: dict, detectorMap: DetectorMap, chiimage: ExposureF
    ) -> MultipagePdfFigure:
        """Make ``extQaImage``

        Parameters
        ----------
        qaStats : `QaDict`
            XXXXX
        dataId : `dict`
            Data ID. Required keys are: "visit", "arm", "spectrograph".
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y
        chiimage : `ExposureF`
            XXXXX

        Returns
        -------
        extQaImage : `MultipagePdfFigure`
            XXXXX
        """
        fiberIds = qaStats["fiberIds"]
        xa = qaStats["xa"]
        pfsArmAve = qaStats["pfsArmAve"]
        targetMask = qaStats["targetMask"]
        chiSquare = qaStats["chiSquare"]
        chiAve = qaStats["chiAve"]
        chiMedian = qaStats["chiMedian"]
        chiStd = qaStats["chiStd"]
        chiPeak = qaStats["chiPeak"]
        xarray = qaStats["Xarray"]
        yarray = qaStats["Yarray"]
        dxarray = qaStats["dx"]
        dwarray = qaStats["dsigma"]

        aveRange = 5.0
        medRange = 5.0
        stdRange = 10.0
        ql = 0.2
        largeAve = chiAve > aveRange
        smallAve = chiAve < -aveRange
        largeMed = chiMedian > medRange
        smallMed = chiMedian < -medRange
        largeStd = chiStd > stdRange

        qaImagePdf = MultipagePdfFigure()
        ct = self.getTargetColors()

        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[0].scatter(
                    fiberIds[targetMask[t]],
                    pfsArmAve[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[0].set_ylabel("Mean flux of pfsArm")
        ax[0].set_xlabel("fiberId")
        ax[0].set_yscale("log")
        ax[0].legend()
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[1].scatter(
                    xa[targetMask[t]],
                    pfsArmAve[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[1].set_xlabel("X (pix)")
        ax[1].set_yscale("log")
        ax[1].legend()
        fig.suptitle("visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d" % dataId)
        qaImagePdf.append(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        ax[0].plot([np.amin(fiberIds), np.amax(fiberIds)], [0, 0], "gray")
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[0].scatter(
                    fiberIds[targetMask[t]],
                    chiAve[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[0].set_ylabel("Chi (average)")
        ax[0].set_xlabel("fiberId")
        ax[0].legend()
        if np.sum(largeAve) > 0:
            ax[0].quiver(
                fiberIds[largeAve],
                np.zeros(np.sum(largeAve)) + aveRange - aveRange * ql,
                0.0,
                aveRange * ql,
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
            )
        if np.sum(smallAve) > 0:
            ax[0].quiver(
                fiberIds[smallAve],
                np.zeros(np.sum(smallAve)) - aveRange + aveRange * ql,
                0.0,
                -aveRange * ql,
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
            )
        ax[0].set_ylim(-aveRange, aveRange)
        ax[1].plot([np.amin(xa), np.amax(xa)], [0, 0], "gray")
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[1].scatter(
                    xa[targetMask[t]],
                    chiAve[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[1].set_xlabel("X (pix)")
        if np.sum(largeAve) > 0:
            ax[1].quiver(
                xa[largeAve],
                np.zeros(np.sum(largeAve)) + aveRange - aveRange * ql,
                0.0,
                aveRange * ql,
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
            )
        if np.sum(smallAve) > 0:
            ax[1].quiver(
                xa[smallAve],
                np.zeros(np.sum(smallAve)) - aveRange + aveRange * ql,
                0.0,
                -aveRange * ql,
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
            )
        ax[1].set_ylim(-aveRange, aveRange)
        ax[1].legend()
        fig.suptitle("visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d" % dataId)
        qaImagePdf.append(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        ax[0].plot([np.amin(fiberIds), np.amax(fiberIds)], [0, 0], "gray")
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[0].scatter(
                    fiberIds[targetMask[t]],
                    chiMedian[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[0].set_ylabel("Chi (median)")
        ax[0].set_xlabel("fiberId")
        ax[0].legend()
        if np.sum(largeMed) > 0:
            ax[0].quiver(
                fiberIds[largeMed],
                np.zeros(np.sum(largeMed)) + medRange - medRange * ql,
                0.0,
                medRange * ql,
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
            )
        if np.sum(smallMed) > 0:
            ax[0].quiver(
                fiberIds[smallMed],
                np.zeros(np.sum(smallMed)) - medRange + medRange * ql,
                0.0,
                -medRange * ql,
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
            )
        ax[0].set_ylim(-medRange, medRange)
        ax[1].plot([np.amin(xa), np.amax(xa)], [0, 0], "gray")
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[1].scatter(
                    xa[targetMask[t]],
                    chiMedian[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[1].set_xlabel("X (pix)")
        if np.sum(largeMed) > 0:
            ax[1].quiver(
                xa[largeMed],
                np.zeros(np.sum(largeMed)) + medRange - medRange * ql,
                0.0,
                medRange * ql,
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
            )
        if np.sum(smallMed) > 0:
            ax[1].quiver(
                xa[smallMed],
                np.zeros(np.sum(smallMed)) - medRange + medRange * ql,
                0.0,
                -medRange * ql,
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
            )
        ax[1].set_ylim(-medRange, medRange)
        fig.suptitle("visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d" % dataId)
        qaImagePdf.append(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        ax[0].plot([np.amin(fiberIds), np.amax(fiberIds)], [1, 1], "gray")
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[0].scatter(
                    fiberIds[targetMask[t]],
                    chiStd[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[0].set_ylabel("Chi (standard deviation)")
        ax[0].set_xlabel("fiberId")
        ax[0].legend()
        if np.sum(largeStd) > 0:
            ax[0].quiver(
                fiberIds[largeStd],
                np.zeros(np.sum(largeStd)) + stdRange - stdRange * ql / 2,
                0.0,
                stdRange * ql / 2,
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
            )
        ax[0].set_ylim(-stdRange * 0.1, stdRange)
        ax[1].plot([np.amin(xa), np.amax(xa)], [1, 1], "gray")
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[1].scatter(
                    xa[targetMask[t]],
                    chiStd[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[1].set_xlabel("X (pix)")
        if np.sum(largeStd) > 0:
            ax[1].quiver(
                xa[largeStd],
                np.zeros(np.sum(largeStd)) + stdRange - stdRange * ql / 2,
                0.0,
                stdRange * ql / 2,
                color="k",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
            )
        ax[1].set_ylim(-stdRange * 0.1, stdRange)
        ax[1].legend()
        fig.suptitle("visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d" % dataId)
        qaImagePdf.append(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[0].scatter(
                    fiberIds[targetMask[t]],
                    chiSquare[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[0].set_ylabel("Chi^2 (average)")
        ax[0].set_xlabel("fiberId")
        ax[0].set_yscale("log")
        ax[0].legend()
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[1].scatter(
                    xa[targetMask[t]],
                    chiSquare[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[1].set_ylabel("Chi^2 (average)")
        ax[1].set_xlabel("X (pix)")
        ax[1].set_yscale("log")
        ax[1].legend()
        fig.suptitle("visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d" % dataId)
        qaImagePdf.append(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[0].scatter(
                    fiberIds[targetMask[t]],
                    chiPeak[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[0].set_ylabel("Chi at profile peak pixels")
        ax[0].set_xlabel("fiberId")
        ax[0].legend()
        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                ax[1].scatter(
                    xa[targetMask[t]],
                    chiPeak[targetMask[t]],
                    10.0,
                    c=ct[t],
                    label="{}: {} fibers".format(t, np.sum(targetMask[t])),
                )
        ax[1].set_xlabel("X (pix)")
        fig.suptitle("visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d" % dataId)
        qaImagePdf.append(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(10, 10))
        disp = afwDisplay.Display(fig)
        disp.scale("linear", -5, 5, Q=1)
        disp.setImageColormap("coolwarm")
        disp.setMaskPlaneColor("REFLINE", afwDisplay.IGNORE)
        disp.mtv(chiimage, title=f"{'%(visit)d %(arm)s%(spectrograph)d' % dataId}")
        addPfsCursor(disp, detectorMap)
        qaImagePdf.append(fig, dpi=self.config.figureDpi, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11.5, 10))
        ax = fig.add_subplot(1, 1, 1)
        mappable = ax.scatter(xarray, yarray, s=2, c=dxarray, cmap="coolwarm", vmin=-0.1, vmax=0.1)
        plt.title("dX (pix)")
        plt.xlabel("X (pix)")
        plt.ylabel("Y (pix)")
        plt.xlim(0, chiimage.getDimensions()[0])
        plt.ylim(0, chiimage.getDimensions()[1])
        fig.colorbar(mappable, ax=ax, pad=0.01)
        qaImagePdf.append(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plt.figure(figsize=(11.5, 10))
        ax = fig.add_subplot(1, 1, 1)
        mappable = ax.scatter(xarray, yarray, s=2, c=dwarray, cmap="coolwarm", vmin=-0.1, vmax=0.1)
        plt.title("d$\\sigma$/$\\sigma$")
        plt.xlabel("X (pix)")
        plt.ylabel("Y (pix)")
        plt.xlim(0, chiimage.getDimensions()[0])
        plt.ylim(0, chiimage.getDimensions()[1])
        fig.colorbar(mappable, ax=ax, pad=0.01)
        qaImagePdf.append(fig, bbox_inches="tight")
        plt.close(fig)

        return qaImagePdf

    def fitGaussiansToPfsArmCutAndCalExpCut(
        self,
        xcoordNarrow: np.ndarray,
        pfsArmCutNarrow: np.ndarray,
        calExpCutNarrow: np.ndarray,
        center0: float,
    ):
        """Fit Gaussians to ``pfsArmCutNarrow`` and ``calExpCutNarrow``.

        Parameters
        ----------
        xcoordNarrow : np.ndarray
            Samples of ``x``.
        pfsArmCutNarrow : np.ndarray
            XXXXX
        calExpCutNarrow : np.ndarray
            XXXXX
        center0: float
            Initial guess of the center of Gaussians.
            (Common to ``pfsArmCutNarrow`` and ``calExpCutNarrow``)

        Returns
        -------
        ok : `bool`
            Did fitting succeed?
        pfsArmCenter : `float`
            mu of Gaussian(mu, sigma) fitted to ``pfsArmCutNarrow``
        pfsArmWidth : `float`
            sigma of Gaussian(mu, sigma) fitted to ``pfsArmCutNarrow``
        calExpCenter : `float`
            mu of Gaussian(mu, sigma) fitted to ``calExpCenter``
        calExpWidth : `float`
            sigma of Gaussian(mu, sigma) fitted to ``calExpCenter``
        """
        PSFFWHM = 1.5

        ok = False
        pfsArmCenter, pfsArmWidth = math.nan, math.nan
        calExpCenter, calExpWidth = math.nan, math.nan

        try:
            if self.config.fixWidth:
                poptPfsArm, pcovPfsArm = curve_fit(
                    gaussianFixedWidth,
                    xcoordNarrow,
                    pfsArmCutNarrow,
                    p0=np.array([np.max(pfsArmCutNarrow), center0]),
                )
                poptCalExp, pcovCalExp = curve_fit(
                    gaussianFixedWidth,
                    xcoordNarrow,
                    calExpCutNarrow,
                    p0=np.array([np.max(calExpCutNarrow), center0]),
                )
                stdErrPfsArm = np.sqrt(np.diag(pcovPfsArm))
                stdErrCalExp = np.sqrt(np.diag(pcovCalExp))
                if (
                    stdErrPfsArm[1] / poptPfsArm[1] < self.config.thresError
                    and stdErrCalExp[1] / poptCalExp[1] < self.config.thresError
                ):
                    ok = True
                    pfsArmCenter, pfsArmWidth = poptPfsArm[1], PSFFWHM
                    calExpCenter, calExpWidth = poptCalExp[1], PSFFWHM
                else:
                    # I don't know the reason of the following two lines,
                    # but I obey the original code
                    pfsArmWidth = PSFFWHM
                    calExpWidth = PSFFWHM
            else:
                poptPfsArm, pcovPfsArm = curve_fit(
                    gaussian_func,
                    xcoordNarrow,
                    pfsArmCutNarrow,
                    p0=np.array([np.max(pfsArmCutNarrow), center0, 1.0]),
                )
                poptCalExp, pcovCalExp = curve_fit(
                    gaussian_func,
                    xcoordNarrow,
                    calExpCutNarrow,
                    p0=np.array([np.max(calExpCutNarrow), center0, 1.0]),
                )
                stdErrPfsArm = np.sqrt(np.diag(pcovPfsArm))
                stdErrCalExp = np.sqrt(np.diag(pcovCalExp))
                if (
                    stdErrPfsArm[1] / poptPfsArm[1] < self.config.thresError
                    and stdErrPfsArm[2] / poptPfsArm[2] < self.config.thresError
                    and stdErrCalExp[1] / poptCalExp[1] < self.config.thresError
                    and stdErrCalExp[2] / poptCalExp[2] < self.config.thresError
                ):
                    ok = True
                    pfsArmCenter, pfsArmWidth = poptPfsArm[1], poptPfsArm[2]
                    calExpCenter, calExpWidth = poptCalExp[1], poptCalExp[2]

        except (ValueError, RuntimeError):
            pass

        return Struct(
            ok=ok,
            pfsArmCenter=pfsArmCenter,
            pfsArmWidth=pfsArmWidth,
            calExpCenter=calExpCenter,
            calExpWidth=calExpWidth,
        )

    @staticmethod
    def getStatsPerFiber(
        data: MaskedImageF, detectorMap: DetectorMap, fiberId: int, xwin: int = 3
    ) -> StatsPerFiber:
        """Get statistics for a fiber.

        Parameters
        ----------
        data : `MaskedImageF`
            XXXXX
        detectorMap : `DetectorMap`
            XXXXX
        fiberId : `int`
            XXXXX
        xwin : `int`
            XXXXX

        Returns
        -------
        stats : `StatsPerFiber`
            Statistics.
        """
        ymin = 0
        ymax = data.getDimensions()[1]
        xmax = data.getDimensions()[0]
        yo = np.arange(ymin, ymax).astype(np.float64)
        xo = detectorMap.getXCenter(fiberId, yo)
        xs = np.round(xo).astype(int)
        ys = np.round(yo).astype(int)
        image = data.image.array
        mask = data.mask.array
        imageFiber = []
        maskFiber = []
        for x, y in zip(xs, ys):
            image_cut = image[y, x - xwin : x + xwin + 1].copy()
            mask_cut = mask[y, x - xwin : x + xwin + 1].copy()
            if x + xwin + 1 > xmax:
                image_cut = np.concatenate([np.zeros(xwin * 2 + 1 - len(image_cut)), image_cut])
                mask_cut = np.concatenate([np.zeros(xwin * 2 + 1 - len(mask_cut)), mask_cut])
            elif x - xwin < 0:
                image_cut = np.concatenate([image_cut, np.zeros(xwin * 2 + 1 - len(image_cut))])
                mask_cut = np.concatenate([mask_cut, np.zeros(xwin * 2 + 1 - len(mask_cut))])
            imageFiber.append(image_cut)
            maskFiber.append(mask_cut)
        imageFiber = np.array(imageFiber)
        maskFiber = np.array(maskFiber)

        chi2 = np.average(imageFiber[maskFiber == 0] ** 2)
        im_ave = np.average(imageFiber[maskFiber == 0])

        return StatsPerFiber(
            chi2=chi2,
            im_ave=im_ave,
            x_ave=np.average(xo),
            chi_f=imageFiber,
            mask_f=maskFiber,
        )

    @staticmethod
    def getTargetColors() -> Dict[TargetType, str]:
        """Get a map from `TargetType` to color name in matplotlib

        Returns
        -------
        targetColors : `dict[TargetType, str]`
            Color name for each `TargetType`.
        """
        colors = [
            "r",
            "b",
            "g",
            "y",
            "gray",
            "orange",
            "cyan",
            "k",
        ]
        return dict(zip(TargetType, itertools.cycle(colors)))
