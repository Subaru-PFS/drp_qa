import dataclasses
import itertools
from typing import Dict

import lsstDebug
import numpy as np
from lsst.afw.image import ExposureF, MaskedImageF
from lsst.pex.config import Field, ListField
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
from scipy.stats import iqr

from pfs.drp.qa.storageClasses import MultipagePdfFigure, QaDict

import warnings


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
    img_f : `np.ndarray` of `float`, shape ``(N, M)``
        XXXXX
    mask_f : `np.ndarray` of `int`, shape ``(N, M)``
        XXXXX
    res_f : `np.ndarray` of `float`, shape ``(N, M)``
        XXXXX
    chi_f : `np.ndarray` of `float`, shape ``(N, M)``
        XXXXX
    """

    chi2: float
    im_ave: float
    x_ave: float
    img_f: np.ndarray
    mask_f: np.ndarray
    res_f: np.ndarray
    chi_f: np.ndarray


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
    calexp = InputConnection(
        name="calexp",
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
        doc="Detail plots. calexp, residual, and chi images, along with related 2D image/"
        "histogram views, are plotted for some fibers with bad extraction quality.",
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

    fiberWidth = Field(dtype=int, default=3, doc="Half width of a fiber region (pix)")
    plotMinChiMed = Field(dtype=float, default=-1.5, doc="Minimum median Chi to plot")
    plotMaxChiMed = Field(dtype=float, default=1.5, doc="Maximum median Chi to plot")
    plotMinChiStd = Field(dtype=float, default=0.0, doc="Minimum standard deviation of Chi to plot")
    plotMaxChiStd = Field(dtype=float, default=3.5, doc="Maximum standard deviation of Chi to plot")
    plotMinChiAtPeak = Field(dtype=float, default=-1.5, doc="Minimum Chi at peak to plot")
    plotMaxChiAtPeak = Field(dtype=float, default=1.5, doc="Maximum Chi at peak to plot")
    plotMinResFrac = Field(dtype=float, default=-5.0, doc="Minimum residual fraction")
    plotMaxResFrac = Field(dtype=float, default=5.0, doc="Maximum residual fraction")
    plotHistRangeScale = Field(dtype=float, default=1.5, doc="The scale factor for the Chi histogram range")
    plotHistNbin = Field(dtype=int, default=100, doc="The number of bins for the Chi histogram")
    targetType = ListField(dtype=str, default=["^ENGINEERING"],
                           doc="Target type for which to calculate statistics")
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
        # Get the dataIds for help with plotting.
        data_id = dict(**inputRefs.pfsArm.dataId.mapping)
        data_id["run"] = inputRefs.pfsArm.run

        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = data_id

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
        calexp: ExposureF,
        fiberProfiles: FiberProfileSet,
        detectorMap: DetectorMap,
        pfsArm: PfsArm,
        pfsConfig: PfsConfig,
        dataId: dict,
    ) -> Struct:
        """QA of extraction by analyzing the residual image.

        Parameters
        ----------
        postISRCCD : `ExposureF`
            2D image before scattered light correction
        calexp : `ExposureF`
            2D image after scattered light correction
        fiberProfiles : `FiberProfileSet`
            Profiles of each fiber.
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y
        pfsArm : `PfsArm`
            Extracted spectra from arm.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers.
        dataId : `dict`
            The dataId for the visit.

        Returns
        -------
        extQaStats : `MultipagePdfFigure`
            Summary plots.
            Results of the residual analysis of extraction are plotted.
        extQaImage : `MultipagePdfFigure`
            Detail plots.
            calexp, residual, and chi images and the comparison of the calexp
            profile and fiberProfiles are plotted for some fibers with bad
            extraction quality.
        extQaImage_pickle : `QaDict`
            Data to be pickled.
            Statistics of the residual analysis.
        """
        self.log.info("Extraction QA on %s", pfsArm.identity)

        spectra = SpectrumSet.fromPfsArm(pfsArm)
        traces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap)
        image = spectra.makeImage(calexp.getDimensions(), traces)
        reconstimage = image.clone()

        subtracted = calexp.clone()
        subtracted.image -= image
        divided = subtracted.clone()
        divided.image /= calexp.image
        chiimage = subtracted.clone()
        with np.errstate(divide="ignore", invalid="ignore"):
            chiimage.image.array /= np.sqrt(calexp.variance.array)

        msk = (
            (pfsConfig.spectrograph == dataId["spectrograph"]) &
            (pfsConfig.fiberStatus == FiberStatus.GOOD) &
            np.isin(pfsConfig.targetType, TargetType.fromList(self.config.targetType))
        )
        fiberIds = pfsConfig[msk].fiberId
        targetMask = {t: pfsConfig[msk].targetType == t for t in TargetType}

        img_data = calexp.maskedImage
        res_data = subtracted.maskedImage
        chi_data = chiimage.maskedImage
        xa = []
        chiSquare = []
        chiAve = []
        chiMed = []
        chiStd = []
        chiAtPeak = []
        pfsArmAve = []
        chiAveSpec = []
        resFrac = []
        for fiberId in fiberIds:
            stats = self.getStatsPerFiber(
                img_data, res_data, chi_data, detectorMap, fiberId, xwin=self.config.fiberWidth
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice")
                stats.chi_f[stats.mask_f != 0] = np.nan
                chiSquare.append(stats.chi2)
                chiAve.append(np.nanmean(stats.chi_f[stats.mask_f == 0]))
                chiMed.append(np.nanmedian(stats.chi_f[stats.mask_f == 0]))
                chiStd.append(iqr(stats.chi_f[stats.mask_f == 0]) / 1.349)
                chiAtPeak.append(
                    np.nanmean(
                        stats.chi_f[:, self.config.fiberWidth][stats.mask_f[:, self.config.fiberWidth] == 0]
                    )
                )

                valid = stats.mask_f == 0
                res_f_valid = np.where(valid, stats.res_f, np.nan)
                img_f_valid = np.where(valid, stats.img_f, np.nan)
                res_sum = np.nansum(res_f_valid.T, axis=0)
                img_sum = np.nansum(img_f_valid.T, axis=0)
                row_ratio = np.full_like(res_sum, np.nan, dtype=float)
                np.divide(res_sum, img_sum, out=row_ratio, where=np.isfinite(img_sum) & (img_sum != 0))
                resFrac.append(np.nanmedian(row_ratio) * 100.)

                pfsArmAve.append(np.nanmean(pfsArm[pfsArm.fiberId == fiberId].flux[0]))
                chiAveSpec.append(np.nanmean(stats.chi_f, axis=1))
                xa.append(stats.x_ave)

        pfsArmAve = np.array(pfsArmAve)
        chiSquare = np.array(chiSquare)
        chiAve = np.array(chiAve)
        chiMed = np.array(chiMed)
        chiStd = np.array(chiStd)
        chiAtPeak = np.array(chiAtPeak)
        xa = np.array(xa)
        resFrac = np.array(resFrac)

        qaStats = QaDict(
            {
                "dataId": dataId,
                "fiberIds": fiberIds,
                "xa": xa,
                "pfsArmAve": pfsArmAve,
                "targetMask": targetMask,
                "chiSquare": chiSquare,
                "chiAve": chiAve,
                "chiMed": chiMed,
                "chiStd": chiStd,
                "chiAtPeak": chiAtPeak,
                "resFrac": resFrac,
            }
        )

        qaImagePdf = self.makeImagePdf(
            dataId, detectorMap, pfsArm, postISRCCD, calexp, reconstimage, chiimage
        )
        qaStatsPdf = self.drawStats(qaStats, dataId)

        return Struct(
            extQaStats=qaStatsPdf,
            extQaImage=qaImagePdf,
            extQaImage_pickle=qaStats,
        )

    def drawStats(
        self,
        qaStats: QaDict,
        dataId: dict,
    ) -> MultipagePdfFigure:
        """Draw figures on new pages of ``qaStatsPdf``.

        Parameters
        ----------
        qaStats : `QaDict`
            XXXXX
        dataId : `dict`
            Data ID. Required keys are: "visit", "arm", "spectrograph".

        Returns
        -------
        extQaStats : `MultipagePdfFigure`
            XXXXX
        """

        fiberIds = qaStats["fiberIds"]
        xa = qaStats["xa"]
        pfsArmAve = qaStats["pfsArmAve"]
        targetMask = qaStats["targetMask"]
        chiMed = qaStats["chiMed"]
        chiStd = qaStats["chiStd"]
        chiAtPeak = qaStats["chiAtPeak"]
        resFrac = qaStats["resFrac"]

        qaStatsPdf = MultipagePdfFigure()
        # ct = self.getTargetColors()

        fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")

        titleStr = "Chi distribution (visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d)\n" % dataId
        titleStr += "RUN=%(run)s" % dataId
        fig.suptitle(titleStr, y=1.05)
        sc = None

        # chiMed vs. fiberId (top left)
        ax[0][0].axhline(0.0, color="gray", linestyle="dashed")

        ymin = self.config.plotMinChiMed
        ymax = self.config.plotMaxChiMed
        ax[0][0].set_ylim(ymin, ymax)

        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                fid = fiberIds[targetMask[t]]
                val = chiMed[targetMask[t]]
                col = pfsArmAve[targetMask[t]]

                under = val < ymin
                over = val > ymax
                inside = ~(under | over)

                sc = ax[0][0].scatter(
                    fid[inside], val[inside], 10.0, c=col[inside],
                    vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    label="{}: {} fibers".format(t, np.sum(targetMask[t]))
                )
                if np.any(over):
                    ax[0][0].scatter(
                        fid[over], np.full(np.sum(over), ymax),
                        marker="^", s=50, c=col[over], edgecolors="k", rasterized=True, clip_on=False,
                        vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    )
                if np.any(under):
                    ax[0][0].scatter(
                        fid[under], np.full(np.sum(under), ymin),
                        marker="v", s=50, c=col[under], edgecolors="k", rasterized=True, clip_on=False,
                        vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    )

        if sc is not None:
            fig.colorbar(sc, ax=ax[0][0], location="right", fraction=0.04, alpha=1.0, label="pfsArmAve")

        ax2 = ax[0][0].twiny()
        ax2.set_xlim(ax[0][0].get_xlim())
        bottom_ticks = ax[0][0].get_xticks()
        sort_idx = np.argsort(fiberIds)
        sorted_fiber_ids = fiberIds[sort_idx]
        sorted_xa = xa[sort_idx]
        unique_fiber_ids, unique_indices = np.unique(sorted_fiber_ids, return_index=True)
        unique_xa = sorted_xa[unique_indices]
        interp_xa = np.interp(bottom_ticks, unique_fiber_ids, unique_xa)
        selected_labels = ["{:.2f}".format(pix) for pix in interp_xa]

        med = round(np.nanmedian(chiMed), 3)
        q3, q1 = np.nanpercentile(chiMed, [75, 25])
        robust_sigma = 0.741*(q3 - q1)
        ax[0][0].text(
            0.05, 0.1, f"median={med:.3}, robustSigma={robust_sigma:.3}", transform=ax[0][0].transAxes
        )
        ax2.set_xticks(bottom_ticks)
        ax2.set_xticklabels(selected_labels, rotation=60)
        ax2.set_xlabel("X (pix)")
        ax[0][0].grid(color="gray", linestyle=":", linewidth=0.5)
        ax[0][0].set_ylabel("Chi (median)")
        ax[0][0].set_xlabel("fiberId")
        ax[0][0].legend()

        # chiStd vs. fiberId (top right)
        ax[0][1].axhline(1.0, color="gray", linestyle="dashed")

        ymin = self.config.plotMinChiStd
        ymax = self.config.plotMaxChiStd
        ax[0][1].set_ylim(ymin, ymax)

        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                fid = fiberIds[targetMask[t]]
                val = chiStd[targetMask[t]]
                col = pfsArmAve[targetMask[t]]

                under = val < ymin
                over = val > ymax
                inside = ~(under | over)

                sc = ax[0][1].scatter(
                    fid[inside], val[inside], 10.0, c=col[inside],
                    vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    label="{}: {} fibers".format(t, np.sum(targetMask[t]))
                )
                if np.any(over):
                    ax[0][1].scatter(
                        fid[over], np.full(np.sum(over), ymax),
                        marker="^", s=50, c=col[over], edgecolors="k", rasterized=True, clip_on=False,
                        vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    )
                if np.any(under):
                    ax[0][1].scatter(
                        fid[under], np.full(np.sum(under), ymin),
                        marker="v", s=50, c=col[under], edgecolors="k", rasterized=True, clip_on=False,
                        vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    )

        if sc is not None:
            fig.colorbar(sc, ax=ax[0][1], location="right", fraction=0.04, alpha=1.0, label="pfsArmAve")

        ax3 = ax[0][1].twiny()
        ax3.set_xlim(ax[0][1].get_xlim())
        bottom_ticks = ax[0][1].get_xticks()
        sort_idx = np.argsort(fiberIds)
        sorted_fiber_ids = fiberIds[sort_idx]
        sorted_xa = xa[sort_idx]
        unique_fiber_ids, unique_indices = np.unique(sorted_fiber_ids, return_index=True)
        unique_xa = sorted_xa[unique_indices]
        interp_xa = np.interp(bottom_ticks, unique_fiber_ids, unique_xa)
        selected_labels = ["{:.2f}".format(pix) for pix in interp_xa]

        ax3.set_xticks(bottom_ticks)
        ax3.set_xticklabels(selected_labels, rotation=60)
        ax3.set_xlabel("X (pix)")

        med = round(np.nanmedian(chiStd), 3)
        q3, q1 = np.nanpercentile(chiStd, [75, 25])
        robust_sigma = 0.741*(q3 - q1)
        ax[0][1].text(
            0.05, self.config.plotMaxChiStd * 0.0075,
            f"median={med:.3}, robustSigma={robust_sigma:.3}", transform=ax[0][1].transAxes
        )
        ax[0][1].set_xlabel("X pixel")
        ax[0][1].grid(color="gray", linestyle=":", linewidth=0.5)
        ax[0][1].set_ylabel("Chi (standard deviation)")
        ax[0][1].set_xlabel("fiberId")
        ax[0][1].legend()

        # chiAtPeak vs. fiberId (bottom left)
        ax[1][0].axhline(0.0, color="gray", linestyle="dashed")

        ymin = self.config.plotMinChiAtPeak
        ymax = self.config.plotMaxChiAtPeak
        ax[1][0].set_ylim(ymin, ymax)

        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                fid = fiberIds[targetMask[t]]
                val = chiAtPeak[targetMask[t]]
                col = pfsArmAve[targetMask[t]]

                under = val < ymin
                over = val > ymax
                inside = ~(under | over)

                sc = ax[1][0].scatter(
                    fid[inside], val[inside], 10.0, c=col[inside],
                    vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    label="{}: {} fibers".format(t, np.sum(targetMask[t]))
                )
                if np.any(over):
                    ax[1][0].scatter(
                        fid[over], np.full(np.sum(over), ymax),
                        marker="^", s=50, c=col[over], edgecolors="k", rasterized=True, clip_on=False,
                        vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    )
                if np.any(under):
                    ax[1][0].scatter(
                        fid[under], np.full(np.sum(under), ymin),
                        marker="v", s=50, c=col[under], edgecolors="k", rasterized=True, clip_on=False,
                        vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    )

        if sc is not None:
            fig.colorbar(sc, ax=ax[1][0], location="right", fraction=0.04, alpha=1.0, label="pfsArmAve")

        ax4 = ax[1][0].twiny()
        ax4.set_xlim(ax[1][0].get_xlim())
        bottom_ticks = ax[1][0].get_xticks()
        sort_idx = np.argsort(fiberIds)
        sorted_fiber_ids = fiberIds[sort_idx]
        sorted_xa = xa[sort_idx]
        unique_fiber_ids, unique_indices = np.unique(sorted_fiber_ids, return_index=True)
        unique_xa = sorted_xa[unique_indices]
        interp_xa = np.interp(bottom_ticks, unique_fiber_ids, unique_xa)
        selected_labels = ["{:.2f}".format(pix) for pix in interp_xa]

        med = round(np.nanmedian(chiAtPeak), 3)
        q3, q1 = np.nanpercentile(chiAtPeak, [75, 25])
        robust_sigma = 0.741*(q3 - q1)
        ax4.text(0.05, 0.1, f"median={med:.3}, robustSigma={robust_sigma:.3}", transform=ax[1][0].transAxes)
        ax4.set_xticks(bottom_ticks)
        ax4.set_xticklabels(selected_labels, rotation=60)
        ax4.set_xlabel("X (pix)")
        ax[1][0].grid(color="gray", linestyle=":", linewidth=0.5)
        ax[1][0].set_ylabel("Chi at profile peak pixels")
        ax[1][0].set_xlabel("fiberId")
        ax[1][0].legend()

        # resFrac vs. fiberId (bottom right)
        ax[1][1].axhline(0.0, color="gray", linestyle="dashed")

        ymin = self.config.plotMinResFrac
        ymax = self.config.plotMaxResFrac
        ax[1][1].set_ylim(ymin, ymax)

        for t in targetMask.keys():
            if np.sum(targetMask[t]) > 0:
                fid = fiberIds[targetMask[t]]
                val = resFrac[targetMask[t]]
                col = pfsArmAve[targetMask[t]]

                under = val < ymin
                over = val > ymax
                inside = ~(under | over)

                sc = ax[1][1].scatter(
                    fid[inside], val[inside], 10.0, c=col[inside],
                    vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    label="{}: {} fibers".format(t, np.sum(targetMask[t]))
                )
                if np.any(over):
                    ax[1][1].scatter(
                        fid[over], np.full(np.sum(over), ymax),
                        marker="^", s=50, c=col[over], edgecolors="k", rasterized=True, clip_on=False,
                        vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    )
                if np.any(under):
                    ax[1][1].scatter(
                        fid[under], np.full(np.sum(under), ymin),
                        marker="v", s=50, c=col[under], edgecolors="k", rasterized=True, clip_on=False,
                        vmin=np.nanmin(pfsArmAve), vmax=np.nanmax(pfsArmAve),
                    )

        if sc is not None:
            fig.colorbar(sc, ax=ax[1][1], location="right", fraction=0.04, alpha=1.0, label="pfsArmAve")

        ax4 = ax[1][1].twiny()
        ax4.set_xlim(ax[1][1].get_xlim())
        bottom_ticks = ax[1][1].get_xticks()
        sort_idx = np.argsort(fiberIds)
        sorted_fiber_ids = fiberIds[sort_idx]
        sorted_xa = xa[sort_idx]
        unique_fiber_ids, unique_indices = np.unique(sorted_fiber_ids, return_index=True)
        unique_xa = sorted_xa[unique_indices]
        interp_xa = np.interp(bottom_ticks, unique_fiber_ids, unique_xa)
        selected_labels = ["{:.2f}".format(pix) for pix in interp_xa]

        med = round(np.nanmedian(resFrac), 3)
        q3, q1 = np.nanpercentile(resFrac, [75, 25])
        robust_sigma = 0.741*(q3 - q1)
        ax4.text(0.05, 0.1, f"median={med:.3}, robustSigma={robust_sigma:.3}", transform=ax[1][1].transAxes)
        ax4.set_xticks(bottom_ticks)
        ax4.set_xticklabels(selected_labels, rotation=60)
        ax4.set_xlabel("X (pix)")
        ax[1][1].grid(color="gray", linestyle=":", linewidth=0.5)
        ax[1][1].set_ylabel("Residual fraction (percent)")
        ax[1][1].set_xlabel("fiberId")
        ax[1][1].legend()

        # plt.subplots_adjust(wspace=0.3)

        qaStatsPdf.append(fig, bbox_inches="tight")
        plt.close(fig)

        return qaStatsPdf

    def makeImagePdf(
        self,
        dataId: dict,
        detectorMap: DetectorMap,
        pfsArm: PfsArm,
        postISRCCD: ExposureF,
        calexp: ExposureF,
        reconstimage: ExposureF,
        chiimage: ExposureF
    ) -> MultipagePdfFigure:
        """Make ``extQaImage``

        Parameters
        ----------
        dataId : `dict`
            Data ID. Required keys are: "visit", "arm", "spectrograph".
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y
        pfsArm : `PfsArm`
            Extracted spectra from arm.
        postISRCCD : `ExposureF`
            2D image before scattered light correction
        calexp : `ExposureF`
            2D image after scattered light correction
        reconstimage : `ExposureF`
            2D image reconstructed from extracted spectra
        chiimage : `ExposureF`
            2D image of chi values for the residuals

        Returns
        -------
        extQaImage : `MultipagePdfFigure`
            XXXXX
        """
        qaImagePdf = MultipagePdfFigure()
        # ct = self.getTargetColors()

        # combined plot for 2D images

        fig, ax = plt.subplots(2, 4, figsize=(25, 10))
        fig.suptitle(
            "Input images (visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d)\nRUN=%(run)s" % dataId)
        # 1. 2D image before scattered light correction (postISRCCD)
        if postISRCCD is not None:
            imagearray = postISRCCD.getImage().array
        else:
            imagearray = np.full_like(calexp.getImage().array, np.nan)
        ax[0, 0].set_title("postISRCCD.image")
        med = round(np.nanmedian(imagearray), 3)
        mean = round(np.nanmean(imagearray), 3)
        vmin = np.nanpercentile(imagearray, 25)
        vmax = np.nanpercentile(imagearray, 75)
        im2 = ax[0, 0].imshow(imagearray, vmin=vmin, vmax=vmax, origin="lower")
        plt.colorbar(im2, shrink=0.8)
        ax[0, 0].set_xlabel("X pixel")
        ax[0, 0].set_ylabel("Y pixel")
        ax[0, 0].text(
            0.0, 0.015, f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox=dict(facecolor="white", alpha=0.5), transform=ax[0, 0].transAxes
        )
        ax[0, 0].set_xlabel("X pixel")
        ax[0, 0].set_ylabel("Y pixel")

        # 2. 2D image after scattered light correction (calexp)
        ax[0, 1].set_title("calexp.image")
        imagearray = calexp.getImage().array
        med = round(np.nanmedian(imagearray), 3)
        mean = round(np.nanmean(imagearray), 3)
        vmin = np.nanpercentile(imagearray, 25)
        vmax = np.nanpercentile(imagearray, 75)
        im2 = ax[0, 1].imshow(imagearray, vmin=vmin, vmax=vmax, origin="lower")
        plt.colorbar(im2, shrink=0.8)
        ax[0, 1].set_xlabel("X pixel")
        ax[0, 1].set_ylabel("Y pixel")
        ax[0, 1].text(
            0.0, 0.015, f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox=dict(facecolor="white", alpha=0.5), transform=ax[0, 1].transAxes
        )
        ax[0, 1].set_xlabel("X pixel")
        ax[0, 1].set_ylabel("Y pixel")

        # 3. pfsArm.flux
        ax[0, 2].set_title("pfsArm.flux")
        imagearray = pfsArm.flux
        data = imagearray.T
        med = round(np.nanmedian(data), 3)
        mean = round(np.nanmean(data), 3)
        vmin = np.nanpercentile(data, 25)
        vmax = np.nanpercentile(data, 75)
        im3 = ax[0, 2].imshow(data, vmin=vmin, vmax=vmax, origin="lower", aspect=0.15)
        plt.colorbar(im3, shrink=0.8)
        ax[0, 2].text(
            0.0, 0.015, f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox=dict(facecolor="white", alpha=0.5), transform=ax[0, 2].transAxes
        )
        ax[0, 2].set_xlabel("fiberId")
        ax[0, 2].set_ylabel("Y pixel")

        # 4. 2D image reconstructed from extracted spectra
        ax[0, 3].set_title("reconstructed.image")
        data = reconstimage.array
        med = round(np.nanmedian(data), 3)
        mean = round(np.nanmean(data), 3)
        vmin = np.nanpercentile(data, 25)
        vmax = np.nanpercentile(data, 75)
        im4 = ax[0, 3].imshow(data, vmin=vmin, vmax=vmax, origin="lower")
        plt.colorbar(im4, shrink=0.8)
        ax[0, 3].set_xlabel("X pixel")
        ax[0, 3].set_ylabel("Y pixel")
        ax[0, 3].text(
            0.0, 0.015, f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox=dict(facecolor="white", alpha=0.5), transform=ax[0, 3].transAxes
        )
        masked_image = chiimage.getMaskedImage()
        variance = masked_image.getVariance().getArray()
        masked_image = masked_image.getImage().getArray()

        # 5. residual fraction

        residual_over_reconst = np.divide(
            (calexp.getMaskedImage().getImage().array - reconstimage.array),
            reconstimage.array,
            out=np.full_like(calexp.getImage().array, np.nan),
            where=(reconstimage.array != 0)
        )
        ax[1, 0].set_title("residual / reconstructed")
        imagearray = residual_over_reconst
        med = round(np.nanmedian(imagearray), 3)
        mean = round(np.nanmean(imagearray), 3)
        vmin = np.nanpercentile(imagearray, 25)
        vmax = np.nanpercentile(imagearray, 75)
        im5 = ax[1, 0].imshow(imagearray, vmin=vmin, vmax=vmax, origin="lower")
        plt.colorbar(im5, shrink=0.8)
        ax[1, 0].text(
            0.0, 0.015, f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox=dict(facecolor="white", alpha=0.5), transform=ax[1, 0].transAxes
        )
        ax[1, 0].set_xlabel("X pixel")
        ax[1, 0].set_ylabel("Y pixel")

        # 6. Variance
        ax[1, 1].set_title("calexp.variance")
        imagearray = variance
        med = round(np.nanmedian(imagearray), 3)
        mean = round(np.nanmean(imagearray), 3)
        vmin = np.nanpercentile(imagearray, 25)
        vmax = np.nanpercentile(imagearray, 75)
        im6 = ax[1, 1].imshow(imagearray, vmin=vmin, vmax=vmax, origin="lower")
        plt.colorbar(im6, shrink=0.8)
        ax[1, 1].text(
            0.0, 0.015, f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox=dict(facecolor="white", alpha=0.5), transform=ax[1, 1].transAxes
        )
        ax[1, 1].set_xlabel("X pixel")
        ax[1, 1].set_ylabel("Y pixel")

        # masked residual chi image
        imagearray = masked_image
        med = round(np.nanmedian(imagearray), 3)
        mean = round(np.nanmean(imagearray), 3)
        q99, q75, q25, q01 = np.nanpercentile(imagearray, [99, 75, 25, 1])
        sigma = round((0.741*(q75 - q25)), 3)

        # 7. 25-75%
        ax[1, 2].set_title("chi image (25-75%)")
        im7 = ax[1, 2].imshow(imagearray, vmin=q25, vmax=q75, origin="lower")
        plt.colorbar(im7, shrink=0.8)
        ax[1, 2].text(
            0.0, 0.015, f"mean={mean:.3} median={med:.3} sigma={sigma:.3}",
            bbox=dict(facecolor="white", alpha=0.5), transform=ax[1, 2].transAxes
        )
        ax[1, 2].set_xlabel("X pixel")
        ax[1, 2].set_ylabel("Y pixel")

        # 8. 1-99%
        ax[1, 3].set_title("chi image (1-99%)")
        im8 = ax[1, 3].imshow(imagearray, vmin=q01, vmax=q99, origin="lower")
        plt.colorbar(im8, shrink=0.8)
        ax[1, 3].text(
            0.0, 0.015, f"mean={mean:.3} median={med:.3} sigma={sigma:.3}",
            bbox=dict(facecolor="white", alpha=0.5), transform=ax[1, 3].transAxes
        )
        ax[1, 3].set_xlabel("X pixel")
        ax[1, 3].set_ylabel("Y pixel")

        qaImagePdf.append(fig, dpi=self.config.figureDpi, bbox_inches="tight")
        plt.close(fig)

        # a single plot of residual/reconstructed (25-75%)

        # med_rr = round(np.nanmedian(residual_over_reconst), 3)
        # mean_rr = round(np.nanmean(residual_over_reconst), 3)
        q95_rr, q75_rr, q25_rr, q05_rr = np.nanpercentile(residual_over_reconst, [95, 75, 25, 5])
        # sigma_rr = round((0.741*(q75_rr - q25_rr)), 3)
        # self.log.info(f"{q05_rr}, {q25_rr}, {q75_rr}, {q95_rr}")

        # fig, ax = plt.subplots(figsize=(10, 10))
        # titleStr = "residual / reconstructed image (25-75%)\n"
        # titleStr += "visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d\n" % dataId
        # titleStr += "RUN=%(run)s" % dataId
        # ax.set_title(titleStr)
        # im = ax.imshow(residual_over_reconst, vmin=q25_rr, vmax=q75_rr, origin="lower")
        # plt.colorbar(im, shrink=0.8, format="%.2f")
        # ax.text(
        #    0.0, 0.015, f"mean={mean_rr:.3} median={med_rr:.3} sigma={sigma_rr:.3}",
        #     bbox=dict(facecolor="white", alpha=0.5), transform=ax.transAxes
        # )
        # ax.set_xlabel("X pixel")
        # ax.set_ylabel("Y pixel")
        # qaImagePdf.append(fig, dpi=self.config.figureDpi, bbox_inches="tight")
        # plt.close(fig)

        # a single plot of chiimage (25-75%)
        fig, ax = plt.subplots(figsize=(10, 10))
        titleStr = "chi image (25-75%)\n"
        titleStr += "visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d\n" % dataId
        titleStr += "RUN=%(run)s" % dataId
        ax.set_title(titleStr)
        im = ax.imshow(imagearray, vmin=q25, vmax=q75, origin="lower")
        plt.colorbar(im, shrink=0.8, format="%.2f")
        ax.text(
            0.0, 0.015, f"mean={mean:.3} median={med:.3} sigma={sigma:.3}",
            bbox=dict(facecolor="white", alpha=0.5), transform=ax.transAxes
        )
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        qaImagePdf.append(fig, dpi=self.config.figureDpi, bbox_inches="tight")
        plt.close(fig)

        # zoom-in plot of chiimage (25-75%) for each grid on the detector

        xc = [500, 2000, 3500]
        yc = [3500, 2000, 500]
        xwin = 75
        ywin = 75

        fig, ax = plt.subplots(3, 3, figsize=(12, 12), layout="constrained")
        titleStr = "visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d" % dataId
        fig.suptitle(titleStr)

        for i, x in enumerate(xc):
            for j, y in enumerate(yc):
                xmin = x - xwin
                xmax = x + xwin
                ymin = y - ywin
                ymax = y + ywin
                data = imagearray[ymin:ymax, xmin:xmax]
                im = ax[j, i].imshow(
                    data, vmin=q25, vmax=q75, origin="lower",
                    extent=[xmin, xmax, ymin, ymax]
                )
                ax[j, i].set_xlabel("X (pix)")
                ax[j, i].set_ylabel("Y (pix)")

        qaImagePdf.append(fig, dpi=self.config.figureDpi, bbox_inches="tight")
        plt.close(fig)

        # a histogram of residual/reconstructed
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        titleStr = "visit=%(visit)d arm=%(arm)s spectrograph=%(spectrograph)d\n" % dataId
        titleStr += "RUN=%(run)s" % dataId
        fig.suptitle(titleStr)

        n, bins, patches = ax[0].hist(
            residual_over_reconst.flatten(),
            range=(self.config.plotHistRangeScale*q05_rr, self.config.plotHistRangeScale*q95_rr),
            bins=self.config.plotHistNbin,
            alpha=0.8,
            label="observed"
        )
        ax[0].set_xlim(self.config.plotHistRangeScale*q05_rr, self.config.plotHistRangeScale*q95_rr)
        ax[0].set_xlabel("residual / reconstructed")

        # a histogram of chi values
        n, bins, patches = ax[1].hist(
            imagearray.flatten(),
            range=(self.config.plotHistRangeScale*q01, self.config.plotHistRangeScale*q99),
            bins=self.config.plotHistNbin,
            alpha=0.8,
            label="observed"
        )
        peak_idx = np.argmax(n)
        amplitude = n[peak_idx]
        x = np.linspace(bins[0], bins[-1], 2*self.config.plotHistNbin)
        y = amplitude * np.exp(-0.5 * (x / 1.0)**2)
        ax[1].plot(x, y, ls="dashed", lw=2, c="k", alpha=0.8, label="N(0,1)")
        ax[1].set_xlim(self.config.plotHistRangeScale*q01, self.config.plotHistRangeScale*q99)
        ax[1].set_xlabel("chi")
        ax[1].legend(loc="upper right")

        qaImagePdf.append(fig, dpi=self.config.figureDpi, bbox_inches="tight")
        plt.close(fig)

        return qaImagePdf

    @staticmethod
    def getStatsPerFiber(
        img_data: MaskedImageF,
        res_data: MaskedImageF,
        chi_data: MaskedImageF,
        detectorMap: DetectorMap,
        fiberId: int,
        xwin: int = 3
    ) -> StatsPerFiber:
        """Get statistics for a fiber.

        Parameters
        ----------
        img_data : `MaskedImageF`
            XXXXX
        res_data : `MaskedImageF`
            XXXXX
        chi_data : `MaskedImageF`
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
        img_arr = img_data.image.array
        msk_arr = img_data.mask.array
        height, width = img_arr.shape
        res_arr = res_data.image.array
        chi_arr = chi_data.image.array

        yo = np.arange(height, dtype=np.float64)
        xo = detectorMap.getXCenter(fiberId, yo)
        xs = np.rint(xo).astype(int)
        ys = yo.astype(int)
        x_offsets = np.arange(-xwin, xwin + 1)

        grid_x = xs[:, None] + x_offsets[None, :]
        grid_y = ys[:, None]
        valid_mask = (grid_x >= 0) & (grid_x < width)
        grid_x_clamped = np.clip(grid_x, 0, width - 1)

        imgFiber = img_arr[grid_y, grid_x_clamped]
        maskFiber = msk_arr[grid_y, grid_x_clamped]
        resFiber = res_arr[grid_y, grid_x_clamped]
        chiFiber = chi_arr[grid_y, grid_x_clamped]

        imgFiber[~valid_mask] = 0
        maskFiber[~valid_mask] = 0
        resFiber[~valid_mask] = 0
        chiFiber[~valid_mask] = 0

        valid_pixels = valid_mask & (maskFiber == 0)
        valid_chi_data = chiFiber[valid_pixels]
        valid_img_data = imgFiber[valid_pixels]

        if valid_chi_data.size > 0:
            chi2 = np.nanmean(valid_chi_data ** 2)
        else:
            chi2 = np.nan

        if valid_img_data.size > 0:
            im_ave = np.nanmean(valid_img_data)
        else:
            im_ave = np.nan

        return StatsPerFiber(
            chi2=chi2,
            im_ave=im_ave,
            x_ave=np.nanmean(xo),
            img_f=imgFiber,
            mask_f=maskFiber,
            res_f=resFiber,
            chi_f=chiFiber,
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
