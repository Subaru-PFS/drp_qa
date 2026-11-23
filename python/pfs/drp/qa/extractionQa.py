import dataclasses
import itertools
import warnings

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
)
from lsst.pipe.base.connectionTypes import (
    Output as OutputConnection,
)
from lsst.pipe.base.connectionTypes import (
    PrerequisiteInput as PrerequisiteConnection,
)
from matplotlib import pyplot as plt
from scipy.stats import iqr

from pfs.datamodel import FiberStatus, PfsConfig, TargetType
from pfs.drp.qa.plotting.extraction import ExtractionPlotRanges, plotExtractionStats
from pfs.drp.qa.storageClasses import MultipagePdfFigure, QaDict
from pfs.drp.stella import DetectorMap, FiberProfileSet, PfsArm, SpectrumSet


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
    """Connections for ExtractionQaTask."""

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
    """Configuration for ExtractionQaTask."""

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
    targetType = ListField(
        dtype=str, default=["^ENGINEERING"], doc="Target type for which to calculate statistics"
    )
    figureDpi = Field(dtype=int, default=72, doc="resolution of plot for residual")


class ExtractionQaTask(PipelineTask):
    """Task for QA of extraction."""

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
            (pfsConfig.spectrograph == dataId["spectrograph"])
            & (pfsConfig.fiberStatus == FiberStatus.GOOD)
            & np.isin(pfsConfig.targetType, TargetType.fromList(self.config.targetType))
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
                resFrac.append(np.nanmedian(row_ratio) * 100.0)

                selected = pfsArm[pfsArm.fiberId == fiberId]
                # A GOOD pfsConfig fiber can be absent from the pfsArm (not extracted); its mean flux is
                # then undefined (NaN), but its chi stats (from the image and detectorMap) stay valid.
                pfsArmAve.append(np.nanmean(selected.flux[0]) if selected.flux.shape[0] > 0 else np.nan)
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
        """Draw the extraction-QA summary page from ``qaStats``.

        The drawing lives in
        `pfs.drp.qa.plotting.extraction.plotExtractionStats`, so the same figure
        can be made from the stored ``extQaImage_pickle`` alone, outside the
        stack (the QA dashboard's per-visit page). This wraps it in the task's
        `MultipagePdfFigure` output, applying the config's plot ranges.

        Parameters
        ----------
        qaStats : `QaDict`
            The per-fiber statistics built in `run`.
        dataId : `dict`
            Data ID. Required keys are: "visit", "arm", "spectrograph", "run".

        Returns
        -------
        extQaStats : `MultipagePdfFigure`
            The summary page.
        """
        ranges = ExtractionPlotRanges(
            chiMed=(self.config.plotMinChiMed, self.config.plotMaxChiMed),
            chiStd=(self.config.plotMinChiStd, self.config.plotMaxChiStd),
            chiAtPeak=(self.config.plotMinChiAtPeak, self.config.plotMaxChiAtPeak),
            resFrac=(self.config.plotMinResFrac, self.config.plotMaxResFrac),
        )
        qaStatsPdf = MultipagePdfFigure()
        qaStatsPdf.append(plotExtractionStats(qaStats, dataId=dataId, ranges=ranges), bbox_inches="tight")
        return qaStatsPdf

    def makeImagePdf(
        self,
        dataId: dict,
        detectorMap: DetectorMap,
        pfsArm: PfsArm,
        postISRCCD: ExposureF,
        calexp: ExposureF,
        reconstimage: ExposureF,
        chiimage: ExposureF,
    ) -> MultipagePdfFigure:
        """Make ``extQaImage``.

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
            "Input images (visit={visit} arm={arm} spectrograph={spectrograph})\nRUN={run}".format(**dataId)
        )
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
            0.0,
            0.015,
            f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox={"facecolor": "white", "alpha": 0.5},
            transform=ax[0, 0].transAxes,
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
            0.0,
            0.015,
            f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox={"facecolor": "white", "alpha": 0.5},
            transform=ax[0, 1].transAxes,
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
            0.0,
            0.015,
            f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox={"facecolor": "white", "alpha": 0.5},
            transform=ax[0, 2].transAxes,
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
            0.0,
            0.015,
            f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox={"facecolor": "white", "alpha": 0.5},
            transform=ax[0, 3].transAxes,
        )
        masked_image = chiimage.getMaskedImage()
        variance = masked_image.getVariance().getArray()
        masked_image = masked_image.getImage().getArray()

        # 5. residual fraction

        residual_over_reconst = np.divide(
            (calexp.getMaskedImage().getImage().array - reconstimage.array),
            reconstimage.array,
            out=np.full_like(calexp.getImage().array, np.nan),
            where=(reconstimage.array != 0),
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
            0.0,
            0.015,
            f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox={"facecolor": "white", "alpha": 0.5},
            transform=ax[1, 0].transAxes,
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
            0.0,
            0.015,
            f"mean={mean:.3} median={med:.3} vrange:25-75%",
            bbox={"facecolor": "white", "alpha": 0.5},
            transform=ax[1, 1].transAxes,
        )
        ax[1, 1].set_xlabel("X pixel")
        ax[1, 1].set_ylabel("Y pixel")

        # masked residual chi image
        imagearray = masked_image
        med = round(np.nanmedian(imagearray), 3)
        mean = round(np.nanmean(imagearray), 3)
        q99, q75, q25, q01 = np.nanpercentile(imagearray, [99, 75, 25, 1])
        sigma = round((0.741 * (q75 - q25)), 3)

        # 7. 25-75%
        ax[1, 2].set_title("chi image (25-75%)")
        im7 = ax[1, 2].imshow(imagearray, vmin=q25, vmax=q75, origin="lower")
        plt.colorbar(im7, shrink=0.8)
        ax[1, 2].text(
            0.0,
            0.015,
            f"mean={mean:.3} median={med:.3} sigma={sigma:.3}",
            bbox={"facecolor": "white", "alpha": 0.5},
            transform=ax[1, 2].transAxes,
        )
        ax[1, 2].set_xlabel("X pixel")
        ax[1, 2].set_ylabel("Y pixel")

        # 8. 1-99%
        ax[1, 3].set_title("chi image (1-99%)")
        im8 = ax[1, 3].imshow(imagearray, vmin=q01, vmax=q99, origin="lower")
        plt.colorbar(im8, shrink=0.8)
        ax[1, 3].text(
            0.0,
            0.015,
            f"mean={mean:.3} median={med:.3} sigma={sigma:.3}",
            bbox={"facecolor": "white", "alpha": 0.5},
            transform=ax[1, 3].transAxes,
        )
        ax[1, 3].set_xlabel("X pixel")
        ax[1, 3].set_ylabel("Y pixel")

        qaImagePdf.append(fig, dpi=self.config.figureDpi, bbox_inches="tight")
        plt.close(fig)

        # a single plot of residual/reconstructed (25-75%)

        # med_rr = round(np.nanmedian(residual_over_reconst), 3)
        # mean_rr = round(np.nanmean(residual_over_reconst), 3)
        q95_rr, _q75_rr, _q25_rr, q05_rr = np.nanpercentile(residual_over_reconst, [95, 75, 25, 5])
        # sigma_rr = round((0.741*(q75_rr - q25_rr)), 3)
        # self.log.info(f"{q05_rr}, {q25_rr}, {q75_rr}, {q95_rr}")

        # fig, ax = plt.subplots(figsize=(10, 10))
        # titleStr = "residual / reconstructed image (25-75%)\n"
        # titleStr += "visit={visit} arm={arm} spectrograph={spectrograph}\n".format(**dataId)
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
        titleStr += "visit={visit} arm={arm} spectrograph={spectrograph}\n".format(**dataId)
        titleStr += "RUN={run}".format(**dataId)
        ax.set_title(titleStr)
        im = ax.imshow(imagearray, vmin=q25, vmax=q75, origin="lower")
        plt.colorbar(im, shrink=0.8, format="%.2f")
        ax.text(
            0.0,
            0.015,
            f"mean={mean:.3} median={med:.3} sigma={sigma:.3}",
            bbox={"facecolor": "white", "alpha": 0.5},
            transform=ax.transAxes,
        )
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        qaImagePdf.append(fig, dpi=self.config.figureDpi, bbox_inches="tight")
        plt.close(fig)

        # a single plot of chiimage (fixed color range defined in the config)
        fig, ax = plt.subplots(figsize=(10, 10))
        titleStr = "chi image\n"
        titleStr += "visit={visit} arm={arm} spectrograph={spectrograph}\n".format(**dataId)
        titleStr += "RUN={run}".format(**dataId)
        ax.set_title(titleStr)
        im = ax.imshow(
            imagearray, vmin=self.config.plotMinChiMed, vmax=self.config.plotMaxChiMed, origin="lower"
        )
        plt.colorbar(im, shrink=0.8, format="%.2f")
        ax.text(
            0.0,
            0.015,
            f"mean={mean:.3} median={med:.3} sigma={sigma:.3}",
            bbox={"facecolor": "white", "alpha": 0.5},
            transform=ax.transAxes,
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
        titleStr = "visit={visit} arm={arm} spectrograph={spectrograph}".format(**dataId)
        fig.suptitle(titleStr)

        for i, x in enumerate(xc):
            for j, y in enumerate(yc):
                xmin = x - xwin
                xmax = x + xwin
                ymin = y - ywin
                ymax = y + ywin
                data = imagearray[ymin:ymax, xmin:xmax]
                im = ax[j, i].imshow(
                    data, vmin=q25, vmax=q75, origin="lower", extent=[xmin, xmax, ymin, ymax]
                )
                ax[j, i].set_xlabel("X (pix)")
                ax[j, i].set_ylabel("Y (pix)")

        qaImagePdf.append(fig, dpi=self.config.figureDpi, bbox_inches="tight")
        plt.close(fig)

        # a histogram of residual/reconstructed
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        titleStr = "visit={visit} arm={arm} spectrograph={spectrograph}\n".format(**dataId)
        titleStr += "RUN={run}".format(**dataId)
        fig.suptitle(titleStr)

        n, bins, _ = ax[0].hist(
            residual_over_reconst.flatten(),
            range=(self.config.plotHistRangeScale * q05_rr, self.config.plotHistRangeScale * q95_rr),
            bins=self.config.plotHistNbin,
            alpha=0.8,
            label="observed",
        )
        ax[0].set_xlim(self.config.plotHistRangeScale * q05_rr, self.config.plotHistRangeScale * q95_rr)
        ax[0].set_xlabel("residual / reconstructed")

        # a histogram of chi values
        n, bins, _patches = ax[1].hist(
            imagearray.flatten(),
            range=(self.config.plotHistRangeScale * q01, self.config.plotHistRangeScale * q99),
            bins=self.config.plotHistNbin,
            alpha=0.8,
            label="observed",
        )
        peak_idx = np.argmax(n)
        amplitude = n[peak_idx]
        x = np.linspace(bins[0], bins[-1], 2 * self.config.plotHistNbin)
        y = amplitude * np.exp(-0.5 * (x / 1.0) ** 2)
        ax[1].plot(x, y, ls="dashed", lw=2, c="k", alpha=0.8, label="N(0,1)")
        ax[1].set_xlim(self.config.plotHistRangeScale * q01, self.config.plotHistRangeScale * q99)
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
        xwin: int = 3,
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

        chi2 = np.nanmean(valid_chi_data**2) if valid_chi_data.size > 0 else np.nan

        im_ave = np.nanmean(valid_img_data) if valid_img_data.size > 0 else np.nan

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
    def getTargetColors() -> dict[TargetType, str]:
        """Get a map from `TargetType` to color name in matplotlib.

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
