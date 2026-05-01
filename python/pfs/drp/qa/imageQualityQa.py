"""Image quality QA pipeline task.

Produces per-detector FWHM and image-shape QA plots from arc-line
second-moment measurements, fiber profile calibrations, or direct
cross-dispersion moment measurements from post-ISR pixel data.
"""

import lsst.afw.image
import numpy as np
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
    Output as OutputConnection,
    PrerequisiteInput as PrerequisiteConnection,
)
from matplotlib import pyplot as plt
from pfs.drp.stella import ArcLineSet, DetectorMap, FiberProfileSet
from pfs.drp.stella.utils.quality import computeImageQuality, opaqueColorbar, plotImageQuality
from pfs.drp.stella.utils.stability import addTraceLambdaToArclines

from pfs.drp.qa.storageClasses import MultipagePdfFigure

_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # sigma → Gaussian-equivalent FWHM

__all__ = ["ImageQualityQaTask", "plotImageQuality"]


class ImageQualityQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Connections for ImageQualityQaTask."""

    arcLines = InputConnection(
        name="lines",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    detectorMap = InputConnection(
        name="detectorMap",
        doc="Calibrated detector mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    iqQaData = OutputConnection(
        name="iqQaData",
        doc="Per-line image quality measurements, including FWHM and position angle.",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    iqQaPlot = OutputConnection(
        name="iqQaPlot",
        doc="Image quality QA plots (one page per enabled plot type).",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc=(
            "Fiber profile shapes; used to derive trace-width FWHM when arc-line"
            " shape measurements are unavailable and no calexp is provided."
        ),
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )

    calexp = InputConnection(
        name="calexp",
        doc=(
            "Calibrated exposure output by ReduceExposureTask.  When present for"
            " non-arc visits, fiber profile widths are measured directly from"
            " pixel data via cross-dispersion 2nd moments rather than read from"
            " the calibration."
        ),
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        minimum=0,
    )


class ImageQualityQaConfig(PipelineTaskConfig, pipelineConnections=ImageQualityQaConnections):
    """Configuration for ImageQualityQaTask."""

    showWhisker = Field(
        dtype=bool,
        default=True,
        doc="Show a whisker/quiver plot of FWHM and position angle.",
    )
    showFWHM = Field(
        dtype=bool,
        default=False,
        doc="Show a 2D spatial hexbin map of FWHM.",
    )
    showFWHMAgainstLambda = Field(
        dtype=bool,
        default=False,
        doc="Plot FWHM vs log(flux) (or S/N if useSN is True).",
    )
    showFWHMHistogram = Field(
        dtype=bool,
        default=False,
        doc="Show a histogram of FWHM values.",
    )
    showFluxHistogram = Field(
        dtype=bool,
        default=False,
        doc="Show a histogram of line fluxes.",
    )
    useSN = Field(
        dtype=bool,
        default=False,
        doc="Use signal/noise rather than log10(flux) in the showFWHMAgainstLambda plot.",
    )
    minFluxPercentile = Field(
        dtype=float,
        default=10.0,
        doc="Minimum flux percentile for line selection in spatial plots.",
    )
    vmin = Field(dtype=float, default=2.5, doc="Minimum FWHM for color scale (pixels).")
    vmax = Field(dtype=float, default=3.5, doc="Maximum FWHM for color scale (pixels).")
    maxFwhm = Field(
        dtype=float,
        default=8.0,
        doc="Upper FWHM cutoff for line selection and histogram binning (pixels).",
    )
    logScale = Field(dtype=bool, default=True, doc="Log y-axis for histogram plots.")
    gridsize = Field(
        dtype=int,
        default=100,
        doc="Grid size for hexbin FWHM map. Use <=0 for scatter plot instead.",
    )
    stride = Field(
        dtype=int,
        default=1,
        doc="Fiber stride for downsampling arc lines in spatial plots.",
    )
    profileHalfWidth = Field(
        dtype=int,
        default=7,
        doc=(
            "Half-width in pixels of the cross-dispersion aperture used when"
            " measuring fiber profile widths directly from the calexp image."
        ),
    )
    profileYStride = Field(
        dtype=int,
        default=50,
        doc=(
            "Row sampling interval (pixels) for image-based fiber profile"
            " width measurements.  Smaller values give denser sampling at"
            " the cost of increased computation time."
        ),
    )


class ImageQualityQaTask(PipelineTask):
    """QA task measuring image quality from arc-line second moments.

    Reads per-detector arc line measurements (``lines`` dataset), enriches
    them with wavelength and trace-position information from the
    ``detectorMap``, computes Gaussian-equivalent FWHM metrics, and produces
    configurable QA plots as a ``MultipagePdfFigure`` (one page per enabled
    plot type).
    """

    ConfigClass = ImageQualityQaConfig
    _DefaultName = "imageQualityQa"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        dataId = dict(**inputRefs.arcLines.dataId.mapping)
        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = dataId
        try:
            outputs = self.run(**inputs)
        except Exception as e:
            self.log.error("ImageQualityQaTask failed for %s: %s", dataId, e)
            raise
        else:
            butlerQC.put(outputs, outputRefs)

    def run(
        self,
        arcLines: ArcLineSet,
        detectorMap: DetectorMap,
        fiberProfiles: FiberProfileSet | None,
        calexp: lsst.afw.image.Exposure | None,
        dataId: dict,
    ) -> Struct:
        """Compute image quality metrics and generate QA plots.

        For arc visits, FWHM is derived from 2D Gaussian second moments of
        the arc lines.  For non-arc visits (e.g. quartz or flat), arc-line
        shape measurements are unavailable.  The task then falls back in
        order of preference:

        1. Direct cross-dispersion 2nd-moment measurement from ``calexp``
           pixel data — reflects the actual optical state of that visit.
        2. Gaussian-equivalent FWHM from the fiber profile calibration stored
           in ``fiberProfiles`` — uses a stale calibration product and
           therefore cannot detect changes in focus between visits.

        Parameters
        ----------
        arcLines : `ArcLineSet`
            Arc line measurements.
        detectorMap : `DetectorMap`
            Calibrated detector mapping from fiberId,wavelength to x,y.
        fiberProfiles : `FiberProfileSet` or `None`
            Fiber profile shapes.  Used as fallback when ``calexp`` is absent.
        calexp : `lsst.afw.image.Exposure` or `None`
            Post-ISR calibrated image.  When provided for non-arc visits,
            fiber widths are measured directly from pixel data.
        dataId : `dict`
            The dataId for this quantum, used to label outputs and annotate
            the output DataFrame with ``visit``, ``arm``, ``spectrograph``.

        Returns
        -------
        iqQaData : `pandas.DataFrame`
            Per-line image quality data including ``fwhm``, ``theta``,
            ``traceOnly``, ``peakRatio``, plus ``visit``, ``arm``,
            ``spectrograph`` for downstream aggregation.
        iqQaPlot : `MultipagePdfFigure`
            QA plots, one page per enabled plot type.
        """
        self.log.info("Computing image quality metrics for %s", dataId)
        als = addTraceLambdaToArclines(arcLines, detectorMap)
        data = computeImageQuality(als)
        data["peakRatio"] = np.nan

        if not data["fwhm"].notna().any():
            if calexp is not None:
                self.log.info(
                    "No finite arc-line FWHM for %s; measuring profile widths from calexp.", dataId
                )
                data = self._buildImageWidthData(calexp, detectorMap)
            elif fiberProfiles is not None:
                self.log.info(
                    "No finite arc-line FWHM for %s; falling back to fiber profile calibration widths.",
                    dataId,
                )
                data = self._buildProfileData(fiberProfiles, detectorMap)
            else:
                self.log.warning(
                    "No finite arc-line FWHM for %s and neither calexp nor fiberProfiles available;"
                    " FWHM will be all NaN.",
                    dataId,
                )

        for key in ("visit", "arm", "spectrograph"):
            if key in dataId:
                data[key] = dataId[key]

        title = "{visit} {arm}{spectrograph}".format(**dataId)
        self.log.info("Generating image quality plots for %s", dataId)
        pdf = self._makePlots(data, title=title)

        return Struct(iqQaData=data, iqQaPlot=pdf)

    def _buildProfileData(self, fiberProfiles: FiberProfileSet, detectorMap: DetectorMap) -> pd.DataFrame:
        """Build an image-quality DataFrame from fiber profile trace widths.

        Computes Gaussian-equivalent FWHM (``2√(2 ln 2) × σ``) per swath from
        `FiberProfile.calculateStatistics` and maps each swath centre to
        detector (x, y) coordinates and wavelength via ``detectorMap``.

        Parameters
        ----------
        fiberProfiles : `FiberProfileSet`
            Fiber profile shapes from the Butler calibration.
        detectorMap : `DetectorMap`
            Calibrated detector mapping.

        Returns
        -------
        `pandas.DataFrame`
            Columns: ``fiberId``, ``x``, ``y``, ``lam``, ``fwhm``, ``theta``,
            ``flux``, ``fluxErr``, ``flag``, ``traceOnly``, ``peakRatio``.
        """
        rows_list, fibers_list, fwhm_list = [], [], []
        x_list, lam_list, flux_list = [], [], []

        for fiberId in fiberProfiles.fiberId:
            profile = fiberProfiles[fiberId]
            stats = profile.calculateStatistics()
            swath_rows = profile.rows
            n = len(swath_rows)
            if n == 0:
                continue

            fwhm = _FWHM_FACTOR * np.asarray(stats.width)

            fiberIds_arr = np.full(n, fiberId, dtype=np.int32)
            swath_rows_f64 = np.asarray(swath_rows, dtype=np.float64)
            x = detectorMap.getXCenter(fiberIds_arr, swath_rows_f64)
            lam = detectorMap.findWavelength(fiberIds_arr, swath_rows_f64)

            if profile.norm is not None and len(profile.norm) > 0:
                idx = np.clip(np.round(swath_rows_f64).astype(int), 0, len(profile.norm) - 1)
                flux = np.asarray(profile.norm[idx], dtype=float)
            else:
                flux = np.ma.sum(profile.profiles, axis=1).filled(np.nan)

            rows_list.append(swath_rows_f64)
            fibers_list.append(fiberIds_arr)
            fwhm_list.append(fwhm)
            x_list.append(x)
            lam_list.append(lam)
            flux_list.append(flux)

        if not rows_list:
            return pd.DataFrame(
                columns=[
                    "fiberId", "x", "y", "lam", "fwhm", "theta",
                    "flux", "fluxErr", "flag", "traceOnly", "peakRatio",
                ]
            )

        n_total = sum(len(r) for r in rows_list)
        return pd.DataFrame(
            {
                "fiberId": np.concatenate(fibers_list),
                "y": np.concatenate(rows_list),
                "x": np.concatenate(x_list),
                "lam": np.concatenate(lam_list),
                "fwhm": np.concatenate(fwhm_list),
                "theta": np.zeros(n_total),
                "flux": np.concatenate(flux_list),
                "fluxErr": np.ones(n_total),
                "flag": np.zeros(n_total, dtype=bool),
                "traceOnly": True,
                "peakRatio": np.full(n_total, np.nan),
            }
        )

    def _buildImageWidthData(
        self,
        calexp: lsst.afw.image.Exposure,
        detectorMap: DetectorMap,
    ) -> pd.DataFrame:
        """Measure fiber profile FWHM directly from post-ISR image pixel data.

        Samples the cross-dispersion intensity profile at regular row
        intervals, computes the background-subtracted 2nd moment (converting
        to a Gaussian-equivalent FWHM), and records the peak-to-total flux
        ratio as a secondary focus indicator.

        Unlike ``_buildProfileData``, this method reflects the actual optical
        state of the exposure rather than a (possibly stale) calibration.

        Parameters
        ----------
        calexp : `lsst.afw.image.Exposure`
            Post-ISR calibrated image for this quantum.
        detectorMap : `DetectorMap`
            Calibrated detector mapping used to locate fiber centers.

        Returns
        -------
        `pandas.DataFrame`
            Columns: ``fiberId``, ``x``, ``y``, ``lam``, ``fwhm``, ``theta``,
            ``flux``, ``fluxErr``, ``flag``, ``traceOnly``, ``peakRatio``.
            ``peakRatio`` is the fraction of background-subtracted flux in
            the peak pixel — a focus proxy independent of absolute brightness.
        """
        image = calexp.image.array.astype(np.float64)
        mask_arr = calexp.mask.array
        nRows, nCols = image.shape

        halfWidth = self.config.profileHalfWidth
        yStride = self.config.profileYStride
        badBits = calexp.mask.getPlaneBitMask(["BAD", "SAT", "CR", "NO_DATA"])

        fiberIds = np.asarray(detectorMap.fiberId, dtype=np.int32)
        y_samples = np.arange(yStride // 2, nRows, yStride, dtype=np.float64)

        all_fiberIds: list = []
        all_y: list = []
        all_x: list = []
        all_lam: list = []
        all_fwhm: list = []
        all_flux: list = []
        all_flag: list = []
        all_peakRatio: list = []

        for y_val in y_samples:
            y_int = int(round(y_val))
            if y_int < 0 or y_int >= nRows:
                continue

            y_arr = np.full(len(fiberIds), y_val, dtype=np.float64)
            x_centers = detectorMap.getXCenter(fiberIds, y_arr)
            lams = detectorMap.findWavelength(fiberIds, y_arr)
            row_image = image[y_int, :]
            row_mask = mask_arr[y_int, :]

            for fiberId, x_cen, lam in zip(fiberIds, x_centers, lams):
                if not (np.isfinite(x_cen) and np.isfinite(lam)):
                    continue

                x_lo = max(0, int(x_cen) - halfWidth)
                x_hi = min(nCols, int(x_cen) + halfWidth + 1)
                if x_hi - x_lo < 5:
                    continue

                strip = row_image[x_lo:x_hi]
                is_bad = (row_mask[x_lo:x_hi] & badBits) != 0

                fwhm_val = np.nan
                flag_val = True
                peak_ratio = np.nan
                flux_val = np.nan

                if is_bad.sum() <= halfWidth:
                    # Background from outermost 2 pixels on each side
                    edge = np.concatenate([strip[:2], strip[-2:]])
                    edge_bad = np.concatenate([is_bad[:2], is_bad[-2:]])
                    bg = float(np.nanmean(np.where(~edge_bad, edge, np.nan)))
                    if not np.isfinite(bg):
                        bg = 0.0

                    strip_bg = np.where(is_bad, 0.0, strip - bg)
                    total = float(strip_bg.sum())

                    if total > 0:
                        x_rel = np.arange(x_lo, x_hi, dtype=np.float64) - x_cen
                        mu = float((x_rel * strip_bg).sum()) / total
                        var = float(((x_rel - mu) ** 2 * strip_bg).sum()) / total
                        if var > 0:
                            fwhm_val = _FWHM_FACTOR * np.sqrt(var)
                            peak_ratio = float(strip_bg.max()) / total
                            flag_val = False
                        flux_val = total

                all_fiberIds.append(int(fiberId))
                all_y.append(y_val)
                all_x.append(float(x_cen))
                all_lam.append(float(lam))
                all_fwhm.append(fwhm_val)
                all_flux.append(flux_val)
                all_flag.append(flag_val)
                all_peakRatio.append(peak_ratio)

        n = len(all_fwhm)
        return pd.DataFrame(
            {
                "fiberId": np.array(all_fiberIds, dtype=np.int32),
                "y": np.array(all_y),
                "x": np.array(all_x),
                "lam": np.array(all_lam),
                "fwhm": np.array(all_fwhm),
                "theta": np.zeros(n),
                "flux": np.array(all_flux),
                "fluxErr": np.ones(n),
                "flag": np.array(all_flag, dtype=bool),
                "traceOnly": True,
                "peakRatio": np.array(all_peakRatio),
            }
        )

    def _makePlots(self, data: pd.DataFrame, title: str = "") -> MultipagePdfFigure:
        """Generate all enabled QA plots.

        Parameters
        ----------
        data : `pandas.DataFrame`
            Output of `computeImageQuality`, annotated with dataId fields.
        title : `str`, optional
            Figure suptitle.

        Returns
        -------
        `MultipagePdfFigure`
            Enabled plot types, one page each.
        """
        cfg = self.config
        pdf = MultipagePdfFigure()

        plotKwargs = dict(
            minFluxPercentile=cfg.minFluxPercentile,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            maxFwhm=cfg.maxFwhm,
            logScale=cfg.logScale,
            gridsize=cfg.gridsize,
            stride=cfg.stride,
            useSN=cfg.useSN,
        )

        for plotMode in (
            "showWhisker",
            "showFWHM",
            "showFWHMAgainstLambda",
            "showFWHMHistogram",
            "showFluxHistogram",
        ):
            if not getattr(cfg, plotMode):
                continue

            fig, ax = plt.subplots(layout="constrained")
            C, colorbarLabel = plotImageQuBesality(ax, data, **{plotMode: True}, **plotKwargs)
            if C is not None:
                with opaqueColorbar(C):
                    fig.colorbar(C, ax=ax, label=colorbarLabel)
            if title:
                fig.suptitle(title)
            pdf.append(fig)
            plt.close(fig)

        return pdf
