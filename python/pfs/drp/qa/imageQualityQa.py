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
        doc=(
            "Per-line image quality measurements, including FWHM and position angle."
            " Written as an empty DataFrame when ``writeIqQaData`` is False."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    iqQaMetrics = OutputConnection(
        name="iqQaMetrics",
        doc=(
            "Per-quantum summary metrics (one row) for downstream cross-visit"
            " aggregation by ImageQualityQaSummaryTask."
            " Columns: ``visit``, ``arm``, ``spectrograph``, ``medFwhm``,"
            " ``pctFlagged``, ``nLines``, ``traceOnly``."
        ),
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
        default=False,
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
    showFWHMvsLambda = Field(
        dtype=bool,
        default=True,
        doc=(
            "Plot FWHM vs wavelength (nm), with a per-10-nm binned median overlay."
            " Useful for detecting chromatic dependence of the PSF across arms."
        ),
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
    minGoodLines = Field(
        dtype=int,
        default=10,
        doc=(
            "Minimum number of good (unflagged, finite-FWHM) arc-line measurements"
            " required to trust the arc-line path.  When fewer good lines are found"
            " the task falls back to the calexp-based width measurement (or fiber"
            " profile calibration if no calexp is available).  This handles IIS"
            " frames and other sparse-illumination cases where the arc-line catalog"
            " does not match the lamp spectrum well enough to yield reliable shape"
            " measurements."
        ),
    )
    minPeakSN = Field(
        dtype=float,
        default=5.0,
        doc=(
            "Minimum peak signal-to-noise ratio required to accept a fiber"
            " profile sample in the calexp-based width measurement."
            " Noise is estimated from the 4 background edge pixels of the"
            " cross-dispersion strip.  Samples below this threshold — e.g."
            " scattered light in an IIS frame, or rows where a fiber is not"
            " illuminated — are flagged and excluded from the FWHM median."
            " Without this check pure-noise strips pass the total>0 gate"
            " roughly 50 % of the time and corrupt the FWHM distribution."
        ),
    )
    maxCalexpFlagRate = Field(
        dtype=float,
        default=0.5,
        doc=(
            "Maximum fraction of calexp profile samples that may be flagged"
            " before the calexp path is considered unreliable and discarded."
            " Continuum/quartz frames illuminate all fibers uniformly so the"
            " post-S/N-filter flag rate is low (< 10 %).  Arc-illuminated"
            " frames — including IIS — illuminate only a small number of"
            " fibers at discrete wavelengths; scattered light from those"
            " bright fibers can pass the S/N gate at many neighbouring"
            " positions and produce a large spurious FWHM (~7 px).  The"
            " resulting flag rate is typically > 90 %, well above the"
            " default threshold of 50 %, so the calexp results are rejected"
            " and the sparse arc-line data are kept instead."
        ),
    )
    writeIqQaData = Field(
        dtype=bool,
        default=False,
        doc=(
            "Write the full per-line ``iqQaData`` DataFrame to the Butler."
            " When False (default), an empty DataFrame is written to satisfy"
            " the output connection while avoiding the storage cost of the"
            " full line-level data.  Enable when per-line spatial plots or"
            " custom downstream analysis are needed."
        ),
    )
    fwhmWarnThreshold = Field(
        dtype=float,
        default=3.2,
        doc=(
            "Median FWHM (pixels) above which the per-quantum status is set to"
            " WARN.  Values above ``fwhmFailThreshold`` take priority and yield"
            " FAIL.  Set to a large value (e.g. 999) to disable."
        ),
    )
    fwhmFailThreshold = Field(
        dtype=float,
        default=3.5,
        doc=(
            "Median FWHM (pixels) above which the per-quantum status is set to"
            " FAIL.  Tuned for arm-b (400–650 nm); adjust for other arms."
            " Set to a large value (e.g. 999) to disable."
        ),
    )
    flagRateWarnThreshold = Field(
        dtype=float,
        default=15.0,
        doc=(
            "Percentage of flagged lines above which the per-quantum status is"
            " set to WARN.  Values above ``flagRateFailThreshold`` yield FAIL."
            " Set to 100 to disable."
        ),
    )
    flagRateFailThreshold = Field(
        dtype=float,
        default=20.0,
        doc=(
            "Percentage of flagged lines above which the per-quantum status is"
            " set to FAIL.  Set to 100 to disable."
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
            iq_data = outputs.iqQaData if self.config.writeIqQaData else pd.DataFrame()
            butlerQC.put(iq_data, outputRefs.iqQaData)
            butlerQC.put(outputs.iqQaMetrics, outputRefs.iqQaMetrics)
            butlerQC.put(outputs.iqQaPlot, outputRefs.iqQaPlot)

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
        the arc lines.  The arc-line path is used only when at least
        ``config.minGoodLines`` unflagged lines with finite FWHM are found;
        otherwise the task falls back in order of preference:

        1. Direct cross-dispersion 2nd-moment measurement from ``calexp``
           pixel data — reflects the actual optical state of that visit.
           This is the preferred fallback and handles IIS frames, quartz
           flats, and other sparse-illumination cases where the arc-line
           catalog does not match the lamp spectrum.
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
            ``spectrograph`` for downstream aggregation.  Written as an
            empty DataFrame when ``writeIqQaData`` is False.
        iqQaMetrics : `pandas.DataFrame`
            Single-row summary with ``medFwhm``, ``pctFlagged``, ``nLines``,
            ``traceOnly``, ``qaStatus``, ``visit``, ``arm``, ``spectrograph``.
            ``qaStatus`` is one of ``"PASS"``, ``"WARN"``, or ``"FAIL"`` based
            on the configured FWHM and flag-rate thresholds.
        iqQaPlot : `MultipagePdfFigure`
            QA plots, one page per enabled plot type.
        """
        self.log.info("Computing image quality metrics for %s", dataId)
        als = addTraceLambdaToArclines(arcLines, detectorMap)
        data = computeImageQuality(als)
        data["peakRatio"] = np.nan

        # Evaluate how many good (unflagged + finite-FWHM) arc-line measurements
        # are available.  IIS frames and lamp-mismatched visits often produce many
        # flagged lines with coincidental finite-fwhm values from partial fits, so
        # checking any-finite-fwhm is insufficient — we require minGoodLines good
        # measurements to trust the arc-line path.
        good_arc = data["fwhm"].notna() & ~data["flag"]
        if "status" in data.columns:
            good_arc &= data["status"] == 0
        n_good_arc = int(good_arc.sum())

        if n_good_arc < self.config.minGoodLines:
            if calexp is not None:
                self.log.info(
                    "Only %d good arc-line FWHM measurements for %s (< minGoodLines=%d);"
                    " attempting calexp-based profile widths.",
                    n_good_arc, dataId, self.config.minGoodLines,
                )
                calexp_data = self._buildImageWidthData(calexp, detectorMap)
                n_good_calexp = int((calexp_data["fwhm"].notna() & ~calexp_data["flag"]).sum())
                n_total_calexp = len(calexp_data)
                calexp_good_frac = n_good_calexp / max(n_total_calexp, 1)
                min_good_frac = 1.0 - self.config.maxCalexpFlagRate
                if n_good_calexp > 0 and calexp_good_frac >= min_good_frac:
                    self.log.info(
                        "calexp path gave %d good profile measurements"
                        " (%.1f%% good) for %s.",
                        n_good_calexp, 100.0 * calexp_good_frac, dataId,
                    )
                    data = calexp_data
                else:
                    # Too many samples flagged: sparse illumination (e.g. IIS
                    # arc frame).  Scattered light from the few bright fibers
                    # can pass the S/N gate at neighbouring positions and
                    # produce a spuriously large FWHM (~7 px).  Reject and
                    # keep the sparse arc-line data instead.
                    self.log.info(
                        "calexp path is too sparse for %s"
                        " (%d good, %.1f%% flagged >= maxCalexpFlagRate=%.0f%%);"
                        " retaining %d arc-line measurements.",
                        dataId, n_good_calexp,
                        100.0 * (1.0 - calexp_good_frac),
                        100.0 * self.config.maxCalexpFlagRate,
                        n_good_arc,
                    )
            elif fiberProfiles is not None:
                self.log.info(
                    "Only %d good arc-line FWHM measurements for %s (< minGoodLines=%d);"
                    " falling back to fiber profile calibration widths.",
                    n_good_arc, dataId, self.config.minGoodLines,
                )
                data = self._buildProfileData(fiberProfiles, detectorMap)
            else:
                self.log.warning(
                    "Only %d good arc-line FWHM measurements for %s (< minGoodLines=%d)"
                    " and neither calexp nor fiberProfiles available; FWHM will be sparse.",
                    n_good_arc, dataId, self.config.minGoodLines,
                )

        for key in ("visit", "arm", "spectrograph"):
            if key in dataId:
                data[key] = dataId[key]

        # Keep only the columns needed for downstream QA analysis and plotting.
        # Drops intermediate shape-moment columns (xx, yy, xy), catalog fields
        # not used in QA (wavelength, xErr, yErr, lamErr, tracePos, fluxNorm,
        # description), and any unnamed index artifact columns.
        _KEEP_COLUMNS = [
            "fiberId", "x", "y", "lam",
            "fwhm", "theta",
            "flux", "fluxErr", "flag",
            "traceOnly", "peakRatio",
            "status",
            "visit", "arm", "spectrograph",
        ]
        data = data[[c for c in _KEEP_COLUMNS if c in data.columns]]

        # Compute per-quantum summary metrics for downstream aggregation.
        good = ~data["flag"] & data["fwhm"].notna()
        if "status" in data.columns:
            good &= data["status"] == 0
        trace_only = bool(data["traceOnly"].all()) if "traceOnly" in data.columns else False
        med_fwhm = float(data.loc[good, "fwhm"].median())
        pct_flagged = 100.0 * data["flag"].sum() / max(len(data), 1)

        title = "{visit} {arm}{spectrograph}".format(**dataId)

        # Determine per-quantum pass/warn/fail status from absolute thresholds.
        # Trace-only visits use the same flag-rate check; FWHM check is skipped
        # when medFwhm is NaN (no valid lines).
        reasons = []
        fwhm_status = "PASS"
        if not trace_only and not np.isnan(med_fwhm):
            if med_fwhm >= self.config.fwhmFailThreshold:
                fwhm_status = "FAIL"
                reasons.append(f"medFWHM={med_fwhm:.2f}px >= fail threshold {self.config.fwhmFailThreshold}px")
            elif med_fwhm >= self.config.fwhmWarnThreshold:
                fwhm_status = "WARN"
                reasons.append(f"medFWHM={med_fwhm:.2f}px >= warn threshold {self.config.fwhmWarnThreshold}px")

        flag_status = "PASS"
        if pct_flagged >= self.config.flagRateFailThreshold:
            flag_status = "FAIL"
            reasons.append(
                f"pctFlagged={pct_flagged:.1f}% >= fail threshold {self.config.flagRateFailThreshold}%"
            )
        elif pct_flagged >= self.config.flagRateWarnThreshold:
            flag_status = "WARN"
            reasons.append(
                f"pctFlagged={pct_flagged:.1f}% >= warn threshold {self.config.flagRateWarnThreshold}%"
            )

        _level = {"PASS": 0, "WARN": 1, "FAIL": 2}
        qa_status = max((fwhm_status, flag_status), key=lambda s: _level[s])

        reason_str = "; ".join(reasons) if reasons else "all metrics nominal"
        self.log.info(
            "IQ QA %-4s  %s  medFWHM=%.2fpx  pctFlagged=%.1f%%  [%s]",
            qa_status, title, med_fwhm, pct_flagged, reason_str,
        )

        metrics = pd.DataFrame(
            {
                "medFwhm": [med_fwhm],
                "pctFlagged": [pct_flagged],
                "nLines": [len(data)],
                "traceOnly": [trace_only],
                "qaStatus": [qa_status],
            }
        )
        for key in ("visit", "arm", "spectrograph"):
            if key in dataId:
                metrics[key] = dataId[key]

        self.log.info("Generating image quality plots for %s", dataId)
        pdf = self._makePlots(data, title=title)

        return Struct(iqQaData=data, iqQaMetrics=metrics, iqQaPlot=pdf)

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
                    # Background from outermost 2 pixels on each side.
                    edge = np.concatenate([strip[:2], strip[-2:]])
                    edge_bad = np.concatenate([is_bad[:2], is_bad[-2:]])
                    good_edge = ~edge_bad
                    bg = float(np.nanmean(np.where(good_edge, edge, np.nan)))
                    if not np.isfinite(bg):
                        bg = 0.0

                    # Per-pixel noise from edge scatter; fall back to sqrt(|bg|)
                    # when fewer than 2 good edge pixels are available.
                    bg_rms = float(np.nanstd(np.where(good_edge, edge, np.nan)))
                    if not np.isfinite(bg_rms) or bg_rms <= 0:
                        bg_rms = max(1.0, np.sqrt(abs(bg)))

                    strip_bg = np.where(is_bad, 0.0, strip - bg)
                    peak_val = float(strip_bg.max())
                    total = float(strip_bg.sum())

                    # Require a minimum peak S/N before accepting this sample.
                    # Without this gate, pure-noise strips (e.g. dark rows in
                    # an IIS frame) pass the total>0 check ~50 % of the time
                    # and contribute garbage FWHM values.
                    if peak_val >= self.config.minPeakSN * bg_rms and total > 0:
                        x_rel = np.arange(x_lo, x_hi, dtype=np.float64) - x_cen
                        mu = float((x_rel * strip_bg).sum()) / total
                        var = float(((x_rel - mu) ** 2 * strip_bg).sum()) / total
                        if var > 0:
                            fwhm_val = _FWHM_FACTOR * np.sqrt(var)
                            peak_ratio = peak_val / total
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
            C, colorbarLabel = plotImageQuality(ax, data, **{plotMode: True}, **plotKwargs)
            if C is not None:
                with opaqueColorbar(C):
                    fig.colorbar(C, ax=ax, label=colorbarLabel)
            if title:
                fig.suptitle(title)
            pdf.append(fig)
            plt.close(fig)

        if cfg.showFWHMvsLambda:
            pdf.append(self._plotFWHMvsLambda(data, title=title))

        return pdf

    def _plotFWHMvsLambda(self, data: pd.DataFrame, title: str = "") -> plt.Figure:
        """Plot FWHM as a function of wavelength.

        Scatter-plots individual arc-line FWHM measurements against their
        wavelength, overlaying a per-10-nm binned median to reveal any
        chromatic PSF dependence.  Only unflagged lines below ``maxFwhm``
        are included.

        Parameters
        ----------
        data : `pandas.DataFrame`
            Output of `computeImageQuality`, annotated with dataId fields.
            Must contain ``lam``, ``fwhm``, and ``flag`` columns.
        title : `str`, optional
            Figure suptitle.

        Returns
        -------
        `matplotlib.figure.Figure`
        """
        cfg = self.config
        good = ~data["flag"] & np.isfinite(data["fwhm"]) & np.isfinite(data["lam"])
        good &= data["fwhm"] < cfg.maxFwhm

        fig, ax = plt.subplots(layout="constrained")

        if good.sum() == 0:
            ax.text(0.5, 0.5, "No valid measurements", transform=ax.transAxes,
                    ha="center", va="center")
            if title:
                fig.suptitle(title)
            return fig

        lam = data["lam"][good].to_numpy()
        fwhm = data["fwhm"][good].to_numpy()

        ax.scatter(lam, fwhm, s=1, alpha=0.15, color="steelblue", rasterized=True,
                   label="Arc lines")

        # Binned median overlay (10 nm bins)
        bin_width = 10.0
        lam_min = np.floor(lam.min() / bin_width) * bin_width
        lam_max = np.ceil(lam.max() / bin_width) * bin_width
        bin_edges = np.arange(lam_min, lam_max + bin_width, bin_width)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_medians = np.array([
            np.nanmedian(fwhm[(lam >= lo) & (lam < hi)])
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:])
        ])
        finite = np.isfinite(bin_medians)
        ax.plot(bin_centers[finite], bin_medians[finite], "o-", color="crimson",
                ms=4, lw=1.5, label="Median (10 nm bins)")

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("FWHM (px)")
        ax.set_title("FWHM vs Wavelength")
        ax.legend(markerscale=4, fontsize=8)
        ax.grid(alpha=0.3)
        if title:
            fig.suptitle(title)
        return fig
