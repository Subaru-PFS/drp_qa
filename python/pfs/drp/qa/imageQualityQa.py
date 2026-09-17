"""Image quality QA pipeline task.

Produces per-detector FWHM and image-shape QA plots from arc-line
second-moment measurements, fiber profile calibrations, or direct
cross-dispersion moment measurements from post-ISR pixel data.
"""

import re
from typing import Any

import lsst.afw.image
import numpy as np
import pandas as pd
from lsst.pex.config import DictField, Field
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

from pfs.datamodel import FiberStatus, PfsConfig, TargetType
from pfs.drp.stella import ArcLineSet, DetectorMap, FiberProfileSet
from pfs.drp.stella.utils.quality import computeImageQuality
from pfs.drp.stella.utils.stability import addTraceLambdaToArclines

_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # sigma → Gaussian-equivalent FWHM

__all__ = ["ImageQualityQaTask"]


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
            "Per-line image quality measurements, including FWHM,"
            " position angle, and spatial offset (dxCenter)."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    iqQaMetrics = OutputConnection(
        name="iqQaMetrics",
        doc=(
            "Per-quantum summary metrics (one row)."
            " Columns: ``visit``, ``arm``, ``spectrograph``, ``medFwhm``,"
            " ``medDxCenter``, ``dxCenterRms``, ``pctFlagged``,"
            " ``pctLowSN``, ``pctMeasFail``,"
            " ``nLines``, ``traceOnly``, ``qaStatus``."
        ),
        storageClass="DataFrame",
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

    detectorMapCalib = PrerequisiteConnection(
        name="detectorMap_calib",
        doc=(
            "Static calibration detectorMap (pre-adjustment).  Used to"
            " measure the spatial offset between calibrated and actual"
            " fiber positions as a flexure diagnostic (dxCenter)."
        ),
        storageClass="DetectorMap",
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

    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc=(
            "Fiber configuration for this visit.  When provided alongside"
            " ``calexp``, only fibers with ``targetType=FLUXSTD`` and"
            " ``fiberStatus=GOOD`` are used for calexp-based FWHM measurement,"
            " avoiding contamination from dark sky fibers."
        ),
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
    )

    isrLog = InputConnection(
        name="isr_log",
        doc="ISR task execution log",
        storageClass="ButlerLogRecords",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        minimum=0,
    )

    cosmicrayLog = InputConnection(
        name="cosmicray_log",
        doc="Cosmic Ray task execution log",
        storageClass="ButlerLogRecords",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        minimum=0,
    )

    reduceExposureLog = InputConnection(
        name="reduceExposure_log",
        doc="reduceExposure task execution log",
        storageClass="ButlerLogRecords",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        minimum=0,
    )


class ImageQualityQaConfig(PipelineTaskConfig, pipelineConnections=ImageQualityQaConnections):
    """Configuration for ImageQualityQaTask."""

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
            " This threshold is not used when pfsConfig FLUXSTD filtering is"
            " active; use ``minFluxstdGoodFrac`` instead."
        ),
    )
    minFluxstdGoodFrac = Field(
        dtype=float,
        default=0.10,
        doc=(
            "Minimum fraction of FLUXSTD calexp profile samples that must pass"
            " the S/N gate (``minPeakSN``) for the FLUXSTD-filtered calexp path"
            " to be considered reliable.  When pfsConfig is provided, only"
            " FLUXSTD+GOOD fibers are sampled; on frames where the standard"
            " stars are faint (sky frames with low S/N), fewer than 10 % of"
            " sampled positions reach S/N threshold and the resulting FWHM"
            " estimate is unreliable.  In that case the FWHM is treated as"
            " sparse (NaN) and no pass/fail status is assigned.  On bright-star"
            " or arc frames where FLUXSTD fibers are well-lit, the good fraction"
            " is typically > 10 % and the FWHM estimate is used.  The"
            " pctFlagged metric is always suppressed (NaN) when the FLUXSTD"
            " path is active, because the flag rate for stellar fibers reflects"
            " exposure depth rather than optical quality."
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
    flagRateWarnThreshold = DictField(
        keytype=str,
        itemtype=float,
        default={
            "b": 50.0,
            "r": 15.0,
            "n": 15.0,
            "m": 15.0,
            "b:HgCd": 15.0,
            "b:Neon": 50.0,
            "b:Krypton": 55.0,
            "b:Xenon": 85.0,
            "b:Argon": 93.0,
        },
        doc=(
            "Percentage of flagged lines above which the per-quantum status"
            " is set to WARN.  Values above ``flagRateFailThreshold`` yield"
            " FAIL.  Keys can be an arm name (``b``) or ``arm:species``"
            " (``b:HgCd``) for lamp-specific overrides.  The species is"
            " extracted from W_SEQNAM (e.g. 'Arc: HgCd' → 'HgCd')."
            " Lookup order: ``arm:species`` → ``arm`` → 15.0."
            " Set to 100 to disable."
        ),
    )
    flagRateFailThreshold = DictField(
        keytype=str,
        itemtype=float,
        default={
            "b": 60.0,
            "r": 20.0,
            "n": 20.0,
            "m": 20.0,
            "b:HgCd": 25.0,
            "b:Neon": 60.0,
            "b:Krypton": 65.0,
            "b:Xenon": 92.0,
            "b:Argon": 97.0,
        },
        doc=(
            "Percentage of flagged lines above which the per-quantum status"
            " is set to FAIL.  Keys can be an arm name (``b``) or"
            " ``arm:species`` (``b:HgCd``) for lamp-specific overrides."
            " Lookup order: ``arm:species`` → ``arm`` → 20.0."
            " Set to 100 to disable."
        ),
    )
    dxCenterWarnThreshold = Field(
        dtype=float,
        default=1.0,
        doc=(
            "Median absolute spatial offset |dxCenter| (pixels) between"
            " the static calibration detectorMap and measured fiber"
            " positions above which the per-quantum status is set to WARN."
        ),
    )
    dxCenterFailThreshold = Field(
        dtype=float,
        default=2.0,
        doc=(
            "Median absolute spatial offset |dxCenter| (pixels) between"
            " the static calibration detectorMap and measured fiber"
            " positions above which the per-quantum status is set to FAIL."
        ),
    )
    maxFluxJitterPct = Field(
        dtype=float,
        default=15.0,
        doc=(
            "Maximum allowed flux jitter percentage (fluxStd/medFlux*100) for"
            " non-flagged arc lines before qaStatus is degraded to WARN."
            " Values above twice this threshold yield FAIL."
            " Set to a large value (e.g. 999) to disable."
        ),
    )
    maxSaturatedLines = Field(
        dtype=int,
        default=50,
        doc=(
            "Maximum number of arc lines with flux above the saturation proxy"
            " threshold (95th percentile of the flux distribution) before"
            " qaStatus is degraded to WARN."
            " Set to a large value (e.g. 99999) to disable."
        ),
    )


class ImageQualityQaTask(PipelineTask):
    """QA task measuring image quality from arc-line second moments.

    Reads per-detector arc line measurements (``lines`` dataset), enriches
    them with wavelength and trace-position information from the
    ``detectorMap``, and computes Gaussian-equivalent FWHM and flexure
    metrics.
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
        # Fetch pfsConfig separately before the bulk get so that a missing
        # dataset degrades gracefully to None rather than aborting the quantum.
        pfsConfig = None
        try:
            pfsConfig = butlerQC.get(inputRefs.pfsConfig)
        except Exception:
            pass
        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = dataId
        inputs["pfsConfig"] = pfsConfig
        try:
            # Perform the actual processing.
            outputs = self.run(**inputs)
        except ValueError as e:
            # An expected processing failure is logged and the outputs are
            # omitted, so one bad quantum does not abort the QA pipeline.
            # Unexpected exceptions are left to propagate.
            self.log.error("ImageQualityQaTask failed for %s: %s", dataId, e)
        else:
            butlerQC.put(outputs.iqQaData, outputRefs.iqQaData)
            butlerQC.put(outputs.iqQaMetrics, outputRefs.iqQaMetrics)

    def run(
        self,
        arcLines: ArcLineSet,
        detectorMap: DetectorMap,
        fiberProfiles: FiberProfileSet | None,
        detectorMapCalib: DetectorMap | None,
        calexp: lsst.afw.image.Exposure | None,
        pfsConfig: PfsConfig | None,
        dataId: dict,
        isrLog: Any = None,
        cosmicrayLog: Any = None,
        reduceExposureLog: Any = None,
    ) -> Struct:
        """Compute image quality metrics and generate QA plots.

        The measurement path is chosen based on the observation type derived
        from ``W_SEQTYP`` and lamp headers (see `_classifyVisit`):

        * **Regular arc** (``scienceArc``, all fibers): arc-line second moments
          are the primary path.  If fewer than ``config.minGoodLines`` good
          measurements are found (e.g. blue arm where catalog lines are sparse),
          a calexp cross-dispersion fallback is attempted.
        * **IIS arc** (``scienceArc``, engineering fibers): the science arc
          catalog does not match the 16 IIS positions; FWHM is reported as
          sparse (no value, no pass/fail).
        * **Regular trace/quartz** (``scienceTrace``, all fibers): calexp
          cross-dispersion moments are the primary path; fiber profile
          calibration is the secondary fallback.
        * **IIS trace/quartz** (``scienceTrace``, engineering fibers): too few
          fibers for a reliable measurement; reported as sparse.
        * **Science / all-sky** (``scienceObject*``, no lamps): when
          ``pfsConfig`` is available, FWHM is measured from FLUXSTD+GOOD
          fibers in ``calexp`` and ``pctFlagged`` is suppressed (stellar fibers
          are not continuously illuminated so the flag rate reflects exposure
          depth, not optical quality).  If the FLUXSTD good fraction is below
          ``config.minFluxstdGoodFrac``, FWHM is treated as sparse.
        * **Unknown** (``W_SEQTYP`` absent): falls back to the original
          heuristic based on ``n_good_arc`` and ``maxCalexpFlagRate``.

        Parameters
        ----------
        arcLines : `ArcLineSet`
            Arc line measurements.
        detectorMap : `DetectorMap`
            Adjusted detector mapping from fiberId,wavelength to x,y.
        fiberProfiles : `FiberProfileSet` or `None`
            Fiber profile shapes.  Used as fallback for regular trace/quartz
            visits when ``calexp`` is absent.
        detectorMapCalib : `DetectorMap` or `None`
            Static calibration detectorMap (pre-adjustment).  Used to
            compute the spatial offset ``dxCenter`` as a flexure diagnostic.
        calexp : `lsst.afw.image.Exposure` or `None`
            Post-ISR calibrated image.
        pfsConfig : `PfsConfig` or `None`
            Fiber configuration for this visit.  Required for science/allsky
            FLUXSTD path; also supplies ``W_SEQTYP`` when ``calexp`` is absent.
        dataId : `dict`
            The dataId for this quantum, used to label outputs and annotate
            the output DataFrame with ``visit``, ``arm``, ``spectrograph``.

        Returns
        -------
        iqQaData : `pandas.DataFrame`
            Per-line image quality data including ``dxCenter`` (spatial
            offset from static calibration detectorMap, in pixels).
        iqQaMetrics : `pandas.DataFrame`
            Single-row summary with ``medFwhm``, ``medDxCenter``,
            ``dxCenterRms``, ``pctFlagged``, ``nLines``, ``traceOnly``,
            ``obsType``, ``seqName``, ``qaStatus``, ``visit``, ``arm``,
            ``spectrograph``.
        """
        self.log.info("Computing image quality metrics for %s", dataId)

        # Classify the observation type from FITS header metadata so that the
        # task can route each visit to the correct measurement path explicitly
        # rather than relying on heuristics like the arc-line count.
        obs_type, is_iis, seq_nam = self._classifyVisit(calexp, pfsConfig)
        self.log.debug(
            "Visit classification: obs_type=%r is_iis=%s seq_nam=%r for %s", obs_type, is_iis, seq_nam, dataId
        )

        als = addTraceLambdaToArclines(arcLines, detectorMap)
        data = computeImageQuality(als)
        data["peakRatio"] = np.nan

        # Flexure diagnostic: offset from static calibration detectorMap.
        # Positive dxCenter means the calibration prediction is to the right
        # of the actual measured fiber position.
        if detectorMapCalib is not None and "x" in data.columns:
            calibX = detectorMapCalib.getXCenter(
                np.asarray(data["fiberId"], dtype=np.int32),
                np.asarray(data["y"], dtype=np.float64),
            )
            data["dxCenter"] = calibX - data["x"]
        else:
            data["dxCenter"] = np.nan

        # Count good (unflagged + finite-FWHM) arc-line measurements.
        good_arc = data["fwhm"].notna() & ~data["flag"]
        if "status" in data.columns:
            good_arc &= data["status"] == 0
        n_good_arc = int(good_arc.sum())

        dense_data = False
        using_fluxstd_filter = False
        # force_sparse bypasses the n_good_arc check for visit types where we
        # know the arc catalog will not match (IIS arcs/traces).
        force_sparse = False

        if obs_type == "arc" and not is_iis:
            # Regular arc (all 600 fibers, arc lamp): primary path is the
            # arc-line shape measurements.  Fall back to calexp only when too
            # few good catalog matches are found (e.g. b arm on sky fields
            # where arc lines are not in the catalog).
            if n_good_arc >= self.config.minGoodLines:
                self.log.debug(
                    "Regular arc %s: using %d good arc-line measurements.",
                    dataId,
                    n_good_arc,
                )
            elif calexp is not None:
                self.log.info(
                    "Regular arc %s: only %d good arc lines (< minGoodLines=%d); trying calexp fallback.",
                    dataId,
                    n_good_arc,
                    self.config.minGoodLines,
                )
                calexp_data = self._buildImageWidthData(calexp, detectorMap, detectorMapCalib)
                n_good_calexp = int((calexp_data["fwhm"].notna() & ~calexp_data["flag"]).sum())
                calexp_good_frac = n_good_calexp / max(len(calexp_data), 1)
                if n_good_calexp > 0 and (1.0 - calexp_good_frac) < self.config.maxCalexpFlagRate:
                    self.log.info(
                        "Regular arc calexp fallback: %d good samples (%.1f%% good) for %s.",
                        n_good_calexp,
                        100.0 * calexp_good_frac,
                        dataId,
                    )
                    data = calexp_data
                    dense_data = True
                else:
                    self.log.warning(
                        "Regular arc calexp fallback too sparse for %s"
                        " (%d good, %.1f%% flagged); keeping %d arc-line measurements.",
                        dataId,
                        n_good_calexp,
                        100.0 * (1.0 - calexp_good_frac),
                        n_good_arc,
                    )
            else:
                self.log.warning(
                    "Regular arc %s: only %d good arc lines (< minGoodLines=%d)"
                    " and no calexp; FWHM will be sparse.",
                    dataId,
                    n_good_arc,
                    self.config.minGoodLines,
                )

        elif obs_type == "arc" and is_iis:
            # IIS arc (16 engineering fibers): the science arc-line catalog
            # does not match the engineering fiber positions, so n_good_arc is
            # always near zero.  Skip calexp too — scattered light from the 16
            # bright fibers produces a spuriously large (~7 px) FWHM.
            force_sparse = True
            self.log.info(
                "IIS arc %s: science arc catalog does not match engineering"
                " fibers; reporting sparse (no FWHM).",
                dataId,
            )

        elif obs_type == "trace" and not is_iis:
            # Regular quartz/trace (all fibers, quartz lamp): no arc lines to
            # fit, so use calexp cross-dispersion moments as the primary path.
            # Any arc-line rows still held in ``data`` describe a catalog that
            # does not apply to a quartz frame, so the visit stays sparse
            # unless one of the two measurement paths succeeds.
            force_sparse = True
            if calexp is not None:
                self.log.info(
                    "Regular trace/quartz %s: measuring FWHM from calexp.",
                    dataId,
                )
                calexp_data = self._buildImageWidthData(calexp, detectorMap, detectorMapCalib)
                n_good_calexp = int((calexp_data["fwhm"].notna() & ~calexp_data["flag"]).sum())
                calexp_good_frac = n_good_calexp / max(len(calexp_data), 1)
                if n_good_calexp > 0 and (1.0 - calexp_good_frac) < self.config.maxCalexpFlagRate:
                    self.log.info(
                        "Quartz calexp: %d good samples (%.1f%% good) for %s.",
                        n_good_calexp,
                        100.0 * calexp_good_frac,
                        dataId,
                    )
                    data = calexp_data
                    dense_data = True
                    force_sparse = False
                else:
                    self.log.warning(
                        "Quartz calexp too sparse for %s (%d good, %.1f%% flagged); FWHM will be sparse.",
                        dataId,
                        n_good_calexp,
                        100.0 * (1.0 - calexp_good_frac),
                    )
            elif fiberProfiles is not None:
                self.log.info(
                    "Regular trace/quartz %s: no calexp; falling back to fiber profile calibration widths.",
                    dataId,
                )
                data = self._buildProfileData(fiberProfiles, detectorMap)
                dense_data = True
                force_sparse = False
            else:
                self.log.warning(
                    "Regular trace/quartz %s: no calexp and no fiberProfiles; FWHM will be sparse.",
                    dataId,
                )

        elif obs_type == "trace" and is_iis:
            # IIS quartz (16 engineering fibers): too few fibers for a
            # reliable full-detector calexp measurement.
            force_sparse = True
            self.log.info("IIS trace/quartz %s: too few fibers; reporting sparse.", dataId)

        elif obs_type in ("science", "allsky"):
            # Science or all-sky plate: no arc lamp, fibers point at sky or
            # targets.  Use FLUXSTD fibers (bright standard stars) sampled
            # from calexp for FWHM when pfsConfig is available.
            #
            # There is no arc lamp, so whatever arc-line rows survived in
            # ``data`` are incidental catalog matches that say nothing about
            # the optics.  Start sparse and only clear that once the FLUXSTD
            # calexp measurement has been accepted.
            force_sparse = True
            if calexp is not None and pfsConfig is not None:
                good_mask = (pfsConfig.targetType == TargetType.FLUXSTD) & (
                    pfsConfig.fiberStatus == FiberStatus.GOOD
                )
                fluxstd_ids: set = {int(f) for f in pfsConfig.fiberId[good_mask]}
                self.log.info(
                    "%s %s: measuring FWHM from %d FLUXSTD fibers in calexp.",
                    obs_type.capitalize(),
                    dataId,
                    len(fluxstd_ids),
                )
                calexp_data = self._buildImageWidthData(
                    calexp, detectorMap, detectorMapCalib, fiberIds=fluxstd_ids
                )
                n_good_calexp = int((calexp_data["fwhm"].notna() & ~calexp_data["flag"]).sum())
                n_total_calexp = len(calexp_data)
                calexp_good_frac = n_good_calexp / max(n_total_calexp, 1)
                if n_good_calexp > 0 and calexp_good_frac >= self.config.minFluxstdGoodFrac:
                    self.log.info(
                        "FLUXSTD calexp: %d good samples (%.1f%% good) for %s.",
                        n_good_calexp,
                        100.0 * calexp_good_frac,
                        dataId,
                    )
                    data = calexp_data
                    dense_data = True
                    using_fluxstd_filter = True
                    force_sparse = False
                else:
                    self.log.info(
                        "FLUXSTD calexp too sparse for %s"
                        " (%d/%d = %.1f%% < minFluxstdGoodFrac=%.0f%%); FWHM will be sparse.",
                        dataId,
                        n_good_calexp,
                        n_total_calexp,
                        100.0 * calexp_good_frac,
                        100.0 * self.config.minFluxstdGoodFrac,
                    )
            else:
                self.log.info(
                    "%s %s: no calexp or pfsConfig available; FWHM will be sparse.",
                    obs_type.capitalize(),
                    dataId,
                )

        else:
            # obs_type == "unknown": W_SEQTYP header absent or unrecognised.
            # Fall back to heuristic: try arc-line path; if too few good lines,
            # try calexp (FLUXSTD-filtered if pfsConfig is available), then
            # fiberProfiles.
            self.log.debug(
                "Visit type unknown for %s; using heuristic fallback (n_good_arc=%d).",
                dataId,
                n_good_arc,
            )
            if n_good_arc < self.config.minGoodLines:
                if calexp is not None:
                    fluxstd_ids_unk: set | None = None
                    if pfsConfig is not None:
                        good_mask = (pfsConfig.targetType == TargetType.FLUXSTD) & (
                            pfsConfig.fiberStatus == FiberStatus.GOOD
                        )
                        fluxstd_ids_unk = {int(f) for f in pfsConfig.fiberId[good_mask]}
                        self.log.info(
                            "Unknown type %s: %d good arc lines (< %d); trying calexp"
                            " with %d FLUXSTD fibers.",
                            dataId,
                            n_good_arc,
                            self.config.minGoodLines,
                            len(fluxstd_ids_unk),
                        )
                    else:
                        self.log.info(
                            "Unknown type %s: %d good arc lines (< %d); trying calexp.",
                            dataId,
                            n_good_arc,
                            self.config.minGoodLines,
                        )
                    calexp_data = self._buildImageWidthData(
                        calexp,
                        detectorMap,
                        detectorMapCalib,
                        fiberIds=fluxstd_ids_unk,
                    )
                    n_good_calexp = int((calexp_data["fwhm"].notna() & ~calexp_data["flag"]).sum())
                    n_total_calexp = len(calexp_data)
                    calexp_good_frac = n_good_calexp / max(n_total_calexp, 1)
                    min_good_frac = (
                        self.config.minFluxstdGoodFrac
                        if fluxstd_ids_unk is not None
                        else 1.0 - self.config.maxCalexpFlagRate
                    )
                    if n_good_calexp > 0 and calexp_good_frac >= min_good_frac:
                        self.log.info(
                            "Unknown type calexp path: %d good samples (%.1f%% good) for %s.",
                            n_good_calexp,
                            100.0 * calexp_good_frac,
                            dataId,
                        )
                        data = calexp_data
                        dense_data = True
                        using_fluxstd_filter = fluxstd_ids_unk is not None
                    elif fluxstd_ids_unk is not None:
                        self.log.info(
                            "Unknown type FLUXSTD calexp too sparse for %s"
                            " (%d/%d = %.1f%% < %.0f%%); FWHM will be sparse.",
                            dataId,
                            n_good_calexp,
                            n_total_calexp,
                            100.0 * calexp_good_frac,
                            100.0 * self.config.minFluxstdGoodFrac,
                        )
                    else:
                        self.log.info(
                            "Unknown type calexp too sparse for %s"
                            " (%d good, %.1f%% flagged); keeping arc-line data.",
                            dataId,
                            n_good_calexp,
                            100.0 * (1.0 - calexp_good_frac),
                        )
                elif fiberProfiles is not None:
                    self.log.info(
                        "Unknown type %s: %d good arc lines (< %d); using fiberProfiles.",
                        dataId,
                        n_good_arc,
                        self.config.minGoodLines,
                    )
                    data = self._buildProfileData(fiberProfiles, detectorMap)
                    dense_data = True
                else:
                    self.log.warning(
                        "Unknown type %s: %d good arc lines (< %d) and no calexp or"
                        " fiberProfiles; FWHM will be sparse.",
                        dataId,
                        n_good_arc,
                        self.config.minGoodLines,
                    )

        for key in ("visit", "arm", "spectrograph"):
            if key in dataId:
                data[key] = dataId[key]

        # Keep only the columns needed for downstream QA analysis and plotting.
        # Drops intermediate shape-moment columns (xx, yy, xy), catalog fields
        # not used in QA (wavelength, xErr, yErr, lamErr, tracePos, fluxNorm,
        # description), and any unnamed index artifact columns.
        _KEEP_COLUMNS = [
            "fiberId",
            "x",
            "y",
            "lam",
            "fwhm",
            "theta",
            "dxCenter",
            "flux",
            "fluxErr",
            "flag",
            "traceOnly",
            "peakRatio",
            "status",
            "visit",
            "arm",
            "spectrograph",
        ]
        data = data[[c for c in _KEEP_COLUMNS if c in data.columns]]

        # Compute per-quantum summary metrics for downstream aggregation.
        good = ~data["flag"] & data["fwhm"].notna()
        if "status" in data.columns:
            good &= data["status"] == 0
        if force_sparse:
            # The visit type rules out every row that is left: on IIS frames
            # the arc catalog does not describe the illuminated fibers, and on
            # quartz/science frames without a usable calexp the surviving arc
            # lines are incidental.  The rows are still written to ``iqQaData``
            # for inspection, but no summary metric may be derived from them.
            good = pd.Series(False, index=data.index)
        trace_only = bool(data["traceOnly"].all()) if "traceOnly" in data.columns else False
        med_fwhm = float(data.loc[good, "fwhm"].median())

        if "dxCenter" in data.columns:
            goodDx = good & data["dxCenter"].notna()
            medDxCenter = float(data.loc[goodDx, "dxCenter"].median())
            dxCenterRms = float(data.loc[goodDx, "dxCenter"].std())
        else:
            medDxCenter = np.nan
            dxCenterRms = np.nan

        # pctFlagged is only meaningful when data coverage is dense (calexp,
        # fiber-profile path, or ≥ minGoodLines arc lines with full-detector
        # illumination).  For sparse illumination (IIS arcs/traces, or
        # lamp-mismatched arc frames) pctFlagged reflects catalog quality
        # rather than optical quality and is set to NaN.
        #
        # The FLUXSTD calexp path also suppresses pctFlagged: stellar fibers
        # only reach S/N threshold at a small fraction of sampled rows, so the
        # flag rate reflects exposure depth rather than optical quality.  FWHM
        # is still reported when the good-fraction threshold is met.
        sparse_fallback = force_sparse or ((n_good_arc < self.config.minGoodLines) and not dense_data)
        if sparse_fallback or using_fluxstd_filter:
            pct_flagged = np.nan
            flagBreakdown = {}
        else:
            nTotal = max(len(data), 1)
            pct_flagged = 100.0 * data["flag"].sum() / nTotal
            flagBreakdown = self._computeFlagBreakdown(
                data,
                nTotal,
                isArcLinePath=not dense_data,
            )

        title = "{visit} {arm}{spectrograph}".format(**dataId)

        # Determine per-quantum pass/warn/fail status from absolute thresholds.
        # Trace-only visits use the same flag-rate check; FWHM check is skipped
        # when medFwhm is NaN (no valid lines).
        reasons = []
        fwhm_status = "PASS"
        if not trace_only and not np.isnan(med_fwhm):
            if med_fwhm >= self.config.fwhmFailThreshold:
                fwhm_status = "FAIL"
                reasons.append(
                    f"medFWHM={med_fwhm:.2f}px >= fail threshold {self.config.fwhmFailThreshold}px"
                )
            elif med_fwhm >= self.config.fwhmWarnThreshold:
                fwhm_status = "WARN"
                reasons.append(
                    f"medFWHM={med_fwhm:.2f}px >= warn threshold {self.config.fwhmWarnThreshold}px"
                )

        flag_status = "PASS"
        if np.isfinite(pct_flagged):
            arm = dataId.get("arm", "")
            species = seq_nam.split(":", 1)[-1].strip() if ":" in seq_nam else ""
            compoundKey = f"{arm}:{species}" if species else ""
            warn_thresh = self.config.flagRateWarnThreshold.get(
                compoundKey, self.config.flagRateWarnThreshold.get(arm, 15.0)
            )
            fail_thresh = self.config.flagRateFailThreshold.get(
                compoundKey, self.config.flagRateFailThreshold.get(arm, 20.0)
            )
            if pct_flagged >= fail_thresh:
                flag_status = "FAIL"
                reasons.append(f"pctFlagged={pct_flagged:.1f}% >= fail threshold {fail_thresh}%")
            elif pct_flagged >= warn_thresh:
                flag_status = "WARN"
                reasons.append(f"pctFlagged={pct_flagged:.1f}% >= warn threshold {warn_thresh}%")

        dx_status = "PASS"
        if np.isfinite(medDxCenter):
            absDx = abs(medDxCenter)
            if absDx >= self.config.dxCenterFailThreshold:
                dx_status = "FAIL"
                reasons.append(
                    f"|dxCenter|={absDx:.3f}px >= fail threshold {self.config.dxCenterFailThreshold}px"
                )
            elif absDx >= self.config.dxCenterWarnThreshold:
                dx_status = "WARN"
                reasons.append(
                    f"|dxCenter|={absDx:.3f}px >= warn threshold {self.config.dxCenterWarnThreshold}px"
                )

        # Compute arc flux jitter and saturation metrics from non-flagged lines,
        # broken down per arc species (description).  Overall (all-species) values
        # are also stored for backward compatibility.  Guard against all-NaN flux
        # (e.g. trace frames) — emit NaN without crashing.
        flux_jitter_pct = np.nan
        n_saturated = np.nan
        flux_status = "PASS"
        # Per-species flux metrics: {species: (fluxJitterPct, nSaturated)}
        species_flux_metrics: dict[str, tuple[float, float]] = {}
        if not force_sparse and hasattr(arcLines, "flux") and hasattr(arcLines, "description"):
            good_base = (arcLines.flag == 0) & np.isfinite(arcLines.flux)
            # Overall (all-species combined)
            if good_base.sum() >= 2:
                good_flux = arcLines.flux[good_base]
                med_flux = float(np.median(good_flux))
                flux_std = float(np.std(good_flux))
                if med_flux > 0:
                    flux_jitter_pct = flux_std / med_flux * 100.0
                sat_threshold = float(np.percentile(good_flux, 95))
                n_saturated = int((good_flux > sat_threshold).sum())
            # Per-species breakdown
            for sp in np.unique(arcLines.description):
                sp_mask = good_base & (arcLines.description == sp)
                if sp_mask.sum() < 2:
                    species_flux_metrics[sp] = (np.nan, np.nan)
                    continue
                sp_flux = arcLines.flux[sp_mask]
                sp_med = float(np.median(sp_flux))
                sp_std = float(np.std(sp_flux))
                sp_jitter = sp_std / sp_med * 100.0 if sp_med > 0 else np.nan
                sp_sat_thresh = float(np.percentile(sp_flux, 95))
                sp_nsat = int((sp_flux > sp_sat_thresh).sum())
                species_flux_metrics[sp] = (sp_jitter, sp_nsat)
            # Evaluate qaStatus from overall metrics
            if not np.isnan(flux_jitter_pct):
                if flux_jitter_pct >= 2.0 * self.config.maxFluxJitterPct:
                    flux_status = "FAIL"
                    reasons.append(
                        f"fluxJitterPct={flux_jitter_pct:.1f}% >= fail threshold"
                        f" {2.0 * self.config.maxFluxJitterPct:.1f}%"
                    )
                elif flux_jitter_pct >= self.config.maxFluxJitterPct:
                    flux_status = "WARN"
                    reasons.append(
                        f"fluxJitterPct={flux_jitter_pct:.1f}% >= warn threshold"
                        f" {self.config.maxFluxJitterPct:.1f}%"
                    )
            if not np.isnan(n_saturated) and n_saturated > self.config.maxSaturatedLines:
                if flux_status != "FAIL":
                    flux_status = "WARN"
                reasons.append(
                    f"nSaturated={n_saturated} > threshold {self.config.maxSaturatedLines}"
                )

        _level = {"PASS": 0, "WARN": 1, "FAIL": 2}
        qa_status = max((fwhm_status, flag_status, dx_status, flux_status), key=lambda s: _level[s])

        reason_str = "; ".join(reasons) if reasons else "all metrics nominal"
        dxStr = f"{medDxCenter:+.3f}px" if np.isfinite(medDxCenter) else "NaN"
        self.log.info(
            "IQ QA %-4s  %s  %-22s  medFWHM=%.2fpx  dxCenter=%s  pctFlagged=%s  [%s]",
            qa_status,
            title,
            seq_nam,
            med_fwhm,
            dxStr,
            f"{pct_flagged:.1f}%" if np.isfinite(pct_flagged) else "NaN",
            reason_str,
        )

        # Parse log metrics if logs are provided
        logMetrics = self._parseLogs(
            self._logToString(isrLog),
            self._logToString(cosmicrayLog),
            self._logToString(reduceExposureLog),
        )

        metricsDict = {
            "medFwhm": [med_fwhm],
            "medDxCenter": [medDxCenter],
            "dxCenterRms": [dxCenterRms],
            "pctFlagged": [pct_flagged],
            "nLines": [len(data)],
            "traceOnly": [trace_only],
            "obsType": [obs_type],
            "seqName": [seq_nam],
            "qaStatus": [qa_status],
            # Log-derived metrics
            "isrBadPixels": [logMetrics["isrBadPixels"]],
            "isrTime": [logMetrics["isrTime"]],
            "cosmicRayCount": [logMetrics["cosmicRayCount"]],
            "cosmicRayPixels": [logMetrics["cosmicRayPixels"]],
            "cosmicRayTime": [logMetrics["cosmicRayTime"]],
            "reduceExposureTime": [logMetrics["reduceExposureTime"]],
            "fitChi2": [logMetrics["fitChi2"]],
            "fitDof": [logMetrics["fitDof"]],
            "fitXRms": [logMetrics["fitXRms"]],
            "fitYRms": [logMetrics["fitYRms"]],
            "fitXSoften": [logMetrics["fitXSoften"]],
            "fitYSoften": [logMetrics["fitYSoften"]],
            "fitNLines": [logMetrics["fitNLines"]],
            "fitTotalLines": [logMetrics["fitTotalLines"]],
            "fitReservedChi2": [logMetrics["fitReservedChi2"]],
            "fitReservedXRms": [logMetrics["fitReservedXRms"]],
            "fitReservedYRms": [logMetrics["fitReservedYRms"]],
            "fitReservedXSoften": [logMetrics["fitReservedXSoften"]],
            "fitReservedYSoften": [logMetrics["fitReservedYSoften"]],
            "fitReservedNLines": [logMetrics["fitReservedNLines"]],
            "fitTraceXRms": [logMetrics["fitTraceXRms"]],
            "fitTraceYRms": [logMetrics["fitTraceYRms"]],
            # Arc flux gatekeeping metrics
            "fluxJitterPct": [flux_jitter_pct],
            "nSaturated": [n_saturated],
            # Fiber arrays
            "fiberIds": [logMetrics["fiberIds"]],
            "fiberXRms": [logMetrics["fiberXRms"]],
            "fiberYRms": [logMetrics["fiberYRms"]],
            "fiberNLines": [logMetrics["fiberNLines"]],
        }
        for bitName, pct in flagBreakdown.items():
            metricsDict[f"pct{bitName}"] = [pct]

        # Add species stats columns dynamically
        for sp, (x_rms, y_rms) in logMetrics["speciesStats"].items():
            metricsDict[f"fitSpeciesXRms_{sp}"] = [x_rms]
            metricsDict[f"fitSpeciesYRms_{sp}"] = [y_rms]

        # Per-species flux jitter and saturation columns (e.g. fluxJitterPct_HgI, nSaturated_CdI)
        for sp, (sp_jitter, sp_nsat) in species_flux_metrics.items():
            metricsDict[f"fluxJitterPct_{sp}"] = [sp_jitter]
            metricsDict[f"nSaturated_{sp}"] = [sp_nsat]

        metrics = pd.DataFrame(metricsDict)
        for key in ("visit", "arm", "spectrograph"):
            if key in dataId:
                metrics[key] = dataId[key]

        return Struct(iqQaData=data, iqQaMetrics=metrics)

    @staticmethod
    def _logToString(logData: Any) -> str:
        """Convert log input to standard string."""
        if logData is None:
            return ""
        if isinstance(logData, str):
            return logData
        if hasattr(logData, "text"):
            return str(logData.text)
        if hasattr(logData, "read"):
            return str(logData.read())
        if hasattr(logData, "readlines"):
            return "\n".join(logData.readlines())
        if isinstance(logData, (list, tuple)):
            return "\n".join(str(r) for r in logData)
        return str(logData)

    def _parseLogs(
        self,
        isrLog: str,
        cosmicrayLog: str,
        reduceExposureLog: str,
    ) -> dict:
        """Parse raw log contents and return a dict of extracted metrics."""
        metrics = {
            "isrBadPixels": 0,
            "isrTime": 0.0,
            "cosmicRayCount": 0,
            "cosmicRayPixels": 0,
            "cosmicRayTime": 0.0,
            "reduceExposureTime": 0.0,
            "fitChi2": float("nan"),
            "fitDof": 0,
            "fitXRms": float("nan"),
            "fitYRms": float("nan"),
            "fitXSoften": float("nan"),
            "fitYSoften": float("nan"),
            "fitNLines": 0,
            "fitTotalLines": 0,
            "fitReservedChi2": float("nan"),
            "fitReservedXRms": float("nan"),
            "fitReservedYRms": float("nan"),
            "fitReservedXSoften": float("nan"),
            "fitReservedYSoften": float("nan"),
            "fitReservedNLines": 0,
            "fitTraceXRms": float("nan"),
            "fitTraceYRms": float("nan"),
            "speciesStats": {},  # species -> (x_rms, y_rms)
            "fiberIds": [],
            "fiberXRms": [],
            "fiberYRms": [],
            "fiberNLines": [],
        }

        re_bad_pixels = re.compile(r"Set (\d+) BAD pixels to")
        re_cr = re.compile(r"(?:Found|Identified) (\d+) cosmic rays (?:\(|covering )(\d+) pixels")
        # Newer drp_stella prefixes its fit summary messages with the detector
        # they belong to ("Final result: arm=b spectrograph=1 chi2=...").  The
        # prefix is optional so that both log formats parse; see
        # ``bin.src/fitDetectorMapLogQa.py``, which does the same.
        armSpec = r"(?:arm=\S+ spectrograph=\d+ )?"
        re_fit_result = re.compile(
            r"Final result: "
            + armSpec
            + r"chi2=(\S+) dof=(\d+) xRMS=(\S+) yRMS=(\S+) xSoften=(\S+) ySoften=(\S+) from (\d+) lines"
        )
        re_fit_lines = re.compile(r"Final fit:.*from (\d+)/(\d+) lines")
        re_reserved_fit = re.compile(
            r"Fit quality from reserved lines:\s*chi2=(\S+)\s+xRMS=(\S+)\s+yRMS=(\S+)(?:\s+\([^\)]+\))?\s+xSoften=(\S+)\s+ySoften=(\S+)\s+from\s+(\d+)\s+lines"
        )
        re_species_stats = re.compile(
            r"Stats for (\w+): " + armSpec + r"chi2=\S+ dof=\d+ xRMS=(\S+) yRMS=(\S+)"
        )
        re_fiber = re.compile(
            r"Stats for fiberId=(\d+): " + armSpec + r"chi2=\S+ dof=\d+ xRMS=(\S+) yRMS=(\S+).*from (\d+) lines"
        )
        re_task_time = re.compile(r"Execution of task '(\w+)' on quantum .* took ([\d\.]+) seconds")

        # 1. Parse ISR log
        if isrLog:
            for line in isrLog.splitlines():
                m = re_bad_pixels.search(line)
                if m:
                    metrics["isrBadPixels"] = int(m.group(1))
                m = re_task_time.search(line)
                if m and m.group(1) == "isr":
                    metrics["isrTime"] = float(m.group(2))

        # 2. Parse Cosmic Ray log
        if cosmicrayLog:
            for line in cosmicrayLog.splitlines():
                m = re_cr.search(line)
                if m:
                    metrics["cosmicRayCount"] += int(m.group(1))
                    metrics["cosmicRayPixels"] += int(m.group(2))
                m = re_task_time.search(line)
                if m and m.group(1) == "cosmicray":
                    metrics["cosmicRayTime"] = float(m.group(2))

        # 3. Parse reduceExposure log
        if reduceExposureLog:
            for line in reduceExposureLog.splitlines():
                m = re_fit_result.search(line)
                if m:
                    try:
                        metrics["fitChi2"] = float(m.group(1))
                        metrics["fitDof"] = int(m.group(2))
                        metrics["fitXRms"] = float(m.group(3))
                        metrics["fitYRms"] = float(m.group(4))
                        metrics["fitXSoften"] = float(m.group(5))
                        metrics["fitYSoften"] = float(m.group(6))
                        metrics["fitNLines"] = int(m.group(7))
                    except ValueError:
                        pass
                m = re_fit_lines.search(line)
                if m:
                    try:
                        metrics["fitNLines"] = int(m.group(1))
                        metrics["fitTotalLines"] = int(m.group(2))
                    except ValueError:
                        pass
                m = re_reserved_fit.search(line)
                if m:
                    try:
                        metrics["fitReservedChi2"] = float(m.group(1))
                        metrics["fitReservedXRms"] = float(m.group(2))
                        metrics["fitReservedYRms"] = float(m.group(3))
                        metrics["fitReservedXSoften"] = float(m.group(4))
                        metrics["fitReservedYSoften"] = float(m.group(5))
                        metrics["fitReservedNLines"] = int(m.group(6))
                    except ValueError:
                        pass
                m = re_species_stats.search(line)
                if m:
                    sp = m.group(1)
                    try:
                        x_rms = float(m.group(2))
                        y_rms = float(m.group(3))
                        if sp == "Trace":
                            metrics["fitTraceXRms"] = x_rms
                            metrics["fitTraceYRms"] = y_rms
                        else:
                            metrics["speciesStats"][sp] = (x_rms, y_rms)
                    except ValueError:
                        pass
                m = re_fiber.search(line)
                if m:
                    try:
                        fid = int(m.group(1))
                        x_rms = float(m.group(2))
                        y_rms = float(m.group(3))
                        n_lines = int(m.group(4))
                        if fid not in metrics["fiberIds"]:
                            metrics["fiberIds"].append(fid)
                            metrics["fiberXRms"].append(x_rms)
                            metrics["fiberYRms"].append(y_rms)
                            metrics["fiberNLines"].append(n_lines)
                    except ValueError:
                        pass
                m = re_task_time.search(line)
                if m and m.group(1) == "reduceExposure":
                    metrics["reduceExposureTime"] = float(m.group(2))

        return metrics

    def _classifyVisit(
        self,
        calexp: lsst.afw.image.Exposure | None,
        pfsConfig: PfsConfig | None,
    ) -> tuple[str, bool, str]:
        """Classify the observation type from FITS header metadata.

        Reads ``W_SEQTYP`` and ``W_SEQNAM`` from either ``calexp`` metadata or
        ``pfsConfig.header``, and determines whether the illumination is IIS
        (engineering fibers) or regular (all 600 science fibers) by inspecting
        lamp headers via `~lsst.obs.pfs.utils.getLamps`.

        Parameters
        ----------
        calexp : `lsst.afw.image.Exposure` or `None`
            Post-ISR calibrated image for this quantum.
        pfsConfig : `PfsConfig` or `None`
            Fiber configuration for this visit.

        Returns
        -------
        obs_type : `str`
            One of ``"arc"``, ``"trace"``, ``"science"``, ``"allsky"``,
            or ``"unknown"`` (when ``W_SEQTYP`` is absent or unrecognised).
        is_iis : `bool`
            True when the illumination comes from the 16 IIS engineering
            fibers (lamp names end with ``"_eng"``).
        seq_nam : `str`
            Raw ``W_SEQNAM`` header value (e.g. ``"Arc: HgCd"``), or an
            empty string when the header is absent.
        """
        # Prefer calexp metadata; fall back to pfsConfig header.
        metadata = None
        if calexp is not None:
            metadata = calexp.getMetadata()
        elif pfsConfig is not None and getattr(pfsConfig, "header", None) is not None:
            metadata = pfsConfig.header

        if metadata is None:
            return "unknown", False, ""

        seq_typ = (metadata.get("W_SEQTYP") or "").strip()
        seq_nam = (metadata.get("W_SEQNAM") or "").strip()

        if seq_typ == "scienceArc":
            obs_type = "arc"
        elif seq_typ == "scienceTrace":
            obs_type = "trace"
        elif seq_typ in ("scienceObject", "scienceObject_windowed", "scienceDark"):
            obs_type = "allsky" if seq_nam.lower().startswith("sky") else "science"
        else:
            if seq_typ:
                self.log.debug("Unrecognised W_SEQTYP=%r; using heuristic fallback.", seq_typ)
            obs_type = "unknown"

        # Distinguish IIS (engineering fiber) illumination from regular (all
        # 600 science fibers).  IIS lamp header names end with "_eng"
        # (e.g. "Ar_eng", "Quartz_eng") while regular lamps do not.
        is_iis = False
        try:
            from lsst.obs.pfs.utils import getLamps

            lamps = getLamps(metadata)
            is_iis = any(name.endswith("_eng") for name in lamps)
        except Exception:
            pass  # obs_pfs unavailable or headers missing; assume regular

        return obs_type, is_iis, seq_nam

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
            ``flux``, ``fluxErr``, ``flag``, ``traceOnly``, ``peakRatio``,
            ``dxCenter`` (always NaN — no live flexure measurement from
            calibration profiles).
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
                    "fiberId",
                    "x",
                    "y",
                    "lam",
                    "fwhm",
                    "theta",
                    "flux",
                    "fluxErr",
                    "flag",
                    "traceOnly",
                    "peakRatio",
                    "dxCenter",
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
                "dxCenter": np.full(n_total, np.nan),
            }
        )

    def _buildImageWidthData(
        self,
        calexp: lsst.afw.image.Exposure,
        detectorMap: DetectorMap,
        detectorMapCalib: DetectorMap | None = None,
        fiberIds: set | None = None,
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
            Adjusted detector mapping used to locate fiber centers.
        detectorMapCalib : `DetectorMap` or `None`, optional
            Static calibration detectorMap.  When provided, the spatial
            offset ``dxCenter`` (calibration prediction minus measured
            center) is recorded for each sample as a flexure diagnostic.
        fiberIds : `set` or `None`, optional
            If provided, only fibers whose IDs are in this set are sampled.
            Use this to restrict measurement to known bright fibers (e.g.
            FLUXSTD fibers from ``pfsConfig``) and avoid dilution from dark
            sky fibers.  When ``None``, all fibers in the detectorMap are
            sampled.

        Returns
        -------
        `pandas.DataFrame`
            Columns: ``fiberId``, ``x``, ``y``, ``lam``, ``fwhm``, ``theta``,
            ``flux``, ``fluxErr``, ``flag``, ``traceOnly``, ``peakRatio``,
            ``dxCenter``.  ``traceOnly`` is `False`: these are live
            measurements of the exposure, not calibration trace widths.
        """
        image = calexp.image.array.astype(np.float64)
        mask_arr = calexp.mask.array
        nRows, nCols = image.shape

        halfWidth = self.config.profileHalfWidth
        yStride = self.config.profileYStride
        badBits = calexp.mask.getPlaneBitMask(["BAD", "SAT", "CR", "NO_DATA"])

        all_det_fiberIds = np.asarray(detectorMap.fiberId, dtype=np.int32)
        if fiberIds is not None:
            mask = np.isin(all_det_fiberIds, list(fiberIds))
            det_fiberIds = all_det_fiberIds[mask]
        else:
            det_fiberIds = all_det_fiberIds
        y_samples = np.arange(yStride // 2, nRows, yStride, dtype=np.float64)

        all_fiberIds: list = []
        all_y: list = []
        all_x: list = []
        all_lam: list = []
        all_fwhm: list = []
        all_flux: list = []
        all_flag: list = []
        all_peakRatio: list = []
        all_dxCenter: list = []

        for y_val in y_samples:
            y_int = round(y_val)
            if y_int < 0 or y_int >= nRows:
                continue

            y_arr = np.full(len(det_fiberIds), y_val, dtype=np.float64)
            x_centers = detectorMap.getXCenter(det_fiberIds, y_arr)
            lams = detectorMap.findWavelength(det_fiberIds, y_arr)
            calibXCenters = (
                detectorMapCalib.getXCenter(det_fiberIds, y_arr) if detectorMapCalib is not None else None
            )
            row_image = image[y_int, :]
            row_mask = mask_arr[y_int, :]

            for ii, (fiberId, x_cen, lam) in enumerate(zip(det_fiberIds, x_centers, lams, strict=False)):
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
                dx_center = np.nan

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
                            if calibXCenters is not None:
                                dx_center = float(calibXCenters[ii]) - (x_cen + mu)
                        flux_val = total

                all_fiberIds.append(int(fiberId))
                all_y.append(y_val)
                all_x.append(float(x_cen))
                all_lam.append(float(lam))
                all_fwhm.append(fwhm_val)
                all_flux.append(flux_val)
                all_flag.append(flag_val)
                all_peakRatio.append(peak_ratio)
                all_dxCenter.append(dx_center)

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
                # These are measurements of the exposure itself, not widths
                # read back from a fiber profile calibration, so the FWHM
                # thresholds apply to them.
                "traceOnly": False,
                "peakRatio": np.array(all_peakRatio),
                "dxCenter": np.array(all_dxCenter),
            }
        )

    @staticmethod
    def _computeFlagBreakdown(
        data: pd.DataFrame,
        nTotal: int,
        isArcLinePath: bool = False,
    ) -> dict[str, float]:
        """Compute per-status-bit and measurement-level flag percentages.

        Parameters
        ----------
        data : `pandas.DataFrame`
            Must contain a ``status`` column (int32 bitmask).  When
            ``isArcLinePath`` is True, ``flag`` and ``flux`` columns are
            used to derive measurement-level breakdown.
        nTotal : `int`
            Total number of lines (denominator for percentages).
        isArcLinePath : `bool`, optional
            When True, compute measurement-level flag breakdown
            (``LowSN`` and ``MeasFail``) from the ``flux`` column.
            Lines with ``flag=True`` and NaN flux were rejected for
            low S/N (measurement skipped); lines with ``flag=True``
            and finite flux had a centroid or photometry failure.

        Returns
        -------
        `dict` [`str`, `float`]
            Mapping of bit/category name to percentage of lines.
        """
        result: dict[str, float] = {}

        if "status" in data.columns:
            status = data["status"].to_numpy(dtype=np.int32)
            bits = [
                ("NotVisible", 0x01),
                ("Blend", 0x02),
                ("Suspect", 0x04),
                ("Rejected", 0x08),
                ("Broad", 0x10),
            ]
            for name, mask in bits:
                result[name] = 100.0 * int(np.sum((status & mask) != 0)) / nTotal

        if isArcLinePath and "flag" in data.columns and "flux" in data.columns:
            flagged = data["flag"].to_numpy(dtype=bool)
            fluxFinite = np.isfinite(data["flux"].to_numpy(dtype=np.float32))
            result["LowSN"] = 100.0 * int(np.sum(flagged & ~fluxFinite)) / nTotal
            result["MeasFail"] = 100.0 * int(np.sum(flagged & fluxFinite)) / nTotal

        return result
