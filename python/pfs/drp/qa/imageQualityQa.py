"""Image quality QA pipeline task.

Produces per-detector FWHM and image-shape QA plots from arc-line
second-moment measurements, fiber profile calibrations, or direct
cross-dispersion moment measurements from post-ISR pixel data.
"""

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
    Output as OutputConnection,
    PrerequisiteInput as PrerequisiteConnection,
)
from matplotlib import pyplot as plt
from pfs.datamodel import FiberStatus, PfsConfig, TargetType
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

    pfsConfig = InputConnection(
        name="pfsConfig",
        doc=(
            "Fiber configuration for this visit.  When provided alongside"
            " ``calexp``, only fibers with ``targetType=FLUXSTD`` and"
            " ``fiberStatus=GOOD`` are used for calexp-based FWHM measurement,"
            " avoiding contamination from dark sky fibers."
        ),
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
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
    flagRateWarnThreshold = DictField(
        keytype=str,
        itemtype=float,
        default={"b": 40.0, "r": 15.0, "n": 15.0, "m": 15.0},
        doc=(
            "Per-arm percentage of flagged lines above which the per-quantum"
            " status is set to WARN.  Values above ``flagRateFailThreshold``"
            " yield FAIL.  The b arm has a naturally higher flag rate (~37%)"
            " due to arc-line crowding, so its threshold is set higher."
            " Arms not listed fall back to 15.0.  Set to 100 to disable."
        ),
    )
    flagRateFailThreshold = DictField(
        keytype=str,
        itemtype=float,
        default={"b": 45.0, "r": 20.0, "n": 20.0, "m": 20.0},
        doc=(
            "Per-arm percentage of flagged lines above which the per-quantum"
            " status is set to FAIL.  The b arm has a naturally higher flag"
            " rate (~37%) due to arc-line crowding, so its threshold is set"
            " higher than r/n/m.  Arms not listed fall back to 20.0."
            " Set to 100 to disable."
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
        pfsConfig: PfsConfig | None,
        dataId: dict,
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
            Calibrated detector mapping from fiberId,wavelength to x,y.
        fiberProfiles : `FiberProfileSet` or `None`
            Fiber profile shapes.  Used as fallback for regular trace/quartz
            visits when ``calexp`` is absent.
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
            Per-line image quality data.  Written as an empty DataFrame when
            ``writeIqQaData`` is False.
        iqQaMetrics : `pandas.DataFrame`
            Single-row summary with ``medFwhm``, ``pctFlagged``, ``nLines``,
            ``traceOnly``, ``qaStatus``, ``visit``, ``arm``, ``spectrograph``.
        iqQaPlot : `MultipagePdfFigure`
            QA plots, one page per enabled plot type.
        """
        self.log.info("Computing image quality metrics for %s", dataId)

        # Classify the observation type from FITS header metadata so that the
        # task can route each visit to the correct measurement path explicitly
        # rather than relying on heuristics like the arc-line count.
        obs_type, is_iis = self._classifyVisit(calexp, pfsConfig)
        self.log.debug(
            "Visit classification: obs_type=%r is_iis=%s for %s", obs_type, is_iis, dataId
        )

        als = addTraceLambdaToArclines(arcLines, detectorMap)
        data = computeImageQuality(als)
        data["peakRatio"] = np.nan

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
                    dataId, n_good_arc,
                )
            elif calexp is not None:
                self.log.info(
                    "Regular arc %s: only %d good arc lines (< minGoodLines=%d);"
                    " trying calexp fallback.",
                    dataId, n_good_arc, self.config.minGoodLines,
                )
                calexp_data = self._buildImageWidthData(calexp, detectorMap)
                n_good_calexp = int((calexp_data["fwhm"].notna() & ~calexp_data["flag"]).sum())
                calexp_good_frac = n_good_calexp / max(len(calexp_data), 1)
                if n_good_calexp > 0 and (1.0 - calexp_good_frac) < self.config.maxCalexpFlagRate:
                    self.log.info(
                        "Regular arc calexp fallback: %d good samples (%.1f%% good) for %s.",
                        n_good_calexp, 100.0 * calexp_good_frac, dataId,
                    )
                    data = calexp_data
                    dense_data = True
                else:
                    self.log.warning(
                        "Regular arc calexp fallback too sparse for %s"
                        " (%d good, %.1f%% flagged); keeping %d arc-line measurements.",
                        dataId, n_good_calexp,
                        100.0 * (1.0 - calexp_good_frac),
                        n_good_arc,
                    )
            else:
                self.log.warning(
                    "Regular arc %s: only %d good arc lines (< minGoodLines=%d)"
                    " and no calexp; FWHM will be sparse.",
                    dataId, n_good_arc, self.config.minGoodLines,
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
            if calexp is not None:
                self.log.info(
                    "Regular trace/quartz %s: measuring FWHM from calexp.", dataId,
                )
                calexp_data = self._buildImageWidthData(calexp, detectorMap)
                n_good_calexp = int((calexp_data["fwhm"].notna() & ~calexp_data["flag"]).sum())
                calexp_good_frac = n_good_calexp / max(len(calexp_data), 1)
                if n_good_calexp > 0 and (1.0 - calexp_good_frac) < self.config.maxCalexpFlagRate:
                    self.log.info(
                        "Quartz calexp: %d good samples (%.1f%% good) for %s.",
                        n_good_calexp, 100.0 * calexp_good_frac, dataId,
                    )
                    data = calexp_data
                    dense_data = True
                else:
                    self.log.warning(
                        "Quartz calexp too sparse for %s"
                        " (%d good, %.1f%% flagged); FWHM will be sparse.",
                        dataId, n_good_calexp,
                        100.0 * (1.0 - calexp_good_frac),
                    )
            elif fiberProfiles is not None:
                self.log.info(
                    "Regular trace/quartz %s: no calexp; falling back to"
                    " fiber profile calibration widths.",
                    dataId,
                )
                data = self._buildProfileData(fiberProfiles, detectorMap)
                dense_data = True
            else:
                self.log.warning(
                    "Regular trace/quartz %s: no calexp and no fiberProfiles;"
                    " FWHM will be sparse.",
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
            if calexp is not None and pfsConfig is not None:
                good_mask = (
                    (pfsConfig.targetType == TargetType.FLUXSTD)
                    & (pfsConfig.fiberStatus == FiberStatus.GOOD)
                )
                fluxstd_ids: set = set(int(f) for f in pfsConfig.fiberId[good_mask])
                self.log.info(
                    "%s %s: measuring FWHM from %d FLUXSTD fibers in calexp.",
                    obs_type.capitalize(), dataId, len(fluxstd_ids),
                )
                calexp_data = self._buildImageWidthData(calexp, detectorMap, fiberIds=fluxstd_ids)
                n_good_calexp = int((calexp_data["fwhm"].notna() & ~calexp_data["flag"]).sum())
                n_total_calexp = len(calexp_data)
                calexp_good_frac = n_good_calexp / max(n_total_calexp, 1)
                if n_good_calexp > 0 and calexp_good_frac >= self.config.minFluxstdGoodFrac:
                    self.log.info(
                        "FLUXSTD calexp: %d good samples (%.1f%% good) for %s.",
                        n_good_calexp, 100.0 * calexp_good_frac, dataId,
                    )
                    data = calexp_data
                    dense_data = True
                    using_fluxstd_filter = True
                else:
                    self.log.info(
                        "FLUXSTD calexp too sparse for %s"
                        " (%d/%d = %.1f%% < minFluxstdGoodFrac=%.0f%%); FWHM will be sparse.",
                        dataId, n_good_calexp, n_total_calexp,
                        100.0 * calexp_good_frac,
                        100.0 * self.config.minFluxstdGoodFrac,
                    )
            else:
                self.log.info(
                    "%s %s: no calexp or pfsConfig available; FWHM will be sparse.",
                    obs_type.capitalize(), dataId,
                )

        else:
            # obs_type == "unknown": W_SEQTYP header absent or unrecognised.
            # Fall back to heuristic: try arc-line path; if too few good lines,
            # try calexp (FLUXSTD-filtered if pfsConfig is available), then
            # fiberProfiles.
            self.log.debug(
                "Visit type unknown for %s; using heuristic fallback (n_good_arc=%d).",
                dataId, n_good_arc,
            )
            if n_good_arc < self.config.minGoodLines:
                if calexp is not None:
                    fluxstd_ids_unk: set | None = None
                    if pfsConfig is not None:
                        good_mask = (
                            (pfsConfig.targetType == TargetType.FLUXSTD)
                            & (pfsConfig.fiberStatus == FiberStatus.GOOD)
                        )
                        fluxstd_ids_unk = set(int(f) for f in pfsConfig.fiberId[good_mask])
                        self.log.info(
                            "Unknown type %s: %d good arc lines (< %d); trying calexp"
                            " with %d FLUXSTD fibers.",
                            dataId, n_good_arc, self.config.minGoodLines, len(fluxstd_ids_unk),
                        )
                    else:
                        self.log.info(
                            "Unknown type %s: %d good arc lines (< %d); trying calexp.",
                            dataId, n_good_arc, self.config.minGoodLines,
                        )
                    calexp_data = self._buildImageWidthData(
                        calexp, detectorMap, fiberIds=fluxstd_ids_unk
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
                            n_good_calexp, 100.0 * calexp_good_frac, dataId,
                        )
                        data = calexp_data
                        dense_data = True
                        using_fluxstd_filter = fluxstd_ids_unk is not None
                    elif fluxstd_ids_unk is not None:
                        self.log.info(
                            "Unknown type FLUXSTD calexp too sparse for %s"
                            " (%d/%d = %.1f%% < %.0f%%); FWHM will be sparse.",
                            dataId, n_good_calexp, n_total_calexp,
                            100.0 * calexp_good_frac,
                            100.0 * self.config.minFluxstdGoodFrac,
                        )
                    else:
                        self.log.info(
                            "Unknown type calexp too sparse for %s"
                            " (%d good, %.1f%% flagged); keeping arc-line data.",
                            dataId, n_good_calexp,
                            100.0 * (1.0 - calexp_good_frac),
                        )
                elif fiberProfiles is not None:
                    self.log.info(
                        "Unknown type %s: %d good arc lines (< %d); using fiberProfiles.",
                        dataId, n_good_arc, self.config.minGoodLines,
                    )
                    data = self._buildProfileData(fiberProfiles, detectorMap)
                    dense_data = True
                else:
                    self.log.warning(
                        "Unknown type %s: %d good arc lines (< %d) and no calexp or"
                        " fiberProfiles; FWHM will be sparse.",
                        dataId, n_good_arc, self.config.minGoodLines,
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
        else:
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
        if np.isfinite(pct_flagged):
            arm = dataId.get("arm", "")
            warn_thresh = self.config.flagRateWarnThreshold.get(arm, 15.0)
            fail_thresh = self.config.flagRateFailThreshold.get(arm, 20.0)
            if pct_flagged >= fail_thresh:
                flag_status = "FAIL"
                reasons.append(
                    f"pctFlagged={pct_flagged:.1f}% >= fail threshold {fail_thresh}%"
                )
            elif pct_flagged >= warn_thresh:
                flag_status = "WARN"
                reasons.append(
                    f"pctFlagged={pct_flagged:.1f}% >= warn threshold {warn_thresh}%"
                )

        _level = {"PASS": 0, "WARN": 1, "FAIL": 2}
        qa_status = max((fwhm_status, flag_status), key=lambda s: _level[s])

        reason_str = "; ".join(reasons) if reasons else "all metrics nominal"
        self.log.info(
            "IQ QA %-4s  %s  medFWHM=%.2fpx  pctFlagged=%s  [%s]",
            qa_status, title, med_fwhm,
            f"{pct_flagged:.1f}%" if np.isfinite(pct_flagged) else "NaN",
            reason_str,
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

    def _classifyVisit(
        self,
        calexp: lsst.afw.image.Exposure | None,
        pfsConfig: PfsConfig | None,
    ) -> tuple[str, bool]:
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
        """
        # Prefer calexp metadata; fall back to pfsConfig header.
        metadata = None
        if calexp is not None:
            metadata = calexp.getMetadata()
        elif pfsConfig is not None and getattr(pfsConfig, "header", None) is not None:
            metadata = pfsConfig.header

        if metadata is None:
            return "unknown", False

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
            from lsst.obs.pfs.utils import getLamps  # noqa: PLC0415

            lamps = getLamps(metadata)
            is_iis = any(name.endswith("_eng") for name in lamps)
        except Exception:
            pass  # obs_pfs unavailable or headers missing; assume regular

        return obs_type, is_iis

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
            Calibrated detector mapping used to locate fiber centers.
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

        for y_val in y_samples:
            y_int = int(round(y_val))
            if y_int < 0 or y_int >= nRows:
                continue

            y_arr = np.full(len(det_fiberIds), y_val, dtype=np.float64)
            x_centers = detectorMap.getXCenter(det_fiberIds, y_arr)
            lams = detectorMap.findWavelength(det_fiberIds, y_arr)
            row_image = image[y_int, :]
            row_mask = mask_arr[y_int, :]

            for fiberId, x_cen, lam in zip(det_fiberIds, x_centers, lams):
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
