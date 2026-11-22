"""Image quality QA pipeline task.

Measures per-detector FWHM and image shape from arc-line
second-moment measurements, cross-dispersion trace widths measured on the
calexp (`pfs.drp.qa.crossDispersion`), or fiber profile calibrations.
It writes numbers only: `pfs.drp.qa.metrics.gate` gives the verdicts.
"""

from typing import Any

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
)
from lsst.pipe.base.connectionTypes import (
    Output as OutputConnection,
)
from lsst.pipe.base.connectionTypes import (
    PrerequisiteInput as PrerequisiteConnection,
)

from pfs.datamodel import FiberStatus, PfsConfig, TargetType
from pfs.drp.qa.crossDispersion import measureImageWidths, profileFwhm
from pfs.drp.qa.metrics.lineFlux import speciesLineFlux
from pfs.drp.qa.metrics.longFormat import metricsFrame, speciesFrame
from pfs.drp.qa.metrics.reductionStats import speciesFitStats
from pfs.drp.qa.metrics.registry import observationType, sequenceOf
from pfs.drp.qa.metrics.saturation import measureSaturation
from pfs.drp.stella import ArcLineSet, DetectorMap, FiberProfileSet
from pfs.drp.stella.utils.quality import computeImageQuality
from pfs.drp.stella.utils.stability import addTraceLambdaToArclines

__all__ = ["ImageQualityQaTask", "engineeringLit", "litFibers"]


def engineeringLit(pfsConfig: PfsConfig, spectrograph: int | None = None) -> bool:
    """Return whether the pfsConfig says the IIS is on: its engineering fibers are all ``GOOD``.

    What drp_stella's ``reduceExposure`` decides by (``checkPfsConfig`` selects the quantum's
    spectrograph from the pfsConfig, then ``isIIS`` asks whether its engineering fibers are
    all ``GOOD``), and so what decides which fibers ``lines`` holds: a design that doesn't
    light the IIS records the engineering fibers as not ``GOOD``. The lamp headers are the
    other witness (`getLamps`), and some n-arm raws record the IIS lamp off (Run30: 148829
    n4, 149134 n2, 149322 n2, ...; PIPE2D-1954).

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.PfsConfig`
        Fiber configuration of the visit.
    spectrograph : `int` or `None`, optional
        Judge this spectrograph's engineering fibers alone, as reduceExposure does;
        every engineering fiber when `None` or the pfsConfig has no ``spectrograph``.

    Returns
    -------
    `bool`
        Whether the engineering fibers (there must be some) are all ``GOOD``.
    """
    engineering = np.asarray(pfsConfig.targetType) == TargetType.ENGINEERING
    spectrographs = getattr(pfsConfig, "spectrograph", None)
    if spectrograph is not None and spectrographs is not None:
        engineering &= np.asarray(spectrographs) == spectrograph
    if not engineering.any():
        return False
    return bool(np.all(np.asarray(pfsConfig.fiberStatus)[engineering] == FiberStatus.GOOD))


def litFibers(
    pfsConfig: PfsConfig | None, fiberId: np.ndarray, engineering: bool = False
) -> tuple[set[int] | None, int | None]:
    """Return the fibers of a detector that the pfsConfig lights, and how many it could.

    A design lights one fiber group by parking the others' cobras behind their black
    dots, recording them as ``fiberStatus`` ``BLACKSPOT`` (with ``targetType``
    ``UNASSIGNED``, not ``BLACKSPOT``). drp_stella measures lines and traces only on
    ``GOOD`` fibers, so ``lines`` has rows for those alone. A ``GOOD`` fiber without a
    nominal position (an unassigned cobra) is lit too, and counts: leaving it out would
    let an obstructed frame on such a design pass.

    An IIS frame lights the 16 engineering fibers of each spectrograph instead, and
    drp_stella's reduceExposure then measures lines and traces on those alone, whatever
    the science fibers' status: with ``engineering`` the lit fibers are
    the ``GOOD`` engineering fibers.

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.PfsConfig` or `None`
        Fiber configuration of the visit.
    fiberId : `numpy.ndarray` of `int`
        The detector's fibers (``detectorMap.fiberId``).
    engineering : `bool`, optional
        Count the engineering fibers (an IIS frame) rather than the science fibers.

    Returns
    -------
    lit : `set` [`int`] or `None`
        The detector's ``GOOD`` science fibers, or with ``engineering`` its ``GOOD``
        engineering fibers; `None` without a pfsConfig.
    possible : `int` or `None`
        How many of the detector's fibers the design could light: the lit ones
        and those it hides behind a black dot, or with ``engineering`` the
        detector's engineering fibers. `None` without a pfsConfig.
    """
    if pfsConfig is None:
        return None, None
    status = np.asarray(pfsConfig.fiberStatus)
    onDetector = np.isin(np.asarray(pfsConfig.fiberId), np.asarray(fiberId))
    isEngineering = np.asarray(pfsConfig.targetType) == TargetType.ENGINEERING
    onDetector &= isEngineering if engineering else ~isEngineering
    lit = onDetector & (status == FiberStatus.GOOD)
    possible = onDetector if engineering else lit | (onDetector & (status == FiberStatus.BLACKSPOT))
    return {int(f) for f in np.asarray(pfsConfig.fiberId)[lit]}, int(possible.sum())


class _WithoutDataType:
    """A header's ``get``, less ``DATA-TYP``, so `getLamps` reads the lamp headers of a flat."""

    def __init__(self, metadata):
        self.metadata = metadata

    def get(self, key, default=None):
        return default if key == "DATA-TYP" else self.metadata.get(key, default)


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

    # Written only with doWriteLineData. The connection stays even when off: every
    # stored imageQualityQa_config sets config.connections.iqQaData, and a config with no such
    # field can't be read back.
    iqQaData = OutputConnection(
        name="iqQaData",
        doc=(
            "Per-line image quality measurements, including FWHM,"
            " position angle, and spatial offset (dxCenter); only with ``doWriteLineData``."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    detectorMap = InputConnection(
        name="detectorMap",
        doc="Calibrated detector mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    iqQaMetrics = OutputConnection(
        name="iqQaMetrics",
        doc=(
            "Per-quantum summary metrics (one row): numbers and what identifies the frame, no verdict"
            " (`pfs.drp.qa.metrics.gate` judges them afterwards)."
            " Columns: ``visit``, ``arm``, ``spectrograph``, ``medFwhm``,"
            " ``medDxCenter``, ``dxCenterRms``, ``pctFlagged``,"
            " ``pctLowSN``, ``pctMeasFail``,"
            " ``nLines`` (rows of ``lines``), ``linesPerFiber`` (unflagged lamp"
            " lines, or on a quartz trace centroids, per lit fiber), ``nLitFibers``"
            " and ``nFibers`` (the detector's fibers the pfsConfig lights, and"
            " those it lights or hides behind a black dot), ``nTraceRows`` (trace"
            " centroids in ``lines``), ``traceOnly``, ``obsType`` (which measurement was made),"
            " ``illumination`` (``full``,"
            " the science fibers, or ``iis``, the 16 engineering fibers), ``seqName`` and ``seqType``"
            " (W_SEQNAM and W_SEQTYP as read: what the frame is judged as comes from them,"
            " `pfs.drp.qa.metrics.registry.frameType`), ``lamps`` (the lamps the header says were lit,"
            " obs_pfs's names joined by ``+``; empty when none),"
            " ``calexpRun`` (the RUN"
            " collection of the calexp measured, null without one)."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    iqQaSpeciesMetrics = OutputConnection(
        name="iqQaSpeciesMetrics",
        doc=(
            "Per-species metrics in long format: ``visit``, ``arm``, ``spectrograph``,"
            " ``description`` (line species, ``Trace`` included), ``metric``, ``value``."
            " From ``reduceExposure_metadata``, the detectorMap fit's ``fitXRms``, ``fitYRms``"
            " (weighted RMS) and ``fitXSoften``, ``fitYSoften`` (the softening that brings"
            " chi^2/dof to 1). From ``lines``, on lit fibers: ``medLineFlux``,"
            " ``robustRmsLineFlux`` and ``nFluxLines`` (the good lines they are over), and with"
            " a calexp ``nSaturatedLines`` (`pfs.drp.qa.metrics.lineFlux`)."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc=(
            "Fiber profile shapes; their widths stand in for the trace FWHM when"
            " neither arc lines nor the calexp give a usable measurement. A"
            " calibration, not a measurement: ``traceOnly`` marks such a FWHM."
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
            " non-arc visits, trace widths are measured directly from its"
            " pixels, each fiber fitted jointly with its neighbours"
            " (`pfs.drp.qa.crossDispersion`), rather than read from the"
            " calibration."
        ),
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        minimum=0,
    )

    # A prerequisite, as in extractionQa and extractionQaCombined: a dataset
    # type that is a prerequisite to one task in a graph must be one to all of
    # them, or the pipeline doesn't build. Prerequisites are resolved when the
    # graph is built and default to minimum=1, so minimum=0 is what lets a
    # collection without a pfsConfig still get a quantum.
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc=(
            "Fiber configuration for this visit.  When provided alongside"
            " ``calexp``, only fibers with ``targetType=FLUXSTD`` and"
            " ``fiberStatus=GOOD`` are used for calexp-based FWHM measurement,"
            " avoiding contamination from dark sky fibers. On arcs and quartz it"
            " says which fibers are lit (`litFibers`): ``linesPerFiber``,"
            " ``pctFlagged`` and the calexp trace width are taken on those."
        ),
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
        minimum=0,
    )

    reduceExposureMetadata = InputConnection(
        name="reduceExposure_metadata",
        doc=(
            "reduceExposure task metadata: the detectorMap fit's per-species statistics"
            " (``speciesName`` and the parallel ``speciesXRms`` … arrays recorded by"
            " ``fitDetectorMap``), read into ``iqQaSpeciesMetrics`` when the adjustment applied"
            " (`pfs.drp.qa.metrics.reductionStats.speciesFitStats`). Absent when only ISR and"
            " fitDetectorMap ran. The whole-detector fit scalars and the ISR/CR counts are read"
            " from the same metadata by ``reductionStats``, not here."
        ),
        storageClass="TaskMetadata",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        minimum=0,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config:
            return
        if not config.doWriteLineData:
            self.outputs.remove("iqQaData")


class ImageQualityQaConfig(PipelineTaskConfig, pipelineConnections=ImageQualityQaConnections):
    """Configuration for ImageQualityQaTask."""

    doWriteLineData = Field(
        dtype=bool,
        default=False,
        doc=(
            "Write the per-line measurements (``iqQaData``)? About 21 MB per detector for an arc and"
            " 80 MB for a twilight, mostly unsampled trace centroids, and nothing reads them back;"
            " the gate and the normal values read ``iqQaMetrics``."
        ),
    )

    profileHalfWidth = Field(
        dtype=int,
        default=7,
        doc="Unused: the calexp trace width is fitted with its neighbours, with no aperture.",
        deprecated=(
            "The calexp trace width no longer uses an aperture; this field is ignored and will be removed."
        ),
    )
    profileYStride = Field(
        dtype=int,
        default=100,
        doc=(
            "Row sampling interval (pixels) for the calexp trace-width measurement."
            "  The fit's cost is proportional to the rows sampled; 100 gives about 40"
            " rows, some 25,000 samples on a quartz."
        ),
    )
    minGoodLines = Field(
        dtype=int,
        default=10,
        doc=(
            "Minimum number of good (unflagged, finite-FWHM) arc-line measurements"
            " required to trust the arc-line path.  When fewer good lines are found"
            " the task falls back to the calexp-based width measurement (or fiber"
            " profile calibration if no calexp is available).  An IIS arc's lines"
            " are on its 16 engineering fibers per spectrograph, and count."
        ),
    )
    minPeakSN = Field(
        dtype=float,
        default=5.0,
        doc=(
            "Minimum significance of a trace in the calexp-based width"
            " measurement: its fitted flux over that flux's standard error,"
            " from the joint fit's covariance and the calexp variance plane."
            " Samples below it (rows where a fiber is not illuminated,"
            " scattered light in an IIS frame) are flagged and excluded from"
            " the FWHM median."
        ),
    )
    maxCalexpFlagRate = Field(
        dtype=float,
        default=0.5,
        doc=(
            "Maximum fraction of calexp profile samples that may be flagged"
            " before the calexp path is considered unreliable and discarded."
            " Continuum/quartz frames illuminate all fibers uniformly so the"
            " post-S/N-filter flag rate is low (< 10 %), and only the lit fibers"
            " are sampled (`litFibers`: the 16 engineering fibers on an IIS quartz)."
            "  Arc-illuminated frames light the fibers only at"
            " discrete wavelengths; scattered light from those"
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
            " sparse (NaN).  On bright-star"
            " or arc frames where FLUXSTD fibers are well-lit, the good fraction"
            " is typically > 10 % and the FWHM estimate is used.  The"
            " pctFlagged metric is always suppressed (NaN) when the FLUXSTD"
            " path is active, because the flag rate for stellar fibers reflects"
            " exposure depth rather than optical quality."
        ),
    )


class ImageQualityQaTask(PipelineTask):
    """QA task measuring image quality from arc-line second moments.

    Reads per-detector arc line measurements (``lines`` dataset), enriches
    them with wavelength and trace-position information from the
    ``detectorMap``, and computes Gaussian-equivalent FWHM and flexure
    metrics.

    The task measures and stores numbers; it gives no verdict and holds no
    limit. `pfs.drp.qa.metrics.gate` judges ``iqQaMetrics`` afterwards.
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
            # Perform the actual processing.
            outputs = self.run(**inputs)
        except ValueError as e:
            # An expected processing failure is logged and the outputs are
            # omitted, so one bad quantum does not abort the QA pipeline.
            # Unexpected exceptions are left to propagate.
            self.log.error("ImageQualityQaTask failed for %s: %s", dataId, e)
        else:
            # The RUN of the calexp measured: its pfsArm, written beside it, records the pipeline
            # that reduced it.
            calexpRef = getattr(inputRefs, "calexp", None)
            outputs.iqQaMetrics["calexpRun"] = calexpRef.run if calexpRef is not None else None
            if self.config.doWriteLineData:
                butlerQC.put(outputs.lineData, outputRefs.iqQaData)
            butlerQC.put(outputs.iqQaMetrics, outputRefs.iqQaMetrics)
            butlerQC.put(outputs.iqQaSpeciesMetrics, outputRefs.iqQaSpeciesMetrics)

    def run(
        self,
        arcLines: ArcLineSet,
        detectorMap: DetectorMap,
        fiberProfiles: FiberProfileSet | None,
        detectorMapCalib: DetectorMap | None,
        calexp: lsst.afw.image.Exposure | None,
        pfsConfig: PfsConfig | None,
        dataId: dict,
        reduceExposureMetadata: Any = None,
    ) -> Struct:
        """Compute image quality metrics.

        The measurement path is chosen based on the observation type derived
        from ``W_SEQTYP`` and lamp headers (see `_classifyVisit`):

        * **Regular arc** (``scienceArc``, all fibers): arc-line second moments
          are the primary path.  If fewer than ``config.minGoodLines`` good
          measurements are found (e.g. blue arm where catalog lines are sparse),
          a calexp cross-dispersion fallback is attempted.
        * **IIS arc** (``scienceArc``, the IIS lamps): as a regular arc, on the
          16 engineering fibers per spectrograph that reduceExposure measures
          lines on; the pfsConfig says whether it did
          (`engineeringLit`), the lamp headers without one. ``illumination`` is
          ``iis``, which makes it a frame type of its own
          (`pfs.drp.qa.metrics.registry.frameType`).
        * **Regular trace/quartz** (``scienceTrace``, all fibers): calexp
          trace widths are the primary path. When they are absent
          or unusable, the fiber profile calibration widths are reported with
          ``traceOnly``, and ``pctFlagged`` is null.
        * **IIS trace/quartz** (``scienceTrace``, the IIS halogen lamp): as a
          regular quartz, the calexp widths and the trace centroids taken on
          the engineering fibers.
        * **Science / all-sky** (``scienceObject*``, no lamps): when
          ``pfsConfig`` is available, FWHM is measured from FLUXSTD+GOOD
          fibers in ``calexp`` and ``pctFlagged`` is suppressed (stellar fibers
          are not continuously illuminated so the flag rate reflects exposure
          depth, not optical quality).  If the FLUXSTD good fraction is below
          ``config.minFluxstdGoodFrac``, FWHM is treated as sparse.
        * **Twilight** (``scienceObject``, ``W_SEQNAM`` "Twilight ..."): not
          measured; its FWHM would come from the sky lines.
        * **Dark** and **bias** (``darks``, ``masterDarks``, ``biases``,
          ``masterBiases``): nothing to measure.
        * **Unknown** (``W_SEQTYP`` absent, or a sequence type with no
          measurement of its own: fiber profiles, dot roaches, engineering
          exposures, ...): falls back to the original
          heuristic based on ``n_good_arc`` and ``maxCalexpFlagRate``.

        A quantum with nothing to measure still writes its row: what could
        not be measured is null, and the identifying columns say what the
        frame is.

        Parameters
        ----------
        arcLines : `ArcLineSet`
            Arc line measurements.
        detectorMap : `DetectorMap`
            Adjusted detector mapping from fiberId,wavelength to x,y.
        fiberProfiles : `FiberProfileSet` or `None`
            Fiber profile shapes.  Used as fallback for regular trace/quartz
            visits when ``calexp`` is absent or its measurement unusable.
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
        lineData : `pandas.DataFrame`
            Per-line (or per-trace-sample) image quality data including
            ``dxCenter`` (spatial offset from static calibration detectorMap,
            in pixels).  Stored as ``iqQaData`` only with ``doWriteLineData``.
        iqQaMetrics : `pandas.DataFrame`
            Single-row summary with ``medFwhm``, ``medDxCenter``,
            ``dxCenterRms``, ``pctFlagged``, ``nLines``, ``linesPerFiber``,
            ``nLitFibers``, ``nFibers``, ``nTraceRows``, ``nSaturatedLines``,
            ``pctSaturatedPixels``, ``traceOnly``,
            ``obsType``, ``illumination``, ``seqName``, ``seqType``, ``lamps``,
            the flag breakdown (``pct<Bit>``), ``visit``, ``arm``,
            ``spectrograph``.  Numbers only: no verdict column.
        iqQaSpeciesMetrics : `pandas.DataFrame`
            Per-species fit statistics and line fluxes in long format
            (`pfs.drp.qa.metrics.longFormat`).
        """
        self.log.info("Computing image quality metrics for %s", dataId)

        # Classify the observation type from FITS header metadata so that the
        # task can route each visit to the correct measurement path explicitly
        # rather than relying on heuristics like the arc-line count.
        obs_type, is_iis, seq_nam, seq_typ, lamps = self._classifyVisit(
            calexp, pfsConfig, dataId.get("spectrograph")
        )
        self.log.debug(
            "Visit classification: obs_type=%r is_iis=%s seq_typ=%r seq_nam=%r lamps=%r for %s",
            obs_type,
            is_iis,
            seq_typ,
            seq_nam,
            lamps,
            dataId,
        )

        als = addTraceLambdaToArclines(arcLines, detectorMap)
        data = computeImageQuality(als)
        data["peakRatio"] = np.nan

        # Flexure diagnostic: offset from static calibration detectorMap.
        # Measured minus calibration, as drp_stella's residuals: positive
        # dxCenter means the fiber sits to the right of the calibration's prediction.
        if detectorMapCalib is not None and "x" in data.columns:
            calibX = detectorMapCalib.getXCenter(
                np.asarray(data["fiberId"], dtype=np.int32),
                np.asarray(data["y"], dtype=np.float64),
            )
            data["dxCenter"] = data["x"] - calibX
        else:
            data["dxCenter"] = np.nan

        # Count good (unflagged + finite-FWHM) arc-line measurements.
        good_arc = data["fwhm"].notna() & ~data["flag"]
        if "status" in data.columns:
            good_arc &= data["status"] == 0
        n_good_arc = int(good_arc.sum())
        # nLines counts the rows of ``lines`` (lines and trace centroids found by
        # fitDetectorMap), whichever path measures the FWHM. linesPerFiber is the
        # count that compares between frames: the measurements per lit fiber of what the
        # frame is for, lamp lines on an arc and trace centroids on a quartz. Per
        # lit fiber, so that a design lighting one fiber group reads like a full
        # slit; without the trace centroids an arc also records, whose number
        # follows the illumination and the pipeline rather than the lamp.
        n_lines = len(data)
        # On an IIS frame the lit fibers are the engineering fibers: reduceExposure
        # measures lines and traces on those alone.
        lit_fibers, n_fibers = litFibers(pfsConfig, detectorMap.fiberId, engineering=is_iis)
        illumination = "iis" if is_iis else "full"
        n_lit_fibers = len(lit_fibers) if lit_fibers is not None else None
        if "description" in data.columns:
            is_trace_row = (data["description"] == "Trace").to_numpy()
        else:
            is_trace_row = np.zeros(len(data), dtype=bool)
        on_lit = (
            data["fiberId"].isin(lit_fibers).to_numpy()
            if lit_fibers is not None
            else np.ones(len(data), bool)
        )
        # Lamp-line rows on lit fibers: what pctFlagged is computed over on the arc-line path.
        lamp_line_rows = ~is_trace_row & on_lit
        lines_per_fiber = np.nan
        if n_lit_fibers and obs_type in ("arc", "trace"):
            counted = is_trace_row if obs_type == "trace" else ~is_trace_row
            if obs_type == "trace" and "description" not in data.columns:
                counted = np.ones(len(data), dtype=bool)  # no descriptions: every row is a centroid
            unflagged = ~data["flag"].to_numpy(dtype=bool)
            lines_per_fiber = float((counted & on_lit & unflagged).sum()) / n_lit_fibers

        # Saturation, against the detector's own level (the calexp's SAT plane), and each
        # species' line flux for cross-visit lamp stability. Measured, not judged.
        # Null without a calexp, not zero.
        line_saturated, pct_saturated_pixels = None, np.nan
        if calexp is not None:
            is_saturated = (calexp.mask.array & calexp.mask.getPlaneBitMask(["SAT"])) != 0
            line_saturated, _, _, pct_saturated_pixels = measureSaturation(
                is_saturated, detectorMap, data, lit_fibers
            )
        n_saturated_lines = (
            int((line_saturated & lamp_line_rows).sum()) if line_saturated is not None else np.nan
        )
        species_flux = speciesLineFlux(data, select=on_lit, saturated=line_saturated)

        dense_data = False
        using_fluxstd_filter = False
        # The FWHM is read from the fiberProfiles calibration (``traceOnly``).
        using_profile_widths = False
        # force_sparse rules out the rows left for visit types the arc catalog says
        # nothing about: a quartz without a usable calexp, a twilight, a dark.
        force_sparse = False
        # The frame as the log names it: "Arc", "Quartz", "IIS arc", "IIS quartz".
        kind = {"arc": "arc", "trace": "quartz"}.get(obs_type, obs_type)
        kind = f"IIS {kind}" if is_iis else kind.capitalize()

        if obs_type == "arc":
            # Arc (the arc lamp on the lit fibers: all 600, one fiber group, or an
            # IIS arc's 16 engineering fibers): primary path is the arc-line shape
            # measurements.  Fall back to calexp only when too few good catalog
            # matches are found (e.g. b arm on sky fields where arc lines are not
            # in the catalog).
            if n_good_arc >= self.config.minGoodLines:
                self.log.debug(
                    f"{kind} %s: using %d good arc-line measurements.",
                    dataId,
                    n_good_arc,
                )
            elif calexp is not None:
                self.log.info(
                    f"{kind} %s: only %d good arc lines (< minGoodLines=%d); trying calexp fallback.",
                    dataId,
                    n_good_arc,
                    self.config.minGoodLines,
                )
                calexp_data = self._buildImageWidthData(
                    calexp, detectorMap, detectorMapCalib, fiberIds=lit_fibers
                )
                n_good_calexp = int((calexp_data["fwhm"].notna() & ~calexp_data["flag"]).sum())
                calexp_good_frac = n_good_calexp / max(len(calexp_data), 1)
                if n_good_calexp > 0 and (1.0 - calexp_good_frac) < self.config.maxCalexpFlagRate:
                    self.log.info(
                        f"{kind} calexp fallback: %d good samples (%.1f%% good) for %s.",
                        n_good_calexp,
                        100.0 * calexp_good_frac,
                        dataId,
                    )
                    data = calexp_data
                    dense_data = True
                else:
                    self.log.warning(
                        f"{kind} calexp fallback too sparse for %s"
                        " (%d good, %.1f%% flagged); keeping %d arc-line measurements.",
                        dataId,
                        n_good_calexp,
                        100.0 * (1.0 - calexp_good_frac),
                        n_good_arc,
                    )
            else:
                self.log.warning(
                    f"{kind} %s: only %d good arc lines (< minGoodLines=%d)"
                    " and no calexp; FWHM will be sparse.",
                    dataId,
                    n_good_arc,
                    self.config.minGoodLines,
                )

        elif obs_type == "trace":
            # Quartz/trace (the quartz lamp on the lit fibers, an IIS quartz's
            # engineering fibers included): no arc lines to fit, so use calexp
            # trace widths as the primary path.
            # Any arc-line rows still held in ``data`` describe a catalog that
            # does not apply to a quartz frame, so the visit stays sparse
            # unless one of the two measurement paths succeeds.
            force_sparse = True
            calexp_failed = False
            if calexp is not None:
                self.log.info(
                    f"{kind} %s: measuring FWHM from calexp.",
                    dataId,
                )
                # Only the lit fibers are sampled: on a design lighting one fiber
                # group, the dark ones would otherwise make most samples flagged
                # and the calexp unusable (maxCalexpFlagRate).
                calexp_data = self._buildImageWidthData(
                    calexp, detectorMap, detectorMapCalib, fiberIds=lit_fibers
                )
                n_good_calexp = int((calexp_data["fwhm"].notna() & ~calexp_data["flag"]).sum())
                calexp_good_frac = n_good_calexp / max(len(calexp_data), 1)
                if n_good_calexp > 0 and (1.0 - calexp_good_frac) < self.config.maxCalexpFlagRate:
                    self.log.info(
                        f"{kind} calexp: %d good samples (%.1f%% good) for %s.",
                        n_good_calexp,
                        100.0 * calexp_good_frac,
                        dataId,
                    )
                    data = calexp_data
                    dense_data = True
                    force_sparse = False
                else:
                    # The denominator is the number of cross-dispersion samples
                    # (tens of thousands), so a count and a rounded percentage
                    # alone read as a contradiction ("10 good, 100.0% flagged").
                    calexp_failed = True
                    self.log.warning(
                        f"{kind} calexp unusable for %s: %d of %d cross-dispersion samples measured"
                        " cleanly (%.3f%%, need > %.0f%%).",
                        dataId,
                        n_good_calexp,
                        len(calexp_data),
                        100.0 * calexp_good_frac,
                        100.0 * (1.0 - self.config.maxCalexpFlagRate),
                    )

            # The fiber profile widths are the fallback whenever the calexp gave
            # no usable measurement, whether or not a calexp exists. They are a
            # calibration, not a measurement of this exposure: ``traceOnly``
            # marks them, and medFwhm then counts as not measured
            # (`pfs.drp.qa.metrics.registry`).
            if not dense_data and fiberProfiles is not None:
                # At the level of the problem it resolves, so that an operator
                # filtering at WARNING who saw the failure also sees the outcome.
                report = self.log.warning if calexp_failed else self.log.info
                report(
                    f"{kind} %s: using fiber profile calibration widths;"
                    " FWHM is not measured from this exposure.",
                    dataId,
                )
                data = self._buildProfileData(fiberProfiles, detectorMap)
                dense_data = True
                using_profile_widths = True
                force_sparse = False
            elif not dense_data:
                self.log.warning(
                    f"{kind} %s: neither a usable calexp nor fiberProfiles; FWHM will be sparse.",
                    dataId,
                )

        elif obs_type == "twilight":
            # Twilight sky: every fiber sees the same sky. Its FWHM belongs to
            # the sky lines, which are not measured yet; FLUXSTD fibers carry no
            # star here, so the science path below does not apply either.
            force_sparse = True
            self.log.info("Twilight %s: FWHM is not measured on twilight frames.", dataId)

        elif obs_type in ("dark", "bias"):
            # Nothing is illuminated: no metric of this task applies.
            force_sparse = True
            self.log.info("%s %s: nothing to measure.", obs_type.capitalize(), dataId)

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
            # obs_type == "unknown": W_SEQTYP header absent, or a sequence type with no path of its own.
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
                    using_profile_widths = True
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
            "fwhmIntrinsic",
            "theta",
            "dxCenter",
            "flux",
            "fluxErr",
            "flag",
            "traceOnly",
            "peakRatio",
            "snr",
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
            # The visit type rules out every row that is left: on quartz/science
            # frames without a usable calexp the surviving arc lines are
            # incidental, and a twilight or dark has nothing to measure.  The rows
            # are still returned in ``lineData`` for inspection, but no summary
            # metric may be derived from them.
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
        # illumination).  For sparse data (fewer than minGoodLines arc lines, as
        # on a lamp-mismatched arc frame) pctFlagged reflects catalog quality
        # rather than optical quality and is set to NaN.
        #
        # The FLUXSTD calexp path also suppresses pctFlagged: stellar fibers
        # only reach S/N threshold at a small fraction of sampled rows, so the
        # flag rate reflects exposure depth rather than optical quality.  FWHM
        # is still reported when the good-fraction threshold is met.
        #
        # The fiber-profile path flags nothing (`_buildProfileData`), so its
        # pctFlagged would be zero by construction rather than by measurement.
        sparse_fallback = force_sparse or ((n_good_arc < self.config.minGoodLines) and not dense_data)
        if sparse_fallback or using_fluxstd_filter or using_profile_widths:
            pct_flagged = np.nan
            flagBreakdown = {}
        else:
            # On the arc-line path, the lamp lines on lit fibers: the trace centroids
            # an arc also records are almost never flagged, and how many there are
            # follows the illumination, so counting them made pctFlagged depend on
            # the fibers lit rather than on the lines. The calexp
            # samples are of lit fibers already.
            flagData = data if dense_data else data[lamp_line_rows]
            nTotal = max(len(flagData), 1)
            pct_flagged = 100.0 * flagData["flag"].sum() / nTotal
            flagBreakdown = self._computeFlagBreakdown(
                flagData,
                nTotal,
                isArcLinePath=not dense_data,
            )

        title = "{visit} {arm}{spectrograph}".format(**dataId)

        metricsDict = {
            "medFwhm": [med_fwhm],
            "medDxCenter": [medDxCenter],
            "dxCenterRms": [dxCenterRms],
            "pctFlagged": [pct_flagged],
            "nLines": [n_lines],
            "linesPerFiber": [lines_per_fiber],
            "nLitFibers": [n_lit_fibers if n_lit_fibers is not None else np.nan],
            "nFibers": [n_fibers if n_fibers is not None else np.nan],
            "nTraceRows": [int(is_trace_row.sum())],
            "nSaturatedLines": [n_saturated_lines],
            "pctSaturatedPixels": [pct_saturated_pixels],
            "traceOnly": [trace_only],
            "obsType": [obs_type],
            "illumination": [illumination],
            "seqName": [seq_nam],
            "seqType": [seq_typ],
            "lamps": [lamps],
        }
        for bitName, pct in flagBreakdown.items():
            metricsDict[f"pct{bitName}"] = [pct]

        metrics = pd.DataFrame(metricsDict)
        for key in ("visit", "arm", "spectrograph"):
            if key in dataId:
                metrics[key] = dataId[key]

        dxStr = f"{medDxCenter:+.3f}px" if np.isfinite(medDxCenter) else "NaN"
        self.log.info(
            "IQ QA  %s  %-22s  medFWHM=%.2fpx  dxCenter=%s  pctFlagged=%s",
            title,
            seq_nam,
            med_fwhm,
            dxStr,
            f"{pct_flagged:.1f}%" if np.isfinite(pct_flagged) else "NaN",
        )

        # Empty frames are left out of the concatenation: pandas would let them decide the dtypes.
        species_frames = [
            frame
            for frame in (
                speciesFrame(dataId, speciesFitStats(reduceExposureMetadata)),
                metricsFrame(dataId, species_flux),
            )
            if len(frame)
        ]
        species_metrics = (
            pd.concat(species_frames, ignore_index=True) if species_frames else speciesFrame(dataId, {})
        )

        return Struct(
            lineData=data,
            iqQaMetrics=metrics,
            iqQaSpeciesMetrics=species_metrics,
        )

    def _classifyVisit(
        self,
        calexp: lsst.afw.image.Exposure | None,
        pfsConfig: PfsConfig | None,
        spectrograph: int | None = None,
    ) -> tuple[str, bool, str, str, str]:
        """Classify the observation type from FITS header metadata.

        Reads ``W_SEQTYP`` and ``W_SEQNAM`` from the ``calexp`` metadata, or
        from ``pfsConfig.header`` when the calexp has no ``W_SEQTYP``
        (`pfs.drp.qa.metrics.registry.sequenceOf`), and determines whether
        the illumination is IIS
        (engineering fibers) or regular (all 600 science fibers) from the lamp
        headers (`~lsst.obs.pfs.utils.getLamps`) or the pfsConfig
        (`engineeringLit`), which decides when present.

        Parameters
        ----------
        calexp : `lsst.afw.image.Exposure` or `None`
            Post-ISR calibrated image for this quantum.
        pfsConfig : `PfsConfig` or `None`
            Fiber configuration for this visit.
        spectrograph : `int` or `None`, optional
            The quantum's spectrograph, for `engineeringLit`.

        Returns
        -------
        obs_type : `str`
            `pfs.drp.qa.metrics.registry.observationType` of the sequence:
            ``"arc"``, ``"trace"``, ``"science"``, ``"allsky"``
            (``W_SEQNAM`` starting "sky"), ``"twilight"`` (starting
            "twilight"), ``"dark"``, ``"bias"``, or ``"unknown"`` (when
            ``W_SEQTYP`` is absent or has no measurement of its own).
        is_iis : `bool`
            True when the illumination comes from the 16 IIS engineering
            fibers. The pfsConfig decides when there is one (`engineeringLit`:
            the spectrograph's engineering fibers all ``GOOD``), since it is
            what reduceExposure went by and so says which fibers ``lines``
            holds; otherwise a lamp name ending in ``"_eng"``.
        seq_nam : `str`
            Raw ``W_SEQNAM`` header value (e.g. ``"Arc: HgCd"``), or an
            empty string when the header is absent.
        seq_typ : `str`
            Raw ``W_SEQTYP`` header value (e.g. ``"scienceArc"``), or an
            empty string when no header has it.
        lamps : `str`
            The lamps the lamp headers say were lit, obs_pfs's names sorted
            and joined by ``+`` (``"Ne"``, ``"Kr_eng"``); empty when none is
            lit or the headers cannot be read.
        """
        # Prefer calexp metadata; fall back to pfsConfig header.
        calexpHeader = calexp.getMetadata() if calexp is not None else None
        configHeader = getattr(pfsConfig, "header", None) if pfsConfig is not None else None
        metadata = calexpHeader if calexpHeader is not None else configHeader

        if metadata is None:
            return "unknown", False, "", "", ""

        seq_typ, seq_nam = sequenceOf(calexpHeader, configHeader)
        if not seq_typ:
            # No header has the sequence type; the name is kept as the header at hand gives it.
            seq_nam = (metadata.get("W_SEQNAM") or "").strip()
        obs_type = observationType(seq_typ, seq_nam)
        if obs_type == "unknown" and seq_typ:
            self.log.debug("W_SEQTYP=%r has no measurement of its own; using heuristic fallback.", seq_typ)

        # Distinguish IIS (engineering fiber) illumination from regular (all
        # 600 science fibers).  IIS lamp header names end with "_eng"
        # (e.g. "Ar_eng", "Quartz_eng") while regular lamps do not.
        # getLamps answers {"Quartz"} for any DATA-TYP FLAT before reading the
        # lamp headers, which hides an IIS quartz (W_IISQTH): the lamp headers
        # are read without DATA-TYP (PIPE2D-1953).
        is_iis = False
        lamps = ""
        try:
            from lsst.obs.pfs.utils import getLamps

            lit = getLamps(_WithoutDataType(metadata))
            is_iis = any(name.endswith("_eng") for name in lit)
            lamps = "+".join(sorted(lit))
        except Exception:
            pass  # obs_pfs unavailable or headers missing; assume regular
        # The pfsConfig is the witness reduceExposure goes by, so it decides which fibers
        # ``lines`` holds and which are lit; the headers decide only without one. Some n-arm
        # raws record the IIS lamp off.
        if pfsConfig is not None:
            byConfig = engineeringLit(pfsConfig, spectrograph)
            if byConfig != is_iis:
                self.log.info(
                    "The pfsConfig says %s (engineering fibers %sGOOD); the lamp headers say %s. Taking the pfsConfig.",
                    "IIS" if byConfig else "not IIS",
                    "" if byConfig else "not all ",
                    "IIS" if is_iis else "not IIS",
                )
            is_iis = byConfig

        return obs_type, is_iis, seq_nam, seq_typ, lamps

    def _buildProfileData(self, fiberProfiles: FiberProfileSet, detectorMap: DetectorMap) -> pd.DataFrame:
        """Build an image-quality DataFrame from fiber profile trace widths.

        Reads the FWHM of each swath's profile directly, at half its maximum
        (`pfs.drp.qa.crossDispersion.profileFwhm`), and maps each swath centre
        to detector (x, y) coordinates and wavelength via ``detectorMap``. The
        profile is not Gaussian: a Gaussian fitted to it understates this
        width by 8-10 %, and drp_stella's second moment
        (``calculateStatistics().width``) gave 2.3-3.7 px on Run25 detectors
        whose profiles a Gaussian fit put at 2.7-2.8 px
        (``docs/trace-widths.ipynb``).

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
            swath_rows = profile.rows
            n = len(swath_rows)
            if n == 0:
                continue

            fwhm = profileFwhm(profile.index, profile.profiles)

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
        """Measure trace widths directly from the calexp's pixels.

        Every ``profileYStride``-th row is fitted with
        `pfs.drp.qa.crossDispersion.measureImageWidths`: each fiber jointly
        with its neighbours, as pixel-integrated Gaussians sharing one width,
        plus a background. ``fwhm`` includes the pixel response, as arc-line
        ``SdssShape`` widths and ``fiberProfiles`` do; ``fwhmIntrinsic`` does
        not. Unlike ``_buildProfileData``, this reflects the optical state of
        the exposure rather than of a (possibly stale) calibration.

        Parameters
        ----------
        calexp : `lsst.afw.image.Exposure`
            Calibrated exposure for this quantum.
        detectorMap : `DetectorMap`
            This visit's detector map, giving the trace centers the fit starts
            from and the wavelengths.
        detectorMapCalib : `DetectorMap` or `None`, optional
            Static calibration detectorMap.  When provided, the spatial
            offset ``dxCenter`` (calibration prediction minus measured
            center) is recorded for each sample as a flexure diagnostic.
        fiberIds : `set` or `None`, optional
            If provided, only these fibers are reported (e.g. FLUXSTD fibers
            from ``pfsConfig``, or the fibers it lights, `litFibers`). Every
            fiber is still fitted, because a fiber's fit depends on its
            neighbours.

        Returns
        -------
        `pandas.DataFrame`
            One row per (fiber, sampled row), with the columns of
            `pfs.drp.qa.crossDispersion.IMAGE_WIDTH_COLUMNS`. ``traceOnly``
            is `False`: these are measurements of the exposure.
        """
        height = calexp.image.array.shape[0]
        rows = np.arange(self.config.profileYStride // 2, height, self.config.profileYStride)
        allFiberIds = np.asarray(detectorMap.fiberId, dtype=np.int32)
        yy = np.repeat(rows[:, None].astype(np.float64), len(allFiberIds), axis=1).ravel()
        ff = np.tile(allFiberIds, len(rows))
        shape = (len(rows), len(allFiberIds))
        xCenters = np.asarray(detectorMap.getXCenter(ff, yy), dtype=np.float64).reshape(shape)
        wavelengths = np.asarray(detectorMap.findWavelength(ff, yy), dtype=np.float64).reshape(shape)
        calibXCenters = None
        if detectorMapCalib is not None:
            calibXCenters = np.asarray(detectorMapCalib.getXCenter(ff, yy), dtype=np.float64).reshape(shape)
        badBits = calexp.mask.getPlaneBitMask(["BAD", "SAT", "CR", "NO_DATA"])
        return measureImageWidths(
            calexp.image.array[rows],
            calexp.variance.array[rows],
            (calexp.mask.array[rows] & badBits) != 0,
            rows,
            allFiberIds,
            xCenters,
            wavelengths,
            calibXCenters,
            select=fiberIds,
            minPeakSN=self.config.minPeakSN,
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
