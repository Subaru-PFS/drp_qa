"""Line residuals against the calibration and the visit's detectorMap: measurement and verdict.

``dmLineResiduals`` measures every line of ``lines`` against two references
(``reference``):

- ``calib``: ``detectorMap_calib``, the calibration the visit is reduced
  against (rule R1 of ``docs/qa-principles.md``). An offset from it is drift:
  has the instrument moved away from the calibration?
- ``visit``: the visit's own ``detectorMap``, which ``reduceExposure``
  adjusted to these lines. The lines the fit used measure the fit; those it
  held back (``RESERVED``) measure how well the adjusted map predicts lines it
  was not fitted to, as it must for a science frame.

The lines are the fit's own: the good rows of ``lines`` (unflagged, no ``BAD``
status bit, as ``fitDetectorMap``'s ``lineFlags``) that it marked
``DETECTORMAP_USED`` or ``DETECTORMAP_RESERVED``. There is no other selection
and no clipping (R5: the statistics are robust), and trace centroids are kept
whatever the lamp.

- Trace centroids (``description == "Trace"``) measure x only: their row is
  fixed, so ``dx = x - detectorMap.getXCenter(fiberId, y)``.
- Lamp lines measure both: ``(dx, dy) = (x, y) - detectorMap.findPoint(fiberId,
  wavelength)``.

Residuals are measured minus reference, the sign of drp_stella's. Per
reference, species (``description``, R7) and ``statusType``, the median and
robust RMS of ``dx`` and ``dy``, ``nLines`` and ``linesPerFiber`` (per lit
fiber, so a design lighting one fiber group is not read as sparse,
PIPE2D-1935). A species with fewer than ``minLines`` lines gets NaN
statistics: not measured. The components are kept apart, never combined.

``statusType`` is ``BOTH`` (``USED`` and ``RESERVED`` together), ``USED`` or
``RESERVED``. When the visit's fit marked no line, it did not run or failed,
and the visit is reduced against the calibration unadjusted: every good line is
then measured once, as ``UNFITTED``, and the summary's ``fitted`` is false.

``obsType`` is the frame's observation type, from the ``W_SEQTYP`` header of
the visit's pfsConfig as ``imageQualityQa`` classifies it (`observationType`:
``trace`` for a quartz, ``arc``, ``twilight``, ...). It is a population column:
trace centroids are measured on every frame, but on an arc the continuum is
faint and they scatter two to three times more than on a quartz, so a
threshold on their scatter is derived and judged per frame type.

Judged (`JUDGED`), everything else stored as a diagnostic:

- against ``calib``, over ``BOTH`` (or ``UNFITTED``) lines: the drift. The
  config judges the median x offset of trace centroids and the median y
  offset of lamp lines; lamp lines' x offset is stored, and judged only where
  a thresholds file names it.
- against ``visit``, over ``RESERVED`` lines: the robust RMS, the scatter the
  adjusted map leaves on lines it was not fitted to. The weighted RMS and the
  softening are not recomputed here: ``imageQualityQa`` reads drp_stella's own
  from the ``reduceExposure`` log (``iqQaMetrics``, ``iqQaSpeciesMetrics``).

Stack-free: the detectorMaps are duck-typed (``fiberId``, ``getXCenter``,
``findPoint``), so it is tested with fakes.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pfs.drp.qa.metrics.gate import STATUS_ORDER, judge

__all__ = [
    "BAD_STATUS",
    "BOTH",
    "CALIB",
    "JUDGED",
    "LINE_COLUMNS",
    "QUARTZ",
    "RECOMMENDED_ACTIONS",
    "REFERENCES",
    "RESERVED",
    "RESIDUAL_COLUMNS",
    "RESIDUAL_METRICS",
    "SKY_SPECIES",
    "STATUS_TYPES",
    "SUMMARY_COLUMNS",
    "TRACE",
    "UNFITTED",
    "USED",
    "VISIT",
    "addObsType",
    "cameraSet",
    "frameType",
    "lineResiduals",
    "lineTable",
    "longResiduals",
    "observationType",
    "residualThresholds",
    "residualVerdict",
    "robustRms",
    "speciesResiduals",
    "wideResiduals",
]

#: The ``description`` of trace centroids in ``lines``.
TRACE = "Trace"
#: Line species of the night sky, measured on twilights and science frames.
SKY_SPECIES = ("OH", "O2", "OI", "NaI")
#: The ``obsType`` of a quartz frame: the only frames trace scatter is judged on.
QUARTZ = "trace"

#: drp_stella's ``ReferenceLineStatus.BAD`` (NOT_VISIBLE, BLEND, SUSPECT,
#: REJECTED, BROAD): the status bits ``fitDetectorMap`` drops by default.
BAD_STATUS = 0x1F
#: drp_stella's ``ReferenceLineStatus.DETECTORMAP_USED`` and ``DETECTORMAP_RESERVED``.
_USED_BIT, _RESERVED_BIT = 0x20, 0x40

#: Values of ``reference``: ``detectorMap_calib``, and the visit's adjusted ``detectorMap``.
CALIB, VISIT = "calib", "visit"
REFERENCES = (CALIB, VISIT)

#: Values of ``statusType``: lines the visit's detectorMap fit used, those it
#: held back, and both together.
USED, RESERVED, BOTH = "USED", "RESERVED", "BOTH"
STATUS_TYPES = (BOTH, USED, RESERVED)
#: ``statusType`` of every good line of a detector whose fit marked none.
UNFITTED = "UNFITTED"

#: The residual statistics, with the component each comes from.
RESIDUAL_METRICS = {
    "medianDeltaX": "x",
    "robustRmsDeltaX": "x",
    "medianDeltaY": "y",
    "robustRmsDeltaY": "y",
}

#: What is judged: per reference, the ``statusType`` values and the metrics.
#: Any other row, or metric, is stored without a verdict.
JUDGED = {
    CALIB: ((BOTH, UNFITTED), tuple(RESIDUAL_METRICS)),
    VISIT: ((RESERVED,), ("robustRmsDeltaX", "robustRmsDeltaY")),
}

#: The counts reported for every species.
_COUNT_METRICS = ("nLines", "linesPerFiber")

#: The operator's actions, least to most severe (`recommendedAction`). A
#: detector takes the most severe of its crossed metrics' actions:
#:
#: - a median offset from the calibration at WARN: ``APPLY_SHIFT``. The
#:   calibration no longer matches, but the visit's first-order detectorMap
#:   adjustment absorbs a shift;
#: - a median offset at FAIL: ``RECALIBRATE``. The offset nears the 3 px radius
#:   within which drp_stella finds lines (``centroidLines.peakSearch``,
#:   ``centroidTraces.searchRadius``), so the adjustment cannot be trusted;
#: - a robust RMS at WARN or FAIL, against either reference: ``RECALIBRATE``.
#:   Scatter is change a shift cannot absorb;
#: - any crossing on ``UNFITTED`` lines: ``RECALIBRATE``. No adjustment was
#:   applied, so the offset is in the reduced data.
#:
#: ``INSUFFICIENT_DATA`` when nothing was measurable; `residualVerdict` also
#: gives ``UNJUDGED`` when lines were measured but no threshold applies.
RECOMMENDED_ACTIONS = ("NOMINAL", "APPLY_SHIFT", "RECALIBRATE")

#: Columns of ``dmLineResidualMetrics``, in order, with their dtypes: one row
#: per detector, reference, species, ``statusType`` and metric, with the
#: metric's verdict (empty when not judged).
RESIDUAL_COLUMNS = {
    "visit": "int64",
    "arm": "string",
    "spectrograph": "int64",
    "cameraSet": "string",
    "obsType": "string",
    "reference": "string",
    "description": "string",
    "statusType": "string",
    "metric": "string",
    "value": "float64",
    "status": "string",
}

#: Columns of ``dmLineResidualData``, in order, with their dtypes: one row per
#: line measured, with its residual against each reference (NaN where that
#: reference predicts nothing, and ``dy`` for trace centroids).
LINE_COLUMNS = {
    "visit": "int64",
    "arm": "string",
    "spectrograph": "int64",
    "fiberId": "int32",
    "wavelength": "float64",
    "x": "float64",
    "y": "float64",
    "xErr": "float64",
    "yErr": "float64",
    "description": "string",
    "statusType": "string",
    **{f"d{c}{ref.capitalize()}": "float64" for ref in REFERENCES for c in "xy"},
}

#: Columns of ``dmLineResidualSummary`` set by `residualVerdict`.
SUMMARY_COLUMNS = (
    "fitted",
    "nSpecies",
    "nSpeciesMeasured",
    "qaStatus",
    "qaDecidedBy",
    "qaReason",
    "recommendedAction",
)

#: Arms in the order camera sets are named (``brn``, ``bmn``, ``bm``).
_ARM_ORDER = "brmn"
#: Arm of each bit of ``PfsConfig.camMask``: cameras b1 r1 n1 m1 b2 ... (``PfsConfig._allCams``).
_CAM_MASK_ARMS = "brnm" * 4


def robustRms(values: Any) -> float:
    """Return the robust RMS of the finite values: 0.741 times the interquartile range.

    The definition of ``pfs.drp.stella.utils.math.robustRms``, which this
    stack-free module cannot import (rule R5).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    lower, upper = np.percentile(values, (25.0, 75.0))
    return float(0.741 * (upper - lower))


def cameraSet(pfsConfig: Any, arm: str | None = None) -> str | None:
    """Return the arms a visit read, e.g. ``"brn"``, ``"bmn"``, ``"br"`` or ``"bm"``.

    b-arm offsets from the calibration depend on it: in Run25 the b cameras sit
    +0.45 to +0.52 px from ``detectorMap_calib`` when read with m, and about 0
    with r (PIPE2D-1927). It is a population column, nothing more.

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.PfsConfig` or `None`
        Duck-typed: ``camMask`` (``W_CAMMSK``) when non-zero, else ``arms``
        (``ARMS``, which datamodel defaults to ``"brn"`` when the header
        lacks it).
    arm : `str`, optional
        The detector's own arm, always included.

    Returns
    -------
    `str` or `None`
        The arms in ``b, r, m, n`` order; `None` without a pfsConfig.
    """
    if pfsConfig is None:
        return None
    camMask = int(getattr(pfsConfig, "camMask", 0) or 0)
    if camMask:
        arms = {_CAM_MASK_ARMS[bit] for bit in range(len(_CAM_MASK_ARMS)) if camMask & (1 << bit)}
    else:
        arms = set(str(getattr(pfsConfig, "arms", "") or ""))
    if arm:
        arms.add(arm)
    named = "".join(a for a in _ARM_ORDER if a in arms)
    return named or None


def observationType(header: Any) -> str | None:
    """Return the frame's observation type, as ``imageQualityQa`` classifies it.

    Parameters
    ----------
    header : mapping or `None`
        The visit's header (``PfsConfig.header``, or any ``get``): ``W_SEQTYP``
        and ``W_SEQNAM``.

    Returns
    -------
    `str` or `None`
        ``arc``, ``trace`` (a quartz), ``dark``, ``twilight``, ``allsky`` or
        ``science`` (``scienceObject`` by its sequence name), ``unknown`` when
        ``W_SEQTYP`` is absent or unrecognised; `None` without a header. The
        values of ``iqQaMetrics.obsType``, so one thresholds population
        vocabulary serves both tasks.
    """
    if header is None:
        return None
    seqType = str(header.get("W_SEQTYP") or "").strip()
    seqName = str(header.get("W_SEQNAM") or "").strip().lower()
    if seqType == "scienceArc":
        return "arc"
    if seqType == "scienceTrace":
        return QUARTZ
    if seqType == "scienceDark":
        return "dark"
    if seqType in ("scienceObject", "scienceObject_windowed"):
        if seqName.startswith("twilight"):
            return "twilight"
        return "allsky" if seqName.startswith("sky") else "science"
    return "unknown"


def frameType(rawHeader: Any, pfsConfigHeader: Any) -> str | None:
    """Return the frame's observation type from the raw header, the pfsConfig's standing in.

    The raw header always carries ``W_SEQTYP``; Run25 and Run26 pfsConfigs do
    not. The pfsConfig's type is used when the raw's is missing or
    ``unknown`` and the pfsConfig's is known.

    Parameters
    ----------
    rawHeader, pfsConfigHeader : mapping or `None`
        The raw exposure's header and ``PfsConfig.header``.

    Returns
    -------
    `str` or `None`
        As `observationType`: ``unknown`` when neither header says, `None`
        only when neither header is at hand.
    """
    obsType = observationType(rawHeader)
    if obsType not in (None, "unknown"):
        return obsType
    fallback = observationType(pfsConfigHeader)
    if fallback not in (None, "unknown"):
        return fallback
    # Neither says: unknown when either header is at hand, None when neither is.
    return obsType or fallback


def addObsType(metrics: pd.DataFrame, kinds: pd.DataFrame | None) -> pd.DataFrame:
    """Return ``metrics`` with ``obsType`` filled from ``kinds`` where it is missing, null or ``unknown``.

    Tables stored before ``obsType`` was a column (PIPE2D-1956), or by a task
    whose header lacked ``W_SEQTYP`` (Run25 and Run26 pfsConfigs, before the
    raw header was read), have it from ``iqQaMetrics``, which classifies the
    same frame from the calexp's header.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        Rows with ``visit``, ``arm`` and ``spectrograph``.
    kinds : `pandas.DataFrame` or `None`
        Rows with the same keys and ``obsType``; `None` leaves ``metrics`` as
        it is.
    """
    if kinds is None or "obsType" not in kinds.columns:
        return metrics
    keys = ["visit", "arm", "spectrograph"]
    lookup = kinds[[*keys, "obsType"]].drop_duplicates(keys).rename(columns={"obsType": "_obsType"})
    merged = metrics.merge(lookup, on=keys, how="left")
    merged.index = metrics.index
    if "obsType" in merged.columns:
        known = merged["obsType"].notna() & (merged["obsType"] != "unknown")
        merged["obsType"] = merged["obsType"].where(known, merged["_obsType"])
    else:
        merged["obsType"] = merged["_obsType"]
    return merged.drop(columns="_obsType")


def lineResiduals(
    lines: pd.DataFrame, references: Mapping[str, Any], badStatus: int = BAD_STATUS
) -> pd.DataFrame:
    """Return each line's residual from each reference detectorMap.

    Parameters
    ----------
    lines : `pandas.DataFrame`
        ``lines`` (``ArcLineSet.data``): ``fiberId``, ``wavelength``, ``x``,
        ``y``, ``flag``, ``status``, ``description``; ``xErr`` and ``yErr``
        are copied when present.
    references : mapping [`str`, `pfs.drp.stella.DetectorMap` or `None`]
        The detectorMaps by `REFERENCES` name; one that is absent or `None`
        gives NaN residuals. Duck-typed: ``fiberId``, vectorised
        ``getXCenter(fiberId, y)`` and ``findPoint(fiberId, wavelength)``
        (an ``(N, 2)`` array).
    badStatus : `int`, optional
        Status bits that exclude a line, as ``fitDetectorMap``'s ``lineFlags``.

    Returns
    -------
    `pandas.DataFrame`
        One row per line measured: good (unflagged, no ``badStatus`` bit,
        finite position, and a finite wavelength for a lamp line), with a
        finite residual against some reference (a detectorMap predicts NaN
        off the detector, and nothing for a fiber it lacks), and marked by the
        visit's fit; or, when the fit marked none, every such line. Columns
        ``fiberId``, ``wavelength``, ``x``, ``y``, ``xErr``, ``yErr``,
        ``description``, ``statusType`` (``USED``, ``RESERVED`` or
        `UNFITTED`), and ``dx<Ref>``, ``dy<Ref>`` per reference
        (``dxCalib``, ``dyVisit``, ...), measured minus reference; ``dy`` is
        NaN for trace centroids.
    """
    num = len(lines)
    fiberId = lines["fiberId"].to_numpy(dtype=np.int32)
    x, y = lines["x"].to_numpy(dtype=float), lines["y"].to_numpy(dtype=float)
    description = lines["description"].astype(str).to_numpy()
    wavelength = (
        lines["wavelength"].to_numpy(dtype=float) if "wavelength" in lines.columns else np.full(num, np.nan)
    )
    status = lines["status"].to_numpy(dtype=np.int64) if "status" in lines.columns else np.zeros(num, int)
    isTrace = description == TRACE

    good = ~lines["flag"].to_numpy(dtype=bool)
    good &= (status & badStatus) == 0
    good &= np.isfinite(x) & np.isfinite(y)
    good &= isTrace | np.isfinite(wavelength)

    columns: dict[str, np.ndarray] = {}
    measured = np.zeros(num, dtype=bool)
    for name in REFERENCES:
        dx, dy = _residuals(references.get(name), fiberId, x, y, wavelength, isTrace, good)
        columns[f"dx{name.capitalize()}"] = dx
        columns[f"dy{name.capitalize()}"] = dy
        # The component a line measures: a NaN prediction would count as a line with no value.
        measured |= np.where(isTrace, np.isfinite(dx), np.isfinite(dy))
    good &= measured

    # USED wins over RESERVED, as in dmResiduals; a fit never sets both.
    statusType = np.where(status & _USED_BIT, USED, np.where(status & _RESERVED_BIT, RESERVED, ""))
    marked = statusType != ""
    if (good & marked).any():
        good &= marked
    else:
        statusType = np.full(num, UNFITTED)

    def column(name: str) -> np.ndarray:
        return lines[name].to_numpy(dtype=float) if name in lines.columns else np.full(num, np.nan)

    frame = {
        "fiberId": fiberId,
        "wavelength": wavelength,
        "x": x,
        "y": y,
        "xErr": column("xErr"),
        "yErr": column("yErr"),
        "description": description,
        "statusType": statusType,
        **columns,
    }
    return pd.DataFrame({key: value[good] for key, value in frame.items()})


def _residuals(
    detectorMap: Any,
    fiberId: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    wavelength: np.ndarray,
    isTrace: np.ndarray,
    good: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dx, dy)`` of the ``good`` lines from one detectorMap; NaN elsewhere."""
    dx = np.full(len(fiberId), np.nan)
    dy = np.full(len(fiberId), np.nan)
    if detectorMap is None:
        return dx, dy
    # getXCenter raises on a fiber a splined detectorMap lacks.
    use = good & np.isin(fiberId, np.asarray(detectorMap.fiberId))
    traces = use & isTrace
    if traces.any():
        dx[traces] = x[traces] - np.asarray(
            detectorMap.getXCenter(np.ascontiguousarray(fiberId[traces]), np.ascontiguousarray(y[traces])),
            dtype=float,
        )
    lamps = use & ~isTrace
    if lamps.any():
        points = np.asarray(
            detectorMap.findPoint(
                np.ascontiguousarray(fiberId[lamps]), np.ascontiguousarray(wavelength[lamps])
            ),
            dtype=float,
        )
        dx[lamps] = x[lamps] - points[:, 0]
        dy[lamps] = y[lamps] - points[:, 1]
    return dx, dy


def lineTable(
    residuals: pd.DataFrame, dataId: Mapping[str, Any], maxTraceRowsPerFiber: int = 0
) -> pd.DataFrame:
    """Return ``dmLineResidualData``: `lineResiduals` with the detector, in `LINE_COLUMNS`.

    A fit can keep millions of trace rows (twilight b3/b4 kept about 2
    million), so the trace centroids may be sampled: those the fit used, and
    those it held back (``RESERVED``), each to at most ``maxTraceRowsPerFiber``
    per fiber. Lamp and sky lines, and ``UNFITTED`` rows, are always kept
    whole. Only the stored table is sampled, never the statistics, so a trace
    statistic recomputed from the table is an estimate of the stored one.

    Parameters
    ----------
    residuals : `pandas.DataFrame`
        `lineResiduals`.
    dataId : mapping
        ``visit``, ``arm``, ``spectrograph``.
    maxTraceRowsPerFiber : `int`, optional
        Most ``USED``, and most ``RESERVED``, trace rows kept per fiber,
        evenly spaced in y; 0 keeps them all.

    Returns
    -------
    `pandas.DataFrame`
        `LINE_COLUMNS`, in the order of ``residuals``; empty, with the same
        columns, when there are no lines.
    """
    if maxTraceRowsPerFiber > 0:
        residuals = residuals[_sampleTraces(residuals, maxTraceRowsPerFiber)]
    frame = residuals.assign(visit=dataId["visit"], arm=dataId["arm"], spectrograph=dataId["spectrograph"])
    return frame.reindex(columns=list(LINE_COLUMNS)).astype(LINE_COLUMNS)


def _sampleTraces(residuals: pd.DataFrame, maxRows: int) -> np.ndarray:
    """Return which rows to keep: at most ``maxRows`` of each fiber's USED, and RESERVED, trace rows; all others.

    Each fiber's rows of one status type are kept evenly spaced in y.
    """
    keep = np.ones(len(residuals), dtype=bool)
    sampled = (
        (residuals["description"] == TRACE) & residuals["statusType"].isin((USED, RESERVED))
    ).to_numpy()
    if not sampled.any():
        return keep
    traces = residuals[sampled]
    position = np.flatnonzero(sampled)
    # Each row's rank in y within its fiber and status type, and that group's row count.
    groups = traces.groupby(["fiberId", "statusType"])["y"]
    rank = groups.rank(method="first").to_numpy(dtype=int) - 1
    count = groups.transform("size").to_numpy(dtype=int)
    # Keep the ranks nearest maxRows evenly spaced points, as np.linspace(0, count - 1, maxRows) would.
    # The point a rank falls nearest, capped at the last: with one point, both ends would round to it.
    step = (count - 1) / max(maxRows - 1, 1)
    point = np.minimum(np.round(rank / np.where(step > 0, step, 1)), maxRows - 1)
    nearest = np.round(point * step)
    keep[position] = (count <= maxRows) | (rank == nearest)
    return keep


def speciesResiduals(
    residuals: pd.DataFrame, minLines: int, litFibers: set[int] | None = None
) -> pd.DataFrame:
    """Return the residual statistics of each species, per reference and ``statusType``.

    Parameters
    ----------
    residuals : `pandas.DataFrame`
        `lineResiduals`.
    minLines : `int`
        Fewest lines a species needs, in each set, for its statistics; below
        it they are NaN (not measured).
    litFibers : `set` [`int`], optional
        The fibers the pfsConfig lights (``imageQualityQa.litFibers``);
        ``linesPerFiber`` is NaN without them, or when none is lit.

    Returns
    -------
    `pandas.DataFrame`
        One row per reference (`REFERENCES` order), species (sorted) and
        ``statusType`` (`STATUS_TYPES` order, or `UNFITTED` alone):
        ``reference``, ``description``, ``statusType``, ``nLines`` (lines
        with a residual against that reference in the component the species
        measures: x for trace centroids, y for lamp lines), ``linesPerFiber``
        (of those, the lines on lit fibers per lit fiber), and the
        `RESIDUAL_METRICS`; the y statistics are NaN for trace centroids.
    """
    rows = []
    unfitted = (residuals["statusType"] == UNFITTED).any()
    for reference in REFERENCES:
        suffix = reference.capitalize()
        for description, group in residuals.groupby("description", sort=True):
            isTrace = description == TRACE
            primary = group[f"dx{suffix}" if isTrace else f"dy{suffix}"]
            group = group[np.isfinite(primary.to_numpy(dtype=float))]
            if unfitted:
                sets = [(UNFITTED, group)]
            else:
                sets = [(BOTH, group)] + [(s, group[group["statusType"] == s]) for s in (USED, RESERVED)]
            for statusType, lines in sets:
                nLines = len(lines)
                row = {
                    "reference": reference,
                    "description": description,
                    "statusType": statusType,
                    "nLines": nLines,
                    "linesPerFiber": np.nan,
                }
                if litFibers:
                    row["linesPerFiber"] = float(lines["fiberId"].isin(litFibers).sum()) / len(litFibers)
                for metric, component in RESIDUAL_METRICS.items():
                    values = lines[f"d{component}{suffix}"]
                    if nLines < minLines or (isTrace and component == "y"):
                        row[metric] = np.nan
                    elif metric.startswith("median"):
                        row[metric] = float(np.nanmedian(values)) if values.notna().any() else np.nan
                    else:
                        row[metric] = robustRms(values)
                rows.append(row)
    columns = ["reference", "description", "statusType", *_COUNT_METRICS, *RESIDUAL_METRICS]
    return pd.DataFrame(rows, columns=columns).astype({"nLines": "int64"})


def residualThresholds(
    xWarn: float,
    xFail: float,
    yWarn: float,
    yFail: float,
    xRmsWarn: float = np.nan,
    xRmsFail: float = np.nan,
    yRmsWarn: float = np.nan,
    yRmsFail: float = np.nan,
    source: str = "dmLineResiduals config",
) -> pd.DataFrame:
    """Return the config's thresholds as a `gate` table.

    - ``medianDeltaX`` of trace centroids and ``medianDeltaY``, against
      ``calib``, judged on their absolute value: the drift;
    - ``robustRmsDeltaX`` and ``robustRmsDeltaY`` of ``RESERVED`` lines
      against ``visit``: the scatter the adjusted map leaves.

    The population columns are ``reference``, ``description`` and
    ``statusType``; an entry matches any value of a column it leaves null. A
    NaN threshold is not applied; an entry with both NaN is not judged, and
    stops the search for that population (`pfs.drp.qa.metrics.gate`).
    """
    rows = []
    for metric, warn, fail, name, population in (
        ("medianDeltaX", xWarn, xFail, "xWarn/FailThreshold", {"reference": CALIB, "description": TRACE}),
        ("medianDeltaY", yWarn, yFail, "yWarn/FailThreshold", {"reference": CALIB}),
        (
            "robustRmsDeltaX",
            xRmsWarn,
            xRmsFail,
            "xRmsWarn/FailThreshold",
            {"reference": VISIT, "statusType": RESERVED},
        ),
        (
            "robustRmsDeltaY",
            yRmsWarn,
            yRmsFail,
            "yRmsWarn/FailThreshold",
            {"reference": VISIT, "statusType": RESERVED},
        ),
    ):
        rows.append(
            {
                "metric": metric,
                "reference": None,
                "description": None,
                "statusType": None,
                **population,
                "warn": float(warn),
                "fail": float(fail),
                "higherIsWorse": True,
                "absolute": metric.startswith("median"),
                "provenance": f"{source} {name}.",
            }
        )
    return pd.DataFrame(rows).astype({"reference": object, "description": object, "statusType": object})


def residualVerdict(
    species: pd.DataFrame,
    dataId: Mapping[str, Any],
    thresholds: pd.DataFrame | Sequence[pd.DataFrame],
    minLines: int,
    camera: str | None = None,
    obsType: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Judge the `JUDGED` rows and metrics of each species, and the detector.

    Parameters
    ----------
    species : `pandas.DataFrame`
        `speciesResiduals`.
    dataId : mapping
        ``visit``, ``arm``, ``spectrograph``.
    thresholds : `pandas.DataFrame` or a sequence of them
        Threshold tables, highest priority first (`pfs.drp.qa.metrics.gate`).
        Population columns may name ``reference``, ``statusType``, ``arm``,
        ``spectrograph``, ``description``, ``cameraSet`` and ``obsType``.
    minLines : `int`
        As given to `speciesResiduals`; for the reason string.
    camera : `str`, optional
        `cameraSet`.
    obsType : `str`, optional
        `observationType`.

    Returns
    -------
    judged : `pandas.DataFrame`
        `pfs.drp.qa.metrics.gate.judge` of the judged rows, with
        ``reference``, ``description`` and ``statusType``. A metric `JUDGED`
        leaves out for a reference is judged as NaN: no verdict, whatever the
        thresholds.
    summary : `dict`
        `SUMMARY_COLUMNS`. ``fitted`` is false when the visit's fit marked
        no line (`UNFITTED`). ``nSpecies`` and ``nSpeciesMeasured`` count
        species against the calibration. ``qaStatus`` is the worst verdict, or
        ``UNKNOWN`` when no judged row, against either reference, has
        ``minLines`` good lines: not
        measurable, never measured but not combinable. ``qaDecidedBy`` is
        ``"<reference> <description> <metric>"``.
    """
    keep = np.zeros(len(species), dtype=bool)
    for reference, (statusTypes, _) in JUDGED.items():
        keep |= (species["reference"] == reference).to_numpy() & species["statusType"].isin(statusTypes)
    wide = (
        species[keep]
        .assign(
            visit=dataId["visit"],
            arm=dataId["arm"],
            spectrograph=dataId["spectrograph"],
            cameraSet=camera,
            obsType=obsType,
        )
        .reset_index(drop=True)
    )
    for reference, (_, metrics) in JUDGED.items():
        blank = [metric for metric in RESIDUAL_METRICS if metric not in metrics]
        wide.loc[wide["reference"] == reference, blank] = np.nan
    judged = judge(wide, thresholds)
    rows = judged["row"].to_numpy(dtype=int)
    for column in ("reference", "description", "statusType"):
        judged[column] = wide[column].to_numpy()[rows]

    calib = wide[wide["reference"] == CALIB]
    measured = int((calib["nLines"] >= minLines).sum())
    summary: dict[str, Any] = {
        "fitted": not (species["statusType"] == UNFITTED).any(),
        "nSpecies": len(calib),
        "nSpeciesMeasured": measured,
        "qaStatus": "UNKNOWN",
        "qaDecidedBy": "",
        "qaReason": "",
        "recommendedAction": "INSUFFICIENT_DATA",
    }
    # Measurable against either reference: the visit's map can predict lines the calibration cannot.
    if not (wide["nLines"] >= minLines).any():
        summary["qaReason"] = (
            f"fewer than {minLines} good lines in every species" if len(calib) else "no good lines"
        )
        return judged, summary
    verdicts = judged[judged["status"] != ""]
    if verdicts.empty:
        summary["qaReason"] = "lines measured, but no threshold applies"
        summary["recommendedAction"] = "UNJUDGED"
        return judged, summary
    rank = verdicts["status"].map(STATUS_ORDER.index)
    worst = STATUS_ORDER[int(rank.max())]
    summary["qaStatus"] = worst
    summary["recommendedAction"] = max(
        (_action(row) for row in verdicts.itertuples()), key=RECOMMENDED_ACTIONS.index
    )
    if worst != "PASS":
        decider = verdicts.loc[rank.idxmax()]
        summary["qaDecidedBy"] = f"{decider['reference']} {decider['description']} {decider['metric']}"
        crossed = verdicts[verdicts["reason"] != ""]
        summary["qaReason"] = "; ".join(
            f"{ref} {d}: {r}"
            for ref, d, r in zip(crossed["reference"], crossed["description"], crossed["reason"], strict=True)
        )
    return judged, summary


def _action(verdict: Any) -> str:
    """Return the `RECOMMENDED_ACTIONS` entry for one judged metric (a `judge` row)."""
    if verdict.status == "PASS":
        return "NOMINAL"
    if verdict.statusType == UNFITTED or not verdict.metric.startswith("median"):
        return "RECALIBRATE"
    return "APPLY_SHIFT" if verdict.status == "WARN" else "RECALIBRATE"


def wideResiduals(
    metrics: pd.DataFrame,
    reference: str,
    statusTypes: Sequence[str],
    summary: pd.DataFrame | None = None,
    kinds: pd.DataFrame | None = None,
    excludeCalibInputs: bool = False,
) -> pd.DataFrame:
    """Return ``dmLineResidualMetrics`` rows of one reference as one row per detector and species.

    The long table pivoted on ``metric``, for deriving thresholds
    (`pfs.drp.qa.metrics.calibration.calibrate` takes a column per metric) and
    for the test fixtures made from the validation visits.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        ``dmLineResidualMetrics`` rows (`RESIDUAL_COLUMNS`), of any number of
        detectors.
    reference : `str`
        `CALIB` or `VISIT`: the rows kept.
    statusTypes : sequence of `str`
        The ``statusType`` values kept: ``(BOTH, UNFITTED)`` for the drift
        against the calibration, ``(RESERVED,)`` for the scatter the adjusted
        map leaves.
    summary : `pandas.DataFrame`, optional
        ``dmLineResidualSummary`` rows of the same detectors: adds the
        calibration columns (``calibId``, ``calibRun``, ``calibCreated``,
        ``calibFirstVisit``, ``calibLastVisit``, those present).
    kinds : `pandas.DataFrame`, optional
        A table with ``visit``, ``arm``, ``spectrograph`` and ``obsType`` or
        ``seqName``, e.g. ``iqQaMetrics``: adds ``seqName``, so that the
        validation set's ``seqType`` selectors match the rows, and fills
        ``obsType`` where the stored rows lack it (`addObsType`: tables made
        before it was a column).
    excludeCalibInputs : `bool`, optional
        Drop the detectors whose visit lies within the input visits of the
        calibration they were measured against (``calibFirstVisit`` to
        ``calibLastVisit``, inclusive): a calibration's own inputs sit at zero
        against it by construction, and their adjusted maps are fitted to the
        lines the calibration was. Needs ``summary``.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph``, ``cameraSet``, ``obsType`` (when
        stored), ``reference``, ``description``, ``statusType``, a column per
        metric (``nLines``, ``linesPerFiber`` and `RESIDUAL_METRICS`, NaN
        where the species measures no such component), then the calibration
        and kind columns; sorted by detector and species.

    Raises
    ------
    ValueError
        If ``excludeCalibInputs`` without ``summary``.
    """
    if excludeCalibInputs and summary is None:
        raise ValueError("excludeCalibInputs needs the summary, which holds each calibration's input visits")
    keys = ["visit", "arm", "spectrograph"]
    stored = ["obsType"] if "obsType" in metrics.columns else []
    index = [*keys, "cameraSet", *stored, "reference", "description", "statusType"]
    columns = [*index, *_COUNT_METRICS, *RESIDUAL_METRICS]
    rows = metrics[(metrics["reference"] == reference) & metrics["statusType"].isin(tuple(statusTypes))]
    if rows.empty:
        wide = pd.DataFrame(columns=columns)
    else:
        # pivot, not pivot_table: a detector with no pfsConfig has a null cameraSet, which a groupby drops.
        wide = rows.pivot(index=index, columns="metric", values="value").reset_index()
        wide.columns.name = None
        wide = wide.reindex(columns=columns)
    if summary is not None:
        calibColumns = [
            column
            for column in ("calibId", "calibRun", "calibCreated", "calibFirstVisit", "calibLastVisit")
            if column in summary
        ]
        wide = wide.merge(summary[[*keys, *calibColumns]].drop_duplicates(keys), on=keys, how="left")
        if excludeCalibInputs:
            first = pd.to_numeric(wide["calibFirstVisit"], errors="coerce")
            last = pd.to_numeric(wide["calibLastVisit"], errors="coerce")
            own = first.notna() & last.notna() & (wide["visit"] >= first) & (wide["visit"] <= last)
            wide = wide[~own.to_numpy()]
    if kinds is not None:
        wide = addObsType(wide, kinds)
        if "seqName" in kinds:
            wide = wide.merge(kinds[[*keys, "seqName"]].drop_duplicates(keys), on=keys, how="left")
    return wide.sort_values([*keys, "description"], ignore_index=True)


def longResiduals(
    species: pd.DataFrame,
    judged: pd.DataFrame,
    dataId: Mapping[str, Any],
    camera: str | None = None,
    obsType: str | None = None,
) -> pd.DataFrame:
    """Return ``dmLineResidualMetrics``: one row per reference, species, ``statusType`` and metric.

    Each has rows for ``nLines``, ``linesPerFiber`` and the statistics of the
    components the species measures (x for trace centroids, x and y for lamp
    lines), NaN when it has too few lines. Only `JUDGED` rows carry a verdict.

    Parameters
    ----------
    species : `pandas.DataFrame`
        `speciesResiduals`.
    judged : `pandas.DataFrame`
        From `residualVerdict`.
    dataId : mapping
        ``visit``, ``arm``, ``spectrograph``.
    camera : `str`, optional
        `cameraSet`.
    obsType : `str`, optional
        `observationType`.

    Returns
    -------
    `pandas.DataFrame`
        `RESIDUAL_COLUMNS`; empty, with the same columns, when there are no species.
    """
    status = {
        (ref, d, t, m): s
        for ref, d, t, m, s in zip(
            judged["reference"],
            judged["description"],
            judged["statusType"],
            judged["metric"],
            judged["status"],
            strict=True,
        )
    }
    rows = []
    for record in species.to_dict("records"):
        reference, description = record["reference"], record["description"]
        components = "x" if description == TRACE else "xy"
        metrics = [*_COUNT_METRICS, *(m for m, c in RESIDUAL_METRICS.items() if c in components)]
        for metric in metrics:
            rows.append(
                {
                    "visit": dataId["visit"],
                    "arm": dataId["arm"],
                    "spectrograph": dataId["spectrograph"],
                    "cameraSet": camera,
                    "obsType": obsType,
                    "reference": reference,
                    "description": description,
                    "statusType": record["statusType"],
                    "metric": metric,
                    "value": float(record[metric]),
                    "status": status.get((reference, description, record["statusType"], metric), ""),
                }
            )
    return pd.DataFrame(rows, columns=list(RESIDUAL_COLUMNS)).astype(RESIDUAL_COLUMNS)
