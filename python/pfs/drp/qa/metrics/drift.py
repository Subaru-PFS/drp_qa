"""Detector-map drift against the calibration: measurement and verdict.

``dmDriftMonitor`` asks whether the instrument has moved away from the
calibration a visit is reduced against. The reference is ``detectorMap_calib``
(rule R1 of ``docs/qa-principles.md``), never the visit's own ``detectorMap``:
``fitDetectorMap`` fitted that to these very lines, so comparing the two
measures fit residuals, not drift.

Per line species (``description``, R7), over the good rows of ``lines`` (no
``BAD`` status bit, as ``fitDetectorMap``'s ``lineFlags``) that the visit's
detectorMap fit marked ``DETECTORMAP_USED`` (fitted) or
``DETECTORMAP_RESERVED`` (held back to test it), as ``dmResiduals`` selects:

- trace centroids (``description == "Trace"``) measure x: their row is fixed,
  so ``dx = x - calib.getXCenter(fiberId, y)``;
- lamp lines measure y: ``dy = y - calib.findPoint(fiberId, wavelength)[:, 1]``.

The offsets are measured minus calibration, the sign of drp_stella's
residuals. Each species gives the median and robust RMS of its component,
``nLines``, and ``linesPerFiber`` (per lit fiber, so a design lighting one fiber group is not
read as sparse, PIPE2D-1935). A species with fewer than ``minLines`` good lines
gets NaN statistics: not measured. The components are gated, never a combined
magnitude: trace lines constrain x only, lamp lines y only.

Each species is measured three ways (``statusType``): over ``USED`` and
``RESERVED`` lines together (``BOTH``), which is judged, and over each alone.
Neither set was fitted to the calibration, so against it both measure the
drift; a difference between them is a selection effect of the visit's fit, kept
as a diagnostic, as ``dmResiduals`` keeps the two apart.

Stack-free: the calibration is duck-typed (``fiberId``, ``getXCenter``,
``findPoint``), so it is tested with a fake.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pfs.drp.qa.metrics.gate import STATUS_ORDER, judge

__all__ = [
    "BAD_STATUS",
    "BOTH",
    "DRIFT_COLUMNS",
    "DRIFT_METRICS",
    "RECOMMENDED_ACTIONS",
    "RESERVED",
    "STATUS_TYPES",
    "SUMMARY_COLUMNS",
    "TRACE",
    "USED",
    "cameraSet",
    "driftOffsets",
    "driftThresholds",
    "driftVerdict",
    "longDrift",
    "robustRms",
    "speciesDrift",
]

#: The ``description`` of trace centroids in ``lines``.
TRACE = "Trace"

#: drp_stella's ``ReferenceLineStatus.BAD`` (NOT_VISIBLE, BLEND, SUSPECT,
#: REJECTED, BROAD): the status bits ``fitDetectorMap`` drops by default.
BAD_STATUS = 0x1F
#: drp_stella's ``ReferenceLineStatus.DETECTORMAP_USED`` and ``DETECTORMAP_RESERVED``.
_USED_BIT, _RESERVED_BIT = 0x20, 0x40

#: Values of ``statusType``: lines the visit's detectorMap fit used, those it
#: held back, and both together, the set that is judged.
USED, RESERVED, BOTH = "USED", "RESERVED", "BOTH"
STATUS_TYPES = (BOTH, USED, RESERVED)

#: The drift statistics, with the component each comes from.
DRIFT_METRICS = {
    "medianDeltaX": "x",
    "robustRmsDeltaX": "x",
    "medianDeltaY": "y",
    "robustRmsDeltaY": "y",
}

#: The counts reported for every species.
_COUNT_METRICS = ("nLines", "linesPerFiber")

#: The operator's action for each detector verdict. `driftVerdict` also gives
#: ``UNJUDGED``: lines were measured, but no threshold applies (thresholds disabled).
RECOMMENDED_ACTIONS = {
    "PASS": "NOMINAL",
    "WARN": "APPLY_SHIFT",
    "FAIL": "RECALIBRATE",
    "UNKNOWN": "INSUFFICIENT_DATA",
}

#: Columns of ``dmDriftMetrics``, in order, with their dtypes: one row per
#: detector, species, ``statusType`` and metric, with the metric's verdict
#: (empty when not judged, as every ``USED`` and ``RESERVED`` row is).
DRIFT_COLUMNS = {
    "visit": "int64",
    "arm": "string",
    "spectrograph": "int64",
    "cameraSet": "string",
    "description": "string",
    "statusType": "string",
    "metric": "string",
    "value": "float64",
    "status": "string",
}

#: Columns of ``dmDriftSummary`` set by `driftVerdict`.
SUMMARY_COLUMNS = (
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


def driftOffsets(lines: pd.DataFrame, calib: Any, badStatus: int = BAD_STATUS) -> pd.DataFrame:
    """Return each good line's offset from the calibration.

    Parameters
    ----------
    lines : `pandas.DataFrame`
        ``lines`` (``ArcLineSet.data``): ``fiberId``, ``wavelength``, ``x``,
        ``y``, ``flag``, ``status``, ``description``.
    calib : `pfs.drp.stella.DetectorMap`
        ``detectorMap_calib``; duck-typed: ``fiberId``, vectorised
        ``getXCenter(fiberId, y)`` and ``findPoint(fiberId, wavelength)``
        (an ``(N, 2)`` array).
    badStatus : `int`, optional
        Status bits that exclude a line, as ``fitDetectorMap``'s ``lineFlags``.

    Returns
    -------
    `pandas.DataFrame`
        One row per good line (unflagged, no ``badStatus`` bit, finite
        position, on a fiber of the calibration, and a finite calibration
        prediction: ``findPoint`` gives NaN off the detector): ``fiberId``,
        ``description``, ``statusType`` (``USED``, ``RESERVED``, or empty
        when the visit's fit did neither: such a line is not measured),
        ``dx`` (trace centroids; NaN for lamp lines) and ``dy`` (lamp
        lines; NaN for trace centroids), measured minus calibration.
    """
    fiberId = lines["fiberId"].to_numpy(dtype=np.int32)
    x, y = lines["x"].to_numpy(dtype=float), lines["y"].to_numpy(dtype=float)
    description = lines["description"].astype(str).to_numpy()
    good = ~lines["flag"].to_numpy(dtype=bool)
    status = (
        lines["status"].to_numpy(dtype=np.int64) if "status" in lines.columns else np.zeros(len(lines), int)
    )
    good &= (status & badStatus) == 0
    good &= np.isfinite(x) & np.isfinite(y)
    # getXCenter raises on a fiber a splined detectorMap lacks.
    good &= np.isin(fiberId, np.asarray(calib.fiberId))
    isTrace = description == TRACE
    wavelength = lines["wavelength"].to_numpy(dtype=float) if "wavelength" in lines.columns else None
    if wavelength is None:
        good &= isTrace
    else:
        good &= isTrace | np.isfinite(wavelength)

    dx = np.full(len(lines), np.nan)
    dy = np.full(len(lines), np.nan)
    traces = good & isTrace
    if traces.any():
        dx[traces] = x[traces] - np.asarray(
            calib.getXCenter(np.ascontiguousarray(fiberId[traces]), np.ascontiguousarray(y[traces])),
            dtype=float,
        )
    lamps = good & ~isTrace
    if lamps.any():
        points = np.asarray(
            calib.findPoint(np.ascontiguousarray(fiberId[lamps]), np.ascontiguousarray(wavelength[lamps])),
            dtype=float,
        )
        dy[lamps] = y[lamps] - points[:, 1]
    # A NaN prediction would make its species' median NaN while still counting as a line.
    good &= np.where(isTrace, np.isfinite(dx), np.isfinite(dy))
    # USED wins over RESERVED, as in dmResiduals; a fit never sets both.
    statusType = np.where(status & _USED_BIT, USED, np.where(status & _RESERVED_BIT, RESERVED, ""))
    return pd.DataFrame(
        {
            "fiberId": fiberId[good],
            "description": description[good],
            "statusType": statusType[good],
            "dx": dx[good],
            "dy": dy[good],
        }
    )


def speciesDrift(offsets: pd.DataFrame, minLines: int, litFibers: set[int] | None = None) -> pd.DataFrame:
    """Return the drift statistics of each species, over each `STATUS_TYPES` set.

    Parameters
    ----------
    offsets : `pandas.DataFrame`
        `driftOffsets`; lines with no ``statusType`` are left out.
    minLines : `int`
        Fewest lines a species needs, in each set, for its statistics; below
        it they are NaN (not measured).
    litFibers : `set` [`int`], optional
        The fibers the pfsConfig lights (``imageQualityQa.litFibers``);
        ``linesPerFiber`` is NaN without them, or when none is lit.

    Returns
    -------
    `pandas.DataFrame`
        One row per species and ``statusType`` (`STATUS_TYPES`, in that
        order), species sorted: ``description``, ``statusType``, ``nLines``,
        ``linesPerFiber`` (lines on lit fibers, per lit fiber), and the
        `DRIFT_METRICS`, NaN for the component the species does not measure.
        A species is listed when it has a ``USED`` or ``RESERVED`` line.
    """
    rows = []
    fitted = offsets[offsets["statusType"].isin((USED, RESERVED))]
    for (description, statusType), group in _statusSets(fitted):
        nLines = len(group)
        row = {
            "description": description,
            "statusType": statusType,
            "nLines": nLines,
            "linesPerFiber": np.nan,
        }
        if litFibers:
            row["linesPerFiber"] = float(group["fiberId"].isin(litFibers).sum()) / len(litFibers)
        for metric, component in DRIFT_METRICS.items():
            measures = (component == "x") == (description == TRACE)
            values = group[f"d{component}"]
            if not measures or nLines < minLines:
                row[metric] = np.nan
            elif metric.startswith("median"):
                row[metric] = float(np.median(values))
            else:
                row[metric] = robustRms(values)
        rows.append(row)
    columns = ["description", "statusType", *_COUNT_METRICS, *DRIFT_METRICS]
    return pd.DataFrame(rows, columns=columns).astype({"nLines": "int64"})


def _statusSets(fitted: pd.DataFrame):
    """Yield ``((description, statusType), lines)`` for each species and `STATUS_TYPES` set."""
    for description, group in fitted.groupby("description", sort=True):
        yield (description, BOTH), group
        for statusType in (USED, RESERVED):
            yield (description, statusType), group[group["statusType"] == statusType]


def driftThresholds(
    xWarn: float, xFail: float, yWarn: float, yFail: float, source: str = "dmDriftMonitor config"
) -> pd.DataFrame:
    """Return the config's thresholds on the median offsets as a `gate` table.

    One entry each for ``medianDeltaX`` and ``medianDeltaY``, judged on their
    absolute value, with no population: they match every detector and
    species, beneath any thresholds file. A NaN threshold is not applied; a
    metric with both NaN is not judged.
    """
    rows = []
    for metric, warn, fail, name in (
        ("medianDeltaX", xWarn, xFail, "xWarn/FailThreshold"),
        ("medianDeltaY", yWarn, yFail, "yWarn/FailThreshold"),
    ):
        rows.append(
            {
                "metric": metric,
                "warn": float(warn),
                "fail": float(fail),
                "higherIsWorse": True,
                "absolute": True,
                "provenance": f"{source} {name}.",
            }
        )
    return pd.DataFrame(rows)


def driftVerdict(
    species: pd.DataFrame,
    dataId: Mapping[str, Any],
    thresholds: pd.DataFrame | Sequence[pd.DataFrame],
    minLines: int,
    camera: str | None = None,
    numUnfitted: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Judge each species' drift, over its ``BOTH`` lines, and the detector.

    Parameters
    ----------
    species : `pandas.DataFrame`
        `speciesDrift`.
    dataId : mapping
        ``visit``, ``arm``, ``spectrograph``.
    thresholds : `pandas.DataFrame` or a sequence of them
        Threshold tables, highest priority first (`pfs.drp.qa.metrics.gate`).
        Population columns may name ``arm``, ``spectrograph``,
        ``description`` and ``cameraSet``.
    minLines : `int`
        As given to `speciesDrift`; for the reason string.
    camera : `str`, optional
        `cameraSet`.
    numUnfitted : `int`, optional
        Good lines the visit's fit neither used nor reserved; for the reason
        string when no species could be measured.

    Returns
    -------
    judged : `pandas.DataFrame`
        `pfs.drp.qa.metrics.gate.judge` of each species' ``BOTH`` row, with
        ``description`` and ``statusType``.
    summary : `dict`
        `SUMMARY_COLUMNS`. ``qaStatus`` is the worst species verdict, or
        ``UNKNOWN`` when no species has ``minLines`` good lines: not
        measurable, never measured but not combinable. ``qaDecidedBy`` is
        ``"<description> <metric>"``.
    """
    species = species[species["statusType"] == BOTH]
    wide = species.assign(
        visit=dataId["visit"],
        arm=dataId["arm"],
        spectrograph=dataId["spectrograph"],
        cameraSet=camera,
    ).reset_index(drop=True)
    judged = judge(wide, thresholds)
    rows = judged["row"].to_numpy(dtype=int)
    judged["description"] = wide["description"].to_numpy()[rows]
    judged["statusType"] = wide["statusType"].to_numpy()[rows]
    measured = int((species["nLines"] >= minLines).sum())
    summary: dict[str, Any] = {
        "nSpecies": len(species),
        "nSpeciesMeasured": measured,
        "qaStatus": "UNKNOWN",
        "qaDecidedBy": "",
        "qaReason": "",
        "recommendedAction": RECOMMENDED_ACTIONS["UNKNOWN"],
    }
    if measured == 0:
        if len(species):
            summary["qaReason"] = f"fewer than {minLines} good lines in every species"
        elif numUnfitted:
            summary["qaReason"] = (
                f"no line marked DETECTORMAP_USED or DETECTORMAP_RESERVED ({numUnfitted} good lines):"
                " the visit's detectorMap fit did not run or failed"
            )
        else:
            summary["qaReason"] = "no good lines"
        return judged, summary
    verdicts = judged[judged["status"] != ""]
    if verdicts.empty:
        summary["qaReason"] = "lines measured, but no threshold applies"
        summary["recommendedAction"] = "UNJUDGED"
        return judged, summary
    rank = verdicts["status"].map(STATUS_ORDER.index)
    worst = STATUS_ORDER[int(rank.max())]
    summary["qaStatus"] = worst
    summary["recommendedAction"] = RECOMMENDED_ACTIONS[worst]
    if worst != "PASS":
        decider = verdicts.loc[rank.idxmax()]
        summary["qaDecidedBy"] = f"{decider['description']} {decider['metric']}"
        crossed = verdicts[verdicts["reason"] != ""]
        summary["qaReason"] = "; ".join(
            f"{d}: {r}" for d, r in zip(crossed["description"], crossed["reason"], strict=True)
        )
    return judged, summary


def longDrift(
    species: pd.DataFrame, judged: pd.DataFrame, dataId: Mapping[str, Any], camera: str | None = None
) -> pd.DataFrame:
    """Return ``dmDriftMetrics``: one row per species, ``statusType`` and metric, with its verdict.

    Each species and ``statusType`` has rows for ``nLines``, ``linesPerFiber``
    and the statistics of the component it measures (x for trace centroids, y
    for lamp lines), NaN when it has too few lines; none for the other
    component. Only ``BOTH`` rows carry a verdict.

    Parameters
    ----------
    species : `pandas.DataFrame`
        `speciesDrift`.
    judged : `pandas.DataFrame`
        From `driftVerdict`.
    dataId : mapping
        ``visit``, ``arm``, ``spectrograph``.
    camera : `str`, optional
        `cameraSet`.

    Returns
    -------
    `pandas.DataFrame`
        `DRIFT_COLUMNS`; empty, with the same columns, when there are no species.
    """
    status = {
        (d, t, m): s
        for d, t, m, s in zip(
            judged["description"], judged["statusType"], judged["metric"], judged["status"], strict=True
        )
    }
    rows = []
    for record in species.to_dict("records"):
        description = record["description"]
        component = "x" if description == TRACE else "y"
        metrics = [*_COUNT_METRICS, *(m for m, c in DRIFT_METRICS.items() if c == component)]
        for metric in metrics:
            rows.append(
                {
                    "visit": dataId["visit"],
                    "arm": dataId["arm"],
                    "spectrograph": dataId["spectrograph"],
                    "cameraSet": camera,
                    "description": description,
                    "statusType": record["statusType"],
                    "metric": metric,
                    "value": float(record[metric]),
                    "status": status.get((description, record["statusType"], metric), ""),
                }
            )
    return pd.DataFrame(rows, columns=list(DRIFT_COLUMNS)).astype(DRIFT_COLUMNS)
