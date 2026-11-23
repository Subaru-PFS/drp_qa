"""Per-fiber slit offsets of a detectorMap, and the fibers whose offset is an outlier.

A detectorMap carries a spatial and a spectral slit offset per fiber
(``getSpatialOffsets``, ``getSpectralOffsets``), in pixels, aligned to its
``fiberId``. They are measured in ``drp_stella.fitDetectorMap.measureSlitOffsets``
as the per-fiber median of ``(line - model) / scale``: a fiber with a bad PSF,
whose line centroids are biased, gets a biased offset. The offsets are carved
into ``detectorMap_calib`` when calibrations are built (``doSlitOffsets=True``,
after a bootstrap) and, in normal per-visit reduction (``doSlitOffsets=False``),
left unchanged for the visit. So a bad offset is a property of the calibration,
and the signal is the offset being an outlier among its slit neighbours, not a
line residual: the fiber's lines line up with its own (wrong) map.

This module is the stack-free measurement. Two jobs:

- **Outliers within a calibration** (`slitOffsetResiduals`, `flagOutliers`,
  `offsetSummary`): detrend the smooth along-slit variation within each slit
  block (MTP ferrule) by subtracting the block's median, then flag the fibers
  whose residual is large against the pooled robust scatter. These are the
  bad-slit-offset candidates.
- **Invariance of the visit against the calibration** (`offsetInvariance`): in
  normal reduction the visit's offsets equal the calibration's; a non-zero
  difference means ``doSlitOffsets`` was on, or the visit was reduced against a
  different calibration than assumed.

The Butler side -- reading the detectorMap's offset arrays and the ``pfs_utils``
MTP mapping -- is elsewhere (the calibration-comparison and ``dmLineResiduals``
layers). Nothing here imports the LSST stack.
"""

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CHANGE_FIELDS",
    "CONFIRMED_COLUMNS",
    "INVARIANCE_FIELDS",
    "OFFSET_COLUMNS",
    "SUMMARY_FIELDS",
    "changeSummary",
    "compareRuns",
    "crossArmConfirmed",
    "flagOutliers",
    "invarianceRows",
    "offsetInvariance",
    "offsetSummary",
    "offsetTrajectory",
    "robustRms",
    "slitOffsetResiduals",
]


#: Columns of the per-fiber slit-offset table, in order, with their dtypes: one
#: row per fiber of a detector, its two offsets, their residual from the slit
#: block's median, and whether it is an outlier.
OFFSET_COLUMNS = {
    "fiberId": "int32",
    "group": "string",
    "spatial": "float64",
    "spectral": "float64",
    "spatialResid": "float64",
    "spectralResid": "float64",
    "selected": "boolean",
    "spatialOutlier": "boolean",
    "spectralOutlier": "boolean",
    "outlier": "boolean",
}

#: Keys of the per-detector rollup `offsetSummary` returns.
SUMMARY_FIELDS = (
    "nFibers",
    "nSelected",
    "nSlitOffsetOutliers",
    "worstFiberId",
    "maxSpatialResid",
    "maxSpectralResid",
    "spatialScale",
    "spectralScale",
)

#: Keys of the per-detector rollup `offsetInvariance` returns.
INVARIANCE_FIELDS = (
    "nFibers",
    "nSlitOffsetChanged",
    "maxSpatialOffsetDelta",
    "maxSpectralOffsetDelta",
    "worstFiberId",
)

#: Columns of the cross-arm confirmed-outlier table `crossArmConfirmed` returns;
#: `compareRuns` adds ``status``.
CONFIRMED_COLUMNS = {
    "spectrograph": "int64",
    "fiberId": "int32",
    "nArms": "int64",
    "arms": "string",
    "maxSpatialResid": "float64",
    "maxSpectralResid": "float64",
}

#: Keys of the run-to-run rollup `changeSummary` returns.
CHANGE_FIELDS = ("nConfirmed", "nNew", "nCleared", "nPersisted")


def robustRms(values: Any) -> float:
    """Return the robust RMS of the finite values: 0.741 times the interquartile range.

    The definition of ``pfs.drp.stella.utils.math.robustRms``, which this
    stack-free module cannot import (rule R5); shared with
    `pfs.drp.qa.metrics.lineResiduals.robustRms`.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    lower, upper = np.percentile(values, (25.0, 75.0))
    return float(0.741 * (upper - lower))


def slitOffsetResiduals(
    fiberId: Any,
    spatial: Any,
    spectral: Any,
    group: Any,
    *,
    select: Any = None,
) -> pd.DataFrame:
    """Return each fiber's slit offsets and their residual from its slit block.

    The smooth variation of the offset along the slit is removed by subtracting,
    within each ``group`` (an MTP ferrule or other slit block), the median offset
    of the selected fibers in that group. What is left -- the residual -- is flat
    and centred on zero for a healthy block, so a bad fiber stands out.

    Parameters
    ----------
    fiberId : array_like of `int`
        The detector's fibers (``detectorMap.fiberId``).
    spatial, spectral : array_like of `float`
        The spatial and spectral slit offsets, in pixels, aligned to ``fiberId``
        (``detectorMap.getSpatialOffsets()`` / ``getSpectralOffsets()``).
    group : array_like
        A slit-block label per fiber (e.g. the MTP ferrule id). Fibers are
        detrended within their group; a group is never pooled with another.
    select : array_like of `bool`, optional
        Which fibers to trust for the group median and the outlier scale (e.g.
        the ``GOOD`` science fibers). Default: every fiber with finite offsets.
        A residual is computed for every fiber regardless, so an unselected
        fiber is placed but does not set the trend.

    Returns
    -------
    `pandas.DataFrame`
        One row per fiber (`OFFSET_COLUMNS` without the outlier columns, which
        `flagOutliers` adds): ``fiberId``, ``group``, ``spatial``, ``spectral``,
        ``spatialResid``, ``spectralResid``, ``selected``.
    """
    fiberId = np.asarray(fiberId, dtype=np.int32)
    spatial = np.asarray(spatial, dtype=float)
    spectral = np.asarray(spectral, dtype=float)
    group = np.asarray(group).astype(str)
    if not (len(fiberId) == len(spatial) == len(spectral) == len(group)):
        raise ValueError("fiberId, spatial, spectral and group must be the same length")

    finite = np.isfinite(spatial) & np.isfinite(spectral)
    selected = finite if select is None else (np.asarray(select, dtype=bool) & finite)

    frame = pd.DataFrame(
        {
            "fiberId": fiberId,
            "group": group,
            "spatial": spatial,
            "spectral": spectral,
            "selected": selected,
        }
    )

    # The slit-block level is the median of the selected fibers; a group with no
    # selected fiber falls back to all its fibers, so its members are still placed.
    def _level(column: str) -> pd.Series:
        trusted = frame[frame["selected"]]
        level = trusted.groupby("group")[column].median()
        missing = frame["group"][~frame["group"].isin(level.index)].unique()
        if len(missing):
            fallback = frame[frame["group"].isin(missing)].groupby("group")[column].median()
            level = pd.concat([level, fallback])
        return frame["group"].map(level)

    frame["spatialResid"] = frame["spatial"] - _level("spatial")
    frame["spectralResid"] = frame["spectral"] - _level("spectral")
    return frame[["fiberId", "group", "spatial", "spectral", "spatialResid", "spectralResid", "selected"]]


def flagOutliers(
    residuals: pd.DataFrame,
    *,
    threshold: float = 5.0,
    minScale: float = 0.02,
) -> pd.DataFrame:
    """Flag the fibers whose slit-offset residual is an outlier.

    The scale is the pooled robust RMS of the selected fibers' residuals -- the
    residuals are already block-centred, so pooling across blocks is valid --
    floored at ``minScale`` so that an unusually quiet detector does not flag
    pixel-level noise. A fiber is an outlier when its residual exceeds
    ``threshold`` scales in spatial or in spectral.

    Parameters
    ----------
    residuals : `pandas.DataFrame`
        The output of `slitOffsetResiduals`.
    threshold : `float`, optional
        How many robust scales a residual must exceed to be an outlier. A per-fiber
        cut, not a gate limit; the gate's WARN/FAIL on the outlier *count* lives in
        `pfs.drp.qa.metrics.registry` and is set from measured spreads.
    minScale : `float`, optional
        Floor on the robust scale, in pixels, so a quiet detector does not flag
        noise as an outlier.

    Returns
    -------
    `pandas.DataFrame`
        ``residuals`` with ``spatialOutlier``, ``spectralOutlier`` and ``outlier``
        added, and ``attrs['spatialScale']`` / ``attrs['spectralScale']`` set.
    """
    frame = residuals.copy()
    trusted = frame[frame["selected"]]
    spatialScale = max(robustRms(trusted["spatialResid"]), minScale)
    spectralScale = max(robustRms(trusted["spectralResid"]), minScale)

    frame["spatialOutlier"] = (frame["spatialResid"].abs() > threshold * spatialScale).fillna(False)
    frame["spectralOutlier"] = (frame["spectralResid"].abs() > threshold * spectralScale).fillna(False)
    # An unselected fiber is placed but not judged: it never raises an alarm on its own.
    frame.loc[~frame["selected"], ["spatialOutlier", "spectralOutlier"]] = False
    frame["outlier"] = frame["spatialOutlier"] | frame["spectralOutlier"]
    frame.attrs["spatialScale"] = spatialScale
    frame.attrs["spectralScale"] = spectralScale
    return frame


def offsetSummary(flagged: pd.DataFrame) -> dict[str, Any]:
    """Return the per-detector rollup of `flagOutliers` (`SUMMARY_FIELDS`).

    ``nSlitOffsetOutliers`` is the gated number; ``worstFiberId`` is the fiber
    with the largest residual (in units of the robust scale), for the report.
    """
    spatialScale = flagged.attrs.get("spatialScale", np.nan)
    spectralScale = flagged.attrs.get("spectralScale", np.nan)
    selected = flagged[flagged["selected"]]

    worstFiberId: int | None = None
    if len(selected):
        # Rank by the worse of the two residuals, each in units of its own scale.
        spatialZ = selected["spatialResid"].abs() / spatialScale
        spectralZ = selected["spectralResid"].abs() / spectralScale
        worst = np.fmax(spatialZ.to_numpy(), spectralZ.to_numpy())
        if np.any(np.isfinite(worst)):
            worstFiberId = int(selected["fiberId"].to_numpy()[np.nanargmax(worst)])

    return {
        "nFibers": len(flagged),
        "nSelected": int(flagged["selected"].sum()),
        "nSlitOffsetOutliers": int(flagged["outlier"].sum()),
        "worstFiberId": worstFiberId,
        "maxSpatialResid": float(selected["spatialResid"].abs().max()) if len(selected) else np.nan,
        "maxSpectralResid": float(selected["spectralResid"].abs().max()) if len(selected) else np.nan,
        "spatialScale": float(spatialScale),
        "spectralScale": float(spectralScale),
    }


def offsetInvariance(
    fiberId: Any,
    spatial: Any,
    spectral: Any,
    spatialRef: Any,
    spectralRef: Any,
    *,
    select: Any = None,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Return how far a detectorMap's slit offsets have moved from a reference's.

    In normal reduction ``adjustDetectorMap`` runs with ``doSlitOffsets=False``,
    so the visit's offsets equal the calibration's. A non-zero difference means
    the offsets were refit per visit, or the visit was reduced against a
    different calibration than the one compared here.

    Parameters
    ----------
    fiberId : array_like of `int`
        The detector's fibers; the two offset pairs are aligned to it.
    spatial, spectral : array_like of `float`
        The detectorMap's offsets (the visit's).
    spatialRef, spectralRef : array_like of `float`
        The reference's offsets (the calibration's).
    select : array_like of `bool`, optional
        Which fibers to compare. Default: fibers finite in both.
    tol : `float`, optional
        A difference at or below this, in pixels, counts as unchanged.

    Returns
    -------
    `dict`
        `INVARIANCE_FIELDS`: ``nFibers``, ``nSlitOffsetChanged``,
        ``maxSpatialOffsetDelta``, ``maxSpectralOffsetDelta``, ``worstFiberId``.
    """
    fiberId = np.asarray(fiberId, dtype=np.int32)
    dSpatial = np.asarray(spatial, dtype=float) - np.asarray(spatialRef, dtype=float)
    dSpectral = np.asarray(spectral, dtype=float) - np.asarray(spectralRef, dtype=float)

    finite = np.isfinite(dSpatial) & np.isfinite(dSpectral)
    use = finite if select is None else (np.asarray(select, dtype=bool) & finite)

    if not np.any(use):
        return {
            "nFibers": 0,
            "nSlitOffsetChanged": 0,
            "maxSpatialOffsetDelta": np.nan,
            "maxSpectralOffsetDelta": np.nan,
            "worstFiberId": None,
        }

    absSpatial = np.abs(dSpatial[use])
    absSpectral = np.abs(dSpectral[use])
    changed = (absSpatial > tol) | (absSpectral > tol)
    # The worst fiber is only meaningful once something has moved.
    worst = np.fmax(absSpatial, absSpectral)
    worstFiberId = int(fiberId[use][int(np.nanargmax(worst))]) if np.any(changed) else None

    return {
        "nFibers": int(use.sum()),
        "nSlitOffsetChanged": int(changed.sum()),
        "maxSpatialOffsetDelta": float(absSpatial.max()),
        "maxSpectralOffsetDelta": float(absSpectral.max()),
        "worstFiberId": worstFiberId,
    }


def crossArmConfirmed(flagged: pd.DataFrame, *, minArms: int = 2) -> pd.DataFrame:
    """Return the fibers that are slit-offset outliers in several arms of one spectrograph.

    A single-arm outlier is often marginal, but a fault at or before the slit (a bad
    PSF, a displaced fiber) biases the offset in every arm, so a fiber flagged in two or
    more arms of its spectrograph is a robust bad-fiber signal. This is the headline of
    the slit-offset QA: it rediscovered the documented bad fibers (SM3 1358/1373/1415,
    SM2 940/941) with no single-detector false positives.

    Parameters
    ----------
    flagged : `pandas.DataFrame`
        `flagOutliers` output stacked over a run's detectors, with ``spectrograph`` and
        ``arm`` columns added. Only ``selected`` outliers count.
    minArms : `int`, optional
        Fewest arms of a spectrograph a fiber must be an outlier in to be confirmed.

    Returns
    -------
    `pandas.DataFrame`
        One row per confirmed (spectrograph, fiberId), sorted, with `CONFIRMED_COLUMNS`:
        ``nArms`` (how many arms flagged it), ``arms`` (e.g. ``"b,m,n,r"``) and the
        largest absolute residual over those arms in each direction.
    """
    hits = flagged[flagged["outlier"] & flagged["selected"]]
    if hits.empty:
        return pd.DataFrame({name: pd.Series(dtype=dt) for name, dt in CONFIRMED_COLUMNS.items()})

    rows = []
    for (spec, fiberId), group in hits.groupby(["spectrograph", "fiberId"], sort=True):
        arms = sorted(str(a) for a in group["arm"].unique())
        if len(arms) < minArms:
            continue
        rows.append(
            {
                "spectrograph": int(spec),
                "fiberId": int(fiberId),
                "nArms": len(arms),
                "arms": ",".join(arms),
                "maxSpatialResid": float(group["spatialResid"].abs().max()),
                "maxSpectralResid": float(group["spectralResid"].abs().max()),
            }
        )
    confirmed = pd.DataFrame(rows, columns=list(CONFIRMED_COLUMNS))
    return confirmed.astype(CONFIRMED_COLUMNS)


def compareRuns(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Return the status of this run's confirmed outliers against the previous run's.

    The slit offsets are physical and should persist across runs: a fiber that is a
    confirmed outlier in this run but not the previous is ``new`` -- the slit, cabling or
    the fiber changed, a regression worth a verdict. A fiber in the previous run but not
    this one is ``cleared``; one in both is ``persisted`` (a chronic bad fiber, not a
    regression). Keyed on (spectrograph, fiberId).

    Parameters
    ----------
    current, previous : `pandas.DataFrame`
        `crossArmConfirmed` outputs for the two runs.

    Returns
    -------
    `pandas.DataFrame`
        The union of both runs' confirmed fibers, with `CONFIRMED_COLUMNS` (from the run
        that has the fiber; the current run's values when both do) and a ``status`` of
        ``new``, ``cleared`` or ``persisted``, sorted by (spectrograph, fiberId).
    """
    key = ["spectrograph", "fiberId"]
    cur = {(int(r.spectrograph), int(r.fiberId)) for r in current.itertuples()}
    prev = {(int(r.spectrograph), int(r.fiberId)) for r in previous.itertuples()}

    def status(spec: int, fiberId: int) -> str:
        here, there = (spec, fiberId) in cur, (spec, fiberId) in prev
        return "persisted" if (here and there) else ("new" if here else "cleared")

    merged = pd.concat([current, previous[~previous.set_index(key).index.isin(current.set_index(key).index)]])
    merged = merged.sort_values(key).reset_index(drop=True)
    merged["status"] = [status(int(r.spectrograph), int(r.fiberId)) for r in merged.itertuples()]
    return merged


def changeSummary(comparison: pd.DataFrame) -> dict[str, int]:
    """Return the run-to-run rollup of `compareRuns` (`CHANGE_FIELDS`).

    ``nNew`` is the gated number: a new confirmed outlier is a regression.
    """
    status = comparison["status"] if "status" in comparison else pd.Series(dtype="string")
    return {
        "nConfirmed": int((status != "cleared").sum()),
        "nNew": int((status == "new").sum()),
        "nCleared": int((status == "cleared").sum()),
        "nPersisted": int((status == "persisted").sum()),
    }


#: The metrics `invarianceRows` writes, in the order a check takes the worst.
_INVARIANCE_METRICS = ("maxSpatialOffsetDelta", "maxSpectralOffsetDelta", "nSlitOffsetChanged")


def invarianceRows(
    dataId: Any,
    result: dict[str, Any],
    *,
    camera: str | None = None,
    obsType: str | None = None,
    seqType: str | None = None,
) -> pd.DataFrame:
    """Return `offsetInvariance`'s numbers as ``dmLineResidualMetrics`` rows.

    The rows carry ``reference`` ``"slit"``, ``description`` ``"SlitOffset"`` and
    ``statusType`` ``"ALL"``, so they ride the visit's line-residual metrics table into
    qadb and `pfs.drp.qa.metrics.gate` judges them (the ``slitOffsetInvariance`` check).
    ``nSlitOffsetChanged`` rides along for context and is not judged.

    Parameters
    ----------
    dataId : mapping
        ``visit``, ``arm``, ``spectrograph``.
    result : `dict`
        `offsetInvariance`'s return (`INVARIANCE_FIELDS`).
    camera, obsType, seqType : `str`, optional
        The frame's `cameraSet`, observation type and sequence type, as the other
        rows carry them.

    Returns
    -------
    `pandas.DataFrame`
        `pfs.drp.qa.metrics.lineResiduals.RESIDUAL_COLUMNS`, one row per metric.
    """
    from pfs.drp.qa.metrics.lineResiduals import RESIDUAL_COLUMNS

    rows = [
        {
            "visit": dataId["visit"],
            "arm": dataId["arm"],
            "spectrograph": dataId["spectrograph"],
            "cameraSet": camera,
            "obsType": obsType,
            "seqType": seqType,
            "reference": "slit",
            "description": "SlitOffset",
            "statusType": "ALL",
            "metric": metric,
            "value": float(result[metric]) if result.get(metric) is not None else float("nan"),
        }
        for metric in _INVARIANCE_METRICS
    ]
    return pd.DataFrame(rows, columns=list(RESIDUAL_COLUMNS)).astype(RESIDUAL_COLUMNS)


def offsetTrajectory(confirmedByRun: "dict[str, pd.DataFrame]") -> pd.DataFrame:
    """Return a fiber-by-run matrix of cross-arm confirmation across runs.

    A slit offset is re-fit whenever a ``detectorMap_calib`` is built, so one run's confirmed
    outlier can clear in the next from a different build, not only a physical change. Laid out
    across runs, a fiber that is confirmed in every run is chronic, one that steps on and stays
    changed, and one that flickers is build-to-build fit wobble. This is the interpretive view;
    the ``new`` count against the previous run (`compareRuns`) is the alert.

    Parameters
    ----------
    confirmedByRun : mapping of `str` to `pandas.DataFrame`
        `crossArmConfirmed` output per run, by run label, **in run order** (the column order is
        the mapping's order).

    Returns
    -------
    `pandas.DataFrame`
        One row per (``spectrograph``, ``fiberId``) confirmed in at least one run, sorted, then one
        integer column per run: the arms it was confirmed in that run (0 when it was not), so a
        healthy fiber is a row of zeros and never appears.
    """
    runs = list(confirmedByRun)
    perRun = {
        run: {(int(r.spectrograph), int(r.fiberId)): int(r.nArms) for r in frame.itertuples()}
        for run, frame in confirmedByRun.items()
    }
    keys = sorted({key for counts in perRun.values() for key in counts})
    rows = []
    for spectrograph, fiberId in keys:
        row = {"spectrograph": spectrograph, "fiberId": fiberId}
        row |= {run: perRun[run].get((spectrograph, fiberId), 0) for run in runs}
        rows.append(row)
    columns = ["spectrograph", "fiberId", *runs]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)
