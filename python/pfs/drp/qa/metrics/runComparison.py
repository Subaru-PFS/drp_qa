"""Compare a run's metrics with a reference's: each detector against where it was.

A gate verdict judges one image against thresholds. This asks the other
question of a night or a run: has a detector moved since a time it was known
to be good? `compareRuns` takes the metrics of the visits in question and of
one or more references (the previous run's set; the run's own first
calibration night) and returns, per reference, metric, camera and population
(illumination, frame type, lamp), the two medians, their difference, and a
status where the metric declares a tolerance (`MetricSpec.tolerance`: the
fractions the gate's thresholds are derived with).

Stack-free: DataFrames in, a DataFrame out, no database. The tables are the
Butler ones or those `pfs.drp.qa.qadb.reader.readQadb` returns: ``iqQaMetrics``
(a column per metric) or a long table (``metric`` and ``value`` columns:
``dmLineResidualMetrics``).
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from pfs.drp.qa.metrics.calibration import addSpecies
from pfs.drp.qa.metrics.registry import specFor
from pfs.drp.qa.metrics.visitRollup import EXIT_FAIL, EXIT_NO_DATA, EXIT_PASS, EXIT_WARN
from pfs.drp.qa.qadb.tables import IMAGE_KEYS, cameraName, worstStatus

__all__ = [
    "COMPARED_METRICS",
    "COMPARISON_COLUMNS",
    "POPULATION_COLUMNS",
    "compareRuns",
    "exitCode",
    "metricRows",
    "populationMedians",
    "toleranceStatus",
]

#: The metrics compared by default: those of ``iqQaMetrics``, then those of ``dmLineResidualMetrics``.
COMPARED_METRICS = (
    "medFwhm",
    "linesPerFiber",
    "pctFlagged",
    "medDxCenter",
    "medianDeltaX",
    "robustRmsDeltaX",
    "medianDeltaY",
    "robustRmsDeltaY",
)

#: Columns separating the populations compared, besides the camera; one a table lacks is empty.
#: ``illumination`` keeps IIS frames beside PFI ones, never pooled with them.
POPULATION_COLUMNS = ("illumination", "obsType", "species", "reference", "statusType", "description")

_KEYS = ("metric", "arm", "spectrograph", *POPULATION_COLUMNS)

#: Columns of `compareRuns`, in order.
COMPARISON_COLUMNS = (
    "referenceName",
    "metric",
    "camera",
    "arm",
    "spectrograph",
    *POPULATION_COLUMNS,
    "nRun",
    "nReference",
    "runMedian",
    "referenceMedian",
    "difference",
    "relative",
    "warn",
    "fail",
    "status",
)

_ROW_COLUMNS = (*IMAGE_KEYS, *POPULATION_COLUMNS, "metric", "value", "verdict")


def metricRows(table: pd.DataFrame, metrics: Sequence[str] = COMPARED_METRICS) -> pd.DataFrame:
    """Return a metrics table as one row per image, population and metric.

    Parameters
    ----------
    table : `pandas.DataFrame`
        ``iqQaMetrics`` (a column per metric, with ``visit``, ``arm`` and
        ``spectrograph``), or a long table with ``metric`` and ``value``.
    metrics : sequence of `str`
        The metrics wanted; those the table lacks are skipped.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph``, `POPULATION_COLUMNS` (empty
        strings where the table has no such column; ``species`` from
        ``seqName``), ``metric``, ``value`` and ``verdict``: the image's
        ``qaStatus``, or a long table's ``status`` of that metric. Values a
        metric's `MetricSpec.notMeasuredWhen` marks, and non-finite ones, are
        dropped; an ``absolute`` metric's values are made absolute.
    """
    if table.empty:
        return pd.DataFrame(columns=list(_ROW_COLUMNS))
    if {"metric", "value"} <= set(table.columns):
        rows = table[table["metric"].isin(list(metrics))].copy()
        rows["verdict"] = rows.get("status", "")
    else:
        table = addSpecies(table)
        shared = [c for c in (*IMAGE_KEYS, *POPULATION_COLUMNS) if c in table.columns]
        parts = []
        for name in metrics:
            if name not in table.columns:
                continue
            measured = table[~specFor(name).notMeasured(table)]
            part = measured[shared].assign(metric=name, value=measured[name])
            part["verdict"] = measured.get("qaStatus", "")
            parts.append(part)
        if not parts:
            return pd.DataFrame(columns=list(_ROW_COLUMNS))
        rows = pd.concat(parts, ignore_index=True)
    for column in POPULATION_COLUMNS:
        if column not in rows:
            rows[column] = ""
        rows[column] = rows[column].fillna("").astype(str)
    rows["verdict"] = rows["verdict"].fillna("").astype(str)
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce").astype("float64")
    absolute = rows["metric"].map(lambda name: specFor(name).absolute).astype(bool)
    rows.loc[absolute, "value"] = rows.loc[absolute, "value"].abs()
    rows = rows[np.isfinite(rows["value"])]
    rows = rows.astype({"visit": "int64", "spectrograph": "int64"})
    return rows[list(_ROW_COLUMNS)].reset_index(drop=True)


def populationMedians(rows: pd.DataFrame) -> pd.DataFrame:
    """Return the median and count of `metricRows` per metric, camera and population."""
    grouped = rows.groupby(list(_KEYS), sort=True)["value"]
    return grouped.agg(median="median", n="count").reset_index()


def compareRuns(
    run: pd.DataFrame | Sequence[pd.DataFrame],
    references: Mapping[str, pd.DataFrame | Sequence[pd.DataFrame]],
    metrics: Sequence[str] = COMPARED_METRICS,
    knownGood: bool = True,
) -> pd.DataFrame:
    """Compare a run's medians with each reference's, per camera and population.

    Parameters
    ----------
    run : `pandas.DataFrame` or sequence of them
        Metrics tables of the visits in question (see `metricRows`).
    references : mapping of `str` to tables
        The same for each reference, by its name (``"run30"``, ``"firstNight"``).
    metrics : sequence of `str`
        The metrics to compare.
    knownGood : `bool`
        Leave a reference's ``WARN`` and ``FAIL`` values out of its medians;
        the run's values are all used.

    Returns
    -------
    `pandas.DataFrame`
        `COMPARISON_COLUMNS`: one row per reference and population the run has
        values in. ``difference`` is the run's median less the reference's and
        ``relative`` that over the reference's median (missing when it is not
        positive). ``warn`` and ``fail`` are the metric's tolerance fractions.
        ``status`` is ``PASS``, ``WARN`` or ``FAIL`` by ``relative`` in the
        metric's bad direction; ``UNKNOWN`` when the metric has no tolerance,
        the reference no values, or its median is not positive.
    """
    ours = populationMedians(_rows(run, metrics))
    parts = []
    for name, tables in references.items():
        theirs = _rows(tables, metrics)
        if knownGood:
            theirs = theirs[~theirs["verdict"].isin(["WARN", "FAIL"])]
        both = ours.merge(
            populationMedians(theirs), on=list(_KEYS), how="left", suffixes=("Run", "Reference")
        )
        both.insert(0, "referenceName", name)
        parts.append(both)
    if not parts:
        return pd.DataFrame(columns=list(COMPARISON_COLUMNS))
    result = pd.concat(parts, ignore_index=True).rename(
        columns={"medianRun": "runMedian", "medianReference": "referenceMedian"}
    )
    result["nReference"] = result["nReference"].fillna(0).astype("int64")
    result["camera"] = [
        cameraName(arm, spec) for arm, spec in zip(result["arm"], result["spectrograph"], strict=True)
    ]
    result["difference"] = result["runMedian"] - result["referenceMedian"]
    positive = result["referenceMedian"] > 0
    result["relative"] = (result["difference"] / result["referenceMedian"]).where(positive)

    result[["warn", "fail", "status"]] = toleranceStatus(result["metric"], result["relative"])
    result = result.sort_values(["referenceName", "metric", "spectrograph", "arm", *POPULATION_COLUMNS])
    return result[list(COMPARISON_COLUMNS)].reset_index(drop=True)


def toleranceStatus(metric: pd.Series, relative: pd.Series) -> pd.DataFrame:
    """Return the status of relative differences from a reference, by each metric's tolerance.

    Parameters
    ----------
    metric : `pandas.Series`
        The metric each difference is of.
    relative : `pandas.Series`
        The differences, as fractions of the reference; on the same index.

    Returns
    -------
    `pandas.DataFrame`
        ``warn`` and ``fail``, the metric's `MetricSpec.tolerance` fractions
        (missing when it declares none), and ``status``: ``PASS``, ``WARN`` or
        ``FAIL`` by the difference in the metric's bad direction, ``UNKNOWN``
        without a tolerance or a difference.
    """
    limits = [specFor(name).tolerance or (np.nan, np.nan) for name in metric]
    warn = pd.Series([float(limit[0]) for limit in limits], index=metric.index, dtype="float64")
    fail = pd.Series([float(limit[1]) for limit in limits], index=metric.index, dtype="float64")
    outwards = pd.Series(
        [1.0 if specFor(name).higherIsWorse else -1.0 for name in metric], index=metric.index, dtype="float64"
    )
    excess = relative.astype("float64") * outwards
    status = pd.Series("PASS", index=metric.index, dtype=object)
    status[excess >= warn] = "WARN"
    status[excess >= fail] = "FAIL"
    status[excess.isna() | warn.isna()] = "UNKNOWN"
    return pd.DataFrame({"warn": warn, "fail": fail, "status": status})


def exitCode(comparison: pd.DataFrame) -> int:
    """Return the exit status for a comparison: its worst status.

    `EXIT_FAIL`, `EXIT_WARN` or `EXIT_PASS` of
    `pfs.drp.qa.metrics.visitRollup`; `EXIT_NO_DATA` when nothing was judged
    (no rows, or every one ``UNKNOWN``).
    """
    worst = worstStatus(comparison["status"]) if len(comparison) else "UNKNOWN"
    return {"FAIL": EXIT_FAIL, "WARN": EXIT_WARN, "PASS": EXIT_PASS}.get(worst, EXIT_NO_DATA)


def _rows(tables: pd.DataFrame | Sequence[pd.DataFrame], metrics: Sequence[str]) -> pd.DataFrame:
    """Return `metricRows` of one table or several, together."""
    if isinstance(tables, pd.DataFrame):
        tables = [tables]
    parts = [rows for rows in (metricRows(table, metrics) for table in tables) if len(rows)]
    if not parts:
        return pd.DataFrame(columns=list(_ROW_COLUMNS)).astype({"value": "float64"})
    return pd.concat(parts, ignore_index=True)
