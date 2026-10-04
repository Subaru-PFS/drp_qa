"""Derive thresholds for ``iqQaMetrics`` columns from the validation visit set.

`labelRows` marks each metrics row as known-good, known-bad or unconfirmed for
one metric, and `calibrate` turns those labels into a table of suggested
thresholds: one row per metric and population. The table is what a notebook
shows and what a dashboard can store; `pfs.drp.qa.plotting.thresholds` draws
it.

Populations are separated by default. Arc moments, calexp trace widths and sky
frames are different estimators, the b arm behaves differently from the
others, and flag rates depend on the lamp (rule R7): a percentile of a blend is
a property of the blend.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from pfs.drp.qa.metrics.thresholds import deriveThresholds, verifyKnownBad
from pfs.drp.qa.metrics.validationVisits import ValidationVisitSet, matchRows

__all__ = [
    "DEFAULT_METRICS",
    "METRIC_SPECS",
    "MetricSpec",
    "addSpecies",
    "calibrate",
    "compareRuns",
    "labelRows",
    "selectGroup",
    "summarizeRuns",
]


@dataclass(frozen=True)
class MetricSpec:
    """How to derive thresholds for one metrics column.

    Attributes
    ----------
    name : `str`
        Column name in ``iqQaMetrics``.
    higherIsWorse : `bool`
        Direction of the metric.
    absolute : `bool`
        Calibrate on the absolute value, as the gate sees it.
    groupBy : `tuple` [`str`]
        Columns separating populations. Columns absent from the data are
        dropped from the grouping.
    physicalLimit : `float` or `None`
        A physical limit to use for FAIL.
    """

    name: str
    higherIsWorse: bool = True
    absolute: bool = False
    groupBy: tuple[str, ...] = ("arm", "obsType")
    physicalLimit: float | None = None


#: The ``iqQaMetrics`` columns with a known treatment. ``pctFlagged`` is split
#: by species because ``flagRate*Threshold`` is keyed by ``arm:species``;
#: ``nLines`` by lamp because the line count is a property of the lamp.
METRIC_SPECS = {
    spec.name: spec
    for spec in (
        MetricSpec("medFwhm"),
        MetricSpec("medDxCenter", absolute=True),
        MetricSpec("dxCenterRms"),
        MetricSpec("pctFlagged", groupBy=("obsType", "arm", "species")),
        MetricSpec("nLines", higherIsWorse=False, groupBy=("arm", "seqName")),
    )
}

#: The metrics whose thresholds are derived from the reference run.
#: ``medDxCenter`` is gated by ``imageQualityQa`` but not derived: an offset
#: from ``detectorMap_calib`` is near zero in the run the calibrations were made
#: from, so its percentiles say nothing about the drift to tolerate. Its
#: thresholds are a tolerance (PIPE2D-1921); `summarizeRuns` reports it.
DEFAULT_METRICS = ("medFwhm", "pctFlagged")

#: Values of the ``validation`` column added by `labelRows`: known good in a
#: reference run (thresholds come from these), known good in another run (held
#: out and compared), known bad, and suspected bad.
GOOD, HELD_OUT, BAD_WARN, BAD_FAIL, UNCONFIRMED = "good", "heldOut", "bad:WARN", "bad:FAIL", "unconfirmed"


def addSpecies(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return ``metrics`` with a ``species`` column taken from ``seqName``.

    The species is what ``imageQualityQa`` keys flag-rate thresholds on: the
    part of ``W_SEQNAM`` after the colon (``"Arc: HgCd"`` gives ``"HgCd"``),
    or an empty string when there is no colon.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        Metrics with a ``seqName`` column.

    Returns
    -------
    `pandas.DataFrame`
        A copy with ``species`` added; unchanged if ``seqName`` is absent.
    """
    if "seqName" not in metrics.columns:
        return metrics
    seqName = metrics["seqName"].fillna("").astype(str)
    return metrics.assign(
        species=seqName.map(lambda name: name.split(":", 1)[-1].strip() if ":" in name else "")
    )


def labelRows(metrics: pd.DataFrame, visitSet: ValidationVisitSet, metric: str) -> pd.DataFrame:
    """Label the metrics rows the validation set has a verdict on.

    A ``known_bad`` entry asserts something only about the metric it names, or
    about every metric if it names none, and removes its rows from the good
    data of only those metrics: a b-arm arc whose flag rate is a pipeline
    limitation still measures FWHM. Known-good rows of a reference run are
    ``good``; those of other runs are ``heldOut``.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        Metrics rows with ``visit`` and, where available, ``arm``,
        ``spectrograph`` and ``seqName``.
    visitSet : `ValidationVisitSet`
        The validation visit set.
    metric : `str`
        The metric being calibrated.

    Returns
    -------
    `pandas.DataFrame`
        The covered rows with a ``validation`` column (``good``, ``heldOut``,
        ``bad:WARN``, ``bad:FAIL`` or ``unconfirmed``), ``run`` (`None` when
        the set has no runs table) and ``species`` (see `addSpecies`). Rows
        with no verdict for ``metric`` are dropped.
    """
    metrics = addSpecies(metrics)

    def forMetric(entries: Iterable) -> list:
        return [entry for entry in entries if entry.metric in (None, metric)]

    label = np.full(len(metrics), None, dtype=object)
    label[matchRows(metrics, visitSet.referenceGood)] = GOOD
    label[matchRows(metrics, visitSet.heldOutGood)] = HELD_OUT
    label[matchRows(metrics, forMetric(visitSet.knownBad))] = None
    label[matchRows(metrics, forMetric(visitSet.unconfirmedBad))] = UNCONFIRMED
    for expect, value in (("WARN", BAD_WARN), ("FAIL", BAD_FAIL)):
        entries = [entry for entry in forMetric(visitSet.confirmedBad) if entry.expect == expect]
        label[matchRows(metrics, entries)] = value
    labelled = metrics.assign(validation=label)
    labelled = labelled[labelled["validation"].notna()]
    return labelled.assign(run=[visitSet.runOf(int(visit)) for visit in labelled["visit"]])


def selectGroup(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Return the rows of ``frame`` in the population of one `calibrate` row.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        Labelled metrics rows, with the grouping columns named in
        ``row["groupBy"]``.
    row : `pandas.Series`
        A row of the `calibrate` table.

    Returns
    -------
    `pandas.DataFrame`
        The rows whose grouping columns equal the row's (nulls match nulls).
    """
    mask = np.ones(len(frame), dtype=bool)
    for column in (column for column in str(row["groupBy"]).split("/") if column):
        value = row[column]
        series = frame[column]
        mask &= (series.isna() if pd.isna(value) else series == value).to_numpy()
    return frame[mask]


def calibrate(
    metrics: pd.DataFrame,
    visitSet: ValidationVisitSet,
    metricNames: Sequence[str] = DEFAULT_METRICS,
    groupBy: Sequence[str] | None = None,
    warnPercentile: float = 95.0,
    failPercentile: float = 99.0,
    derivedOn: date | None = None,
) -> pd.DataFrame:
    """Suggest thresholds for metrics columns, one population at a time.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        Concatenated ``iqQaMetrics`` rows (or any table with a ``visit`` column
        and the metric columns).
    visitSet : `ValidationVisitSet`
        The validation visit set.
    metricNames : sequence of `str`, optional
        Columns to calibrate. Defaults to `DEFAULT_METRICS`. A name not in
        `METRIC_SPECS` is treated as higher-is-worse, grouped by arm and
        ``obsType``.
    groupBy : sequence of `str`, optional
        Override every metric's grouping. An empty sequence pools everything.
    warnPercentile, failPercentile : `float`, optional
        Percentiles for WARN and FAIL. Defaults are 95 and 99.
    derivedOn : `datetime.date`, optional
        Derivation date for the provenance. Defaults to today.

    Returns
    -------
    `pandas.DataFrame`
        One row per metric and population, with columns:

        ``metric``, ``group`` (e.g. ``"b/arc"``), ``groupBy``
            What was calibrated.
        the grouping columns
            The population's values.
        ``nGood``, ``nGoodVisits``, ``visitRange``
            The known-good sample of the reference runs. Detectors of one visit, and repeated
            exposures, are correlated, so ``nGoodVisits`` is the more honest
            sample size.
        ``warn``, ``fail``, ``warnRaw``, ``failRaw``, ``failLow``, ``failHigh``
            The thresholds, unrounded values and the interval on FAIL.
        ``median``, ``robustRms``, ``goodFlaggedWarn``, ``goodFlaggedFail``
            The good distribution, and what the thresholds flag of it.
        ``reliable``, ``failBounded``, ``degenerate``
            Reasons not to trust the suggestion.
        ``nBad``, ``badOk``, ``badSummary``
            The known-bad check: ``badOk`` is True when each known-bad value
            reaches the verdict its entry expects, and NA when no known-bad
            rows name this metric in this population.
        ``nUnconfirmed``, ``unconfirmedFlagged``
            Suspected faults and how many the suggestion would flag at WARN
            or worse. Reported, never judged.
        ``nHeldOut``, ``heldOutFlaggedWarn``, ``heldOutFlaggedFail``
            Known-good data of the other runs, and the fractions flagged: an
            out-of-sample false-alarm rate. Per run in `compareRuns`.
        ``provenance``
            The sentence for the config field's ``doc``.

        A population with known-bad or unconfirmed rows and no known-good rows
        gets a row with ``nGood`` 0 and no thresholds, so the gap shows.

    Raises
    ------
    KeyError
        If a requested metric is not a column of ``metrics``.
    """
    rows = []
    for name in metricNames:
        if name not in metrics.columns:
            raise KeyError(f"{name!r} is not a column of the metrics table")
        spec = METRIC_SPECS.get(name, MetricSpec(name))
        labelled = labelRows(metrics, visitSet, name)
        columns = [column for column in (spec.groupBy if groupBy is None else groupBy) if column in labelled]
        values = labelled[name].astype(float)
        labelled = labelled.assign(_value=values.abs() if spec.absolute else values)
        groups = labelled.groupby(columns, dropna=False, sort=True) if columns else [((), labelled)]
        for key, subset in groups:
            key = key if isinstance(key, tuple) else (key,)
            base = {
                "metric": name,
                "group": "/".join(str(value) for value in key) if columns else "all",
                "groupBy": "/".join(columns),
                **dict(zip(columns, key, strict=True)),
            }
            rows.append(
                base
                | _calibrateGroup(
                    subset, spec, name, base["group"], warnPercentile, failPercentile, derivedOn
                )
            )
    return pd.DataFrame(rows)


def _calibrateGroup(
    subset: pd.DataFrame,
    spec: MetricSpec,
    name: str,
    group: str,
    warnPercentile: float,
    failPercentile: float,
    derivedOn: date | None,
) -> dict:
    """Derive and check the thresholds for one population.

    Parameters
    ----------
    subset : `pandas.DataFrame`
        The population's labelled rows, with the calibrated value in
        ``_value``.
    spec : `MetricSpec`
        The metric's treatment.
    name : `str`
        The metric name.
    group : `str`
        The population label.
    warnPercentile, failPercentile : `float`
        Percentiles for WARN and FAIL.
    derivedOn : `datetime.date` or `None`
        Derivation date.

    Returns
    -------
    `dict`
        The table columns described in `calibrate`, less the identifying ones.
    """
    good = subset[(subset["validation"] == GOOD) & np.isfinite(subset["_value"])]
    heldOut = subset[(subset["validation"] == HELD_OUT) & np.isfinite(subset["_value"])]
    bad = subset[subset["validation"].isin((BAD_WARN, BAD_FAIL))]
    unconfirmed = subset[subset["validation"] == UNCONFIRMED]
    row = {
        "nGood": len(good),
        "nGoodVisits": good["visit"].nunique(),
        "visitRange": f"{good['visit'].min()}-{good['visit'].max()}" if len(good) else "",
        "nBad": len(bad),
        "nUnconfirmed": len(unconfirmed),
        "nHeldOut": len(heldOut),
        "badOk": pd.NA,
        "badSummary": "",
    }
    if good.empty:
        row["badSummary"] = "no known-good rows in this population"
        return row

    suggestion = deriveThresholds(
        good["_value"],
        metric=f"{name}[{group}]",
        higherIsWorse=spec.higherIsWorse,
        warnPercentile=warnPercentile,
        failPercentile=failPercentile,
        physicalLimit=spec.physicalLimit,
        visitRange=row["visitRange"],
        derivedOn=derivedOn,
    )
    row |= {
        "warn": suggestion.warn,
        "fail": suggestion.fail,
        "warnRaw": suggestion.warnRaw,
        "failRaw": suggestion.failRaw,
        "failLow": suggestion.failInterval[0],
        "failHigh": suggestion.failInterval[1],
        "median": suggestion.median,
        "robustRms": suggestion.robustRms,
        "goodFlaggedWarn": suggestion.goodFlaggedWarn,
        "goodFlaggedFail": suggestion.goodFlaggedFail,
        "reliable": suggestion.reliable,
        "failBounded": suggestion.failBounded,
        "degenerate": suggestion.degenerate,
        "provenance": suggestion.provenance,
    }

    checks = []
    for label, expect in ((BAD_FAIL, "FAIL"), (BAD_WARN, "WARN")):
        values = bad.loc[bad["validation"] == label, "_value"]
        if values.empty:
            continue
        try:
            checks.append(verifyKnownBad(values, suggestion, expect))
        except ValueError as exc:
            # Known-bad rows with no usable value: a broken check, not a pass.
            checks.append((False, str(exc)))
    if checks:
        row["badOk"] = all(ok for ok, _ in checks)
        row["badSummary"] = "; ".join(message for _, message in checks)

    for name, level in (("heldOutFlaggedWarn", suggestion.warn), ("heldOutFlaggedFail", suggestion.fail)):
        flagged = heldOut["_value"] >= level if spec.higherIsWorse else heldOut["_value"] <= level
        row[name] = float(flagged.mean()) if len(heldOut) else np.nan

    worse = (
        unconfirmed["_value"] >= suggestion.warn
        if spec.higherIsWorse
        else unconfirmed["_value"] <= suggestion.warn
    )
    row["unconfirmedFlagged"] = int(worse.sum())
    return row


def compareRuns(
    metrics: pd.DataFrame,
    visitSet: ValidationVisitSet,
    table: pd.DataFrame,
    metricNames: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Compare each run's known-good data with the reference runs' thresholds.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        Concatenated metrics rows, as given to `calibrate`.
    visitSet : `ValidationVisitSet`
        The validation visit set.
    table : `pandas.DataFrame`
        The `calibrate` table for ``metrics``.
    metricNames : sequence of `str`, optional
        Metrics to compare. Defaults to `DEFAULT_METRICS`.

    Returns
    -------
    `pandas.DataFrame`
        One row per metric, population and run with known-good data:
        ``metric``, ``group``, ``run``, ``reference`` (the thresholds were
        derived from this run), ``n``, ``median`` and the fractions flagged at
        WARN or worse and at FAIL. A held-out run flagged far more often than
        the reference has moved, or its data differ, which is the comparison
        this exists for.
    """
    rows = []
    for name in metricNames:
        spec = METRIC_SPECS.get(name, MetricSpec(name))
        labelled = labelRows(metrics, visitSet, name)
        labelled = labelled[labelled["validation"].isin((GOOD, HELD_OUT))]
        values = labelled[name].astype(float)
        labelled = labelled.assign(_value=values.abs() if spec.absolute else values)
        for _, suggestion in table[(table["metric"] == name) & (table["nGood"] > 0)].iterrows():
            population = selectGroup(labelled, suggestion)
            population = population[np.isfinite(population["_value"])]
            for run, data in population.groupby("run", dropna=False):
                value = data["_value"]
                if spec.higherIsWorse:
                    flaggedWarn, flaggedFail = value >= suggestion["warn"], value >= suggestion["fail"]
                else:
                    flaggedWarn, flaggedFail = value <= suggestion["warn"], value <= suggestion["fail"]
                rows.append(
                    {
                        "metric": name,
                        "group": suggestion["group"],
                        "run": run,
                        "reference": bool((data["validation"] == GOOD).all()),
                        "n": len(data),
                        "median": float(value.median()),
                        "flaggedWarn": float(flaggedWarn.mean()),
                        "flaggedFail": float(flaggedFail.mean()),
                    }
                )
    return pd.DataFrame(rows)


def summarizeRuns(
    metrics: pd.DataFrame,
    visitSet: ValidationVisitSet,
    metricNames: Sequence[str],
    groupBy: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Describe each run's known-good values of metrics that have no derived thresholds.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        Concatenated metrics rows, as given to `calibrate`.
    visitSet : `ValidationVisitSet`
        The validation visit set.
    metricNames : sequence of `str`
        Metrics to describe.
    groupBy : sequence of `str`, optional
        Override each metric's grouping, as in `calibrate`.

    Returns
    -------
    `pandas.DataFrame`
        One row per metric, population and run: ``metric``, ``group``, ``run``,
        ``reference``, ``n``, ``median``, ``robustRms``, ``p99`` and ``max``,
        of the absolute values for a metric gated on them.
    """
    rows = []
    for name in metricNames:
        spec = METRIC_SPECS.get(name, MetricSpec(name))
        labelled = labelRows(metrics, visitSet, name)
        labelled = labelled[labelled["validation"].isin((GOOD, HELD_OUT))]
        values = labelled[name].astype(float)
        labelled = labelled.assign(_value=values.abs() if spec.absolute else values)
        labelled = labelled[np.isfinite(labelled["_value"])]
        columns = [column for column in (spec.groupBy if groupBy is None else groupBy) if column in labelled]
        for key, data in labelled.groupby([*columns, "run"], dropna=False, sort=True):
            *group, run = key if isinstance(key, tuple) else (key,)
            value = data["_value"].to_numpy()
            q25, q75 = np.percentile(value, [25.0, 75.0])
            rows.append(
                {
                    "metric": name,
                    "group": "/".join(str(item) for item in group) if columns else "all",
                    "run": run,
                    "reference": bool((data["validation"] == GOOD).all()),
                    "n": len(value),
                    "median": float(np.median(value)),
                    "robustRms": float(0.741 * (q75 - q25)),
                    "p99": float(np.percentile(value, 99.0)),
                    "max": float(value.max()),
                }
            )
    return pd.DataFrame(rows)
