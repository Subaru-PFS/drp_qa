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
    "labelRows",
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

#: The metrics gated by ``imageQualityQa``.
DEFAULT_METRICS = ("medFwhm", "pctFlagged", "medDxCenter")

#: Values of the ``validation`` column added by `labelRows`.
GOOD, BAD_WARN, BAD_FAIL, UNCONFIRMED = "good", "bad:WARN", "bad:FAIL", "unconfirmed"


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
    about every metric if it names none. A row covered by any ``known_bad``
    entry is never used as known-good, even for another metric.

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
        The covered rows with a ``validation`` column: ``good``, ``bad:WARN``,
        ``bad:FAIL`` or ``unconfirmed``, and ``species`` (see `addSpecies`).
        Rows with no verdict for ``metric`` are dropped.
    """
    metrics = addSpecies(metrics)

    def forMetric(entries: Iterable) -> list:
        return [entry for entry in entries if entry.metric in (None, metric)]

    label = np.full(len(metrics), None, dtype=object)
    label[matchRows(metrics, visitSet.knownGood)] = GOOD
    label[matchRows(metrics, visitSet.knownBad)] = None
    label[matchRows(metrics, forMetric(visitSet.unconfirmedBad))] = UNCONFIRMED
    for expect, value in (("WARN", BAD_WARN), ("FAIL", BAD_FAIL)):
        entries = [entry for entry in forMetric(visitSet.confirmedBad) if entry.expect == expect]
        label[matchRows(metrics, entries)] = value
    labelled = metrics.assign(validation=label)
    return labelled[labelled["validation"].notna()]


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
            The known-good sample. Detectors of one visit, and repeated
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
    bad = subset[subset["validation"].isin((BAD_WARN, BAD_FAIL))]
    unconfirmed = subset[subset["validation"] == UNCONFIRMED]
    row = {
        "nGood": len(good),
        "nGoodVisits": good["visit"].nunique(),
        "visitRange": f"{good['visit'].min()}-{good['visit'].max()}" if len(good) else "",
        "nBad": len(bad),
        "nUnconfirmed": len(unconfirmed),
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

    worse = (
        unconfirmed["_value"] >= suggestion.warn
        if spec.higherIsWorse
        else unconfirmed["_value"] <= suggestion.warn
    )
    row["unconfirmedFlagged"] = int(worse.sum())
    return row
