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

A metric with a `MetricSpec.tolerance` also gets a row per finer population
(`MetricSpec.toleranceBy`: ``medFwhm`` per detector and lamp) whose WARN and
FAIL are a fixed fraction beyond that population's known-good median
(``method`` ``tolerance``). The gate prefers them, being more specific, and
falls back to the percentile rows (``method`` ``percentile``) for an image
whose finer population has no known-good values.
"""

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pfs.drp.qa.metrics.registry import METRIC_SPECS, MetricSpec, specFor
from pfs.drp.qa.metrics.thresholds import deriveThresholds, deriveTolerance, verifyKnownBad
from pfs.drp.qa.metrics.validationVisits import ValidationVisitSet, matchRows
from pfs.drp.qa.utils.provenance import PIPELINE_PRODUCTS

__all__ = [
    "DEFAULT_METRICS",
    "METRIC_SPECS",
    "MetricSpec",
    "addSpecies",
    "calibrate",
    "labelRows",
    "readThresholds",
    "selectGroup",
    "summarizeGood",
    "writeThresholds",
]

#: Schema version of the thresholds file written by `writeThresholds`. Version 2
#: added each entry's ``method``; a reader that predates it would take
#: ``method`` for a population column, match nothing, and judge nothing, so
#: such readers must reject the file. The optional ``pipelineVersions`` needed no
#: new version: a reader that predates it ignores it and judges as before.
THRESHOLDS_VERSION = 2

#: Versions `readThresholds` reads. A version-1 entry is a percentile one.
_READABLE_VERSIONS = (1, 2)

#: Columns of a `calibrate` row kept in the thresholds file, besides the population.
_THRESHOLD_FIELDS = (
    "method",
    "warn",
    "fail",
    "nGood",
    "nGoodVisits",
    "visitRange",
    "reliable",
    "failBounded",
    "degenerate",
    "provenance",
)


#: The metrics whose thresholds are derived from the reference run.
#: ``medDxCenter`` is gated by ``imageQualityQa`` but not derived: an offset
#: from ``detectorMap_calib`` is near zero in the run the calibrations were made
#: from, so its percentiles say nothing about the drift to tolerate. Its
#: thresholds are a tolerance (PIPE2D-1921); `summarizeGood` reports it.
DEFAULT_METRICS = ("medFwhm", "pctFlagged")

#: Values of the ``validation`` column added by `labelRows`: known good
#: (thresholds come from these), known bad, and suspected bad.
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
    about every metric if it names none, and removes its rows from the good
    data of only those metrics: a b-arm arc whose flag rate is a pipeline
    limitation still measures FWHM. Known-good rows are ``good``; a
    ``known_good`` entry with ``metrics`` is good for those metrics only.

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
        The covered rows with a ``validation`` column (``good``, ``bad:WARN``, ``bad:FAIL`` or ``unconfirmed``), ``run`` (`None` when
        the set has no runs table) and ``species`` (see `addSpecies`). Rows
        with no verdict for ``metric``, and rows whose value its `MetricSpec`
        marks as not measured, are dropped.
    """
    metrics = addSpecies(metrics)

    def forMetric(entries: Iterable) -> list:
        return [entry for entry in entries if entry.metric in (None, metric)]

    label = np.full(len(metrics), None, dtype=object)
    good = [entry for entry in visitSet.knownGood if not entry.metrics or metric in entry.metrics]
    label[matchRows(metrics, good)] = GOOD
    label[matchRows(metrics, forMetric(visitSet.knownBad))] = None
    label[matchRows(metrics, forMetric(visitSet.unconfirmedBad))] = UNCONFIRMED
    for expect, value in (("WARN", BAD_WARN), ("FAIL", BAD_FAIL)):
        entries = [entry for entry in forMetric(visitSet.confirmedBad) if entry.expect == expect]
        label[matchRows(metrics, entries)] = value
    label[specFor(metric).notMeasured(metrics)] = None
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
        ``method``
            ``percentile``, or ``tolerance`` for a row of a metric's
            `MetricSpec.toleranceBy` populations (``group`` e.g.
            ``"b/2/arc/Krypton"``): WARN and FAIL a fixed fraction beyond the
            population's known-good median.
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
            Reasons not to trust the suggestion. For a tolerance,
            ``failLow``/``failHigh`` and ``failBounded`` describe the interval
            on the median, scaled to FAIL.
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
        spec = specFor(name)
        labelled = labelRows(metrics, visitSet, name)
        values = labelled[name].astype(float)
        labelled = labelled.assign(_value=values.abs() if spec.absolute else values)
        populations = [("percentile", spec.groupBy if groupBy is None else groupBy)]
        if spec.tolerance is not None and groupBy is None:
            populations.append(("tolerance", spec.toleranceBy))
        for method, by in populations:
            columns = [column for column in by if column in labelled]
            groups = labelled.groupby(columns, dropna=False, sort=True) if columns else [((), labelled)]
            for key, subset in groups:
                key = key if isinstance(key, tuple) else (key,)
                base = {
                    "metric": name,
                    "group": "/".join(str(value) for value in key) if columns else "all",
                    "groupBy": "/".join(columns),
                    "method": method,
                    **dict(zip(columns, key, strict=True)),
                }
                rows.append(
                    base
                    | _calibrateGroup(
                        subset, spec, name, base["group"], method, warnPercentile, failPercentile, derivedOn
                    )
                )
    return pd.DataFrame(rows)


def _calibrateGroup(
    subset: pd.DataFrame,
    spec: MetricSpec,
    name: str,
    group: str,
    method: str,
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
    method : `str`
        ``percentile``, or ``tolerance`` for `MetricSpec.tolerance`.
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

    if method == "tolerance":
        suggestion = deriveTolerance(
            good["_value"],
            metric=f"{name}[{group}]",
            warnFraction=spec.tolerance[0],
            failFraction=spec.tolerance[1],
            higherIsWorse=spec.higherIsWorse,
            visitRange=row["visitRange"],
            derivedOn=derivedOn,
            units=spec.unitSuffix,
        )
    else:
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


def summarizeGood(
    metrics: pd.DataFrame,
    visitSet: ValidationVisitSet,
    metricNames: Sequence[str],
    groupBy: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Describe the known-good values of metrics that have no derived thresholds.

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
        One row per metric and population: ``metric``, ``group``, ``n``,
        ``median``, ``robustRms``, ``p99`` and ``max``, of the absolute values
        for a metric gated on them.
    """
    rows = []
    for name in metricNames:
        spec = specFor(name)
        labelled = labelRows(metrics, visitSet, name)
        labelled = labelled[labelled["validation"] == GOOD]
        values = labelled[name].astype(float)
        labelled = labelled.assign(_value=values.abs() if spec.absolute else values)
        labelled = labelled[np.isfinite(labelled["_value"])]
        if labelled.empty:
            continue
        columns = [column for column in (spec.groupBy if groupBy is None else groupBy) if column in labelled]
        groups = labelled.groupby(columns, dropna=False, sort=True) if columns else [((), labelled)]
        for key, data in groups:
            key = key if isinstance(key, tuple) else (key,)
            value = data["_value"].to_numpy()
            q25, q75 = np.percentile(value, [25.0, 75.0])
            rows.append(
                {
                    "metric": name,
                    "group": "/".join(str(item) for item in key) if columns else "all",
                    "n": len(value),
                    "median": float(np.median(value)),
                    "robustRms": float(0.741 * (q75 - q25)),
                    "p99": float(np.percentile(value, 99.0)),
                    "max": float(value.max()),
                }
            )
    return pd.DataFrame(rows)


def writeThresholds(
    table: pd.DataFrame,
    path: Path | str,
    collection: str,
    referenceRuns: Sequence[int],
    drpQaVersion: str,
    derivedOn: date | None = None,
    pipelineVersions: Mapping[str, str] | None = None,
) -> Path:
    """Write the derived thresholds to a versioned YAML file.

    The file is what the judgement step reads, in real time and in comparison:
    one entry per metric and population with a known-good sample, keyed by the
    population's columns, with the provenance that says how far to trust it.

    Parameters
    ----------
    table : `pandas.DataFrame`
        The `calibrate` table.
    path : `pathlib.Path` or `str`
        Where to write.
    collection : `str`
        The collection the metrics came from.
    referenceRuns : sequence of `int`
        The runs the thresholds were derived from.
    drpQaVersion : `str`
        The drp_qa version that measured them, e.g. ``git describe``.
    derivedOn : `datetime.date`, optional
        Derivation date. Defaults to today.
    pipelineVersions : mapping [`str`, `str`], optional
        The pipeline that reduced the visits, as `PIPELINE_PRODUCTS` versions
        (`pfs.drp.qa.utils.provenance.commonPipeline`). The thresholds hold for
        it: the gate marks a verdict on a reduction made with another
        provisional. Omitted when unknown.

    Returns
    -------
    `pathlib.Path`
        The file written.
    """
    entries = []
    for _, row in table[table["nGood"] > 0].iterrows():
        columns = [column for column in str(row["groupBy"]).split("/") if column]
        if row.get("method") == "tolerance":
            # A trace has no lamp: its species is "", which says nothing obsType doesn't.
            columns = [column for column in columns if row[column] != ""]
        spec = specFor(row["metric"])
        entry = {
            "metric": row["metric"],
            "population": {column: _populationValue(row[column]) for column in columns},
            "higherIsWorse": spec.higherIsWorse,
            "absolute": spec.absolute,
        }
        entry |= {name: _plain(row[name]) for name in _THRESHOLD_FIELDS}
        entries.append(entry)
    document = {
        "version": THRESHOLDS_VERSION,
        "derivedOn": (derivedOn or date.today()).isoformat(),
        "collection": collection,
        "referenceRuns": [int(run) for run in referenceRuns],
        "drpQaVersion": drpQaVersion,
        **({"pipelineVersions": _pipeline(pipelineVersions)} if pipelineVersions else {}),
        "thresholds": entries,
    }
    path = Path(path)
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=110))
    return path


def readThresholds(path: Path | str) -> tuple[dict, pd.DataFrame]:
    """Read a file written by `writeThresholds`.

    Parameters
    ----------
    path : `pathlib.Path` or `str`
        The file.

    Returns
    -------
    metadata : `dict`
        ``derivedOn``, ``collection``, ``referenceRuns``, ``drpQaVersion``,
        and ``pipelineVersions``: the reducing pipeline's version per product,
        empty when the file records none.
    thresholds : `pandas.DataFrame`
        One row per metric and population: ``metric``, the population's
        columns, ``higherIsWorse``, ``absolute`` and the threshold fields.

    Raises
    ------
    ValueError
        If the file's version is not one this reader knows (1 or
        `THRESHOLDS_VERSION`).
    """
    document = yaml.safe_load(Path(path).read_text())
    version = document.get("version")
    if version not in _READABLE_VERSIONS:
        raise ValueError(f"{path}: thresholds version {version!r}, expected one of {_READABLE_VERSIONS}")
    if version == 1:
        document["thresholds"] = [{"method": "percentile"} | entry for entry in document["thresholds"]]
    rows = [
        {"metric": entry["metric"], **entry["population"]}
        | {key: value for key, value in entry.items() if key not in ("metric", "population")}
        for entry in document["thresholds"]
    ]
    table = pd.DataFrame(rows)
    # Keep population values as written: a spectrograph named by some entries
    # only would otherwise read as 2.0, and print so in the gate's reasons.
    for column in {column for entry in document["thresholds"] for column in entry["population"]}:
        table[column] = pd.Series([row.get(column) for row in rows], index=table.index, dtype=object)
    metadata = {key: document[key] for key in ("derivedOn", "collection", "referenceRuns", "drpQaVersion")}
    metadata["pipelineVersions"] = _pipeline(document.get("pipelineVersions") or {})
    return metadata, table


def _pipeline(versions: Mapping[str, str]) -> dict[str, str]:
    """Return pipeline versions as plain strings, in `PIPELINE_PRODUCTS` order, the others after."""
    ordered = [
        *(p for p in PIPELINE_PRODUCTS if p in versions),
        *(p for p in versions if p not in PIPELINE_PRODUCTS),
    ]
    return {product: str(versions[product]) for product in ordered}


def _populationValue(value):
    """Return a population value for the file: a spectrograph 2, not the 2.0 a column with nulls holds."""
    value = _plain(value)
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _plain(value):
    """Return ``value`` as a YAML-safe Python scalar (numpy types and NaN included)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else ("inf" if value > 0 else "-inf")
    return value
