"""Judge metrics against thresholds: the one gating path.

Measurement and judgement are separate (rule R3 of ``docs/qa-principles.md``):
``imageQualityQa`` measures, and `gate` turns a table of metrics into a verdict
per image, whether in the task, in real time, or over a whole run's stored
``iqQaMetrics``.

Thresholds come as tables in the format `readThresholds` returns: one row per
metric and population, with ``warn``, ``fail``, ``higherIsWorse``, ``absolute``
and ``provenance``. Every other column is a population column, and a null in
one means "any value". For each image and metric, the entry used is

1. from the first table, in the order given, with an entry whose population
   matches the image, so a derived file can be layered over the task config;
2. within that table, the matching entry naming the most population columns,
   so ``arm=b species=HgCd`` beats ``arm=b``, which beats an entry naming
   none. Two equally specific matches are an error.

An entry with neither ``warn`` nor ``fail`` stops the search: the metric is
deliberately not judged in that population.

A missing (NaN) value, one with no thresholds, or one its `MetricSpec` marks
as not measured (a FWHM read from ``fiberProfiles``) gets no verdict; an
infinite one is judged, and fails a higher-is-worse metric. A value at a
threshold has crossed it.
"""

from collections.abc import Sequence
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pfs.drp.qa.metrics.calibration import addSpecies, readThresholds
from pfs.drp.qa.metrics.registry import specFor

__all__ = [
    "STATUS_ORDER",
    "VERDICT_COLUMNS",
    "configThresholds",
    "gate",
    "judge",
    "loadThresholds",
    "thresholdsPath",
]

#: Verdicts from best to worst.
STATUS_ORDER = ("PASS", "WARN", "FAIL")

#: Columns of a thresholds table that are not population columns.
_FIELDS = frozenset(
    (
        "metric",
        "higherIsWorse",
        "absolute",
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
)

#: Columns `gate` returns: the verdict, the metric that decided it, and why.
VERDICT_COLUMNS = ("qaStatus", "qaDecidedBy", "qaReason")

#: Columns identifying an image, copied to `judge`'s output when present.
_ID_COLUMNS = ("visit", "arm", "spectrograph")

#: Fallback flag-rate thresholds of ``imageQualityQa`` for an arm with no entry.
_FLAG_RATE_FALLBACK = (15.0, 20.0)

ThresholdsLike = pd.DataFrame | Path | str


def loadThresholds(thresholds: ThresholdsLike | Sequence[ThresholdsLike]) -> list[pd.DataFrame]:
    """Return threshold tables, reading any paths with `readThresholds`.

    Parameters
    ----------
    thresholds : `pandas.DataFrame`, path, or a sequence of them
        Tables, or thresholds files, in priority order.

    Returns
    -------
    `list` [`pandas.DataFrame`]
        The tables, in the order given.
    """
    if isinstance(thresholds, (pd.DataFrame, Path, str)):
        thresholds = [thresholds]
    return [table if isinstance(table, pd.DataFrame) else readThresholds(table)[1] for table in thresholds]


def thresholdsPath(name: str) -> Path:
    """Return the path of a thresholds file named in a task config.

    Parameters
    ----------
    name : `str`
        An absolute path, or a path relative to the thresholds files shipped
        with the package (``pfs/drp/qa/metrics/data``), e.g.
        ``"iqQaThresholds-run25.yaml"``.

    Returns
    -------
    `pathlib.Path`
        The file. Not checked for existence.
    """
    path = Path(name)
    return path if path.is_absolute() else Path(str(files("pfs.drp.qa.metrics") / "data")) / path


def configThresholds(config: Any) -> pd.DataFrame:
    """Return the thresholds of an ``imageQualityQa`` config as a table.

    Parameters
    ----------
    config : `Any`
        An ``ImageQualityQaConfig``, or any object with its threshold
        attributes; duck-typed so this is tested without ``lsst.pex.config``.

    Returns
    -------
    `pandas.DataFrame`
        Entries for ``medFwhm``, ``pctFlagged`` (per ``arm`` and
        ``arm:species`` key, and the 15/20 % fallback for any other arm) and
        ``medDxCenter`` (absolute), in that order, with ``arm`` and
        ``species`` population columns.
    """
    rows = [
        _entry("medFwhm", config.fwhmWarnThreshold, config.fwhmFailThreshold, "fwhmWarn/FailThreshold"),
        _entry("pctFlagged", *_FLAG_RATE_FALLBACK, "fallback for an arm with no flagRate*Threshold entry"),
    ]
    warn, fail = dict(config.flagRateWarnThreshold), dict(config.flagRateFailThreshold)

    def resolve(source: dict, key: str, fallback: float) -> float:
        # The task looks each level up on its own: arm:species, then arm, then
        # the fallback. A key in one dict only takes the other from its arm.
        return source[key] if key in source else source.get(key.split(":", 1)[0], fallback)

    for key in sorted(set(warn) | set(fail)):
        arm, _, species = key.partition(":")
        entry = _entry(
            "pctFlagged",
            resolve(warn, key, _FLAG_RATE_FALLBACK[0]),
            resolve(fail, key, _FLAG_RATE_FALLBACK[1]),
            f"flagRateWarn/FailThreshold[{key!r}]",
        )
        rows.append(entry | {"arm": arm} | ({"species": species} if species else {}))
    rows.append(
        _entry(
            "medDxCenter",
            config.dxCenterWarnThreshold,
            config.dxCenterFailThreshold,
            "dxCenterWarn/FailThreshold",
        )
    )

    table = pd.DataFrame(rows)
    for column in ("arm", "species"):
        if column not in table:
            table[column] = None
    table["higherIsWorse"] = [specFor(name).higherIsWorse for name in table["metric"]]
    table["absolute"] = [specFor(name).absolute for name in table["metric"]]
    return table.astype({"arm": object, "species": object})


def _entry(metric: str, warn: float, fail: float, source: str) -> dict:
    """Return a `configThresholds` row: a pair of thresholds from ``source``."""
    provenance = f"imageQualityQa config {source}; origin unrecorded."
    return {"metric": metric, "warn": float(warn), "fail": float(fail), "provenance": provenance}


def judge(metrics: pd.DataFrame, thresholds: ThresholdsLike | Sequence[ThresholdsLike]) -> pd.DataFrame:
    """Judge each metric of each image.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        One row per image (``iqQaMetrics`` rows), with a column per metric
        and the population columns the thresholds name. ``species`` is added
        from ``seqName`` when absent (`addSpecies`).
    thresholds : `pandas.DataFrame`, path, or a sequence of them
        Threshold tables or files, highest priority first; see the module
        docstring.

    Returns
    -------
    `pandas.DataFrame`
        One row per image and metric named by any table, images in the order
        of ``metrics`` and metrics in the order the tables first name them:

        ``row``
            Position of the image in ``metrics``.
        ``visit``, ``arm``, ``spectrograph``
            When ``metrics`` has them.
        ``metric``, ``value``
            The value judged: absolute, for an ``absolute`` entry.
        ``status``
            ``PASS``, ``WARN`` or ``FAIL``; empty when there is no verdict,
            including for a value its `MetricSpec` marks as not measured.
        ``reason``
            Why, for ``WARN`` and ``FAIL``.
        ``layer``, ``population``, ``warn``, ``fail``, ``provenance``
            The entry used: its table's position in ``thresholds``, its
            population (``"arm=b species=HgCd"``; ``""`` matches any image),
            and its thresholds. ``layer`` is -1 when no entry matched.

    Raises
    ------
    KeyError
        If a table names a metric that is not a column of ``metrics``: an
        unmeasured metric must not pass silently.
    ValueError
        If two entries of one table match an image equally specifically.
    """
    tables = loadThresholds(thresholds)
    metrics = addSpecies(metrics).reset_index(drop=True)
    numRows = len(metrics)
    names = list(dict.fromkeys(name for table in tables for name in table["metric"]))
    missing = [name for name in names if name not in metrics.columns]
    if missing:
        raise KeyError(f"Thresholds name metrics that are not columns of the metrics table: {missing}")

    frames = []
    for name in names:
        chosen = _choose(metrics, tables, name)
        values = metrics[name].astype(float).to_numpy()
        notMeasured = specFor(name).notMeasured(metrics)
        columns: dict[str, list] = {key: [] for key in ("value", "status", "reason", "layer", "population")}
        columns |= {"warn": [], "fail": [], "provenance": []}
        for row in range(numRows):
            layer, entry = chosen[row]
            value, status, reason = values[row], "", ""
            if entry is None:
                columns["layer"].append(-1)
                columns["population"].append("")
                columns["warn"].append(np.nan)
                columns["fail"].append(np.nan)
                columns["provenance"].append("")
            else:
                spec = replace(
                    specFor(name),
                    higherIsWorse=bool(entry.get("higherIsWorse", specFor(name).higherIsWorse)),
                )
                if bool(entry.get("absolute", spec.absolute)):
                    value = abs(value)
                warn, fail = _limit(entry.get("warn")), _limit(entry.get("fail"))
                if not (np.isnan(value) or notMeasured[row]) and (warn is not None or fail is not None):
                    status = "PASS"
                    for level, limit in (("FAIL", fail), ("WARN", warn)):
                        if limit is not None and spec.crossed(value, limit):
                            status, reason = level, spec.describe(value, level, limit)
                            break
                columns["layer"].append(layer)
                columns["population"].append(entry["_population"])
                columns["warn"].append(np.nan if warn is None else warn)
                columns["fail"].append(np.nan if fail is None else fail)
                columns["provenance"].append(entry.get("provenance") or "")
            columns["value"].append(value)
            columns["status"].append(status)
            columns["reason"].append(reason)
        frame = pd.DataFrame({"row": np.arange(numRows), "metric": name} | columns)
        for key in reversed(_ID_COLUMNS):
            if key in metrics.columns:
                frame.insert(1, key, metrics[key].to_numpy())
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["row", "metric", "value", "status", "reason", "layer", "population"])
    return pd.concat(frames, ignore_index=True).sort_values(["row"], kind="stable", ignore_index=True)


def gate(
    metrics: pd.DataFrame,
    thresholds: ThresholdsLike | Sequence[ThresholdsLike],
    default: str = "PASS",
) -> pd.DataFrame:
    """Give each image a verdict, and the metric that decided it.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        One row per image; see `judge`.
    thresholds : `pandas.DataFrame`, path, or a sequence of them
        Threshold tables or files, highest priority first.
    default : `str`, optional
        The verdict for an image with no metric judged.

    Returns
    -------
    `pandas.DataFrame`
        Indexed like ``metrics``, with `VERDICT_COLUMNS`:

        ``qaStatus``
            The worst verdict of the image's metrics, or ``default``.
        ``qaDecidedBy``
            The first metric with that verdict when it is ``WARN`` or
            ``FAIL``; empty otherwise.
        ``qaReason``
            The reasons of every ``WARN`` and ``FAIL``, joined by ``"; "``.
    """
    judged = judge(metrics, thresholds)
    status, decidedBy, reason = [default] * len(metrics), [""] * len(metrics), [""] * len(metrics)
    for row, group in judged.groupby("row", sort=True):
        verdicts = group[group["status"] != ""]
        if verdicts.empty:
            continue
        rank = verdicts["status"].map(STATUS_ORDER.index)
        worst = STATUS_ORDER[int(rank.max())]
        status[row] = worst
        if worst != "PASS":
            decidedBy[row] = verdicts.loc[rank.idxmax(), "metric"]
            reason[row] = "; ".join(verdicts.loc[verdicts["reason"] != "", "reason"])
    return pd.DataFrame(
        {"qaStatus": status, "qaDecidedBy": decidedBy, "qaReason": reason}, index=metrics.index
    )


def _limit(value: Any) -> float | None:
    """Return a threshold as a float, or `None` when it is null."""
    return None if _isNull(value) else float(value)


def _choose(metrics: pd.DataFrame, tables: list[pd.DataFrame], name: str) -> list[tuple[int, dict | None]]:
    """Choose the entry for metric ``name`` for each image.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        The images, with a default index.
    tables : `list` [`pandas.DataFrame`]
        Threshold tables, highest priority first.
    name : `str`
        The metric.

    Returns
    -------
    `list` [`tuple` [`int`, `dict` or `None`]]
        Per image, the table's position and the entry (with its population as
        ``_population``), or ``(-1, None)`` when nothing matched.
    """
    numRows = len(metrics)
    chosen: list[tuple[int, dict | None]] = [(-1, None)] * numRows
    pending = np.ones(numRows, dtype=bool)
    for layer, table in enumerate(tables):
        entries = table[table["metric"] == name].reset_index(drop=True)
        if entries.empty or not pending.any():
            continue
        populationColumns = [column for column in entries.columns if column not in _FIELDS]
        match = np.ones((numRows, len(entries)), dtype=bool)
        specificity = np.zeros(len(entries), dtype=int)
        labels = [[] for _ in range(len(entries))]
        for column in populationColumns:
            wanted = entries[column].to_numpy(dtype=object)
            named = np.array([not _isNull(value) for value in wanted])
            specificity += named
            for index in np.flatnonzero(named):
                labels[index].append(f"{column}={wanted[index]}")
                if column in metrics.columns:
                    match[:, index] &= metrics[column].to_numpy(dtype=object) == wanted[index]
                else:
                    match[:, index] = False
        score = np.where(match, specificity, -1)
        best = score.max(axis=1)
        ties = ((score == best[:, None]).sum(axis=1) > 1) & (best >= 0) & pending
        if ties.any():
            row = int(np.flatnonzero(ties)[0])
            tied = [" ".join(labels[i]) or "(any)" for i in np.flatnonzero(score[row] == best[row])]
            raise ValueError(f"Thresholds table {layer} has equally specific entries for {name}: {tied}")
        best_index = score.argmax(axis=1)
        for row in np.flatnonzero(pending & (best >= 0)):
            entry = entries.iloc[best_index[row]].to_dict()
            entry["_population"] = " ".join(labels[best_index[row]])
            chosen[row] = (layer, entry)
        pending &= best < 0
    return chosen


def _isNull(value: Any) -> bool:
    """Return whether a population value is null, meaning "any"."""
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))
