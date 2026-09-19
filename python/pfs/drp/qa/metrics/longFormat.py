"""Long-format per-species QA metrics.

Per-species values used to become wide columns -- ``fitSpeciesXRms_HgI``,
``fitSpeciesXRms_CdI``, and so on. The column set therefore varied per quantum:
concatenating across quanta produced a ragged frame padded with NaN, every
downstream ``groupby`` had to know the species in advance, and a species that
appeared in one visit but not the next was indistinguishable from one that was
measured and came back empty.

Long format fixes all three. One row per ``(detector, species, metric)``:

===== === ============ =========== ================== ====== ======
visit arm spectrograph description metric             value  status
===== === ============ =========== ================== ====== ======
12345 b   1            HgI         spatialWeightedRms 0.021  PASS
12345 b   1            ArI         spatialWeightedRms 0.088  FAIL
===== === ============ =========== ================== ====== ======

The schema is fixed, concatenation is trivial, gating is the one path in
`pfs.drp.qa.metrics.registry`, and ``groupby("description")`` is all a plot
needs. A wide view remains available through `widen` for operators who prefer
one, as a derived convenience rather than as the stored form.

See ``doc/qa-rebuild-plan.md`` section 1.3. This module imports neither the LSST
stack nor the Butler.
"""

from collections.abc import Mapping, Sequence

import pandas as pd

from pfs.drp.qa.metrics.registry import MetricRegistry

__all__ = [
    "LONG_COLUMNS",
    "longRecords",
    "toLongFrame",
    "widen",
]

#: The stable column set. Every quantum emits exactly these, in this order.
LONG_COLUMNS = ("visit", "arm", "spectrograph", "description", "metric", "value", "status")

#: Columns that identify the row, as opposed to carrying the measurement.
_INDEX_COLUMNS = ("visit", "arm", "spectrograph", "description")

_DTYPES = {
    "visit": "Int64",
    "arm": "string",
    "spectrograph": "Int64",
    "description": "string",
    "metric": "string",
    "value": "float64",
    "status": "string",
}


def longRecords(
    dataId: Mapping[str, object],
    perSpecies: Mapping[str, Mapping[str, float]],
    registry: MetricRegistry | None = None,
) -> list[dict]:
    """Build long-format records for one quantum.

    Parameters
    ----------
    dataId : `Mapping`
        The quantum's data ID. ``visit``, ``arm`` and ``spectrograph`` are read
        from it; anything else is ignored. A missing key becomes ``None`` rather
        than an error, so a partial data ID still yields usable rows.
    perSpecies : `Mapping` [`str`, `Mapping` [`str`, `float`]]
        ``{description: {metric: value}}``, e.g.
        ``{"HgI": {"fitSpeciesXRms": 0.021}}``. The species names are the
        ``description`` values from the line lists (``HgI``, ``ArI``, ...), not
        the lamp names from ``W_SEQNAM``.
    registry : `MetricRegistry`, optional
        Used to gate each value. When omitted, or when a metric is absent from
        it, ``status`` is left empty: a metric that was not gated must not
        report a verdict it never earned.

    Returns
    -------
    `list` [`dict`]
        One record per ``(description, metric)``, with the keys in
        `LONG_COLUMNS`. Ordered by species and then by metric, so that the
        output is reproducible.

    Notes
    -----
    Gating is tried with the keys ``arm:description`` and then ``arm``, matching
    the per-species then per-arm lookup the flag-rate thresholds use. A metric
    with no such overrides simply falls through to its base thresholds.
    """
    arm = dataId.get("arm")

    records = []
    for description in sorted(perSpecies):
        values = perSpecies[description]
        speciesKeys = _gateKeys(arm, description)
        for metric in sorted(values):
            value = values[metric]
            status = None
            if registry is not None and metric in registry:
                result = registry.gate(metric, value, keys=speciesKeys)
                status = None if result is None else result.status
            records.append(
                {
                    "visit": dataId.get("visit"),
                    "arm": arm,
                    "spectrograph": dataId.get("spectrograph"),
                    "description": description,
                    "metric": metric,
                    "value": value,
                    "status": status,
                }
            )
    return records


def toLongFrame(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Turn long-format records into a DataFrame with a stable schema.

    Parameters
    ----------
    records : `Sequence` [`Mapping`]
        Records as produced by `longRecords`.

    Returns
    -------
    `pandas.DataFrame`
        A frame with exactly `LONG_COLUMNS`, in that order, with fixed dtypes.
        An empty input yields an empty frame with the same columns and dtypes,
        so that ``pd.concat`` over a mix of empty and non-empty quanta neither
        widens the schema nor changes a column's type.

        A missing string -- above all a ``status`` of `None`, "no verdict" -- is
        stored as the empty string, never as null. The Butler's parquet writer
        sizes each string column from its longest non-null value and raises
        on a non-empty column with none, which is every quantum whose species
        metrics are all ungated. ``""`` still carries no verdict: it is not in
        `~pfs.drp.qa.metrics.registry.STATUS_ORDER`.
    """
    frame = pd.DataFrame(list(records), columns=list(LONG_COLUMNS)).astype(_DTYPES)
    for column, dtype in _DTYPES.items():
        if dtype == "string":
            frame[column] = frame[column].fillna("")
    return frame


def widen(frame: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long-format frame to one column per metric.

    A derived convenience for operators reading a table by eye, not a storage
    format: the wide schema varies with which metrics are present, which is
    what long format exists to avoid.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        A long-format frame, as produced by `toLongFrame`.

    Returns
    -------
    `pandas.DataFrame`
        One row per ``(visit, arm, spectrograph, description)`` and one column
        per metric, with the index reset. An empty input yields an empty frame
        carrying just the index columns.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(_INDEX_COLUMNS)).astype(
            {column: _DTYPES[column] for column in _INDEX_COLUMNS}
        )
    return (
        frame.pivot_table(
            index=list(_INDEX_COLUMNS),
            columns="metric",
            values="value",
            aggfunc="first",
            observed=True,
        )
        .rename_axis(columns=None)
        .reset_index()
    )


def _gateKeys(arm: object, description: str | None = None) -> tuple[str, ...]:
    """Build the override lookup keys for a detector and species.

    Parameters
    ----------
    arm : `object`
        The arm name, or ``None`` when the data ID does not carry one.
    description : `str`, optional
        The species name.

    Returns
    -------
    `tuple` [`str`]
        Keys most specific first: ``("b:HgI", "b")``. Empty when there is no
        arm to key on.
    """
    if not arm:
        return ()
    if description:
        return (f"{arm}:{description}", str(arm))
    return (str(arm),)
