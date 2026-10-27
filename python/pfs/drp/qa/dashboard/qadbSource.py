"""Read the QA results qadb holds back as the dashboard's Tier 1 table (PIPE2D-1924).

During a run qaActor loads every visit it judges into qadb (INSTRM-3040). This
reads them back in the Butler tables' shapes (`pfs.drp.qa.qadb.tables`, in
reverse) and builds the Tier 1 table `serving.buildTier1` builds from an
export, so the app shows a run as it happens the same way it shows a past one.

Read-only: one query per table, with bound parameters. An image judged in more
than one Butler RUN of the collection is shown as the latest RUN judged it.
Visit times are qadb's ``pfs_visit.issued_at`` (HST); sequences are inferred
from runs of one ``seqName`` (`serving.inferSequences`), as qadb holds no IIC
sequence.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from pfs.drp.qa.dashboard.serving import buildTier1, inferSequences
from pfs.drp.qa.qadb.tables import (
    IMAGE_QUALITY,
    LINE_RESIDUAL,
    LINE_RESIDUAL_METRIC,
    TableSpec,
    snakeCase,
)

__all__ = ["FIRST_VISIT", "LAST_VISIT", "readQadb", "tier1FromQadb", "withSequences"]

#: The widest visit range: every visit.
FIRST_VISIT, LAST_VISIT = 0, 2**31 - 1

#: The tables the dashboard reads, by Butler dataset type.
_SPECS = {spec.source: spec for spec in (IMAGE_QUALITY, LINE_RESIDUAL, LINE_RESIDUAL_METRIC)}

# Each image's latest judgement in the collection: the highest processing (Butler RUN) that judged it.
_LATEST = """
WITH latest AS (
    SELECT g.processing_id, g.pfs_visit_id, g.camera_name, g.decided_at,
        ROW_NUMBER() OVER (
            PARTITION BY g.pfs_visit_id, g.camera_name ORDER BY g.processing_id DESC
        ) AS n
    FROM qa_gate g JOIN data_processing_pipe2d p ON p.processing_id = g.processing_id
    WHERE p.collection_name = :collection AND g.pfs_visit_id BETWEEN :first AND :last
)
"""

_VISITS_SQL = """
SELECT pfs_visit_id, issued_at AS time_exp_start FROM pfs_visit WHERE pfs_visit_id = ANY(:visits)
"""


def _tableSql(table: str, since: bool) -> str:
    # The table name is one of _SPECS', never input.
    return (
        _LATEST
        + f"SELECT t.*, l.decided_at FROM {table} t JOIN latest l ON l.processing_id = t.processing_id"
        " AND l.pfs_visit_id = t.pfs_visit_id AND l.camera_name = t.camera_name WHERE l.n = 1"
        + (" AND l.decided_at >= :since" if since else "")
    )


def _butlerFrame(rows: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Return qadb rows of ``spec`` with the Butler table's column names and types."""
    names = {snakeCase(column): column for column in (*spec.extraKeys, *spec.columns)}
    frame = rows.rename(columns={**names, "pfs_visit_id": "visit"})
    camera = frame["camera_name"].astype(str)
    if "arm" not in frame:
        frame["arm"] = camera.str[0]
    if "spectrograph" not in frame:
        frame["spectrograph"] = camera.str[1:]
    frame["visit"] = frame["visit"].astype("int64")
    frame["spectrograph"] = frame["spectrograph"].astype("int64")
    for column, kind in spec.columns.items():
        if column not in frame or kind not in ("float", "bool"):
            continue
        if kind == "float":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        else:
            # SQLite and a null give 0/1 and None: keep a missing value missing.
            frame[column] = frame[column].map(lambda value: None if pd.isna(value) else bool(value))
    return frame.drop(columns=["processing_id", "camera_name"])


def readQadb(
    qadb: Any,
    collection: str,
    first: int = FIRST_VISIT,
    last: int = LAST_VISIT,
    since: Any = None,
) -> dict[str, pd.DataFrame]:
    """Read a collection's QA results from qadb, as the Butler tables they were loaded from.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The qadb, read-only (``query_dataframe`` with ``:name`` parameters).
    collection : `str`
        The collection the QA tasks wrote to, as loaded (``qaActor/reductions``).
    first, last : `int`
        The visit range, inclusive.
    since : optional
        Only images whose latest judgement was loaded at or after this
        (``qa_gate.decided_at``, as a previous read returned it); default all.

    Returns
    -------
    `dict` [`str`, `pandas.DataFrame`]
        ``iqQaMetrics``, ``dmLineResidualSummary`` and ``dmLineResidualMetrics``
        with their Butler columns (those qadb stores) and ``decidedAt``; and
        ``visitTimes``, qadb's visit listing in the opdb listing's columns
        (``pfs_visit_id``, ``time_exp_start``, ``iic_sequence_id``, the last
        always missing).
    """
    params = {"collection": collection, "first": int(first), "last": int(last)}
    if since is not None:
        params["since"] = since
    tables = {}
    for source, spec in _SPECS.items():
        rows = qadb.query_dataframe(_tableSql(spec.name, since is not None), params=params)
        frame = _butlerFrame(rows, spec).rename(columns={"decided_at": "decidedAt"})
        tables[source] = frame.reset_index(drop=True)
    visits = sorted({int(visit) for visit in tables["iqQaMetrics"]["visit"]})
    times = (
        qadb.query_dataframe(_VISITS_SQL, params={"visits": visits})
        if visits
        else pd.DataFrame(columns=["pfs_visit_id", "time_exp_start"])
    )
    times["time_exp_start"] = pd.to_datetime(times["time_exp_start"])
    tables["visitTimes"] = times.assign(iic_sequence_id=pd.array([pd.NA] * len(times), dtype="Int64"))
    return tables


def tier1FromQadb(tables: Mapping[str, pd.DataFrame], layers: Mapping[str, Sequence]) -> pd.DataFrame:
    """Return the Tier 1 table for what `readQadb` read; empty when it read no image.

    Parameters
    ----------
    tables : mapping
        From `readQadb`.
    layers : mapping
        The thresholds each task judges with, as `serving.taskThresholds`
        returns them, to record each metric's limits beside its stored value.
    """
    iq = tables["iqQaMetrics"]
    if iq.empty:
        return pd.DataFrame()
    summary, metrics = tables["dmLineResidualSummary"], tables["dmLineResidualMetrics"]
    hasLines = not summary.empty and not metrics.empty
    return buildTier1(
        iq.drop(columns="decidedAt"),
        layers["iqQaMetrics"],
        summary.drop(columns="decidedAt") if hasLines else None,
        metrics.drop(columns="decidedAt") if hasLines else None,
        tables["visitTimes"],
        layers["dmLineResidualMetrics"],
    )


def withSequences(table: pd.DataFrame, visitTimes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a Tier 1 table with its sequences and times inferred again over all its visits.

    A table put together from several reads has each read's sequences, numbered
    apart; this numbers them as one (`serving.inferSequences`).
    """
    images = table[["visit", "seqName"]].drop_duplicates("visit")
    sequences = inferSequences(images, visitTimes)
    return (
        table.drop(columns=["sequenceKey", "sequenceSource", "timeExpStart"])
        .merge(sequences, on="visit", how="left")
        .reindex(columns=table.columns)
    )
