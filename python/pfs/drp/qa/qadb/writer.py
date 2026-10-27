"""Idempotent writes to qadb.

Every write is ``INSERT ... ON CONFLICT (keys) DO UPDATE``: loading an image
again replaces its rows, so a collection can be loaded as often as it is
re-judged, and a load that stopped halfway is finished by running it again.
The database is a `pfs.utils.database.qadb.QaDB` (or anything with its
``commit`` and ``query_scalar``), queried with bound parameters only.
"""

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from pfs.drp.qa.metrics.calibration import readThresholds
from pfs.drp.qa.qadb.tables import QA_THRESHOLD_ENTRY, QA_THRESHOLDS, thresholdRows

__all__ = [
    "PIPELINE_VERSION_COLUMNS",
    "UNKNOWN_VERSION",
    "deleteRows",
    "ensureVisits",
    "processingId",
    "records",
    "registerThresholds",
    "sha256Of",
    "upsert",
    "upsertSql",
]

#: The drp_qa version recorded when a RUN's ``packages`` dataset names none; never overwrites a known one.
UNKNOWN_VERSION = "unknown"

#: ``data_processing_pipe2d`` column of each product's version (``pipe2d_version`` is drp_stella's).
PIPELINE_VERSION_COLUMNS = {
    "drp_stella": "pipe2d_version",
    "datamodel": "datamodel_version",
    "obs_pfs": "obs_pfs_version",
}

_PFS_VISIT_COLUMNS = ("pfs_visit_id", "pfs_visit_description", "pfs_design_id", "issued_at")

_PFS_VISIT_SQL = """
SELECT pfs_visit_id, pfs_visit_description, pfs_design_id, issued_at
FROM pfs_visit
WHERE pfs_visit_id = ANY(:visits)
"""


def upsertSql(table: str, columns: Sequence[str], keys: Sequence[str]) -> str:
    """Return the ``INSERT ... ON CONFLICT`` statement that writes or replaces a row.

    Parameters
    ----------
    table : `str`
        The table.
    columns : sequence of `str`
        Every column written, the keys included; each is bound as ``:column``.
    keys : sequence of `str`
        The columns of the primary key or a unique constraint.

    Returns
    -------
    `str`
        The statement. The non-key columns are updated from ``EXCLUDED`` on a
        conflict; with no non-key column the conflicting row is left alone.
    """
    missing = [key for key in keys if key not in columns]
    if missing:
        raise ValueError(f"Keys {missing} are not among the columns of {table}")
    names = ", ".join(columns)
    values = ", ".join(f":{column}" for column in columns)
    updates = ", ".join(f"{column} = EXCLUDED.{column}" for column in columns if column not in keys)
    action = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
    return f"INSERT INTO {table} ({names}) VALUES ({values}) ON CONFLICT ({', '.join(keys)}) {action}"


def _native(value: Any) -> Any:
    """Return ``value`` as a Python scalar the database driver binds: `None` for NaN and NaT."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a table's rows as dicts of Python scalars, for binding."""
    return [
        {column: _native(value) for column, value in zip(frame.columns, row, strict=True)}
        for row in frame.itertuples(index=False, name=None)
    ]


def _run(qadb: Any, sql: str, params: list[dict[str, Any]], conn: Any) -> None:
    """Run a statement for every row of ``params``: on ``conn`` when given, else in its own transaction."""
    if conn is not None:
        conn.execute(text(sql), params)
    else:
        qadb.commit(sql, params=params)


def upsert(qadb: Any, table: str, frame: pd.DataFrame, keys: Sequence[str], conn: Any = None) -> int:
    """Write a table's rows, replacing any already there.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The database.
    table : `str`
        The table.
    frame : `pandas.DataFrame`
        The rows, every column a column of the table.
    keys : sequence of `str`
        The columns that identify a row.
    conn : `sqlalchemy.engine.Connection`, optional
        From ``qadb.connection()``, to put several writes in one transaction.

    Returns
    -------
    `int`
        Rows written.
    """
    if frame.empty:
        return 0
    _run(qadb, upsertSql(table, list(frame.columns), keys), records(frame), conn)
    return len(frame)


def deleteRows(qadb: Any, table: str, keys: pd.DataFrame, conn: Any = None) -> int:
    """Delete the rows of a table that match each row of ``keys`` on every column of ``keys``.

    For a long table before its reload: an upsert alone would leave the rows
    of a species or metric the new judgement no longer has.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The database.
    table : `str`
        The table.
    keys : `pandas.DataFrame`
        One row per key to delete, e.g. the images loaded; duplicates are
        deleted once.
    conn : `sqlalchemy.engine.Connection`, optional
        As for `upsert`.

    Returns
    -------
    `int`
        Keys deleted (not rows).
    """
    keys = keys.drop_duplicates()
    if keys.empty:
        return 0
    where = " AND ".join(f"{column} = :{column}" for column in keys.columns)
    _run(qadb, f"DELETE FROM {table} WHERE {where}", records(keys), conn)
    return len(keys)


def ensureVisits(qadb: Any, visits: Iterable[int], opdb: Any = None) -> int:
    """Make sure qadb's ``pfs_visit`` has a row for each visit, which the QA tables reference.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The database.
    visits : iterable of `int`
        The visits.
    opdb : `pfs.utils.database.opdb.OpDB`, optional
        When given, each visit's description, design and issue time are copied
        from the opdb's ``pfs_visit`` (one query) and replace what qadb holds.
        Otherwise a visit qadb lacks gets a row with its id alone, and one it
        has is left as it is.

    Returns
    -------
    `int`
        Rows written.
    """
    visits = sorted({int(visit) for visit in visits})
    if not visits:
        return 0
    if opdb is None:
        frame = pd.DataFrame({"pfs_visit_id": visits}, dtype=object)
        return upsert(qadb, "pfs_visit", frame, ["pfs_visit_id"])
    known = opdb.query_dataframe(_PFS_VISIT_SQL, params={"visits": visits})
    known = known.reindex(columns=list(_PFS_VISIT_COLUMNS)).astype(object)
    written = upsert(qadb, "pfs_visit", known, ["pfs_visit_id"])
    unknown = sorted(set(visits) - {int(v) for v in known["pfs_visit_id"]})
    if unknown:
        written += upsert(
            qadb, "pfs_visit", pd.DataFrame({"pfs_visit_id": unknown}, dtype=object), ["pfs_visit_id"]
        )
    return written


def processingId(
    qadb: Any,
    collection: str,
    run: str,
    drpQaVersion: str | None,
    pipeline: Mapping[str, str | None] | None = None,
) -> int:
    """Return the ``processing_id`` of a collection and Butler RUN, making the row if needed.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The database.
    collection : `str`
        The collection loaded (``data_processing_pipe2d.collection_name``).
    run : `str`
        The Butler RUN holding the QA tables (``run_name``).
    drpQaVersion : `str` or `None`
        The drp_qa that wrote them.
    pipeline : mapping [`str`, `str`], optional
        The reduction's versions by product (`PIPELINE_VERSION_COLUMNS`).

    Returns
    -------
    `int`
        The id. A version given here replaces the existing row's; one not
        given (`None`, or `UNKNOWN_VERSION` for drp_qa) leaves it, so a later
        load that did not read the versions keeps what an earlier one recorded.
    """
    pipeline = pipeline or {}
    params = {
        "collection_name": collection,
        "run_name": run,
        "drp_qa_version": None if drpQaVersion in (None, UNKNOWN_VERSION) else drpQaVersion,
        **{column: pipeline.get(product) for product, column in PIPELINE_VERSION_COLUMNS.items()},
    }
    columns = list(params)
    keys = ["collection_name", "run_name"]
    updates = ", ".join(
        f"{c} = COALESCE(EXCLUDED.{c}, data_processing_pipe2d.{c})" for c in columns if c not in keys
    )
    sql = (
        f"INSERT INTO data_processing_pipe2d ({', '.join(columns)}) VALUES ({', '.join(f':{c}' for c in columns)})"
        f" ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {updates} RETURNING processing_id"
    )
    found = qadb.query_scalar(sql, params=params)
    if found is None:
        raise RuntimeError(f"No processing_id returned for {collection!r} run {run!r}")
    return int(found)


def sha256Of(path: Path | str) -> str:
    """Return the hex SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registerThresholds(qadb: Any, path: Path | str, registeredAt: datetime | None = None) -> int:
    """Register a thresholds file, returning its ``thresholds_id``.

    The file is identified by its content: registering the same thresholds
    again, under any name, returns the existing id and leaves
    ``registered_at`` as it was. Its entries are written to
    ``qa_threshold_entry``.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The database.
    path : `pathlib.Path` or `str`
        The file, as `pfs.drp.qa.metrics.calibration.writeThresholds` writes it.
    registeredAt : `datetime.datetime`, optional
        Default now.

    Returns
    -------
    `int`
        The id.
    """
    path = Path(path)
    digest = sha256Of(path)
    metadata, thresholds = readThresholds(path)
    header, _ = thresholdRows(metadata, thresholds, 0, path.name, digest, registeredAt or datetime.now())
    columns = [column for column in QA_THRESHOLDS if column != "thresholds_id"]
    params = records(header[columns])[0]
    sql = (
        f"INSERT INTO qa_thresholds ({', '.join(columns)}) VALUES ({', '.join(f':{c}' for c in columns)})"
        " ON CONFLICT (sha256) DO UPDATE SET name = EXCLUDED.name RETURNING thresholds_id"
    )
    found = qadb.query_scalar(sql, params=params)
    if found is None:
        raise RuntimeError(f"No thresholds_id returned for {path}")
    thresholdsId = int(found)
    _, entries = thresholdRows(metadata, thresholds, thresholdsId, path.name, digest, registeredAt)
    upsert(qadb, "qa_threshold_entry", entries, list(QA_THRESHOLD_ENTRY[:3]))
    return thresholdsId
