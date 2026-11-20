"""Read the QA numbers, verdicts and normal values qadb holds, in the shapes the gate reads.

The inverse of `pfs.drp.qa.qadb.tables`: what `pfs.drp.qa.qadb.loader` wrote for a
collection comes back as ``iqQaMetrics``, ``dmLineResidualSummary``,
``dmLineResidualMetrics`` and the twilight metrics, for readers without a Butler:
the commands that judge stored numbers again and make normal values
(`pfs.drp.qa.qadb.judge`), and the QA dashboard (``pfs_qa_dashboard``).

Read-only: one query per table, with bound parameters. An image measured in more
than one Butler RUN of the collection is read, table by table, as the newest
RUN that wrote that table: a task rerun alone into a newer RUN shows its new
rows beside the other task's older ones. ``since`` is the image's latest gate row.
Visit times are qadb's ``pfs_visit.issued_at`` (HST); qadb holds no IIC sequence.
"""

from typing import Any

import pandas as pd

from pfs.drp.qa.qadb.tables import (
    IMAGE_KEYS,
    IMAGE_QUALITY,
    LINE_RESIDUAL,
    LINE_RESIDUAL_METRIC,
    NORMAL_VALUE,
    QA_GATE,
    TWILIGHT_QA,
    TableSpec,
    normalValueFrame,
)

__all__ = [
    "FIRST_VISIT",
    "GATE_COLUMNS",
    "LAST_VISIT",
    "PROCESSING_ID",
    "collectionLoaded",
    "readGate",
    "readNormalValues",
    "readQadb",
    "readTwilight",
    "visitsJudged",
]

#: The widest visit range: every visit.
FIRST_VISIT, LAST_VISIT = 0, 2**31 - 1

#: The column of `readQadb`'s tables holding each row's processing (``data_processing_pipe2d``).
PROCESSING_ID = "processingId"

_SPECS = {spec.source: spec for spec in (IMAGE_QUALITY, LINE_RESIDUAL, LINE_RESIDUAL_METRIC, TWILIGHT_QA)}


def _camelCase(name: str) -> str:
    """Return a snake_case column as the camelCase the Butler tables use."""
    first, *rest = name.split("_")
    return first + "".join(part.capitalize() for part in rest)


# Each image's latest gate row in the collection: its key and when it was decided.
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

# Each image's latest gate row in the collection, whole.
_GATE_SQL = """
WITH latest AS (
    SELECT g.*, p.run_name,
        ROW_NUMBER() OVER (
            PARTITION BY g.pfs_visit_id, g.camera_name ORDER BY g.processing_id DESC
        ) AS n
    FROM qa_gate g JOIN data_processing_pipe2d p ON p.processing_id = g.processing_id
    WHERE p.collection_name = :collection AND g.pfs_visit_id BETWEEN :first AND :last
)
SELECT * FROM latest WHERE n = 1 ORDER BY pfs_visit_id, camera_name
"""

_COLLECTION_SQL = "SELECT COUNT(*) FROM data_processing_pipe2d WHERE collection_name = :collection"

_VISITS_JUDGED_SQL = """
SELECT DISTINCT g.pfs_visit_id FROM qa_gate g JOIN data_processing_pipe2d p ON p.processing_id = g.processing_id
WHERE p.collection_name = :collection AND g.pfs_visit_id BETWEEN :first AND :last ORDER BY g.pfs_visit_id
"""

#: Columns of `readGate`: the image, then ``qa_gate`` in camelCase, then the Butler RUN it was measured in.
GATE_COLUMNS = (
    *IMAGE_KEYS,
    *(_camelCase(c) for c in QA_GATE if c not in ("processing_id", "pfs_visit_id", "camera_name")),
    "run",
    PROCESSING_ID,
)

_VISITS_SQL = """
SELECT pfs_visit_id, issued_at AS time_exp_start FROM pfs_visit WHERE pfs_visit_id = ANY(:visits)
"""

_NORMAL_SQL = (
    "SELECT * FROM qa_normal_value WHERE collection_name = :collection"
    " ORDER BY valid_from, metric, camera, frame_type, selector"
)


def _tableSql(table: str, since: bool) -> str:
    # The table name is _SPECS', never input. Each image's rows come from the newest processing of the
    # collection that wrote this table for it; the gate row (latest) gives decided_at and the since filter.
    return (
        _LATEST + f""", newest AS (
    SELECT t.pfs_visit_id, t.camera_name, MAX(t.processing_id) AS processing_id
    FROM {table} t JOIN data_processing_pipe2d p ON p.processing_id = t.processing_id
    WHERE p.collection_name = :collection AND t.pfs_visit_id BETWEEN :first AND :last
    GROUP BY t.pfs_visit_id, t.camera_name
)
SELECT t.*, l.decided_at FROM {table} t"""
        " JOIN newest w ON w.processing_id = t.processing_id"
        " AND w.pfs_visit_id = t.pfs_visit_id AND w.camera_name = t.camera_name"
        " JOIN latest l ON l.pfs_visit_id = t.pfs_visit_id AND l.camera_name = t.camera_name WHERE l.n = 1"
        + (" AND l.decided_at >= :since" if since else "")
    )


def _butlerFrame(rows: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Return qadb rows of ``spec`` with the Butler table's column names and types."""
    names = {spec.qadbColumn(column): column for column in (*spec.extraKeys, *spec.columns)}
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
            frame[column] = frame[column].map(_nullableBool)
    return frame.rename(columns={"processing_id": PROCESSING_ID}).drop(columns=["camera_name"])


def _nullableBool(value: Any) -> bool | None:
    """Return a stored boolean as `bool`, a null as `None`: SQLite gives 0/1, and a null must stay missing."""
    return None if pd.isna(value) else bool(value)


def readQadb(
    qadb: Any,
    collection: str,
    first: int = FIRST_VISIT,
    last: int = LAST_VISIT,
    since: Any = None,
) -> dict[str, pd.DataFrame]:
    """Read a collection's QA numbers from qadb, as the tables they were loaded from.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The qadb, read-only (``query_dataframe`` with ``:name`` parameters).
    collection : `str`
        The collection the QA tasks wrote to, as loaded (``qaActor/reductions``).
    first, last : `int`
        The visit range, inclusive.
    since : optional
        Only images whose latest gate row was decided at or after this
        (``qa_gate.decided_at``, as a previous read returned it); default all.

    Returns
    -------
    `dict` [`str`, `pandas.DataFrame`]
        ``iqQaMetrics``, ``dmLineResidualSummary``, ``dmLineResidualMetrics``
        and ``twilightMetrics`` with their columns (those qadb stores),
        `PROCESSING_ID` and ``decidedAt``; the line-residual rows carry their
        image's ``cameraSet``, ``obsType`` and ``seqType``, as the Butler table does, so
        `pfs.drp.qa.metrics.gate.gate` reads them as they are. And
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
    keys = list(IMAGE_KEYS)
    summary = tables["dmLineResidualSummary"]
    tables["dmLineResidualMetrics"] = tables["dmLineResidualMetrics"].merge(
        summary[[*keys, "cameraSet", "obsType", "seqType"]].drop_duplicates(keys), on=keys, how="left"
    )
    visits = sorted({int(visit) for table in tables.values() for visit in table["visit"]})
    times = (
        qadb.query_dataframe(_VISITS_SQL, params={"visits": visits})
        if visits
        else pd.DataFrame(columns=["pfs_visit_id", "time_exp_start"])
    )
    times["time_exp_start"] = pd.to_datetime(times["time_exp_start"])
    tables["visitTimes"] = times.assign(iic_sequence_id=pd.array([pd.NA] * len(times), dtype="Int64"))
    return tables


def readTwilight(qadb: Any, collection: str, first: int, last: int) -> pd.DataFrame:
    """Read the twilight metrics qadb holds for a visit range of a collection, as `readQadb` returns them."""
    params = {"collection": collection, "first": int(first), "last": int(last)}
    rows = qadb.query_dataframe(_tableSql(TWILIGHT_QA.name, False), params=params)
    return _butlerFrame(rows, TWILIGHT_QA).rename(columns={"decided_at": "decidedAt"}).reset_index(drop=True)


def visitsJudged(qadb: Any, collection: str, first: int = FIRST_VISIT, last: int = LAST_VISIT) -> list[int]:
    """Return the visits of a collection with a gate row in a visit range, sorted."""
    params = {"collection": collection, "first": int(first), "last": int(last)}
    rows = qadb.query_dataframe(_VISITS_JUDGED_SQL, params=params)
    return [int(visit) for visit in rows["pfs_visit_id"]]


def readNormalValues(qadb: Any, collection: str) -> pd.DataFrame:
    """Read a collection's normal values.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The qadb, read-only.
    collection : `str`
        The collection they judge.

    Returns
    -------
    `pandas.DataFrame`
        `pfs.drp.qa.metrics.normalValues.NORMAL_COLUMNS`, every accepted row
        of the collection: `pfs.drp.qa.metrics.gate.gate` takes the one in
        force for each image.
    """
    rows = qadb.query_dataframe(_NORMAL_SQL, params={"collection": collection})
    frame = normalValueFrame(rows.reindex(columns=list(NORMAL_VALUE)))
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").astype("float64")
    for column in ("count", "validFrom"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("int64")
    for column in ("frameType", "selector"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def collectionLoaded(qadb: Any, collection: str) -> bool:
    """Return whether qadb holds any processing of a collection.

    Tells a collection nothing was loaded for (a mistyped name) from one with
    no verdicts in a visit range, which read back alike.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The qadb, read-only.
    collection : `str`
        The collection, as loaded.
    """
    return bool(qadb.query_scalar(_COLLECTION_SQL, params={"collection": collection}))


def readGate(qadb: Any, collection: str, first: int = FIRST_VISIT, last: int = LAST_VISIT) -> pd.DataFrame:
    """Read each image's gate verdict in a collection.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The qadb, read-only.
    collection : `str`
        The collection the QA tasks wrote to, as loaded.
    first, last : `int`
        The visit range, inclusive.

    Returns
    -------
    `pandas.DataFrame`
        `GATE_COLUMNS`, one row per image, sorted by visit and camera: the row
        of the highest processing (Butler RUN, ``run``, and `PROCESSING_ID`) of
        the collection that judged the image. `pfs.drp.qa.metrics.visitRollup.rollupVisits`
        takes it.
    """
    params = {"collection": collection, "first": int(first), "last": int(last)}
    rows = qadb.query_dataframe(_GATE_SQL, params=params)
    names = {column: _camelCase(column) for column in QA_GATE if column != "camera_name"}
    frame = rows.rename(columns={**names, "pfs_visit_id": "visit", "run_name": "run"})
    frame[PROCESSING_ID] = pd.to_numeric(frame[PROCESSING_ID]).astype("int64")
    camera = frame["camera_name"].astype(str)
    frame["arm"] = camera.str[0]
    frame["spectrograph"] = pd.to_numeric(camera.str[1:]).astype("int64")
    frame["visit"] = frame["visit"].astype("int64")
    # A database that returns times as text writes the fraction of a second only when there is one.
    frame["decidedAt"] = pd.to_datetime(frame["decidedAt"], format="mixed")
    return frame.reindex(columns=list(GATE_COLUMNS)).reset_index(drop=True)
