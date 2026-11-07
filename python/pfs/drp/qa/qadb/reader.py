"""Read the QA tables qadb holds back in the Butler tables' shapes.

The inverse of `pfs.drp.qa.qadb.tables`: what `pfs.drp.qa.qadb.loader` wrote for a
collection (by qaActor after each visit, or ``bin.src/loadQadb.py`` after a run)
comes back as ``iqQaMetrics``, ``dmLineResidualSummary`` and
``dmLineResidualMetrics``, for readers without a Butler (the QA dashboard,
``pfs_qa_dashboard``).

Each image comes back with the thresholds it was judged with. A task judges each
metric by the first of its layers with an entry for the image's population: its
thresholds files, then its config's threshold fields. ``qa_gate_thresholds``
cites every layer in that order (``thresholdsIds``, the ids joined by commas;
`thresholdsLayers` splits them). The cited layers' entries (``thresholds``) and
headers (``thresholdsFiles``, a config layer named by
`pfs.drp.qa.metrics.gate.configLayerName`) are read back too, so a reader gives
`pfs.drp.qa.metrics.gate.judge` the layers the image was judged by
(`layeredThresholds`), whatever this drp_qa ships, rather than ones rebuilt from
today's files and defaults.

Read-only: one query per table, with bound parameters. An image judged in more
than one Butler RUN of the collection is read as the latest RUN judged it.
Visit times are qadb's ``pfs_visit.issued_at`` (HST); qadb holds no IIC sequence.
"""

import re
from typing import Any

import pandas as pd

from pfs.drp.qa.metrics.gate import THRESHOLD_FIELDS
from pfs.drp.qa.qadb.tables import (
    IMAGE_QUALITY,
    LINE_RESIDUAL,
    LINE_RESIDUAL_METRIC,
    QA_THRESHOLD_ENTRY,
    QA_THRESHOLDS,
    TableSpec,
    cameraName,
    snakeCase,
)

__all__ = [
    "FIRST_VISIT",
    "LAST_VISIT",
    "THRESHOLDS_COLUMNS",
    "THRESHOLDS_FILES_COLUMNS",
    "THRESHOLDS_ID",
    "THRESHOLDS_IDS",
    "THRESHOLDS_LAYERS_COLUMNS",
    "layeredThresholds",
    "readQadb",
    "thresholdsLayers",
    "thresholdsTables",
]

#: The widest visit range: every visit.
FIRST_VISIT, LAST_VISIT = 0, 2**31 - 1

#: The id column of ``thresholds``, ``thresholdsFiles`` and ``thresholdsLayers``.
THRESHOLDS_ID = "thresholdsId"

#: The column of every table read back naming every layer its image's task judged with: the ids in the order
#: judged, joined by commas (`thresholdsLayers`); empty when none.
THRESHOLDS_IDS = "thresholdsIds"

#: Columns of ``thresholdsLayers``: one row per image, task and layer cited.
THRESHOLDS_LAYERS_COLUMNS = ("visit", "arm", "spectrograph", "task", "layer", THRESHOLDS_ID)

#: The tables read back, by Butler dataset type.
_SPECS = {spec.source: spec for spec in (IMAGE_QUALITY, LINE_RESIDUAL, LINE_RESIDUAL_METRIC)}

#: The task whose layers ``qa_gate_thresholds`` holds for each table read back.
_GATE_TASK = {
    IMAGE_QUALITY.source: "imageQualityQa",
    LINE_RESIDUAL.source: "dmLineResiduals",
    LINE_RESIDUAL_METRIC.source: "dmLineResiduals",
}


def _camelCase(name: str) -> str:
    """Return a snake_case column as the camelCase the Butler tables and thresholds files use."""
    first, *rest = name.split("_")
    return first + "".join(part.capitalize() for part in rest)


#: Population columns ``qa_threshold_entry`` stores in their own columns, camelCase, in its order.
_STORED_POPULATION = tuple(
    _camelCase(column)
    for column in QA_THRESHOLD_ENTRY
    if column not in ("thresholds_id", "metric", "population") and _camelCase(column) not in THRESHOLD_FIELDS
)

#: The threshold fields, camelCase, in ``qa_threshold_entry``'s order.
_FIELDS = tuple(
    _camelCase(column)
    for column in QA_THRESHOLD_ENTRY
    if column != "metric" and _camelCase(column) in THRESHOLD_FIELDS
)

#: Columns of ``thresholds``: the id, the metric, the stored population key, the population columns it
#: stores (the keys it only names in the key, such as ``species``, follow), then the threshold fields.
THRESHOLDS_COLUMNS = (THRESHOLDS_ID, "metric", "population", *_STORED_POPULATION, *_FIELDS)

#: Columns of ``thresholdsFiles``: ``qa_thresholds`` in camelCase (``derivedOn``, ``referenceRuns``, ...).
THRESHOLDS_FILES_COLUMNS = tuple(
    THRESHOLDS_ID if c == "thresholds_id" else _camelCase(c) for c in QA_THRESHOLDS
)

_BOOL_FIELDS = ("higherIsWorse", "absolute", "reliable", "failBounded", "degenerate")
_INT_FIELDS = ("nGood", "nGoodVisits")
_FLOAT_FIELDS = ("warn", "fail")

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

_ENTRIES_SQL = """
SELECT * FROM qa_threshold_entry WHERE thresholds_id = ANY(:ids) ORDER BY thresholds_id, metric, population
"""

_FILES_SQL = "SELECT * FROM qa_thresholds WHERE thresholds_id = ANY(:ids) ORDER BY thresholds_id"

_LAYERS_SQL = (
    _LATEST
    + "SELECT l.pfs_visit_id, l.camera_name, t.task, t.layer, t.thresholds_id FROM qa_gate_thresholds t"
    " JOIN latest l ON l.processing_id = t.processing_id"
    " AND l.pfs_visit_id = t.pfs_visit_id AND l.camera_name = t.camera_name WHERE l.n = 1{since}"
    " ORDER BY l.pfs_visit_id, l.camera_name, t.task, t.layer"
)


def _tableSql(table: str, since: bool) -> str:
    # The table name is _SPECS', never input.
    return (
        _LATEST + f"SELECT t.*, l.decided_at FROM {table} t"
        " JOIN latest l ON l.processing_id = t.processing_id"
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
            frame[column] = frame[column].map(_nullableBool)
    return frame.drop(columns=["processing_id", "camera_name"])


def _nullableBool(value: Any) -> bool | None:
    """Return a stored boolean as `bool`, a null as `None`: SQLite gives 0/1, and a null must stay missing."""
    return None if pd.isna(value) else bool(value)


def _populationValue(value: Any) -> Any:
    """Return a population value as the files hold it: an integral number as `int`, a null as `None`."""
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def _parsePopulation(key: Any) -> dict[str, Any]:
    """Return the population a stored key names (`pfs.drp.qa.qadb.tables.populationKey`): ``arm=b species=Neon``.

    A value may hold spaces (``seqName=Arc: Neon``): the key splits before each ``name=``.
    """
    if not isinstance(key, str) or not key:
        return {}
    pairs = (part.split("=", 1) for part in re.split(r" (?=[A-Za-z_]\w*=)", key))
    return {name: _populationValue(value) for name, value in pairs}


def _entriesFrame(rows: pd.DataFrame) -> pd.DataFrame:
    """Return ``qa_threshold_entry`` rows as `THRESHOLDS_COLUMNS`, typed as the thresholds files are read.

    The stored population columns come back typed; a population key the table
    has no column for (``species``, ``seqName``) is parsed from ``population``
    into a column of its own, so the entries match images as the file's did.
    """
    frame = rows.reindex(columns=list(QA_THRESHOLD_ENTRY)).rename(
        columns={column: _camelCase(column) for column in QA_THRESHOLD_ENTRY}
    )
    populations = [_parsePopulation(key) for key in frame["population"]]
    extra = sorted({name for population in populations for name in population} - set(_STORED_POPULATION))
    for column in _STORED_POPULATION:
        values = [_populationValue(value) for value in frame[column].tolist()]
        frame[column] = pd.Series(values, index=frame.index, dtype=object)
    for column in extra:
        values = [population.get(column) for population in populations]
        frame[column] = pd.Series(values, index=frame.index, dtype=object)
    for column in _BOOL_FIELDS:
        values = [_nullableBool(value) for value in frame[column].tolist()]
        frame[column] = pd.Series(values, index=frame.index, dtype=object)
    for column in _FLOAT_FIELDS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in _INT_FIELDS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame[THRESHOLDS_ID] = frame[THRESHOLDS_ID].astype("int64")
    order = [THRESHOLDS_ID, "metric", "population", *_STORED_POPULATION, *extra, *_FIELDS]
    return frame[order].reset_index(drop=True)


def _filesFrame(rows: pd.DataFrame) -> pd.DataFrame:
    """Return ``qa_thresholds`` rows as `THRESHOLDS_FILES_COLUMNS`."""
    frame = rows.reindex(columns=list(QA_THRESHOLDS)).rename(
        columns=dict(zip(QA_THRESHOLDS, THRESHOLDS_FILES_COLUMNS, strict=True))
    )
    frame[THRESHOLDS_ID] = frame[THRESHOLDS_ID].astype("int64")
    for column in ("derivedOn", "registeredAt"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["nEntries"] = pd.to_numeric(frame["nEntries"], errors="coerce").astype("Int64")
    return frame.reset_index(drop=True)


def thresholdsLayers(key: Any) -> list[int]:
    """Return the ``thresholds_id`` of each layer a ``thresholdsIds`` value names, in the order judged.

    Parameters
    ----------
    key : `str` or null
        A `THRESHOLDS_IDS` value: ``"12,7"``. Empty or null names none.
    """
    if not isinstance(key, str) or not key:
        return []
    return [int(part) for part in key.split(",") if part]


def layeredThresholds(key: Any, tables: dict[int, pd.DataFrame]) -> list[pd.DataFrame]:
    """Return the thresholds tables of the layers a ``thresholdsIds`` value names, in the order judged.

    What `pfs.drp.qa.metrics.gate.judge` takes to judge an image as its task
    did: each metric by the first table with an entry for the image.

    Parameters
    ----------
    key : `str` or null
        A `THRESHOLDS_IDS` value.
    tables : `dict` [`int`, `pandas.DataFrame`]
        `thresholdsTables` of the read's ``thresholds``.

    Returns
    -------
    `list` [`pandas.DataFrame`]
        The tables; a layer ``tables`` lacks is left out.
    """
    return [tables[thresholdsId] for thresholdsId in thresholdsLayers(key) if thresholdsId in tables]


def _layersFrame(rows: pd.DataFrame) -> pd.DataFrame:
    """Return ``qa_gate_thresholds`` rows as `THRESHOLDS_LAYERS_COLUMNS`."""
    camera = rows["camera_name"].astype(str)
    return pd.DataFrame(
        {
            "visit": rows["pfs_visit_id"].astype("int64").to_numpy(),
            "arm": camera.str[0].to_numpy(),
            "spectrograph": camera.str[1:].astype("int64").to_numpy(),
            "task": rows["task"].astype(str).to_numpy(),
            "layer": rows["layer"].astype("int64").to_numpy(),
            THRESHOLDS_ID: rows["thresholds_id"].astype("int64").to_numpy(),
        },
        columns=list(THRESHOLDS_LAYERS_COLUMNS),
    )


def _layerKeys(layers: pd.DataFrame) -> dict[tuple[int, str, str], str]:
    """Return each image and task's `THRESHOLDS_IDS` value from its ``thresholdsLayers`` rows."""
    keys: dict[tuple[int, str, str], str] = {}
    ordered = layers.sort_values(["visit", "arm", "spectrograph", "task", "layer"], kind="stable")
    for (visit, arm, spectrograph, task), group in ordered.groupby(
        ["visit", "arm", "spectrograph", "task"], sort=False
    ):
        keys[(int(visit), cameraName(arm, spectrograph), str(task))] = ",".join(
            str(int(thresholdsId)) for thresholdsId in group[THRESHOLDS_ID]
        )
    return keys


def thresholdsTables(thresholds: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return each cited file's entries as a thresholds table `pfs.drp.qa.metrics.gate.judge` takes, by id.

    `judge` takes every column but the threshold fields for a population
    column, so ``thresholdsId`` and the stored ``population`` key go, as do the
    population columns the file never names (null throughout: "any" to the
    gate, but noise to a reader). Population values are as the files hold them
    (a spectrograph `int`, a null `None`), whatever a parquet round trip made of
    them.

    Parameters
    ----------
    thresholds : `pandas.DataFrame`
        `readQadb`'s ``thresholds``.

    Returns
    -------
    `dict` [`int`, `pandas.DataFrame`]
        By ``thresholdsId``: the file's entries, one row per metric and
        population, as `pfs.drp.qa.metrics.calibration.readThresholds` returns
        them.
    """
    tables = {}
    for thresholdsId, rows in thresholds.groupby(THRESHOLDS_ID, sort=True):
        table = rows.drop(columns=[THRESHOLDS_ID, "population"]).reset_index(drop=True)
        for column in [c for c in table.columns if c not in THRESHOLD_FIELDS]:
            values = [_populationValue(value) for value in table[column].tolist()]
            table[column] = pd.Series(values, index=table.index, dtype=object)
        unnamed = [c for c in table.columns if c not in THRESHOLD_FIELDS and table[c].isna().all()]
        tables[int(thresholdsId)] = table.drop(columns=unnamed)
    return tables


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
        with their Butler columns (those qadb stores), ``decidedAt``, and
        ``thresholdsIds``, every thresholds layer the image's task judged with
        in the order judged (`THRESHOLDS_IDS`, a string, empty when none;
        `layeredThresholds` gives `judge` their tables); ``thresholdsLayers``,
        one row per image, task and layer (`THRESHOLDS_LAYERS_COLUMNS`);
        ``thresholds``, the entries of every layer cited by the images read
        (`THRESHOLDS_COLUMNS`; `thresholdsTables` of it, by id);
        ``thresholdsFiles``, one row per layer (`THRESHOLDS_FILES_COLUMNS`);
        and ``visitTimes``, qadb's visit listing in the opdb listing's columns
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
    layers = _layersFrame(
        qadb.query_dataframe(
            _LAYERS_SQL.format(since=" AND l.decided_at >= :since" if since else ""), params=params
        )
    )
    keys = _layerKeys(layers)
    for source, frame in tables.items():
        task = _GATE_TASK[source]
        frame[THRESHOLDS_IDS] = [
            keys.get((int(visit), cameraName(arm, spectrograph), task), "")
            for visit, arm, spectrograph in frame[["visit", "arm", "spectrograph"]].itertuples(
                index=False, name=None
            )
        ]
    ids = sorted({int(i) for i in layers[THRESHOLDS_ID]})
    tables["thresholdsLayers"] = layers
    if ids:
        entries = qadb.query_dataframe(_ENTRIES_SQL, params={"ids": ids})
        files = qadb.query_dataframe(_FILES_SQL, params={"ids": ids})
    else:
        entries = pd.DataFrame(columns=list(QA_THRESHOLD_ENTRY))
        files = pd.DataFrame(columns=list(QA_THRESHOLDS))
    tables["thresholds"] = _entriesFrame(entries)
    tables["thresholdsFiles"] = _filesFrame(files)
    visits = sorted({int(visit) for visit in tables["iqQaMetrics"]["visit"]})
    times = (
        qadb.query_dataframe(_VISITS_SQL, params={"visits": visits})
        if visits
        else pd.DataFrame(columns=["pfs_visit_id", "time_exp_start"])
    )
    times["time_exp_start"] = pd.to_datetime(times["time_exp_start"])
    tables["visitTimes"] = times.assign(iic_sequence_id=pd.array([pd.NA] * len(times), dtype="Int64"))
    return tables
