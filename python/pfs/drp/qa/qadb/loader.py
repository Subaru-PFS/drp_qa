"""Read a collection's QA tables from a Butler and load them into qadb.

`loadCollection` is the one entry point: qaActor calls it for a visit as soon
as its pipetask has run (`loadVisit`), comparison mode for a run, and
``bin.src/loadQadb.py`` for a backfill. Each Butler RUN found becomes a
``data_processing_pipe2d`` row, with the drp_qa that wrote it (from the RUN's
``packages`` dataset) and the pipeline that reduced the images it judged (from
their ``pfsArm`` headers, as comparison mode reads them); the thresholds layers
each image was judged with are registered and cited, in order, in
``qa_gate_thresholds``, and its gate verdict is written to ``qa_gate``.

The tasks record those layers with each image, by name and SHA-256 as they were
at judgement time (`pfs.drp.qa.metrics.gate.thresholdsRecord`), so what the
gate cites does not depend on the drp_qa that loads: its thresholds files, then
its config's threshold fields as a document of their own
(`pfs.drp.qa.metrics.gate.thresholdsDocument`, named by
`pfs.drp.qa.metrics.gate.configLayerName`). A file is registered from the copy
this drp_qa ships with the recorded content; the config layer is rebuilt from
the RUN's stored task config (`CONFIG_LAYERS`) and registered when it hashes as
recorded, so a layer this drp_qa would build differently is left out, and
logged, rather than misattributed. A table written before the tasks recorded
anything has its RUN's stored config read instead: through today's config class,
with today's files under those names, and its fields as the config layer.
"""

import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pfs.drp.qa.metrics.gate import (
    THRESHOLDS_COLUMNS,
    configLayerName,
    configThresholds,
    isConfigLayer,
    recordedThresholds,
    sha256Of,
    thresholdsDocument,
    thresholdsNames,
    thresholdsPath,
)
from pfs.drp.qa.metrics.lineResiduals import residualConfigThresholds
from pfs.drp.qa.metrics.readers import readMetrics, readReductionPipelines
from pfs.drp.qa.qadb.tables import (
    IMAGE_KEYS,
    IMAGE_QUALITY,
    IMAGE_QUALITY_SPECIES,
    KEYS,
    LINE_RESIDUAL,
    LINE_RESIDUAL_METRIC,
    QA_GATE,
    QA_GATE_THRESHOLDS,
    TASK_TABLES,
    TWILIGHT_QA,
    TableSpec,
    gateRows,
    gateThresholdsRows,
    imageRows,
)
from pfs.drp.qa.qadb.writer import (
    UNKNOWN_VERSION,
    deleteRows,
    ensureVisits,
    processingId,
    registerThresholds,
    registerThresholdsContent,
    upsert,
)
from pfs.drp.qa.utils.provenance import commonPipeline

__all__ = [
    "CONFIG_LAYERS",
    "MODES",
    "SPECS",
    "collectionExists",
    "configDocumentOf",
    "drpQaVersionOf",
    "loadCollection",
    "loadVisit",
    "readTables",
    "thresholdsFilesOf",
]

#: How each task's config threshold fields become its last thresholds layer: the table the task judges with
#: after its files, built from its config as the task builds it. ``twilightQa`` has no config layer.
CONFIG_LAYERS: dict[str, Callable[[Any], pd.DataFrame]] = {
    "imageQualityQa": configThresholds,
    "dmLineResiduals": residualConfigThresholds,
}

#: How a load came about: qaActor as the run goes, comparison mode over a run, or later.
MODES = ("realtime", "comparison", "backfill")

#: The qadb table of each Butler dataset type loaded.
SPECS: dict[str, TableSpec] = {
    spec.source: spec
    for spec in (IMAGE_QUALITY, IMAGE_QUALITY_SPECIES, LINE_RESIDUAL, LINE_RESIDUAL_METRIC, TWILIGHT_QA)
}


def collectionExists(butler: Any, collection: str) -> bool:
    """Say whether a Butler has a collection of this exact name.

    A missing collection would otherwise surface deep inside the first
    registry query as a long traceback.
    """
    try:
        found = butler.collections.query(collection)
    except Exception as error:  # daf_butler raises MissingCollectionError for an unknown exact name
        if type(error).__name__ == "MissingCollectionError":
            return False
        raise
    return collection in set(found)


def readTables(
    butler: Any,
    collection: str,
    visits: Iterable[int] | None = None,
    visitRange: tuple[int, int] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, pd.DataFrame]:
    """Read every dataset type of `SPECS` from a collection, with each row's Butler RUN.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A Butler; read-only is enough.
    collection : `str`
        The collection.
    visits, visitRange
        Which visits; see `pfs.drp.qa.metrics.readers.readMetrics`.
    log : callable, optional
        Where progress goes.

    Returns
    -------
    `dict` [`str`, `pandas.DataFrame`]
        By dataset type; a type the collection lacks gives an empty table.
    """
    tables = {}
    for datasetType in SPECS:
        try:
            tables[datasetType] = readMetrics(
                butler,
                visits,
                datasetType,
                collections=[collection],
                visitRange=visitRange,
                progress=True,
                withRun=True,
                log=log,
            )
        except LookupError:
            # Not a problem by itself: a task that did not run on these visits (twilightFluxQa on a science
            # visit, say) leaves none.
            log(f"{datasetType}: none for these visits in {collection}")
            tables[datasetType] = pd.DataFrame(columns=[*IMAGE_KEYS, "run"])
    return tables


def drpQaVersionOf(butler: Any, run: str) -> str:
    """Return the drp_qa version that wrote a RUN, from its ``packages`` dataset.

    pipetask records the versions of every product set up when it ran.
    `UNKNOWN_VERSION` when the RUN has no readable ``packages``: the loader's
    own version would be the wrong answer, and `processingId` never
    overwrites a recorded version with it.
    """
    try:
        packages = butler.get("packages", collections=[run])
        version = packages.get("drp_qa") if hasattr(packages, "get") else packages["drp_qa"]
        if version:
            return str(version)
    except Exception:  # any Butler failure means the RUN recorded none
        pass
    return UNKNOWN_VERSION


def thresholdsFilesOf(butler: Any, run: str, task: str) -> list[Path]:
    """Return the thresholds files a RUN's stored config for a task names, as this drp_qa ships them.

    The config's ``thresholdsFile`` first, then its ``extraThresholdsFiles``
    (``imageQualityQa``'s further layers, judged after it). Empty when the RUN
    has no config for the task, or it names no file. Only for tables that
    record none themselves: the config is read through today's config class,
    so a field added since has today's default.
    """
    try:
        config = butler.get(f"{task}_config", collections=[run])
    except Exception:  # a RUN without the task has no config
        return []
    return [thresholdsPath(name) for name in thresholdsNames(config)]


def configDocumentOf(butler: Any, run: str, task: str) -> bytes | None:
    """Return a RUN's stored config for a task as the task's config thresholds layer.

    The table `CONFIG_LAYERS` builds from the config, as a
    `pfs.drp.qa.metrics.gate.thresholdsDocument`: what the task recorded the
    SHA-256 of, when its drp_qa built the same table from the same fields.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A Butler.
    run : `str`
        The RUN, whose ``<task>_config`` is read.
    task : `str`
        The task.

    Returns
    -------
    `bytes` or `None`
        The document; `None` when the task has no config layer
        (``twilightQa``), the RUN has no config for it, or the config lacks
        the threshold fields (read through today's config class, a config
        stored by a drp_qa without them has none).
    """
    build = CONFIG_LAYERS.get(task)
    if build is None:
        return None
    try:
        config = butler.get(f"{task}_config", collections=[run])
    except Exception:  # a RUN without the task has no config
        return None
    try:
        table = build(config)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return thresholdsDocument(table, configLayerName(task))


def _thresholdsIdsOf(
    qadb: Any, butler: Any, run: str, task: str, now: datetime, log: Callable
) -> list[int | None]:
    """Register the thresholds layers a RUN's stored config gives; return their ids in the order judged.

    For tables written before the tasks recorded their layers. Every file the
    config names is registered, so each is in ``qa_thresholds`` by content,
    then the config's fields as the task's config layer (`configDocumentOf`).
    `None` in a file's place, with a log line, when this drp_qa no longer
    ships it.
    """
    log(
        f"{run}: {task} recorded no thresholds layers; citing those its stored config gives, as this drp_qa has them"
    )
    ids: list[int | None] = []
    for path in thresholdsFilesOf(butler, run, task):
        try:
            ids.append(registerThresholds(qadb, path, now))
        except FileNotFoundError:
            log(
                f"{run}: {task} judged with {path.name}, which this drp_qa no longer has; thresholds left null"
            )
            ids.append(None)
    document = configDocumentOf(butler, run, task)
    if document is not None:
        ids.append(registerThresholdsContent(qadb, document, configLayerName(task), now))
    elif task in CONFIG_LAYERS:
        log(f"{run}: {task}'s stored config gives no threshold fields; its config layer is not cited")
    return ids


def _shippedWith(name: str, digest: str) -> Path | None:
    """Return a thresholds file this drp_qa ships with a given content: under its name, or renamed."""
    shipped = thresholdsPath(name)
    for path in [shipped, *sorted(thresholdsPath("").glob("*.yaml"))]:
        if path.is_file() and sha256Of(path) == digest:
            return path
    return None


def _recordedId(
    qadb: Any,
    name: str,
    digest: str,
    now: datetime,
    log: Callable,
    rebuild: Callable[[], bytes | None] | None = None,
) -> int | None:
    """Return the ``thresholds_id`` of a layer an image records judging with, registering it if needed.

    By content: qadb's row with that SHA-256, else the same content found
    again and registered under the name judged with: for a file, a copy this
    drp_qa ships (`_shippedWith`); for a config layer, the document
    ``rebuild`` returns (`configDocumentOf` on the RUN's stored config) when it
    hashes as recorded. `None`, with a log line, when there is neither: the
    file changed or went before it was registered, or this drp_qa builds the
    config layer differently. A row with fewer entries than it counts (a
    registration that stopped between its two writes) is registered again
    when the content is at hand.
    """
    found = qadb.query_dataframe(
        "SELECT t.thresholds_id, t.n_entries,"
        " (SELECT count(*) FROM qa_threshold_entry e WHERE e.thresholds_id = t.thresholds_id) AS stored"
        " FROM qa_thresholds t WHERE t.sha256 = :sha256",
        params={"sha256": digest},
    )
    complete = found.empty or (
        pd.notna(found["n_entries"].iloc[0]) and found["stored"].iloc[0] >= found["n_entries"].iloc[0]
    )
    if not found.empty and complete:
        return int(found["thresholds_id"].iloc[0])
    content, why = None, "shipped with this drp_qa"
    if isConfigLayer(name):
        document = rebuild() if rebuild is not None else None
        if document is None:
            why = "given by the RUN's stored config"
        elif sha256Of(document) != digest:
            why = f"what this drp_qa builds from the RUN's stored config (sha256 {sha256Of(document)[:12]})"
        else:
            content = document
    else:
        path = _shippedWith(name, digest)
        content = path.read_bytes() if path is not None else None
    if content is None and not found.empty:
        log(
            f"thresholds {name} (sha256 {digest[:12]}) has entries missing in qadb, and is not {why} to restore them"
        )
        return int(found["thresholds_id"].iloc[0])
    if content is None:
        log(f"thresholds {name} (sha256 {digest[:12]}) is neither in qadb nor {why}; layer left null")
        return None
    return registerThresholdsContent(qadb, content, Path(name).name, now)


def _thresholdsIds(
    qadb: Any,
    butler: Any,
    task: str,
    judged: pd.DataFrame,
    registered: dict[tuple, int | None],
    now: datetime,
    log: Callable[[str], None],
) -> list[list[int | None]]:
    """Return the ``thresholds_id`` of each layer each image of a task was judged with, in the order judged.

    Every recorded layer is registered (``registered`` caches them across the
    load: a file by name and SHA-256, a config layer by those and the RUN,
    whose stored config rebuilds it); a layer that cannot be cited is `None`
    in its place. An image that records none falls back on its RUN's stored
    config (`_thresholdsIdsOf`); one whose record is unreadable gets no
    layers.
    """
    recorded = (
        judged[list(THRESHOLDS_COLUMNS)].itertuples(index=False, name=None)
        if set(THRESHOLDS_COLUMNS) <= set(judged.columns)
        else [(None, None)] * len(judged)
    )
    fromConfig: dict[str, list[int | None]] = {}
    ids: list[list[int | None]] = []
    for (files, digests), run in zip(recorded, judged["run"], strict=True):
        try:
            layers = recordedThresholds(files, digests)
        except ValueError as error:
            # One unreadable record must not stop the load (qaActor runs it for every visit).
            log(f"{run}: {task} {error}; thresholds left null")
            ids.append([])
            continue
        if layers is None:
            if run not in fromConfig:
                fromConfig[run] = _thresholdsIdsOf(qadb, butler, run, task, now, log)
            ids.append(fromConfig[run])
            continue
        found: list[int | None] = []
        for name, digest in layers:
            if isConfigLayer(name):
                key: tuple = (name, digest, run)

                def rebuild(run: str = run) -> bytes | None:
                    return configDocumentOf(butler, run, task)

            else:
                key, rebuild = (name, digest), None  # type: ignore[assignment]
            if key not in registered:
                registered[key] = _recordedId(qadb, name, digest, now, log, rebuild=rebuild)
            found.append(registered[key])
        ids.append(found)
    return ids


def _pipelineByRun(
    butler: Any,
    metrics: pd.DataFrame,
    runs: Iterable[str],
    log: Callable[[str], None],
    readHeader: Callable[[Any], Mapping[str, Any]] | None,
) -> dict[str, dict[str, str] | None]:
    """Return the one reduction pipeline of each RUN's judged images, `None` where there isn't one."""
    pipeline: dict[str, dict[str, str] | None] = dict.fromkeys(runs)
    if metrics.empty:
        return pipeline
    log(f"reading the pipeline of {len(metrics)} reductions from their pfsArm headers")
    versions = readReductionPipelines(butler, metrics, readHeader=readHeader)
    for run, rows in versions.groupby(metrics["run"].to_numpy()):
        try:
            pipeline[run] = commonPipeline(rows)
        except ValueError as error:
            log(f"{run}: no single reduction pipeline ({error}); versions left null")
    return pipeline


class _DryRun:
    """A stand-in for the database that writes each table to parquet instead."""

    def __init__(self, directory: Path):
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ids: dict[tuple[str, str], int] = {}

    def processingId(self, collection: str, run: str) -> int:
        return self.ids.setdefault((collection, run), -(len(self.ids) + 1))

    def write(self, table: str, frame: pd.DataFrame) -> int:
        path = self.directory / f"{table}.parquet"
        frame.to_parquet(path)
        return len(frame)


def loadCollection(
    butler: Any,
    qadb: Any,
    collection: str,
    *,
    visits: Iterable[int] | None = None,
    visitRange: tuple[int, int] | None = None,
    opdb: Any = None,
    mode: str = "backfill",
    pipelineVersions: bool = True,
    dryRun: Path | str | None = None,
    log: Callable[[str], None] = print,
    now: Callable[[], datetime] = datetime.now,
    readHeader: Callable[[Any], Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    """Load a collection's QA tables, thresholds and gate verdicts into qadb.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A Butler; read-only is enough.
    qadb : `pfs.utils.database.qadb.QaDB`
        The database, with a user that may write. Ignored with ``dryRun``.
    collection : `str`
        The collection the QA tasks wrote to (a CHAINED collection is searched
        in order; each dataset's own RUN is recorded).
    visits, visitRange
        Which visits; see `pfs.drp.qa.metrics.readers.readMetrics`.
    opdb : `pfs.utils.database.opdb.OpDB`, optional
        To copy each visit's description, design and issue time into qadb's
        ``pfs_visit``; see `pfs.drp.qa.qadb.writer.ensureVisits`.
    mode : `str`, optional
        One of `MODES`; recorded in ``qa_gate``.
    pipelineVersions : `bool`, optional
        Read the reducing pipeline's versions from the judged images' pfsArm
        headers (one file read per image). `False` leaves them null.
    dryRun : `pathlib.Path` or `str`, optional
        Write what would be loaded to parquet files in this directory, one per
        table, and touch no database. Processing ids are then negative
        placeholders and thresholds are not registered.
    log : callable, optional
        Where progress goes.
    now : callable, optional
        The clock, for ``decided_at`` and ``registered_at``.
    readHeader : callable, optional
        For `pfs.drp.qa.metrics.readers.readReductionPipelines`.

    Returns
    -------
    `dict` [`str`, `int`]
        Rows written per table (``qa_gate_thresholds`` counts layers cited),
        and ``processings``: the RUNs found.

    Raises
    ------
    ValueError
        If ``mode`` is not one of `MODES`, or there is neither a database nor
        a ``dryRun`` directory.
    LookupError
        If the collection has none of the tables for these visits.
    """
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    if qadb is None and dryRun is None:
        raise ValueError("Give a qadb to write to, or a dryRun directory")
    start = time.monotonic()
    tables = readTables(butler, collection, visits, visitRange, log=log)
    if all(table.empty for table in tables.values()):
        raise LookupError(f"No QA tables for these visits in {collection}")
    sink = _DryRun(dryRun) if dryRun is not None else None

    # The per-image tables; a long table's rows take the processing of the image they belong to.
    images = {task: tables[perImage] for task, (perImage, _) in TASK_TABLES.items()}
    runs = sorted(
        {run for table in images.values() for run in table.get("run", pd.Series(dtype=object)).dropna()}
    )
    log(f"{len(runs)} RUN(s) in {collection}: {', '.join(runs)}")
    pipeline = (
        _pipelineByRun(butler, tables["iqQaMetrics"], runs, log, readHeader)
        if pipelineVersions
        else dict.fromkeys(runs)
    )
    ids: dict[str, int] = {}
    for run in runs:
        if sink is not None:
            ids[run] = sink.processingId(collection, run)
        else:
            ids[run] = processingId(qadb, collection, run, drpQaVersionOf(butler, run), pipeline.get(run))

    # The thresholds each image was judged with, as its task recorded them: every layer, in order.
    if sink is None:
        registered: dict[tuple, int | None] = {}
        cited = {}
        for task, judged in images.items():
            if judged.empty:
                cited[task] = judged
                continue
            layers = _thresholdsIds(qadb, butler, task, judged, registered, now(), log)
            cited[task] = judged.assign(thresholdsIds=pd.Series(layers, index=judged.index, dtype=object))
        images = cited

    allVisits = {int(v) for table in images.values() for v in table.get("visit", pd.Series(dtype=object))}
    counts: dict[str, int] = {"processings": len(runs)}
    if sink is None:
        counts["pfs_visit"] = ensureVisits(qadb, allVisits, opdb)

    def write(table: str, frame: pd.DataFrame, keys: list[str], conn: Any = None) -> None:
        counts[table] = (
            sink.write(table, frame) if sink is not None else upsert(qadb, table, frame, keys, conn=conn)
        )
        log(f"{table}: {counts[table]} rows")

    for perImage, longTable in TASK_TABLES.values():
        spec = SPECS[perImage]
        image = tables[perImage]
        if image.empty:
            continue
        imageKeys = imageRows(image, spec, ids)
        write(spec.name, imageKeys, list(spec.keys))
        if longTable is None:
            continue
        longSpec = SPECS[longTable]
        rows = tables[longTable]
        # The long table's rows belong to the image's processing, whatever RUN they were read from.
        keyed = rows.drop(columns=["run"], errors="ignore").merge(
            image[[*IMAGE_KEYS, "run"]], on=list(IMAGE_KEYS), how="inner"
        )
        if len(keyed) < len(rows):
            log(f"{longTable}: {len(rows) - len(keyed)} rows without a {perImage} row, skipped")
        longRows = imageRows(keyed, longSpec, ids)
        if sink is not None:
            write(longSpec.name, longRows, list(longSpec.keys))
            continue
        # A reload replaces every loaded image's rows, in one transaction: what the new judgement no
        # longer has must go, and no reader sees an image with a verdict and no rows in between.
        with qadb.connection() as conn:
            deleteRows(qadb, longSpec.name, imageKeys[list(KEYS)], conn=conn)
            write(longSpec.name, longRows, list(longSpec.keys), conn=conn)

    # One gate row per image across every task's RUN: the twilight judge stores its verdict in a RUN of
    # its own beside pipetask's, and the gate is the worst of all the image's tasks. The row
    # takes the first present task's processing (`gateRows`, in TASK_TABLES order), so a sequence's reload
    # replaces the row the per-visit load wrote under pipetask's RUN rather than adding one.
    present = {task: table for task, table in images.items() if not table.empty}
    gate = gateRows(present, ids, mode, now()) if present else gateRows({}, 0, mode, now())
    layers = gateThresholdsRows(gate, present)
    if sink is not None:
        write("qa_gate", gate, list(QA_GATE[:3]))
        write("qa_gate_thresholds", layers, list(QA_GATE_THRESHOLDS[:5]))
    else:
        # Every layer of an image's new judgement replaces the old ones, with its gate row, in one transaction.
        with qadb.connection() as conn:
            write("qa_gate", gate, list(QA_GATE[:3]), conn=conn)
            deleteRows(qadb, "qa_gate_thresholds", gate[list(KEYS)], conn=conn)
            write("qa_gate_thresholds", layers, list(QA_GATE_THRESHOLDS[:5]), conn=conn)
    log(f"loaded {collection} in {time.monotonic() - start:.0f} s")
    return counts


def loadVisit(
    butler: Any, qadb: Any, collection: str, visit: int, mode: str = "realtime", **kwargs: Any
) -> dict[str, int]:
    """Load one visit's QA tables: `loadCollection` for qaActor, after the visit's pipetask."""
    return loadCollection(butler, qadb, collection, visits=[int(visit)], mode=mode, **kwargs)
