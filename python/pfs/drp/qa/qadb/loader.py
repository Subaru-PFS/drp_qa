"""Read a collection's QA tables from a Butler and load them into qadb.

`loadCollection` is the one entry point: qaActor calls it for a visit as soon
as its pipetask has run (`loadVisit`), comparison mode for a run, and
``bin.src/loadQadb.py`` for a backfill. Each Butler RUN found becomes a
``data_processing_pipe2d`` row, with the drp_qa that wrote it (from the RUN's
``packages`` dataset) and the pipeline that reduced the images it judged (from
their ``pfsArm`` headers, as comparison mode reads them); the thresholds file
each task's config in that RUN names is registered, and the gate verdict of
every image is written to ``qa_gate``.
"""

import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pfs.drp.qa.metrics.gate import thresholdsPath
from pfs.drp.qa.metrics.readers import readMetrics, readReductionPipelines
from pfs.drp.qa.qadb.tables import (
    IMAGE_KEYS,
    IMAGE_QUALITY,
    IMAGE_QUALITY_SPECIES,
    KEYS,
    LINE_RESIDUAL,
    LINE_RESIDUAL_METRIC,
    QA_GATE,
    TASK_TABLES,
    TableSpec,
    gateRows,
    imageRows,
)
from pfs.drp.qa.qadb.writer import (
    UNKNOWN_VERSION,
    deleteRows,
    ensureVisits,
    processingId,
    registerThresholds,
    upsert,
)
from pfs.drp.qa.utils.provenance import commonPipeline

__all__ = [
    "MODES",
    "SPECS",
    "collectionExists",
    "drpQaVersionOf",
    "loadCollection",
    "loadVisit",
    "readTables",
    "thresholdsFilesOf",
]

#: How a load came about: qaActor as the run goes, comparison mode over a run, or later.
MODES = ("realtime", "comparison", "backfill")

#: The qadb table of each Butler dataset type loaded.
SPECS: dict[str, TableSpec] = {
    spec.source: spec for spec in (IMAGE_QUALITY, IMAGE_QUALITY_SPECIES, LINE_RESIDUAL, LINE_RESIDUAL_METRIC)
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
            log(f"{datasetType}: none in {collection}")
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
    """Return the thresholds files a task judged a RUN with, from the config pipetask stored there.

    The config's ``thresholdsFile`` first, then its ``extraThresholdsFiles``
    (``imageQualityQa``'s further layers, judged after it). Empty when the RUN
    has no config for the task, or it names no file.
    """
    try:
        config = butler.get(f"{task}_config", collections=[run])
    except Exception:  # a RUN without the task has no config
        return []
    names = [getattr(config, "thresholdsFile", None), *(getattr(config, "extraThresholdsFiles", None) or [])]
    return [thresholdsPath(name) for name in names if name]


def _thresholdsIdOf(qadb: Any, butler: Any, run: str, task: str, now: datetime, log: Callable) -> int | None:
    """Register the thresholds files a task judged a RUN with; return the first layer's id.

    Every file is registered, so each is in ``qa_thresholds`` by content;
    ``qa_gate`` cites the first, the layer judged first. `None`, with a log
    line, for a file this drp_qa no longer ships.
    """
    ids = []
    for path in thresholdsFilesOf(butler, run, task):
        try:
            ids.append(registerThresholds(qadb, path, now))
        except FileNotFoundError:
            # The file this drp_qa ships is not the one that judged (PIPE2D-1976 records it at judgement time).
            log(
                f"{run}: {task} judged with {path.name}, which this drp_qa no longer has; thresholds left null"
            )
            ids.append(None)
    return ids[0] if ids else None


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
        Rows written per table, and ``processings``: the RUNs found.

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

    # The thresholds each task judged each RUN with: pipetask stores the task's config in the RUN.
    thresholdsIds: dict[str, dict[str, int | None]] = {run: {} for run in runs}
    if sink is None:
        for run in runs:
            for task in TASK_TABLES:
                judged = images[task]
                if not judged.empty and (judged["run"] == run).any():
                    thresholdsIds[run][task] = _thresholdsIdOf(qadb, butler, run, task, now(), log)

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
        spec, longSpec = SPECS[perImage], SPECS[longTable]
        image = tables[perImage]
        if image.empty:
            continue
        imageKeys = imageRows(image, spec, ids)
        write(spec.name, imageKeys, list(spec.keys))
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

    gates = []
    for run in runs:
        byRun = {task: table[table["run"] == run] for task, table in images.items() if not table.empty}
        gates.append(gateRows(byRun, ids[run], mode, now(), thresholdsIds[run]))
    gate = pd.concat(gates, ignore_index=True) if gates else gateRows({}, 0, mode, now())
    write("qa_gate", gate, list(QA_GATE[:3]))
    log(f"loaded {collection} in {time.monotonic() - start:.0f} s")
    return counts


def loadVisit(
    butler: Any, qadb: Any, collection: str, visit: int, mode: str = "realtime", **kwargs: Any
) -> dict[str, int]:
    """Load one visit's QA tables: `loadCollection` for qaActor, after the visit's pipetask."""
    return loadCollection(butler, qadb, collection, visits=[int(visit)], mode=mode, **kwargs)
