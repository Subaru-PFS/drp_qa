"""Read a collection's QA tables from a Butler and load them into qadb.

`loadCollection` is the one entry point: qaActor calls it for a visit as soon
as its pipetask has run (`loadVisit`), comparison mode for a run, and
``bin.src/loadQadb.py`` for a backfill. Each Butler RUN found becomes a
``data_processing_pipe2d`` row, with the drp_qa that wrote it (from the RUN's
``packages`` dataset) and the pipeline that reduced the images it judged (from
their ``pfsArm`` headers, as comparison mode reads them); the thresholds files
the tasks' configs name are registered, and the gate verdict of every image is
written to ``qa_gate``.
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
from pfs.drp.qa.qadb.writer import deleteRows, ensureVisits, processingId, registerThresholds, upsert
from pfs.drp.qa.utils.provenance import commonPipeline

__all__ = [
    "MODES",
    "SPECS",
    "drpQaVersionOf",
    "loadCollection",
    "loadVisit",
    "readTables",
    "thresholdsFileOf",
]

#: How a load came about: qaActor as the run goes, comparison mode over a run, or later.
MODES = ("realtime", "comparison", "backfill")

#: The qadb table of each Butler dataset type loaded.
SPECS: dict[str, TableSpec] = {
    spec.source: spec for spec in (IMAGE_QUALITY, IMAGE_QUALITY_SPECIES, LINE_RESIDUAL, LINE_RESIDUAL_METRIC)
}

_UNKNOWN = "unknown"


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
            )
        except LookupError:
            log(f"{datasetType}: none in {collection}")
            tables[datasetType] = pd.DataFrame(columns=[*IMAGE_KEYS, "run"])
    return tables


def drpQaVersionOf(butler: Any, run: str) -> str:
    """Return the drp_qa version that wrote a RUN, from its ``packages`` dataset.

    pipetask records the versions of every product set up when it ran. Falls
    back to this package's own version, then ``unknown``.
    """
    try:
        packages = butler.get("packages", collections=[run])
        version = packages.get("drp_qa") if hasattr(packages, "get") else packages["drp_qa"]
        if version:
            return str(version)
    except Exception:  # any Butler failure means the RUN recorded none
        pass
    try:
        from pfs.drp.qa.version import __version__

        return str(__version__)
    except ImportError:
        return _UNKNOWN


def thresholdsFileOf(butler: Any, collection: str, task: str) -> Path | None:
    """Return the thresholds file a task judged with, from its config in the collection.

    `None` when the config is not there or names no file.
    """
    try:
        config = butler.get(f"{task}_config", collections=[collection])
    except Exception:  # a collection without the task has no config
        return None
    name = getattr(config, "thresholdsFile", None)
    return thresholdsPath(name) if name else None


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
        If ``mode`` is not one of `MODES`.
    LookupError
        If the collection has none of the tables for these visits.
    """
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
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

    # The thresholds of each task that judged something here; a task's config alone says nothing.
    thresholdsIds: dict[str, int | None] = {}
    for task in TASK_TABLES:
        path = thresholdsFileOf(butler, collection, task) if not images[task].empty else None
        thresholdsIds[task] = registerThresholds(qadb, path, now()) if path and sink is None else None

    allVisits = {int(v) for table in images.values() for v in table.get("visit", pd.Series(dtype=object))}
    counts: dict[str, int] = {"processings": len(runs)}
    if sink is None:
        counts["pfs_visit"] = ensureVisits(qadb, allVisits, opdb)

    def write(table: str, frame: pd.DataFrame, keys: list[str]) -> None:
        counts[table] = sink.write(table, frame) if sink is not None else upsert(qadb, table, frame, keys)
        log(f"{table}: {counts[table]} rows")

    for perImage, longTable in TASK_TABLES.values():
        spec, longSpec = SPECS[perImage], SPECS[longTable]
        image = tables[perImage]
        if not image.empty:
            write(spec.name, imageRows(image, spec, ids), list(spec.keys))
        rows = tables[longTable]
        if not rows.empty:
            # The long table's rows belong to the image's processing, whatever RUN they were read from.
            keyed = rows.drop(columns=["run"], errors="ignore").merge(
                image[[*IMAGE_KEYS, "run"]], on=list(IMAGE_KEYS), how="inner"
            )
            if len(keyed) < len(rows):
                log(f"{longTable}: {len(rows) - len(keyed)} rows without a {perImage} row, skipped")
            longRows = imageRows(keyed, longSpec, ids)
            if sink is None:
                # A reload replaces the image's rows: what the new judgement no longer has must go too.
                deleteRows(qadb, longSpec.name, longRows[list(KEYS)])
            write(longSpec.name, longRows, list(longSpec.keys))

    gate = gateRows(images, ids, mode, now(), thresholdsIds)
    write("qa_gate", gate, list(QA_GATE[:3]))
    log(f"loaded {collection} in {time.monotonic() - start:.0f} s")
    return counts


def loadVisit(
    butler: Any, qadb: Any, collection: str, visit: int, mode: str = "realtime", **kwargs: Any
) -> dict[str, int]:
    """Load one visit's QA tables: `loadCollection` for qaActor, after the visit's pipetask."""
    return loadCollection(butler, qadb, collection, visits=[int(visit)], mode=mode, **kwargs)
