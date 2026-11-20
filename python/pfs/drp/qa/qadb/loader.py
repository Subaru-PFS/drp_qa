"""Read a collection's QA numbers from a Butler, load them into qadb, and judge them.

`loadCollection` is the one entry point: qaActor calls it for a visit as soon
as its pipetask has run (`loadVisit`), and ``drpQa.py load`` for a backfill.
Each Butler RUN found becomes a ``data_processing_pipe2d`` row, with the drp_qa
that wrote it (from the RUN's ``packages`` dataset) and the pipeline that reduced
its images (from their ``pfsArm`` headers).

The tasks store numbers only. The verdict is given here, once per image, by
`pfs.drp.qa.metrics.gate.gate`, from the numbers just read and the collection's
normal values in qadb, and written to ``qa_gate``.

A twilight exposure is judged against the other exposures of its sequence, so
it has no numbers to judge until the sequence is complete. When a visit being
loaded is a twilight, its IIC sequence is looked up in the opdb; once the
sequence has finished and every visit of it has its ``twilightFiberFlux``, the
whole sequence is measured (`pfs.drp.qa.metrics.twilight.sequenceConsistency`),
stored in ``twilight_qa`` and judged, the visits loaded earlier included. Until
then, and without an opdb, a twilight image is ``UNKNOWN``.
"""

import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pfs.drp.qa.metrics.gate import gate
from pfs.drp.qa.metrics.readers import datasetDetectors, readMetrics, readReductionPipelines
from pfs.drp.qa.metrics.twilight import isTwilightSequence, sequenceConsistency, sequenceReadiness
from pfs.drp.qa.qadb.reader import PROCESSING_ID, readGate, readNormalValues
from pfs.drp.qa.qadb.tables import (
    IMAGE_KEYS,
    IMAGE_QUALITY,
    IMAGE_QUALITY_SPECIES,
    KEYS,
    LINE_RESIDUAL,
    LINE_RESIDUAL_METRIC,
    QA_GATE,
    TWILIGHT_QA,
    TableSpec,
    gateRows,
    imageRows,
)
from pfs.drp.qa.qadb.writer import UNKNOWN_VERSION, deleteRows, ensureVisits, processingId, upsert
from pfs.drp.qa.utils.provenance import commonPipeline

__all__ = [
    "MODES",
    "SPECS",
    "TABLES",
    "collectionExists",
    "drpQaVersionOf",
    "imageRuns",
    "loadCollection",
    "loadVisit",
    "readTables",
    "twilightMetrics",
]

#: How a load came about, recorded in ``qa_gate``.
MODES = ("realtime", "comparison", "backfill")

#: The Butler tables loaded: each per-image table with the long table whose rows take the image's processing.
TABLES = (("iqQaMetrics", "iqQaSpeciesMetrics"), ("dmLineResidualSummary", "dmLineResidualMetrics"))

#: The qadb table of each Butler dataset type loaded.
SPECS: dict[str, TableSpec] = {
    spec.source: spec for spec in (IMAGE_QUALITY, IMAGE_QUALITY_SPECIES, LINE_RESIDUAL, LINE_RESIDUAL_METRIC)
}

#: The per-fiber table ``twilightFluxQa`` stores, which a sequence's metrics are measured from.
TWILIGHT_FLUX = "twilightFiberFlux"


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
            # Not a problem by itself: a task that did not run on these visits leaves none.
            log(f"{datasetType}: none for these visits in {collection}")
            tables[datasetType] = pd.DataFrame(columns=[*IMAGE_KEYS, "run"])
    return tables


def twilightMetrics(
    butler: Any,
    opdb: Any,
    collection: str,
    visits: Iterable[int],
    log: Callable[[str], None] = print,
    readSequence: Callable[[Any, int], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Measure the complete twilight sequences some twilight visits belong to.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A Butler; read-only is enough.
    opdb : `pfs.utils.database.opdb.OpDB`
        The opdb, read-only: which IIC sequence a visit belongs to, and
        whether it has finished.
    collection : `str`
        The collection holding ``twilightFiberFlux``.
    visits : iterable of `int`
        Twilight visits being loaded.
    log : callable, optional
        Where progress goes.
    readSequence : callable, optional
        ``(opdb, visit)`` to the sequence's visits; default
        `pfs.drp.qa.comparison.queries.readSequenceOfVisit`.

    Returns
    -------
    `pandas.DataFrame`
        `pfs.drp.qa.metrics.twilight.sequenceConsistency` rows, one per image
        of every visit of each sequence that is ready (a finished twilight
        with every visit's table stored), with ``run``, the Butler RUN of the
        image's ``twilightFiberFlux``. No rows when none is ready: what each
        is waiting for is logged.
    """
    if readSequence is None:
        from pfs.drp.qa.comparison.queries import readSequenceOfVisit as readSequence
    parts, seen = [], set()
    for visit in sorted({int(visit) for visit in visits}):
        if visit in seen:
            continue
        sequence = readSequence(opdb, visit)
        members = [int(member) for member in sequence["pfs_visit_id"]]
        seen.update(members)
        if not isTwilightSequence(sequence):
            log(f"twilight: visit {visit}: {sequenceReadiness(sequence, {}, {})[1]}")
            continue
        reduced = datasetDetectors(butler, "pfsArm", members, collections=[collection])
        stored = datasetDetectors(butler, TWILIGHT_FLUX, members, collections=[collection])
        ready, why = sequenceReadiness(sequence, reduced, stored)
        log(f"twilight: visit {visit}: {why}")
        if not ready:
            continue
        fibers = readMetrics(
            butler, members, TWILIGHT_FLUX, collections=[collection], progress=True, withRun=True, log=log
        )
        fibers["sequence"] = int(sequence["iic_sequence_id"].iloc[0])
        metrics = sequenceConsistency(fibers.drop(columns=["run"]))
        runs = fibers[[*IMAGE_KEYS, "run"]].drop_duplicates(list(IMAGE_KEYS))
        parts.append(metrics.merge(runs, on=list(IMAGE_KEYS), how="left"))
    if not parts:
        return pd.DataFrame(columns=[*IMAGE_KEYS, "run"])
    return pd.concat(parts, ignore_index=True)


def imageRuns(*tables: pd.DataFrame) -> pd.DataFrame:
    """Return the Butler RUN of each image: that of the first table, in the order given, with a row for it."""
    keys = list(IMAGE_KEYS)
    parts = [table[[*keys, "run"]] for table in tables if len(table)]
    if not parts:
        return pd.DataFrame(columns=[*keys, "run"])
    runs = pd.concat(parts, ignore_index=True).dropna(subset=["run"]).drop_duplicates(keys)
    return runs.astype({"visit": "int64", "spectrograph": "int64"}).reset_index(drop=True)


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
    normalValues: pd.DataFrame | None = None,
    log: Callable[[str], None] = print,
    now: Callable[[], datetime] = datetime.now,
    readHeader: Callable[[Any], Mapping[str, Any]] | None = None,
    readSequence: Callable[[Any, int], pd.DataFrame] | None = None,
) -> dict[str, int]:
    """Load a collection's QA numbers into qadb, and judge every image loaded.

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
        ``pfs_visit`` (`pfs.drp.qa.qadb.writer.ensureVisits`), and to find
        the sequence of a twilight visit (`twilightMetrics`). Without it no
        twilight is judged.
    mode : `str`, optional
        One of `MODES`; recorded in ``qa_gate``.
    pipelineVersions : `bool`, optional
        Read the reducing pipeline's versions from the images' pfsArm headers
        (one file read per image). `False` leaves them null.
    dryRun : `pathlib.Path` or `str`, optional
        Write what would be loaded to parquet files in this directory, one per
        table, and touch no database. Processing ids are then negative
        placeholders.
    normalValues : `pandas.DataFrame`, optional
        The normal values to judge with
        (`pfs.drp.qa.metrics.normalValues.NORMAL_COLUMNS`); default the
        collection's in qadb. With ``dryRun`` and none given, the checks that
        need one give no verdict.
    log : callable, optional
        Where progress goes.
    now : callable, optional
        The clock, for ``decided_at``.
    readHeader : callable, optional
        For `pfs.drp.qa.metrics.readers.readReductionPipelines`.
    readSequence : callable, optional
        For `twilightMetrics`.

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
    iq, summary = tables["iqQaMetrics"], tables["dmLineResidualSummary"]

    # A twilight is measured with its sequence, once that is complete.
    twilightVisits = sorted(set(iq.loc[iq["obsType"] == "twilight", "visit"])) if "obsType" in iq else []
    twilight = pd.DataFrame(columns=[*IMAGE_KEYS, "run"])
    if twilightVisits and opdb is None:
        log(
            f"twilight: {len(twilightVisits)} twilight visit(s) not judged: no opdb to find their sequence in"
        )
    elif twilightVisits:
        twilight = twilightMetrics(
            butler, opdb, collection, twilightVisits, log=log, readSequence=readSequence
        )

    runs = sorted(
        {
            run
            for table in (iq, summary, twilight)
            for run in table.get("run", pd.Series(dtype=object)).dropna()
        }
    )
    log(f"{len(runs)} RUN(s) in {collection}: {', '.join(runs)}")
    pipeline = (
        _pipelineByRun(
            butler, iq, sorted(set(iq.get("run", pd.Series(dtype=object)).dropna())), log, readHeader
        )
        if pipelineVersions
        else {}
    )
    ids: dict[str, int] = {}
    for run in runs:
        if sink is not None:
            ids[run] = sink.processingId(collection, run)
        else:
            ids[run] = processingId(qadb, collection, run, drpQaVersionOf(butler, run), pipeline.get(run))

    allVisits = {
        int(v) for table in (iq, summary, twilight) for v in table.get("visit", pd.Series(dtype=object))
    }
    counts: dict[str, int] = {"processings": len(runs)}
    if sink is None:
        counts["pfs_visit"] = ensureVisits(qadb, allVisits, opdb)

    def write(table: str, frame: pd.DataFrame, keys: list[str], conn: Any = None) -> None:
        counts[table] = (
            sink.write(table, frame) if sink is not None else upsert(qadb, table, frame, keys, conn=conn)
        )
        log(f"{table}: {counts[table]} rows")

    for perImage, longTable in TABLES:
        spec = SPECS[perImage]
        image = tables[perImage]
        if image.empty:
            continue
        imageKeys = imageRows(image, spec, ids)
        write(spec.name, imageKeys, list(spec.keys))
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
        # A reload replaces every loaded image's rows, in one transaction: what the new measurement no
        # longer has must go.
        with qadb.connection() as conn:
            deleteRows(qadb, longSpec.name, imageKeys[list(KEYS)], conn=conn)
            write(longSpec.name, longRows, list(longSpec.keys), conn=conn)
    if not twilight.empty:
        write(TWILIGHT_QA.name, imageRows(twilight, TWILIGHT_QA, ids), list(TWILIGHT_QA.keys))

    # One verdict per image, from every number it has. The row takes the processing of the image's first
    # table loaded (imageQualityQa's, then dmLineResiduals', then the twilight's), so a later load replaces it.
    if normalValues is None and sink is None:
        normalValues = readNormalValues(qadb, collection)
    verdicts = gate(
        iq=iq, lines=tables["dmLineResidualMetrics"], twilight=twilight, normalValues=normalValues
    )
    verdicts = verdicts.merge(imageRuns(iq, summary, twilight), on=list(IMAGE_KEYS), how="left")
    orphans = verdicts["run"].isna()
    if orphans.any():
        log(f"qa_gate: {int(orphans.sum())} images with line residuals and no summary row, skipped")
        verdicts = verdicts[~orphans]
    verdicts = verdicts.assign(run=verdicts["run"].map(ids).astype("int64"))
    # A twilight sequence judged now includes visits loaded earlier: their verdict replaces the gate row
    # they have, whatever RUN their fiber fluxes are in.
    loaded = set(iq.get("visit", pd.Series(dtype="int64"))) | set(
        summary.get("visit", pd.Series(dtype="int64"))
    )
    earlier = ~verdicts["visit"].isin(loaded)
    if sink is None and earlier.any():
        visitsEarlier = verdicts.loc[earlier, "visit"]
        existing = readGate(qadb, collection, int(visitsEarlier.min()), int(visitsEarlier.max()))
        existing = existing[[*IMAGE_KEYS, PROCESSING_ID]].drop_duplicates(list(IMAGE_KEYS))
        verdicts = verdicts.merge(existing, on=list(IMAGE_KEYS), how="left")
        replace = earlier.to_numpy() & verdicts[PROCESSING_ID].notna().to_numpy()
        verdicts.loc[replace, "run"] = verdicts.loc[replace, PROCESSING_ID].astype("int64")
        verdicts = verdicts.drop(columns=[PROCESSING_ID])
    processings = {int(value): int(value) for value in verdicts["run"].unique()}
    write("qa_gate", gateRows(verdicts, processings, mode, now()), list(QA_GATE[:3]))
    tally = verdicts["qaStatus"].value_counts().to_dict()
    log(
        "verdicts: "
        + ", ".join(
            f"{tally[status]} {status}" for status in ("PASS", "WARN", "FAIL", "UNKNOWN") if status in tally
        )
    )
    log(f"loaded {collection} in {time.monotonic() - start:.0f} s")
    return counts


def loadVisit(
    butler: Any, qadb: Any, collection: str, visit: int, mode: str = "realtime", **kwargs: Any
) -> dict[str, int]:
    """Load one visit's QA numbers and judge it: `loadCollection` for qaActor, after the visit's pipetask."""
    return loadCollection(butler, qadb, collection, visits=[int(visit)], mode=mode, **kwargs)
