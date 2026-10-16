"""Read stored QA metrics from a Butler, the pipeline that made their reductions, and name the runs.

The only code in `pfs.drp.qa.metrics` that touches a Butler. It takes the
Butler as an argument and never imports one, so the package stays stack-free
and the reader is tested with a stub.
"""

import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from pfs.drp.qa.utils.provenance import PIPELINE_COLUMNS, pipelineVersions

__all__ = [
    "chooseRun",
    "exportMetrics",
    "readCalibValidity",
    "readMetrics",
    "readPipelineVersions",
    "readReductionPipelines",
]

#: Data ID keys copied onto each row when the stored table lacks them.
_DATA_ID_KEYS = ("visit", "arm", "spectrograph")


def readMetrics(
    butler,
    visits: Iterable[int] | None = None,
    datasetType: str = "iqQaMetrics",
    collections: str | Iterable[str] | None = None,
    instrument: str = "PFS",
    visitRange: tuple[int, int] | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Read and concatenate a per-detector metrics table for some visits.

    One registry query for every visit, not a speculative ``get`` per data ID:
    a mistyped collection and a missing detector must not look the same.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A Butler; read-only is enough.
    visits : iterable of `int`, optional
        The visits to read, e.g. ``ValidationVisitSet.visits``.
    datasetType : `str`, optional
        The dataset type. Default is ``iqQaMetrics``.
    collections : `str` or iterable of `str`, optional
        Collections to search, in order. Defaults to the Butler's own.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.
    visitRange : `tuple` [`int`, `int`], optional
        Every visit from the first to the last, inclusive, e.g. a whole run;
        instead of ``visits``.
    progress : `bool`, optional
        Print how many datasets there are, then done/total, elapsed and ETA
        every 100 and at the end.

    Returns
    -------
    `pandas.DataFrame`
        The tables concatenated, sorted by visit, arm and spectrograph. Where
        the stored table lacks ``visit``, ``arm`` or ``spectrograph``, they are
        taken from the data ID.

    Raises
    ------
    ValueError
        If ``visits`` is empty, or not exactly one of ``visits`` and
        ``visitRange`` is given.
    LookupError
        If no datasets match: an empty result is an error to report, not an
        empty table to calibrate on.
    """
    if (visits is None) == (visitRange is None):
        raise ValueError("Give one of visits and visitRange")
    if visitRange is not None:
        first, last = (int(visit) for visit in visitRange)
        where = "instrument = instrumentName AND visit >= firstVisit AND visit <= lastVisit"
        bind = {"instrumentName": instrument, "firstVisit": first, "lastVisit": last}
        described = f"visits {first}-{last}"
    else:
        visits = tuple(sorted({int(visit) for visit in visits}))
        if not visits:
            raise ValueError("No visits to read")
        where = "instrument = instrumentName AND visit IN (visits)"
        bind = {"instrumentName": instrument, "visits": visits}
        described = f"{len(visits)} visits ({visits[0]}-{visits[-1]})"
    refs = butler.query_datasets(
        datasetType,
        collections=collections,
        where=where,
        bind=bind,
        # One dataset per data ID, from the first collection that has it. A
        # detector reduced twice in a chain would otherwise count twice.
        find_first=True,
        limit=None,
        explain=False,
    )
    if not refs:
        raise LookupError(f"No {datasetType} datasets for {described} in collections {collections!r}")

    total, start, tables = len(refs), time.monotonic(), []
    if progress:
        print(f"{datasetType}: reading {total} datasets for {described}", flush=True)
    for done, ref in enumerate(refs, 1):
        table = butler.get(ref)
        dataId = ref.dataId
        for key in _DATA_ID_KEYS:
            if key not in table.columns and key in dataId:
                table[key] = dataId[key]
        tables.append(table)
        if progress and (done % 100 == 0 or done == total):
            elapsed = time.monotonic() - start
            eta = elapsed / done * (total - done)
            print(f"{datasetType}: {done}/{total}  {elapsed:.0f} s elapsed, ETA {eta:.0f} s", flush=True)
    metrics = pd.concat(tables, ignore_index=True)
    sortKeys = [key for key in _DATA_ID_KEYS if key in metrics.columns]
    return metrics.sort_values(sortKeys, ignore_index=True)


def exportMetrics(
    butler,
    collection: str,
    datasetTypes: Sequence[str],
    out: Path | str,
    visits: Iterable[int] | None = None,
    visitRange: tuple[int, int] | None = None,
    instrument: str = "PFS",
) -> list[Path]:
    """Write per-detector tables from one collection to parquet, for work away from the Butler.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A Butler; read-only is enough.
    collection : `str`
        The collection to read.
    datasetTypes : sequence of `str`
        Per-detector tables, e.g. ``dmLineResidualSummary``, ``dmLineResidualMetrics``,
        ``iqQaMetrics``.
    out : `pathlib.Path` or `str`
        Directory to write to; made if missing.
    visits, visitRange
        Which visits; see `readMetrics`.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.

    Returns
    -------
    `list` [`pathlib.Path`]
        One file per dataset type, ``<datasetType>-<collection>.parquet``
        with ``/`` in the collection made ``_``.
    """
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stem = collection.replace("/", "_")
    paths = []
    for datasetType in datasetTypes:
        table = readMetrics(
            butler,
            visits,
            datasetType,
            collections=[collection],
            instrument=instrument,
            visitRange=visitRange,
            progress=True,
        )
        path = out / f"{datasetType}-{stem}.parquet"
        table.to_parquet(path)
        print(f"{datasetType}: {len(table)} rows -> {path} ({path.stat().st_size / 1e6:.2f} MB)", flush=True)
        paths.append(path)
    return paths


def readPipelineVersions(
    butler,
    visits: Iterable[int],
    collections: str | Iterable[str] | None = None,
    datasetType: str = "pfsArm",
    instrument: str = "PFS",
    readHeader: Callable[[Any], Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Read the pipeline each detector image of some visits was reduced with.

    drp_stella records the versions in the ``pfsArm`` written beside the
    calexp, not in the calexp. Only the primary header is read, from the file.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A Butler; read-only is enough.
    visits : iterable of `int`
        The visits.
    collections : `str` or iterable of `str`, optional
        Collections to search, in order: those the judged calexps were found
        in, so each ``pfsArm`` is the one made beside its calexp.
    datasetType : `str`, optional
        The dataset whose header records the pipeline. Default is ``pfsArm``.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.
    readHeader : callable, optional
        Returns a primary header as a mapping, given a dataset's URI; reads
        the FITS file with astropy by default.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph`` and `PIPELINE_COLUMNS`, one row
        per detector with a dataset, sorted; a version its header doesn't
        record is null.
    """
    columns = [*_DATA_ID_KEYS, *PIPELINE_COLUMNS.values()]
    visits = tuple(sorted({int(visit) for visit in visits}))
    if not visits:
        return pd.DataFrame(columns=columns)
    refs = butler.query_datasets(
        datasetType,
        collections=collections,
        where="instrument = instrumentName AND visit IN (visits)",
        bind={"instrumentName": instrument, "visits": visits},
        find_first=True,
        limit=None,
        explain=False,
    )
    readHeader = readHeader or _primaryHeader
    rows = []
    for ref in refs:
        versions = pipelineVersions(readHeader(butler.getURI(ref)))
        row = {key: ref.dataId[key] for key in _DATA_ID_KEYS}
        rows.append(row | {column: versions.get(product) for product, column in PIPELINE_COLUMNS.items()})
    return pd.DataFrame(rows, columns=columns).sort_values(list(_DATA_ID_KEYS), ignore_index=True)


def readCalibValidity(
    butler,
    calibIds: Iterable[str],
    collections: str | Iterable[str],
    datasetType: str = "detectorMap_calib",
) -> pd.DataFrame:
    """Read the validity ranges of some calibrations, by dataset ID.

    A quantum cannot read the registry, so ``dmLineResidualSummary`` records the
    ``calibId`` it was reduced against; this looks up when that calibration
    is certified for. A large drift against a calibration certified long
    before the visit is a calibration problem, not an instrument one.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A Butler; read-only is enough.
    calibIds : iterable of `str`
        Dataset IDs (``dmLineResidualSummary.calibId``).
    collections : `str` or iterable of `str`
        Calibration collections, or chains of them, to search.
    datasetType : `str`, optional
        Default is ``detectorMap_calib``.

    Returns
    -------
    `pandas.DataFrame`
        ``calibId``, ``collection``, ``validFrom`` and ``validTo`` (ISO TAI;
        null when unbounded), one row per certification of a requested
        calibration, sorted; a calibration certified in none of
        ``collections`` has no row.
    """
    columns = ["calibId", "collection", "validFrom", "validTo"]
    wanted = {str(calibId) for calibId in calibIds}
    rows = []
    if wanted:
        for association in butler.registry.queryDatasetAssociations(
            datasetType, collections=collections, flattenChains=True
        ):
            calibId = str(association.ref.id)
            timespan = association.timespan
            if calibId not in wanted or timespan is None:
                continue
            rows.append(
                {
                    "calibId": calibId,
                    "collection": association.collection,
                    "validFrom": _isoTime(timespan.begin),
                    "validTo": _isoTime(timespan.end),
                }
            )
    # dtype object, so an unbounded end stays None rather than NaN.
    frame = pd.DataFrame(rows, columns=columns, dtype=object)
    return frame.sort_values(["calibId", "validFrom"], ignore_index=True)


def _isoTime(time) -> str | None:
    """Return an `astropy.time.Time` as ISO TAI, or `None` for an unbounded end."""
    return None if time is None else time.tai.isot


def readReductionPipelines(
    butler,
    metrics: pd.DataFrame,
    fallback: str | Iterable[str] | None = None,
    datasetType: str = "pfsArm",
    instrument: str = "PFS",
    readHeader: Callable[[Any], Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Read the pipeline that made the reduction each metrics row judged.

    Each ``pfsArm`` is read from the RUN of the calexp ``imageQualityQa``
    judged (``calexpRun``), so it is the one written beside that calexp. A
    search of a chain could find a later reduction of the same detector.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A Butler; read-only is enough.
    metrics : `pandas.DataFrame`
        ``iqQaMetrics`` rows, with ``visit``, ``arm``, ``spectrograph`` and
        ``calexpRun`` (null, or absent in metrics written before it, when
        unknown).
    fallback : `str` or iterable of `str`, optional
        Collections to search for a row with no ``calexpRun``. Only right
        where each detector has one reduction, e.g. a fresh collection. By
        default such a row's pipeline is unknown.
    datasetType, instrument, readHeader
        As for `readPipelineVersions`.

    Returns
    -------
    `pandas.DataFrame`
        `PIPELINE_COLUMNS`, one row per row of ``metrics`` in its order and
        with its index; null where unknown.
    """
    keys = list(_DATA_ID_KEYS)
    columns = list(PIPELINE_COLUMNS.values())
    images = metrics[keys].reset_index(drop=True)
    if "calexpRun" in metrics:
        runs = pd.Series(metrics["calexpRun"].to_numpy(dtype=object), index=images.index)
    else:
        runs = pd.Series(None, index=images.index, dtype=object)
    known = runs.notna()
    groups = [([run], images[known & (runs == run)]) for run in sorted(set(runs[known]))]
    if fallback:
        groups.append((fallback, images[~known]))
    result = pd.DataFrame(None, index=images.index, columns=columns, dtype=object)
    for collections, subset in groups:
        if subset.empty:
            continue
        versions = readPipelineVersions(
            butler,
            subset["visit"],
            collections=collections,
            datasetType=datasetType,
            instrument=instrument,
            readHeader=readHeader,
        )
        found = subset.reset_index().merge(versions, on=keys, how="inner")
        result.loc[found["index"].to_numpy(), columns] = found[columns].to_numpy(dtype=object)
    result.index = metrics.index
    return result


def _primaryHeader(uri) -> dict[str, Any]:
    """Return the primary header of the FITS file at ``uri`` (an `lsst.resources.ResourcePath`)."""
    from astropy.io import fits

    with uri.as_local() as local:  # a local file is used where it is
        return dict(fits.getheader(local.ospath, 0))


def chooseRun(existing: Iterable[str], prefix: str, newRun: bool, runName: str | None = None) -> str:
    """Return the collection of a numbered threshold-derivation run.

    Runs are ``<prefix>/001``, ``<prefix>/002``, ... A new run takes the
    next free number; otherwise the latest run is read again.

    Parameters
    ----------
    existing : iterable of `str`
        Collection names that exist, e.g. the output of
        ``butler query-collections <repo> "<prefix>/*"``; lines that are not
        numbered runs under ``prefix`` are ignored.
    prefix : `str`
        The runs' common prefix, e.g. ``"u/someone/qa-thresholds"``.
    newRun : `bool`
        True for the next free number, False for the latest run.
    runName : `str`, optional
        An explicit collection, returned unchanged: to read an older run. Not
        allowed with ``newRun``, which would write into an existing run.

    Returns
    -------
    `str`
        The collection.

    Raises
    ------
    ValueError
        If both ``runName`` and ``newRun`` are given.
    LookupError
        If ``newRun`` is False and there is no run yet.
    """
    if runName and newRun:
        raise ValueError("RUN_NAME is for reading an older run: leave it unset to make a new one")
    if runName:
        return runName
    pattern = re.compile(rf"(?:^|\s){re.escape(prefix)}/(\d+)(?:\s|$)")
    numbers = sorted({int(match.group(1)) for line in existing for match in pattern.finditer(line)})
    if newRun:
        return f"{prefix}/{(numbers[-1] if numbers else 0) + 1:03d}"
    if not numbers:
        raise LookupError(f"No runs under {prefix}/ yet: set RUN_PIPELINE = True to make the first")
    return f"{prefix}/{numbers[-1]:03d}"
