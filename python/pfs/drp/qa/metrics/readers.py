"""Read stored QA metrics from a Butler, the pipeline that made their reductions, and name the runs.

The only code in `pfs.drp.qa.metrics` that touches a Butler. It takes the
Butler as an argument and never imports one, so the package stays stack-free
and the reader is tested with a stub.
"""

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pandas as pd

from pfs.drp.qa.utils.provenance import PIPELINE_COLUMNS, pipelineVersions

__all__ = ["chooseRun", "readMetrics", "readPipelineVersions"]

#: Data ID keys copied onto each row when the stored table lacks them.
_DATA_ID_KEYS = ("visit", "arm", "spectrograph")


def readMetrics(
    butler,
    visits: Iterable[int],
    datasetType: str = "iqQaMetrics",
    collections: str | Iterable[str] | None = None,
    instrument: str = "PFS",
) -> pd.DataFrame:
    """Read and concatenate a per-detector metrics table for some visits.

    One registry query for every visit, not a speculative ``get`` per data ID:
    a mistyped collection and a missing detector must not look the same.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A Butler; read-only is enough.
    visits : iterable of `int`
        The visits to read, e.g. ``ValidationVisitSet.visits``.
    datasetType : `str`, optional
        The dataset type. Default is ``iqQaMetrics``.
    collections : `str` or iterable of `str`, optional
        Collections to search, in order. Defaults to the Butler's own.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.

    Returns
    -------
    `pandas.DataFrame`
        The tables concatenated, sorted by visit, arm and spectrograph. Where
        the stored table lacks ``visit``, ``arm`` or ``spectrograph``, they are
        taken from the data ID.

    Raises
    ------
    ValueError
        If ``visits`` is empty.
    LookupError
        If no datasets match: an empty result is an error to report, not an
        empty table to calibrate on.
    """
    visits = tuple(sorted({int(visit) for visit in visits}))
    if not visits:
        raise ValueError("No visits to read")
    refs = butler.query_datasets(
        datasetType,
        collections=collections,
        where="instrument = instrumentName AND visit IN (visits)",
        bind={"instrumentName": instrument, "visits": visits},
        # One dataset per data ID, from the first collection that has it. A
        # detector reduced twice in a chain would otherwise count twice.
        find_first=True,
        limit=None,
        explain=False,
    )
    if not refs:
        raise LookupError(
            f"No {datasetType} datasets for {len(visits)} visits ({visits[0]}-{visits[-1]})"
            f" in collections {collections!r}"
        )

    tables = []
    for ref in refs:
        table = butler.get(ref)
        dataId = ref.dataId
        for key in _DATA_ID_KEYS:
            if key not in table.columns and key in dataId:
                table[key] = dataId[key]
        tables.append(table)
    metrics = pd.concat(tables, ignore_index=True)
    sortKeys = [key for key in _DATA_ID_KEYS if key in metrics.columns]
    return metrics.sort_values(sortKeys, ignore_index=True)


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
