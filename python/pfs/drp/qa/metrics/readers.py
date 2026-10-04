"""Read stored QA metrics from a Butler.

The only code in `pfs.drp.qa.metrics` that touches a Butler. It takes the
Butler as an argument and never imports one, so the package stays stack-free
and the reader is tested with a stub.
"""

from collections.abc import Iterable

import pandas as pd

__all__ = ["readMetrics"]

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
