"""Store a per-image QA table in a Butler after the fact (INSTRM-3089).

A judgement made outside ``pipetask``, such as the twilight sequence check that
needs every exposure of a sequence, still belongs in the Butler: it is the
record the qadb loader reads, and a reload must find it. `storeImageTable`
writes one dataset per detector image into a new RUN of the collection the
judge read from, prepended to that CHAINED collection so readers of the
collection find the new tables first, as they find ``pipetask``'s outputs.
Only this module writes to a Butler; the stack is imported inside the function.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from pfs.drp.qa.qadb.tables import IMAGE_KEYS

__all__ = ["IMAGE_DIMENSIONS", "runName", "storeImageTable"]

#: The dimensions of a per-detector-image dataset.
IMAGE_DIMENSIONS = ("instrument", "visit", "arm", "spectrograph")


def runName(collection: str, datasetType: str, when: datetime | None = None) -> str:
    """Return the RUN a store goes to: ``<collection>/<datasetType>/<UTC timestamp>``, as pipetask names its RUNs."""
    when = when or datetime.now(UTC)
    return f"{collection}/{datasetType}/{when:%Y%m%dT%H%M%SZ}"


def storeImageTable(
    butler: Any,
    collection: str,
    datasetType: str,
    table: pd.DataFrame,
    *,
    instrument: str = "PFS",
    run: str | None = None,
    log: Callable[[str], None] = print,
) -> str:
    """Store a table as one ``DataFrame`` dataset per detector image, in a new RUN of a CHAINED collection.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A writeable Butler.
    collection : `str`
        The CHAINED collection the tables belong to, e.g. the QA output
        collection; the new RUN is prepended to it.
    datasetType : `str`
        The dataset type, registered if the repository lacks it, with
        `IMAGE_DIMENSIONS` and storage class ``DataFrame``.
    table : `pandas.DataFrame`
        Rows with `pfs.drp.qa.qadb.tables.IMAGE_KEYS`; each image's rows are
        stored as its dataset.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.
    run : `str`, optional
        The RUN to make; `runName` by default.
    log : callable, optional
        Where progress goes.

    Returns
    -------
    `str`
        The RUN written.

    Raises
    ------
    ValueError
        If ``collection`` is not a CHAINED collection: a RUN has no chain to
        prepend to, and the tables would be invisible to the collection's readers.
    """
    from lsst.daf.butler import DatasetType

    info = butler.collections.get_info(collection)
    if getattr(info.type, "name", str(info.type)) != "CHAINED":
        raise ValueError(
            f"{collection!r} is a {info.type} collection, not CHAINED: nothing to prepend a RUN to"
        )
    run = run or runName(collection, datasetType)
    dimensions = butler.dimensions.conform(IMAGE_DIMENSIONS)
    butler.registry.registerDatasetType(
        DatasetType(datasetType, dimensions, "DataFrame", universe=butler.dimensions)
    )
    butler.registry.registerRun(run)
    images = list(table.groupby(list(IMAGE_KEYS), sort=True))
    log(f"{datasetType}: storing {len(images)} images in {run}")
    for (visit, arm, spectrograph), rows in images:
        dataId = {
            "instrument": instrument,
            "visit": int(visit),
            "arm": str(arm),
            "spectrograph": int(spectrograph),
        }
        butler.put(rows.reset_index(drop=True), datasetType, dataId=dataId, run=run)
    butler.collections.prepend_chain(collection, [run])
    log(f"{datasetType}: {run} prepended to {collection}")
    return run
