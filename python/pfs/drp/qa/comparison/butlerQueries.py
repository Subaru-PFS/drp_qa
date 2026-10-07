"""What a Butler holds for a period's visits: raw data, reductions and verdicts.

This is the only module in `pfs.drp.qa.comparison` that touches a Butler. It takes the Butler
as an argument and never imports one, so the package stays stack-free; read-only is enough.
"""

from collections.abc import Iterable, Sequence

import pandas as pd

__all__ = ["DETECTOR_KEYS", "collectionExists", "datasetDetectors", "detectorHoldings"]

#: The data ID keys of a detector image.
DETECTOR_KEYS = ("visit", "arm", "spectrograph")


def collectionExists(butler, name: str) -> bool:
    """Return whether a collection exists.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The Butler.
    name : `str`
        The collection.

    Returns
    -------
    `bool`
    """
    return bool(butler.collections.query(name))


def datasetDetectors(
    butler,
    datasetType: str,
    visits: Iterable[int],
    collections: str | Sequence[str],
    instrument: str = "PFS",
) -> pd.DataFrame:
    """Return the detectors of some visits that have a dataset.

    One registry query for every visit, not a ``get`` per data ID.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The Butler.
    datasetType : `str`
        E.g. ``raw``, ``calexp``, ``iqQaMetrics``.
    visits : iterable of `int`
        The visits.
    collections : `str` or sequence of `str`
        Collections to search.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm`` and ``spectrograph``, one row per detector, sorted.
    """
    visits = sorted({int(visit) for visit in visits})
    if not visits:
        return pd.DataFrame(columns=list(DETECTOR_KEYS))
    refs = butler.query_datasets(
        datasetType,
        collections=collections,
        where="instrument = instrumentName AND visit IN (visits)",
        bind={"instrumentName": instrument, "visits": visits},
        find_first=True,
        limit=None,
        explain=False,
    )
    rows = [{key: ref.dataId[key] for key in DETECTOR_KEYS} for ref in refs]
    detectors = pd.DataFrame(rows, columns=list(DETECTOR_KEYS)).drop_duplicates()
    return detectors.sort_values(list(DETECTOR_KEYS), ignore_index=True)


def detectorHoldings(
    butler,
    visits: Iterable[int],
    *,
    raw: str | Sequence[str],
    reductions: str | Sequence[str],
    output: str,
    instrument: str = "PFS",
) -> pd.DataFrame:
    """Return, for each detector of some visits, what the Butler holds.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The Butler.
    visits : iterable of `int`
        The visits.
    raw : `str` or sequence of `str`
        Collections holding the raw data, e.g. ``PFS/defaults``.
    reductions : `str` or sequence of `str`
        Collections holding reductions to reuse, e.g. ``drpActor/reductions``.
    output : `str`
        The comparison's output collection; it need not exist yet.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph`` and three `bool` columns:
        ``raw``, ``reduced`` (a ``calexp`` in ``reductions`` or ``output``)
        and ``judged`` (an ``iqQaMetrics`` in ``output``). One row per
        detector with any of them.
    """
    visits = sorted({int(visit) for visit in visits})
    outputExists = collectionExists(butler, output)
    reductions = [reductions] if isinstance(reductions, str) else list(reductions)
    found = {
        "raw": datasetDetectors(butler, "raw", visits, raw, instrument),
        "reduced": datasetDetectors(
            butler, "calexp", visits, [*reductions, *([output] if outputExists else [])], instrument
        ),
        "judged": (
            datasetDetectors(butler, "iqQaMetrics", visits, output, instrument)
            if outputExists
            else pd.DataFrame(columns=list(DETECTOR_KEYS))
        ),
    }
    holdings = pd.DataFrame(columns=list(DETECTOR_KEYS))
    for name, detectors in found.items():
        holdings = holdings.merge(detectors.assign(**{name: True}), on=list(DETECTOR_KEYS), how="outer")
    for name in found:
        holdings[name] = holdings[name].astype("boolean").fillna(False).astype(bool)
    return holdings.sort_values(list(DETECTOR_KEYS), ignore_index=True)
