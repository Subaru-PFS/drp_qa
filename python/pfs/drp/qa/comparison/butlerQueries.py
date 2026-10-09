"""What a Butler holds for a period's visits: raw data, reductions and verdicts.

This is the only module in `pfs.drp.qa.comparison` that touches a Butler. It takes the Butler
as an argument and never imports one, so the package stays stack-free; read-only is enough.
"""

import re
from collections.abc import Iterable, Sequence

import pandas as pd

__all__ = [
    "DETECTOR_KEYS",
    "collectionExists",
    "datasetDetectors",
    "datasetVisits",
    "detectorHoldings",
    "earlierReductions",
]

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
    # Querying an explicit name that doesn't exist raises rather than returning nothing. The
    # exception is matched by name: this module doesn't import the stack.
    try:
        return bool(butler.collections.query(name))
    except Exception as error:
        if type(error).__name__ == "MissingCollectionError":
            return False
        raise


def earlierReductions(butler, prefix: str, period: str, reductions: str, exclude: str) -> list[str]:
    """Return a period's comparison collections made with the same fresh reductions.

    Fresh reductions depend on the pipeline, not on drp_qa, so a new drp_qa version reuses those
    an earlier one made with the same pipeline, and only judges again.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The Butler.
    prefix : `str`
        The comparison prefix, e.g. ``u/someone/comparison``.
    period : `str`
        The period, e.g. ``run30``.
    reductions : `str`
        What made them, e.g. ``drp_stella-w.2026.40``.
    exclude : `str`
        The current output collection.

    Returns
    -------
    `list` [`str`]
        ``<prefix>/<period>/<any version>/<reductions>``, newest name first.
    """
    base = f"{prefix.rstrip('/')}/{period}"
    pattern = re.compile(rf"{re.escape(base)}/[^/]+/{re.escape(reductions)}")
    names = butler.collections.query(f"{base}/*")
    return sorted((name for name in names if pattern.fullmatch(name) and name != exclude), reverse=True)


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
        Collections to search; none finds nothing.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm`` and ``spectrograph``, one row per detector, sorted.
    """
    visits = sorted({int(visit) for visit in visits})
    if not visits or not collections:
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


def datasetVisits(
    butler,
    datasetType: str,
    visits: Iterable[int],
    collections: str | Sequence[str],
    instrument: str = "PFS",
) -> set[int]:
    """Return the visits that have a per-visit dataset, such as ``pfsConfig``.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The Butler.
    datasetType : `str`
        The dataset type.
    visits : iterable of `int`
        The visits.
    collections : `str` or sequence of `str`
        Collections to search.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.

    Returns
    -------
    `set` [`int`]
    """
    visits = sorted({int(visit) for visit in visits})
    if not visits:
        return set()
    refs = butler.query_datasets(
        datasetType,
        collections=collections,
        where="instrument = instrumentName AND visit IN (visits)",
        bind={"instrumentName": instrument, "visits": visits},
        find_first=True,
        limit=None,
        explain=False,
    )
    return {int(ref.dataId["visit"]) for ref in refs}


def detectorHoldings(
    butler,
    visits: Iterable[int],
    *,
    raw: str | Sequence[str],
    reductions: str | Sequence[str],
    output: str,
    inputs: str | Sequence[str] | None = None,
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
    inputs : `str` or sequence of `str`, optional
        The pipeline's input collections, searched for ``pfsConfig``.
        Defaults to ``reductions`` then ``raw``.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph`` and four `bool` columns:
        ``raw``, ``reduced`` (a ``calexp`` in ``reductions`` or ``output``),
        ``judged`` (an ``iqQaMetrics`` in ``output``) and ``pfsConfig`` (the
        visit has one in ``inputs``: without it ``reduceExposure`` can't run).
        One row per detector with any of the first three.
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
    raw = [raw] if isinstance(raw, str) else list(raw)
    inputs = [inputs] if isinstance(inputs, str) else list(inputs or [*reductions, *raw])
    holdings["pfsConfig"] = holdings["visit"].isin(
        datasetVisits(butler, "pfsConfig", visits, inputs, instrument)
    )
    return holdings.sort_values(list(DETECTOR_KEYS), ignore_index=True)
