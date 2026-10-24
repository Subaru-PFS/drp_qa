"""Whether each known-bad validation entry is caught: its check runs and gives its verdict.

A ``known_bad`` entry names the metric that should catch it (``docs/validation-visits.md``).
It is *caught* when every image it covers with that metric measured is judged, under the
thresholds the gate uses, and reaches the expected verdict: a ``WARN`` entry is reached by
``WARN`` or ``FAIL``. An entry that cannot be checked says why in its ``unchecked`` field.
One that is neither is a check that silently does not run, which is how entries naming a
metric that stopped being judged went unnoticed (PIPE2D-1934); and an ``unchecked`` entry that
is caught has outlived its reason. `knownBadCoverage` reports both.

An image is a (visit, arm, spectrograph). A table may hold several rows per image (the
``dmLineResiduals`` tables have one per species, status type and reference), so an image is
measured when any of its rows has a value for the metric, and judged when any of those rows
gets a verdict, at the worst of them.

Nothing here imports the LSST stack.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pfs.drp.qa.metrics.gate import STATUS_ORDER, ThresholdsLike, judge
from pfs.drp.qa.metrics.validationVisits import ValidationVisit, ValidationVisitSet, matchRows

__all__ = ["CAUGHT", "CoverageSource", "knownBadCoverage"]

#: Results of `knownBadCoverage`, in the order they are tested.
NO_METRIC, NO_ROWS, NOT_MEASURED, MISSED, NOT_JUDGED, CAUGHT = (
    "no metric",
    "no rows",
    "not measured",
    "missed",
    "not judged",
    "caught",
)

#: Columns identifying an image.
_IMAGE = ["visit", "arm", "spectrograph"]


@dataclass(frozen=True)
class CoverageSource:
    """A metrics table, the thresholds that judge it, and the metrics it answers for.

    Attributes
    ----------
    metrics : `pandas.DataFrame`
        Metrics rows with ``visit``, ``arm``, ``spectrograph`` and the
        selector columns of the validation set (``seqName``).
    thresholds : `pandas.DataFrame`, path, or a sequence of them
        The thresholds the gate judges this table with, highest priority
        first (`pfs.drp.qa.metrics.gate.judge`).
    names : `tuple` [`str`]
        The metrics entries are checked on in this table. A column of the same
        name in another table (``nLines`` and ``linesPerFiber`` are in both
        ``iqQaMetrics`` and the ``dmLineResiduals`` summary) is not consulted.
    """

    metrics: pd.DataFrame
    thresholds: ThresholdsLike | Sequence[ThresholdsLike]
    names: tuple[str, ...]


def knownBadCoverage(visitSet: ValidationVisitSet, sources: Sequence[CoverageSource]) -> pd.DataFrame:
    """Return, for each known-bad entry, whether its check runs and gives its verdict.

    Parameters
    ----------
    visitSet : `ValidationVisitSet`
        The validation set. Placeholders are skipped.
    sources : `list` [`CoverageSource`]
        The metrics tables and their thresholds. An entry is checked on the
        source whose ``names`` hold its metric.

    Returns
    -------
    `pandas.DataFrame`
        One row per known-bad entry, in the set's order: ``visits``,
        ``arms``, ``spectrographs`` (the entry, as text), ``metric``,
        ``expect``, ``unconfirmed``, ``unchecked`` (the reason, or empty);
        ``images`` it covers in the table, ``measured``, ``judged`` and
        ``reached`` (images reaching ``expect``); ``result``, one of
        ``caught``, ``not judged`` (a measured image gets no verdict),
        ``missed`` (a judged image falls short of ``expect``), ``not measured``,
        ``no rows`` (the entry matches nothing) and ``no metric`` (no source
        answers for its metric); and ``ok``: a confirmed entry is ok when it
        is caught or unchecked, not both. Unconfirmed entries never decide
        anything, so they are always ok.
    """
    bySource = {name: source for source in sources for name in source.names}
    judged = {id(source): judge(source.metrics, source.thresholds) for source in sources}
    rows = []
    for entry in visitSet.knownBad:
        if entry.placeholder:
            continue
        source = bySource.get(entry.metric)
        counts = _count(entry, source, judged[id(source)]) if source is not None else {}
        result = _result(counts)
        rows.append(
            {
                "visits": _visitsText(entry),
                "arms": ", ".join(entry.arms) or "all",
                "spectrographs": ", ".join(str(s) for s in entry.spectrographs) or "all",
                "metric": entry.metric or "",
                "expect": entry.expect,
                "unconfirmed": entry.unconfirmed,
                "unchecked": entry.unchecked or "",
                **{key: counts.get(key, 0) for key in ("images", "measured", "judged", "reached")},
                "result": result,
                "ok": entry.unconfirmed or ((result == CAUGHT) != bool(entry.unchecked)),
            }
        )
    return pd.DataFrame(rows)


def _count(entry: ValidationVisit, source: CoverageSource, judged: pd.DataFrame) -> dict[str, int]:
    """Return the images ``entry`` covers in ``source``, measured, judged and reaching its verdict."""
    frame = source.metrics
    covered = np.flatnonzero(matchRows(frame, [entry]))
    measured = covered[frame[entry.metric].iloc[covered].notna().to_numpy()]
    verdicts = judged[(judged["metric"] == entry.metric) & judged["row"].isin(measured)]
    rank = verdicts["status"].map({status: i for i, status in enumerate(STATUS_ORDER)}).fillna(-1)
    worst = rank.groupby([verdicts[column] for column in _IMAGE]).max()
    return {
        "images": frame.iloc[covered].groupby(_IMAGE).ngroups,
        "measured": frame.iloc[measured].groupby(_IMAGE).ngroups,
        "judged": int((worst >= 0).sum()),
        "reached": int((worst >= STATUS_ORDER.index(entry.expect)).sum()),
    }


def _result(counts: dict[str, int]) -> str:
    """Return the coverage result for an entry's counts (`knownBadCoverage`)."""
    if not counts:
        return NO_METRIC
    if counts["images"] == 0:
        return NO_ROWS
    if counts["measured"] == 0:
        return NOT_MEASURED
    if counts["reached"] < counts["judged"]:
        return MISSED
    if counts["judged"] < counts["measured"]:
        return NOT_JUDGED
    return CAUGHT


def _visitsText(entry: ValidationVisit) -> str:
    """Return an entry's visits as ``first`` or ``first–last``."""
    first, last = entry.visits[0], entry.visits[-1]
    return str(first) if first == last else f"{first}–{last}"
