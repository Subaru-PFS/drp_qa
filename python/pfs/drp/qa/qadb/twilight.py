"""The twilight verdicts as qadb rows, and their write (INSTRM-3089).

`pfs.drp.qa.metrics.twilight` judges a twilight sequence's exposures against
each other once the sequence is complete; ``bin.src/twilightQa.py`` runs it
after every visit from ics_qaActor. The verdict (one row per exposure and arm)
goes to ``twilight_qa`` and the measurements behind it (one row per detector
image) to ``twilight_qa_metric``, both of ``spt_qa_database`` 0.3.0, keyed by
the visit alone: a judgement is of a sequence, not of a Butler RUN, and
judging it again replaces its rows. Nothing here reads a Butler; the shaping
functions take and return `pandas.DataFrame`.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd

from pfs.drp.qa.metrics.twilight import METRIC_COLUMNS, VERDICT_COLUMNS
from pfs.drp.qa.qadb.tables import cameraName, snakeCase
from pfs.drp.qa.qadb.writer import deleteRows, ensureVisits, upsert

__all__ = [
    "TWILIGHT_METRIC_KEYS",
    "TWILIGHT_QA",
    "TWILIGHT_QA_METRIC",
    "TWILIGHT_VERDICT_KEYS",
    "drpQaVersion",
    "metricRows",
    "verdictRows",
    "writeTwilight",
]

#: The verdict table and its key: one row per exposure and arm.
TWILIGHT_QA = "twilight_qa"
TWILIGHT_VERDICT_KEYS = ("pfs_visit_id", "arm")

#: The measurements table and its key: one row per detector image.
TWILIGHT_QA_METRIC = "twilight_qa_metric"
TWILIGHT_METRIC_KEYS = ("pfs_visit_id", "camera_name")

#: `pfs.drp.qa.metrics.twilight.VERDICT_COLUMNS` whose qadb name is not their `snakeCase`.
_VERDICT_RENAMES = {
    "visit": "pfs_visit_id",
    "sequence": "iic_sequence_id",
    "status": "qa_status",
    "reason": "qa_reason",
}

#: Likewise for `pfs.drp.qa.metrics.twilight.METRIC_COLUMNS`.
_METRIC_RENAMES = {"visit": "pfs_visit_id", "sequence": "iic_sequence_id"}


def drpQaVersion() -> str | None:
    """Return this drp_qa's version, or `None` when nothing records it.

    The installed package's ``version`` module first (``lsst-versions`` writes
    it), then the EUPS environment (``setup -r .`` writes no module).
    """
    try:
        from pfs.drp.qa.version import __version__

        return str(__version__)
    except ImportError:
        pass
    try:
        from lsst.utils.packages import getEnvironmentPackages

        version = getEnvironmentPackages().get("drp_qa")
        return str(version) if version else None
    except Exception:
        return None


def verdictRows(
    verdict: pd.DataFrame, collection: str | None, drpQaVersion: str | None, judgedAt: datetime
) -> pd.DataFrame:
    """Return `pfs.drp.qa.metrics.twilight.twilightVerdict` rows as ``twilight_qa`` rows.

    Parameters
    ----------
    verdict : `pandas.DataFrame`
        `pfs.drp.qa.metrics.twilight.VERDICT_COLUMNS`.
    collection : `str` or `None`
        The collection the ``twilightFiberFlux`` tables were read from.
    drpQaVersion : `str` or `None`
        The drp_qa judging.
    judgedAt : `datetime.datetime`
        When.

    Returns
    -------
    `pandas.DataFrame`
        ``pfs_visit_id``, ``arm``, ``iic_sequence_id``, ``qa_status``,
        ``qa_reason``, ``twilight_arm_mean``, ``twilight_arm_half_diff``,
        ``collection_name``, ``drp_qa_version``, ``judged_at``.
    """
    rows = verdict.reindex(columns=list(VERDICT_COLUMNS)).rename(columns=_VERDICT_RENAMES)
    rows.columns = [snakeCase(column) for column in rows.columns]
    rows["collection_name"] = collection
    rows["drp_qa_version"] = drpQaVersion
    rows["judged_at"] = judgedAt
    return rows.reset_index(drop=True)


def metricRows(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return `pfs.drp.qa.metrics.twilight.sequenceConsistency` rows as ``twilight_qa_metric`` rows.

    Every `pfs.drp.qa.metrics.twilight.METRIC_COLUMNS` column, as `snakeCase`,
    with ``visit`` as ``pfs_visit_id``, ``sequence`` as ``iic_sequence_id``,
    and the ``camera_name`` (``b1``) of each row's arm and spectrograph. A
    non-finite value (``twilightLevelOffset`` is ``inf`` with no measured
    fiber) is written as null by `pfs.drp.qa.qadb.writer.records`.
    """
    rows = metrics.reindex(columns=list(METRIC_COLUMNS)).rename(columns=_METRIC_RENAMES)
    rows.columns = [snakeCase(column) for column in rows.columns]
    rows.insert(
        1,
        "camera_name",
        [
            cameraName(arm, spectrograph)
            for arm, spectrograph in zip(rows["arm"], rows["spectrograph"], strict=True)
        ],
    )
    return rows.reset_index(drop=True)


def writeTwilight(
    qadb: Any,
    verdict: pd.DataFrame,
    metrics: pd.DataFrame,
    collection: str | None,
    *,
    version: str | None = None,
    opdb: Any = None,
    now: Callable[[], datetime] = datetime.now,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Write a judgement's verdicts and measurements to qadb, replacing the judged visits' rows.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The database, with a user that may write.
    verdict : `pandas.DataFrame`
        `pfs.drp.qa.metrics.twilight.twilightVerdict`'s table.
    metrics : `pandas.DataFrame`
        `pfs.drp.qa.metrics.twilight.sequenceConsistency`'s table.
    collection : `str` or `None`
        The collection the ``twilightFiberFlux`` tables were read from.
    version : `str`, optional
        The drp_qa judging; `drpQaVersion` by default.
    opdb : `pfs.utils.database.opdb.OpDB`, optional
        To copy the visits' descriptions into qadb's ``pfs_visit``; see
        `pfs.drp.qa.qadb.writer.ensureVisits`.
    now : callable, optional
        The clock, for ``judged_at``.
    log : callable, optional
        Where progress goes.

    Returns
    -------
    `dict` [`str`, `int`]
        Rows written per table.

    Notes
    -----
    Every visit in either table has its rows of both tables deleted and
    rewritten in one transaction, so an arm or a detector the new judgement
    lacks (an exposure dropped from the sequence, an arm no longer judged) is
    gone, and no reader sees a visit half-written.
    """
    visits = sorted({int(v) for v in verdict.get("visit", [])} | {int(v) for v in metrics.get("visit", [])})
    counts: dict[str, int] = {}
    if not visits:
        log("twilight: nothing to write")
        return counts
    counts["pfs_visit"] = ensureVisits(qadb, visits, opdb)
    version = drpQaVersion() if version is None else version
    verdicts = verdictRows(verdict, collection, version, now())
    measured = metricRows(metrics)
    keys = pd.DataFrame({"pfs_visit_id": visits})
    with qadb.connection() as conn:
        deleteRows(qadb, TWILIGHT_QA_METRIC, keys, conn=conn)
        deleteRows(qadb, TWILIGHT_QA, keys, conn=conn)
        counts[TWILIGHT_QA] = upsert(qadb, TWILIGHT_QA, verdicts, list(TWILIGHT_VERDICT_KEYS), conn=conn)
        counts[TWILIGHT_QA_METRIC] = upsert(
            qadb, TWILIGHT_QA_METRIC, measured, list(TWILIGHT_METRIC_KEYS), conn=conn
        )
    log(f"{TWILIGHT_QA}: {counts[TWILIGHT_QA]} rows; {TWILIGHT_QA_METRIC}: {counts[TWILIGHT_QA_METRIC]} rows")
    return counts
