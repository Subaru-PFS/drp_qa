"""Mark a visit's reduceExposure failure in qadb, so the dashboard shows it as a verdict.

When reduceExposure fails for a visit the qaActor skips QA, so no QA numbers are written and
the dashboard -- reading qadb -- shows the visit as AWAITING (newer than the latest judged) or
NO RESULT (older), indistinguishable from a visit still being reduced. `markReduceFailed` writes
a ``qa_gate`` verdict of ``REDUCE FAILED`` for each detector the visit exposed, under the actor's
own collection and a RUN of its own, so the dashboard reads the failure as a verdict and the visit
leaves AWAITING / NO RESULT.

No qadb schema change: ``qa_gate.qa_status`` is free text, and the row needs only a
``data_processing_pipe2d`` row (made here by ``writer.processingId``) and a ``pfs_visit`` row
(``writer.ensureVisits``). The opdb is read once, for which detectors the visit exposed.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd

from pfs.drp.qa.comparison.queries import readVisitCameras
from pfs.drp.qa.qadb.tables import QA_GATE, gateRows
from pfs.drp.qa.qadb.writer import ensureVisits, processingId, upsert

__all__ = ["REDUCE_FAILED_STATUS", "markReduceFailed"]

#: The ``qa_gate.qa_status`` a reduceExposure failure gets. Not a gate verdict: written directly, and
#: rendered as a verdict of its own by the dashboard. Kept here so the writer and the reader agree on it.
REDUCE_FAILED_STATUS = "REDUCE FAILED"

#: ``qa_gate.qa_decided_by`` for the marker: the step that failed, not a QA check.
_DECIDED_BY = "reduceExposure"


def _reason(returnCode: int | None, status: str) -> str:
    """Return the ``qa_gate.qa_reason`` text, e.g. ``reduceExposure failed (returnCode 1: FAILED)``.

    Either part may be absent: ``(returnCode 1)``, ``(FAILED)`` or nothing beyond ``reduceExposure failed``.
    """
    parts = []
    if returnCode is not None:
        parts.append(f"returnCode {returnCode}")
    if status:
        parts.append(status)
    detail = ": ".join(parts)
    return f"reduceExposure failed ({detail})" if detail else "reduceExposure failed"


def markReduceFailed(
    qadb: Any,
    opdb: Any,
    collection: str,
    run: str,
    visit: int,
    *,
    returnCode: int | None = None,
    status: str = "",
    drpQaVersion: str | None = None,
    now: Callable[[], datetime] = datetime.now,
    log: Callable[[str], None] | None = None,
) -> int:
    """Write a ``REDUCE FAILED`` ``qa_gate`` verdict for each detector ``visit`` exposed.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The qadb, written: one ``qa_gate`` row per exposed detector, plus the
        ``data_processing_pipe2d`` and ``pfs_visit`` rows they reference.
    opdb : `pfs.utils.database.opdb.OpDB`
        The opdb, read once for the visit's exposed SpS cameras (`readVisitCameras`).
    collection : `str`
        The collection the verdicts are keyed under: the actor's output collection, the one
        the dashboard reads, so the marker shows on the same page as the judged visits.
    run : `str`
        The Butler RUN for the failed attempt (``data_processing_pipe2d.run_name``). Keeps the
        marker's processing apart from a real reduction's.
    visit : `int`
        The visit whose reduceExposure failed.
    returnCode, status : `int`, `str`, optional
        The failure's returnCode and status string, for the reason text.
    drpQaVersion : `str`, optional
        The drp_qa writing the marker; recorded on the ``data_processing_pipe2d`` row.
    now : callable, optional
        The clock, for ``decided_at`` (naive, as the loader's).
    log : callable, optional
        Takes one line, for progress.

    Returns
    -------
    `int`
        ``qa_gate`` rows written (one per exposed detector); 0 when the opdb lists no camera
        for the visit, in which case nothing is written.
    """
    cameras = readVisitCameras(opdb, int(visit), int(visit))
    if cameras.empty:
        if log is not None:
            log(f"visit {visit}: the opdb lists no exposed camera; nothing marked")
        return 0
    verdicts = pd.DataFrame(
        {
            "visit": cameras["visit"].astype(int),
            "arm": [str(camera)[0] for camera in cameras["camera"]],
            "spectrograph": [int(str(camera)[1:]) for camera in cameras["camera"]],
            "frameType": None,
            "qaStatus": REDUCE_FAILED_STATUS,
            "qaDecidedBy": _DECIDED_BY,
            "qaReason": _reason(returnCode, status),
        }
    )
    ensureVisits(qadb, [int(visit)], opdb)
    pid = processingId(qadb, collection, run, drpQaVersion)
    rows = gateRows(verdicts, pid, mode="realtime", decidedAt=now())
    written = upsert(qadb, "qa_gate", rows, list(QA_GATE[:3]))
    if log is not None:
        log(f"visit {visit}: {written} detector(s) marked {REDUCE_FAILED_STATUS} under {collection} / {run}")
    return written
