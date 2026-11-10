"""Roll the gate's per-image verdicts up to one verdict per visit.

``qa_gate`` holds the worst verdict of the QA tasks on each image (a visit's
exposure on one camera). A nightly job and the dashboard ask about visits: was
the visit good, on how many cameras, what decided it, and did every camera that
exposed get judged. `rollupVisits` answers from the gate rows
(`pfs.drp.qa.qadb.reader.readGate`) and, for the last, the cameras the opdb says
exposed (`pfs.drp.qa.comparison.queries.readVisitCameras`).

Stack-free: DataFrames in, a DataFrame out, no database.

A camera that exposed but has no gate row is reported (``missingCameras``), not
judged: the visit's verdict is the worst of the images that were judged. A visit
with no judged image at all is ``UNKNOWN``; it is listed only when the expected
cameras are given, since the gate alone cannot know of it.
"""

from collections.abc import Iterable
from typing import Any

import pandas as pd

from pfs.drp.qa.qadb.tables import GATE_ORDER, cameraName, worstStatus

__all__ = [
    "EXIT_FAIL",
    "EXIT_NO_DATA",
    "EXIT_PASS",
    "EXIT_WARN",
    "ROLLUP_COLUMNS",
    "exitCode",
    "rollupVisits",
]

#: Exit statuses of ``bin.src/qaVisitSummary.py``: the worst visit verdict, or that nothing was judged.
EXIT_PASS = 0
EXIT_WARN = 1
EXIT_FAIL = 2
EXIT_NO_DATA = 3

#: Columns of `rollupVisits`, in order.
ROLLUP_COLUMNS = (
    "visit",
    "qaStatus",
    "nImages",
    "nPass",
    "nWarn",
    "nFail",
    "nUnknown",
    "decidedByCamera",
    "decidedByTask",
    "qaDecidedBy",
    "qaReason",
    "provisional",
    "nExpected",
    "missingCameras",
    "decidedAt",
)

_COUNTS = {"PASS": "nPass", "WARN": "nWarn", "FAIL": "nFail", "UNKNOWN": "nUnknown"}


def _text(value: Any) -> str:
    """Return a stored string, empty for a null."""
    return value if isinstance(value, str) else ""


def _cameras(names: Iterable[str]) -> list[str]:
    """Return camera names in order of spectrograph, then arm."""
    return sorted(set(names), key=lambda name: (name[1:], name[0]))


def rollupVisits(gate: pd.DataFrame, expected: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per visit: the worst of its images' verdicts, and what is missing.

    Parameters
    ----------
    gate : `pandas.DataFrame`
        One row per judged image (`pfs.drp.qa.qadb.reader.readGate`): ``visit``,
        ``arm``, ``spectrograph``, ``qaStatus``, and where known
        ``decidedByTask``, ``qaDecidedBy``, ``qaReason``, ``provisional`` and
        ``decidedAt``.
    expected : `pandas.DataFrame`, optional
        The cameras that exposed: ``visit`` and ``camera`` (``b1``), one row per
        image (`pfs.drp.qa.comparison.queries.readVisitCameras`). Without it
        ``nExpected`` and ``missingCameras`` are missing.

    Returns
    -------
    `pandas.DataFrame`
        `ROLLUP_COLUMNS`, sorted by visit. ``qaStatus`` is the worst of the
        visit's images by `pfs.drp.qa.qadb.tables.GATE_ORDER` (``UNKNOWN`` when
        none was judged); the ``decidedBy*`` columns and ``qaReason`` are those
        of the first image, by spectrograph then arm, with that verdict when it
        is ``WARN`` or ``FAIL``, and empty otherwise; ``provisional`` is true
        when any image's verdict is; ``missingCameras`` lists, comma-separated,
        the expected cameras with no gate row; ``decidedAt`` is the latest of
        the images'.
    """
    gate = gate.copy()
    for column in ("decidedByTask", "qaDecidedBy", "qaReason", "provisional", "decidedAt"):
        if column not in gate:
            gate[column] = None
    gate["camera"] = [
        cameraName(arm, spec) for arm, spec in zip(gate["arm"], gate["spectrograph"], strict=True)
    ]
    judged = {int(visit): rows for visit, rows in gate.groupby("visit", sort=True)}
    wanted: dict[int, list[str]] = {}
    if expected is not None:
        wanted = {int(visit): _cameras(rows["camera"]) for visit, rows in expected.groupby("visit")}

    records = []
    for visit in sorted(set(judged) | set(wanted)):
        rows = judged.get(visit)
        record: dict[str, Any] = dict.fromkeys(ROLLUP_COLUMNS)
        record.update(visit=visit, qaStatus="UNKNOWN", nImages=0, provisional=False)
        record.update(dict.fromkeys(_COUNTS.values(), 0))
        record.update(decidedByCamera="", decidedByTask="", qaDecidedBy="", qaReason="")
        seen: list[str] = []
        if rows is not None:
            rows = rows.sort_values(["spectrograph", "arm"])
            statuses = [s if s in GATE_ORDER else "UNKNOWN" for s in rows["qaStatus"]]
            worst = worstStatus(statuses)
            record.update(qaStatus=worst, nImages=len(rows))
            for status, column in _COUNTS.items():
                record[column] = statuses.count(status)
            if worst in ("WARN", "FAIL"):
                first = rows.iloc[statuses.index(worst)]
                record.update(
                    decidedByCamera=first["camera"],
                    decidedByTask=_text(first["decidedByTask"]),
                    qaDecidedBy=_text(first["qaDecidedBy"]),
                    qaReason=_text(first["qaReason"]),
                )
            record["provisional"] = any(value is True or value == 1 for value in rows["provisional"])
            times = pd.to_datetime(rows["decidedAt"]).dropna()
            record["decidedAt"] = times.max() if len(times) else None
            seen = list(rows["camera"])
        if expected is not None:
            cameras = wanted.get(visit, [])
            record["nExpected"] = len(cameras)
            record["missingCameras"] = ",".join(c for c in cameras if c not in seen)
        records.append(record)
    rollup = pd.DataFrame(records, columns=list(ROLLUP_COLUMNS))
    rollup["visit"] = rollup["visit"].astype("int64")
    for column in ("nImages", *_COUNTS.values()):
        rollup[column] = rollup[column].astype("int64")
    rollup["nExpected"] = rollup["nExpected"].astype("Int64")
    rollup["provisional"] = rollup["provisional"].astype(bool)
    return rollup


def exitCode(rollup: pd.DataFrame) -> int:
    """Return the exit status for a rollup: its worst visit verdict.

    `EXIT_FAIL` when any visit failed, else `EXIT_WARN` when any warned, else
    `EXIT_PASS` when any passed; `EXIT_NO_DATA` when no visit was judged (no
    rows, or every visit ``UNKNOWN``). Missing cameras don't change it.
    """
    worst = worstStatus(rollup["qaStatus"]) if len(rollup) else "UNKNOWN"
    return {"FAIL": EXIT_FAIL, "WARN": EXIT_WARN, "PASS": EXIT_PASS}.get(worst, EXIT_NO_DATA)
