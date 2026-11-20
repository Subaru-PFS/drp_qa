"""Each camera's normal values: made from a run's calibration frames, compared, accepted.

A normal value is the median of one metric on one camera (or arm), for one
frame type and lamp or species, over the frames a run's calibrations are built
from. The relative checks of `pfs.drp.qa.metrics.registry.CHECKS` judge a frame
by the latest accepted value at or before it (`pfs.drp.qa.metrics.gate`), so
the previous run's values stay in force until the new ones are accepted.

- `candidates` makes them from the calibration frames' stored numbers, reading
  each check's numbers as the gate does (`pfs.drp.qa.metrics.gate.measurements`).
- `compare` sets each beside the value in force, with its shift.
- `accepted` returns the rows to store: a value shifted beyond its check's
  FAIL is refused unless a reason is given, and one made from fewer than
  `pfs.drp.qa.metrics.registry.MIN_NORMAL_FRAMES` frames is left out, so the
  previous value stays in force for both.

The table also holds, per camera, the offset of a visit read with the m arm
(`pfs.drp.qa.metrics.registry.M_ARM_OFFSETS`): the calibration is made with the
r arm, and with m the b arm sits a steady fraction of a pixel off in x and the
n arm in y. Each is the median with m minus the median with r, from the run's
quartz (b, in x) and Neon arcs (n, in y), with at least
`pfs.drp.qa.metrics.registry.MIN_NORMAL_FRAMES` visits in each mode.

Nothing here imports the LSST stack.
"""

from collections.abc import Iterable
from datetime import datetime

import numpy as np
import pandas as pd

from pfs.drp.qa.metrics.gate import IMAGE_KEYS, NORMAL_KEYS, _inForce, measurements
from pfs.drp.qa.metrics.registry import (
    ARC,
    CHECKS,
    FIXED,
    M_ARM_OFFSETS,
    MIN_NORMAL_FRAMES,
    QUARTZ,
    RELATIVE,
    specFor,
)

__all__ = [
    "CANDIDATE_COLUMNS",
    "M_ARM_SPECIES",
    "NORMAL_COLUMNS",
    "accepted",
    "candidates",
    "compare",
    "fixedFailures",
    "formatComparison",
    "visitRanges",
]

#: Columns of the table of normal values, as qadb's ``qa_normal_value`` holds them.
NORMAL_COLUMNS = (
    *NORMAL_KEYS,
    "value",
    "count",
    "collection",
    "validFrom",
    "sourceVisits",
    "acceptedBy",
    "acceptedAt",
    "reason",
)

#: Columns of `candidates`.
CANDIDATE_COLUMNS = (*NORMAL_KEYS, "value", "count", "firstVisit", "lastVisit", "sourceVisits")

#: The lamp species whose y offset on the n arm gives the m-arm offset in y.
M_ARM_SPECIES = "NeI"


def visitRanges(visits: Iterable[int]) -> str:
    """Return visits as text: ``"133025-133053,135830"``."""
    ordered = sorted({int(visit) for visit in visits})
    ranges, start = [], None
    for index, visit in enumerate(ordered):
        if start is None:
            start = visit
        if index + 1 == len(ordered) or ordered[index + 1] != visit + 1:
            ranges.append(str(start) if start == visit else f"{start}-{visit}")
            start = None
    return ",".join(ranges)


def _relativeChecks() -> dict[tuple[str, str], str]:
    """Return the relative check of each (metric, frame type)."""
    return {
        (metric, frame): check.name
        for check in CHECKS.values()
        if check.kind == RELATIVE
        for metric in check.metrics
        for frame in check.frames
    }


def candidates(
    iq: pd.DataFrame | None = None, lines: pd.DataFrame | None = None, exclude: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Make candidate normal values from calibration frames' numbers.

    Parameters
    ----------
    iq, lines : `pandas.DataFrame`, optional
        ``iqQaMetrics`` and ``dmLineResidualMetrics`` rows of the frames the
        calibrations are built from.
    exclude : `pandas.DataFrame`, optional
        Images (``visit``, ``arm``, ``spectrograph``) to leave out: those
        that failed a fixed check.

    Returns
    -------
    `pandas.DataFrame`
        `CANDIDATE_COLUMNS`, sorted: per metric, camera or arm, frame type
        and lamp or species, the median ``value`` over the images and the
        ``count`` of exposures (visits) it rests on, with those visits. A value
        kept per arm takes an image from each spectrograph of an exposure, and
        still counts the exposure once. The m-arm offsets
        (`pfs.drp.qa.metrics.registry.M_ARM_OFFSETS`) have an empty frame
        type, and count the visits of the mode with fewer.
    """
    measured = measurements(iq, lines)
    if exclude is not None and len(exclude):
        keys = list(IMAGE_KEYS)
        dropped = exclude[keys].astype({"visit": "int64", "spectrograph": "int64"}).drop_duplicates()
        measured = measured.merge(dropped.assign(_drop=True), on=keys, how="left")
        measured = measured[measured["_drop"].isna()].drop(columns="_drop")
    keys = list(NORMAL_KEYS)
    rows = []
    relativeChecks = [name for name, check in CHECKS.items() if check.kind == RELATIVE]
    relative = measured[measured["check"].isin(relativeChecks) & np.isfinite(measured["value"])]
    for key, group in relative.groupby(keys, sort=True):
        rows.append(
            dict(zip(keys, key, strict=True))
            | {
                "value": float(group["value"].median()),
                "count": int(group["visit"].nunique()),
                "firstVisit": int(group["visit"].min()),
                "lastVisit": int(group["visit"].max()),
                "sourceVisits": visitRanges(group["visit"]),
            }
        )
    rows += _mArmOffsets(measured)
    return pd.DataFrame(rows, columns=list(CANDIDATE_COLUMNS))


def _mArmOffsets(measured: pd.DataFrame) -> list[dict]:
    """Return the m-arm offset rows: per camera, the median with m minus the median with r."""
    offsets = measured[(measured["check"] == "calibOffset") & np.isfinite(measured["value"])]
    rows = []
    for name, arm in M_ARM_OFFSETS.items():
        if name.endswith("X"):
            use = (offsets["metric"] == "medianDeltaX") & (offsets["frameType"] == QUARTZ)
        else:
            use = (
                (offsets["metric"] == "medianDeltaY")
                & (offsets["frameType"] == ARC)
                & (offsets["part"] == f"y {M_ARM_SPECIES}")
            )
        frame = offsets[use & (offsets["arm"] == arm)]
        for spectrograph, group in frame.groupby("spectrograph", sort=True):
            withM, withR = group[group["withM"]], group[~group["withM"]]
            count = min(withM["visit"].nunique(), withR["visit"].nunique())
            if count == 0:
                continue
            rows.append(
                {
                    "metric": name,
                    "camera": f"{arm}{int(spectrograph)}",
                    "frameType": "",
                    "selector": "",
                    "value": float(withM["value"].median() - withR["value"].median()),
                    "count": int(count),
                    "firstVisit": int(group["visit"].min()),
                    "lastVisit": int(group["visit"].max()),
                    "sourceVisits": visitRanges(group["visit"]),
                }
            )
    return rows


def compare(made: pd.DataFrame, normalValues: pd.DataFrame | None, validFrom: int) -> pd.DataFrame:
    """Set each candidate beside the value in force before ``validFrom``.

    Parameters
    ----------
    made : `pandas.DataFrame`
        `candidates`.
    normalValues : `pandas.DataFrame`, optional
        The collection's stored normal values (`NORMAL_COLUMNS`).
    validFrom : `int`
        The visit the candidates would be valid from.

    Returns
    -------
    `pandas.DataFrame`
        ``made`` with:

        ``previous``
            The value in force for the last visit before ``validFrom``; NaN
            when there is none.
        ``shift``
            The candidate less ``previous``: in per cent of it for a relative
            check's metric, in the metric's units for an m-arm offset.
        ``status``
            ``NEW`` without a previous value; ``TOO_FEW`` for a candidate
            resting on fewer than
            `pfs.drp.qa.metrics.registry.MIN_NORMAL_FRAMES` frames; else
            ``PASS``, ``WARN`` or ``FAIL`` by the limits of the check the
            value is for, in the metric's bad direction. An m-arm offset has
            no limits and is ``PASS``.
    """
    compared = made.reset_index(drop=True).copy()
    previous = _inForce(compared.assign(visit=int(validFrom) - 1), normalValues)["normal"]
    compared["previous"] = previous.to_numpy()
    checks = _relativeChecks()
    shift, status = [], []
    for row in compared.itertuples():
        name = checks.get((row.metric, row.frameType))
        if name is None:
            change = row.value - row.previous
        else:
            change = (row.value / row.previous - 1) * 100 if row.previous else np.nan
        shift.append(change)
        if row.count < MIN_NORMAL_FRAMES:
            status.append("TOO_FEW")
        elif not np.isfinite(row.previous):
            status.append("NEW")
        elif name is None:
            status.append("PASS")
        else:
            check = CHECKS[name]
            bad = change if specFor(row.metric).higherIsWorse else -change
            status.append("FAIL" if bad >= check.fail else "WARN" if bad >= check.warn else "PASS")
    compared["shift"] = shift
    compared["status"] = status
    return compared


def accepted(
    compared: pd.DataFrame,
    collection: str,
    validFrom: int,
    acceptedBy: str,
    reason: str = "",
    when: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the normal values to store, and the candidates refused.

    Parameters
    ----------
    compared : `pandas.DataFrame`
        `compare`.
    collection : `str`
        The collection the values judge.
    validFrom : `int`
        The first visit they judge.
    acceptedBy : `str`
        Who accepts them.
    reason : `str`, optional
        Why a value shifted beyond FAIL is accepted. Without it such a value
        is refused.
    when : `datetime.datetime`, optional
        When they are accepted; now by default.

    Returns
    -------
    stored : `pandas.DataFrame`
        `NORMAL_COLUMNS`.
    refused : `pandas.DataFrame`
        The rows of ``compared`` left out: ``TOO_FEW``, and ``FAIL`` without
        a reason. The previous value stays in force for each.
    """
    refuse = (compared["status"] == "TOO_FEW") | ((compared["status"] == "FAIL") & (not reason))
    stored = compared[~refuse].assign(
        collection=collection,
        validFrom=int(validFrom),
        acceptedBy=acceptedBy,
        acceptedAt=when or datetime.now(),
        reason=np.where(compared.loc[~refuse, "status"] == "FAIL", reason, ""),
    )
    return stored.reindex(columns=list(NORMAL_COLUMNS)).reset_index(drop=True), compared[refuse].reset_index(
        drop=True
    )


def formatComparison(compared: pd.DataFrame, onlyShifted: bool = False) -> str:
    """Return `compare`'s table as text: each value beside the one in force, with its shift.

    Parameters
    ----------
    compared : `pandas.DataFrame`
        `compare`.
    onlyShifted : `bool`, optional
        Leave out the ``PASS`` rows.
    """
    rows = compared[compared["status"] != "PASS"] if onlyShifted else compared
    lines = [
        f"{'metric':<16} {'camera':<6} {'frames':<10} {'lamp/species':<12} {'n':>4} {'value':>9} {'in force':>9} {'shift':>9}  status"
    ]
    for row in rows.itertuples():
        unit = "" if (row.metric, row.frameType) not in _relativeChecks() else " %"
        shift = f"{row.shift:+.2f}{unit}" if np.isfinite(row.shift) else "-"
        previous = f"{row.previous:.4g}" if np.isfinite(row.previous) else "-"
        lines.append(
            f"{row.metric:<16} {row.camera:<6} {row.frameType or '-':<10} {row.selector or '-':<12} "
            f"{row.count:>4d} {row.value:>9.4g} {previous:>9} {shift:>9}  {row.status}"
        )
    counts = compared["status"].value_counts().to_dict()
    lines.append(
        ", ".join(f"{count} {status}" for status, count in sorted(counts.items())) or "no candidates"
    )
    return "\n".join(lines)


def fixedFailures(judged: pd.DataFrame) -> pd.DataFrame:
    """Return the images that fail a fixed check: those `candidates` should leave out.

    Parameters
    ----------
    judged : `pandas.DataFrame`
        `pfs.drp.qa.metrics.gate.judge`.
    """
    fixed = judged["check"].isin([name for name, check in CHECKS.items() if check.kind == FIXED])
    return (
        judged.loc[fixed & (judged["status"] == "FAIL"), list(IMAGE_KEYS)]
        .drop_duplicates()
        .reset_index(drop=True)
    )
