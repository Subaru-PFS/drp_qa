"""Judge numbers already in qadb, and make, compare and accept a collection's normal values.

Both work on stored numbers alone (`pfs.drp.qa.qadb.reader.readQadb`): no
Butler, and no LSST stack.

- `judgeStored` gives every image of a visit range its verdict again, with the
  limits as they are now (`pfs.drp.qa.metrics.registry.CHECKS`) and the normal
  values in qadb. It is what follows a change of limit.
- `makeValues` makes candidate normal values from the calibration frames named
  (the visits a run's calibrations are built from), sets each beside the value
  in force, and, when asked to store them, writes those accepted to
  ``qa_normal_value`` and judges again every frame they cover.

Nothing here imports the LSST stack.
"""

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import pandas as pd

from pfs.drp.qa.metrics.gate import IMAGE_KEYS, gate, judge, measurements
from pfs.drp.qa.metrics.normalValues import accepted, candidates, compare, fixedFailures
from pfs.drp.qa.qadb.reader import LAST_VISIT, PROCESSING_ID, readGate, readNormalValues, readQadb
from pfs.drp.qa.qadb.tables import NORMAL_VALUE_KEYS, QA_GATE, gateRows, normalValueRows
from pfs.drp.qa.qadb.writer import upsert

__all__ = ["judgeStored", "makeValues", "storedNumbers"]


def storedNumbers(
    qadb: Any, collection: str, visitRanges: Sequence[tuple[int, int]], arms: Sequence[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Read a collection's numbers for some visit ranges, as `pfs.drp.qa.metrics.gate.gate` takes them.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The qadb, read-only.
    collection : `str`
        The collection, as loaded.
    visitRanges : sequence of (`int`, `int`)
        First and last visit of each range, inclusive.
    arms : sequence of `str`, optional
        Only these arms.

    Returns
    -------
    `dict` [`str`, `pandas.DataFrame`]
        ``iq``, ``lines`` and ``twilight``, each with
        `pfs.drp.qa.qadb.reader.PROCESSING_ID`.
    """
    first, last = min(lo for lo, _ in visitRanges), max(hi for _, hi in visitRanges)
    tables = readQadb(qadb, collection, first, last)

    def select(table: pd.DataFrame) -> pd.DataFrame:
        keep = pd.Series(False, index=table.index)
        for lo, hi in visitRanges:
            keep |= table["visit"].between(int(lo), int(hi))
        if arms is not None:
            keep &= table["arm"].isin(list(arms))
        return table[keep].reset_index(drop=True)

    return {
        "iq": select(tables["iqQaMetrics"]),
        "lines": select(tables["dmLineResidualMetrics"]),
        "twilight": select(tables["twilightMetrics"]),
    }


def judgeStored(
    qadb: Any,
    collection: str,
    first: int,
    last: int = LAST_VISIT,
    mode: str = "backfill",
    dryRun: bool = False,
    log: Callable[[str], None] = print,
    now: Callable[[], datetime] = datetime.now,
) -> pd.DataFrame:
    """Judge again every image of a collection in a visit range, from the numbers qadb holds.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The qadb, with a user that may write unless ``dryRun``.
    collection : `str`
        The collection, as loaded.
    first, last : `int`
        The visit range, inclusive.
    mode : `str`, optional
        Recorded in ``qa_gate``.
    dryRun : `bool`, optional
        Judge and report, and write nothing.
    log : callable, optional
        Where progress goes.
    now : callable, optional
        The clock, for ``decided_at``.

    Returns
    -------
    `pandas.DataFrame`
        `pfs.drp.qa.metrics.gate.gate`'s verdicts, one row per image.
    """
    numbers = storedNumbers(qadb, collection, [(int(first), int(last))])
    verdicts = gate(**numbers, normalValues=readNormalValues(qadb, collection))
    if verdicts.empty:
        log(f"no numbers for visits {first} to {last} in {collection}")
        return verdicts
    # The verdict replaces the image's latest gate row, the one readers take; an image with none gets its
    # first table's processing, as the loader gives it.
    keys = list(IMAGE_KEYS)
    processing = pd.concat(
        [
            readGate(qadb, collection, first, last)[[*keys, PROCESSING_ID]],
            *(table[[*keys, PROCESSING_ID]] for table in numbers.values() if len(table)),
        ],
        ignore_index=True,
    ).drop_duplicates(keys)
    keyed = verdicts.merge(processing.rename(columns={PROCESSING_ID: "run"}), on=keys, how="left")
    ids = {int(value): int(value) for value in keyed["run"].unique()}
    rows = gateRows(keyed.assign(run=keyed["run"].astype(int)), ids, mode, now())
    tally = verdicts["qaStatus"].value_counts().to_dict()
    counts = ", ".join(f"{tally[s]} {s}" for s in ("PASS", "WARN", "FAIL", "UNKNOWN") if s in tally)
    if dryRun:
        log(f"{len(rows)} images judged, nothing written: {counts}")
        return verdicts
    upsert(qadb, "qa_gate", rows, list(QA_GATE[:3]))
    log(f"qa_gate: {len(rows)} images judged again: {counts}")
    return verdicts


def makeValues(
    qadb: Any,
    collection: str,
    visitRanges: Sequence[tuple[int, int]],
    validFrom: int | None = None,
    arms: Sequence[str] | None = None,
    store: bool = False,
    acceptedBy: str = "",
    reason: str = "",
    log: Callable[[str], None] = print,
    now: Callable[[], datetime] = datetime.now,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make a collection's normal values from its calibration frames, and store those accepted.

    Parameters
    ----------
    qadb : `pfs.utils.database.qadb.QaDB`
        The qadb, with a user that may write when ``store``.
    collection : `str`
        The collection, as loaded.
    visitRanges : sequence of (`int`, `int`)
        The calibration frames: first and last visit of each range, inclusive.
    validFrom : `int`, optional
        The first visit the values judge; default the visit after the last
        of ``visitRanges``.
    arms : sequence of `str`, optional
        Only these arms, for a calibration set of one camera group.
    store : `bool`, optional
        Write the accepted values, then judge again every frame from
        ``validFrom`` on. Without it nothing is written.
    acceptedBy : `str`, optional
        Who accepts them; needed to store.
    reason : `str`, optional
        Why values shifted beyond FAIL are accepted; without it they are
        refused, and the values in force stay.
    log : callable, optional
        Where progress goes.
    now : callable, optional
        The clock.

    Returns
    -------
    compared : `pandas.DataFrame`
        `pfs.drp.qa.metrics.normalValues.compare`: every candidate beside the
        value in force.
    stored : `pandas.DataFrame`
        The values accepted (`pfs.drp.qa.metrics.normalValues.NORMAL_COLUMNS`):
        written when ``store``.
    refused : `pandas.DataFrame`
        The candidates left out.
    """
    if store and not acceptedBy:
        raise ValueError("Storing normal values needs the name of who accepts them")
    validFrom = max(hi for _, hi in visitRanges) + 1 if validFrom is None else int(validFrom)
    numbers = storedNumbers(qadb, collection, visitRanges, arms)
    inForce = readNormalValues(qadb, collection)
    # The calibration frames are first judged by the values in force: the start-of-run check. A frame that
    # fails a fixed check is not a calibration frame to take a normal value from.
    judged = judge(measurements(numbers["iq"], numbers["lines"]), inForce[inForce["validFrom"] < validFrom])
    failed = fixedFailures(judged)
    if len(failed):
        log(f"{len(failed)} images fail a fixed check and are left out of the normal values")
    made = candidates(numbers["iq"], numbers["lines"], exclude=failed)
    compared = compare(made, inForce, validFrom)
    stored, refused = accepted(compared, collection, validFrom, acceptedBy, reason=reason, when=now())
    if store and len(stored):
        upsert(qadb, "qa_normal_value", normalValueRows(stored), list(NORMAL_VALUE_KEYS))
        log(f"qa_normal_value: {len(stored)} values valid from visit {validFrom}")
        judgeStored(qadb, collection, validFrom, log=log, now=now)
    return compared, stored, refused
