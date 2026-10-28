"""A Tier 1 table kept up to date from qadb while a run is observed (PIPE2D-1924).

`QadbTier1Store` answers the app's queries as `store.Tier1Store` does, from a
table held in memory instead of a file. `refresh` reads from qadb only the
images loaded since the last read (`qadbSource.readQadb`'s ``since``) and
replaces their rows, so a refresh costs what the new visits cost. One store
serves every browser session; a refresh within ``minInterval`` of the last is
skipped, so many sessions polling it make one query.
"""

import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from pfs.drp.qa.dashboard.qadbSource import FIRST_VISIT, LAST_VISIT, readQadb, tier1FromQadb, withSequences
from pfs.drp.qa.dashboard.schema import TIER1_COLUMNS
from pfs.drp.qa.dashboard.serving import firstThresholdsFile, taskThresholds, thresholdsProvenance
from pfs.drp.qa.dashboard.store import Tier1Store

__all__ = ["HST", "QadbTier1Store", "emptyTier1"]

HST = ZoneInfo("Pacific/Honolulu")

_IMAGE = ["visit", "arm", "spectrograph"]


def emptyTier1() -> pd.DataFrame:
    """Return a Tier 1 table with no rows and `TIER1_COLUMNS`' types."""
    return pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in TIER1_COLUMNS.items()})


class QadbTier1Store(Tier1Store):
    """The Tier 1 table of a collection in qadb, re-read as visits are loaded.

    Parameters
    ----------
    read : callable
        ``read(since)`` returns new rows as `qadbSource.readQadb` does: every
        image when ``since`` is None, else those loaded at or after it.
    layers : mapping
        The thresholds each task judges with (`serving.taskThresholds`).
    provenance : mapping
        What the app's provenance bar shows (``collection``, ``thresholds``, ...).
    minInterval : `float`
        Seconds within which a second `refresh` reads nothing.
    clock : callable
        Seconds, for ``minInterval``; `time.monotonic`.
    """

    live = True

    def __init__(
        self,
        read: Callable[[Any], Mapping[str, pd.DataFrame]],
        layers: Mapping[str, list],
        provenance: Mapping,
        minInterval: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._read = read
        self.layers = layers
        self.provenance = dict(provenance)
        self.minInterval = minInterval
        self._clock = clock
        self._lock = threading.Lock()
        self._lastRead: float | None = None
        self.since: Any = None
        self._seenAtSince: set[tuple] = set()
        self.updatedAt: datetime | None = None
        self.version = 0
        self.table = emptyTier1()
        self._visitTimes = pd.DataFrame(columns=["pfs_visit_id", "time_exp_start", "iic_sequence_id"])
        self._db = duckdb.connect()
        self._register()
        self.refresh()

    @classmethod
    def fromQadb(
        cls,
        qadb: Any,
        collection: str,
        first: int = FIRST_VISIT,
        last: int = LAST_VISIT,
        minInterval: float = 15.0,
    ) -> "QadbTier1Store":
        """Return the store of one collection's visits ``first``..``last`` in a qadb (read-only)."""
        layers, sources = taskThresholds()
        provenance = {
            "collection": f"{collection} (qadb, live)",
            "thresholds": thresholdsProvenance(firstThresholdsFile(layers["iqQaMetrics"])),
            "thresholdLayers": [f"{key}: {source}" for key, source in sources.items()],
            "drpQaVersion": _drpQaVersion(),
            "builtAt": "live",
        }
        return cls(
            lambda since: readQadb(qadb, collection, first, last, since), layers, provenance, minInterval
        )

    def _register(self) -> None:
        strings = [column for column, dtype in TIER1_COLUMNS.items() if dtype == "string"]
        frame = self.table.astype(dict.fromkeys(strings, "string"))
        self._db.register("t1", frame)

    def refresh(self) -> bool:
        """Read the images loaded since the last read and replace their rows.

        Returns
        -------
        `bool`
            Whether the table changed. False within ``minInterval`` of the last read.
        """
        with self._lock:
            now = self._clock()
            if self._lastRead is not None and now - self._lastRead < self.minInterval:
                return False
            self._lastRead = now
            tables = self._read(self.since)
            self.updatedAt = datetime.now(HST)
            iq = tables["iqQaMetrics"]
            if iq.empty:
                return False
            # ``since`` is inclusive, so the last read's newest loads come back every time: only
            # an image load not seen before (image and time) is new.
            loads = set(iq[[*_IMAGE, "decidedAt"]].itertuples(index=False, name=None))
            if loads <= self._seenAtSince:
                return False
            new = tier1FromQadb(tables, self.layers)
            judged = pd.MultiIndex.from_frame(iq[_IMAGE])
            kept = self.table[~pd.MultiIndex.from_frame(self.table[_IMAGE]).isin(judged)]
            times = pd.concat([self._visitTimes, tables["visitTimes"]], ignore_index=True)
            self._visitTimes = times.drop_duplicates("pfs_visit_id", keep="last")
            table = pd.concat([kept, new], ignore_index=True)
            table = withSequences(table, self._visitTimes)
            self.table = table.sort_values([*_IMAGE, "source"], kind="stable", ignore_index=True)
            # ``since`` is inclusive: rows loaded in the same instant as the last ones are read again.
            stamps = pd.concat(
                [tables[name]["decidedAt"] for name in ("iqQaMetrics", "dmLineResidualSummary")]
            )
            self.since = stamps.max()
            self._seenAtSince = {load for load in loads if load[-1] == self.since}
            self.version += 1
            self._register()
            return True


def _drpQaVersion() -> str:
    """Return this drp_qa's version, or ``?`` when it can't be read (no git checkout)."""
    try:
        from pfs.drp.qa.comparison.cli import drpQaVersion

        return drpQaVersion()
    except Exception:
        return "?"
