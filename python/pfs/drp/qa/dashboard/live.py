"""The metrics table of a collection in qadb, kept up to date while a run is observed (PIPE2D-1924).

`QadbMetricsStore` is a `store.MetricsStore` whose `refresh` reads from qadb
only the images loaded since the last read (`qadbSource.readQadb`'s ``since``)
and replaces their rows, so a refresh costs what the new visits cost. One store
serves every browser session; a refresh within ``minInterval`` of the last is
skipped, so many sessions polling it make one query. Offline, the same store
reads a snapshot of qadb (`qadbSource.writeSnapshot`) once.
"""

import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from pfs.drp.qa.dashboard.qadbSource import (
    FIRST_VISIT,
    LAST_VISIT,
    metricsTableFromQadb,
    readQadb,
    readSnapshot,
    withSequences,
)
from pfs.drp.qa.dashboard.serving import firstThresholdsFile, taskThresholds, thresholdsProvenance
from pfs.drp.qa.dashboard.store import MetricsStore

__all__ = ["HST", "QadbMetricsStore", "qadbProvenance"]

HST = ZoneInfo("Pacific/Honolulu")

_IMAGE = ["visit", "arm", "spectrograph"]


def qadbProvenance(collection: str, **extra: Any) -> dict:
    """Return the provenance of a collection read from qadb: the thresholds and drp_qa judging it, and ``extra``."""
    layers, sources = taskThresholds()
    return {
        "collection": collection,
        "thresholds": thresholdsProvenance(firstThresholdsFile(layers["iqQaMetrics"])),
        "thresholdLayers": [f"{key}: {source}" for key, source in sources.items()],
        "drpQaVersion": _drpQaVersion(),
        **extra,
    }


class QadbMetricsStore(MetricsStore):
    """The metrics table of a collection in qadb, re-read as visits are loaded.

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
        self.minInterval = minInterval
        self._clock = clock
        self._lock = threading.Lock()
        self._lastRead: float | None = None
        self.since: Any = None
        self._seenAtSince: set[tuple] = set()
        self.updatedAt: datetime | None = None
        self.version = 0
        self._visitTimes = pd.DataFrame(columns=["pfs_visit_id", "time_exp_start", "iic_sequence_id"])
        super().__init__(provenance=provenance)
        self.refresh()

    @classmethod
    def fromQadb(
        cls,
        qadb: Any,
        collection: str,
        first: int = FIRST_VISIT,
        last: int = LAST_VISIT,
        minInterval: float = 15.0,
    ) -> "QadbMetricsStore":
        """Return the store of one collection's visits ``first``..``last`` in a qadb (read-only)."""
        return cls(
            lambda since: readQadb(qadb, collection, first, last, since),
            taskThresholds()[0],
            qadbProvenance(f"{collection} (qadb, live)", builtAt="live"),
            minInterval,
        )

    @classmethod
    def fromSnapshot(cls, directory: Path | str) -> "QadbMetricsStore":
        """Return the store of a qadb snapshot (`qadbSource.writeSnapshot`), read once; it never refreshes.

        The stored verdicts are the snapshot's; the limits shown beside each value are this drp_qa's.
        """
        tables, provenance = readSnapshot(directory)
        none = {name: frame.iloc[:0] for name, frame in tables.items()}
        store = cls(lambda since: tables if since is None else none, taskThresholds()[0], provenance, 0)
        store.live = False
        return store

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
            new = metricsTableFromQadb(tables, self.layers)
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
