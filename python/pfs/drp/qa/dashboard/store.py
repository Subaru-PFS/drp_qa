"""Query a Tier 1 table with DuckDB: what the dashboard reads, and nothing else.

Each method is one SQL query over the Parquet file `serving.writeTier1` wrote,
with bound parameters, returning a DataFrame. No Butler, no opdb.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from pfs.drp.qa.dashboard.schema import IMAGE_STATUS_ORDER, NOTE, VERDICT, readProvenance

__all__ = ["VERDICT_SOURCES", "Tier1Store", "visitTimeline"]

#: Sources whose verdicts roll up into an image's status.
VERDICT_SOURCES = ("iqQaMetrics", "dmLineResidualSummary")

# Worst verdict of an image over VERDICT_SOURCES. Standing in for the visit
# rollup of PIPE2D-1923 (qaVisitSummary), which should store it in Tier 1.
_RANK = "CASE status WHEN 'FAIL' THEN 3 WHEN 'WARN' THEN 2 WHEN 'PASS' THEN 1 ELSE 0 END"
_IMAGES_SQL = f"""
SELECT visit, arm, spectrograph,
    any_value(obsType) AS obsType, any_value(seqName) AS seqName,
    any_value(sequenceKey) AS sequenceKey, any_value(sequenceSource) AS sequenceSource,
    any_value(timeExpStart) AS timeExpStart,
    max({_RANK}) AS rank,
    max(CASE WHEN source = 'iqQaMetrics' THEN status END) AS iqStatus,
    max(CASE WHEN source = 'dmLineResidualSummary' THEN status END) AS lineStatus,
    string_agg(CASE WHEN reason <> '' THEN source || ': ' || reason END, ' | ') AS reasons
FROM t1
WHERE metric = $verdict AND source IN ('iqQaMetrics', 'dmLineResidualSummary')
    AND visit BETWEEN $first AND $last
GROUP BY visit, arm, spectrograph
ORDER BY visit, arm, spectrograph
"""


class Tier1Store:
    """A Tier 1 Parquet file, queried with DuckDB.

    Parameters
    ----------
    path : path
        The file `pfs.drp.qa.dashboard.serving.writeTier1` wrote.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.provenance = readProvenance(self.path)
        self._db = duckdb.connect()
        # A view, not a copy: every query reads the file.
        self._db.execute(f"CREATE VIEW t1 AS SELECT * FROM read_parquet('{self.path.as_posix()}')")

    def _query(self, sql: str, **params) -> pd.DataFrame:
        frame = self._db.execute(sql, params).df()
        # Plain object strings: HoloViews reshapes columns, which Arrow-backed strings refuse.
        strings = frame.select_dtypes(include=["string"]).columns
        return frame.astype(dict.fromkeys(strings, object))

    def visits(self) -> pd.DataFrame:
        """Return one row per visit: ``visit``, ``obsType``, ``seqName``, sequence and time."""
        return self._query(
            """
            SELECT visit, any_value(obsType) AS obsType, any_value(seqName) AS seqName,
                any_value(sequenceKey) AS sequenceKey, any_value(sequenceSource) AS sequenceSource,
                any_value(timeExpStart) AS timeExpStart
            FROM t1 GROUP BY visit ORDER BY visit
            """
        )

    def images(self, first: int = 0, last: int = 2**31) -> pd.DataFrame:
        """Return one row per image of visits ``first``..``last`` with its rolled-up ``status``.

        ``status`` is the worst of the stored verdicts of `VERDICT_SOURCES`,
        ``UNKNOWN`` when none judged the image.
        """
        images = self._query(_IMAGES_SQL, verdict=VERDICT, first=first, last=last)
        images["status"] = [IMAGE_STATUS_ORDER[int(rank)] for rank in images.pop("rank")]
        images["reasons"] = images["reasons"].fillna("")
        return images

    def notes(self, first: int = 0, last: int = 2**31) -> pd.DataFrame:
        """Return the opdb notes on visits ``first``..``last``: ``visit``, ``arm``, ``spectrograph``, ``source``, ``note``."""
        return self._query(
            """
            SELECT DISTINCT visit, arm, spectrograph, source, reason AS note
            FROM t1 WHERE metric = $note AND visit BETWEEN $first AND $last
            ORDER BY visit, arm, spectrograph, source, note
            """,
            note=NOTE,
            first=first,
            last=last,
        )

    def visitMetrics(self, visit: int) -> pd.DataFrame:
        """Return every row of one visit, verdicts first."""
        return self._query(
            """
            SELECT * FROM t1 WHERE visit = $visit
            ORDER BY arm, spectrograph, metric <> $verdict, source, metric, reference, statusType, description
            """,
            visit=visit,
            verdict=VERDICT,
        )

    def trend(
        self,
        source: str,
        metric: str,
        first: int = 0,
        last: int = 2**31,
        reference: str | None = None,
        statusType: str | None = None,
    ) -> pd.DataFrame:
        """Return one metric's values over visits ``first``..``last``.

        Parameters
        ----------
        source, metric : `str`
            The table and metric, e.g. ``"iqQaMetrics"``, ``"medFwhm"``.
        first, last : `int`
            The visit range, inclusive.
        reference, statusType : `str`, optional
            For ``dmLineResidualMetrics``: ``calib`` or ``visit``, and
            ``BOTH``, ``USED`` or ``RESERVED``.

        Returns
        -------
        `pandas.DataFrame`
            One row per image, population and species, sorted by visit.
        """
        return self._query(
            """
            SELECT visit, arm, spectrograph, obsType, seqName, description, value, status, warn, fail,
                population, timeExpStart
            FROM t1
            WHERE source = $source AND metric = $metric AND visit BETWEEN $first AND $last
                AND ($reference IS NULL OR reference = $reference)
                AND ($statusType IS NULL OR statusType = $statusType)
                AND isfinite(value)
            ORDER BY visit, arm, spectrograph
            """,
            source=source,
            metric=metric,
            first=first,
            last=last,
            reference=reference,
            statusType=statusType,
        )


def visitTimeline(visits: pd.DataFrame) -> pd.DataFrame:
    """Return each visit's place in time: ``visit``, ``position``, ``time``, ``night``, ``clock``.

    ``position`` counts the visits in order from 0; ``time`` is the start of the visit's first SpS
    exposure (HST, from the opdb listing; NaT without it); ``night`` is the date of the evening the
    night began (``time`` less 12 hours), so a night's visits share it across midnight.
    """
    timeline = visits[["visit", "timeExpStart"]].sort_values("visit").reset_index(drop=True)
    timeline = timeline.rename(columns={"timeExpStart": "time"})
    timeline["time"] = pd.to_datetime(timeline["time"])
    timeline["position"] = np.arange(len(timeline))
    timeline["night"] = (timeline["time"] - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d").fillna("")
    timeline["clock"] = timeline["time"].dt.strftime("%H:%M").fillna("")
    return timeline
