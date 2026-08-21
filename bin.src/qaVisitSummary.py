#!/usr/bin/env python3
"""QA visit summary — query and display per-visit QA results from the Butler.

Provides both a Python API (``VisitSummary``) and a CLI entry point.

Python API example
------------------
>>> from qaVisitSummary import VisitSummary
>>> vs = VisitSummary("/path/to/butler", "output/collection")
>>> summary = vs.getSummary(visit=12345)
>>> print(summary.dmDetectorStats)
>>> print(summary.iqMetrics)
>>> print(summary.driftMetrics)
>>> print(summary.dmResidualStats)

CLI example
-----------
    python bin.src/qaVisitSummary.py \\
        --butler /path/to/butler \\
        --collection output/collection \\
        --visit 12345
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


__all__ = ["VisitSummary", "VisitSummaryResult"]

# Arms and spectrograph modules present in PFS.
_ARMS = ("b", "r", "n", "m")
_SPECTROGRAPHS = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class VisitSummaryResult:
    """Container for all QA DataFrames for a single visit.

    Attributes
    ----------
    visit : int
        The visit ID.
    dmDetectorStats : pd.DataFrame or None
        Cross-detector summary from ``dmCombinedResiduals`` (``dmQaDetectorStats``).
        One row per ``(arm, spectrograph)`` with DM residual and IQ columns.
    iqMetrics : pd.DataFrame or None
        Per-detector image quality metrics from ``imageQualityQa`` (``iqQaMetrics``).
        Concatenated across all available ``(arm, spectrograph)`` quanta.
    driftMetrics : pd.DataFrame or None
        Per-detector drift metrics from ``dmDriftMonitor`` (``dmDriftMetrics``).
        Concatenated across all available ``(arm, spectrograph)`` quanta.
    dmResidualStats : pd.DataFrame or None
        Per-detector, per-species residual statistics from ``dmResiduals``
        (``dmQaResidualStats``).  Concatenated across all available quanta.
    missing : list[str]
        Dataset types that were not found in the collection.
    """

    visit: int
    dmDetectorStats: Optional[pd.DataFrame] = None
    iqMetrics: Optional[pd.DataFrame] = None
    driftMetrics: Optional[pd.DataFrame] = None
    dmResidualStats: Optional[pd.DataFrame] = None
    missing: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def overallStatus(self) -> str:
        """Return the worst ``qaStatus`` across all available datasets.

        Returns
        -------
        str
            ``"PASS"``, ``"WARN"``, ``"FAIL"``, or ``"UNKNOWN"`` when no
            status information is available.
        """
        _level = {"PASS": 0, "WARN": 1, "FAIL": 2, "UNKNOWN": -1}
        worst = "UNKNOWN"
        for df in (self.dmDetectorStats, self.iqMetrics, self.driftMetrics, self.dmResidualStats):
            if df is None or "qaStatus" not in df.columns:
                continue
            for val in df["qaStatus"].dropna():
                if val in _level and _level.get(val, -1) > _level.get(worst, -1):
                    worst = val
        return worst

    def printSummary(self, file=None) -> None:
        """Print a human-readable summary table to *file* (default: stdout).

        Parameters
        ----------
        file : file-like, optional
            Output stream.  Defaults to ``sys.stdout``.
        """
        if file is None:
            file = sys.stdout

        print(f"\n{'='*70}", file=file)
        print(f"  QA Visit Summary — visit {self.visit}", file=file)
        print(f"  Overall status: {self.overallStatus()}", file=file)
        print(f"{'='*70}", file=file)

        if self.missing:
            print(f"\n  [Missing datasets: {', '.join(self.missing)}]", file=file)

        # --- Combined detector stats (best single-table view) ---
        if self.dmDetectorStats is not None:
            vRows = self.dmDetectorStats
            if "visit" in vRows.columns:
                vRows = vRows[vRows["visit"] == self.visit]
            cols = [c for c in ("arm", "spectrograph", "spatialRms", "wavelengthRms",
                                "lineYieldFrac", "medFwhm", "pctFlagged",
                                "fluxJitterPct", "nSaturated", "qaStatus")
                    if c in vRows.columns]
            print(f"\n--- dmQaDetectorStats (cross-detector summary) ---", file=file)
            print(vRows[cols].to_string(index=False), file=file)

        # --- IQ metrics ---
        if self.iqMetrics is not None:
            cols = [c for c in ("visit", "arm", "spectrograph", "medFwhm", "pctFlagged",
                                "fluxJitterPct", "nSaturated", "medDxCenter", "qaStatus")
                    if c in self.iqMetrics.columns]
            print(f"\n--- iqQaMetrics (image quality per detector) ---", file=file)
            print(self.iqMetrics[cols].to_string(index=False), file=file)

        # --- DM residual stats ---
        if self.dmResidualStats is not None:
            cols = [c for c in ("visit", "arm", "spectrograph", "description",
                                "spatialRms", "wavelengthRms", "lineYieldFrac",
                                "medResolution", "qaStatus")
                    if c in self.dmResidualStats.columns]
            print(f"\n--- dmQaResidualStats (per-detector, per-species) ---", file=file)
            print(self.dmResidualStats[cols].to_string(index=False), file=file)

        # --- Drift metrics ---
        if self.driftMetrics is not None:
            cols = [c for c in ("visit", "arm", "spectrograph", "description",
                                "deltaX", "deltaY", "driftMag", "deltaWx",
                                "qaStatus", "recommendedAction")
                    if c in self.driftMetrics.columns]
            print(f"\n--- dmDriftMetrics (daily drift vs static detectormap) ---", file=file)
            print(self.driftMetrics[cols].to_string(index=False), file=file)

        print(f"\n{'='*70}\n", file=file)


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------


class VisitSummary:
    """Query QA results for one or more visits from a Butler collection.

    Parameters
    ----------
    butlerRoot : str
        Path to the Butler repository root.
    collection : str
        Output collection to query (e.g. ``"u/user/output"``).
    instrument : str, optional
        Instrument name.  Defaults to ``"PFS"``.

    Examples
    --------
    >>> vs = VisitSummary("/path/to/butler", "output/collection")
    >>> result = vs.getSummary(12345)
    >>> result.printSummary()
    >>> df = result.dmDetectorStats
    """

    def __init__(self, butlerRoot: str, collection: str, instrument: str = "PFS"):
        import lsst.daf.butler as daf_butler

        self._butler = daf_butler.Butler(butlerRoot, collections=[collection])
        self._collection = collection
        self._instrument = instrument

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _getPerDetector(self, datasetType: str, visit: int) -> Optional[pd.DataFrame]:
        """Fetch and concatenate a per-detector DataFrame across all quanta.

        Parameters
        ----------
        datasetType : str
            Butler dataset type name.
        visit : int
            Visit ID.

        Returns
        -------
        pd.DataFrame or None
            Concatenated DataFrame, or ``None`` if no quanta were found.
        """
        frames = []
        for arm in _ARMS:
            for sm in _SPECTROGRAPHS:
                try:
                    df = self._butler.get(
                        datasetType,
                        visit=visit,
                        arm=arm,
                        spectrograph=sm,
                        instrument=self._instrument,
                    )
                    # Tag with dataId columns if not already present.
                    if "arm" not in df.columns:
                        df = df.copy()
                        df["arm"] = arm
                    if "spectrograph" not in df.columns:
                        df = df.copy()
                        df["spectrograph"] = sm
                    if "visit" not in df.columns:
                        df = df.copy()
                        df["visit"] = visit
                    frames.append(df)
                except Exception:
                    pass
        return pd.concat(frames, ignore_index=True) if frames else None

    def _getInstrumentLevel(self, datasetType: str) -> Optional[pd.DataFrame]:
        """Fetch an instrument-level (aggregated) DataFrame.

        Parameters
        ----------
        datasetType : str
            Butler dataset type name.

        Returns
        -------
        pd.DataFrame or None
        """
        try:
            return self._butler.get(datasetType, instrument=self._instrument)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def getSummary(self, visit: int) -> VisitSummaryResult:
        """Fetch all QA DataFrames for *visit* and return a ``VisitSummaryResult``.

        Parameters
        ----------
        visit : int
            Visit ID to query.

        Returns
        -------
        VisitSummaryResult
            Container with all available QA DataFrames and a list of missing
            dataset types.
        """
        result = VisitSummaryResult(visit=visit)

        # dmQaDetectorStats is instrument-level; filter to this visit afterwards.
        dmDetectorStats = self._getInstrumentLevel("dmQaDetectorStats")
        if dmDetectorStats is not None:
            if "visit" in dmDetectorStats.columns:
                dmDetectorStats = dmDetectorStats[dmDetectorStats["visit"] == visit]
                if dmDetectorStats.empty:
                    dmDetectorStats = None
            result.dmDetectorStats = dmDetectorStats
        else:
            result.missing.append("dmQaDetectorStats")

        # Per-detector datasets.
        iqMetrics = self._getPerDetector("iqQaMetrics", visit)
        if iqMetrics is not None:
            result.iqMetrics = iqMetrics
        else:
            result.missing.append("iqQaMetrics")

        driftMetrics = self._getPerDetector("dmDriftMetrics", visit)
        if driftMetrics is not None:
            result.driftMetrics = driftMetrics
        else:
            result.missing.append("dmDriftMetrics")

        dmResidualStats = self._getPerDetector("dmQaResidualStats", visit)
        if dmResidualStats is not None:
            result.dmResidualStats = dmResidualStats
        else:
            result.missing.append("dmQaResidualStats")

        return result

    def getMultiVisitSummary(self, visits: list[int]) -> pd.DataFrame:
        """Fetch the combined detector stats for multiple visits.

        Queries ``dmQaDetectorStats`` (instrument-level) and filters to the
        requested visits.  This is the most efficient way to build a
        cross-visit summary table.

        Parameters
        ----------
        visits : list[int]
            Visit IDs to include.

        Returns
        -------
        pd.DataFrame
            Rows from ``dmQaDetectorStats`` for the requested visits, or an
            empty DataFrame if the dataset is unavailable.
        """
        df = self._getInstrumentLevel("dmQaDetectorStats")
        if df is None:
            return pd.DataFrame()
        if "visit" in df.columns:
            df = df[df["visit"].isin(visits)]
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print a per-visit QA summary from Butler output DataFrames produced by drpQA.yaml. "
            "Queries dmQaDetectorStats, iqQaMetrics, dmQaResidualStats, and dmDriftMetrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single visit
  python bin.src/qaVisitSummary.py \\
      --butler /path/to/butler \\
      --collection u/user/output \\
      --visit 12345

  # Multiple visits (prints one block per visit)
  python bin.src/qaVisitSummary.py \\
      --butler /path/to/butler \\
      --collection u/user/output \\
      --visit 12345 12346 12347

  # Save to a file
  python bin.src/qaVisitSummary.py \\
      --butler /path/to/butler \\
      --collection u/user/output \\
      --visit 12345 --output summary.txt
""",
    )
    parser.add_argument("--butler", required=True, metavar="PATH", help="Butler repository root path.")
    parser.add_argument(
        "--collection", required=True, metavar="COLLECTION", help="Output collection to query."
    )
    parser.add_argument(
        "--visit",
        required=True,
        nargs="+",
        type=int,
        metavar="VISIT",
        help="One or more visit IDs to summarize.",
    )
    parser.add_argument(
        "--instrument",
        default="PFS",
        metavar="INSTRUMENT",
        help="Instrument name (default: PFS).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write summary to FILE instead of stdout.",
    )
    return parser


def main(argv=None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv : list[str], optional
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 = all PASS, 1 = at least one WARN, 2 = at least one FAIL,
        3 = no QA data found.
    """
    parser = _buildParser()
    args = parser.parse_args(argv)

    try:
        vs = VisitSummary(args.butler, args.collection, instrument=args.instrument)
    except Exception as exc:
        print(f"ERROR: Could not open Butler at {args.butler!r}: {exc}", file=sys.stderr)
        return 3

    _level = {"UNKNOWN": -1, "PASS": 0, "WARN": 1, "FAIL": 2}
    worstLevel = -1

    outFile = None
    try:
        if args.output:
            outFile = open(args.output, "w")

        for visit in args.visit:
            result = vs.getSummary(visit)
            result.printSummary(file=outFile)
            lvl = _level.get(result.overallStatus(), -1)
            if lvl > worstLevel:
                worstLevel = lvl
    finally:
        if outFile is not None:
            outFile.close()

    if worstLevel < 0:
        return 3
    return min(worstLevel, 2)


if __name__ == "__main__":
    sys.exit(main())
