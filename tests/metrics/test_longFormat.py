"""Tests for the long-format per-species metrics."""

import math

import pandas as pd

from pfs.drp.qa.metrics.longFormat import SPECIES_COLUMNS, speciesFrame, speciesStats

DATA_ID = {"instrument": "PFS", "visit": 12345, "arm": "r", "spectrograph": 1}


def testSchema():
    """Fixed columns and dtypes, one row per species and metric, sorted by species."""
    frame = speciesFrame(DATA_ID, {"NeI": (0.02, 0.03), "ArI": (0.05, math.nan)})
    assert list(frame.columns) == list(SPECIES_COLUMNS)
    assert frame.dtypes.astype(str).to_dict() == SPECIES_COLUMNS
    assert list(frame["description"]) == ["ArI", "ArI", "NeI", "NeI"]
    assert list(frame["metric"]) == ["fitXRms", "fitYRms"] * 2
    assert (frame["visit"] == 12345).all()


def testEmptyKeepsSchema():
    """No species gives no rows, and concatenation keeps the dtypes."""
    empty = speciesFrame(DATA_ID, {})
    assert empty.empty and empty.dtypes.astype(str).to_dict() == SPECIES_COLUMNS
    both = pd.concat([empty, speciesFrame(DATA_ID, {"HgI": (0.01, 0.02)})], ignore_index=True)
    assert both.dtypes.astype(str).to_dict() == SPECIES_COLUMNS


def testRoundTrip():
    """`speciesStats` inverts `speciesFrame`, NaN included."""
    stats = {"NeI": (0.02, 0.03), "ArI": (0.05, math.nan)}
    back = speciesStats(speciesFrame(DATA_ID, stats))
    assert set(back) == set(stats)
    assert back["NeI"] == stats["NeI"]
    assert back["ArI"][0] == 0.05 and math.isnan(back["ArI"][1])
