"""Tests for the long-format per-species metrics."""

import math

import pandas as pd
import pytest

from pfs.drp.qa.metrics.longFormat import SPECIES_COLUMNS, parseSpeciesStats, speciesFrame, speciesStats

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


def testSoftening():
    """Four values per species give rows for the softening too; the round trip keeps the RMS."""
    frame = speciesFrame(DATA_ID, {"NeI": (0.02, 0.03, 0.01, 0.0), "Trace": (0.04, math.nan, 0.02, 0.0)})
    assert list(frame["metric"]) == ["fitXRms", "fitYRms", "fitXSoften", "fitYSoften"] * 2
    back = speciesStats(frame)
    assert back == {"NeI": (0.02, 0.03)}
    # Negative control: Trace is in the frame, only the lamp-species mapping leaves it out.
    assert "Trace" in set(frame["description"])


@pytest.mark.parametrize(
    "line, expected",
    [
        (
            "Stats for NeI: chi2=0.9 dof=30 xRMS=0.028 yRMS=0.038 xSoften=0.011 ySoften=0.000 from 31 lines",
            ("NeI", (0.028, 0.038, 0.011, 0.0)),
        ),
        (
            "Stats for Trace: arm=b spectrograph=1 chi2=0.9 dof=30 xRMS=0.05 yRMS=nan xSoften=0.02 ySoften=0",
            ("Trace", (0.05, math.nan, 0.02, 0.0)),
        ),
        ("Stats for HgI: chi2=0.9 dof=30 xRMS=0.028 yRMS=0.038", ("HgI", (0.028, 0.038, math.nan, math.nan))),
        ("Stats for fiberId=42: chi2=0.8 dof=10 xRMS=0.025 yRMS=0.035 xSoften=0 ySoften=0", None),
        ("Final result: chi2=0.9 dof=30 xRMS=0.028 yRMS=0.038 xSoften=0 ySoften=0 from 3 lines", None),
    ],
)
def testParseSpeciesStats(line, expected):
    parsed = parseSpeciesStats(line)
    if expected is None:
        assert parsed is None
        return
    assert parsed[0] == expected[0]
    assert parsed[1] == pytest.approx(expected[1], nan_ok=True)
