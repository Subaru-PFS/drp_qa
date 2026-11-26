"""Tests for `pfs.drp.qa.comparison.runs`."""

import datetime

import pandas as pd
import pytest
import yaml

from pfs.drp.qa.comparison.runs import loadPeriods, nightOf, periodOf


def testPackagedTableLoads():
    periods = loadPeriods()
    assert "run25" in periods
    assert periods["run29"].firstNight == datetime.date(2026, 7, 7)
    assert periods["run29-pre"].run == "run29"
    assert periods["run29-pre"].isPreRun
    assert not periods["run29"].isPreRun
    assert list(periods) == sorted(periods, key=lambda name: periods[name].firstNight)


def testNightRollsOverAtMidnightUtc():
    """A night begins at 14:00 HST, 00:00 UTC, when the raw files' date changes: its exposures share one UTC date."""
    hst = pd.Series(
        pd.to_datetime(
            [
                "2026-07-23 13:59",
                "2026-07-23 14:00",
                "2026-07-24 03:00",
                "2026-07-24 13:00",
                "2026-07-24 14:00",
            ]
        )
    )
    nights = list(nightOf(hst))

    assert nights == [
        datetime.date(2026, 7, 22),
        datetime.date(2026, 7, 23),
        datetime.date(2026, 7, 23),
        datetime.date(2026, 7, 23),
        datetime.date(2026, 7, 24),
    ]
    # Every time of a night has the same UTC date, the day after the night's own.
    utc = (hst + pd.Timedelta(hours=10)).dt.date
    assert all(date == night + datetime.timedelta(days=1) for date, night in zip(utc, nights, strict=True))
    # Negative control: with a noon start, an exposure at 13:00 took the next night, and the UTC date of the one before.
    noon = (hst - pd.Timedelta(hours=12)).dt.date
    assert noon[3] == datetime.date(2026, 7, 24) and utc[3] != noon[3] + datetime.timedelta(days=1)


def testPeriodBoundaries(periods):
    run1 = periods["run1"]
    assert run1.start == datetime.datetime(2026, 1, 2, 14)
    assert run1.end == datetime.datetime(2026, 1, 6, 14)  # the afternoon after the last night
    when = pd.to_datetime(
        [
            "2026-01-02 13:59",  # the afternoon before the first night
            "2026-01-02 14:00",
            "2026-01-06 13:00",  # the last night's darks, the afternoon after
            "2026-01-06 14:00",
            "2026-02-02 20:00",
            "2026-02-11 20:00",
        ]
    )
    assert list(periodOf(when, periods.values())) == [None, "run1", "run1", None, "run2-pre", "run2"]


@pytest.mark.parametrize(
    "runs, message",
    [
        ([{"name": "a", "nights": ["2026-01-05", "2026-01-02"]}], "ends before"),
        (
            [
                {"name": "a", "nights": ["2026-01-01", "2026-01-05"]},
                {"name": "b", "nights": ["2026-01-05", "2026-01-08"]},
            ],
            "overlap",
        ),
        (
            [{"name": "a", "nights": ["2026-01-08", "2026-01-09"], "preRun": ["2026-01-05", "2026-01-08"]}],
            "overlap",
        ),
        ([{"name": "a"}], "needs a name and nights"),
        ([{"name": "a", "nights": ["2026-01-01"]}], "expected \\[first, last\\]"),
    ],
)
def testMalformedTables(tmp_path, runs, message):
    path = tmp_path / "runs.yaml"
    path.write_text(yaml.safe_dump({"runs": runs}))
    with pytest.raises(ValueError, match=message):
        loadPeriods(path)
