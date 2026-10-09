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


def testNightRunsNoonToNoon():
    when = pd.Series(
        pd.to_datetime(["2026-07-23 11:59", "2026-07-23 12:00", "2026-07-24 03:00", "2026-07-24 12:00"])
    )
    assert list(nightOf(when)) == [
        datetime.date(2026, 7, 22),
        datetime.date(2026, 7, 23),
        datetime.date(2026, 7, 23),
        datetime.date(2026, 7, 24),
    ]


def testPeriodBoundaries(periods):
    run1 = periods["run1"]
    assert run1.start == datetime.datetime(2026, 1, 2, 12)
    assert run1.end == datetime.datetime(2026, 1, 6, 12)  # the morning after the last night
    when = pd.to_datetime(
        [
            "2026-01-02 11:59",  # the morning before the first night
            "2026-01-02 12:00",
            "2026-01-06 05:00",  # the last night's morning
            "2026-01-06 12:00",
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
