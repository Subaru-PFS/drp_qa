"""Tests for `pfs.drp.qa.comparison.findings`."""

import pandas as pd
import pytest

from pfs.drp.qa.comparison.classify import classifyVisits
from pfs.drp.qa.comparison.findings import (
    describeSetup,
    extentOf,
    findings,
    judgeImages,
    lampsOf,
)

THRESHOLDS = pd.DataFrame(
    {
        "metric": ["medFwhm", "nLines"],
        "warn": [2.6, 200.0],
        "fail": [2.8, 100.0],
        "higherIsWorse": [True, False],
        "absolute": [False, False],
        "provenance": ["test", "test"],
    }
)


@pytest.mark.parametrize(
    "cmdStr, lamps",
    [
        (
            "iic scienceArc exptime=5 head='sps iis on=neon warmingTime=150' tail='sps iis off=neon'",
            ["neon (IIS)"],
        ),
        ("iic scienceArc iisNeon=30 duplicate=2", ["neon (IIS)"]),
        ("iic scienceArc hgcd=0.0 argon=10 xenon=0.0 neon=0.0 krypton=0.0 duplicate=3", ["argon"]),
        ("iic scienceTrace halogen=20 duplicate=2", ["halogen"]),
        ("iic dark exptime=300", []),
        (None, []),
    ],
)
def testLampsOf(cmdStr, lamps):
    assert lampsOf(cmdStr) == lamps


def testDescribeSetupHidesScienceNames():
    calibration = pd.Series(
        {
            "sequence_type": "scienceArc",
            "sequence_name": "Arc: Neon",
            "cmd_str": "neon=5",
            "category": "calibration",
        }
    )
    science = pd.Series(
        {
            "sequence_type": "scienceObject",
            "sequence_name": "program field",
            "category": "science",
            "exptime": 900,
        }
    )
    assert describeSetup(calibration) == "scienceArc 'Arc: Neon', neon"
    assert describeSetup(science) == "scienceObject; 900 s"


def _verdicts(bad):
    """Two spectrographs × three arms, ``bad`` the failing (arm, spectrograph)s."""
    rows = [(1, arm, sm, "FAIL" if (arm, sm) in bad else "PASS") for sm in (1, 2) for arm in "brn"]
    return pd.DataFrame(rows, columns=["visit", "arm", "spectrograph", "status"])


@pytest.mark.parametrize(
    "bad, extent",
    [
        ({("b", 1), ("r", 1), ("n", 1)}, "SM1"),
        ({("b", 1), ("b", 2)}, "b arm"),
        ({("b", 1)}, "detector"),
        ({(arm, sm) for arm in "brn" for sm in (1, 2)}, "visit"),
        ({(arm, sm) for arm in "brn" for sm in (1, 2)} - {("n", 2)}, "visit (5 of 6)"),
    ],
)
def testExtent(bad, extent):
    verdicts = _verdicts(bad)
    result = extentOf(verdicts)
    assert set(result[verdicts["status"] == "FAIL"]) == {extent}
    assert set(result[verdicts["status"] == "PASS"]) <= {""}


@pytest.fixture
def defocus(periods, visit, listing):
    """Return an arc sequence of three visits with SM1 defocused, and a focus sweep with few sky lines."""
    rows = [visit(v, f"2026-01-03 18:0{v}", "scienceArc", sequence=50) for v in (1, 2, 3)]
    rows += [
        visit(v, f"2026-01-03 22:0{v - 10}", "scienceObject", sequence=60 + v, name="M39") for v in (11, 12)
    ]
    visits = classifyVisits(listing(*rows), periods.values())
    visits.loc[visits["pfs_visit_id"] > 10, "focusSweep"] = True
    metrics = pd.DataFrame(
        [
            (v, arm, sm, 3.2 if sm == 1 and v < 10 else 2.3, 50 if v > 10 else 400, list(range(600)))
            for v in (1, 2, 3, 11, 12)
            for sm in (1, 2)
            for arm in "br"
        ],
        columns=["visit", "arm", "spectrograph", "medFwhm", "nLines", "fiberIds"],
    )
    return visits, metrics


def testDefocusedSpectrographIsOneFinding(defocus):
    visits, metrics = defocus
    judged = judgeImages(metrics, visits, THRESHOLDS)
    notes = pd.DataFrame(
        {
            "pfs_visit_id": pd.array([2, None], dtype="Int64"),
            "iic_sequence_id": pd.array([None, 50], dtype="Int64"),
            "camera": None,
            "data_flag": None,
            "note": ["SM1 slit moved", "check focus"],
            "source": ["obslog", "obslog_sequence"],
        }
    )
    result = findings(metrics, judged, visits, notes)

    assert set(zip(result["visit"], result["spectrograph"], strict=True)) == {(v, 1) for v in (1, 2, 3)}
    assert set(result["extent"]) == {"SM1, whole sequence"}
    assert result["metrics"].str.contains("medFWHM=3.20px >= fail threshold 2.8px", regex=False).all()
    assert set(result["fibers"]) == {600.0}
    assert result.loc[result["visit"] == 2, "notes"].tolist() == ["SM1 slit moved | check focus"] * 2
    assert result.loc[result["visit"] == 1, "notes"].tolist() == ["check focus"] * 2


def testFocusSweepFluxNotJudged(defocus):
    visits, metrics = defocus
    judged = judgeImages(metrics, visits, THRESHOLDS)
    sweep = judged[judged["visit"] > 10]
    assert (sweep.loc[sweep["metric"] == "nLines", "status"] == "").all()
    assert (sweep.loc[sweep["metric"] == "medFwhm", "status"] == "PASS").all()

    # Negative control: the same visits not marked as a sweep fail on nLines.
    visits = visits.assign(focusSweep=False)
    judged = judgeImages(metrics, visits, THRESHOLDS)
    sweep = judged[judged["visit"] > 10]
    assert (sweep.loc[sweep["metric"] == "nLines", "status"] == "FAIL").all()


def testNoFindings(defocus):
    visits, metrics = defocus
    metrics = metrics.assign(medFwhm=2.3, nLines=400)
    result = findings(metrics, judgeImages(metrics, visits, THRESHOLDS), visits)
    assert result.empty
    assert "extent" in result.columns


def testDailyLitFiberMetricsNotJudged(periods, visit, listing):
    visits = classifyVisits(
        listing(
            visit(1, "2026-01-03 17:00", "scienceArc", sequence=50, name="Arc: Neon"),  # daily, one group lit
            *(visit(10 + i, f"2026-01-03 18:0{i}", "scienceArc", sequence=52) for i in range(3)),  # a set
        ),
        periods.values(),
    )
    metrics = pd.DataFrame(
        [(v, "b", 1, 2.3, 50, 70.0) for v in (1, 10, 11, 12)],
        columns=["visit", "arm", "spectrograph", "medFwhm", "nLines", "pctFlagged"],
    )
    thresholds = pd.concat(
        [THRESHOLDS, THRESHOLDS.iloc[[0]].assign(metric="pctFlagged", warn=50.0, fail=60.0)],
        ignore_index=True,
    )
    judged = judgeImages(metrics, visits, thresholds).set_index(["visit", "metric"])["status"]
    assert judged[(1, "nLines")] == "" and judged[(1, "pctFlagged")] == ""
    assert judged[(1, "medFwhm")] == "PASS"  # still judged against the set thresholds
    # Negative control: the same values in a set fail.
    assert judged[(10, "nLines")] == "FAIL" and judged[(10, "pctFlagged")] == "FAIL"
    assert describeSetup(visits.set_index("pfs_visit_id").loc[1]).startswith("scienceArc 'Arc: Neon' (daily)")
