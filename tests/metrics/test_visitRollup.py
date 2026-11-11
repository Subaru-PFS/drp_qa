"""Tests for the per-visit rollup of the gate (`pfs.drp.qa.metrics.visitRollup`). Visits are synthetic."""

import pandas as pd

from pfs.drp.qa.metrics.visitRollup import (
    EXIT_FAIL,
    EXIT_NO_DATA,
    EXIT_PASS,
    EXIT_WARN,
    ROLLUP_COLUMNS,
    exitCode,
    rollupVisits,
)


def image(visit, camera, status, task="", by="", reason="", provisional=False, at="2026-10-26 12:00"):
    return {
        "visit": visit,
        "arm": camera[0],
        "spectrograph": int(camera[1:]),
        "qaStatus": status,
        "decidedByTask": task or None,
        "qaDecidedBy": by or None,
        "qaReason": reason or None,
        "provisional": provisional,
        "decidedAt": pd.Timestamp(at),
    }


def gate(*rows):
    return pd.DataFrame(list(rows))


def expect(pairs):
    return pd.DataFrame([{"visit": visit, "camera": camera} for visit, camera in pairs])


GATE = gate(
    image(100, "r2", "FAIL", "dmLineResiduals", "medianDeltaX", "medianDeltaX=0.9 > 0.5"),
    image(100, "b1", "WARN", "imageQualityQa", "medFwhm", "medFwhm=3.1 > 3.0"),
    image(100, "r1", "PASS", at="2026-10-26 12:07"),
    image(101, "b1", "PASS"),
    image(101, "r1", "UNKNOWN"),
    image(102, "n1", "WARN", "imageQualityQa", "pctFlagged", "pctFlagged=12 > 10", provisional=True),
    image(102, "b1", "WARN", "imageQualityQa", "medFwhm", "medFwhm=3.2 > 3.0"),
)


def row(rollup, visit):
    return rollup.set_index("visit").loc[visit]


def testWorstImageDecidesTheVisit():
    rollup = rollupVisits(GATE)
    assert list(rollup.columns) == list(ROLLUP_COLUMNS)
    assert rollup["visit"].tolist() == [100, 101, 102]
    assert rollup["qaStatus"].tolist() == ["FAIL", "PASS", "WARN"]
    # Negative control: not its first image by camera order, which only warned.
    first = GATE[GATE["visit"] == 100].sort_values(["spectrograph", "arm"]).iloc[0]
    assert first["qaStatus"] == "WARN" != row(rollup, 100)["qaStatus"]

    failed = row(rollup, 100)
    assert (failed["nImages"], failed["nFail"], failed["nWarn"], failed["nPass"]) == (3, 1, 1, 1)
    assert failed["decidedByCamera"] == "r2"
    assert failed["decidedByTask"] == "dmLineResiduals"
    assert failed["qaDecidedBy"] == "medianDeltaX"
    assert failed["qaReason"] == "medianDeltaX=0.9 > 0.5"
    assert failed["decidedAt"] == pd.Timestamp("2026-10-26 12:07")


def testUnknownImageDoesNotHideAPass():
    """An image no task could judge is counted, and the visit is still as good as its judged images."""
    passed = row(rollupVisits(GATE), 101)
    assert passed["qaStatus"] == "PASS" and passed["nUnknown"] == 1
    assert passed["decidedByCamera"] == "" and passed["qaReason"] == ""


def testDecidingImageIsFirstBySpectrographThenArm():
    warned = row(rollupVisits(GATE), 102)
    assert warned["decidedByCamera"] == "b1" and warned["qaDecidedBy"] == "medFwhm"
    # Negative control: not the row that happens to come first in the gate.
    assert GATE[GATE["visit"] == 102].iloc[0]["arm"] == "n"
    assert bool(warned["provisional"]) is True
    assert bool(row(rollupVisits(GATE), 100)["provisional"]) is False


def testWithoutExpectedCamerasNothingIsCalledMissing():
    rollup = rollupVisits(GATE)
    assert rollup["nExpected"].isna().all() and rollup["missingCameras"].isna().all()


def testMissingCamerasAreReportedNotJudged():
    expected = expect(
        [
            (100, "b1"),
            (100, "r1"),
            (100, "r2"),
            (101, "b1"),
            (101, "r1"),
            (101, "n1"),
            (101, "b2"),
            (103, "b1"),
        ]
    )
    rollup = rollupVisits(GATE, expected)
    assert row(rollup, 100)["missingCameras"] == "" and row(rollup, 100)["nExpected"] == 3
    # Spectrograph, then arm.
    assert row(rollup, 101)["missingCameras"] == "n1,b2"
    assert row(rollup, 101)["qaStatus"] == "PASS"
    # A visit that exposed and was never judged is listed, UNKNOWN.
    unjudged = row(rollup, 103)
    assert (unjudged["qaStatus"], unjudged["nImages"], unjudged["missingCameras"]) == ("UNKNOWN", 0, "b1")
    # Negative control: without the expected cameras the gate cannot know of it.
    assert 103 not in set(rollupVisits(GATE)["visit"])
    # A judged visit the opdb lists nothing for expects nothing.
    assert row(rollup, 102)["nExpected"] == 0 and row(rollup, 102)["missingCameras"] == ""


def testNoRows():
    empty = rollupVisits(pd.DataFrame(columns=["visit", "arm", "spectrograph", "qaStatus"]))
    assert empty.empty and list(empty.columns) == list(ROLLUP_COLUMNS)
    assert exitCode(empty) == EXIT_NO_DATA


def testMinimalColumns():
    """The verdict columns alone suffice (a hand-built table)."""
    rollup = rollupVisits(GATE[["visit", "arm", "spectrograph", "qaStatus"]])
    assert rollup["qaStatus"].tolist() == ["FAIL", "PASS", "WARN"]
    assert row(rollup, 100)["decidedByTask"] == ""


def testExitCodeIsTheWorstVisit():
    rollup = rollupVisits(GATE)
    assert exitCode(rollup) == EXIT_FAIL == 2
    assert exitCode(rollup[rollup["visit"] != 100]) == EXIT_WARN == 1
    assert exitCode(rollup[rollup["visit"] == 101]) == EXIT_PASS == 0
    unknown = rollupVisits(gate(image(7, "b1", "UNKNOWN")))
    assert exitCode(unknown) == EXIT_NO_DATA == 3
    # Negative control: nothing judged is not a pass.
    assert exitCode(unknown) != EXIT_PASS


def testMediumResolutionCamerasAreNotMissing():
    """The opdb names a medium-resolution exposure's camera ``m1``, as the gate's arm ``m`` does."""
    gate = pd.DataFrame(
        {"visit": [200, 200], "arm": ["b", "m"], "spectrograph": [1, 1], "qaStatus": ["PASS"] * 2}
    )
    rollup = rollupVisits(gate, expect([(200, "b1"), (200, "m1")]))
    assert row(rollup, 200)["missingCameras"] == ""
    # Negative control: the red camera is another camera, and is missed.
    assert row(rollupVisits(gate, expect([(200, "b1"), (200, "r1")])), 200)["missingCameras"] == "r1"


def testNullProvisionalIsNotProvisional():
    gate = pd.DataFrame(
        {
            "visit": [300, 300, 301],
            "arm": ["b", "r", "b"],
            "spectrograph": [1, 1, 1],
            "qaStatus": ["PASS"] * 3,
            "provisional": pd.array([pd.NA, False, True], dtype="boolean"),
        }
    )
    rollup = rollupVisits(gate)
    assert not row(rollup, 300)["provisional"]
    # Negative control: a set flag still shows.
    assert row(rollup, 301)["provisional"]
