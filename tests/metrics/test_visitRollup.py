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


def image(visit, camera, status, by="", reason="", at="2026-10-26 12:00"):
    """Return a gate row: the image, its verdict, the check that decided it and why."""
    return {
        "visit": visit,
        "arm": camera[0],
        "spectrograph": int(camera[1:]),
        "qaStatus": status,
        "qaDecidedBy": by or None,
        "qaReason": reason or None,
        "decidedAt": pd.Timestamp(at),
    }


def gate(*rows):
    return pd.DataFrame(list(rows))


def expect(pairs):
    return pd.DataFrame([{"visit": visit, "camera": camera} for visit, camera in pairs])


GATE = gate(
    image(100, "r2", "FAIL", "calibOffset", "x offset +2.10 px, limit 2 px"),
    image(100, "b1", "WARN", "width", "width 3.10 px is +3.3 % from normal 3.00 px, limit +3 %"),
    image(100, "r1", "PASS", at="2026-10-26 12:07"),
    image(101, "b1", "PASS"),
    image(101, "r1", "UNKNOWN", reason="no normal value"),
    image(
        102, "n1", "WARN", "scatterX", "NeI x scatter 0.050 px is +25.0 % from normal 0.040 px, limit +20 %"
    ),
    image(102, "b1", "WARN", "width", "width 3.20 px is +3.2 % from normal 3.10 px, limit +3 %"),
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
    assert failed["qaDecidedBy"] == "calibOffset"
    assert failed["qaReason"] == "x offset +2.10 px, limit 2 px"
    assert failed["decidedAt"] == pd.Timestamp("2026-10-26 12:07")


def testUnknownImageDoesNotHideAPass():
    """An image no check could judge is counted, and the visit is still as good as its judged images."""
    passed = row(rollupVisits(GATE), 101)
    assert passed["qaStatus"] == "PASS" and passed["nUnknown"] == 1
    # Why the unjudged image has no verdict is its own row's business: a passed visit names no decider.
    assert passed["decidedByCamera"] == "" and passed["qaDecidedBy"] == "" and passed["qaReason"] == ""


def testDecidingImageIsFirstBySpectrographThenArm():
    warned = row(rollupVisits(GATE), 102)
    assert warned["decidedByCamera"] == "b1" and warned["qaDecidedBy"] == "width"
    # Negative control: not the row that happens to come first in the gate.
    first = GATE[GATE["visit"] == 102].iloc[0]
    assert first["arm"] == "n" and first["qaDecidedBy"] != warned["qaDecidedBy"]


def testTheRollupNamesNoTaskAndNoProvisionalVerdict():
    """One gate decides, by a check: the rollup has no column for a deciding task or a provisional verdict."""
    assert not {"decidedByTask", "provisional"} & set(ROLLUP_COLUMNS)
    # A gate table that still carries them (an old export) is rolled up all the same, without them.
    old = GATE.assign(decidedByTask="imageQualityQa", provisional=True)
    rollup = rollupVisits(old)
    assert list(rollup.columns) == list(ROLLUP_COLUMNS)
    pd.testing.assert_frame_equal(rollup, rollupVisits(GATE))


def testTheRollupNamesTheVisitsFrameType():
    """A visit's frame type tells an UNKNOWN that is not judged by design from one to look at."""
    typed = gate(
        image(200, "b1", "UNKNOWN") | {"frameType": "dark"},
        image(200, "n1", "UNKNOWN") | {"frameType": "dark"},
        image(201, "b1", "PASS") | {"frameType": "iisArc"},
        image(201, "r1", "PASS") | {"frameType": "iisArc"},
        image(201, "r2", "PASS") | {"frameType": "arc"},
        image(202, "b1", "UNKNOWN") | {"frameType": None},
    )
    rollup = rollupVisits(typed)
    assert rollup["frameType"].tolist() == ["dark", "iisArc", ""]
    assert rollup["qaStatus"].tolist() == ["UNKNOWN", "PASS", "UNKNOWN"]
    # A gate table without frame types (a hand-built one) rolls up with none.
    assert rollupVisits(GATE)["frameType"].tolist() == ["", "", ""]


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
    assert row(rollup, 100)["decidedByCamera"] == "r2" and row(rollup, 100)["qaDecidedBy"] == ""


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
