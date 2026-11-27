"""The report on candidate ``fiberProfiles`` calibrations, and the command that measures them."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.comparison import calibProfilesCli
from pfs.drp.qa.comparison.butlerQueries import readProfileQa
from pfs.drp.qa.comparison.calibProfiles import (
    DETECTOR_COLUMNS,
    buildProfileReport,
    calibrationSources,
    detectorTable,
    fibersOverLimit,
    negativeProfiles,
    overLimit,
    whatToDo,
)
from pfs.drp.qa.metrics.fiberProfiles import METRICS, PROFILE_MINIMUM_LIMIT, WING_EXCESS_LIMIT


def calibration(
    arm, spectrograph, excess=None, replaced=None, visit0=153013, count=20, first=1, run="u/me/profilesGen/1"
):
    """Return the two tables ``fiberProfilesQa`` stores for one detector.

    ``excess`` gives the wing excess of chosen fibers; ``replaced`` names the copies and their donors.
    """
    excess, replaced = excess or {}, replaced or {}
    fiberId = np.arange(first, first + count)
    wingExcess = np.array([excess.get(int(ff), 0.0005) for ff in fiberId])
    fibers = pd.DataFrame(
        {
            "visit": 156021,
            "arm": arm,
            "spectrograph": spectrograph,
            "calibVisit0": visit0,
            "calibObsDate": "2026-11-02",
            "calibRun": run,
            "fiberId": fiberId,
            "fwhm": 3.04,
            "wingFrac": 0.012 + wingExcess,
            "wingExcess": wingExcess,
            "replaced": [int(ff) in replaced for ff in fiberId],
            "donors": [replaced.get(int(ff), "") for ff in fiberId],
        }
    )
    measured = fibers[~fibers["replaced"]]
    values = dict.fromkeys(METRICS, 0.0) | {
        "nProfiles": count,
        "nReplaced": len(replaced),
        "medProfileFwhm": 3.04,
        "medWingFrac": float(measured["wingFrac"].median()),
        "maxWingExcess": float(measured["wingExcess"].max()),
    }
    metrics = pd.DataFrame(
        {
            "visit": 156021,
            "arm": arm,
            "spectrograph": spectrograph,
            "calibVisit0": visit0,
            "calibRun": run,
            "metric": list(values),
            "value": list(values.values()),
        }
    )
    return fibers, metrics


def joined(*detectors):
    """Return several detectors' tables concatenated, as the report reads them."""
    return tuple(pd.concat(tables, ignore_index=True) for tables in zip(*detectors, strict=True))


#: A candidate with a neighbour's light on r4 (fiber 9, and its copy 8), against a clean calibration in force.
NEW = joined(calibration("r", 4, {9: 0.19, 8: 0.06, 3: 0.004}, {8: "6 7 9"}), calibration("b", 1))
ASIS = joined(calibration("r", 4, visit0=149068), calibration("b", 1, visit0=149068))


def testOverLimitIsOneRowPerFiberAndArm():
    over = overLimit(NEW[0])
    assert list(over["fiberId"]) == [8, 9] and set(over["detector"]) == {"r4"}
    assert list(over["replaced"]) == [True, False] and over["donors"].iloc[0] == "6 7 9"
    # Negative control: fiber 3, at 0.004, is under the limit and comes in only below it.
    assert WING_EXCESS_LIMIT == 0.01
    assert 3 in set(overLimit(NEW[0], limit=0.003)["fiberId"])
    assert overLimit(ASIS[0]).empty


def testFibersOverLimitHaveAnArmPerColumn():
    """One fiber, over the limit in two arms, is one row; a copy is told from a measured profile."""
    both = joined(
        calibration("b", 4, {9: 0.07}),
        calibration("r", 4, {9: 0.19, 8: 0.06}, {8: "6 7 9"}),
        calibration("n", 4, {9: 0.21, 12: 0.03}),
    )
    table = fibersOverLimit(both[0])
    assert list(table.columns) == ["spectrograph", "fiberId", "b", "r", "n", "copy", "donors"]
    assert list(table["fiberId"]) == [8, 9, 12]  # by fiber, not by arm
    nine = table.set_index("fiberId").loc[9]
    assert (nine["b"], nine["r"], nine["n"]) == pytest.approx((0.07, 0.19, 0.21))
    eight = table.set_index("fiberId").loc[8]
    assert pd.isna(eight["b"]) and eight["r"] == pytest.approx(0.06) and bool(eight["copy"])
    assert eight["donors"] == "6 7 9" and not bool(nine["copy"])
    assert fibersOverLimit(ASIS[0]).empty


def testNewIsAgainstTheCalibrationInForce():
    before = joined(calibration("r", 4, {12: 0.03}, visit0=149068), calibration("b", 1, visit0=149068))
    after = joined(calibration("r", 4, {9: 0.19, 12: 0.03}), calibration("b", 1))
    table = fibersOverLimit(after[0], before[0]).set_index("fiberId")
    assert bool(table.loc[9, "new"]) and not bool(table.loc[12, "new"])
    # Negative control: against a clean calibration both are new.
    assert fibersOverLimit(after[0], ASIS[0])["new"].all()


def testDetectorTableCountsMeasuredAndCopiesApart():
    table = detectorTable(*NEW, ASIS[0])
    assert list(table.columns) == list(DETECTOR_COLUMNS)
    assert list(table["detector"]) == ["b1", "r4"]  # blue first, then red
    r4 = table.set_index("detector").loc["r4"]
    assert (r4["nOverMeasured"], r4["nOverCopies"], r4["nReplaced"]) == (1, 1, 1)
    # Candidate minus previous, over the fibers both measured: one of 19 rose, so the median did not move.
    assert r4["wingFracChange"] == pytest.approx(0.0, abs=1e-9)
    assert table.set_index("detector").loc["b1", "nOverMeasured"] == 0
    # Without a calibration to set it against there is no change.
    assert detectorTable(*NEW)["wingFracChange"].isna().all()


def testWhatToDoNamesTheFibersNotToCertify():
    byFiber, detectors = fibersOverLimit(NEW[0], ASIS[0]), detectorTable(*NEW, ASIS[0])
    items = whatToDo(byFiber, detectors, [], "asis")
    assert len(items) == 2
    assert items[0].startswith(
        "**Do not certify as built: 1 measured fiber is over the limit and not in asis.**"
    )
    assert items[0].endswith("Spectrograph 4: 9.")
    assert items[1].startswith("1 copy is over the limit. Nothing to do for them directly")
    # A clean candidate: one item, saying so.
    clean = whatToDo(fibersOverLimit(ASIS[0], ASIS[0]), detectorTable(*ASIS, ASIS[0]), [], "asis")
    assert clean == [next(iter(clean))] and clean[0].startswith("Nothing here stands against certifying")


def testWhatToDoSeparatesWhatWasAlreadyThere():
    before = joined(calibration("r", 4, {12: 0.03}, visit0=149068))
    after = joined(calibration("r", 4, {9: 0.19, 12: 0.03}))
    items = whatToDo(fibersOverLimit(after[0], before[0]), detectorTable(*after, before[0]), [], "asis")
    assert "1 measured fiber is over the limit and not in asis" in items[0] and "4: 9." in items[0]
    assert (
        items[1].startswith("1 measured fiber is over the limit as already in asis") and "4: 12." in items[1]
    )


def testWhatToDoNamesAWholeDetectorThatMoved():
    """Every fiber of a detector up by 0.015: no fiber stands out, so the detector is named."""
    before = joined(calibration("b", 4, visit0=149068))
    fibers, metrics = calibration("b", 4)
    fibers = fibers.assign(wingFrac=fibers["wingFrac"] + 0.015)
    metrics = metrics.assign(
        value=[
            v + 0.015 if m == "medWingFrac" else v
            for m, v in zip(metrics["metric"], metrics["value"], strict=True)
        ]
    )
    items = whatToDo(
        fibersOverLimit(fibers, before[0]), detectorTable(fibers, metrics, before[0]), [], "asis"
    )
    assert items == [next(iter(items))] and items[0].startswith(
        "**b4: the whole detector's median wing fraction changed by +0.015**"
    )
    # Negative control: a move of half the limit is not named.
    small = fibers.assign(wingFrac=fibers["wingFrac"] - 0.010)
    assert whatToDo(fibersOverLimit(small, before[0]), detectorTable(small, metrics, before[0]), [], "asis")[
        0
    ].startswith("Nothing")


def testWhatToDoStopsAtMissingAndUnmeasuredProfiles():
    fibers, metrics = calibration("r", 4)
    metrics = metrics.assign(
        value=[
            2.0 if m == "nMissing" else 1.0 if m == "nUnmeasured" else v
            for m, v in zip(metrics["metric"], metrics["value"], strict=True)
        ]
    )
    items = whatToDo(fibersOverLimit(fibers), detectorTable(fibers, metrics), [])
    assert items[0] == "**Do not certify: 2 expected fibers have no profile** (r4)."
    assert items[1].startswith("**Do not certify: 1 profile could not be measured** (r4)")


def withMinimum(tables, lowest):
    """Return a detector's tables with the stored ``minimum`` column, ``lowest`` giving chosen fibers' values."""
    fibers, metrics = tables
    fibers = fibers.assign(
        minimum=[lowest.get(int(ff), -0.002) for ff in fibers["fiberId"]], placeholder=False
    )
    return fibers, metrics


def testProfilesFarBelowZeroAreListed():
    """A profile fitted where the detector is masked holds values far below zero; it is named with its detector."""
    fibers, metrics = withMinimum(calibration("n", 2, replaced={7: "5 6 8"}), {5: -1.49, 7: -0.5, 11: -0.05})
    negative = negativeProfiles(fibers)
    assert list(negative["fiberId"]) == [5, 7] and list(negative["detector"]) == ["n2", "n2"]
    assert list(negative["copy"]) == [False, True] and negative["minimum"].iloc[0] == pytest.approx(-1.49)
    # Negative control: a profile at -0.05, within what noise gives, is not listed, and nor is one at the limit.
    assert 11 not in set(negative["fiberId"])
    assert negativeProfiles(fibers.assign(minimum=PROFILE_MINIMUM_LIMIT)).empty
    # A table without the column (an older one) lists nothing.
    assert negativeProfiles(fibers.drop(columns="minimum")).empty
    report = buildProfileReport({"new": (fibers, metrics)}, {"new": "`x`"}, "t")
    sections = [line for line in report.splitlines() if line.startswith("## ")]
    assert "## new: profiles far below zero" in sections
    assert "| n2 | 5 | -1.49 |  |" in report and "| n2 | 7 | -0.50 | 5 6 8 |" in report
    # Negative control: the same detector without them has no such section, and nothing stands against it.
    clean = buildProfileReport({"new": withMinimum(calibration("n", 2), {})}, {"new": "`x`"}, "t")
    assert "profiles far below zero" not in clean


def testReportLeadsWithWhatToDo():
    report = buildProfileReport(
        {"asis": ASIS, "new": NEW},
        {"asis": "the inputs as they are: `PFS/defaults`", "new": "`u/me/profilesGen`"},
        "fiberProfiles before certification",
        previous="asis",
        note="Resolved for visit 156021.",
        figures={"new": "profiles-new.png"},
    )
    assert report.startswith("# fiberProfiles before certification\n\nResolved for visit 156021.")
    assert "| asis (in force) | the inputs as they are: `PFS/defaults` |" in report
    assert "| new (candidate) | `u/me/profilesGen` |" in report
    sections = [line for line in report.splitlines() if line.startswith("## ")]
    assert sections == [
        "## new: what to do",
        "## new: fibers over the limit",
        "## new: detectors",
    ]  # none for asis
    assert "![Wing light above the neighbours, along the slits (new)](profiles-new.png)" in report
    assert "### Spectrograph 4" in report
    assert "| fiber | b | r | copy of | new |" in report
    assert "| 9 |  | 0.190 |  | new |" in report
    assert "| 8 |  | 0.060 | 6 7 9 | new |" in report
    assert report.index("what to do") < report.index("fibers over the limit") < report.index(": detectors")
    # Without a calibration in force there is no change column and no "new".
    alone = buildProfileReport({"new": NEW}, {"new": "`x`"}, "t")
    assert (
        "| fiber | b | r | copy of |\n" in alone
        and "change" not in alone.split("## new: detectors")[1].split("\n")[2]
    )


def testADetectorTheCalibrationInForceLacksIsNotCompared():
    """The collection in force has no b1: what the candidate shows there counts as new, and the report says so."""
    before = joined(
        calibration("r", 4, visit0=149068, run="u/me/old/1"),
        calibration("b", 1, {5: 0.03}, run="PFS/other/2"),
    )
    after = joined(calibration("r", 4), calibration("b", 1, {5: 0.03}))
    collections = {"old": "u/me/old", "new": "u/me/profilesGen"}
    report = buildProfileReport(
        {"old": before, "new": after},
        {"old": "`u/me/old`", "new": "`u/me/profilesGen`"},
        "t",
        previous="old",
        collections=collections,
    )
    assert "**No comparison for b1.**" in report
    assert "| 5 | 0.030 |  |  | new |" in report
    # Negative control: read from its own collection, the same fiber is already there and is not new.
    same = joined(
        calibration("r", 4, visit0=149068, run="u/me/old/1"), calibration("b", 1, {5: 0.03}, run="u/me/old/1")
    )
    report = buildProfileReport(
        {"old": same, "new": after},
        {"old": "`u/me/old`", "new": "`u/me/profilesGen`"},
        "t",
        previous="old",
        collections=collections,
    )
    assert "No comparison" not in report and "| 5 | 0.030 |  |  |  |" in report


def testADetectorTheCandidateDoesNotHoldIsNamed():
    """The candidate's collection has no b1: the inputs after it are measured, under the candidate's name."""
    mixed = joined(calibration("r", 4), calibration("b", 1, run="PFS/calib/inForce/2"))
    sources, elsewhere = calibrationSources(mixed[0], "u/me/profilesGen")
    assert sources == {"PFS/calib/inForce/2": ["b1"], "u/me/profilesGen/1": ["r4"]}
    assert elsewhere == ["b1"]
    # The candidate's RUN is inside its collection, or is it; the inputs as they are name nothing.
    assert calibrationSources(mixed[0], "u/me/profilesGen/1")[1] == ["b1"]
    assert calibrationSources(mixed[0], "")[1] == []
    # Negative control: a collection that only shares a prefix with the RUN is not its parent.
    assert calibrationSources(mixed[0], "u/me/profiles")[1] == ["b1", "r4"]

    collections = {"new": "u/me/profilesGen"}
    report = buildProfileReport({"new": mixed}, {"new": "`u/me/profilesGen`"}, "t", collections=collections)
    assert "- `u/me/profilesGen/1`: r4" in report and "- `PFS/calib/inForce/2`: b1" in report
    assert "**Measure again: b1 are not the candidate's.**" in report
    clean = buildProfileReport({"new": NEW}, {"new": "`u/me/profilesGen`"}, "t", collections=collections)
    assert "Measure again" not in clean


def arguments(**overrides):
    values = {
        "visit": [156021],
        "detectors": None,
        "jobs": 8,
        "butler": "/work/datastore",
        "inputs": ["PFS/defaults"],
        "prefix": "u/me/profiles/run31",
    }
    return SimpleNamespace(**(values | overrides))


def testCandidateCollectionGoesFirstInTheInputs():
    command = calibProfilesCli.candidateCommand(arguments(), "new", "u/me/profilesGen")
    assert command[command.index("-i") + 1] == "u/me/profilesGen,PFS/defaults"
    assert command[command.index("-o") + 1] == "u/me/profiles/run31/new"
    assert command[command.index("-p") + 1].endswith("pipelines/fiberProfilesQa.yaml")
    assert command[command.index("-c") + 1] == "fiberProfilesQa:anyFrameType=True"
    assert command[command.index("-d") + 1] == "instrument = 'PFS' AND visit IN (156021)"
    # The inputs as they are: nothing is put before them.
    asis = calibProfilesCli.candidateCommand(arguments(), "asis", "")
    assert asis[asis.index("-i") + 1] == "PFS/defaults"
    # Negative control: the candidate after the inputs would resolve the calibration in force instead.
    assert not command[command.index("-i") + 1].startswith("PFS/defaults")


def testSeveralVisitsCoverTheArmsOneVisitDoesNot():
    """A visit exposes the red arm in low or in medium resolution: two visits measure all four arms."""
    command = calibProfilesCli.candidateCommand(arguments(visit=[156021, 155123]), "new", "u/me/gen")
    assert command[command.index("-d") + 1] == "instrument = 'PFS' AND visit IN (156021, 155123)"


def testACandidateNeedsItsEqualsSign():
    """Without it the item would be a label with no collection: the calibration in force, under the build's name."""
    for item in ("u/me/profilesGen", "=u/me/profilesGen"):
        with pytest.raises(SystemExit, match="LABEL=COLLECTION"):
            calibProfilesCli.candidates(SimpleNamespace(candidate=["asis=", item]))
    with pytest.raises(SystemExit, match="given twice"):
        calibProfilesCli.candidates(SimpleNamespace(candidate=["new=a", "new=b"]))
    assert calibProfilesCli.candidates(SimpleNamespace(candidate=["asis=", "new=u/me/gen"])) == {
        "asis": "",
        "new": "u/me/gen",
    }


def testADetectorMeasuredOnTwoVisitsIsReportedOnce():
    twice = tuple(pd.concat([table, table], ignore_index=True) for table in NEW)
    once = buildProfileReport({"new": NEW}, {"new": "`x`"}, "t")
    assert buildProfileReport({"new": twice}, {"new": "`x`"}, "t") == once


def testDetectorsNarrowTheDataQuery():
    command = calibProfilesCli.candidateCommand(arguments(detectors=["r4", "b1"]), "new", "u/me/gen")
    assert command[command.index("-d") + 1] == (
        "instrument = 'PFS' AND visit IN (156021) AND ((arm = 'r' AND spectrograph = 4) OR (arm = 'b' AND spectrograph = 1))"
    )


def testDryRunPrintsTheCommandsAndRunsNothing(capsys, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a dry run must not run pipetask")

    monkeypatch.setattr(calibProfilesCli.subprocess, "run", refuse)
    argv = [
        "run",
        "--visit",
        "156021",
        "--prefix",
        "u/me/p",
        "--candidate",
        "asis=",
        "--candidate",
        "new=u/me/gen",
    ]
    assert calibProfilesCli.main([*argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "# asis" in out and "# new" in out and "u/me/gen,PFS/defaults" in out


def testPreviousMustBeACandidate():
    argv = [
        "report",
        "--visit",
        "1",
        "--prefix",
        "p",
        "--candidate",
        "new=x",
        "--previous",
        "asis",
        "-o",
        "r.md",
    ]
    with pytest.raises(SystemExit, match="not one of the candidates"):
        calibProfilesCli.main(argv)


def testReadProfileQaConcatenatesTheDetectorsOfAVisit():
    stored = {
        ("fiberProfilesQaFibers", "r", 4): NEW[0][NEW[0]["arm"] == "r"],
        ("fiberProfilesQaFibers", "b", 1): NEW[0][NEW[0]["arm"] == "b"],
        ("fiberProfilesQaMetrics", "r", 4): NEW[1][NEW[1]["arm"] == "r"],
        ("fiberProfilesQaMetrics", "b", 1): NEW[1][NEW[1]["arm"] == "b"],
    }
    asked = []

    def queryDatasets(name, collections, findFirst, where=None):
        asked.append((name, tuple(collections), findFirst, where))
        return [
            SimpleNamespace(dataId={"visit": 156021, "arm": arm, "spectrograph": spectrograph}, name=name)
            for (kind, arm, spectrograph) in stored
            if kind == name
        ]

    def get(ref):
        return stored[(ref.name, ref.dataId["arm"], ref.dataId["spectrograph"])]

    butler = SimpleNamespace(registry=SimpleNamespace(queryDatasets=queryDatasets), get=get)
    fibers, metrics = readProfileQa(butler, "u/me/profiles/run31/new", 156021)
    assert len(fibers) == 40 and list(fibers["arm"].unique()) == ["b", "r"]
    assert len(metrics) == 2 * len(METRICS)
    assert asked[0] == (
        "fiberProfilesQaFibers",
        ("u/me/profiles/run31/new",),
        True,
        "instrument = 'PFS' AND visit IN (156021)",
    )
    # Nothing stored: empty frames, for the command to say so.
    empty = SimpleNamespace(registry=SimpleNamespace(queryDatasets=lambda *a, **k: []), get=get)
    assert all(table.empty for table in readProfileQa(empty, "u/me/none"))
