"""Tests for `pfs.drp.qa.comparison.plan`."""

import pandas as pd
import pytest

from pfs.drp.qa.comparison.classify import classifyVisits
from pfs.drp.qa.comparison.plan import (
    Pass,
    coverage,
    drpActorConfig,
    expectedDetectors,
    outputCollection,
    passes,
    pipetaskCommand,
    summarize,
)


@pytest.mark.parametrize(
    "sequenceType, adjust, quickCds, normalize",
    [
        ("scienceObject", True, False, True),
        ("scienceObject_windowed", False, True, True),
        ("scienceArc", False, True, True),
        ("scienceTrace", False, True, True),
        ("masterDarks", False, False, True),
        ("darks", False, True, False),
    ],
)
def testDrpActorConfig(sequenceType, adjust, quickCds, normalize):
    assert drpActorConfig(sequenceType) == {
        "reduceExposure:requireAdjustDetectorMap": adjust,
        "isr:h4.quickCDS": quickCds,
        "cosmicray:doNormalizeChiRms": normalize,
    }


def testExpectedDetectors():
    visits = pd.DataFrame({"pfs_visit_id": [1, 2], "cameras": ["b1,n1,r1", "m3,bogus"]})
    assert expectedDetectors(visits).values.tolist() == [[1, "b", 1], [1, "n", 1], [1, "r", 1], [2, "m", 3]]


@pytest.fixture
def classified(periods, visit, listing):
    """Classify an arc sequence of two visits, a trace, a sky frame and a dark."""
    return classifyVisits(
        listing(
            visit(1, "2026-01-03 18:00", "scienceArc", sequence=50, cameras="b1,r1"),
            visit(2, "2026-01-03 18:01", "scienceArc", sequence=50, cameras="b1,r1"),
            visit(3, "2026-01-03 18:05", "scienceTrace", sequence=51, cameras="b1,r1"),
            visit(4, "2026-01-03 20:00", "scienceObject", sequence=52, cameras="b1,r1"),
            visit(5, "2026-01-04 15:00", "darks", sequence=53, cameras="b1,r1"),
        ),
        periods.values(),
    )


def _holdings(rows):
    return pd.DataFrame(rows, columns=["visit", "arm", "spectrograph", "raw", "reduced", "judged"])


def testCoverage(classified):
    holdings = _holdings(
        [
            (1, "b", 1, True, True, True),  # judged
            (1, "r", 1, True, True, False),  # reduced by drpActor, not judged yet
            (2, "b", 1, True, False, False),  # raw only
            # 2 r1: listed by opdb, no raw
            (3, "b", 1, True, True, True),
            (3, "r", 1, True, True, True),
            (3, "b", 2, True, False, False),  # raw that opdb didn't list
            (4, "b", 1, True, False, False),
            (4, "r", 1, True, False, False),
            (5, "b", 1, True, False, False),  # a dark: not judged, so not covered
        ]
    )
    detectors = coverage(classified, holdings).set_index(["visit", "arm", "spectrograph"])
    assert detectors["status"].to_dict() == {
        (1, "b", 1): "judged",
        (1, "r", 1): "to judge",
        (2, "b", 1): "to reduce",
        (2, "r", 1): "no raw",
        (3, "b", 1): "judged",
        (3, "b", 2): "raw not in opdb",
        (3, "r", 1): "judged",
        (4, "b", 1): "to reduce",
        (4, "r", 1): "to reduce",
    }
    assert detectors.loc[(4, "b", 1), "sequence_type"] == "scienceObject"

    summary = summarize(classified, detectors.reset_index()).set_index("sequence_type")
    arcs = summary.loc["scienceArc"]
    assert (arcs["visits"], arcs["judged"], arcs["to judge"], arcs["to reduce"], arcs["no raw"]) == (
        2,
        1,
        1,
        1,
        1,
    )
    assert summary.loc["darks", "reason"] == "no method for darks"
    assert summary.loc["darks", "to reduce"] == 0  # not expanded into detectors


def testPassesGroupByConfigAndSequence(classified):
    holdings = _holdings([(visit, arm, 1, True, False, False) for visit in (1, 2, 3, 4) for arm in "br"])
    detectors = coverage(classified, holdings)
    result = passes(classified, detectors)

    assert [item.name for item in result] == ["calibration", "sky"]
    assert result[0].types == ("scienceArc", "scienceTrace")
    arcsAndTraces, sky = result
    assert arcsAndTraces.visits == (1, 2, 3)
    assert arcsAndTraces.groups == {1: 1, 2: 1, 3: 3}  # cosmicray combines a sequence, never two
    assert arcsAndTraces.config == drpActorConfig("scienceArc")
    assert sky.config["reduceExposure:requireAdjustDetectorMap"] is True
    assert "config.groups = {1: 1, 2: 1, 3: 3}" in arcsAndTraces.cosmicrayConfig()


def testNothingPendingNoPasses(classified):
    holdings = _holdings([(visit, arm, 1, True, True, True) for visit in (1, 2, 3, 4) for arm in "br"])
    assert passes(classified, coverage(classified, holdings)) == []


def testPipetaskCommand():
    item = Pass(
        name="calibration",
        types=("scienceArc",),
        visits=(10, 11, 12, 20),
        config=drpActorConfig("scienceArc"),
        groups={},
    )
    command = pipetaskCommand(
        item,
        butler="/work/datastore",
        pipeline="pipelines/qaThresholds.yaml",
        inputs=["drpActor/reductions", "PFS/defaults"],
        output="u/me/comparison/run30/w.2026.41",
        skipExistingIn=["drpActor/reductions"],
        cosmicrayConfigFile="/tmp/cr.py",
        jobs=4,
    )
    assert command[:7] == ["pipetask", "--long-log", "--log-level", "PFS=INFO", "run", "-j", "4"]
    joined = " ".join(command)
    assert "-i drpActor/reductions,PFS/defaults" in joined
    assert "--skip-existing-in drpActor/reductions" in joined
    assert "-c isr:h4.quickCDS=True" in joined
    assert "-C cosmicray:/tmp/cr.py" in joined
    assert command[-2:] == ["-d", "instrument = 'PFS' AND visit IN (10..12, 20)"]


def testOutputCollection():
    assert (
        outputCollection("u/me/comparison/", "run30-pre", "w.2026.41")
        == "u/me/comparison/run30-pre/w.2026.41"
    )
    with pytest.raises(ValueError, match="uncommitted"):
        outputCollection("u/me/comparison", "run30", "w.2026.41-2-gabc1234-dirty")


def testMediumResolutionIsTheRedCamera(periods, visit, listing):
    visits = classifyVisits(
        listing(
            visit(1, "2026-01-03 20:00", "scienceObject", cameras="b1,r1"),  # opdb says r1...
            visit(2, "2026-01-03 20:10", "scienceObject", cameras="b1,m1"),  # ...or m1
        ),
        periods.values(),
    )
    holdings = _holdings(
        [
            (1, "b", 1, True, False, False),
            (1, "m", 1, True, False, False),  # ...where the Butler has m1
            (2, "b", 1, True, False, False),
            (2, "r", 1, True, False, False),  # ...or r1
        ]
    )
    detectors = coverage(visits, holdings).set_index(["visit", "arm", "spectrograph"])["status"]
    assert detectors.to_dict() == {
        (1, "b", 1): "to reduce",
        (1, "m", 1): "to reduce",
        (2, "b", 1): "to reduce",
        (2, "r", 1): "to reduce",
    }
    # Negative control: a different spectrograph is still a mismatch.
    holdings.loc[holdings["arm"] == "m", "spectrograph"] = 2
    statuses = set(coverage(visits, holdings)["status"])
    assert {"no raw", "raw not in opdb"} <= statuses


def testNoPfsConfigIsNotScheduled(classified):
    holdings = _holdings([(visit, arm, 1, True, visit == 3, False) for visit in (1, 2, 3, 4) for arm in "br"])
    holdings["pfsConfig"] = holdings["visit"] != 3  # the trace has none
    detectors = coverage(classified, holdings)
    assert set(detectors.loc[detectors["visit"] == 3, "status"]) == {"no pfsConfig"}
    assert set(detectors.loc[detectors["visit"] != 3, "status"]) == {"to reduce"}
    assert [item.visits for item in passes(classified, detectors)] == [(1, 2), (4,)]


def testUnvalidatedPassComesLast(periods, visit, listing):
    visits = classifyVisits(
        listing(
            visit(1, "2026-01-03 18:00", "scienceArc", sequence=50, cameras="b1"),
            visit(2, "2026-01-03 18:10", "slitThroughFocus", sequence=51, cameras="b1", expType="arc"),
            visit(3, "2026-01-03 18:20", "dotRoach", sequence=52, cameras="b1", expType="flat"),
            visit(4, "2026-01-03 18:30", "darks", sequence=53, cameras="b1"),
        ),
        periods.values(),
    )
    holdings = _holdings([(v, "b", 1, True, False, False) for v in (1, 2, 3, 4)])
    result = passes(visits, coverage(visits, holdings))
    assert [(item.name, item.types, item.visits) for item in result] == [
        ("calibration", ("scienceArc",), (1,)),
        ("unvalidated-calibration", ("dotRoach", "slitThroughFocus"), (2, 3)),
    ]  # the darks aren't measured
