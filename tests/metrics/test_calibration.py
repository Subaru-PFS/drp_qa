"""Tests for the threshold table built from metrics and the validation visit set."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.calibration import addSpecies, calibrate, compareRuns, labelRows
from pfs.drp.qa.metrics.validationVisits import ValidationVisit, ValidationVisitSet

ARC_VISITS = tuple(range(1000, 1030))
TRACE_VISITS = tuple(range(2000, 2030))


def makeMetrics() -> pd.DataFrame:
    """Return synthetic ``iqQaMetrics``: arcs at 2.6 px, traces at 3.1 px, two arms.

    Plus known-bad and unconfirmed visits, and a duplicated index as a plain
    ``pd.concat`` leaves it.
    """
    rng = np.random.default_rng(3)
    rows = []

    def add(visit, obsType, seqName, fwhm, arm="b", spectrograph=1, **extra):
        rows.append(
            {
                "visit": visit,
                "arm": arm,
                "spectrograph": spectrograph,
                "obsType": obsType,
                "seqName": seqName,
                "medFwhm": fwhm,
                "medDxCenter": rng.normal(0.0, 0.05),
                "pctFlagged": extra.pop("pctFlagged", abs(rng.normal(5.0, 1.0))),
                "nLines": extra.pop("nLines", 500),
            }
        )

    for visit in ARC_VISITS:
        for arm in ("b", "r"):
            add(visit, "arc", "Arc: Neon", rng.normal(2.6, 0.03), arm=arm)
    for visit in TRACE_VISITS:
        for arm in ("b", "r"):
            add(visit, "trace", "Trace", rng.normal(3.1, 0.03), arm=arm)
    add(3000, "arc", "Arc: Neon", 3.0)  # bad arc FWHM, inside the trace population
    add(3001, "arc", "Arc: Neon", 2.62, nLines=40)  # partial illumination
    add(3002, "arc", "Arc: Neon", 2.9)  # suspected
    add(3003, "arc", "Arc: Neon", 2.6, arm="m")  # bad, in a population with no good data
    frame = pd.DataFrame(rows)
    frame.index = np.arange(len(frame)) // 2
    return frame


def makeVisitSet() -> ValidationVisitSet:
    return ValidationVisitSet(
        knownGood=(ValidationVisit(visits=ARC_VISITS + TRACE_VISITS, expect="PASS"),),
        knownBad=(
            ValidationVisit(visits=(3000,), expect="FAIL", metric="medFwhm"),
            ValidationVisit(visits=(3001,), expect="WARN", metric="nLines"),
            ValidationVisit(visits=(3002,), expect="FAIL", unconfirmed=True),
            ValidationVisit(visits=(3003,), expect="FAIL", metric="medFwhm"),
        ),
    )


def row(table: pd.DataFrame, metric: str, group: str) -> pd.Series:
    (index,) = np.flatnonzero((table["metric"] == metric) & (table["group"] == group))
    return table.iloc[index]


@pytest.fixture
def table() -> pd.DataFrame:
    return calibrate(makeMetrics(), makeVisitSet(), ["medFwhm", "nLines"], derivedOn=date(2026, 10, 4))


class TestLabelRows:
    def testLabels(self):
        labelled = labelRows(makeMetrics(), makeVisitSet(), "medFwhm")
        byVisit = labelled.groupby("visit")["validation"].first()
        assert byVisit[1000] == "good"
        assert byVisit[3000] == "bad:FAIL"
        assert byVisit[3002] == "unconfirmed"

    def testBadEntryForAnotherMetricIsNeitherBadNorGood(self):
        """3001 is bad for nLines: it says nothing about medFwhm, and is not good either."""
        assert 3001 not in labelRows(makeMetrics(), makeVisitSet(), "medFwhm")["visit"].tolist()
        nLines = labelRows(makeMetrics(), makeVisitSet(), "nLines")
        assert nLines.loc[nLines["visit"] == 3001, "validation"].tolist() == ["bad:WARN"]

    def testBadWinsOverGood(self):
        visitSet = ValidationVisitSet(
            knownGood=(ValidationVisit(visits=(1000,), expect="PASS"),),
            knownBad=(ValidationVisit(visits=(1000,), expect="FAIL", arms=("b",)),),
        )
        labelled = labelRows(makeMetrics(), visitSet, "medFwhm")
        assert labelled.set_index("arm")["validation"].to_dict() == {"b": "bad:FAIL", "r": "good"}


class TestCalibrate:
    def testPopulationsAreSeparated(self, table):
        fwhm = table[table["metric"] == "medFwhm"]
        assert set(fwhm["group"]) == {"b/arc", "r/arc", "b/trace", "r/trace", "m/arc"}
        assert row(table, "medFwhm", "b/arc")["fail"] < 2.8 < row(table, "medFwhm", "b/trace")["warn"]

    def testSeparationCatchesWhatPoolingMisses(self, table):
        """The bad arc at 3.0 px fails against arcs, and hides among traces when pooled."""
        separated = row(table, "medFwhm", "b/arc")
        assert separated["badOk"]
        pooled = calibrate(makeMetrics(), makeVisitSet(), ["medFwhm"], groupBy=[])
        assert pooled["group"].tolist() == ["all"]
        assert not pooled.loc[0, "badOk"]

    def testTableColumns(self, table):
        b = row(table, "medFwhm", "b/arc")
        assert b["arm"] == "b" and b["obsType"] == "arc"
        assert b["nGood"] == len(ARC_VISITS)
        assert b["nGoodVisits"] == len(ARC_VISITS)
        assert b["visitRange"] == f"{ARC_VISITS[0]}-{ARC_VISITS[-1]}"
        assert b["reliable"] and not b["failBounded"]
        assert "Derived 2026-10-04 from validation visits 1000-1029 (n=30)" in b["provenance"]

    def testWarnEntryIsCheckedAtWarn(self, table):
        nLines = row(table, "nLines", "b/Arc: Neon")
        assert nLines["badOk"]
        assert "expecting WARN" in nLines["badSummary"]

    def testUnconfirmedIsReportedNotJudged(self, table):
        b = row(table, "medFwhm", "b/arc")
        assert b["nUnconfirmed"] == 1
        assert b["unconfirmedFlagged"] == 1
        assert b["nBad"] == 1, "the unconfirmed row is not part of the known-bad check"

    def testNoBadRowsLeavesTheCheckOpen(self, table):
        assert pd.isna(row(table, "medFwhm", "r/arc")["badOk"])

    def testBadRowsWithoutGoodDataAreShown(self, table):
        m = row(table, "medFwhm", "m/arc")
        assert m["nGood"] == 0 and m["nBad"] == 1
        assert "no known-good rows" in m["badSummary"]
        assert pd.isna(m["fail"])

    def testFlagRatesAreSplitBySpecies(self):
        table = calibrate(makeMetrics(), makeVisitSet(), ["pctFlagged"])
        assert "arc/b/Neon" in set(table["group"])
        assert "trace/b/" in set(table["group"])

    def testAbsoluteMetricIsCalibratedOnMagnitude(self):
        metrics = makeMetrics().reset_index(drop=True)
        arcs = metrics["obsType"] == "arc"
        metrics.loc[arcs, "medDxCenter"] = -np.linspace(0.0, 1.0, arcs.sum())
        b = row(calibrate(metrics, makeVisitSet(), ["medDxCenter"]), "medDxCenter", "b/arc")
        assert b["warn"] > 0.8

    def testUnknownMetricRaises(self):
        with pytest.raises(KeyError, match="nope"):
            calibrate(makeMetrics(), makeVisitSet(), ["nope"])


def testAddSpeciesFollowsTheTask():
    """Same rule as imageQualityQa's flag-rate keys."""
    frame = addSpecies(pd.DataFrame({"seqName": ["Arc: HgCd", "Trace", None]}))
    assert frame["species"].tolist() == ["HgCd", "", ""]


def makeRunVisitSet() -> ValidationVisitSet:
    """Arcs of visits 1000-1014 are the reference run; 1015-1029 are another run."""
    runs = {25: (1000, 1014), 27: (1015, 3999)}
    good = (
        ValidationVisit(visits=tuple(range(1000, 1015)), expect="PASS", sequenceType="scienceArc", run=25),
        ValidationVisit(visits=tuple(range(1015, 1030)), expect="PASS", sequenceType="scienceArc", run=27),
    )
    return ValidationVisitSet(knownGood=good, runs=runs, referenceRuns=(25,))


class TestRuns:
    def testOtherRunsAreHeldOut(self):
        labelled = labelRows(makeMetrics(), makeRunVisitSet(), "medFwhm")
        assert set(labelled.loc[labelled["visit"] < 1015, "validation"]) == {"good"}
        assert set(labelled.loc[labelled["visit"] >= 1015, "validation"]) == {"heldOut"}
        assert set(labelled["run"]) == {25, 27}

    def testThresholdsComeFromTheReferenceOnly(self):
        metrics = makeMetrics().reset_index(drop=True)
        metrics.loc[metrics["visit"].between(1015, 1029), "medFwhm"] += 1.0  # the other run is worse
        b = row(calibrate(metrics, makeRunVisitSet(), ["medFwhm"]), "medFwhm", "b/arc")
        assert b["nGood"] == 15 and b["nHeldOut"] == 15
        assert b["fail"] < 2.8, "the held-out run did not move the thresholds"
        assert b["heldOutFlaggedFail"] == 1.0

    def testCompareRunsPerRun(self):
        metrics = makeMetrics().reset_index(drop=True)
        metrics.loc[metrics["visit"].between(1015, 1029), "medFwhm"] += 1.0
        visitSet = makeRunVisitSet()
        comparison = compareRuns(metrics, visitSet, calibrate(metrics, visitSet, ["medFwhm"]), ["medFwhm"])
        b = comparison[comparison["group"] == "b/arc"].set_index("run")
        assert b.loc[25, "reference"] and not b.loc[27, "reference"]
        assert b.loc[25, "flaggedFail"] <= 0.1 and b.loc[27, "flaggedFail"] == 1.0


def testBadEntryExcludesOnlyItsMetric():
    """A known_bad entry naming pctFlagged leaves its rows good for medFwhm."""
    visitSet = ValidationVisitSet(
        knownGood=(ValidationVisit(visits=ARC_VISITS, expect="PASS"),),
        knownBad=(ValidationVisit(visits=(1000,), expect="WARN", metric="pctFlagged"),),
    )
    fwhm = labelRows(makeMetrics(), visitSet, "medFwhm")
    flags = labelRows(makeMetrics(), visitSet, "pctFlagged")
    assert set(fwhm.loc[fwhm["visit"] == 1000, "validation"]) == {"good"}
    assert set(flags.loc[flags["visit"] == 1000, "validation"]) == {"bad:WARN"}
