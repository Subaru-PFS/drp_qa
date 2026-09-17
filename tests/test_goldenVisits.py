"""Tests for the golden visit set and its loader.

Stack-free: imports only the standard library, pyyaml and
``pfs.drp.qa.metrics.goldenVisits``.
"""

from textwrap import dedent

import pytest

from pfs.drp.qa.metrics.goldenVisits import (
    GoldenVisit,
    defaultGoldenVisitsPath,
    loadGoldenVisits,
)
from pfs.drp.qa.metrics.thresholds import MIN_SAMPLES


@pytest.fixture
def writeYaml(tmp_path):
    """Return a helper writing YAML text to a temporary file."""

    def write(text):
        path = tmp_path / "goldenVisits.yaml"
        path.write_text(dedent(text))
        return path

    return write


class TestCheckedInSet:
    """The golden set that ships with the repository must stay loadable."""

    def testDefaultPathResolves(self):
        path = defaultGoldenVisitsPath()
        assert path.name == "goldenVisits.yaml"
        assert path.exists(), f"golden visit set missing at {path}"

    def testLoads(self):
        golden = loadGoldenVisits()
        assert golden.path == defaultGoldenVisitsPath()

    def testSm1FocusRangeIsKnownBad(self):
        """The documented SM1 optics failure is the anchor known_bad entry."""
        golden = loadGoldenVisits()
        sm1 = [entry for entry in golden.knownBad if 140005 in entry.visits]
        assert len(sm1) == 1, "expected exactly one entry covering the SM1 focus range"
        (entry,) = sm1
        assert entry.expect == "FAIL"
        assert entry.spectrographs == (1,)
        assert entry.visits[-1] == 140138
        assert len(entry.visits) == 140138 - 140005 + 1
        assert entry.metric, "a known_bad entry must name the metric that identifies the fault"
        assert entry.reason

    def testKnownGoodIsPopulated(self):
        """The Run25 stable set; thresholds cannot be derived without it."""
        golden = loadGoldenVisits()
        assert golden.knownGood, "no usable known_good entries"
        assert golden.goodVisits[0] == 133025
        assert golden.goodVisits[-1] == 135850

    def testKnownGoodEntriesAreScopedToTheArmsThatWereRead(self):
        """Run25 block A read b/r/n and block B read b/m; neither covers the other."""
        golden = loadGoldenVisits()
        assert golden.expectationFor(133037, arm="b", spectrograph=1) == "PASS"
        assert golden.expectationFor(133037, arm="m", spectrograph=1) is None, "m was not read in block A"
        assert golden.expectationFor(133042, arm="m", spectrograph=1) == "PASS"
        assert golden.expectationFor(133042, arm="n", spectrograph=1) is None, "n was not read in block B"

    def testHgCdClearsTheThresholdSampleFloor(self):
        """b:HgCd needs both HgCd blocks to reach the 20-sample floor."""
        detectors = sum(
            len(entry.visits) * len(entry.spectrographs)
            for entry in loadGoldenVisits().knownGood
            if entry.seqType == "Arc: HgCd" and "b" in entry.arms
        )
        assert detectors >= MIN_SAMPLES, f"only {detectors} b-arm HgCd detectors"

    def testCloudyTwilightIsKnownBadButTheClearOnesAreNot(self):
        """Same seqType, opposite verdicts; matching is by visit, not sequence name."""
        golden = loadGoldenVisits()
        assert golden.expectationFor(134334, arm="b", spectrograph=1) == "FAIL"
        assert golden.expectationFor(134880, arm="b", spectrograph=1) == "PASS"

    def testEveryKnownGoodEntryNamesItsSequence(self):
        """The flag-rate threshold key is derived from seqType; it cannot be blank."""
        for entry in loadGoldenVisits().knownGood:
            assert entry.seqType, f"{entry.visits[0]} does not name its W_SEQNAM"

    def testPlaceholdersExcludedByDefault(self):
        """An unfilled entry must never silently validate a threshold."""
        assert not any(entry.placeholder for entry in loadGoldenVisits())
        assert any(entry.placeholder for entry in loadGoldenVisits(includePlaceholders=True))

    def testPlaceholdersCarryNoVisits(self):
        golden = loadGoldenVisits(includePlaceholders=True)
        for entry in golden:
            if entry.placeholder:
                assert entry.visits == ()


class TestReferenceSection:
    """Entries that are tracked but assert nothing."""

    def testReferenceEntriesCarryNoExpectation(self):
        """The Run30 calib block is what is under test, not a premise."""
        golden = loadGoldenVisits()
        assert golden.reference
        assert all(entry.expect is None for entry in golden.reference)
        assert golden.expectationFor(148312, arm="b", spectrograph=1) is None

    def testReferenceEntriesAreStillTracked(self):
        """No expectation is not the same as not being in the set."""
        golden = loadGoldenVisits()
        assert len(golden.find(148312, arm="b", spectrograph=1)) == 1

    def testBothEpochsHaveACalibBlockToCompare(self):
        """The cross-run comparison needs a block on each side of it."""
        golden = loadGoldenVisits()
        assert golden.epochs == ("Run25", "Run30")
        assert golden.withRole("calibBlock", "Run25")
        assert golden.withRole("calibBlock", "Run30")

    def testDriftSeriesIsTrackedButNotKnownGood(self):
        """It is measured across, not gated on; it must constrain no threshold."""
        golden = loadGoldenVisits()
        drift = golden.withRole("driftSeries")
        assert drift
        assert all(entry.expect is None for entry in drift)
        assert not any(entry.role == "driftSeries" for entry in golden.knownGood)

    def testAReferenceEntryMayNotAssertAVerdict(self, writeYaml):
        path = writeYaml("version: 1\nreference:\n  - {visit: 1, expect: PASS}\n")
        with pytest.raises(ValueError, match="must not state 'expect'"):
            loadGoldenVisits(path)


class TestRun30KnownBad:
    def testObstructedFramesFail(self):
        golden = loadGoldenVisits()
        assert golden.expectationFor(150115, arm="b", spectrograph=1) == "FAIL"
        assert golden.expectationFor(150641, arm="b", spectrograph=1) == "FAIL"

    def testPartialIlluminationWarnsOnTheLineCount(self):
        """The defect is how many fibers were measured, not their shape."""
        golden = loadGoldenVisits()
        (entry,) = [e for e in golden.knownBad if 149398 in e.visits]
        assert entry.expect == "WARN"
        assert entry.metric == "nLines"

    def testFlexureCaseIsAttributedToTheCalibComparison(self):
        """Phase 2's reference case: a real offset against detectorMap_calib."""
        golden = loadGoldenVisits()
        flexure = golden.withRole("flexure")
        assert len(flexure) == 2
        assert all(entry.metric == "medDxCenter" for entry in flexure)


class TestEntryMatching:
    def testMatchesRespectsSelectors(self):
        entry = GoldenVisit(visits=(100, 101), expect="FAIL", arms=("b",), spectrographs=(1,))
        assert entry.matches(100, arm="b", spectrograph=1)
        assert not entry.matches(102, arm="b", spectrograph=1)
        assert not entry.matches(100, arm="r", spectrograph=1)
        assert not entry.matches(100, arm="b", spectrograph=2)

    def testOmittedSelectorMatchesEverything(self):
        entry = GoldenVisit(visits=(100,), expect="PASS")
        assert entry.matches(100, arm="n", spectrograph=4, seqType="Quartz")

    def testUnknownSelectorIsNotApplied(self):
        """Passing ``None`` means "do not filter on this", not "no match"."""
        entry = GoldenVisit(visits=(100,), expect="PASS", arms=("b",))
        assert entry.matches(100, arm=None)


class TestExpectation:
    def testUnknownVisitHasNoExpectation(self):
        """Absence from the set is "no expectation", never an implied PASS."""
        golden = loadGoldenVisits()
        assert golden.expectationFor(1) is None

    def testWorstExpectationWins(self, writeYaml):
        path = writeYaml(
            """
            version: 1
            known_good:
              - visit: 100
            known_bad:
              - visit: 100
                expect: WARN
              - visit: 100
                expect: FAIL
            """
        )
        assert loadGoldenVisits(path).expectationFor(100) == "FAIL"


class TestParsing:
    def testVisitRangeIsInclusive(self, writeYaml):
        path = writeYaml("version: 1\nknown_good:\n  - visitRange: [10, 12]\n")
        (entry,) = loadGoldenVisits(path).knownGood
        assert entry.visits == (10, 11, 12)
        assert entry.expect == "PASS", "known_good defaults to PASS"

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ("version: 2\n", "unsupported schema version"),
            ("version: 1\nknown_good: 3\n", "must be a list"),
            ("version: 1\nknown_good:\n  - 3\n", "expected a mapping"),
            ("version: 1\nknown_good:\n  - {}\n", "one of 'visit' or 'visitRange' is required"),
            ("version: 1\nknown_good:\n  - {visit: 1, visitRange: [1, 2]}\n", "not both"),
            ("version: 1\nknown_good:\n  - {visitRange: [9, 1]}\n", "inverted"),
            ("version: 1\nknown_good:\n  - {visitRange: [1]}\n", "two-element list"),
            ("version: 1\nknown_good:\n  - {visit: notanint}\n", "must be an integer"),
            ("version: 1\nknown_good:\n  - {visit: true}\n", "must be an integer"),
            ("version: 1\nknown_good:\n  - {visit: 1, expect: MAYBE}\n", "invalid expect"),
            ("version: 1\nknown_good:\n  - {visit: 1, arms: b}\n", "must be a list"),
            ("version: 1\nknown_bad:\n  - {visit: 1}\n", "'expect' is required"),
        ],
    )
    def testMalformedEntriesRaise(self, writeYaml, body, message):
        """Validation is strict: a dropped entry stops being a reference."""
        with pytest.raises(ValueError, match=message):
            loadGoldenVisits(writeYaml(body))

    def testMissingFileRaises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            loadGoldenVisits(tmp_path / "nope.yaml")
