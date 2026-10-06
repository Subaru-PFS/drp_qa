"""Tests for the validation visit set and its loader."""

from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

from pfs.drp.qa.metrics.thresholds import MIN_SAMPLES
from pfs.drp.qa.metrics.validationVisits import (
    ValidationVisit,
    ValidationVisitSet,
    defaultValidationVisitsPath,
    formatTables,
    loadValidationVisits,
    main,
    matchRows,
    selectRows,
    sequenceGroups,
    sequenceTypeMismatches,
    unmatchedEntries,
    visitExpression,
)


@pytest.fixture
def writeYaml(tmp_path):
    """Return a helper writing YAML text to a temporary file."""

    def write(text):
        path = tmp_path / "validationVisits.yaml"
        path.write_text(dedent(text))
        return path

    return write


class TestCheckedInSet:
    """The validation set that ships with the repository must stay loadable."""

    def testDefaultPathResolves(self):
        path = defaultValidationVisitsPath()
        assert path.name == "validationVisits.yaml"
        assert path.exists(), f"validation visit set missing at {path}"

    def testShipsAsPackageData(self):
        """The set is read at run time, so it lives in the package, not in tests/."""
        assert defaultValidationVisitsPath().parent.parent.name == "metrics"

    def testLoads(self):
        visitSet = loadValidationVisits()
        assert visitSet.path == defaultValidationVisitsPath()

    def testSm1DefocusIsKnownBad(self):
        """The Run27 SM1-defocused sequences: spectrograph 1 only, caught by medFwhm."""
        visitSet = loadValidationVisits()
        sm1 = [entry for entry in visitSet.knownBad if entry.reason and "SM1 defocused" in entry.reason]
        assert len(sm1) == 9
        visits = {visit for entry in sm1 for visit in entry.visits}
        assert visits >= {140005, 140032, 140035, 140138, 140139, 140148}
        assert not visits & {140033, 140034}, "no raw data"
        assert all(entry.spectrographs == (1,) and entry.metric == "medFwhm" for entry in sm1)
        assert visitSet.expectationFor(140130, arm="b", spectrograph=1, seqType="Arc: Neon") == "FAIL"
        assert visitSet.expectationFor(140130, arm="b", spectrograph=2, seqType="Arc: Neon") is None
        assert visitSet.expectationFor(140075) is None, "unlogged visits in the range carry no verdict"

    def testUnlitSpectrographsAreKnownBad(self):
        visitSet = loadValidationVisits()
        assert visitSet.expectationFor(140640, arm="b", spectrograph=2) == "FAIL"
        assert visitSet.expectationFor(140640, arm="b", spectrograph=1) is None

    def testOtherRunsGiveFaultsOnly(self):
        """Run27 and Run30 are fault examples: known-bad entries, no known-good ones."""
        visitSet = loadValidationVisits()
        faults = [
            entry
            for entry in visitSet.knownBad
            if entry.reason
            and any(text in entry.reason for text in ("SM1 defocused", "No light", "didn't turn on"))
        ]
        assert len(faults) == 14 and {entry.run for entry in faults} == {27}
        assert {entry.run for entry in visitSet.confirmedBad} == {25, 27, 30}
        assert {entry.run for entry in visitSet.knownGood} == {25}

    def testSingleVisitHeliumTestsNeedNoSeqType(self):
        """The hand-written summary misspells 150779's sequence; no selector depends on it."""
        visitSet = loadValidationVisits()
        for visit in (150779, 150782):
            (entry,) = visitSet.find(visit, arm="b", spectrograph=1)
            assert entry.seqType is None
            assert entry.metric == "medDxCenter"

    def testKnownGoodIsPopulated(self):
        """The Run25 stable set; thresholds cannot be derived without it."""
        visitSet = loadValidationVisits()
        assert visitSet.knownGood, "no usable known_good entries"
        assert visitSet.goodVisits[0] == 133025
        assert visitSet.goodVisits[-1] == 135850

    def testKnownGoodEntriesAreScopedToTheArmsThatWereRead(self):
        """Run25 block A read b/r/n and block B read b/m; neither covers the other."""
        visitSet = loadValidationVisits()
        assert visitSet.expectationFor(133037, arm="b", spectrograph=1) == "PASS"
        assert visitSet.expectationFor(133037, arm="m", spectrograph=1) is None, "m was not read in block A"
        assert visitSet.expectationFor(133042, arm="m", spectrograph=1) == "PASS"
        assert visitSet.expectationFor(133042, arm="n", spectrograph=1) is None, "n was not read in block B"

    def testHgCdClearsTheThresholdSampleFloor(self):
        """b:HgCd needs both HgCd blocks to reach the 20-sample floor."""
        detectors = sum(
            len(entry.visits) * len(entry.spectrographs)
            for entry in loadValidationVisits().knownGood
            if entry.seqType == "Arc: HgCd" and "b" in entry.arms
        )
        assert detectors >= MIN_SAMPLES, f"only {detectors} b-arm HgCd detectors"

    def testCloudyTwilightIsKnownBadButTheClearOnesAreNot(self):
        """Same seqType, opposite verdicts; matching is by visit, not sequence name."""
        visitSet = loadValidationVisits()
        assert visitSet.expectationFor(134334, arm="b", spectrograph=1) == "FAIL"
        assert visitSet.expectationFor(134880, arm="b", spectrograph=1) == "PASS"

    def testEveryEntryCarriesAVerdict(self):
        """The file's entire content is verdicts; a visit with none does not belong."""
        for entry in loadValidationVisits(includePlaceholders=True):
            assert entry.expect in ("PASS", "WARN", "FAIL")

    def testTraceVisitsClearTheThresholdSampleFloor(self):
        """The trace FWHM gate needs thresholds derived from quartz, not arcs.

        b, r and n clear the floor; m does not, because the only m-arm traces are
        the two short block B and block C sequences.
        """
        visitSet = loadValidationVisits()
        perArm = {}
        for entry in visitSet.knownGood:
            if entry.seqType != "Trace":
                continue
            for arm in entry.arms:
                perArm[arm] = perArm.get(arm, 0) + len(entry.visits) * len(entry.spectrographs)
        for arm in ("b", "r", "n"):
            assert perArm.get(arm, 0) >= MIN_SAMPLES, f"{arm}: only {perArm.get(arm, 0)} trace detectors"

    def testEveryKnownGoodEntryNamesItsSequence(self):
        """The flag-rate threshold key is derived from seqType; it cannot be blank."""
        for entry in loadValidationVisits().knownGood:
            assert entry.seqType, f"{entry.visits[0]} does not name its W_SEQNAM"

    def testPlaceholdersExcludedByDefault(self):
        """An unfilled entry must never silently validate a threshold."""
        assert not any(entry.placeholder for entry in loadValidationVisits())
        assert any(entry.placeholder for entry in loadValidationVisits(includePlaceholders=True))

    def testPlaceholdersCarryNoVisits(self):
        visitSet = loadValidationVisits(includePlaceholders=True)
        for entry in visitSet:
            if entry.placeholder:
                assert entry.visits == ()


class TestRun30KnownBad:
    def testObstructedFramesFail(self):
        visitSet = loadValidationVisits()
        assert visitSet.expectationFor(150115, arm="b", spectrograph=1) == "FAIL"
        assert visitSet.expectationFor(150641, arm="b", spectrograph=1) == "FAIL"

    def testPartialIlluminationWarnsOnTheLineCount(self):
        """The defect is how many fibers were measured, not their shape."""
        visitSet = loadValidationVisits()
        (entry,) = [e for e in visitSet.knownBad if 149398 in e.visits]
        assert entry.expect == "WARN"
        assert entry.metric == "nLines"

    def testFlexureCaseIsAttributedToTheCalibComparison(self):
        """Phase 2's reference case: a real offset against detectorMap_calib."""
        visitSet = loadValidationVisits()
        flexure = [e for e in visitSet.knownBad if e.visits[0] in (150779, 150782)]
        assert len(flexure) == 2
        assert all(entry.metric == "medDxCenter" for entry in flexure)
        assert all(entry.expect == "WARN" for entry in flexure)


class TestUnconfirmedVerdicts:
    """Suspected faults, recorded so somebody checks them."""

    def testUnconfirmedEntriesAreLoaded(self):
        """They are real visits; hiding them defeats the point of recording them."""
        visitSet = loadValidationVisits()
        suspect = {entry.visits[0] for entry in visitSet.knownBad if entry.unconfirmed}
        assert suspect == {149883, 150661}

    def testConfirmedBadExcludesThem(self):
        """A guess must not decide whether a threshold separates the bad data."""
        visitSet = loadValidationVisits()
        assert len(visitSet.confirmedBad) == len(visitSet.knownBad) - 2
        assert all(not entry.unconfirmed for entry in visitSet.confirmedBad)

    def testTheyStillCarryAVerdictAndAReason(self):
        """Unconfirmed means "not established", not "unspecified"."""
        for entry in loadValidationVisits().knownBad:
            if entry.unconfirmed:
                assert entry.expect == "FAIL"
                assert entry.reason

    def testConfirmedEntriesDefaultToConfirmed(self, writeYaml):
        path = writeYaml("version: 1\nknown_bad:\n  - {visit: 1, sequenceType: scienceArc, expect: FAIL}\n")
        (entry,) = loadValidationVisits(path).knownBad
        assert not entry.unconfirmed


class TestEntryMatching:
    def testMatchesRespectsSelectors(self):
        entry = ValidationVisit(visits=(100, 101), expect="FAIL", arms=("b",), spectrographs=(1,))
        assert entry.matches(100, arm="b", spectrograph=1)
        assert not entry.matches(102, arm="b", spectrograph=1)
        assert not entry.matches(100, arm="r", spectrograph=1)
        assert not entry.matches(100, arm="b", spectrograph=2)

    def testOmittedSelectorMatchesEverything(self):
        entry = ValidationVisit(visits=(100,), expect="PASS")
        assert entry.matches(100, arm="n", spectrograph=4, seqType="Quartz")

    def testUnknownSelectorIsNotApplied(self):
        """Passing ``None`` means "do not filter on this", not "no match"."""
        entry = ValidationVisit(visits=(100,), expect="PASS", arms=("b",))
        assert entry.matches(100, arm=None)


class TestExpectation:
    def testUnknownVisitHasNoExpectation(self):
        """Absence from the set is "no expectation", never an implied PASS."""
        visitSet = loadValidationVisits()
        assert visitSet.expectationFor(1) is None

    def testWorstExpectationWins(self, writeYaml):
        path = writeYaml(
            """
            version: 1
            known_good:
              - visit: 100
                sequenceType: scienceArc
            known_bad:
              - visit: 100
                sequenceType: scienceArc
                expect: WARN
              - visit: 100
                sequenceType: scienceArc
                expect: FAIL
            """
        )
        assert loadValidationVisits(path).expectationFor(100) == "FAIL"


class TestParsing:
    def testVisitRangeIsInclusive(self, writeYaml):
        path = writeYaml("version: 1\nknown_good:\n  - {visitRange: [10, 12], sequenceType: scienceTrace}\n")
        (entry,) = loadValidationVisits(path).knownGood
        assert entry.visits == (10, 11, 12)
        assert entry.expect == "PASS", "known_good defaults to PASS"

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ("version: 2\n", "unsupported schema version"),
            ("version: 1\nknown_good: 3\n", "must be a list"),
            ("version: 1\nknown_good: 0\n", "must be a list"),
            ("version: 1\nruns: []\n", "must be a mapping"),
            ("version: 1\nruns: 0\n", "must be a mapping"),
            ("version: 1\nknown_good:\n  - 3\n", "expected a mapping"),
            ("version: 1\nknown_good:\n  - {}\n", "one of 'visit' or 'visitRange' is required"),
            ("version: 1\nknown_good:\n  - {visit: 1, visitRange: [1, 2]}\n", "not both"),
            ("version: 1\nknown_good:\n  - {visitRange: [9, 1]}\n", "inverted"),
            ("version: 1\nknown_good:\n  - {visitRange: [1]}\n", "two-element list"),
            ("version: 1\nknown_good:\n  - {visit: notanint}\n", "must be an integer"),
            ("version: 1\nknown_good:\n  - {visit: true}\n", "must be an integer"),
            ("version: 1\nknown_good:\n  - {visit: 1, expect: MAYBE}\n", "invalid expect"),
            (
                "version: 1\nknown_good:\n  - {visit: 1, sequenceType: scienceArc, arms: b}\n",
                "must be a list",
            ),
            ("version: 1\nknown_bad:\n  - {visit: 1}\n", "'expect' is required"),
            (
                "version: 1\nknown_good:\n  - {visit: 1, sequenceType: scienceArc, expect: FAIL}\n",
                "must expect PASS",
            ),
            ("version: 1\nknown_good:\n  - {visit: 1}\n", "'sequenceType' is required"),
            ("version: 1\nknown_good:\n  - {visit: 1, sequenceType: arc}\n", "invalid sequenceType"),
            ("version: 1\nknown_good:\n  - {visit: 1, arm: [b]}\n", "unknown keys: arm"),
            ("version: 1\nknown_goods: []\n", "unknown top-level keys"),
            ('version: 1\nknown_good:\n  - {visit: 1, placeholder: "false"}\n', "must be true or false"),
            (
                "version: 1\nknown_bad:\n  - {visit: 1, sequenceType: scienceArc, expect: FAIL, unconfirmed: yes please}\n",
                "must be true or false",
            ),
        ],
    )
    def testMalformedEntriesRaise(self, writeYaml, body, message):
        """Validation is strict: a dropped entry stops being a reference."""
        with pytest.raises(ValueError, match=message):
            loadValidationVisits(writeYaml(body))

    def testMissingFileRaises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            loadValidationVisits(tmp_path / "nope.yaml")


class TestRowMatching:
    """Matching a metrics table against entries, as `calibrate` does."""

    @staticmethod
    def frame():
        return pd.DataFrame(
            {
                "visit": [100, 100, 101, 102],
                "arm": ["b", "r", "b", "b"],
                "spectrograph": [1, 1, 2, 1],
                "seqName": ["Arc: Neon", "Arc: Neon", None, "Trace"],
            },
            # Duplicated, as after a plain pd.concat: matching must be positional.
            index=[0, 0, 1, 1],
        )

    def testSelectorsApply(self):
        entry = ValidationVisit(visits=(100, 101, 102), expect="PASS", arms=("b",), seqType="Arc: Neon")
        assert matchRows(self.frame(), [entry]).tolist() == [True, False, True, False]

    def testNullColumnValueDoesNotRestrict(self):
        """Visit 101 has no seqName: unknown, so the seqType selector is not applied."""
        entry = ValidationVisit(visits=(101,), expect="PASS", seqType="Arc: Neon")
        assert matchRows(self.frame(), [entry]).tolist() == [False, False, True, False]

    def testMissingColumnDoesNotRestrict(self):
        entry = ValidationVisit(visits=(100,), expect="PASS", spectrographs=(1,))
        frame = self.frame().drop(columns="spectrograph")
        assert matchRows(frame, [entry]).tolist() == [True, True, False, False]

    def testAgreesWithEntryMatches(self):
        """The vectorised match and `ValidationVisit.matches` are the same rule."""
        visitSet = loadValidationVisits()
        frame = pd.DataFrame(
            [
                {"visit": visit, "arm": arm, "spectrograph": spectrograph, "seqName": seqName}
                for visit in (133025, 133037, 133042, 134334, 140005, 149883, 150779)
                for arm in ("b", "r", "m")
                for spectrograph in (1, 2)
                for seqName in ("Arc: Argon", "Arc: HgCd", "Twilight sky", "Arc: Neon")
            ]
        )
        for entry in visitSet:
            expected = [
                entry.matches(row.visit, row.arm, row.spectrograph, row.seqName) for row in frame.itertuples()
            ]
            assert matchRows(frame, [entry]).tolist() == expected

    def testSequenceTypeMismatches(self):
        """The obsType column comes from W_SEQTYP; a disagreeing sequenceType is reported."""
        frame = self.frame().assign(obsType=["arc", "arc", "trace", "trace"])
        arcs = ValidationVisit(visits=(100, 101), expect="PASS", sequenceType="scienceArc")
        traces = ValidationVisit(visits=(102,), expect="PASS", sequenceType="scienceTrace")
        mismatched = sequenceTypeMismatches(frame, ValidationVisitSet(knownGood=(arcs, traces)))
        assert mismatched["visit"].tolist() == [101]
        assert mismatched["sequenceType"].tolist() == ["scienceArc"]

    def testVisitsOfType(self):
        visitSet = loadValidationVisits()
        objects = visitSet.visitsOfType("scienceObject")
        assert 134880 in objects and 133025 not in objects
        assert set(visitSet.visitsOfType("scienceArc", "scienceTrace", "scienceObject")) == set(
            visitSet.visits
        )

    def testUnmatchedEntries(self):
        matched = ValidationVisit(visits=(100,), expect="PASS", seqType="Arc: Neon")
        typo = ValidationVisit(visits=(100,), expect="PASS", seqType="Arc: Noen")
        absent = ValidationVisit(visits=(999,), expect="PASS")
        visitSet = ValidationVisitSet(knownGood=(matched, typo, absent))
        assert unmatchedEntries(self.frame(), visitSet) == (typo, absent)

    def testSelectRowsKeepsOrder(self):
        entry = ValidationVisit(visits=(100, 102), expect="PASS")
        assert selectRows(self.frame(), [entry])["visit"].tolist() == [100, 100, 102]

    def testNoVisitColumnRaises(self):
        with pytest.raises(KeyError, match="visit"):
            matchRows(pd.DataFrame({"arm": ["b"]}), [])


class TestVisitExpression:
    def testCollapsesRuns(self):
        assert visitExpression([5, 1, 2, 3, 3, 9, 10]) == "visit IN (1..3, 5, 9..10)"

    def testCoversTheSet(self):
        """Every visit, and only those: expand the ranges back and compare."""
        visitSet = loadValidationVisits()
        terms = visitExpression(visitSet.visits).removeprefix("visit IN (").removesuffix(")").split(", ")
        expanded = set()
        for term in terms:
            first, _, last = term.partition("..")
            expanded.update(range(int(first), int(last or first) + 1))
        assert expanded == set(visitSet.visits)

    def testEmptyRaises(self):
        with pytest.raises(ValueError):
            visitExpression([])

    def testCommandLine(self, capsys):
        main(["expression"])
        assert capsys.readouterr().out.startswith("visit IN (133025..")


class TestTables:
    def testSectionsAndRows(self):
        visitSet = loadValidationVisits(includePlaceholders=True)
        text = formatTables(visitSet)
        for heading in ("### Known good", "### Known bad", "### Unconfirmed", "### Placeholders"):
            assert heading in text
        assert (
            sum(line.startswith("| ") and "---" not in line for line in text.splitlines())
            == len(visitSet) + 4
        ), "one row per entry plus a header per table"
        assert "| 140005–140006 | 27 | scienceArc | b, r, n | 1 | Arc: Argon | FAIL | `medFwhm` |" in text

    def testEveryRowHasTheHeadersColumns(self):
        """A cell added to the rows but not the header breaks the Markdown table."""
        for block in formatTables(loadValidationVisits(includePlaceholders=True)).split("### ")[1:]:
            rows = [line for line in block.splitlines() if line.startswith("|")]
            widths = {line.count(" | ") for line in rows if "---" not in line}
            assert len(widths) == 1, block.splitlines()[0]

    def testPipesAreEscaped(self):
        entry = ValidationVisit(visits=(1,), expect="PASS", note="a | b")
        assert "a \\| b" in formatTables(ValidationVisitSet(knownGood=(entry,)))

    def testDocsMatchTheYaml(self, capsys):
        """docs/validation-visits.md carries the generated tables; regenerate them when the YAML changes."""
        doc = (Path(__file__).parents[2] / "docs" / "validation-visits.md").read_text()
        begin = "<!-- BEGIN GENERATED: validationVisits tables -->\n"
        end = "<!-- END GENERATED: validationVisits tables -->"
        stored = doc[doc.index(begin) + len(begin) : doc.index(end)]
        main(["tables"])
        assert stored == capsys.readouterr().out, (
            "docs/validation-visits.md is out of date: paste the output of "
            "`python -m pfs.drp.qa.metrics.validationVisits tables` between its GENERATED markers"
        )


class TestRuns:
    BODY = """
        version: 1
        runs:
          25: [100, 199]
          27: [300, 399]
        referenceRuns: [25]
        known_good:
          - {visit: 100, sequenceType: scienceArc}
        known_bad:
          - {visit: 300, sequenceType: scienceArc, expect: FAIL}
        """

    def testEntriesGetTheirRun(self, writeYaml):
        visitSet = loadValidationVisits(writeYaml(self.BODY))
        assert [entry.run for entry in visitSet] == [25, 27]
        assert visitSet.runOf(150) == 25 and visitSet.runOf(250) is None

    def testKnownGoodOutsideTheReferenceRunsRaises(self, writeYaml):
        """Thresholds come from the reference runs; another run's good visit has no use."""
        body = self.BODY.replace(
            "{visit: 100, sequenceType: scienceArc}", "{visit: 301, sequenceType: scienceArc}"
        )
        with pytest.raises(
            ValueError, match=r"known_good\[0\]: known_good entries must lie in referenceRuns"
        ):
            loadValidationVisits(writeYaml(body))
        # Negative control: without referenceRuns, any run may hold known-good visits.
        assert loadValidationVisits(writeYaml(body.replace("        referenceRuns: [25]\n", ""))).knownGood

    def testNoRunsTableMeansNoConstraint(self, writeYaml):
        visitSet = loadValidationVisits(
            writeYaml("version: 1\nknown_good:\n  - {visit: 1, sequenceType: scienceArc}\n")
        )
        assert visitSet.knownGood[0].run is None and not visitSet.referenceRuns

    @pytest.mark.parametrize(
        ("runs", "entry", "message"),
        [
            ("{25: [100, 199], 27: [150, 399]}", "{visit: 100, sequenceType: scienceArc}", "overlap"),
            ("{25: [100, 199]}", "{visit: 250, sequenceType: scienceArc}", "within one run"),
            (
                "{25: [100, 199], 27: [200, 299]}",
                "{visitRange: [190, 210], sequenceType: scienceArc}",
                "within one run",
            ),
        ],
    )
    def testMalformedRunsRaise(self, writeYaml, runs, entry, message):
        with pytest.raises(ValueError, match=message):
            loadValidationVisits(writeYaml(f"version: 1\nruns: {runs}\nknown_good:\n  - {entry}\n"))

    def testReferenceRunMustBeARun(self, writeYaml):
        with pytest.raises(ValueError, match="referenceRuns not in runs"):
            loadValidationVisits(writeYaml("version: 1\nruns: {25: [1, 9]}\nreferenceRuns: [30]\n"))

    def testCheckedInSetDerivesFromRun25Only(self):
        visitSet = loadValidationVisits()
        assert visitSet.referenceRuns == (25,)
        assert {entry.run for entry in visitSet.knownGood} == {25}

    def testBArmFlagRatesAreRecordedAsBad(self):
        """The baseline for drp_stella's adjustDetectorMap work: bad for pctFlagged only."""
        visitSet = loadValidationVisits()
        flagged = [
            entry for entry in visitSet.knownBad if entry.metric == "pctFlagged" and entry.arms == ("b",)
        ]
        assert {entry.seqType for entry in flagged} == {
            "Arc: Argon",
            "Arc: Xenon",
            "Arc: Neon",
            "Arc: Krypton",
        }
        assert all(entry.run == 25 for entry in flagged)


class TestSequenceGroups:
    def testOneGroupPerSequence(self):
        a = ValidationVisit(visits=(10, 11, 12), expect="PASS")
        b = ValidationVisit(visits=(20, 21), expect="PASS")
        groups = sequenceGroups(ValidationVisitSet(knownGood=(a, b)))
        assert groups == {10: 10, 11: 10, 12: 10, 20: 20, 21: 20}

    def testEntriesSharingAVisitAreOneSequence(self):
        """A fault and its control cover the same exposures."""
        fault = ValidationVisit(visits=(10, 11), expect="FAIL", spectrographs=(1,))
        control = ValidationVisit(visits=(11, 12), expect="PASS", spectrographs=(2,))
        groups = sequenceGroups(ValidationVisitSet(knownGood=(control,), knownBad=(fault,)))
        assert set(groups.values()) == {10}

    def testRestrictedToVisits(self):
        a = ValidationVisit(visits=(10, 11), expect="PASS")
        assert sequenceGroups(ValidationVisitSet(knownGood=(a,)), visits=[11, 99]) == {11: 10}

    def testCheckedInSetKeepsBlocksApart(self):
        """Run25 block A and block B argon are different sequences."""
        groups = sequenceGroups(loadValidationVisits())
        assert groups[133025] == groups[133027] != groups[133042]
