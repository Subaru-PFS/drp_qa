"""Tests for the threshold derivation procedure.

Stack-free: numpy plus ``pfs.drp.qa.metrics.thresholds``.
"""

from datetime import date

import numpy as np
import pytest

from pfs.drp.qa.metrics.thresholds import (
    ThresholdSuggestion,
    deriveThresholds,
    formatProvenance,
    roundToReadable,
    verifyKnownBad,
)


class TestRoundToReadable:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (3.4712, 3.5),
            (0.021384, 0.021),
            (94.7213, 95.0),
            (1234.5, 1200.0),
            (-3.4712, -3.5),
            (0.0, 0.0),
        ],
    )
    def testRounding(self, value, expected):
        assert roundToReadable(value) == pytest.approx(expected)

    def testNonFinitePassesThrough(self):
        assert np.isnan(roundToReadable(float("nan")))
        assert np.isinf(roundToReadable(float("inf")))

    def testSignificantDigitsAreHonoured(self):
        assert roundToReadable(3.4712, significant=3) == pytest.approx(3.47)


class TestDeriveThresholds:
    def testPercentilesOfAKnownDistribution(self):
        """Uniform 0-100 puts p95 at 95 and p99 at 99, to rounding."""
        values = np.linspace(0.0, 100.0, 1001)
        result = deriveThresholds(values, metric="test")
        assert result.warnRaw == pytest.approx(95.0, abs=0.2)
        assert result.failRaw == pytest.approx(99.0, abs=0.2)
        assert result.median == pytest.approx(50.0, abs=0.2)
        assert result.numSamples == 1001
        assert result.reliable

    def testNonFiniteValuesAreDropped(self):
        values = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
        result = deriveThresholds(values, metric="test")
        assert result.numSamples == 3
        assert result.median == pytest.approx(2.0)

    def testLowerIsWorseReflectsThePercentiles(self):
        """For a metric that fails low, FAIL must sit below WARN."""
        values = np.linspace(0.0, 100.0, 1001)
        result = deriveThresholds(values, metric="nLines", higherIsWorse=False)
        assert result.warnRaw == pytest.approx(5.0, abs=0.2)
        assert result.failRaw == pytest.approx(1.0, abs=0.2)
        assert result.fail < result.warn

    def testPhysicalLimitOverridesTheFailPercentile(self):
        values = np.linspace(0.0, 100.0, 1001)
        result = deriveThresholds(values, metric="test", physicalLimit=65535.0)
        assert result.fail == pytest.approx(66000.0)
        assert "physical limit" in result.provenance

    def testTooFewSamplesIsFlaggedNotHidden(self):
        result = deriveThresholds([1.0, 2.0, 3.0], metric="test")
        assert not result.reliable
        assert "UNRELIABLE" in str(result)

    def testProvenanceRecordsTheQuantilesActuallyUsed(self):
        """A metric that fails low is derived from p5/p1, and must say so."""
        result = deriveThresholds(np.linspace(0.0, 10.0, 100), metric="nLines", higherIsWorse=False)
        assert "p5" in result.provenance and "p1" in result.provenance
        assert "p95" not in result.provenance and "p99" not in result.provenance

    def testProvenanceRecordsVisitRangeAndDate(self):
        result = deriveThresholds(
            np.linspace(0.0, 10.0, 100),
            metric="test",
            visitRange="140200-140260",
            derivedOn=date(2026, 9, 17),
        )
        assert "140200-140260" in result.provenance
        assert "2026-09-17" in result.provenance

    def testUnreliableProvenanceSaysSo(self):
        """A 16-detector derivation must not read like a 400-detector one."""
        result = deriveThresholds([3.0, 3.1, 3.2], metric="medFwhm", visitRange="133054-135850")
        assert not result.reliable
        assert "NOT RELIABLE" in result.provenance
        assert "below the 20-sample floor" in result.provenance
        # The numbers are still there: the point is to report them, flagged.
        assert result.warn > 0 and result.fail > 0

    def testReliableProvenanceCarriesNoWarning(self):
        result = deriveThresholds(np.linspace(2.5, 3.5, 100), metric="medFwhm")
        assert result.reliable
        assert "NOT RELIABLE" not in result.provenance

    def testEmptyInputRaises(self):
        with pytest.raises(ValueError, match="No finite values"):
            deriveThresholds([np.nan, np.nan], metric="test")

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"warnPercentile": 101.0}, "not a percentile"),
            ({"failPercentile": -1.0}, "not a percentile"),
            ({"warnPercentile": 99.0, "failPercentile": 95.0}, "below warnPercentile"),
        ],
    )
    def testInvalidPercentilesRaise(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            deriveThresholds([1.0, 2.0], metric="test", **kwargs)


class TestVerifyKnownBad:
    @staticmethod
    def suggestion(higherIsWorse=True, fail=3.5):
        return ThresholdSuggestion(
            metric="medFwhm",
            warn=3.2,
            fail=fail,
            warnRaw=3.2,
            failRaw=fail,
            median=2.8,
            robustRms=0.1,
            numSamples=100,
            higherIsWorse=higherIsWorse,
            reliable=True,
            provenance="test",
        )

    def testKnownBadAboveFailPasses(self):
        """The documented SM1 FWHM range, against the shipped 3.5 px FAIL."""
        ok, message = verifyKnownBad([3.83, 4.2, 4.86], self.suggestion())
        assert ok
        assert "3/3" in message

    def testKnownBadBelowFailIsReported(self):
        ok, message = verifyKnownBad([3.0, 3.6], self.suggestion())
        assert not ok
        assert "1/2" in message
        assert "does not separate" in message

    def testLowerIsWorseComparesDownwards(self):
        ok, _ = verifyKnownBad([1.0, 2.0], self.suggestion(higherIsWorse=False, fail=5.0))
        assert ok

    def testNoKnownBadDataIsNotAPass(self):
        with pytest.raises(ValueError, match="cannot verify"):
            verifyKnownBad([np.nan], self.suggestion())


class TestFormatProvenance:
    def testMentionsGoldenSetWhenNoRangeGiven(self):
        text = formatProvenance(None, date(2026, 1, 2), 42, 95.0, 99.0)
        assert "golden visit set" in text
        assert "n=42" in text
        assert "p95" in text and "p99" in text
