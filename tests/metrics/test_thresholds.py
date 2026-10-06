"""Tests for the threshold derivation procedure."""

import math
from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from pfs.drp.qa.metrics.thresholds import (
    MIN_SAMPLES,
    deriveThresholds,
    formatProvenance,
    quantileInterval,
    resolutionFor,
    roundToReadable,
    roundToStep,
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
    def testNearest(self, value, expected):
        assert roundToReadable(value) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("value", "direction", "expected"),
        [
            (3.41, 1, 3.5),
            (3.49, -1, 3.4),
            (0.0213, 1, 0.022),
            (-3.41, 1, -3.4),
            (-3.41, -1, -3.5),
            # Exact values stay put: 3.5 * 10 is 35.000000000000004 in floats.
            (3.5, 1, 3.5),
            (0.07, 1, 0.07),
        ],
    )
    def testDirected(self, value, direction, expected):
        assert roundToReadable(value, direction=direction) == pytest.approx(expected)

    def testNonFinitePassesThrough(self):
        assert np.isnan(roundToReadable(float("nan")))
        assert np.isinf(roundToReadable(float("inf"), direction=1))

    def testSignificantDigitsAreHonoured(self):
        assert roundToReadable(3.4712, significant=3) == pytest.approx(3.47)


class TestRoundToStep:
    @pytest.mark.parametrize(
        ("value", "step", "direction", "expected"),
        [
            (2.8431, 0.001, 1, 2.844),
            (2.8431, 0.001, -1, 2.843),
            (2.8431, 0.01, 0, 2.84),
            (3.5, 0.1, 1, 3.5),
            (94.2, 1.0, 1, 95.0),
            (-0.0123, 0.001, 1, -0.012),
        ],
    )
    def testRounding(self, value, step, direction, expected):
        assert roundToStep(value, step, direction) == expected

    @pytest.mark.parametrize(
        ("robustRms", "expected"),
        [(0.08, 0.001), (37.0, 1.0), (0.5, 0.01), (0.0, None), (float("nan"), None)],
    )
    def testResolution(self, robustRms, expected):
        assert resolutionFor(robustRms) == expected


class TestQuantileInterval:
    def testCoverage(self):
        """The interval covers the true quantile at least as often as claimed.

        Negative control: the neighbouring order statistics of the sample
        quantile, a too-narrow interval, cover far less often, so the test can
        tell a valid interval from a plausible-looking one.
        """
        rng = np.random.default_rng(1)
        n, q, trials = 400, 0.95, 2000
        truth = -math.log(1.0 - q)  # exponential quantile
        rank = int(n * q)
        covered = narrow = 0
        for _ in range(trials):
            sample = rng.exponential(size=n)
            low, high = quantileInterval(sample, q)
            covered += low <= truth <= high
            ordered = np.sort(sample)
            narrow += ordered[rank - 2] <= truth <= ordered[rank]
        assert covered / trials >= 0.94
        assert narrow / trials < 0.6

    def testSmallSampleCannotBoundTheTail(self):
        """24 detectors cannot bound p99: the upper end is open."""
        low, high = quantileInterval(np.linspace(2.5, 3.5, 24), 0.99)
        assert math.isfinite(low)
        assert high == math.inf

    def testLargeSampleBoundsTheTail(self):
        low, high = quantileInterval(np.linspace(0.0, 1.0, 2000), 0.99)
        assert 0.97 < low < 0.99 < high < 1.0

    def testLowQuantileIsOpenBelow(self):
        low, high = quantileInterval(np.linspace(0.0, 1.0, 24), 0.01)
        assert low == -math.inf
        assert math.isfinite(high)

    def testNonFiniteValuesAreDropped(self):
        assert quantileInterval([np.nan, np.inf], 0.5) == (-math.inf, math.inf)


class TestDeriveThresholds:
    def testPercentilesOfAKnownDistribution(self):
        """Uniform 0-100 puts p95 at 95 and p99 at 99."""
        values = np.linspace(0.0, 100.0, 1001)
        result = deriveThresholds(values, metric="test")
        assert result.warnRaw == pytest.approx(95.0, abs=0.2)
        assert result.failRaw == pytest.approx(99.0, abs=0.2)
        assert result.median == pytest.approx(50.0, abs=0.2)
        assert result.numSamples == 1001
        assert result.reliable
        assert result.failBounded

    def testNonFiniteValuesAreDropped(self):
        result = deriveThresholds([1.0, 2.0, np.nan, 3.0, np.inf], metric="test")
        assert result.numSamples == 3
        assert result.median == pytest.approx(2.0)

    def testRoundingIsOutwards(self):
        """Rounding never makes a threshold flag more of the good data.

        Negative control: rounding to nearest does, for some samples.
        """
        rng = np.random.default_rng(6)
        nearestFlagsMore = 0
        for _ in range(50):
            values = rng.normal(2.7, 0.03, 500)
            result = deriveThresholds(values, metric="medFwhm")
            assert result.warn >= result.warnRaw and result.fail >= result.failRaw
            assert result.goodFlaggedFail <= np.mean(values >= result.failRaw)
            nearest = roundToStep(result.failRaw, resolutionFor(result.robustRms))
            nearestFlagsMore += np.mean(values >= nearest) > np.mean(values >= result.failRaw)
        assert nearestFlagsMore > 0

    def testLowerIsWorseReflectsAndRoundsDown(self):
        values = np.linspace(0.0, 100.0, 1001)
        result = deriveThresholds(values, metric="nLines", higherIsWorse=False)
        assert result.warnRaw == pytest.approx(5.0, abs=0.2)
        assert result.failRaw == pytest.approx(1.0, abs=0.2)
        assert result.fail < result.warn
        assert result.warn <= result.warnRaw and result.fail <= result.failRaw

    def testPhysicalLimitIsUsedExactly(self):
        """Rounding 65535 to two digits would put FAIL above saturation."""
        result = deriveThresholds(np.linspace(0.0, 100.0, 1001), metric="peak", physicalLimit=65535.0)
        assert result.fail == 65535.0
        assert result.failBounded
        assert "physical limit 65535" in result.provenance

    def testInSampleFlagRates(self):
        result = deriveThresholds(np.linspace(0.0, 100.0, 1001), metric="test")
        assert result.goodFlaggedWarn == pytest.approx(0.05, abs=0.01)
        assert result.goodFlaggedFail == pytest.approx(0.01, abs=0.005)

    def testTooFewSamplesIsFlaggedNotHidden(self):
        result = deriveThresholds([1.0, 2.0, 3.0], metric="test")
        assert not result.reliable
        assert "UNRELIABLE" in str(result)
        assert "NOT RELIABLE" in result.provenance
        assert result.warn > 0 and result.fail > 0

    def testUnboundedFailIsFlagged(self):
        """Above the sample floor but too small to bound p99: the Run25 situation."""
        result = deriveThresholds(np.linspace(2.5, 3.5, MIN_SAMPLES + 4), metric="medFwhm")
        assert result.reliable
        assert not result.failBounded
        assert "FAIL UNBOUNDED" in str(result)
        assert "FAIL UNBOUNDED" in result.provenance

    def testProvenanceRecordsTheQuantilesActuallyUsed(self):
        result = deriveThresholds(np.linspace(0.0, 10.0, 100), metric="nLines", higherIsWorse=False)
        assert "p5" in result.provenance and "p1" in result.provenance
        assert "p95" not in result.provenance and "p99" not in result.provenance

    def testProvenanceRecordsVisitRangeAndDate(self):
        result = deriveThresholds(
            np.linspace(0.0, 10.0, 100),
            metric="test",
            visitRange="133025-135850",
            derivedOn=date(2026, 9, 17),
        )
        assert "validation visits 133025-135850" in result.provenance
        assert "2026-09-17" in result.provenance

    def testTightOffsetDistributionKeepsItsWarnLevel(self):
        """FWHM-like: 2.7 px with 0.03 px scatter. p95 and p99 stay apart.

        Negative control: two significant figures would round both to 2.8.
        """
        values = np.random.default_rng(5).normal(2.7, 0.03, 1000)
        result = deriveThresholds(values, metric="medFwhm")
        assert not result.degenerate
        assert result.fail - result.warn > 0.01
        assert roundToReadable(result.warnRaw, direction=1) == roundToReadable(result.failRaw, direction=1)

    def testDegeneratePairIsFlagged(self):
        """Ties in the tail, as for a metric that is mostly one value."""
        tied = np.r_[np.full(36, 3.2), np.full(4, 3.0)]
        result = deriveThresholds(tied, metric="medFwhm")
        assert result.warn == result.fail
        assert result.degenerate
        assert "DEGENERATE" in str(result)
        assert "WARN can never fire" in result.provenance

    def testPhysicalLimitBelowWarnIsDegenerate(self):
        result = deriveThresholds(np.linspace(2.0, 4.0, 200), metric="medFwhm", physicalLimit=3.0)
        assert result.degenerate

    def testAHealthySpreadIsNotDegenerate(self):
        result = deriveThresholds(np.linspace(2.0, 4.0, 200), metric="medFwhm")
        assert not result.degenerate
        assert "DEGENERATE" not in result.provenance

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
    def suggestion(higherIsWorse=True):
        values = np.linspace(2.0, 3.0, 1000) if higherIsWorse else np.linspace(5.0, 6.0, 1000)
        result = deriveThresholds(values, metric="medFwhm", higherIsWorse=higherIsWorse)
        return replace(result, warn=3.2, fail=3.5) if higherIsWorse else replace(result, warn=5.0, fail=4.0)

    def testKnownBadBeyondFailPasses(self):
        """The documented SM1 FWHM range."""
        ok, message = verifyKnownBad([3.83, 4.2, 4.86], self.suggestion())
        assert ok
        assert "3 >= FAIL" in message

    def testKnownBadShortOfFailIsReported(self):
        ok, message = verifyKnownBad([3.3, 3.6], self.suggestion())
        assert not ok
        assert "1 >= FAIL" in message
        assert "expected FAIL" in message

    def testWarnEntriesAreHeldToWarn(self):
        """A WARN entry is satisfied by WARN; holding it to FAIL would fail it wrongly."""
        assert verifyKnownBad([3.3, 3.4], self.suggestion(), expect="WARN")[0]
        assert not verifyKnownBad([3.3, 3.4], self.suggestion(), expect="FAIL")[0]
        assert not verifyKnownBad([3.0, 3.4], self.suggestion(), expect="WARN")[0]

    def testLowerIsWorseComparesDownwards(self):
        assert verifyKnownBad([1.0, 4.0], self.suggestion(higherIsWorse=False))[0]
        assert not verifyKnownBad([4.5], self.suggestion(higherIsWorse=False))[0]

    def testNoKnownBadDataIsNotAPass(self):
        with pytest.raises(ValueError, match="cannot verify"):
            verifyKnownBad([np.nan], self.suggestion())

    def testExpectMustBeABadVerdict(self):
        with pytest.raises(ValueError, match="WARN or FAIL"):
            verifyKnownBad([4.0], self.suggestion(), expect="PASS")


class TestFormatProvenance:
    def testMentionsTheSetWhenNoRangeGiven(self):
        text = formatProvenance(None, date(2026, 1, 2), 42, 95.0, 99.0)
        assert "validation visit set" in text
        assert "n=42" in text
        assert "p95" in text and "p99" in text
