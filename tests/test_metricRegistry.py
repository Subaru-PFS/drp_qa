"""Tests for the metric registry and the gating path built on it.

Stack-free: no Butler, no ``lsst.pex.config``. The ``imageQualityQa`` config is
stood in for by a `types.SimpleNamespace`, which is exactly the point of
`buildImageQualityRegistry` taking a duck-typed object.
"""

from types import SimpleNamespace

import pytest

from pfs.drp.qa.metrics.definitions import (
    IQ_FLAG_RATE_FALLBACK,
    buildImageQualityRegistry,
    imageQualityMetricDefs,
)
from pfs.drp.qa.metrics.registry import (
    STATUS_ORDER,
    UNKNOWN,
    MetricDef,
    MetricRegistry,
    Thresholds,
    worstStatus,
)


@pytest.fixture
def fwhm():
    """Return a metric standing in for medFwhm."""
    return MetricDef(
        name="medFwhm",
        units="pixels",
        reference="optical design spot size",
        thresholds=Thresholds(warn=3.2, fail=3.5),
        label="medFWHM",
        unitSuffix="px",
        valueFormat=".2f",
    )


@pytest.fixture
def iqConfig():
    """Return the shipped imageQualityQa threshold defaults as a plain namespace."""
    return SimpleNamespace(
        fwhmWarnThreshold=3.2,
        fwhmFailThreshold=3.5,
        traceFwhmWarnThreshold=3.2,
        traceFwhmFailThreshold=3.5,
        dxCenterWarnThreshold=1.0,
        dxCenterFailThreshold=2.0,
        flagRateWarnThreshold={
            "b": 50.0,
            "r": 15.0,
            "n": 15.0,
            "m": 15.0,
            "b:HgCd": 15.0,
            "b:Neon": 50.0,
            "b:Krypton": 55.0,
            "b:Xenon": 85.0,
            "b:Argon": 93.0,
        },
        flagRateFailThreshold={
            "b": 60.0,
            "r": 20.0,
            "n": 20.0,
            "m": 20.0,
            "b:HgCd": 25.0,
            "b:Neon": 60.0,
            "b:Krypton": 65.0,
            "b:Xenon": 92.0,
            "b:Argon": 97.0,
        },
    )


class TestGate:
    @pytest.mark.parametrize(
        ("value", "status"),
        [(2.9, "PASS"), (3.19, "PASS"), (3.2, "WARN"), (3.49, "WARN"), (3.5, "FAIL"), (9.0, "FAIL")],
    )
    def testThresholdsAreInclusive(self, fwhm, value, status):
        """Matches the ``>=`` comparisons the tasks used before the registry."""
        assert fwhm.gate(value).status == status

    def testReasonNamesTheThresholdThatWasCrossed(self, fwhm):
        result = fwhm.gate(3.6)
        assert result.reason == "medFWHM=3.60px >= fail threshold 3.5px"
        assert result.threshold == 3.5

    def testPassCarriesNoReason(self, fwhm):
        result = fwhm.gate(2.5)
        assert result.status == "PASS"
        assert result.reason is None
        assert result.threshold is None

    @pytest.mark.parametrize("value", [float("nan"), None])
    def testUnmeasuredValuesGetNoVerdict(self, fwhm, value):
        """``None`` means "not judged", which is not the same as PASS."""
        assert fwhm.gate(value) is None

    def testInfiniteValueStillFails(self, fwhm):
        """A measurement that blew up is a fault, not a missing measurement."""
        assert fwhm.gate(float("inf")).status == "FAIL"

    def testUngatedMetricGetsNoVerdict(self):
        """A metric with no thresholds must not claim to have been checked."""
        metric = MetricDef(name="dxCenterRms", units="pixels", reference="detectorMap_calib")
        assert metric.gate(12345.0) is None

    def testLowerIsWorseComparesDownwards(self):
        metric = MetricDef(
            name="nLines",
            units="count",
            reference="line list",
            higherIsWorse=False,
            thresholds=Thresholds(warn=20.0, fail=10.0),
        )
        assert metric.gate(50.0).status == "PASS"
        assert metric.gate(15.0).status == "WARN"
        assert metric.gate(5.0).status == "FAIL"
        assert "<=" in metric.gate(5.0).reason

    def testUseAbsoluteGatesOnMagnitude(self):
        metric = MetricDef(
            name="medDxCenter",
            units="pixels",
            reference="detectorMap_calib",
            thresholds=Thresholds(warn=1.0, fail=2.0),
            useAbsolute=True,
            label="|dxCenter|",
            unitSuffix="px",
            valueFormat=".3f",
        )
        assert metric.gate(-2.1).status == "FAIL"
        assert metric.gate(-2.1).reason == "|dxCenter|=2.100px >= fail threshold 2.0px"


class TestOverrides:
    @pytest.fixture
    def flagRate(self):
        return MetricDef(
            name="pctFlagged",
            units="percent",
            reference="fitDetectorMap flags",
            thresholds=Thresholds(warn=15.0, fail=20.0),
            overrides={"b": Thresholds(50.0, 60.0), "b:Argon": Thresholds(93.0, 97.0)},
            unitSuffix="%",
            valueFormat=".1f",
        )

    def testMostSpecificKeyWins(self, flagRate):
        """``arm:species`` before ``arm`` before the base pair."""
        assert flagRate.gate(90.0, keys=("b:Argon", "b")).status == "PASS"
        assert flagRate.gate(90.0, keys=("b:Neon", "b")).status == "FAIL"
        assert flagRate.gate(90.0, keys=("r:Neon", "r")).status == "FAIL"

    def testMatchedKeyIsReported(self, flagRate):
        assert flagRate.gate(95.0, keys=("b:Argon", "b")).key == "b:Argon"
        assert flagRate.gate(95.0, keys=("b:Neon", "b")).key == "b"
        assert flagRate.gate(95.0, keys=("r:Neon",)).key is None

    def testEmptyKeysAreSkipped(self, flagRate):
        """An unclassified visit yields ``""`` for the species half of the key."""
        assert flagRate.gate(55.0, keys=("", "b")).key == "b"


class TestRegistry:
    def testDuplicateRegistrationIsRefused(self, fwhm):
        registry = MetricRegistry([fwhm])
        with pytest.raises(ValueError, match="already registered"):
            registry.register(fwhm)

    def testGatingAnUnregisteredMetricRaises(self, fwhm):
        """Silently passing an unknown metric would hide a whole metric."""
        with pytest.raises(KeyError):
            MetricRegistry([fwhm]).gate("noSuchMetric", 1.0)

    def testMembershipAndOrder(self, fwhm):
        other = MetricDef(name="pctFlagged", units="percent", reference="flags")
        registry = MetricRegistry([fwhm, other])
        assert registry.names == ("medFwhm", "pctFlagged")
        assert "medFwhm" in registry
        assert len(registry) == 2
        assert registry.get("nope") is None


class TestWorstStatus:
    def testWorstWins(self):
        assert worstStatus(["PASS", "FAIL", "WARN"]) == "FAIL"
        assert worstStatus(["PASS", "WARN"]) == "WARN"

    def testUnjudgedEntriesAreSkipped(self, fwhm):
        assert worstStatus([None, None]) == "PASS"
        assert worstStatus([None, fwhm.gate(3.6)]) == "FAIL"

    def testGateResultsAndStringsMix(self, fwhm):
        assert worstStatus([fwhm.gate(2.0), "WARN"]) == "WARN"

    def testUnknownStatusRaises(self):
        with pytest.raises(ValueError, match="Unknown status"):
            worstStatus(["MAYBE"])

    def testStatusOrderIsBestToWorst(self):
        assert STATUS_ORDER == ("PASS", "WARN", "FAIL")

    def testUnknownIsOutsideTheOrdering(self):
        """It is not a severity; a detector that could not be measured is unassessed."""
        assert UNKNOWN not in STATUS_ORDER

    def testDefaultIsReturnedWhenNothingWasJudged(self):
        """The case that makes a vacuous PASS impossible."""
        assert worstStatus([None, None], default=UNKNOWN) == UNKNOWN
        assert worstStatus([], default=UNKNOWN) == UNKNOWN

    def testOneRealVerdictBeatsTheDefault(self, fwhm):
        """A single measurable metric still decides; UNKNOWN means none of them were."""
        assert worstStatus([None, fwhm.gate(2.0), None], default=UNKNOWN) == "PASS"
        assert worstStatus([None, fwhm.gate(3.6)], default=UNKNOWN) == "FAIL"


class TestImageQualityDefinitions:
    def testEveryMetricStatesItsReferenceAndProvenance(self):
        """R1 and R2 are only enforceable if the fields are never left blank."""
        for metric in imageQualityMetricDefs():
            assert metric.reference.strip(), f"{metric.name} states no external reference"
            assert metric.provenance.strip(), f"{metric.name} states no threshold provenance"

    def testRegistryReproducesTheShippedVerdicts(self, iqConfig):
        """The migration must not move a single verdict boundary."""
        registry = buildImageQualityRegistry(iqConfig)
        assert registry.gate("medFwhm", 3.4).status == "WARN"
        assert registry.gate("medFwhm", 3.5).status == "FAIL"
        assert registry.gate("medDxCenter", -1.5).status == "WARN"
        assert registry.gate("pctFlagged", 55.0, keys=("b:HgCd", "b")).status == "FAIL"
        assert registry.gate("pctFlagged", 55.0, keys=("b:Neon", "b")).status == "WARN"
        assert registry.gate("pctFlagged", 55.0, keys=("b:Argon", "b")).status == "PASS"
        assert registry.gate("pctFlagged", 16.0, keys=("r:Neon", "r")).status == "WARN"

    def testArmWithNoEntryFallsBackToTheOldDefault(self, iqConfig):
        """The 15/20 fallback the task hard-coded is preserved."""
        registry = buildImageQualityRegistry(iqConfig)
        assert registry.gate("pctFlagged", 16.0, keys=("x:Neon", "x")).status == "WARN"
        assert registry.gate("pctFlagged", 21.0, keys=("x:Neon", "x")).status == "FAIL"

    def testConfigOverridesTakeEffect(self, iqConfig):
        """Operators override thresholds on the command line; the registry must follow."""
        iqConfig.fwhmWarnThreshold = 9.0
        iqConfig.fwhmFailThreshold = 99.0
        registry = buildImageQualityRegistry(iqConfig)
        assert registry.gate("medFwhm", 4.0).status == "PASS"

    def testPartialOverrideResolvesTheMissingSideThroughTheArm(self, iqConfig):
        """A species with no FAIL entry takes its arm's, not the global fallback.

        Going straight to the fallback would rewrite a configured verdict: with
        b:Argon warning at 93 and only the arm's FAIL of 60 configured, a 70 %
        flag rate is a FAIL at 60 and would wrongly have been one at 20 too --
        but a 25 % rate must stay PASS, which the fallback would have failed.
        """
        iqConfig.flagRateWarnThreshold = {"b": 50.0, "b:Argon": 93.0}
        iqConfig.flagRateFailThreshold = {"b": 60.0}
        registry = buildImageQualityRegistry(iqConfig)
        result = registry.gate("pctFlagged", 25.0, keys=("b:Argon", "b"))
        assert result.status == "PASS", "the global 20 % fallback must not apply here"
        assert registry.gate("pctFlagged", 70.0, keys=("b:Argon", "b")).threshold == 60.0

    def testGlobalFallbackStillAppliesWhenTheArmHasNoEntryEither(self, iqConfig):
        iqConfig.flagRateWarnThreshold = {"z:Neon": 30.0}
        iqConfig.flagRateFailThreshold = {}
        registry = buildImageQualityRegistry(iqConfig)
        assert registry.gate("pctFlagged", 25.0, keys=("z:Neon",)).threshold == IQ_FLAG_RATE_FALLBACK.fail

    def testTraceFwhmGatesOnItsOwnThresholds(self, iqConfig):
        """A fiber-profile width is not an arc-line second moment.

        It borrows the arc values today, but through its own key, so re-deriving
        them from the golden set's quartz visits is a config change rather than a
        code change.
        """
        iqConfig.traceFwhmWarnThreshold = 4.0
        iqConfig.traceFwhmFailThreshold = 4.5
        registry = buildImageQualityRegistry(iqConfig)
        # 4.2 px fails the arc thresholds but only warns against the trace ones.
        assert registry.gate("medFwhm", 4.2).status == "FAIL"
        assert registry.gate("medFwhm", 4.2, keys=("trace",)).status == "WARN"
        assert registry.gate("medFwhm", 3.3, keys=("trace",)).status == "PASS"

    def testTraceFwhmIsActuallyGated(self, iqConfig):
        """It used to be skipped entirely, which left the trace path ungateable."""
        registry = buildImageQualityRegistry(iqConfig)
        assert registry.gate("medFwhm", 9.0, keys=("trace",)) is not None
        assert registry.gate("medFwhm", 9.0, keys=("trace",)).status == "FAIL"

    def testUngatedMetricsAreRegisteredButSilent(self, iqConfig):
        registry = buildImageQualityRegistry(iqConfig)
        assert "dxCenterRms" in registry
        assert registry.gate("dxCenterRms", 99.0) is None
