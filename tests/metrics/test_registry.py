"""The registry: the 13 checks, the frame types and the lamp names."""

import pytest

from pfs.drp.qa.metrics.registry import (
    ALLSKY,
    ARC,
    CHECKS,
    EXCLUSIONS,
    FIXED,
    FRAME_TYPES,
    IIS_ARC,
    IIS_QUARTZ,
    METRIC_SPECS,
    QUARTZ,
    RELATIVE,
    SCIENCE,
    TWILIGHT,
    Check,
    frameType,
    lampOf,
    specFor,
)

#: Every pair of limits, as the plan of record states them: check -> (kind, WARN, FAIL).
PAIRS = {
    "skyOffset": (FIXED, 0.5, 1.0),
    "skyLineResidual": (FIXED, 0.5, 1.0),
    "skyLineResidualBlue": (FIXED, 1.5, 2.5),
    "calibOffset": (FIXED, 1.0, 2.0),
    "traceCentreOffset": (FIXED, 1.0, 2.0),
    "twilightLevel": (FIXED, 2.0, 3.0),
    "twilightScatter": (FIXED, 1.5, 2.5),
    "twilightPatch": (FIXED, 2.5, 4.0),
    "width": (RELATIVE, 3.0, 5.0),
    "scatterX": (RELATIVE, 20.0, 50.0),
    "scatterY": (RELATIVE, 50.0, 150.0),
    "linesPerFiberArc": (RELATIVE, 25.0, 50.0),
    "linesPerFiberQuartz": (RELATIVE, 2.5, 5.0),
}


def testThirteenPairs():
    assert {name: (check.kind, check.warn, check.fail) for name, check in CHECKS.items()} == PAIRS
    assert len(CHECKS) == 13


def testChecksAreWellFormed():
    for name, check in CHECKS.items():
        assert check.name == name and check.title
        assert 0 < check.warn < check.fail
        assert check.frames and set(check.frames) <= set(FRAME_TYPES)
        assert check.metrics and all(metric in METRIC_SPECS for metric in check.metrics)
        # A relative check says what its normal value is kept per; a fixed one has none.
        assert bool(check.by) == (check.kind == RELATIVE)
        if check.by:
            assert check.by[0] in ("camera", "arm")


def testScienceFramesHaveOnlyTheFixedChecks():
    for kind in (SCIENCE, ALLSKY):
        names = {name for name, check in CHECKS.items() if kind in check.frames}
        assert names == {"skyOffset", "skyLineResidual", "skyLineResidualBlue"}
        assert all(CHECKS[name].kind == FIXED for name in names)
    # The two residual checks split the arms between them.
    assert set(CHECKS["skyLineResidual"].arms) | set(CHECKS["skyLineResidualBlue"].arms) == set("brnm")
    assert not set(CHECKS["skyLineResidual"].arms) & set(CHECKS["skyLineResidualBlue"].arms)


def testOneRelativeCheckPerMetricAndFrameType():
    """A normal value is keyed by metric and frame type: two relative checks may not share one."""
    seen = set()
    for check in CHECKS.values():
        if check.kind != RELATIVE:
            continue
        for metric in check.metrics:
            for frame in check.frames:
                assert (metric, frame) not in seen
                seen.add((metric, frame))


def testTwilightNArmIsNotJudged():
    for name in ("twilightLevel", "twilightScatter", "twilightPatch"):
        assert CHECKS[name].frames == (TWILIGHT,) and "n" not in CHECKS[name].arms and CHECKS[name].arms


@pytest.mark.parametrize(
    "obsType, illumination, expected",
    [
        ("science", "full", SCIENCE),
        ("allsky", None, ALLSKY),
        ("arc", "full", ARC),
        ("trace", "full", QUARTZ),
        ("arc", "iis", IIS_ARC),
        ("trace", "iis", IIS_QUARTZ),
        ("twilight", "full", TWILIGHT),
        ("unknown", "full", None),
        ("dark", None, None),
        (None, None, None),
        (float("nan"), "iis", None),
        ("science", "iis", None),
    ],
)
def testFrameType(obsType, illumination, expected):
    assert frameType(obsType, illumination) == expected


def testFrameTypeKeepsTheIlluminationsApart():
    # Negative control: an engineering-fiber arc is not an arc, so no full-slit check or normal value reaches it.
    assert frameType("arc", "iis") != frameType("arc", "full")
    assert frameType("arc", "iis") not in CHECKS["scatterX"].frames


@pytest.mark.parametrize(
    "seqName, lamp",
    [
        ("Arc: HgCd", "HgCd"),
        ("ImageQuality (Ne)", "Ne"),
        ("Trace", ""),
        ("", ""),
        (None, ""),
        (float("nan"), ""),
    ],
)
def testLampOf(seqName, lamp):
    assert lampOf(seqName) == lamp


def testLimits():
    assert CHECKS["skyOffset"].limits() == (0.5, 1.0)
    warn, fail = CHECKS["width"].limits(normal=2.0, higherIsWorse=True)
    assert (warn, fail) == pytest.approx((2.06, 2.10))
    warn, fail = CHECKS["linesPerFiberArc"].limits(normal=80.0, higherIsWorse=False)
    assert (warn, fail) == pytest.approx((60.0, 40.0))
    # Negative control: the direction matters.
    assert CHECKS["linesPerFiberArc"].limits(normal=80.0, higherIsWorse=True)[0] != pytest.approx(60.0)


def testSpecs():
    assert specFor("linesPerFiber").higherIsWorse is False
    assert specFor("medianDeltaYKms").absolute and specFor("medianDeltaX").absolute
    assert specFor("noSuchMetric").higherIsWorse is True
    assert specFor("medianDeltaX").show(-0.514) == "-0.51 px"
    assert isinstance(CHECKS["width"], Check)


def testExclusionsGiveReasons():
    assert len(EXCLUSIONS) >= 8
    assert all(what and why for what, why in EXCLUSIONS)
