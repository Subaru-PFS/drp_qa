"""The registry: the 13 checks, what each sequence type is, the frame types and the lamp names."""

import pytest

from pfs.drp.qa.metrics.registry import (
    ALLSKY,
    ARC,
    BIAS,
    CHECKS,
    DARK,
    DITHERED,
    DOME_FLAT,
    DOT_ROACH,
    ENGINEERING,
    EXCLUSIONS,
    FIBER_PROFILE,
    FIXED,
    FRAME_TYPES,
    IIS_ARC,
    IIS_QUARTZ,
    METRIC_SPECS,
    NOT_JUDGED,
    QUARTZ,
    RELATIVE,
    SCIENCE,
    SEQUENCE_TYPES,
    THROUGH_FOCUS,
    TWILIGHT,
    WINDOWED,
    Check,
    frameType,
    lampName,
    lampOf,
    observationType,
    sequenceOf,
    specFor,
    whyNotJudged,
)

#: Every pair of limits, as the plan of record states them: check -> (kind, WARN, FAIL).
PAIRS = {
    "skyOffset": (FIXED, 0.5, 1.0),
    "skyLineResidual": (FIXED, 0.5, 1.0),
    "skyLineResidualBlue": (FIXED, 1.5, 2.5),
    "calibOffset": (FIXED, 1.0, 2.0),
    "traceCentreOffset": (FIXED, 1.0, 2.0),
    "slitOffsetInvariance": (FIXED, 0.02, 0.10),
    "twilightLevel": (FIXED, 2.0, 3.0),
    "twilightScatter": (FIXED, 1.5, 2.5),
    "twilightPatch": (FIXED, 2.5, 4.0),
    "width": (RELATIVE, 3.0, 5.0),
    "scatterX": (RELATIVE, 20.0, 50.0),
    "scatterY": (RELATIVE, 50.0, 150.0),
    "linesPerFiberArc": (RELATIVE, 25.0, 50.0),
    "linesPerFiberQuartz": (RELATIVE, 2.5, 5.0),
}


def testFourteenPairs():
    assert {name: (check.kind, check.warn, check.fail) for name, check in CHECKS.items()} == PAIRS
    assert len(CHECKS) == 14


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


#: The sequence types with SpS exposures in Runs 25 to 31 (the opdb's ``iic_sequence``), and the frame type of each.
SEEN = {
    "scienceObject": SCIENCE,
    "scienceObject_windowed": SCIENCE,
    "scienceArc": ARC,
    "scienceTrace": QUARTZ,
    "scienceTrace_windowed": WINDOWED,
    "darks": DARK,
    "masterDarks": DARK,
    "biases": BIAS,
    "masterBiases": BIAS,
    "fiberProfiles": FIBER_PROFILE,
    "dotRoach": DOT_ROACH,
    "dotRoachInit": DOT_ROACH,
    "dotRoachNew": DOT_ROACH,
    "dotRoach_continued": DOT_ROACH,
    "dotScan": DOT_ROACH,
    "detThroughFocus": THROUGH_FOCUS,
    "fpaThroughFocus": THROUGH_FOCUS,
    "slitThroughFocus": THROUGH_FOCUS,
    "arcs": ENGINEERING,
    "arcs_binned": ENGINEERING,
    "flats": ENGINEERING,
    "flats_binned": ENGINEERING,
    "domeFlat": DOME_FLAT,
}


@pytest.mark.parametrize("seqType, expected", sorted(SEEN.items()))
def testEverySequenceTypeSeenHasAFrameType(seqType, expected):
    assert frameType(seqType, "") == expected
    # A frame type is either judged or says why it is not, never both and never neither.
    assert (expected in FRAME_TYPES) != (expected in NOT_JUDGED)


def testFrameTypesAreJudgedOrSayWhyNot():
    kinds = set(SEQUENCE_TYPES.values())
    assert kinds <= set(FRAME_TYPES) | set(NOT_JUDGED)
    assert not set(FRAME_TYPES) & set(NOT_JUDGED)
    assert all(reason.endswith(".") for reason in NOT_JUDGED.values())
    # No check names a frame type that is not judged.
    assert not {frame for check in CHECKS.values() for frame in check.frames} & set(NOT_JUDGED)
    assert {DITHERED, THROUGH_FOCUS, FIBER_PROFILE, DOT_ROACH} <= set(NOT_JUDGED)


@pytest.mark.parametrize(
    "seqType, seqName, illumination, expected",
    [
        ("scienceObject", "field 3", "full", SCIENCE),
        ("scienceObject_windowed", "field 3", None, SCIENCE),
        ("scienceObject", "Sky flat", "full", ALLSKY),
        ("scienceObject", "Twilight sky", "full", TWILIGHT),
        ("scienceArc", "Arc: Neon", "full", ARC),
        ("scienceArc", "ImageQuality (Kr)", "iis", IIS_ARC),
        ("scienceTrace", "Trace", "full", QUARTZ),
        ("scienceTrace", "ImageQuality (Halogen)", "iis", IIS_QUARTZ),
        (" scienceTrace ", None, None, QUARTZ),  # padded, as a FITS card can be
        # A dark's pfsConfig can have the engineering fibers GOOD: it is a dark all the same.
        ("darks", "Morning Dark", "iis", DARK),
        ("fiberProfiles", "allHidden", "full", FIBER_PROFILE),
        ("arcs_binned", "cplTest_ne_bin1", "iis", ENGINEERING),
        # Science with the engineering fibers lit is nothing the checks were made for.
        ("scienceObject", "field 3", "iis", None),
        # Not in the table: a new or unclassified type shows up as untyped.
        ("scienceObject_iis", "field 3", "full", None),
        ("rampObject", "", "full", None),
        ("scienceArc_binned", "", "full", None),
        ("", "Trace", "full", None),
        (None, None, None, None),
        (float("nan"), "Trace", "iis", None),
    ],
)
def testFrameType(seqType, seqName, illumination, expected):
    assert frameType(seqType, seqName, illumination) == expected


def testObsTypeIsNotReadWhenASequenceTypeIsStored():
    assert frameType("fiberProfiles", "x", "full", obsType="trace") == FIBER_PROFILE
    assert frameType("rampObject", "x", "full", obsType="arc") is None
    # Negative control: without the sequence type the same obsType does type the row.
    assert frameType(None, "x", "full", obsType="arc") == ARC


def testFrameTypeIsNotGuessedFromTheName():
    # Negative control: a quartz is never read as an arc, whatever its sequence name says, and a sequence
    # named like a recipe arc is not one unless its type says so.
    assert frameType("scienceTrace", "Arc: Neon") == QUARTZ != ARC
    assert frameType("fiberProfiles", "Trace") != QUARTZ
    assert frameType("ditheredArcs", "Arc: Neon") == DITHERED != ARC
    assert frameType("defocusedArcs", "Arc: Neon") == THROUGH_FOCUS != ARC


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
        ("dark", None, DARK),
        ("bias", "iis", BIAS),
        ("unknown", "full", None),
        (None, None, None),
        (float("nan"), "iis", None),
        ("science", "iis", None),
    ],
)
def testFrameTypeOfARowStoredWithoutItsSequenceType(obsType, illumination, expected):
    """Rows from before ``seqType`` was stored are typed by the tasks' ``obsType``."""
    for seqType in (None, "", float("nan")):
        assert frameType(seqType, "anything", illumination, obsType) == expected


def testFrameTypeKeepsTheIlluminationsApart():
    # Negative control: an engineering-fiber arc is not an arc, so no full-slit check or normal value reaches it.
    assert frameType("scienceArc", "", "iis") != frameType("scienceArc", "", "full")
    assert frameType("scienceArc", "", "iis") not in CHECKS["scatterX"].frames
    assert frameType(obsType="arc", illumination="iis") != frameType(obsType="arc", illumination="full")


def testAScienceSequenceWithoutItsNameIsSplitByObsType():
    """The line-residual tables have no sequence name; their obsType was made from it."""
    assert frameType("scienceObject", None, None, "twilight") == TWILIGHT
    assert frameType("scienceObject", "", None, "allsky") == ALLSKY
    assert frameType("scienceObject", None, None, "science") == SCIENCE
    # Negative control: with the name at hand the name decides, and obsType splits no other sequence.
    assert frameType("scienceObject", "field 3", None, "twilight") == SCIENCE
    assert frameType("scienceArc", None, None, "twilight") == ARC


@pytest.mark.parametrize(
    "seqType, seqName, expected",
    [
        ("scienceTrace", "Trace", "trace"),
        ("scienceArc", "Arc: Neon", "arc"),
        ("scienceObject", "Twilight flat", "twilight"),
        ("scienceObject_windowed", "sky test", "allsky"),
        ("scienceObject", "field", "science"),
        ("darks", "Morning Dark", "dark"),
        ("masterDarks", None, "dark"),
        ("scienceDark", None, "dark"),
        ("biases", None, "bias"),
        ("fiberProfiles", "allHidden", "unknown"),
        ("ditheredFlats", None, "unknown"),
        ("", "Trace", "unknown"),
        (None, None, "unknown"),
    ],
)
def testObservationType(seqType, seqName, expected):
    assert observationType(seqType, seqName) == expected


class Header(dict):
    """A header: ``get`` is all `sequenceOf` asks of it."""


@pytest.mark.parametrize(
    "headers, expected",
    [
        (({"W_SEQTYP": "scienceTrace", "W_SEQNAM": "Trace"}, {}), ("scienceTrace", "Trace")),
        (({}, {"W_SEQTYP": "scienceArc", "W_SEQNAM": "Arc: Neon"}), ("scienceArc", "Arc: Neon")),
        ((None, {"W_SEQTYP": " darks ", "W_SEQNAM": None}), ("darks", "")),
        (
            ({"W_SEQNAM": "Trace"}, {"W_SEQTYP": "scienceArc"}),
            ("scienceArc", ""),
        ),  # the name of the same header
        (({}, None), ("", "")),
        ((None, None), ("", "")),
        ((), ("", "")),
    ],
)
def testSequenceOf(headers, expected):
    assert sequenceOf(*(None if header is None else Header(header) for header in headers)) == expected


def testSequenceOfReadsTheFirstHeaderThatSays():
    raw, config = Header({"W_SEQTYP": "fiberProfiles"}), Header({"W_SEQTYP": "scienceTrace"})
    assert sequenceOf(raw, config)[0] == "fiberProfiles"
    # Negative control: the order is the order of trust.
    assert sequenceOf(config, raw)[0] == "scienceTrace" != sequenceOf(raw, config)[0]


def testWhyNotJudged():
    assert whyNotJudged(DARK) == "dark frames are not judged: No lamp or sky is exposed."
    assert whyNotJudged(None) == "no frame type"
    assert whyNotJudged(None, "rampObject") == "no frame type for sequence type rampObject"
    assert whyNotJudged(float("nan"), None) == "no frame type"
    # A judged frame type has no such reason: what its checks lacked is said elsewhere.
    assert all(whyNotJudged(kind, "scienceArc") == "" for kind in FRAME_TYPES)


@pytest.mark.parametrize(
    "seqName, lamps, lamp",
    [
        ("Arc: HgCd", None, "HgCd"),
        ("Arc: Neon", None, "Ne"),
        ("Arc: Krypton", "", "Kr"),
        ("ImageQuality (Ne)", None, "Ne"),
        ("DetectorMap test (Helium)", None, "He"),
        ("ImageQuality (Halogen)", None, "Quartz"),
        ("Image Quality test (SM1)", None, "SM1"),  # no known lamp: the name's part as it is
        ("Trace", None, ""),
        ("", None, ""),
        (None, None, ""),
        (float("nan"), float("nan"), ""),
        # The header's lamps, when stored, whatever the name says.
        ("ImageQuality", "Kr_eng", "Kr"),
        ("ImageQuality (Ne)", "Kr_eng", "Kr"),
        ("Arc: Neon", "Ne", "Ne"),
        ("Bootstrap", "HgAr_eng", "HgAr"),
        ("x", "Ne+HgAr", "HgAr+Ne"),
        ("x", " Quartz_eng ", "Quartz"),
        ("x", "Kr_eng + Ne_eng", "Kr+Ne"),
        # Several lamps in a name are one key with the header's, whatever their order and spelling.
        ("Arc: Neon+Krypton", None, "Kr+Ne"),
        ("Arc: Ne+HgAr", "", "HgAr+Ne"),
    ],
)
def testLampOf(seqName, lamps, lamp):
    assert lampOf(seqName, lamps) == lamp


@pytest.mark.parametrize(
    "spelled, name",
    [
        ("Neon", "Ne"),
        ("Ne_eng", "Ne"),
        (" argon ", "Ar"),
        ("Halogen", "Quartz"),
        ("HgAr+Neon", "HgAr+Ne"),
        ("SM1", "SM1"),
        ("", ""),
        (None, ""),
        (float("nan"), ""),
    ],
)
def testLampName(spelled, name):
    assert lampName(spelled) == name
    assert lampName(lampName(spelled)) == name


def testLampOfGivesOneNamePerLamp():
    spellings = [lampOf("Arc: Neon"), lampOf("ImageQuality (Ne)"), lampOf("x", "Ne"), lampOf("x", "Ne_eng")]
    assert set(spellings) == {"Ne"}
    # Negative control: two lamps stay two.
    assert lampOf("x", "Kr_eng") != lampOf("x", "Ne_eng")


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
    # Every frame type that is not judged is an exclusion, with its reason.
    reasons = dict(EXCLUSIONS)
    for kind, reason in NOT_JUDGED.items():
        assert reasons[f"Every number measured on {kind} frames"] == reason


def testComparisonModeAgreesOnWhatASequenceTypeIs():
    """Comparison mode plans by sequence type too: its lists and the registry's table say the same."""
    from pfs.drp.qa.comparison import classify

    assert {frameType(seqType) for seqType in classify.VALIDATED_TYPES} == {ARC, QUARTZ}
    assert {frameType(seqType) for seqType in classify.NOT_MEASURED_TYPES} == {DARK, BIAS}
    assert {frameType(seqType, "field") for seqType in classify.SKY_TYPES} == {SCIENCE}
