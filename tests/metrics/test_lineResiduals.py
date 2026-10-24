"""Tests for line residuals against the calibration and the visit's detectorMap (dmLineResiduals).

Drift against the calibration is PIPE2D-1921; the visit reference, PIPE2D-1956.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import judge
from pfs.drp.qa.metrics.lineResiduals import (
    BAD_STATUS,
    BOTH,
    CALIB,
    LINE_COLUMNS,
    QUARTZ,
    RESERVED,
    RESIDUAL_COLUMNS,
    SKY_SPECIES,
    STATUS_TYPES,
    TRACE,
    UNFITTED,
    USED,
    VISIT,
    addObsType,
    cameraSet,
    frameType,
    lineResiduals,
    lineTable,
    longResiduals,
    observationType,
    residualThresholds,
    residualVerdict,
    robustRms,
    speciesResiduals,
    wideResiduals,
)
from pfs.drp.qa.metrics.registry import specFor

DATA_ID = {"visit": 1, "arm": "b", "spectrograph": 2}
NAN = float("nan")
#: drp_stella's ReferenceLineStatus bits.
USED_BIT, RESERVED_BIT = 0x20, 0x40


class FakeCalib:
    """Straight vertical traces 6 px apart; y is 10 px per nm above 500 nm."""

    def __init__(self, numFibers: int = 20):
        self.fiberId = np.arange(1, numFibers + 1, dtype=np.int32)

    def getXCenter(self, fiberId, y):
        fiberId = np.asarray(fiberId)
        if not np.isin(fiberId, self.fiberId).all():
            raise IndexError("fiberId not in the detectorMap")  # as a SplinedDetectorMap does
        return 6.0 * fiberId + 0.0 * np.asarray(y, dtype=float)

    def findPoint(self, fiberId, wavelength):
        fiberId = np.asarray(fiberId, dtype=float)
        return np.column_stack([6.0 * fiberId, 10.0 * (np.asarray(wavelength, dtype=float) - 500.0)])


def makeLines(
    calib: FakeCalib,
    traceRows: int = 20,
    lampLines: int = 5,
    dx: float = 0.0,
    dy: float = 0.0,
    species: str = "NeI",
    fibers=None,
    noise: float = 0.0,
    seed: int = 1,
) -> pd.DataFrame:
    """Return ``lines``: trace centroids shifted by ``dx`` and lamp lines by ``dy`` from ``calib``."""
    rng = np.random.default_rng(seed)
    fibers = calib.fiberId if fibers is None else np.asarray(fibers)
    rows = []
    for fiberId in fibers:
        for y in np.linspace(10.0, 3900.0, traceRows):
            x = float(calib.getXCenter([fiberId], [y])[0]) + dx + rng.normal(0.0, noise)
            rows.append((fiberId, NAN, x, y, TRACE))
        for wavelength in np.linspace(520.0, 880.0, lampLines):
            x, y = calib.findPoint([fiberId], [wavelength])[0]
            rows.append((fiberId, wavelength, x, y + dy + rng.normal(0.0, noise), species))
    frame = pd.DataFrame(rows, columns=["fiberId", "wavelength", "x", "y", "description"])
    frame["flag"] = False
    frame["status"] = USED_BIT  # not a bad bit
    return frame


class ShiftedMap(FakeCalib):
    """`FakeCalib` moved by ``(dx, dy)``: a visit's detectorMap adjusted away from the calibration."""

    def __init__(self, dx: float = 0.0, dy: float = 0.0, numFibers: int = 20):
        super().__init__(numFibers)
        self.dx, self.dy = dx, dy

    def getXCenter(self, fiberId, y):
        return super().getXCenter(fiberId, y) + self.dx

    def findPoint(self, fiberId, wavelength):
        return super().findPoint(fiberId, wavelength) + np.array([self.dx, self.dy])


def residuals(lines, calib, visit=None, **kwargs):
    """Return `lineResiduals` against ``calib`` and ``visit`` (``calib`` when not given)."""
    return lineResiduals(lines, {CALIB: calib, VISIT: calib if visit is None else visit}, **kwargs)


def speciesOf(lines, calib, minLines=10, litFibers=None, visit=None):
    """Return `speciesResiduals` of ``lines``."""
    return speciesResiduals(residuals(lines, calib, visit), minLines, litFibers)


def both(species: pd.DataFrame, reference: str = CALIB) -> pd.DataFrame:
    """Return the ``BOTH`` (or ``UNFITTED``) rows of `speciesResiduals` against ``reference``, by species."""
    rows = species[(species["reference"] == reference) & species["statusType"].isin((BOTH, UNFITTED))]
    return rows.set_index("description")


def reserved(species: pd.DataFrame, reference: str = VISIT) -> pd.DataFrame:
    """Return the ``RESERVED`` rows of `speciesResiduals` against ``reference``, by species."""
    rows = species[(species["reference"] == reference) & (species["statusType"] == RESERVED)]
    return rows.set_index("description")


def measure(
    lines, calib, minLines=10, litFibers=None, thresholds=None, camera=None, visit=None, obsType=None
):
    """Return ``(both(species), judged, summary, species)`` for ``lines``."""
    full = speciesOf(lines, calib, minLines, litFibers, visit)
    if thresholds is None:
        thresholds = residualThresholds(0.1, 0.15, 0.1, 0.15)
    judged, summary = residualVerdict(full, DATA_ID, thresholds, minLines, camera, obsType)
    return both(full), judged, summary, full


def testZeroOffsetPasses():
    calib = FakeCalib()
    species, _, summary, _ = measure(makeLines(calib), calib)
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.0, abs=1e-12)
    assert species.loc["NeI", "medianDeltaY"] == pytest.approx(0.0, abs=1e-12)
    assert summary["qaStatus"] == "PASS"
    assert summary["recommendedAction"] == "NOMINAL"
    assert summary["nSpeciesMeasured"] == 2


def testInjectedShiftIsRecoveredAndFails():
    calib = FakeCalib()
    lines = makeLines(calib, dx=0.2, noise=0.02)
    species, _, summary, _ = measure(lines, calib)
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.2, abs=0.01)
    assert species.loc[TRACE, "robustRmsDeltaX"] == pytest.approx(0.02, rel=0.25)
    assert summary["qaStatus"] == "FAIL"
    assert summary["recommendedAction"] == "RECALIBRATE"
    assert summary["qaDecidedBy"] == f"calib {TRACE} medianDeltaX"
    assert "fail threshold 0.15px" in summary["qaReason"]
    # Negative control: the offset is measured minus calibration, not calibration minus measured
    # (imageQualityQa's dxCenter): a +0.2 px shift does not read as -0.2 px.
    assert not np.isclose(species.loc[TRACE, "medianDeltaX"], -0.2, atol=0.01)


def testInjectedYShiftIsRecovered():
    calib = FakeCalib()
    species, _, summary, _ = measure(makeLines(calib, dy=-0.12), calib)
    assert species.loc["NeI", "medianDeltaY"] == pytest.approx(-0.12)
    assert summary["qaStatus"] == "WARN"
    assert summary["recommendedAction"] == "APPLY_SHIFT"
    assert summary["qaDecidedBy"] == "calib NeI medianDeltaY"
    # Negative control: the sign is kept; only the gate takes the absolute value.
    assert species.loc["NeI", "medianDeltaY"] < 0


def testTooFewLinesIsInsufficientData():
    calib = FakeCalib(numFibers=1)
    species, judged, summary, _ = measure(makeLines(calib, traceRows=5, lampLines=5, dx=5.0), calib)
    assert species["nLines"].tolist() == [5, 5]
    assert species[["medianDeltaX", "medianDeltaY"]].isna().all().all()
    assert summary["qaStatus"] == "UNKNOWN"
    assert summary["recommendedAction"] == "INSUFFICIENT_DATA"
    assert "fewer than 10 good lines" in summary["qaReason"]
    assert (judged["status"] == "").all()


def testNoLinesIsInsufficientData():
    calib = FakeCalib()
    lines = makeLines(calib).assign(flag=True)
    _, _, summary, _ = measure(lines, calib)
    assert summary["qaStatus"] == "UNKNOWN"
    assert summary["recommendedAction"] == "INSUFFICIENT_DATA"
    assert summary["qaReason"] == "no good lines"
    assert summary["fitted"]


def testTraceOnlyFillsXAndDecides():
    calib = FakeCalib()
    lines = makeLines(calib, lampLines=0, dx=0.05)
    species, judged, summary, full = measure(lines, calib)
    assert list(species.index) == [TRACE]
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.05)
    assert np.isnan(species.loc[TRACE, "medianDeltaY"])
    assert np.isnan(species.loc[TRACE, "robustRmsDeltaY"])
    assert summary["qaStatus"] == "PASS"
    long = longResiduals(full, judged, DATA_ID)
    assert set(long["metric"]) == {"nLines", "linesPerFiber", "medianDeltaX", "robustRmsDeltaX"}


def testOneSpeciesShortIsStillJudgedOnTheOther():
    calib = FakeCalib()
    lines = makeLines(calib, lampLines=0, dx=0.2)
    lamp = makeLines(FakeCalib(numFibers=1), traceRows=0, lampLines=3, species="HgI")
    species, _, summary, _ = measure(pd.concat([lines, lamp]), calib)
    assert species.loc["HgI", "nLines"] == 3
    assert np.isnan(species.loc["HgI", "medianDeltaY"])
    assert summary["nSpeciesMeasured"] == 1
    assert summary["qaStatus"] == "FAIL"


def testSpeciesAreNotBlended():
    calib = FakeCalib()
    neon = makeLines(calib, traceRows=0, dy=0.12, species="NeI")
    mercury = makeLines(calib, traceRows=0, dy=0.0, species="HgI")
    species, _, summary, _ = measure(pd.concat([neon, mercury]), calib)
    assert species.loc["NeI", "medianDeltaY"] == pytest.approx(0.12)
    assert species.loc["HgI", "medianDeltaY"] == pytest.approx(0.0)
    assert summary["qaDecidedBy"] == "calib NeI medianDeltaY"


def testBadLinesAreExcluded():
    calib = FakeCalib(numFibers=2)
    lines = makeLines(calib, lampLines=0)
    bad = lines.iloc[:10].assign(x=lines["x"].iloc[:10] + 50.0)
    flagged = bad.assign(flag=True)
    blended = bad.assign(status=0x2 | USED_BIT)  # marked by the fit, but a blend
    offFiber = bad.assign(fiberId=99)  # not in the calibration: getXCenter would raise
    notFinite = bad.assign(x=np.nan)
    offsets = residuals(pd.concat([lines, flagged, blended, offFiber, notFinite]), calib)
    assert len(offsets) == len(lines)
    assert offsets["dxCalib"].abs().max() == pytest.approx(0.0)
    # Negative control: without the status mask, the blended lines get in.
    unmasked = residuals(pd.concat([lines, blended]), calib, badStatus=0)
    assert len(unmasked) > len(lines)
    assert BAD_STATUS & 0x2


def testLinesPerFiberIsPerLitFiber():
    calib = FakeCalib()
    full = both(speciesOf(makeLines(calib), calib, 10, set(calib.fiberId.tolist())))
    group = calib.fiberId[::4]
    shaka = both(speciesOf(makeLines(calib, fibers=group), calib, 10, set(group.tolist())))
    assert shaka["linesPerFiber"].tolist() == full["linesPerFiber"].tolist() == [5.0, 20.0]
    assert shaka["nLines"].tolist() == [len(group) * 5, len(group) * 20]
    # Negative control: counted per fiber of the detector, one group reads as sparse.
    sparse = both(speciesOf(makeLines(calib, fibers=group), calib, 10, set(calib.fiberId.tolist())))
    assert (sparse["linesPerFiber"] < full["linesPerFiber"]).all()
    # Without a pfsConfig, or with no fiber lit, there is no per-fiber count.
    assert speciesOf(makeLines(calib), calib, 10, None)["linesPerFiber"].isna().all()
    assert speciesOf(makeLines(calib), calib, 10, set())["linesPerFiber"].isna().all()


def testThresholdsByPopulation():
    calib = FakeCalib()
    lines = makeLines(calib, dx=0.5)
    file = pd.DataFrame(
        {
            "metric": ["medianDeltaX"],
            "arm": ["b"],
            "cameraSet": ["bmn"],
            "warn": [0.6],
            "fail": [0.8],
            "higherIsWorse": [True],
            "absolute": [True],
            "provenance": ["test"],
        }
    )
    tables = [file, residualThresholds(0.1, 0.15, NAN, NAN)]
    _, judged, withM, _ = measure(lines, calib, thresholds=tables, camera="bmn")
    assert withM["qaStatus"] == "PASS"
    assert judged.loc[judged["metric"] == "medianDeltaX", "population"].iloc[0] == "arm=b cameraSet=bmn"
    # Negative control: read with r, the b camera falls through to the config's thresholds.
    _, _, withR, _ = measure(lines, calib, thresholds=tables, camera="brn")
    assert withR["qaStatus"] == "FAIL"


def testMeasuredButUnthresholdedIsUnjudged():
    calib = FakeCalib()
    lines = makeLines(calib, traceRows=0, dy=5.0)
    _, _, summary, _ = measure(lines, calib, thresholds=residualThresholds(1.0, 2.0, NAN, NAN))
    assert summary["qaStatus"] == "UNKNOWN"
    assert summary["recommendedAction"] == "UNJUDGED"
    assert summary["nSpeciesMeasured"] == 1


def testLongFormat():
    calib = FakeCalib()
    _, judged, _, full = measure(makeLines(calib, dx=0.2), calib, camera="brn")
    long = longResiduals(full, judged, DATA_ID, "brn")
    assert list(long.columns) == list(RESIDUAL_COLUMNS)
    assert long.dtypes.astype(str).tolist() == list(RESIDUAL_COLUMNS.values())
    trace = long[(long["description"] == TRACE) & (long["statusType"] == BOTH)]
    trace = trace[trace["reference"] == CALIB].set_index("metric")
    assert trace.loc["medianDeltaX", "status"] == "FAIL"
    assert trace.loc["nLines", "status"] == ""
    assert (long["cameraSet"] == "brn").all()
    lamp = long[long["description"] == "NeI"]
    assert set(lamp["metric"]) == {
        "nLines",
        "linesPerFiber",
        "medianDeltaX",
        "robustRmsDeltaX",
        "medianDeltaY",
        "robustRmsDeltaY",
    }
    assert list(long["reference"].unique()) == [CALIB, VISIT]
    assert (long.loc[long["statusType"] != BOTH, "status"] == "").all()
    assert list(long["statusType"].unique()) == list(STATUS_TYPES)
    empty = longResiduals(full.iloc[:0], judged.iloc[:0], DATA_ID)
    assert list(empty.columns) == list(RESIDUAL_COLUMNS) and empty.empty


def testRobustRms():
    values = np.random.default_rng(2).normal(0.0, 0.3, 20000)
    assert robustRms(values) == pytest.approx(0.3, rel=0.03)
    assert robustRms(np.append(values, [1e6, np.nan])) == pytest.approx(0.3, rel=0.03)
    assert np.isnan(robustRms([]))


@pytest.mark.parametrize(
    "config, arm, expected",
    [
        (None, "b", None),
        (SimpleNamespace(camMask=0, arms="brn"), "b", "brn"),
        (SimpleNamespace(camMask=0, arms="bmn"), "b", "bmn"),
        (SimpleNamespace(camMask=0, arms="br"), "b", "br"),
        (SimpleNamespace(camMask=0, arms="bm"), "m", "bm"),
        # b1 m1 b2 m2: bits 0, 3, 4, 7 of PfsConfig._allCams (b1 r1 n1 m1 b2 ...).
        (SimpleNamespace(camMask=(1 << 0) | (1 << 3) | (1 << 4) | (1 << 7), arms="brn"), "b", "bm"),
        (SimpleNamespace(camMask=0, arms=""), "m", "m"),
    ],
)
def testCameraSet(config, arm, expected):
    assert cameraSet(config, arm) == expected


class Header(dict):
    """A ``PfsConfig.header``: a mapping with ``get``."""


@pytest.mark.parametrize(
    "header, expected",
    [
        ({"W_SEQTYP": "scienceTrace", "W_SEQNAM": "Trace"}, QUARTZ),
        ({"W_SEQTYP": "scienceArc", "W_SEQNAM": "Arc: Neon"}, "arc"),
        ({"W_SEQTYP": "scienceObject", "W_SEQNAM": "Twilight flat"}, "twilight"),
        ({"W_SEQTYP": "scienceObject_windowed", "W_SEQNAM": "sky test"}, "allsky"),
        ({"W_SEQTYP": "scienceObject", "W_SEQNAM": "field"}, "science"),
        ({"W_SEQTYP": "scienceDark"}, "dark"),
        ({"W_SEQTYP": " scienceTrace ", "W_SEQNAM": None}, QUARTZ),  # padded, as a FITS card can be
        ({"W_SEQNAM": "Trace"}, "unknown"),
        ({"W_SEQTYP": "ditheredFlats"}, "unknown"),
        (None, None),
    ],
)
def testObservationType(header, expected):
    """The frame type is imageQualityQa's ``obsType``, from the pfsConfig header; None without one."""
    assert observationType(None if header is None else Header(header)) == expected
    # Negative control: a quartz is never read as an arc, whatever its sequence name says.
    assert observationType(Header({"W_SEQTYP": "scienceTrace", "W_SEQNAM": "Arc: Neon"})) != "arc"


@pytest.mark.parametrize(
    "raw, config, expected",
    [
        ({"W_SEQTYP": "scienceTrace"}, {}, QUARTZ),  # a Run25/26 pfsConfig lacks the card
        ({}, {"W_SEQTYP": "scienceArc"}, "arc"),  # the pfsConfig stands in for a raw without it
        (None, {"W_SEQTYP": "scienceArc"}, "arc"),
        ({"W_SEQTYP": "ditheredFlats"}, {"W_SEQTYP": "scienceArc"}, "arc"),
        ({}, {}, "unknown"),
        (None, {}, "unknown"),
        ({}, None, "unknown"),
        (None, None, None),
    ],
)
def testFrameTypeReadsTheRawFirst(raw, config, expected):
    """The raw header's type, the pfsConfig's when the raw's is missing or unknown."""
    assert frameType(*(None if header is None else Header(header) for header in (raw, config))) == expected
    # Negative control: when both say, the raw wins over the pfsConfig.
    both = frameType(Header({"W_SEQTYP": "scienceArc"}), Header({"W_SEQTYP": "scienceTrace"}))
    assert both == "arc" and both != QUARTZ


def testTraceScatterIsJudgedPerFrame():
    """A trace-scatter entry naming ``obsType`` applies to that frame type and no other."""
    calib = FakeCalib()
    lines, _ = withReserved(makeLines(calib, traceRows=40), noise=0.2)
    entry = residualThresholds(1.0, 2.0, 1.0, 2.0, 0.05, 0.1).iloc[[2]].assign(obsType=QUARTZ)
    thresholds = [entry, residualThresholds(1.0, 2.0, 1.0, 2.0)]
    _, judgedQuartz, quartz, full = measure(lines, calib, thresholds=thresholds, obsType=QUARTZ)
    _, judgedArc, arc, _ = measure(lines, calib, thresholds=thresholds, obsType="arc")
    _, judgedNone, none, _ = measure(lines, calib, thresholds=thresholds)
    scatter = judgedQuartz[
        (judgedQuartz["metric"] == "robustRmsDeltaX")
        & (judgedQuartz["description"] == TRACE)
        & (judgedQuartz["reference"] == VISIT)
    ]
    assert (scatter["status"] == "FAIL").all() and quartz["recommendedAction"] == "RECALIBRATE"
    for judged, summary in ((judgedArc, arc), (judgedNone, none)):
        rows = judged[
            (judged["metric"] == "robustRmsDeltaX")
            & (judged["description"] == TRACE)
            & (judged["reference"] == VISIT)
        ]
        assert (rows["status"] == "").all(), (
            "an arc's traces, or a frame of unknown type, are not judged by it"
        )
        assert summary["recommendedAction"] != "RECALIBRATE"
    long = longResiduals(full, judgedQuartz, DATA_ID, "brn", QUARTZ)
    assert (long["obsType"] == QUARTZ).all() and long["obsType"].dtype == "string"
    assert longResiduals(full, judgedNone, DATA_ID)["obsType"].isna().all()
    # Negative control: the same entry without obsType judges every frame.
    _, judgedAny, _, _ = measure(
        lines, calib, thresholds=[entry.drop(columns="obsType"), thresholds[1]], obsType="arc"
    )
    rows = judgedAny[
        (judgedAny["metric"] == "robustRmsDeltaX")
        & (judgedAny["description"] == TRACE)
        & (judgedAny["reference"] == VISIT)
    ]
    assert (rows["status"] == "FAIL").all()


def testRegistry():
    for name in ("medianDeltaX", "medianDeltaY"):
        assert specFor(name).absolute
        assert "cameraSet" in specFor(name).groupBy
    assert not specFor("robustRmsDeltaX").absolute
    # Thresholds are derived per reference and line set: the two references are different populations.
    for name in ("medianDeltaX", "robustRmsDeltaX", "medianDeltaY", "robustRmsDeltaY"):
        assert {"reference", "statusType"} <= set(specFor(name).groupBy)


def testUsedAndReservedAreKeptApart():
    calib = FakeCalib()
    lines = makeLines(calib, lampLines=0)
    reserved = np.arange(len(lines)) % 5 == 0  # a fit holds back about a fifth
    lines.loc[reserved, "status"] = RESERVED_BIT
    lines.loc[reserved, "x"] += 0.3
    _, judged, summary, full = measure(lines, calib)
    trace = full[(full["description"] == TRACE) & (full["reference"] == CALIB)].set_index("statusType")
    assert list(trace.index) == list(STATUS_TYPES)
    assert trace.loc[USED, "nLines"] == (~reserved).sum()
    assert trace.loc[RESERVED, "nLines"] == reserved.sum()
    assert trace.loc[BOTH, "nLines"] == len(lines)
    assert trace.loc[USED, "medianDeltaX"] == pytest.approx(0.0)
    assert trace.loc[RESERVED, "medianDeltaX"] == pytest.approx(0.3)
    # Only the union is judged: one verdict per species.
    assert set(judged.loc[judged["reference"] == CALIB, "statusType"]) == {BOTH}
    assert summary["qaStatus"] == "PASS"
    # Negative control: judged on the RESERVED lines alone, it would fail.
    assert abs(trace.loc[RESERVED, "medianDeltaX"]) > 0.15


def testLinesTheFitDidNotMarkAreLeftOut():
    calib = FakeCalib()
    lines = makeLines(calib)
    rejected = makeLines(calib, dx=5.0, dy=5.0).assign(status=0)  # good, but neither used nor reserved
    species, _, summary, _ = measure(pd.concat([lines, rejected]), calib)
    assert species.loc[TRACE, "nLines"] == (lines["description"] == TRACE).sum()
    assert summary["qaStatus"] == "PASS"
    # Negative control: the same lines, marked USED, are measured and fail.
    _, _, marked, _ = measure(pd.concat([lines, rejected.assign(status=USED_BIT)]), calib)
    assert marked["qaStatus"] == "FAIL"


def testUnfittedIsMeasuredOnEveryGoodLine():
    calib = FakeCalib()
    lines = makeLines(calib, dx=0.2)
    species, judged, summary, full = measure(lines.assign(status=0), calib)
    assert set(full["statusType"]) == {UNFITTED}
    assert species.loc[TRACE, "nLines"] == (lines["description"] == TRACE).sum()
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.2)
    assert set(judged["statusType"]) == {UNFITTED}
    assert set(judged["reference"]) == {CALIB}
    assert summary["qaStatus"] == "FAIL"
    assert not summary["fitted"]
    long = longResiduals(full, judged, DATA_ID)
    traceX = long[(long["metric"] == "medianDeltaX") & (long["description"] == TRACE)]
    assert traceX.set_index("reference")["status"].to_dict() == {CALIB: "FAIL", VISIT: ""}
    # Negative control: the same lines marked by a fit are measured as BOTH, USED and RESERVED.
    _, _, marked, markedFull = measure(lines, calib)
    assert marked["fitted"]
    assert UNFITTED not in set(markedFull["statusType"])


class EdgeCalib(FakeCalib):
    """A calibration that predicts nothing off the detector: NaN beyond 3000 px, as ``findPoint`` does."""

    def getXCenter(self, fiberId, y):
        return np.where(np.asarray(y, dtype=float) > 3000.0, np.nan, super().getXCenter(fiberId, y))

    def findPoint(self, fiberId, wavelength):
        points = super().findPoint(fiberId, wavelength)
        points[points[:, 1] > 3000.0] = np.nan
        return points


def testNaNPredictionsAreLeftOut():
    calib = EdgeCalib()
    lines = makeLines(FakeCalib(), dx=0.05, dy=0.05)
    offsets = lineResiduals(lines, {CALIB: calib})
    assert np.isfinite(offsets.loc[offsets["description"] == TRACE, "dxCalib"]).all()
    assert np.isfinite(offsets.loc[offsets["description"] == "NeI", "dyCalib"]).all()
    assert offsets[["dxVisit", "dyVisit"]].isna().all().all()
    species, _, summary, _ = measure(lines, calib)
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.05)
    assert species.loc["NeI", "medianDeltaY"] == pytest.approx(0.05)
    assert species.loc[TRACE, "nLines"] < (lines["description"] == TRACE).sum()
    assert summary["nSpeciesMeasured"] == 2
    # Negative control: the calibration does predict NaN for some of these lines.
    xCenter = calib.getXCenter(lines["fiberId"].to_numpy(), lines["y"].to_numpy())
    assert np.isnan(xCenter[(lines["description"] == TRACE).to_numpy()]).any()


@pytest.mark.parametrize(
    "dx, rms, expected",
    [
        (0.12, None, ("WARN", "APPLY_SHIFT")),  # a shift the adjustment absorbs
        (0.2, None, ("FAIL", "RECALIBRATE")),  # beyond what it can be trusted with
        (0.0, 0.05, ("WARN", "RECALIBRATE")),  # scatter: a shift cannot fix it
    ],
)
def testActionFollowsTheMetric(dx, rms, expected):
    calib = FakeCalib()
    tables = [residualThresholds(0.1, 0.15, NAN, NAN)]
    if rms is not None:
        tables.insert(
            0,
            pd.DataFrame(
                {
                    "metric": ["robustRmsDeltaX"],
                    "warn": [rms],
                    "fail": [1.0],
                    "higherIsWorse": [True],
                    "absolute": [False],
                    "provenance": ["test"],
                }
            ),
        )
    _, _, summary, _ = measure(makeLines(calib, dx=dx, noise=0.1), calib, thresholds=tables)
    assert (summary["qaStatus"], summary["recommendedAction"]) == expected


def testUnfittedCrossingIsRecalibrate():
    calib = FakeCalib()
    lines = makeLines(calib, dx=0.12)
    _, _, unfitted, _ = measure(lines.assign(status=0), calib)
    assert (unfitted["qaStatus"], unfitted["recommendedAction"]) == ("WARN", "RECALIBRATE")
    # Negative control: the same offset, adjusted by the visit's fit, only needs a shift.
    _, _, fitted, _ = measure(lines, calib)
    assert (fitted["qaStatus"], fitted["recommendedAction"]) == ("WARN", "APPLY_SHIFT")


def testValidationVerdicts():
    """The shipped config on the validation run (u/wtg/qa-thresholds/010): no known-good is flagged.

    The calibration's own input visits are known-good too, and sit at zero by construction. The
    sky-line species of the twilights (OH, O2, OI, NaI) are set aside: judging them is PIPE2D-1956.
    """
    from pathlib import Path

    from pfs.drp.qa.metrics.calibration import UNCONFIRMED, labelRows
    from pfs.drp.qa.metrics.validationVisits import loadValidationVisits

    rows = pd.read_parquet(Path(__file__).parent / "data" / "dmDriftMetrics-validation.parquet")
    # The 008 fixture holds dmDriftMonitor's rows (PIPE2D-1921), all against the calibration.
    if "reference" not in rows:
        rows = rows.assign(reference=CALIB)
    rows = rows[(rows["nLines"] >= 10) & (rows["reference"] == CALIB)].reset_index(drop=True)
    sky = rows["description"].isin(["OH", "O2", "OI", "NaI"])
    assert sky.any()  # the twilights are in the fixture; their species are not asserted on below
    config = residualThresholds(1.0, 2.0, 1.0, 2.0)
    judged = judge(rows, config).set_index(["row", "metric"])
    visitSet = loadValidationVisits()
    for metric in ("medianDeltaX", "medianDeltaY"):
        labelled = labelRows(rows, visitSet, metric)
        status = judged.xs(metric, level="metric")["status"]
        good = status.loc[labelled.index[(labelled["validation"] == "good") & ~sky[labelled.index]]]
        assert len(good[good != ""]) > 300 and (good[good != ""] == "PASS").all(), metric
        unconfirmed = status.loc[labelled.index[labelled["validation"] == UNCONFIRMED]]
        if metric == "medianDeltaY":
            # Run25 SM3 after its calibration block: most reach WARN, none FAIL.
            unconfirmed = unconfirmed[unconfirmed != ""]
            assert len(unconfirmed) > 50 and (unconfirmed == "WARN").mean() > 0.8
            assert not (unconfirmed == "FAIL").any()
    # Run27 against a calibration built later: every known-bad row reaches its verdict
    # (SM3 +1.15 px in x, WARN; SM2 -3.15 px in y, FAIL), and the visits after the change pass.
    for metric, expect, count in (("medianDeltaX", "WARN", 80), ("medianDeltaY", "FAIL", 30)):
        labelled = labelRows(rows, visitSet, metric)
        status = judged.xs(metric, level="metric")["status"]
        bad = status.loc[labelled.index[labelled["validation"] == f"bad:{expect}"]]
        bad = bad[bad != ""]
        assert len(bad) > count and (bad == expect).all(), metric
    later = rows.index[
        rows["visit"].isin([140477, 140640, 140649, 140651]) & rows["spectrograph"].isin([2, 3])
    ]
    laterX = judged.xs("medianDeltaX", level="metric")["status"].loc[later]
    assert (laterX[laterX != ""] == "PASS").all()
    # Negative control: the rebuild plan's 0.15 px FAIL would fail known-good Run25 traces.
    strict = judge(rows, residualThresholds(0.1, 0.15, NAN, NAN)).set_index(["row", "metric"])
    labelled = labelRows(rows, visitSet, "medianDeltaX")
    strictGood = strict.xs("medianDeltaX", level="metric")["status"].loc[
        labelled.index[labelled["validation"] == "good"]
    ]
    assert (strictGood == "FAIL").sum() > 50


def rmsThresholds(warn: float = 0.05, fail: float = 0.1) -> pd.DataFrame:
    """Return the config's thresholds with the RESERVED robust RMS judged in x and y."""
    return residualThresholds(1.0, 2.0, 1.0, 2.0, warn, fail, warn, fail)


def withReserved(lines: pd.DataFrame, noise: float = 0.0, seed: int = 3) -> tuple[pd.DataFrame, np.ndarray]:
    """Mark every fifth line RESERVED, scattering those by ``noise`` px in x and y."""
    lines = lines.copy()
    marked = np.arange(len(lines)) % 5 == 0
    rng = np.random.default_rng(seed)
    lines.loc[marked, "status"] = RESERVED_BIT
    lines.loc[marked, "x"] += rng.normal(0.0, noise, marked.sum())
    isLamp = (lines["description"] != TRACE).to_numpy()
    lines.loc[marked & isLamp, "y"] += rng.normal(0.0, noise, (marked & isLamp).sum())
    return lines, marked


def testEachReferenceIsMeasured():
    """An adjusted map 0.5 px from the calibration: the lines sit on it, 0.5 px from the calibration."""
    calib, visit = FakeCalib(), ShiftedMap(dx=0.5, dy=-0.3)
    lines = makeLines(calib, dx=0.5, dy=-0.3)
    full = speciesOf(lines, calib, visit=visit)
    againstCalib, againstVisit = both(full, CALIB), both(full, VISIT)
    assert againstCalib.loc[TRACE, "medianDeltaX"] == pytest.approx(0.5)
    assert againstCalib.loc["NeI", "medianDeltaY"] == pytest.approx(-0.3)
    assert againstVisit.loc[TRACE, "medianDeltaX"] == pytest.approx(0.0)
    assert againstVisit.loc["NeI", "medianDeltaY"] == pytest.approx(0.0)
    # Negative control: the references are not swapped.
    assert not np.isclose(againstVisit.loc[TRACE, "medianDeltaX"], 0.5)
    offsets = residuals(lines, calib, visit)
    assert offsets["dxCalib"].to_numpy() == pytest.approx(offsets["dxVisit"].to_numpy() + 0.5)


def testReservedScatterAgainstTheVisitIsJudged():
    calib = FakeCalib()
    lines, _ = withReserved(makeLines(calib), noise=0.2)
    _, judged, summary, full = measure(lines, calib, thresholds=rmsThresholds())
    rows = reserved(full)
    assert rows.loc[TRACE, "robustRmsDeltaX"] == pytest.approx(0.2, rel=0.3)
    assert rows.loc["NeI", "robustRmsDeltaY"] == pytest.approx(0.2, rel=0.3)
    assert summary["qaStatus"] == "FAIL"
    assert summary["recommendedAction"] == "RECALIBRATE"
    assert summary["qaDecidedBy"].startswith("visit ")
    assert "visit Trace: robustRmsDeltaX=" in summary["qaReason"]
    visitRows = judged[judged["reference"] == VISIT]
    assert set(visitRows["statusType"]) == {RESERVED}
    assert set(visitRows.loc[visitRows["status"] != "", "metric"]) == {"robustRmsDeltaX", "robustRmsDeltaY"}
    # Negative control: the same scatter on lines the fit used is not judged; it is the fit's residual.
    used = lines.assign(status=USED_BIT)
    _, _, onUsed, _ = measure(used, calib, thresholds=rmsThresholds())
    assert onUsed["qaStatus"] == "PASS"


def testReservedScatterIsNotJudgedByDefault():
    """The config's RMS thresholds are NaN until derived from the validation set."""
    calib = FakeCalib()
    lines, _ = withReserved(makeLines(calib), noise=0.5)
    _, judged, summary, _ = measure(lines, calib, thresholds=residualThresholds(1.0, 2.0, 1.0, 2.0))
    assert (judged.loc[judged["reference"] == VISIT, "status"] == "").all()
    assert summary["qaStatus"] == "PASS"
    # Negative control: given thresholds, the same lines fail.
    _, _, judgedSummary, _ = measure(lines, calib, thresholds=rmsThresholds())
    assert judgedSummary["qaStatus"] == "FAIL"


def testVisitMediansAreNotJudged():
    """Only the RESERVED robust RMS is judged against the visit, whatever a thresholds file names."""
    calib, visit = FakeCalib(), ShiftedMap(dx=-5.0)
    lines, _ = withReserved(makeLines(calib))
    file = pd.DataFrame(
        {
            "metric": ["medianDeltaX"],
            "warn": [0.1],
            "fail": [0.15],
            "higherIsWorse": [True],
            "absolute": [True],
            "provenance": ["test: names no reference"],
        }
    )
    _, judged, summary, full = measure(lines, calib, thresholds=[file], visit=visit)
    assert reserved(full).loc[TRACE, "medianDeltaX"] == pytest.approx(5.0)
    assert summary["qaStatus"] == "PASS"
    medians = judged[(judged["reference"] == VISIT) & (judged["metric"] == "medianDeltaX")]
    assert (medians["status"] == "").all()
    # Negative control: the same entry judges the calibration's offsets.
    _, _, shifted, _ = measure(makeLines(calib, dx=5.0), calib, thresholds=[file], visit=visit)
    assert shifted["qaStatus"] == "FAIL"
    assert shifted["qaDecidedBy"].startswith("calib ")


def testLampXIsMeasuredButNotJudgedByTheConfig():
    calib = FakeCalib()
    lines = makeLines(calib)
    isLamp = lines["description"] != TRACE
    lines.loc[isLamp, "x"] += 0.5
    species, judged, summary, _ = measure(lines, calib)
    assert species.loc["NeI", "medianDeltaX"] == pytest.approx(0.5)
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.0)
    assert summary["qaStatus"] == "PASS"
    lampX = judged[(judged["description"] == "NeI") & (judged["metric"] == "medianDeltaX")]
    assert (lampX["status"] == "").all()
    # Negative control: a thresholds entry naming lamp lines judges it.
    file = pd.DataFrame(
        {
            "metric": ["medianDeltaX"],
            "reference": [CALIB],
            "description": ["NeI"],
            "warn": [0.1],
            "fail": [0.15],
            "higherIsWorse": [True],
            "absolute": [True],
            "provenance": ["test"],
        }
    )
    _, _, judgedLamp, _ = measure(lines, calib, thresholds=[file, residualThresholds(0.1, 0.15, 0.1, 0.15)])
    assert judgedLamp["qaStatus"] == "FAIL"
    assert judgedLamp["qaDecidedBy"] == "calib NeI medianDeltaX"


def testTracesAreKeptOnArcs():
    """Unlike dmResiduals, trace centroids are measured alongside lamp lines."""
    calib = FakeCalib()
    species, _, summary, _ = measure(makeLines(calib, dx=0.12), calib)
    assert set(species.index) == {TRACE, "NeI"}
    assert summary["qaDecidedBy"] == f"calib {TRACE} medianDeltaX"
    # Negative control: without the traces, the shift is not seen.
    lampsOnly = makeLines(calib, traceRows=0, dx=0.12)
    _, _, noTraces, _ = measure(lampsOnly, calib)
    assert noTraces["qaStatus"] == "PASS"


def testAFiberTheVisitMapLacksIsMeasuredAgainstTheCalibration():
    calib, visit = FakeCalib(), FakeCalib(numFibers=10)
    lines = makeLines(calib)
    offsets = residuals(lines, calib, visit)
    assert len(offsets) == len(lines)
    outside = offsets["fiberId"] > 10
    assert offsets.loc[outside, "dxVisit"].isna().all()
    assert np.isfinite(offsets.loc[outside, "dxCalib"]).all()
    full = speciesOf(lines, calib, visit=visit)
    assert both(full, VISIT).loc[TRACE, "nLines"] == both(full, CALIB).loc[TRACE, "nLines"] / 2


def testLineTable():
    calib = FakeCalib()
    lines, marked = withReserved(makeLines(calib).assign(xErr=0.01, yErr=0.02))
    rejected = makeLines(calib).iloc[:5].assign(status=0)  # not marked by the fit: not stored
    table = lineTable(residuals(pd.concat([lines, rejected]), calib), DATA_ID)
    assert list(table.columns) == list(LINE_COLUMNS)
    assert table.dtypes.astype(str).tolist() == list(LINE_COLUMNS.values())
    assert len(table) == len(lines)
    assert (table["statusType"] == RESERVED).sum() == marked.sum()
    assert (table["visit"] == DATA_ID["visit"]).all() and (table["arm"] == "b").all()
    assert table.loc[table["description"] == TRACE, ["dyCalib", "dyVisit"]].isna().all().all()
    assert (table["xErr"] == 0.01).all()
    empty = lineTable(residuals(lines.assign(flag=True), calib), DATA_ID)
    assert empty.empty and list(empty.columns) == list(LINE_COLUMNS)


def testLineTableSamplesTraces():
    calib = FakeCalib()
    lines, marked = withReserved(makeLines(calib, traceRows=200))
    offsets = residuals(lines, calib)
    table = lineTable(offsets, DATA_ID, maxTraceRowsPerFiber=10)
    isTrace = (offsets["description"] == TRACE).to_numpy()
    # Every fifth line is RESERVED: 40 trace rows per fiber, above the limit like the USED ones.
    for statusType in (USED, RESERVED):
        stored = table[(table["description"] == TRACE) & (table["statusType"] == statusType)]
        assert (stored.groupby("fiberId").size() == 10).all()
        # Spread over the detector, not the first rows: the lowest and highest y of each fiber are kept.
        first = offsets[isTrace & (offsets["statusType"] == statusType).to_numpy()]
        assert set(stored.groupby("fiberId")["y"].max()) == set(first.groupby("fiberId")["y"].max())
        assert set(stored.groupby("fiberId")["y"].min()) == set(first.groupby("fiberId")["y"].min())
    # Lamp lines, RESERVED ones included, are all kept.
    lamp = table[table["description"] == "NeI"]
    assert len(lamp) == (lines["description"] == "NeI").sum()
    assert (lamp["statusType"] == RESERVED).sum() == (marked & ~isTrace).sum()
    # Negative control: with no limit, every trace row is stored, RESERVED ones included.
    whole = lineTable(offsets, DATA_ID)
    assert len(whole) == len(offsets) > len(table)
    assert (whole["statusType"] == RESERVED).sum() == marked.sum() > (table["statusType"] == RESERVED).sum()
    # The statistics do not depend on it: speciesResiduals reads lineResiduals, not the table.
    assert both(speciesResiduals(offsets, 10)).loc[TRACE, "nLines"] == (lines["description"] == TRACE).sum()


@pytest.mark.parametrize("limit", [1, 2, 7])
def testLineTableSamplingNeverExceedsTheLimit(limit):
    calib = FakeCalib(numFibers=3)
    lines, _ = withReserved(makeLines(calib, traceRows=50, lampLines=0))
    offsets = residuals(lines, calib)
    table = lineTable(offsets, DATA_ID, maxTraceRowsPerFiber=limit)
    # Both status types have more rows per fiber than any limit: each keeps exactly the limit.
    assert (offsets.groupby(["fiberId", "statusType"]).size() > max(limit, 7)).all()
    perGroup = table.groupby(["fiberId", "statusType"]).size()
    assert set(perGroup.index.get_level_values("statusType")) == {USED, RESERVED}
    assert (perGroup == limit).all()
    if limit == 1:
        # As np.linspace(0, n - 1, 1): the lowest row only.
        lowest = offsets.groupby(["fiberId", "statusType"])["y"].min()
        assert table.groupby(["fiberId", "statusType"])["y"].min().equals(lowest)


def testVisitVerdictStandsWhenTheCalibrationMeasuresNothing():
    """The calibration predicts none of the fibers; the visit's map judges its RESERVED scatter."""
    calib, visit = FakeCalib(numFibers=0), FakeCalib()
    lines, _ = withReserved(makeLines(visit), noise=0.2)
    _, _, summary, full = measure(lines, calib, thresholds=rmsThresholds(), visit=visit)
    assert (full.loc[full["reference"] == CALIB, "nLines"] == 0).all()
    assert summary["nSpeciesMeasured"] == 0  # counts species against the calibration
    assert summary["qaStatus"] == "FAIL"
    assert summary["qaDecidedBy"].startswith("visit ")
    assert summary["recommendedAction"] == "RECALIBRATE"
    # Negative control: with no lines either way, it is still insufficient data.
    _, _, empty, _ = measure(lines.assign(flag=True), calib, thresholds=rmsThresholds(), visit=visit)
    assert (empty["qaStatus"], empty["recommendedAction"]) == ("UNKNOWN", "INSUFFICIENT_DATA")


def _longRows(visit, arm, spectrograph, cameraSet, reference, description, statusType, values, obsType="arc"):
    """Return ``dmLineResidualMetrics`` rows of one species."""
    return [
        {
            "visit": visit,
            "arm": arm,
            "spectrograph": spectrograph,
            "cameraSet": cameraSet,
            "obsType": obsType,
            "reference": reference,
            "description": description,
            "statusType": statusType,
            "metric": metric,
            "value": value,
            "status": "",
        }
        for metric, value in values.items()
    ]


def testWideResiduals():
    neon = {
        "nLines": 100,
        "linesPerFiber": 2.0,
        "medianDeltaX": 0.1,
        "robustRmsDeltaX": 0.05,
        "medianDeltaY": 0.2,
        "robustRmsDeltaY": 0.06,
    }
    rows = [
        *_longRows(10, "b", 1, "brn", CALIB, "NeI", BOTH, neon),
        *_longRows(
            10,
            "b",
            1,
            "brn",
            CALIB,
            TRACE,
            BOTH,
            {"nLines": 1000, "linesPerFiber": 20.0, "medianDeltaX": -0.1},
        ),
        *_longRows(10, "b", 1, "brn", VISIT, "NeI", RESERVED, neon | {"robustRmsDeltaY": 0.07}),
        *_longRows(11, "b", 1, None, CALIB, "NeI", UNFITTED, neon),  # no pfsConfig: null camera set
        *_longRows(12, "b", 1, "brn", CALIB, "NeI", BOTH, neon),
    ]
    metrics = pd.DataFrame(rows).astype(RESIDUAL_COLUMNS)
    summary = pd.DataFrame(
        {
            "visit": [10, 11, 12],
            "arm": "b",
            "spectrograph": 1,
            "calibId": "c",
            "calibFirstVisit": 11,
            "calibLastVisit": 12,
        }
    )
    kinds = pd.DataFrame(
        {"visit": [10, 11, 12], "arm": "b", "spectrograph": 1, "obsType": "arc", "seqName": "Arc: Neon"}
    )

    drift = wideResiduals(metrics, CALIB, (BOTH, UNFITTED), summary, kinds)
    assert list(drift.columns[:8]) == [
        "visit",
        "arm",
        "spectrograph",
        "cameraSet",
        "obsType",
        "reference",
        "description",
        "statusType",
    ]
    assert len(drift) == 4  # 10 NeI, 10 Trace, 11 NeI, 12 NeI
    assert (drift["obsType"] == "arc").all()  # stored by the task
    trace = drift[(drift["visit"] == 10) & (drift["description"] == TRACE)].iloc[0]
    assert trace["medianDeltaX"] == -0.1 and np.isnan(trace["medianDeltaY"])  # a trace measures x only
    assert (drift["seqName"] == "Arc: Neon").all() and (drift["calibFirstVisit"] == 11).all()
    assert drift.loc[drift["visit"] == 11, "cameraSet"].isna().all()  # kept, not dropped as a null key
    assert sorted(drift["visit"].unique()) == [10, 11, 12]
    # The calibration's own inputs, 11-12, are left out on request; 10, outside the range, stays.
    own = wideResiduals(metrics, CALIB, (BOTH, UNFITTED), summary, kinds, excludeCalibInputs=True)
    assert sorted(own["visit"].unique()) == [10]
    scatter = wideResiduals(metrics, VISIT, (RESERVED,))
    assert len(scatter) == 1 and scatter.iloc[0]["robustRmsDeltaY"] == 0.07
    assert "calibId" not in scatter and "seqName" not in scatter
    assert wideResiduals(metrics, VISIT, (BOTH,)).empty
    with pytest.raises(ValueError, match="summary"):
        wideResiduals(metrics, CALIB, (BOTH,), excludeCalibInputs=True)
    # A table stored before obsType was a column takes it from iqQaMetrics; stored values win.
    older = metrics.drop(columns="obsType")
    assert "obsType" not in wideResiduals(older, CALIB, (BOTH, UNFITTED), summary).columns
    filled = wideResiduals(older, CALIB, (BOTH, UNFITTED), summary, kinds.assign(obsType="trace"))
    assert (filled["obsType"] == "trace").all()
    assert list(filled.columns[:5]) == [
        *drift.columns[:4],
        "reference",
    ]  # filled at the end, not in the index
    stored = wideResiduals(metrics, CALIB, (BOTH, UNFITTED), summary, kinds.assign(obsType="trace"))
    assert (stored["obsType"] == "arc").all()
    partial = addObsType(
        metrics.assign(obsType=[None, *metrics["obsType"][1:]]), kinds.assign(obsType="trace")
    )
    assert partial["obsType"].iloc[0] == "trace" and (partial["obsType"].iloc[1:] == "arc").all()
    assert addObsType(older, None) is older
    # A task whose header lacked W_SEQTYP stored "unknown": as good as missing.
    unknown = addObsType(metrics.assign(obsType="unknown"), kinds.assign(obsType="trace"))
    assert (unknown["obsType"] == "trace").all()
    # Negative control: without kinds "unknown" stays, so a thresholds entry naming a frame type never matches.
    assert (addObsType(metrics.assign(obsType="unknown"), None)["obsType"] == "unknown").all()


def _driftFixture() -> pd.DataFrame:
    from pathlib import Path

    return pd.read_parquet(Path(__file__).parent / "data" / "dmDriftMetrics-validation.parquet")


def testShippedScatterFileIsTheAdopted():
    """The shipped scatter file is what its script makes from the derivation."""
    import importlib.util
    from pathlib import Path

    import yaml

    from pfs.drp.qa.metrics.gate import thresholdsPath

    script = thresholdsPath("makeLineResidualThresholds.py")
    spec = importlib.util.spec_from_file_location("makeLineResidualThresholds", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = Path(__file__).parent / "data" / "dmLineResidualThresholds-run25.yaml"
    derivedDocument = yaml.safe_load(source.read_text())
    remade = module.adopt(derivedDocument)
    assert remade == yaml.safe_load(thresholdsPath("dmLineResidualThresholds-run25.yaml").read_text())
    # The derivation has a percentile row per arm, species, camera set and observation type, and a
    # tolerance row per detector, species and observation type. A population judged per detector keeps
    # its tolerance rows alone (quartz traces, lamp species but HgCd); the others keep their percentile rows.
    assert len(derivedDocument["thresholds"]) == 329
    assert len(remade["thresholds"]) == 168
    assert {e["method"] for e in remade["thresholds"]} == {"percentile", "tolerance"}
    perDetector = [module.judgedByTolerance(e["population"]) for e in remade["thresholds"]]
    assert all(
        (e["method"] == "tolerance") == flag
        for e, flag in zip(remade["thresholds"], perDetector, strict=True)
    )
    judged = [e for e in remade["thresholds"] if e["warn"] is not None or e["fail"] is not None]
    # Judged, per detector, above the detector's known-good median: trace centroids on quartz frames and the
    # Ar, Kr, Ne and Xe lines in x (WARN 20 %, FAIL 50 %), the lamp lines in y too (WARN 50 %, FAIL 150 %).
    assert len(judged) == 136
    assert all(e["method"] == "tolerance" and e["reliable"] for e in judged)
    x = [e for e in judged if e["metric"] == "robustRmsDeltaX"]
    y = [e for e in judged if e["metric"] == "robustRmsDeltaY"]
    assert len(x) == 76 and len(y) == 60
    assert all(e["warn"] / e["fail"] == pytest.approx(1.2 / 1.5, rel=0.03) for e in x)
    assert all("WARN 20% and FAIL 50% above the known-good median" in e["provenance"] for e in x)
    assert all(e["warn"] / e["fail"] == pytest.approx(1.5 / 2.5, rel=0.03) for e in y)
    assert all("WARN 50% and FAIL 150% above the known-good median" in e["provenance"] for e in y)
    lamps = {
        (arm, species)
        for arm in "bmnr"
        for species in ("ArI", "KrI", "NeI", "XeI")
        if (arm, species) != ("b", "ArI")
    }  # the b arm's Argon has too few lines for a RESERVED set
    assert {(e["population"]["arm"], e["population"]["description"]) for e in x} == lamps | {
        (arm, TRACE) for arm in "bmnr"
    }
    assert {
        (e["population"]["arm"], e["population"]["description"]) for e in y
    } == lamps  # traces measure x only
    assert all(
        e["population"]["obsType"] == (QUARTZ if e["population"]["description"] == TRACE else "arc")
        for e in judged
    )
    assert {e["nGood"] for e in judged} == {4, 5, 8, 10, 14}  # the calibration's own inputs count
    # Not judged, with the reason: sky species, HgCd lines, trace centroids on arcs and twilights.
    unjudged = [e for e in remade["thresholds"] if e["warn"] is None and e["fail"] is None]
    assert len(unjudged) == 32 and all(e["provenance"].startswith("Not judged (") for e in unjudged)
    reasons = {
        "sky": [e for e in unjudged if e["population"]["description"] in SKY_SPECIES],
        "hgcd": [e for e in unjudged if e["population"]["description"] in module.HGCD_SPECIES],
        "arcTraces": [
            e
            for e in unjudged
            if e["population"]["description"] == TRACE and e["population"]["obsType"] != QUARTZ
        ],
    }
    assert {k: len(v) for k, v in reasons.items()} == {"sky": 14, "hgcd": 10, "arcTraces": 8}
    assert all("sky lines' fit" in e["provenance"] for e in reasons["sky"])
    assert all(
        "mercury or cadmium" in e["provenance"] and e["method"] == "percentile" for e in reasons["hgcd"]
    )
    assert all("faint" in e["provenance"] and e["method"] == "percentile" for e in reasons["arcTraces"])
    # Negative control: the derivation gave the arc-frame traces reliable percentile thresholds, the pooled
    # quartz traces a FAIL and the HgCd lines 40 reliable tolerances; withholding them is the adoption's rule.
    percentile = {
        (
            e["population"]["arm"],
            e["population"]["description"],
            e["population"].get("cameraSet"),
            e["population"]["obsType"],
            e["metric"],
        ): e
        for e in derivedDocument["thresholds"]
        if e["method"] == "percentile"
    }
    assert sum(percentile[k]["reliable"] for k in percentile if k[1] == TRACE and k[3] == "arc") == 5
    assert percentile[("r", TRACE, "brn", "trace", "robustRmsDeltaX")]["fail"] is not None
    hgcd = [
        e
        for e in derivedDocument["thresholds"]
        if e["method"] == "tolerance" and e["population"]["description"] in module.HGCD_SPECIES
    ]
    assert sum(e["reliable"] for e in hgcd) == 40


def testShippedScatterVerdicts():
    """The adopted file on the validation run (u/wtg/qa-thresholds/010).

    Trace centroids on quartz frames and the Ar, Kr, Ne and Xe lines are judged in x per detector, the
    lamp lines in y too; trace centroids on arcs and twilights, the HgCd lines and the sky species
    are not. Counts are the fixture's, whose ``obsType`` is imageQualityQa's.
    """
    from pfs.drp.qa.metrics.calibration import labelRows
    from pfs.drp.qa.metrics.gate import thresholdsPath
    from pfs.drp.qa.metrics.validationVisits import loadValidationVisits

    rows = _driftFixture()
    reserved = rows[(rows["reference"] == VISIT) & (rows["statusType"] == RESERVED)].reset_index(drop=True)
    assert reserved["obsType"].value_counts().to_dict() == {"arc": 1427, "twilight": 623, "trace": 576}
    thresholds = [
        thresholdsPath("dmLineResidualThresholds-run25.yaml"),
        residualThresholds(1.0, 2.0, 1.0, 2.0),
    ]
    judged = judge(reserved, thresholds).set_index(["row", "metric"])
    visitSet = loadValidationVisits()
    sky = reserved["description"].isin(SKY_SPECIES)
    traces = reserved["description"] == TRACE
    quartz = reserved["obsType"] == QUARTZ
    hgcd = reserved["description"].isin(["HgI", "CdI"])
    lamps = ~traces & ~sky & ~hgcd
    counts = {
        "robustRmsDeltaX": {
            "quartzTraces": {"PASS": 479, "FAIL": 82, "WARN": 14},
            "quartzByArm": {
                ("b", "FAIL"): 20,
                ("b", "PASS"): 180,
                ("b", "WARN"): 12,
                ("m", "PASS"): 56,
                ("n", "FAIL"): 25,
                ("n", "PASS"): 118,
                ("r", "FAIL"): 37,
                ("r", "PASS"): 125,
                ("r", "WARN"): 2,
            },
            "otherTraces": 862,
            "lamps": {"PASS": 468, "FAIL": 63, "WARN": 23},
            "goodQuartzTraces": {"PASS": 144},  # every known-good quartz trace passes
            "goodLamps": {"PASS": 357, "WARN": 3},  # three b-arm Xenon arcs at the 20 % WARN
            # 150661, the screen in the wrong position (Run30): known-bad on the b arm, where every
            # detector fails; r and n, mixed, carry no verdict (PIPE2D-1934).
            "badQuartzTraces": {"FAIL": 4},
        },
        "robustRmsDeltaY": {
            "quartzTraces": {},  # traces measure x only
            "quartzByArm": {},
            "otherTraces": 0,
            "lamps": {"PASS": 450, "FAIL": 55, "WARN": 49},
            "goodQuartzTraces": {"": 144},
            "goodLamps": {"PASS": 358, "WARN": 2},  # two b-arm Krypton arcs at the 50 % WARN
            "badQuartzTraces": {},  # 150661 is known-bad in x only
        },
    }
    for metric, expected in counts.items():
        table = judged.xs(metric, level="metric")
        status, layer = table["status"], table["layer"]
        # Sky species are unjudged: by an entry with no thresholds where the derivation had values
        # (423 rows), and by absence, the config's NaN, where it had none (8).
        assert sky.sum() == 431 and (status[sky] == "").all() and (layer[sky] == 0).sum() == 423
        measured = reserved[metric].notna()
        quartzTraces = traces & quartz & measured
        assert status[quartzTraces].value_counts().to_dict() == expected["quartzTraces"]
        assert (
            status[quartzTraces].groupby(reserved["arm"]).value_counts().to_dict() == expected["quartzByArm"]
        )
        assert (layer[quartzTraces] == 0).all()  # every quartz trace meets an entry, judged or not
        otherTraces = traces & ~quartz & measured
        assert otherTraces.sum() == expected["otherTraces"] and (status[otherTraces] == "").all()
        assert (layer[otherTraces] == 0).all()  # by an entry with no thresholds, not by absence
        assert (lamps & measured).sum() == 554
        assert status[lamps & measured].value_counts().to_dict() == expected["lamps"]
        assert (layer[lamps & measured] == 0).all()
        assert (hgcd & measured).sum() == 171 and (status[hgcd & measured] == "").all()
        labelled = labelRows(reserved, visitSet, metric)
        good = labelled.index[labelled["validation"] == "good"]
        assert (
            status.loc[good[(traces & quartz)[good]]].value_counts().to_dict() == expected["goodQuartzTraces"]
        )
        assert status.loc[good[lamps[good]]].value_counts().to_dict() == expected["goodLamps"]
        assert not (labelled["validation"] == "unconfirmed").any()  # no scatter verdict is a suspicion
        bad = labelled.index[labelled["validation"].str.startswith("bad")]
        assert status.loc[bad[(traces & quartz)[bad]]].value_counts().to_dict() == expected["badQuartzTraces"]
        assert not lamps[bad].any()
    # Negative control: a flat 0.03/0.04 px scatter threshold fails or warns most known-good quartz traces.
    strict = judge(reserved, [residualThresholds(1.0, 2.0, 1.0, 2.0, 0.03, 0.04, 0.03, 0.04)])
    strictX = strict.set_index(["row", "metric"]).xs("robustRmsDeltaX", level="metric")["status"]
    labelled = labelRows(reserved, visitSet, "robustRmsDeltaX")
    goodTraces = labelled.index[(labelled["validation"] == "good") & (traces & quartz)[labelled.index]]
    assert strictX.loc[goodTraces].value_counts().to_dict() == {"FAIL": 96, "PASS": 30, "WARN": 18}
