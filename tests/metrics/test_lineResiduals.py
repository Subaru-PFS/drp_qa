"""Tests for line residuals against the calibration and the visit's detectorMap (dmLineResiduals)."""

from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.gate import longLines, measurements
from pfs.drp.qa.metrics.lineResiduals import (
    BAD_STATUS,
    BOTH,
    CALIB,
    LINE_COLUMNS,
    QUARTZ,
    REFERENCES,
    RESERVED,
    RESIDUAL_COLUMNS,
    RESIDUAL_METRICS,
    SKY_SPECIES,
    STATUS_TYPES,
    SUMMARY_COLUMNS,
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
    residualSummary,
    robustRms,
    sequenceType,
    speciesResiduals,
    wideResiduals,
)
from pfs.drp.qa.metrics.registry import METRIC_SPECS, specFor

DATA_ID = {"visit": 1, "arm": "b", "spectrograph": 2}
NAN = float("nan")
#: drp_stella's ReferenceLineStatus bits.
USED_BIT, RESERVED_BIT = 0x20, 0x40
#: The speed of light, km/s: written out here, not imported, so the tests check the module's.
C_KMS = 299792.458


class FakeCalib:
    """Straight vertical traces 6 px apart; y is ``pixelsPerNm`` (10) px per nm above 500 nm.

    With 10 px per nm the dispersion is 0.1 nm per pixel everywhere; a negative
    ``pixelsPerNm`` makes wavelength fall with y.
    """

    def __init__(self, numFibers: int = 20, pixelsPerNm: float = 10.0):
        self.fiberId = np.arange(1, numFibers + 1, dtype=np.int32)
        self.pixelsPerNm = pixelsPerNm

    def getXCenter(self, fiberId, y):
        fiberId = np.asarray(fiberId)
        if not np.isin(fiberId, self.fiberId).all():
            raise IndexError("fiberId not in the detectorMap")  # as a SplinedDetectorMap does
        return 6.0 * fiberId + 0.0 * np.asarray(y, dtype=float)

    def row(self, wavelength):
        """Return the row a wavelength falls on."""
        return self.pixelsPerNm * (np.asarray(wavelength, dtype=float) - 500.0)

    def findPoint(self, fiberId, wavelength):
        fiberId = np.asarray(fiberId, dtype=float)
        return np.column_stack([6.0 * fiberId, self.row(wavelength)])

    def getDispersion(self, fiberId, wavelength, dWavelength=0.1):
        """Return nm per pixel as drp_stella's ``DetectorMap`` does: a wavelength step over the rows it spans."""
        # The stack's vectorised overload takes contiguous int32 and float64 arrays, and nothing else.
        fiberId, wavelength = np.asarray(fiberId), np.asarray(wavelength)
        assert fiberId.dtype == np.int32 and wavelength.dtype == np.float64
        assert fiberId.flags.c_contiguous and wavelength.flags.c_contiguous
        y0 = self.findPoint(fiberId, wavelength - 0.5 * dWavelength)[:, 1]
        y1 = self.findPoint(fiberId, wavelength + 0.5 * dWavelength)[:, 1]
        return dWavelength / (y1 - y0)


class CurvedMap(FakeCalib):
    """`FakeCalib` with the dispersion changing along the detector: 10 px per nm at 500 nm, 13.8 at 880 nm."""

    def row(self, wavelength):
        offset = np.asarray(wavelength, dtype=float) - 500.0
        return 10.0 * offset + 0.005 * offset**2

    @staticmethod
    def dispersion(wavelength):
        """Return the dispersion (nm per pixel) from the derivative of `row`, not by `getDispersion`."""
        return 1.0 / (10.0 + 0.01 * (np.asarray(wavelength, dtype=float) - 500.0))


class NoDispersionMap(FakeCalib):
    """A detectorMap that predicts positions but has no ``getDispersion``."""

    getDispersion = None


class PartialDispersionMap(FakeCalib):
    """A detectorMap that gives no dispersion (NaN) redward of 600 nm, as at the edge of a detector."""

    def getDispersion(self, fiberId, wavelength, dWavelength=0.1):
        dispersion = super().getDispersion(fiberId, wavelength, dWavelength)
        return np.where(np.asarray(wavelength) > 600.0, np.nan, dispersion)


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


def measure(lines, calib, minLines=10, litFibers=None, visit=None):
    """Return ``(both(species), summary, species)`` for ``lines``."""
    full = speciesOf(lines, calib, minLines, litFibers, visit)
    return both(full), residualSummary(full, minLines), full


def testZeroOffset():
    calib = FakeCalib()
    species, summary, _ = measure(makeLines(calib), calib)
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.0, abs=1e-12)
    assert species.loc["NeI", "medianDeltaY"] == pytest.approx(0.0, abs=1e-12)
    assert species.loc["NeI", "medianDeltaYKms"] == pytest.approx(0.0, abs=1e-9)
    assert summary == {"fitted": True, "nSpecies": 2, "nSpeciesMeasured": 2}


def testSummaryHasNoVerdict():
    """The task writes numbers only: nothing in the summary or the long table is a verdict."""
    calib = FakeCalib()
    _, summary, full = measure(makeLines(calib, dx=5.0, dy=5.0), calib)
    assert tuple(summary) == SUMMARY_COLUMNS == ("fitted", "nSpecies", "nSpeciesMeasured")
    long = longResiduals(full, DATA_ID)
    for name in ("status", "qaStatus", "qaDecidedBy", "qaReason", "recommendedAction"):
        assert name not in summary and name not in long.columns and name not in RESIDUAL_COLUMNS


def testInjectedShiftIsRecovered():
    calib = FakeCalib()
    lines = makeLines(calib, dx=0.2, noise=0.02)
    species, _, _ = measure(lines, calib)
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.2, abs=0.01)
    assert species.loc[TRACE, "robustRmsDeltaX"] == pytest.approx(0.02, rel=0.25)
    # Negative control: the offset is measured minus calibration, not calibration minus measured
    # (imageQualityQa's dxCenter): a +0.2 px shift does not read as -0.2 px.
    assert not np.isclose(species.loc[TRACE, "medianDeltaX"], -0.2, atol=0.01)


def testInjectedYShiftIsRecovered():
    calib = FakeCalib()
    species, _, _ = measure(makeLines(calib, dy=-0.12), calib)
    assert species.loc["NeI", "medianDeltaY"] == pytest.approx(-0.12)
    # Negative control: the sign is kept; only the gate takes the absolute value.
    assert species.loc["NeI", "medianDeltaY"] < 0
    assert not np.isclose(species.loc["NeI", "medianDeltaY"], 0.12)


def testTooFewLinesIsNotMeasured():
    calib = FakeCalib(numFibers=1)
    lines = makeLines(calib, traceRows=5, lampLines=5, dx=5.0, dy=5.0)
    species, summary, _ = measure(lines, calib)
    assert species["nLines"].tolist() == [5, 5]
    assert species[list(RESIDUAL_METRICS)].isna().all().all()
    assert summary == {"fitted": True, "nSpecies": 2, "nSpeciesMeasured": 0}
    # Negative control: the same lines are measured when five are enough.
    enough, summaryEnough, _ = measure(lines, calib, minLines=5)
    assert enough.loc[TRACE, "medianDeltaX"] == pytest.approx(5.0)
    assert enough.loc["NeI", "medianDeltaY"] == pytest.approx(5.0)
    assert np.isfinite(enough.loc["NeI", "medianDeltaYKms"])
    assert summaryEnough["nSpeciesMeasured"] == 2


def testNoLines():
    calib = FakeCalib()
    lines = makeLines(calib).assign(flag=True)
    species, summary, full = measure(lines, calib)
    assert full.empty and species.empty
    assert summary == {"fitted": True, "nSpecies": 0, "nSpeciesMeasured": 0}


def testTraceOnlyFillsX():
    calib = FakeCalib()
    lines = makeLines(calib, lampLines=0, dx=0.05)
    species, summary, full = measure(lines, calib)
    assert list(species.index) == [TRACE]
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.05)
    for metric in ("medianDeltaY", "robustRmsDeltaY", "medianDeltaYKms"):
        assert np.isnan(species.loc[TRACE, metric])
    assert summary["nSpeciesMeasured"] == 1
    long = longResiduals(full, DATA_ID)
    assert set(long["metric"]) == {"nLines", "linesPerFiber", "medianDeltaX", "robustRmsDeltaX"}


def testOneSpeciesShortLeavesTheOtherMeasured():
    calib = FakeCalib()
    lines = makeLines(calib, lampLines=0, dx=0.2)
    lamp = makeLines(FakeCalib(numFibers=1), traceRows=0, lampLines=3, species="HgI")
    species, summary, _ = measure(pd.concat([lines, lamp]), calib)
    assert species.loc["HgI", "nLines"] == 3
    assert np.isnan(species.loc["HgI", "medianDeltaY"])
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.2)
    assert (summary["nSpecies"], summary["nSpeciesMeasured"]) == (2, 1)


def testSpeciesAreNotBlended():
    calib = FakeCalib()
    neon = makeLines(calib, traceRows=0, dy=0.12, species="NeI")
    mercury = makeLines(calib, traceRows=0, dy=0.0, species="HgI")
    species, _, _ = measure(pd.concat([neon, mercury]), calib)
    assert species.loc["NeI", "medianDeltaY"] == pytest.approx(0.12)
    assert species.loc["HgI", "medianDeltaY"] == pytest.approx(0.0)
    # Negative control: pooled, the two would read as one offset between them.
    pooled, _, _ = measure(pd.concat([neon, mercury.assign(description="NeI")]), calib)
    assert not np.isclose(pooled.loc["NeI", "medianDeltaY"], 0.12)


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


def testLongFormat():
    calib = FakeCalib()
    _, _, full = measure(makeLines(calib, dx=0.2), calib)
    long = longResiduals(full, DATA_ID, "brn", QUARTZ, "scienceTrace")
    assert list(long.columns) == list(RESIDUAL_COLUMNS)
    assert list(RESIDUAL_COLUMNS) == [
        "visit",
        "arm",
        "spectrograph",
        "cameraSet",
        "obsType",
        "seqType",
        "reference",
        "description",
        "statusType",
        "metric",
        "value",
    ]
    assert long.dtypes.astype(str).tolist() == list(RESIDUAL_COLUMNS.values())
    assert set(long["seqType"]) == {"scienceTrace"} and set(long["obsType"]) == {QUARTZ}
    # Without a header the sequence type is null, not a guess.
    assert longResiduals(full, DATA_ID, "brn")["seqType"].isna().all()
    trace = long[(long["description"] == TRACE) & (long["statusType"] == BOTH)]
    trace = trace[trace["reference"] == CALIB].set_index("metric")
    assert trace.loc["medianDeltaX", "value"] == pytest.approx(0.2)
    assert trace.loc["nLines", "value"] == 400
    assert (long["cameraSet"] == "brn").all()
    assert (long["obsType"] == QUARTZ).all() and long["obsType"].dtype == "string"
    lamp = long[long["description"] == "NeI"]
    assert set(lamp["metric"]) == {
        "nLines",
        "linesPerFiber",
        "medianDeltaX",
        "robustRmsDeltaX",
        "medianDeltaY",
        "robustRmsDeltaY",
        "medianDeltaYKms",
    }
    assert list(long["reference"].unique()) == [CALIB, VISIT]
    assert list(long["statusType"].unique()) == list(STATUS_TYPES)
    # Without a pfsConfig or a header, the population columns are null.
    bare = longResiduals(full, DATA_ID)
    assert bare["obsType"].isna().all() and bare["cameraSet"].isna().all()
    empty = longResiduals(full.iloc[:0], DATA_ID)
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
        ({"W_SEQTYP": "darks", "W_SEQNAM": "Morning Dark"}, "dark"),
        ({"W_SEQTYP": "biases"}, "bias"),
        ({"W_SEQTYP": "fiberProfiles", "W_SEQNAM": "allHidden"}, "unknown"),
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
        # A raw whose sequence type has no measurement of its own is not read as what the pfsConfig says.
        ({"W_SEQTYP": "ditheredFlats"}, {"W_SEQTYP": "scienceArc"}, "unknown"),
        ({}, {}, "unknown"),
        (None, {}, "unknown"),
        ({}, None, "unknown"),
        (None, None, None),
    ],
)
def testFrameTypeReadsTheRawFirst(raw, config, expected):
    """The raw header's type, the pfsConfig's when the raw has none."""
    assert frameType(*(None if header is None else Header(header) for header in (raw, config))) == expected
    # Negative control: when both say, the raw wins over the pfsConfig.
    both = frameType(Header({"W_SEQTYP": "scienceArc"}), Header({"W_SEQTYP": "scienceTrace"}))
    assert both == "arc" and both != QUARTZ


@pytest.mark.parametrize(
    "raw, config, expected",
    [
        ({"W_SEQTYP": "fiberProfiles"}, {}, "fiberProfiles"),
        ({}, {"W_SEQTYP": " scienceArc "}, "scienceArc"),
        (None, {"W_SEQTYP": "darks"}, "darks"),
        ({"W_SEQTYP": "rampObject"}, {"W_SEQTYP": "scienceArc"}, "rampObject"),  # as read, classified or not
        ({}, {}, None),
        (None, None, None),
    ],
)
def testSequenceTypeIsStoredAsRead(raw, config, expected):
    """``seqType`` is the header's ``W_SEQTYP`` itself, by the headers `frameType` reads; null without one."""
    assert sequenceType(*(None if header is None else Header(header) for header in (raw, config))) == expected


def testRegistry():
    """Every statistic stored is declared in the registry, with its units."""
    assert set(RESIDUAL_METRICS) <= set(METRIC_SPECS)
    for name in ("medianDeltaX", "medianDeltaY", "medianDeltaYKms"):
        assert specFor(name).absolute
    for name in ("robustRmsDeltaX", "robustRmsDeltaY"):
        assert not specFor(name).absolute
    assert specFor("medianDeltaYKms").units == "km/s"
    assert {specFor(name).units for name in RESIDUAL_METRICS if name != "medianDeltaYKms"} == {"pixels"}
    assert RESIDUAL_METRICS["medianDeltaYKms"] == "y"


def testUsedAndReservedAreKeptApart():
    calib = FakeCalib()
    lines = makeLines(calib, lampLines=0)
    reserved = np.arange(len(lines)) % 5 == 0  # a fit holds back about a fifth
    lines.loc[reserved, "status"] = RESERVED_BIT
    lines.loc[reserved, "x"] += 0.3
    _, summary, full = measure(lines, calib)
    trace = full[(full["description"] == TRACE) & (full["reference"] == CALIB)].set_index("statusType")
    assert list(trace.index) == list(STATUS_TYPES)
    assert trace.loc[USED, "nLines"] == (~reserved).sum()
    assert trace.loc[RESERVED, "nLines"] == reserved.sum()
    assert trace.loc[BOTH, "nLines"] == len(lines)
    assert trace.loc[USED, "medianDeltaX"] == pytest.approx(0.0)
    assert trace.loc[RESERVED, "medianDeltaX"] == pytest.approx(0.3)
    assert trace.loc[BOTH, "medianDeltaX"] == pytest.approx(0.0)
    # The summary counts a species once, on the union.
    assert (summary["nSpecies"], summary["nSpeciesMeasured"]) == (1, 1)
    # Negative control: the held-back lines' offset is not what the union reads.
    assert abs(trace.loc[RESERVED, "medianDeltaX"]) > 0.15 > abs(trace.loc[BOTH, "medianDeltaX"])


def testLinesTheFitDidNotMarkAreLeftOut():
    calib = FakeCalib()
    lines = makeLines(calib)
    rejected = makeLines(calib, dx=5.0, dy=5.0).assign(status=0)  # good, but neither used nor reserved
    species, _, _ = measure(pd.concat([lines, rejected]), calib)
    numTraces = (lines["description"] == TRACE).sum()
    assert species.loc[TRACE, "nLines"] == numTraces
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.0)
    # Negative control: the same lines, marked USED, are measured.
    marked, _, _ = measure(pd.concat([lines, rejected.assign(status=USED_BIT)]), calib)
    assert marked.loc[TRACE, "nLines"] == 2 * numTraces
    assert marked.loc[TRACE, "medianDeltaX"] > 1.0


def testUnfittedIsMeasuredOnEveryGoodLine():
    calib = FakeCalib()
    lines = makeLines(calib, dx=0.2)
    species, summary, full = measure(lines.assign(status=0), calib)
    assert set(full["statusType"]) == {UNFITTED}
    assert species.loc[TRACE, "nLines"] == (lines["description"] == TRACE).sum()
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.2)
    assert summary == {"fitted": False, "nSpecies": 2, "nSpeciesMeasured": 2}
    long = longResiduals(full, DATA_ID)
    assert set(long["statusType"]) == {UNFITTED}
    traceX = long[(long["metric"] == "medianDeltaX") & (long["description"] == TRACE)]
    assert traceX.set_index("reference")["value"].to_dict() == pytest.approx({CALIB: 0.2, VISIT: 0.2})
    # Negative control: the same lines marked by a fit are measured as BOTH, USED and RESERVED.
    _, marked, markedFull = measure(lines, calib)
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
    assert np.isfinite(offsets.loc[offsets["description"] == "NeI", "dyKmsCalib"]).all()
    assert offsets[["dxVisit", "dyVisit", "dyKmsVisit"]].isna().all().all()
    species, summary, _ = measure(lines, calib)
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.05)
    assert species.loc["NeI", "medianDeltaY"] == pytest.approx(0.05)
    assert species.loc[TRACE, "nLines"] < (lines["description"] == TRACE).sum()
    assert summary["nSpeciesMeasured"] == 2
    # Negative control: the calibration does predict NaN for some of these lines.
    xCenter = calib.getXCenter(lines["fiberId"].to_numpy(), lines["y"].to_numpy())
    assert np.isnan(xCenter[(lines["description"] == TRACE).to_numpy()]).any()


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
    assert againstVisit.loc["NeI", "medianDeltaYKms"] == pytest.approx(0.0, abs=1e-9)
    # Negative control: the references are not swapped.
    assert not np.isclose(againstVisit.loc[TRACE, "medianDeltaX"], 0.5)
    assert againstCalib.loc["NeI", "medianDeltaYKms"] < -1.0
    offsets = residuals(lines, calib, visit)
    assert offsets["dxCalib"].to_numpy() == pytest.approx(offsets["dxVisit"].to_numpy() + 0.5)


def testReservedScatterAgainstTheVisitIsMeasured():
    calib = FakeCalib()
    lines, _ = withReserved(makeLines(calib), noise=0.2)
    _, _, full = measure(lines, calib)
    rows = reserved(full)
    assert rows.loc[TRACE, "robustRmsDeltaX"] == pytest.approx(0.2, rel=0.3)
    assert rows.loc["NeI", "robustRmsDeltaY"] == pytest.approx(0.2, rel=0.3)
    # Negative control: the lines the fit used carry none of it, and are stored apart.
    used = full[(full["reference"] == VISIT) & (full["statusType"] == USED)].set_index("description")
    assert used.loc[TRACE, "robustRmsDeltaX"] == pytest.approx(0.0, abs=1e-9)
    assert used.loc["NeI", "robustRmsDeltaY"] == pytest.approx(0.0, abs=1e-9)


def testLampXIsMeasured():
    calib = FakeCalib()
    lines = makeLines(calib)
    isLamp = lines["description"] != TRACE
    lines.loc[isLamp, "x"] += 0.5
    species, _, full = measure(lines, calib)
    assert species.loc["NeI", "medianDeltaX"] == pytest.approx(0.5)
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.0)
    long = longResiduals(full, DATA_ID)
    lampX = long[(long["description"] == "NeI") & (long["metric"] == "medianDeltaX")]
    assert lampX.loc[lampX["statusType"] != RESERVED, "value"].to_numpy() == pytest.approx(0.5)
    assert lampX.loc[lampX["statusType"] == RESERVED, "value"].isna().all()  # the fit held none back
    # Negative control: a lamp line's x offset is not folded into its y statistics.
    assert species.loc["NeI", "medianDeltaY"] == pytest.approx(0.0)


def testTracesAreKeptOnArcs():
    """Unlike dmResiduals, trace centroids are measured alongside lamp lines."""
    calib = FakeCalib()
    species, _, _ = measure(makeLines(calib, dx=0.12), calib)
    assert set(species.index) == {TRACE, "NeI"}
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.12)
    # Negative control: without the traces, there is no trace row to show the shift.
    lampsOnly, _, _ = measure(makeLines(calib, traceRows=0, dx=0.12), calib)
    assert set(lampsOnly.index) == {"NeI"}


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
    lines, marked = withReserved(makeLines(calib, dy=0.1).assign(xErr=0.01, yErr=0.02))
    rejected = makeLines(calib).iloc[:5].assign(status=0)  # not marked by the fit: not stored
    table = lineTable(residuals(pd.concat([lines, rejected]), calib), DATA_ID)
    assert list(table.columns) == list(LINE_COLUMNS)
    assert list(LINE_COLUMNS)[-6:] == ["dxCalib", "dyCalib", "dxVisit", "dyVisit", "dyKmsCalib", "dyKmsVisit"]
    assert table.dtypes.astype(str).tolist() == list(LINE_COLUMNS.values())
    assert len(table) == len(lines)
    assert (table["statusType"] == RESERVED).sum() == marked.sum()
    assert (table["visit"] == DATA_ID["visit"]).all() and (table["arm"] == "b").all()
    isTrace = table["description"] == TRACE
    assert table.loc[isTrace, ["dyCalib", "dyVisit", "dyKmsCalib", "dyKmsVisit"]].isna().all().all()
    # A lamp line's dy in km/s, by the dispersion at the line: 0.1 px at 0.1 nm per pixel.
    lamp = table[~isTrace]
    expected = 0.1 * C_KMS * 0.1 / lamp["wavelength"].to_numpy()
    assert lamp["dyKmsCalib"].to_numpy() == pytest.approx(expected)
    assert lamp["dyKmsVisit"].to_numpy() == pytest.approx(expected)
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


def testTheVisitIsMeasuredWhenTheCalibrationMeasuresNothing():
    """The calibration predicts none of the fibers; the lines are still measured against the visit's map."""
    calib, visit = FakeCalib(numFibers=0), FakeCalib()
    lines, _ = withReserved(makeLines(visit), noise=0.2)
    _, summary, full = measure(lines, calib, visit=visit)
    assert (full.loc[full["reference"] == CALIB, "nLines"] == 0).all()
    # The summary counts species against the calibration.
    assert summary == {"fitted": True, "nSpecies": 2, "nSpeciesMeasured": 0}
    assert reserved(full).loc[TRACE, "robustRmsDeltaX"] == pytest.approx(0.2, rel=0.3)
    assert reserved(full).loc["NeI", "robustRmsDeltaY"] == pytest.approx(0.2, rel=0.3)
    # Negative control: with no lines either way, nothing is measured against the visit either.
    _, empty, emptyFull = measure(lines.assign(flag=True), calib, visit=visit)
    assert emptyFull.empty and empty["nSpecies"] == 0


def testKmsIsEachLinesVelocity():
    """``medianDeltaYKms`` is the median of each line's ``dy * c * dispersion / wavelength``, in km/s."""
    # Units: a line 1 px off at 600 nm, where a pixel is 0.1 nm, is 0.1 / 600 of c: 49.97 km/s.
    flat = FakeCalib()
    one = makeLines(flat, traceRows=0, lampLines=1, species="OH").assign(wavelength=600.0)
    one["y"] = flat.row(600.0) + 1.0
    assert both(speciesOf(one, flat)).loc["OH", "medianDeltaYKms"] == pytest.approx(49.965, abs=1e-3)

    # A dispersion that changes along the detector, and offsets that differ from line to line: large on
    # the three bluest lines of each fiber, small on the others.
    calib = CurvedMap()
    lines = makeLines(calib, traceRows=0, lampLines=7, species="OH")
    perLine = np.tile([0.6, 0.5, 0.4, 0.05, 0.1, 0.1, 0.1], len(calib.fiberId))
    dy = perLine + np.random.default_rng(5).normal(0.0, 0.005, len(lines))
    lines["y"] += dy
    wavelength = lines["wavelength"].to_numpy()
    dispersion = CurvedMap.dispersion(wavelength)  # nm per pixel: 0.098 at 520 nm, 0.072 at 880 nm
    kmsPerPixel = C_KMS * dispersion / wavelength  # 56 km/s at 520 nm, 25 km/s at 880 nm
    offsets = residuals(lines, calib)
    assert offsets["dyCalib"].to_numpy() == pytest.approx(dy)
    assert offsets["dyKmsCalib"].to_numpy() == pytest.approx(dy * kmsPerPixel)
    species = both(speciesResiduals(offsets, 10))
    measured = species.loc["OH", "medianDeltaYKms"]
    assert measured == pytest.approx(np.median(dy * kmsPerPixel))
    # Negative control: it is not the pixel value,
    assert not np.isclose(measured, species.loc["OH", "medianDeltaY"], rtol=0.5)
    # nor the pixel median times one scale for the detector (the dispersion and wavelength at its middle),
    middle = C_KMS * CurvedMap.dispersion(700.0) / 700.0
    assert not np.isclose(measured, species.loc["OH", "medianDeltaY"] * middle, rtol=0.02)
    # nor made with the wrong dispersion (the flat map's 0.1 nm per pixel), nor in m/s, nor in Angstrom.
    assert not np.isclose(measured, np.median(dy * C_KMS * 0.1 / wavelength), rtol=0.02)
    assert not np.isclose(measured, 1e3 * np.median(dy * kmsPerPixel), rtol=0.5)
    assert not np.isclose(measured, 10.0 * np.median(dy * kmsPerPixel), rtol=0.5)


def testKmsUsesEachReferencesOwnDispersion():
    """Against the calibration, the calibration's dispersion; against the visit's map, the visit's."""
    calib, visit = FakeCalib(), CurvedMap()
    lines = makeLines(visit, traceRows=0, lampLines=7, dy=0.2, species="OH")
    full = speciesOf(lines, calib, visit=visit)
    wavelength = lines["wavelength"].to_numpy()
    expected = np.median(0.2 * C_KMS * CurvedMap.dispersion(wavelength) / wavelength)
    assert both(full, VISIT).loc["OH", "medianDeltaYKms"] == pytest.approx(expected)
    # Against the flat calibration the lines sit elsewhere, with its 0.1 nm per pixel.
    offsets = residuals(lines, calib, visit)
    assert offsets["dyKmsCalib"].to_numpy() == pytest.approx(
        offsets["dyCalib"].to_numpy() * C_KMS * 0.1 / wavelength
    )
    # Negative control: the visit's residual with the calibration's dispersion does not match.
    wrong = np.median(0.2 * C_KMS * 0.1 / wavelength)
    assert not np.isclose(both(full, VISIT).loc["OH", "medianDeltaYKms"], wrong, rtol=0.02)


def testKmsSignIsRedward():
    """A line redward of where the reference puts it is positive, whichever way wavelength runs in y."""
    values = {}
    for pixelsPerNm in (10.0, -10.0):
        calib = FakeCalib(pixelsPerNm=pixelsPerNm)
        lines = makeLines(calib, traceRows=0, species="OH")
        # Each line sits where one 0.02 nm longer would: 0.02 nm to the red.
        lines["y"] = calib.row(lines["wavelength"] + 0.02)
        species = both(speciesOf(lines, calib))
        expected = np.median(C_KMS * 0.02 / lines["wavelength"])  # +8.6 km/s at 700 nm
        assert species.loc["OH", "medianDeltaYKms"] == pytest.approx(expected)
        assert species.loc["OH", "medianDeltaY"] == pytest.approx(0.02 * pixelsPerNm)
        values[pixelsPerNm] = species.loc["OH"]
    # Where wavelength rises with y, it has the sign of dy.
    assert values[10.0]["medianDeltaY"] > 0 and values[10.0]["medianDeltaYKms"] > 0
    # Negative control: where wavelength falls with y it has not; the size of the dispersion alone
    # (its absolute value) would give the sign of dy, a blueshift for this redward line.
    assert values[-10.0]["medianDeltaY"] < 0 < values[-10.0]["medianDeltaYKms"]
    unsigned = values[-10.0]["medianDeltaY"] * C_KMS * abs(-0.1) / 700.0
    assert not np.isclose(values[-10.0]["medianDeltaYKms"], unsigned, rtol=0.5)
    # Negative control: a line to the blue is negative.
    blue = makeLines(FakeCalib(), traceRows=0, dy=-0.2, species="OH")
    assert both(speciesOf(blue, FakeCalib())).loc["OH", "medianDeltaYKms"] < 0


def testKmsRowsAreThoseWithAYResidual():
    """Lamp and sky species get it, against each reference and for each line set; trace centroids never."""
    calib = FakeCalib()
    lines = pd.concat(
        [makeLines(calib, dy=0.1, species="OH"), makeLines(calib, traceRows=0, dy=0.1, species="NeI")],
        ignore_index=True,
    )
    lines, _ = withReserved(lines)
    full = speciesOf(lines, calib)
    long = longResiduals(full, DATA_ID)
    kms = long[long["metric"] == "medianDeltaYKms"]
    assert set(kms["description"]) == {"OH", "NeI"}
    for description in ("OH", "NeI"):
        rows = kms[kms["description"] == description]
        assert sorted(zip(rows["reference"], rows["statusType"], strict=True)) == sorted(
            product(REFERENCES, STATUS_TYPES)
        )
    assert np.isfinite(kms["value"]).all() and (kms["value"] > 0).all()
    assert full.loc[full["description"] == TRACE, "medianDeltaYKms"].isna().all()
    # The species' other y statistics have a row wherever it has one.
    perMetric = long[long["description"] == "OH"].groupby("metric").size()
    assert perMetric["medianDeltaYKms"] == perMetric["medianDeltaY"] == perMetric["robustRmsDeltaY"] == 6
    # An unfitted detector has it too, on its one line set.
    unfitted = longResiduals(speciesOf(lines.assign(status=0), calib), DATA_ID)
    rows = unfitted[(unfitted["metric"] == "medianDeltaYKms") & (unfitted["description"] == "OH")]
    assert sorted(zip(rows["reference"], rows["statusType"], strict=True)) == [
        (CALIB, UNFITTED),
        (VISIT, UNFITTED),
    ]
    # The wide form has it as a column, NaN for trace centroids.
    wide = wideResiduals(long, VISIT, (RESERVED,)).set_index("description")
    assert wide.loc["OH", "medianDeltaYKms"] == pytest.approx(reserved(full).loc["OH", "medianDeltaYKms"])
    assert np.isnan(wide.loc[TRACE, "medianDeltaYKms"])
    # Negative control: a trace centroid has no row for it, as it has none for the other y statistics.
    traceMetrics = set(long.loc[long["description"] == TRACE, "metric"])
    assert not traceMetrics & {"medianDeltaY", "robustRmsDeltaY", "medianDeltaYKms"}
    assert "medianDeltaX" in traceMetrics


def testKmsNeedsADispersion():
    """NaN without a dispersion, or with one for fewer than ``minLines`` lines; the pixel statistics stand."""
    lines = makeLines(FakeCalib(), traceRows=0, dy=0.2, species="OH")  # 5 lines on each of 20 fibers
    without = both(speciesOf(lines, NoDispersionMap()))
    assert np.isnan(without.loc["OH", "medianDeltaYKms"])
    assert without.loc["OH", "medianDeltaY"] == pytest.approx(0.2)
    assert residuals(lines, NoDispersionMap())[["dyKmsCalib", "dyKmsVisit"]].isna().all().all()
    # A dispersion for the 520 nm line only: 20 lines, enough, and the median is over those alone.
    partial = both(speciesOf(lines, PartialDispersionMap()))
    assert partial.loc["OH", "nLines"] == 100
    assert partial.loc["OH", "medianDeltaYKms"] == pytest.approx(0.2 * C_KMS * 0.1 / 520.0)
    # On five fibers that is 5 lines with a dispersion, fewer than minLines, though 25 have a dy.
    few = lines[lines["fiberId"] <= 5]
    short = both(speciesOf(few, PartialDispersionMap(numFibers=5)))
    assert short.loc["OH", "nLines"] == 25
    assert np.isnan(short.loc["OH", "medianDeltaYKms"])
    assert short.loc["OH", "medianDeltaY"] == pytest.approx(0.2)
    # Negative control: the same lines with a dispersion throughout are measured.
    whole = both(speciesOf(few, FakeCalib(numFibers=5)))
    assert whole.loc["OH", "medianDeltaYKms"] == pytest.approx(0.2 * C_KMS * 0.1 / 700.0)
    # A table of lines stored without the km/s columns gives NaN, and the pixel statistics as before.
    offsets = residuals(lines, FakeCalib())
    older = both(speciesResiduals(offsets.drop(columns=["dyKmsCalib", "dyKmsVisit"]), 10))
    assert np.isnan(older.loc["OH", "medianDeltaYKms"])
    assert older.loc["OH", "medianDeltaY"] == pytest.approx(0.2)


def testTheGateReadsTheSkyLineResidual():
    """The gate's sky-line check reads ``medianDeltaYKms`` of the held-back OH lines against the visit's map."""
    assert "OH" in SKY_SPECIES
    calib, visit = FakeCalib(), ShiftedMap(dy=-0.3)
    lines, marked = withReserved(makeLines(calib, dy=-0.28, species="OH"))
    full = speciesOf(lines, calib, visit=visit)
    # The held-back OH lines are each fiber's 520 nm line, 0.02 px from the visit's map.
    assert set(lines.loc[marked & (lines["description"] == "OH"), "wavelength"]) == {520.0}
    expected = 0.02 * C_KMS * 0.1 / 520.0  # 1.15 km/s
    assert reserved(full).loc["OH", "medianDeltaYKms"] == pytest.approx(expected)
    dataId = DATA_ID | {"arm": "r"}
    long = longResiduals(full, dataId, "brn", "science")
    for table in (long, longLines(wideResiduals(long, VISIT, (RESERVED,)))):
        read = measurements(lines=table)
        residual = read[read["check"] == "skyLineResidual"]
        assert len(residual) == 1
        assert residual["metric"].iloc[0] == "medianDeltaYKms"
        assert residual["value"].iloc[0] == pytest.approx(expected)
    # Negative control: it is not the offset from the calibration (-0.28 px, -16 km/s), nor the pixel value.
    assert not np.isclose(expected, both(full).loc["OH", "medianDeltaYKms"], rtol=0.5)
    assert not np.isclose(expected, reserved(full).loc["OH", "medianDeltaY"], rtol=0.5)
    # Negative control: on an arc, the gate reads no sky-line residual.
    onArc = measurements(lines=longResiduals(full, dataId, "brn", "arc"))
    assert not (onArc["check"] == "skyLineResidual").any()


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
        "medianDeltaYKms": 8.5,
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
    # A table stored before ``seqType`` was a column: it is pivoted as it is.
    stored = {name: dtype for name, dtype in RESIDUAL_COLUMNS.items() if name != "seqType"}
    metrics = pd.DataFrame(rows).astype(stored)
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
    assert list(drift.columns[8:15]) == ["nLines", "linesPerFiber", *RESIDUAL_METRICS]
    assert len(drift) == 4  # 10 NeI, 10 Trace, 11 NeI, 12 NeI
    assert (drift["obsType"] == "arc").all()  # stored by the task
    trace = drift[(drift["visit"] == 10) & (drift["description"] == TRACE)].iloc[0]
    assert trace["medianDeltaX"] == -0.1 and np.isnan(trace["medianDeltaY"])  # a trace measures x only
    assert np.isnan(trace["medianDeltaYKms"])
    assert (drift.loc[drift["description"] == "NeI", "medianDeltaYKms"] == 8.5).all()
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
    # Negative control: without kinds "unknown" stays, so the frame keeps no type.
    assert (addObsType(metrics.assign(obsType="unknown"), None)["obsType"] == "unknown").all()


def testWideResidualsOfATableStoredEarlier():
    """A table with a ``status`` column and no km/s rows still pivots: the verdict is dropped, km/s is NaN."""
    neon = {"nLines": 100, "linesPerFiber": 2.0, "medianDeltaY": 0.2, "robustRmsDeltaY": 0.06}
    stored = pd.DataFrame(_longRows(10, "b", 1, "brn", CALIB, "NeI", BOTH, neon)).assign(status="PASS")
    wide = wideResiduals(stored, CALIB, (BOTH,))
    assert "status" not in wide.columns
    assert wide.iloc[0]["medianDeltaY"] == 0.2
    assert "medianDeltaYKms" in wide.columns and np.isnan(wide.iloc[0]["medianDeltaYKms"])
    # Negative control: a table that has the rows keeps their values.
    withKms = pd.DataFrame(_longRows(10, "b", 1, "brn", CALIB, "NeI", BOTH, neon | {"medianDeltaYKms": 8.5}))
    assert wideResiduals(withKms, CALIB, (BOTH,)).iloc[0]["medianDeltaYKms"] == 8.5


def testStoredFixtureIsInTheWideForm():
    """The stored fixture's columns are `wideResiduals`'; it may lack the statistic in km/s, added after it."""
    rows = pd.read_parquet(Path(__file__).parent / "data" / "dmDriftMetrics-validation.parquet")
    identifying = ["visit", "arm", "spectrograph", "cameraSet", "obsType", "reference", "description"]
    assert set(identifying) | {"statusType", "nLines", "linesPerFiber"} <= set(rows.columns)
    assert set(RESIDUAL_METRICS) - set(rows.columns) <= {"medianDeltaYKms"}
    assert set(rows["reference"]) == set(REFERENCES)
    assert set(rows.loc[rows["reference"] == VISIT, "statusType"]) == {RESERVED}
    assert set(rows.loc[rows["reference"] == CALIB, "statusType"]) <= {BOTH, UNFITTED}
    # The gate's long form of it has a row per statistic stored.
    long = longLines(rows)
    assert set(long["metric"]) == set(RESIDUAL_METRICS) & set(rows.columns)
    assert len(long) == len(rows) * len(set(long["metric"]))
