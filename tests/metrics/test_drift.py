"""Tests for detector-map drift against the calibration (dmDriftMonitor, PIPE2D-1921)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.drift import (
    BAD_STATUS,
    BOTH,
    DRIFT_COLUMNS,
    RESERVED,
    STATUS_TYPES,
    TRACE,
    USED,
    cameraSet,
    driftOffsets,
    driftThresholds,
    driftVerdict,
    longDrift,
    robustRms,
    speciesDrift,
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


def both(species: pd.DataFrame) -> pd.DataFrame:
    """Return the judged rows of `speciesDrift`, indexed by species."""
    return species[species["statusType"] == BOTH].set_index("description")


def measure(lines, calib, minLines=10, litFibers=None, thresholds=None, camera=None):
    """Return ``(both(species), judged, summary, species)`` for ``lines``."""
    offsets = driftOffsets(lines, calib)
    species = speciesDrift(offsets, minLines, litFibers)
    if thresholds is None:
        thresholds = driftThresholds(0.1, 0.15, 0.1, 0.15)
    numUnfitted = int((offsets["statusType"] == "").sum())
    judged, summary = driftVerdict(species, DATA_ID, thresholds, minLines, camera, numUnfitted=numUnfitted)
    return both(species), judged, summary, species


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
    assert summary["qaDecidedBy"] == f"{TRACE} medianDeltaX"
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
    assert summary["qaDecidedBy"] == "NeI medianDeltaY"
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


def testTraceOnlyFillsXAndDecides():
    calib = FakeCalib()
    lines = makeLines(calib, lampLines=0, dx=0.05)
    species, judged, summary, full = measure(lines, calib)
    assert list(species.index) == [TRACE]
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.05)
    assert np.isnan(species.loc[TRACE, "medianDeltaY"])
    assert np.isnan(species.loc[TRACE, "robustRmsDeltaY"])
    assert summary["qaStatus"] == "PASS"
    long = longDrift(full, judged, DATA_ID)
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
    assert summary["qaDecidedBy"] == "NeI medianDeltaY"


def testBadLinesAreExcluded():
    calib = FakeCalib(numFibers=2)
    lines = makeLines(calib, lampLines=0)
    bad = lines.iloc[:10].assign(x=lines["x"].iloc[:10] + 50.0)
    flagged = bad.assign(flag=True)
    blended = bad.assign(status=0x2)
    offFiber = bad.assign(fiberId=99)  # not in the calibration: getXCenter would raise
    notFinite = bad.assign(x=np.nan)
    offsets = driftOffsets(pd.concat([lines, flagged, blended, offFiber, notFinite]), calib)
    assert len(offsets) == len(lines)
    assert offsets["dx"].abs().max() == pytest.approx(0.0)
    # Negative control: without the status mask, the blended lines get in.
    unmasked = driftOffsets(pd.concat([lines, blended]), calib, badStatus=0)
    assert len(unmasked) > len(lines)
    assert BAD_STATUS & 0x2


def testLinesPerFiberIsPerLitFiber():
    calib = FakeCalib()
    full = both(speciesDrift(driftOffsets(makeLines(calib), calib), 10, set(calib.fiberId.tolist())))
    group = calib.fiberId[::4]
    shaka = both(speciesDrift(driftOffsets(makeLines(calib, fibers=group), calib), 10, set(group.tolist())))
    assert shaka["linesPerFiber"].tolist() == full["linesPerFiber"].tolist() == [5.0, 20.0]
    assert shaka["nLines"].tolist() == [len(group) * 5, len(group) * 20]
    # Negative control: counted per fiber of the detector, one group reads as sparse.
    sparse = both(
        speciesDrift(driftOffsets(makeLines(calib, fibers=group), calib), 10, set(calib.fiberId.tolist()))
    )
    assert (sparse["linesPerFiber"] < full["linesPerFiber"]).all()
    # Without a pfsConfig, or with no fiber lit, there is no per-fiber count.
    assert speciesDrift(driftOffsets(makeLines(calib), calib), 10, None)["linesPerFiber"].isna().all()
    assert speciesDrift(driftOffsets(makeLines(calib), calib), 10, set())["linesPerFiber"].isna().all()


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
    tables = [file, driftThresholds(0.1, 0.15, NAN, NAN)]
    _, judged, withM, _ = measure(lines, calib, thresholds=tables, camera="bmn")
    assert withM["qaStatus"] == "PASS"
    assert judged.loc[judged["metric"] == "medianDeltaX", "population"].iloc[0] == "arm=b cameraSet=bmn"
    # Negative control: read with r, the b camera falls through to the config's thresholds.
    _, _, withR, _ = measure(lines, calib, thresholds=tables, camera="brn")
    assert withR["qaStatus"] == "FAIL"


def testMeasuredButUnthresholdedIsUnjudged():
    calib = FakeCalib()
    lines = makeLines(calib, traceRows=0, dy=5.0)
    _, _, summary, _ = measure(lines, calib, thresholds=driftThresholds(1.0, 2.0, NAN, NAN))
    assert summary["qaStatus"] == "UNKNOWN"
    assert summary["recommendedAction"] == "UNJUDGED"
    assert summary["nSpeciesMeasured"] == 1


def testLongFormat():
    calib = FakeCalib()
    _, judged, _, full = measure(makeLines(calib, dx=0.2), calib, camera="brn")
    long = longDrift(full, judged, DATA_ID, "brn")
    assert list(long.columns) == list(DRIFT_COLUMNS)
    assert long.dtypes.astype(str).tolist() == list(DRIFT_COLUMNS.values())
    trace = long[(long["description"] == TRACE) & (long["statusType"] == BOTH)].set_index("metric")
    assert trace.loc["medianDeltaX", "status"] == "FAIL"
    assert trace.loc["nLines", "status"] == ""
    assert (long["cameraSet"] == "brn").all()
    lamp = long[long["description"] == "NeI"]
    assert set(lamp["metric"]) == {"nLines", "linesPerFiber", "medianDeltaY", "robustRmsDeltaY"}
    assert (long.loc[long["statusType"] != BOTH, "status"] == "").all()
    assert list(long["statusType"].unique()) == list(STATUS_TYPES)
    empty = longDrift(full.iloc[:0], judged.iloc[:0], DATA_ID)
    assert list(empty.columns) == list(DRIFT_COLUMNS) and empty.empty


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


def testRegistry():
    for name in ("medianDeltaX", "medianDeltaY"):
        assert specFor(name).absolute
        assert "cameraSet" in specFor(name).groupBy
    assert not specFor("robustRmsDeltaX").absolute


def testUsedAndReservedAreKeptApart():
    calib = FakeCalib()
    lines = makeLines(calib, lampLines=0)
    reserved = np.arange(len(lines)) % 5 == 0  # a fit holds back about a fifth
    lines.loc[reserved, "status"] = RESERVED_BIT
    lines.loc[reserved, "x"] += 0.3
    _, judged, summary, full = measure(lines, calib)
    trace = full[full["description"] == TRACE].set_index("statusType")
    assert list(trace.index) == list(STATUS_TYPES)
    assert trace.loc[USED, "nLines"] == (~reserved).sum()
    assert trace.loc[RESERVED, "nLines"] == reserved.sum()
    assert trace.loc[BOTH, "nLines"] == len(lines)
    assert trace.loc[USED, "medianDeltaX"] == pytest.approx(0.0)
    assert trace.loc[RESERVED, "medianDeltaX"] == pytest.approx(0.3)
    # Only the union is judged: one verdict per species.
    assert set(judged["statusType"]) == {BOTH}
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


def testNoFitIsInsufficientData():
    calib = FakeCalib()
    _, _, summary, _ = measure(makeLines(calib).assign(status=0), calib)
    assert summary["qaStatus"] == "UNKNOWN"
    assert summary["recommendedAction"] == "INSUFFICIENT_DATA"
    assert "no line marked DETECTORMAP_USED or DETECTORMAP_RESERVED" in summary["qaReason"]


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
    offsets = driftOffsets(lines, calib)
    assert np.isfinite(offsets.loc[offsets["description"] == TRACE, "dx"]).all()
    assert np.isfinite(offsets.loc[offsets["description"] == "NeI", "dy"]).all()
    species, _, summary, _ = measure(lines, calib)
    assert species.loc[TRACE, "medianDeltaX"] == pytest.approx(0.05)
    assert species.loc["NeI", "medianDeltaY"] == pytest.approx(0.05)
    assert species.loc[TRACE, "nLines"] < (lines["description"] == TRACE).sum()
    assert summary["nSpeciesMeasured"] == 2
    # Negative control: the calibration does predict NaN for some of these lines.
    xCenter = calib.getXCenter(lines["fiberId"].to_numpy(), lines["y"].to_numpy())
    assert np.isnan(xCenter[(lines["description"] == TRACE).to_numpy()]).any()
