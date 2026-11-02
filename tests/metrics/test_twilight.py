"""Tests for the twilight sequence check (PIPE2D-1973): synthetic sequences, then the stored metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.registry import specFor
from pfs.drp.qa.metrics.twilight import (
    METRIC_COLUMNS,
    WINDOWS,
    addPositions,
    fiberFlux,
    sequenceConsistency,
    twilightThresholds,
    twilightVerdict,
)
from pfs.drp.qa.metrics.validationVisits import loadValidationVisits

DATA = Path(__file__).parent / "data"
SEQUENCE = 4242
N_GOOD = 3000


def makeSequence(
    snr=(20.0, 30.0, 45.0, 70.0),
    arms=("b", "r"),
    fibersPerDetector=150,
    seed=1,
    step=None,
    patch=None,
    staticPatch=None,
) -> pd.DataFrame:
    """Return ``twilightFiberFlux`` rows of a synthetic twilight sequence.

    Each fiber has a fixed throughput per arm (the instrument), the sky a smooth gradient that changes
    between exposures, and each fiber's median the photon noise of its S/N (b's; r's is twice). ``step``
    is ``(exposure index, arm, spectrograph, factor)``; ``patch`` is ``(exposure index, depth)``: a
    Gaussian dimming of sigma 45 mm in both arms of that exposure; ``staticPatch`` the same depth in
    every exposure.
    """
    rng = np.random.default_rng(seed)
    nFibers = 4 * fibersPerDetector
    fiberId = np.arange(1, nFibers + 1)
    spectrograph = np.repeat([1, 2, 3, 4], fibersPerDetector)
    angle, radius = rng.uniform(0, 2 * np.pi, nFibers), 220 * np.sqrt(rng.uniform(0, 1, nFibers))
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    throughput = {arm: rng.normal(1, 0.05, nFibers) for arm in arms}
    patchShape = np.exp(-(((x - 60) ** 2 + (y - 40) ** 2) / (2 * 45**2)))
    rows = []
    for ii, bSnr in enumerate(snr):
        visit = 1000 + ii
        sky = (1 + 0.002 * ii * x / 100 + 0.001 * y / 100) * (1 + 0.0005 * ii * (x / 100) ** 2)
        if patch is not None and patch[0] == ii:
            sky = sky * (1 - patch[1] * patchShape)
        if staticPatch is not None:
            sky = sky * (1 - staticPatch * patchShape)
        for arm in arms:
            armSnr = bSnr * (2 if arm == "r" else 1)
            level = 100.0 * (ii + 1)
            noise = rng.normal(0, 1 / (armSnr * np.sqrt(N_GOOD)), nFibers)
            flux = level * throughput[arm] * sky * (1 + noise)
            if step is not None and step[0] == ii and step[1] == arm:
                flux = np.where(spectrograph == step[2], flux * step[3], flux)
            rows.append(
                pd.DataFrame(
                    {
                        "visit": visit,
                        "arm": arm,
                        "spectrograph": spectrograph,
                        "sequence": SEQUENCE,
                        "seqName": "Twilight sky",
                        "fiberId": fiberId,
                        "x": x,
                        "y": y,
                        "flux": flux,
                        "snr": armSnr,
                        "nGood": N_GOOD,
                        "satFraction": 0.0,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def verdictOf(fibers: pd.DataFrame) -> pd.DataFrame:
    return twilightVerdict(sequenceConsistency(fibers)).set_index(["visit", "arm"])


def testFiberFluxUsesTheWindowAndGoodPixels():
    wavelength = np.tile(np.linspace(400, 700, 7), (2, 1))  # 450-650 holds columns 1-5
    flux = np.array([[9.0, 1.0, 2.0, 3.0, 4.0, 100.0, 9.0], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])
    mask = np.zeros_like(flux, dtype=int)
    mask[0, 5] = 4  # SAT
    table = fiberFlux(
        np.array([7, 8]),
        wavelength,
        flux,
        np.ones_like(flux),
        np.ones_like(flux),
        mask,
        WINDOWS["b"],
        satBit=4,
    )
    assert table["flux"].tolist() == [2.5, 5.0]  # median of 1, 2, 3, 4: the saturated pixel is left out
    assert table["nGood"].tolist() == [4, 5]
    assert table["satFraction"].tolist() == [0.2, 0.0]
    # Negative control: without the mask the saturated pixel would move the median.
    unmasked = fiberFlux(
        np.array([7, 8]), wavelength, flux, np.ones_like(flux), np.ones_like(flux), 0 * mask, WINDOWS["b"]
    )
    assert unmasked["flux"].iloc[0] != table["flux"].iloc[0]


def testPositionsJoinFromBigEndianFits():
    """pfsConfig.pfiCenter read from FITS is big-endian; the join must still work (it failed on the stack)."""
    table = pd.DataFrame({"fiberId": [3, 1], "flux": [1.0, 2.0]})
    fiberId = np.array([1, 2, 3], dtype=">i4")
    center = np.array([[10.0, -1.0], [20.0, -2.0], [30.0, -3.0]], dtype=">f8")
    joined = addPositions(table, fiberId, center)
    assert joined["x"].tolist() == [30.0, 10.0] and joined["y"].tolist() == [-3.0, -1.0]
    # Negative control: joining on the big-endian arrays as they come is what failed.
    raw = pd.DataFrame({"fiberId": fiberId, "x": center[:, 0], "y": center[:, 1]})
    with pytest.raises(ValueError, match=r"[Bb]ig-endian"):
        table.merge(raw, on="fiberId", how="left")


def testCleanSequencePasses():
    """Fixed throughputs and a changing smooth sky: every exposure is a good flat."""
    metrics = sequenceConsistency(makeSequence())
    assert list(metrics.columns) == list(METRIC_COLUMNS)
    assert metrics["twilightLevelOffset"].abs().max() < 0.2
    assert metrics["twilightResidRms"].max() < 0.2
    assert (verdictOf(makeSequence())["status"] == "PASS").all()


def testDetectorStepIsNamedInItsExposureAlone():
    """A detector 12 % low in one exposure fails there, and nowhere else."""
    fibers = makeSequence(step=(2, "b", 4, 0.88))
    metrics = sequenceConsistency(fibers).set_index(["visit", "arm", "spectrograph"]).sort_index()
    offset = metrics["twilightLevelOffset"]
    assert offset[(1002, "b", 4)] == pytest.approx(-12, abs=1.5)
    # Negative control: the sign follows the detector, low is negative.
    assert not offset[(1002, "b", 4)] > 0
    others = offset.drop((1002, "b", 4))
    assert others.abs().max() < 1.0
    verdict = verdictOf(fibers)["status"]
    assert verdict[(1002, "b")] == "FAIL"
    assert (verdict.drop((1002, "b")) == "PASS").all()
    # The detector's own median removes the step before the RMS: it is not counted twice.
    assert metrics.loc[(1002, "b"), "twilightResidRms"].max() < 0.2


def testPassingCloudFailsThePatchAndIsShared():
    """A 10 % patch in one exposure fails, and shows in both arms: the sky."""
    fibers = makeSequence(patch=(2, 0.10))
    metrics = sequenceConsistency(fibers)
    exposure = metrics[metrics["visit"] == 1002].iloc[0]
    assert exposure["twilightPatch"] > 4.0
    assert exposure["twilightShared"] > 3 * exposure["twilightDiffering"]
    assert verdictOf(fibers).loc[(1002, "b"), "status"] == "FAIL"
    # The robust RMS ignores a patch on a few per cent of the fibers: that is why the patch is measured.
    assert exposure["twilightResidRms"] < 1.5
    assert metrics.loc[metrics["visit"] != 1002, "twilightPatch"].max() < 2.5


def testStationaryPatchIsNotSeen():
    """The blind spot, pinned: a patch in every exposure cancels against the sequence."""
    assert (verdictOf(makeSequence(staticPatch=0.10))["status"] == "PASS").all()
    # Negative control: the same patch in one exposure is seen.
    assert (verdictOf(makeSequence(patch=(2, 0.10)))["status"] == "FAIL").any()


def testFaintExposureIsTooFaintButItsLevelIsJudged():
    fibers = makeSequence(snr=(5.0, 20.0, 30.0, 45.0), step=(0, "b", 3, 0.85))
    metrics = sequenceConsistency(fibers)
    faint = metrics[metrics["visit"] == 1000]
    assert faint["tooFaint"].all() and faint["twilightResidRms"].isna().all()
    assert faint.set_index(["arm", "spectrograph"]).loc[("b", 3), "twilightLevelOffset"] < -10
    verdict = verdictOf(fibers)["status"]
    assert verdict[(1000, "b")] == "FAIL"  # its level step still fails it
    assert verdict[(1000, "r")] == "TOO_FAINT"
    # A faint exposure is no reference: the brighter three each have two.
    assert (metrics.loc[metrics["visit"] != 1000, "nReference"] == 2).all()


def testShortSequenceIsNotMeasured():
    assert sequenceConsistency(makeSequence(snr=(20.0, 30.0))).empty


def testNArmIsNotJudged():
    fibers = makeSequence(arms=("b", "n"), step=(1, "n", 2, 0.8))
    verdict = verdictOf(fibers)["status"]
    assert (verdict.xs("n", level="arm") == "").all()
    # Negative control: the same step in b fails.
    assert verdictOf(makeSequence(arms=("b", "n"), step=(1, "b", 2, 0.8)))["status"][(1001, "b")] == "FAIL"


def testThresholdsAgreeWithTheRegistry():
    thresholds = twilightThresholds()
    for _, entry in thresholds.iterrows():
        spec = specFor(entry["metric"])
        assert entry["higherIsWorse"] == spec.higherIsWorse
        assert entry["absolute"] == spec.absolute
    judged = thresholds[thresholds["arm"].isna()]
    assert (judged["warn"] < judged["fail"]).all()


class TestStoredMetrics:
    """The 16 twilight sequences of 2025-03 to 2026-09, through ``makeTwilightFixture.py``."""

    @pytest.fixture(scope="class")
    def verdict(self) -> pd.DataFrame:
        return twilightVerdict(pd.read_parquet(DATA / "twilightMetrics.parquet"))

    def testKnownBadDetectorsFail(self, verdict):
        statuses = verdict.set_index(["visit", "arm"])["status"]
        for visit in (121949, 121950, 121951, 121952, 123320):
            assert statuses[(visit, "b")] == "FAIL", visit

    def testOddExposureIsNamedAlone(self):
        metrics = pd.read_parquet(DATA / "twilightMetrics.parquet")
        b4 = metrics[
            (metrics["sequence"] == 44625) & (metrics["arm"] == "b") & (metrics["spectrograph"] == 4)
        ]
        offsets = b4.set_index("visit")["twilightLevelOffset"]
        assert offsets[123320] < -10 and offsets.drop(123320).abs().max() < 2

    def testOnlyTheKnownFaultsAndTheRun29PatchFail(self, verdict):
        """Besides the Run21 faults, only Run29's b arm fails.

        A patch at x=-173, y=-28 mm, on SM1 and SM2 fibers alike: +11 %, -5 % and -5 % against the other
        exposures (r within 1.4 %).
        """
        failed = set(map(tuple, verdict.loc[verdict["status"] == "FAIL", ["visit", "arm"]].to_numpy()))
        run21 = {(visit, "b") for visit in (121949, 121950, 121951, 121952, 123320)}
        run29 = {(visit, "b") for visit in (145073, 145074, 145075)}
        assert failed == run21 | run29
        assert (verdict["status"] == "WARN").sum() <= 2

    def testKnownGoodTwilightsAreNotFailed(self, verdict):
        visitSet = loadValidationVisits()
        for _, row in verdict[verdict["arm"].isin(["b", "r"])].iterrows():
            if visitSet.expectationFor(row["visit"], arm=row["arm"], spectrograph=1) == "PASS":
                assert row["status"] in ("PASS", "TOO_FAINT"), (row["visit"], row["arm"], row["reason"])

    def testCloudyTwilightPasses(self, verdict):
        """The obslog's cloudy twilight is a good flat by this check (its entry is unconfirmed)."""
        cloudy = verdict[verdict["visit"].between(134334, 134337) & verdict["arm"].isin(["b", "r"])]
        assert set(cloudy["status"]) <= {"PASS", "TOO_FAINT"}
