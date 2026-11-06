"""Tests for the twilight sequence check (PIPE2D-1973): synthetic sequences, then the stored metrics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.calibration import readThresholds
from pfs.drp.qa.metrics.gate import THRESHOLDS_COLUMNS, thresholdsPath
from pfs.drp.qa.metrics.registry import specFor
from pfs.drp.qa.metrics.twilight import (
    IMAGE_VERDICT_COLUMNS,
    METRIC_COLUMNS,
    TWILIGHT_THRESHOLDS_FILE,
    VERDICT_COLUMNS,
    WINDOWS,
    addPositions,
    fiberFlux,
    isTwilightSequence,
    sequenceConsistency,
    sequenceReadiness,
    twilightImageVerdict,
    twilightThresholds,
    twilightVerdict,
)
from pfs.drp.qa.metrics.validationVisits import loadValidationVisits

DATA = Path(__file__).parent / "data"
SEQUENCE = 4242
N_GOOD = 3000
MEDIAN_ERROR = np.sqrt(np.pi / 2)


def makeSequence(
    snr=(20.0, 30.0, 45.0, 70.0),
    arms=("b", "r"),
    nGood=N_GOOD,
    fibersPerDetector=150,
    skyGradient=True,
    seed=1,
    step=None,
    patch=None,
    staticPatch=None,
) -> pd.DataFrame:
    """Return ``twilightFiberFlux`` rows of a synthetic twilight sequence.

    Each fiber has a fixed throughput per arm, the sky a smooth gradient that changes
    between exposures (a uniform sky without ``skyGradient``: its change has a cubic term the quadratic
    leaves), and each fiber's median the photon noise of its S/N (b's; r's is twice). ``step``
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
        if not skyGradient:
            sky = np.ones(nFibers)
        if patch is not None and patch[0] == ii:
            sky = sky * (1 - patch[1] * patchShape)
        if staticPatch is not None:
            sky = sky * (1 - staticPatch * patchShape)
        for arm in arms:
            armSnr = bSnr * (2 if arm == "r" else 1)
            level = 100.0 * (ii + 1)
            noise = rng.normal(0, MEDIAN_ERROR / (armSnr * np.sqrt(nGood)), nFibers)
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
                        "nGood": nGood,
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
    """A 10 % patch in one exposure fails, and the arm mean carries it, not the arms' difference."""
    fibers = makeSequence(patch=(2, 0.10))
    metrics = sequenceConsistency(fibers)
    exposure = metrics[metrics["visit"] == 1002].iloc[0]
    assert exposure["twilightPatch"] > 4.0
    assert exposure["twilightArmMean"] > 3 * exposure["twilightArmHalfDiff"]
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
    # A faint exposure is no reference: the brighter three's is their own median.
    assert (metrics.loc[metrics["visit"] != 1000, "nReference"] == 3).all()


def testOneBadExposureOfThreeIsNamedAlone():
    """With three bright exposures, a patch in one fails it, and its neighbours pass.

    The reference includes the judged exposure: against the median of the other two, which is their
    mean, the patch shows in both neighbours at half its depth.
    """
    fibers = makeSequence(snr=(30.0, 45.0, 70.0), patch=(2, 0.10))
    metrics = sequenceConsistency(fibers).drop_duplicates(["visit", "arm"]).set_index(["visit", "arm"])
    assert metrics.loc[(1002, "b"), "twilightPatch"] > 4.0
    assert metrics.drop(1002, level="visit")["twilightPatch"].max() < 1.0
    verdict = verdictOf(fibers)["status"]
    assert (verdict.xs(1002, level="visit") == "FAIL").all()
    assert (verdict.drop(1002, level="visit") == "PASS").all()
    # Negative control: against the mean of the other two, a clean neighbour shows the patch at half depth.
    wide = fibers[fibers["arm"] == "b"].pivot_table(index="fiberId", columns="visit", values="flux")
    wide = wide / wide.median()
    mirrored = wide[1000] / wide[[1001, 1002]].mean(axis=1)
    assert mirrored.min() < 0.97 or mirrored.max() > 1.03


def testNoiseIsSubtracted():
    """Fiber noise alone, about 1 % per fiber, leaves little excess RMS though the noise is 0.25-0.9 %.

    A robust RMS of 2400 fibers is good to about 3 %, which subtracting in quadrature turns into an excess
    of up to about a third of the noise.
    """
    for seed in (1, 2, 3):
        fibers = makeSequence(
            snr=(16.0, 20.0, 24.0, 30.0), nGood=40, seed=seed, fibersPerDetector=600, skyGradient=False
        )
        metrics = sequenceConsistency(fibers)
        assert (metrics["noise"] > 0.2).all()
        assert (metrics["twilightResidRms"] < 0.4 * metrics["noise"]).all(), seed
    # Negative control: the noise is there before subtraction; leaving it in would fail the bound.
    unsubtracted = np.hypot(metrics["twilightResidRms"], metrics["noise"])
    assert not (unsubtracted < 0.4 * metrics["noise"]).any()


def testDeadDetectorFails():
    """A detector reading zero, or with no measured fiber, in one exposure fails there rather than vanishing."""
    for value, expected in ((0.0, -100.0), (np.nan, np.inf)):
        fibers = makeSequence()
        dead = (fibers["visit"] == 1002) & (fibers["arm"] == "b") & (fibers["spectrograph"] == 3)
        fibers.loc[dead, "flux"] = value
        metrics = sequenceConsistency(fibers).set_index(["visit", "arm", "spectrograph"]).sort_index()
        # Negative control: the row is there; dropping such fibers would leave the arm judged on b1, b2, b4.
        assert (1002, "b", 3) in metrics.index
        assert metrics.loc[(1002, "b", 3), "twilightLevelOffset"] == pytest.approx(expected)
        if np.isnan(value):
            assert metrics.loc[(1002, "b", 3), "nMeasured"] == 0
        assert verdictOf(fibers).loc[(1002, "b"), "status"] == "FAIL"


def testCameraWithTooFewExposuresIsUnjudged():
    """A camera in only one exposure has no reference: its level is NaN, not a pass against itself."""
    fibers = makeSequence(step=(2, "r", 4, 0.88))
    alone = (fibers["arm"] == "r") & (fibers["spectrograph"] == 4) & (fibers["visit"] != 1002)
    fibers = fibers[~alone]
    metrics = sequenceConsistency(fibers).set_index(["visit", "arm", "spectrograph"]).sort_index()
    level = metrics.loc[(1002, "r", 4), "twilightLevelOffset"]
    # Negative control: a median over its one sample would be the exposure itself, 0 %.
    assert np.isnan(level) and level != pytest.approx(0.0)
    assert metrics.drop((1002, "r", 4))["twilightLevelOffset"].abs().max() < 1.0


def testFaintArmLeftUnjudgedStaysEmpty():
    """TOO_FAINT replaces a PASS; an arm the thresholds leave out keeps no verdict, faint or not."""
    verdict = verdictOf(makeSequence(snr=(5.0, 20.0, 30.0, 45.0), arms=("b", "n")))["status"]
    assert verdict[(1000, "n")] == ""
    # Negative control: the judged arm of the same faint exposure is TOO_FAINT.
    assert verdict[(1000, "b")] == "TOO_FAINT"


def testShortSequenceIsNotMeasured():
    assert sequenceConsistency(makeSequence(snr=(20.0, 30.0))).empty
    # Nothing measured gives a verdict table with its columns, and no rows.
    assert list(twilightVerdict(sequenceConsistency(makeSequence(snr=(20.0, 30.0))))) == list(VERDICT_COLUMNS)


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


def opdbSequence(visits=(150840, 150841, 150842), name="Twilight sky", finished="2026-09-16 05:40:51"):
    """Rows as `readSequenceOfVisit` returns them."""
    return pd.DataFrame(
        {
            "pfs_visit_id": list(visits),
            "iic_sequence_id": 69251,
            "sequence_type": "scienceObject",
            "sequence_name": name,
            "finished_at": pd.to_datetime(finished),
            "sequence_status": "complete" if finished else None,
        }
    )


class TestSequenceReadiness:
    """Judge a sequence only once it is a finished twilight with every visit's tables stored."""

    CAMERAS = frozenset({"b1", "r1", "b2", "r2"})

    def everything(self, visits=(150840, 150841, 150842)):
        return {visit: set(self.CAMERAS) for visit in visits}

    def testReadyWhenEveryVisitIsStored(self):
        ready, why = sequenceReadiness(opdbSequence(), self.everything(), self.everything())
        assert ready and "3 visits, all stored" in why

    def testWaitsForAVisitNotYetReduced(self):
        stored = self.everything()
        del stored[150842]
        reduced = self.everything()
        del reduced[150842]  # not reduced either: no pfsArm, no table
        ready, why = sequenceReadiness(opdbSequence(), reduced, stored)
        assert not ready and "150842 (no tables yet)" in why
        # Negative control: counting only the pfsArms present would call it ready.
        assert all(reduced[visit] <= stored[visit] for visit in reduced)

    def testWaitsForAMissingCamera(self):
        stored = self.everything()
        stored[150841] = stored[150841] - {"r2"}
        ready, why = sequenceReadiness(opdbSequence(), self.everything(), stored)
        assert not ready and "150841 (r2)" in why

    def testACameraNeverReducedIsNotWaitedFor(self):
        reduced = self.everything()
        reduced[150841] = reduced[150841] - {"r2"}  # no pfsArm, so no table is expected
        stored = {visit: set(cameras) for visit, cameras in reduced.items()}
        assert sequenceReadiness(opdbSequence(), reduced, stored)[0]

    def testWaitsForTheSequenceToFinish(self):
        ready, why = sequenceReadiness(opdbSequence(finished=None), self.everything(), self.everything())
        assert not ready and "has not finished" in why

    def testIgnoresOtherSequences(self):
        ready, why = sequenceReadiness(opdbSequence(name="Arc: HgCd"), self.everything(), self.everything())
        assert not ready and "is not a twilight" in why
        assert not sequenceReadiness(opdbSequence(visits=()), {}, {})[0]


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
        """Besides the Run21 faults, only Run29's 145073 b fails.

        A patch at x=-173, y=-28 mm, on SM1 and SM2 fibers alike, +10 % against the rest of its sequence,
        whose other two bright exposures agree within 0.2 %.
        """
        failed = set(map(tuple, verdict.loc[verdict["status"] == "FAIL", ["visit", "arm"]].to_numpy()))
        run21 = {(visit, "b") for visit in (121949, 121950, 121951, 121952, 123320)}
        run29 = {(145073, "b")}
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


def testShippedThresholdsAreTheFunctions():
    """``twilightThresholds.yaml`` (what the judge records) holds `twilightThresholds` (what it means)."""
    _, shipped = readThresholds(thresholdsPath(TWILIGHT_THRESHOLDS_FILE))
    table = twilightThresholds()
    assert len(shipped) == len(table)
    for column in ("metric", "warn", "fail", "higherIsWorse", "absolute"):
        assert list(shipped[column].fillna(-1)) == list(table[column].fillna(-1)), column
    assert [None if pd.isna(a) else a for a in shipped["arm"]] == [
        None if pd.isna(a) else a for a in table["arm"]
    ]
    # Negative control: a changed threshold would be caught.
    assert list(shipped["warn"].fillna(-1)) != list(twilightThresholds(levelWarn=9.0)["warn"].fillna(-1))


def testIsTwilightSequence():
    assert isTwilightSequence(opdbSequence())
    assert not isTwilightSequence(opdbSequence(name="Arc: HgCd"))
    assert not isTwilightSequence(opdbSequence(visits=()))


class TestImageVerdict:
    """One verdict per detector image, for the gate: the sequence verdict seen from each image."""

    @pytest.fixture(scope="class")
    def metrics(self):
        return pd.read_parquet(DATA / "twilightMetrics.parquet")

    @pytest.fixture(scope="class")
    def images(self, metrics):
        return twilightImageVerdict(metrics)

    def testColumnsAndRecord(self, images, metrics):
        assert list(images.columns) == list(IMAGE_VERDICT_COLUMNS)
        assert len(images) == len(metrics)
        assert set(images["thresholdsFiles"]) == {TWILIGHT_THRESHOLDS_FILE}
        assert images["thresholdsSha256"].str.len().eq(64).all()
        assert set(THRESHOLDS_COLUMNS) <= set(images.columns)

    def testTheNArmIsUnknown(self, images):
        assert set(images.loc[images["arm"] == "n", "qaStatus"]) == {"UNKNOWN"}
        assert set(images.loc[images["arm"] != "n", "qaStatus"]) <= {"PASS", "WARN", "FAIL"}

    def testWorstImageIsTheArmVerdict(self, images, metrics):
        """An arm's verdict is the worst of its images', TOO_FAINT read as PASS."""
        arm = twilightVerdict(metrics).set_index(["visit", "arm"])["status"]
        order = {"UNKNOWN": -1, "PASS": 0, "WARN": 1, "FAIL": 2}
        worst = images.groupby(["visit", "arm"])["qaStatus"].agg(lambda s: max(s, key=order.get))
        judged = arm[arm != ""]
        assert len(judged) and (arm == "TOO_FAINT").any()  # the mapping is exercised
        assert (worst[judged.index] == judged.replace("TOO_FAINT", "PASS")).all()

    def testRun21StepsFailTheSteppedDetectorAlone(self, images):
        """b1's level step (121949-121952) and b4's (123320) fail those detectors, not their arm's others."""
        run21 = images[
            images["visit"].isin([121949, 121950, 121951, 121952, 123320]) & (images["arm"] == "b")
        ]
        failed = run21[run21["qaStatus"] == "FAIL"]
        assert set(failed["qaDecidedBy"]) == {"twilightLevelOffset"}
        assert set(zip(failed["visit"], failed["spectrograph"], strict=True)) == {
            (121949, 1), (121950, 1), (121951, 1), (121952, 1), (123320, 4)
        }  # fmt: skip
        # Negative control: the other detectors of those arms are not failed.
        assert (run21.loc[~run21.index.isin(failed.index), "qaStatus"] != "FAIL").all()

    def testEmpty(self):
        empty = twilightImageVerdict(pd.DataFrame(columns=list(METRIC_COLUMNS)))
        assert empty.empty and list(empty.columns) == list(IMAGE_VERDICT_COLUMNS)

    def testOwnThresholdsRecordNothing(self, metrics):
        images = twilightImageVerdict(metrics.iloc[:12], twilightThresholds())
        assert set(images["thresholdsFiles"]) == {""} and set(images["thresholdsSha256"]) == {""}

    def testAFileGivenByPathIsRecorded(self, metrics, tmp_path):
        """The shipped file by its path records as the default does; another file by its path."""
        default = twilightImageVerdict(metrics.iloc[:12])
        explicit = twilightImageVerdict(metrics.iloc[:12], thresholdsPath(TWILIGHT_THRESHOLDS_FILE))
        for column in THRESHOLDS_COLUMNS:
            assert set(explicit[column]) == set(default[column]) == {default[column].iloc[0]}
        assert default["thresholdsFiles"].iloc[0] == TWILIGHT_THRESHOLDS_FILE
        copy = tmp_path / "myThresholds.yaml"
        copy.write_bytes(thresholdsPath(TWILIGHT_THRESHOLDS_FILE).read_bytes())
        other = twilightImageVerdict(metrics.iloc[:12], copy)
        assert set(other["thresholdsFiles"]) == {str(copy)}
        assert set(other["thresholdsSha256"]) == set(default["thresholdsSha256"])  # same content
