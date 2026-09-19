"""Tests for the cross-dispersion trace width estimator.

Stack-free: numpy and scipy only. Every test uses a synthetic row of
pixel-integrated Gaussian traces with Poisson-like noise, at the measured PFS
fiber pitch of 6.17 px unless it says otherwise.

The estimator exists because the one it replaces rejected essentially every
sample on quartz frames. ``TestTheDefectItReplaces`` reproduces that failure
on the same fixture, so a change that stops the fixture exercising it is caught.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

from pfs.drp.qa.crossDispersion import FWHM_FACTOR, IMAGE_WIDTH_COLUMNS, measureImageWidths, measureRow

#: Measured PFS fiber pitch, identical to 0.3 % across b, r and n (Run25).
PFS_PITCH = 6.17

N_COLS = 4096


def traceRow(
    sigma: float = 1.3,
    pitch: float = PFS_PITCH,
    amplitude: float = 20000.0,
    background: float = 300.0,
    lit=None,
    shift: float = 0.0,
    seed: int = 0,
    readNoise: float = 5.0,
):
    """Build one image row of Gaussian fiber traces.

    Parameters
    ----------
    sigma : `float`, optional
        True trace sigma, in pixels.
    pitch : `float`, optional
        Distance between fiber centers, in pixels.
    amplitude : `float`, optional
        Total flux per lit trace.
    background : `float`, optional
        Constant background level.
    lit : array-like of `bool`, optional
        Which fibers are illuminated; default all, as on a quartz frame.
    shift : `float`, optional
        Offset of the true trace centers from the centers returned to the
        caller, as a flexure or detector-map error would produce.
    seed : `int`, optional
        Random seed.
    readNoise : `float`, optional
        Read noise, in the same units as ``amplitude``.

    Returns
    -------
    row, variance, centers : `numpy.ndarray`
        The noisy row, its variance, and the *predicted* centers (without
        ``shift``), as a detector map would supply them.
    """
    rng = np.random.default_rng(seed)
    centers = 20.0 + pitch * np.arange(700) + rng.uniform(-0.3, 0.3, 700)
    centers = centers[centers < N_COLS - 20]
    lit = np.ones(len(centers), dtype=bool) if lit is None else np.asarray(lit)[: len(centers)]
    edges = np.arange(N_COLS + 1) - 0.5
    model = np.full(N_COLS, float(background))
    for center, on in zip(centers, lit, strict=True):
        if on:
            model += amplitude * np.diff(0.5 * (1 + erf((edges - center - shift) / (np.sqrt(2) * sigma))))
    variance = model + readNoise**2
    return model + rng.normal(0.0, np.sqrt(variance)), variance, centers


def measure(row, variance, centers, bad=None, **kwargs):
    bad = np.zeros(N_COLS, dtype=bool) if bad is None else bad
    return measureRow(row, bad, variance, centers, **kwargs)


class TestTheDefectItReplaces:
    """The failure that motivated the rewrite, reproduced on the test fixture."""

    @staticmethod
    def legacyUsableFraction(row, centers, halfWidth=7, minPeakSN=5.0):
        """``_buildImageWidthData``'s old per-sample logic, transcribed."""
        usable = 0
        for center in centers:
            lo, hi = max(0, int(center) - halfWidth), min(N_COLS, int(center) + halfWidth + 1)
            strip = row[lo:hi]
            edge = np.concatenate([strip[:2], strip[-2:]])
            bg, rms = float(np.mean(edge)), float(np.std(edge))
            rms = rms if rms > 0 else max(1.0, np.sqrt(abs(bg)))
            signal = strip - bg
            if signal.max() >= minPeakSN * rms and signal.sum() > 0:
                usable += 1
        return usable / len(centers)

    def testTheOldEstimatorFailsAtThePfsPitch(self):
        """Its 'background' pixels at +/-6..7 px sit on the neighbours at 6.17 px."""
        row, _, centers = traceRow()
        assert self.legacyUsableFraction(row, centers) < 0.01

    def testTheOldEstimatorWorkedWhenFibersWereFarApart(self):
        """Which is why the defect went unnoticed: it is specific to tight pitch."""
        row, _, centers = traceRow(pitch=12.0)
        assert self.legacyUsableFraction(row, centers) > 0.95

    def testTheNewOneMeasuresTheSameRow(self):
        row, variance, centers = traceRow()
        result = measure(row, variance, centers)
        assert (~result["flag"]).mean() > 0.95


class TestWidth:
    @pytest.mark.parametrize("sigma", [0.9, 1.1, 1.3, 1.5])
    def testRecoversTheTrueSigmaAtThePfsPitch(self, sigma):
        """The operating range: PFS traces are FWHM 2.6-3.7 px."""
        row, variance, centers = traceRow(sigma=sigma)
        result = measure(row, variance, centers)
        measured = np.median(result["sigma"][~result["flag"]])
        assert measured == pytest.approx(sigma, rel=0.02)

    def testFwhmIsSigmaScaled(self):
        row, variance, centers = traceRow()
        result = measure(row, variance, centers)
        good = ~result["flag"]
        np.testing.assert_allclose(result["fwhm"][good], FWHM_FACTOR * result["sigma"][good])

    def testScatterIsSmall(self):
        row, variance, centers = traceRow(sigma=1.3)
        result = measure(row, variance, centers)
        sigmas = result["sigma"][~result["flag"]]
        assert np.std(sigmas) / 1.3 < 0.05

    def testBroaderTracesMeasureBroader(self):
        """Monotonic well past the operating range.

        A defocused trace is exactly what QA exists to catch, so the estimator
        must keep reporting larger widths as traces broaden toward the pitch --
        even where it is biased. The saddle-correction iteration first tried
        here diverged in this regime.
        """
        medians = []
        for sigma in (1.0, 1.3, 1.6, 2.0, 2.6):
            row, variance, centers = traceRow(sigma=sigma, seed=11)
            result = measure(row, variance, centers)
            medians.append(np.median(result["sigma"][~result["flag"]]))
        assert all(np.diff(medians) > 0), medians
        assert all(np.isfinite(medians))

    def testIndependentOfTheBackgroundLevel(self):
        low = measure(*traceRow(background=50.0))
        high = measure(*traceRow(background=5000.0))
        assert np.median(low["sigma"][~low["flag"]]) == pytest.approx(
            np.median(high["sigma"][~high["flag"]]), rel=0.02
        )

    @pytest.mark.parametrize("residual", [0.01, 0.02, 0.05])
    def testRobustToResidualBackground(self, residual):
        """Where a second-moment width inflates ~5 % per 1 % of background.

        The constant background is fitted, so a residual one does not widen
        the trace.
        """
        peakPixelFlux = 20000.0 * 0.29
        row, variance, centers = traceRow(background=300.0 + residual * peakPixelFlux)
        result = measure(row, variance, centers)
        assert np.median(result["sigma"][~result["flag"]]) == pytest.approx(1.3, rel=0.02)


class TestPosition:
    @pytest.mark.parametrize("shift", [-0.4, 0.2, 0.6])
    def testRecoversAShiftFromThePredictedCenters(self, shift):
        """The spatial offset QA reports as dxCenter."""
        row, variance, centers = traceRow(shift=shift)
        result = measure(row, variance, centers)
        good = ~result["flag"]
        offset = np.median(result["centroid"][good] - centers[good])
        assert offset == pytest.approx(shift, abs=0.02)

    def testShiftDoesNotBiasTheWidth(self):
        """A center error must be fitted, not absorbed into sigma."""
        row, variance, centers = traceRow(shift=0.5)
        result = measure(row, variance, centers)
        assert np.median(result["sigma"][~result["flag"]]) == pytest.approx(1.3, rel=0.02)


class TestRejection:
    def testDarkFibersAreFlaggedAndLitOnesMeasured(self):
        """Sparse illumination, as on a science frame or an IIS arc."""
        lit = np.arange(700) % 5 == 0
        row, variance, centers = traceRow(lit=lit)
        result = measure(row, variance, centers)
        litHere = lit[: len(centers)]
        assert (~result["flag"][litHere]).mean() > 0.95
        assert result["flag"][~litHere].mean() > 0.95
        assert np.median(result["sigma"][litHere & ~result["flag"]]) == pytest.approx(1.3, rel=0.02)

    def testNoiseOnlyRowIsRejected(self):
        rng = np.random.default_rng(5)
        variance = np.full(N_COLS, 100.0)
        row = 300.0 + rng.normal(0.0, 10.0, N_COLS)
        centers = 20.0 + PFS_PITCH * np.arange(600)
        result = measure(row, variance, centers[centers < N_COLS - 20])
        assert result["flag"].mean() > 0.99

    def testMaskedCenterPixelIsFlagged(self):
        row, variance, centers = traceRow()
        bad = np.zeros(N_COLS, dtype=bool)
        target = 100
        bad[int(np.rint(centers[target]))] = True
        result = measure(row, variance, centers, bad=bad)
        assert result["flag"][target]
        assert not result["flag"][target + 5]

    def testNonFiniteCentersAreFlaggedWithoutShiftingOthers(self):
        row, variance, centers = traceRow()
        withNan = centers.copy()
        withNan[[3, 50]] = np.nan
        result = measure(row, variance, withNan)
        assert len(result["flag"]) == len(centers)
        assert result["flag"][3] and result["flag"][50]
        assert not result["flag"][4]

    def testAWidthOutsideTheSearchRangeIsFlagged(self):
        """A best fit on the edge of the grid means the truth lies outside it."""
        row, variance, centers = traceRow(sigma=1.3)
        result = measure(row, variance, centers, sigmaRange=(2.5, 4.0))
        assert result["flag"].mean() > 0.95


class TestInterface:
    def testInputOrderIsPreserved(self):
        """Detector maps need not list fibers left to right."""
        row, variance, centers = traceRow(shift=0.3)
        rng = np.random.default_rng(9)
        perm = rng.permutation(len(centers))
        shuffled = measure(row, variance, centers[perm])
        ordered = measure(row, variance, centers)
        np.testing.assert_allclose(shuffled["sigma"], ordered["sigma"][perm], equal_nan=True)
        np.testing.assert_allclose(shuffled["centroid"], ordered["centroid"][perm], equal_nan=True)

    def testEveryOutputHasOneEntryPerFiber(self):
        row, variance, centers = traceRow()
        result = measure(row, variance, centers)
        for key, value in result.items():
            assert len(value) == len(centers), key

    def testEmptyInputIsHandled(self):
        row, variance, _ = traceRow()
        result = measure(row, variance, np.array([]))
        assert all(len(value) == 0 for value in result.values())


def traceImage(nRows=3, **kwargs):
    """Stack `traceRow` rows into the arrays `measureImageWidths` takes."""
    rows = [traceRow(seed=seed, **kwargs) for seed in range(nRows)]
    image = np.stack([row for row, _, _ in rows])
    variance = np.stack([var for _, var, _ in rows])
    # One set of centers per row; the seeds jitter them, so keep each row's own.
    xCenters = np.stack([centers for _, _, centers in rows])
    fiberIds = np.arange(1, xCenters.shape[1] + 1, dtype=np.int32)
    return image, variance, np.zeros(image.shape, dtype=bool), xCenters, fiberIds


class TestImageWidths:
    """The detector-level wrapper `imageQualityQa._buildImageWidthData` calls."""

    def testOneSamplePerFiberPerRow(self):
        image, variance, bad, xCenters, fiberIds = traceImage()
        rows = np.array([100.0, 200.0, 300.0])
        table = measureImageWidths(image, variance, bad, rows, fiberIds, xCenters, np.ones_like(xCenters))
        assert list(table.columns) == list(IMAGE_WIDTH_COLUMNS)
        assert len(table) == xCenters.size
        assert sorted(table["y"].unique()) == [100.0, 200.0, 300.0]
        assert not table["traceOnly"].any()
        assert (~table["flag"]).mean() > 0.95

    def testSelectReportsASubsetButMeasuresAgainstEveryNeighbour(self):
        """A fiber's width must not change because its neighbours were not requested."""
        image, variance, bad, xCenters, fiberIds = traceImage(nRows=1)
        lam = np.ones_like(xCenters)
        every = measureImageWidths(image, variance, bad, [0], fiberIds, xCenters, lam)
        chosen = {10, 11, 300}
        subset = measureImageWidths(image, variance, bad, [0], fiberIds, xCenters, lam, select=chosen)
        assert set(subset["fiberId"]) == chosen
        expected = every.set_index("fiberId").loc[sorted(chosen), "fwhm"].to_numpy()
        np.testing.assert_allclose(subset.sort_values("fiberId")["fwhm"], expected)

    @pytest.mark.parametrize("shift", [-0.4, 0.3])
    def testDxCenterIsCalibMinusMeasured(self, shift):
        """Traces displaced by +shift from the calib prediction give dxCenter = -shift."""
        image, variance, bad, xCenters, fiberIds = traceImage(nRows=1, shift=shift)
        table = measureImageWidths(
            image, variance, bad, [0], fiberIds, xCenters, np.ones_like(xCenters), calibXCenters=xCenters
        )
        assert table["dxCenter"].median() == pytest.approx(-shift, abs=0.02)

    def testDxCenterIsNanWithoutACalib(self):
        image, variance, bad, xCenters, fiberIds = traceImage(nRows=1)
        table = measureImageWidths(image, variance, bad, [0], fiberIds, xCenters, np.ones_like(xCenters))
        assert table["dxCenter"].isna().all()

    def testSamplesWithoutAWavelengthAreDropped(self):
        image, variance, bad, xCenters, fiberIds = traceImage(nRows=1)
        lam = np.ones_like(xCenters)
        lam[0, :5] = np.nan
        table = measureImageWidths(image, variance, bad, [0], fiberIds, xCenters, lam)
        assert len(table) == xCenters.shape[1] - 5
        assert not set(fiberIds[:5]) & set(table["fiberId"])

    def testNoRowsGivesAnEmptyTableWithTheSchema(self):
        table = measureImageWidths(
            np.empty((0, N_COLS)), np.empty((0, N_COLS)), np.empty((0, N_COLS), dtype=bool),
            [], [1, 2], np.empty((0, 2)), np.empty((0, 2)),
        )  # fmt: skip
        assert table.empty
        assert list(table.columns) == list(IMAGE_WIDTH_COLUMNS)


DATA = Path(__file__).parent / "data"


class TestRealQuartz:
    """Real Run25 quartz rows (visit 133040); see ``tests/data/README.md``.

    Every 12th exported row, to keep the suite fast. On these frames the old
    estimator found 0.157 % (b2) and 0.004 % (r2) of samples usable.
    """

    @staticmethod
    def measure(detector):
        data = np.load(DATA / f"quartz_133040_{detector}.npz")
        rows = slice(None, None, 12)
        table = measureImageWidths(
            data["image"][rows],
            data["variance"][rows],
            data["bad"][rows],
            data["rows"][rows],
            data["fiberIds"],
            data["xCenters"][rows],
            np.ones_like(data["xCenters"][rows]),
        )
        calibFwhm = FWHM_FACTOR * float(np.nanmedian(data["calibWidth"]))
        return table, calibFwhm

    @pytest.mark.parametrize("detector", ["b2", "r2"])
    def testMostSamplesAreUsable(self, detector):
        """Well clear of the task's ``maxCalexpFlagRate`` acceptance (50 % usable).

        All 84 rows give 92.0 % (b2) and 95.7 % (r2); this subsample, b2 83 %.
        """
        table, _ = self.measure(detector)
        assert (~table["flag"]).mean() > 0.75

    def testB2AgreesWithItsCalibration(self):
        table, calibFwhm = self.measure("b2")
        assert table.loc[~table["flag"], "fwhm"].median() == pytest.approx(calibFwhm, rel=0.02)

    def testR2IsInsideTheExpectedRange(self):
        """Only a range: r2 reads ~10 % wider than its calib, which is unexplained.

        Do not tighten this to the calib value until that is resolved; see
        ``doc/qa-rebuild-plan.md``.
        """
        table, _ = self.measure("r2")
        assert 2.6 < table.loc[~table["flag"], "fwhm"].median() < 3.7
