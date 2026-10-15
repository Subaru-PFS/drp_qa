"""Tests for the cross-dispersion trace width estimator.

Stack-free: numpy, scipy and pandas. The synthetic tests use a row of
pixel-integrated Gaussian traces with Poisson-like noise, at the measured PFS
fiber pitch of 6.17 px unless they say otherwise. ``TestRealQuartz`` reads real
calexp rows made by ``data/makeQuartzFixtures.py``.

The estimator exists because the one it replaces rejected essentially every
sample on quartz frames. ``TestTheDefectItReplaces`` reproduces that failure
on the same fixture, so a change that stops the fixture exercising it is caught.
"""

import functools
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

from pfs.drp.qa.crossDispersion import (
    FWHM_FACTOR,
    IMAGE_WIDTH_COLUMNS,
    PIXEL_VARIANCE,
    measureImageWidths,
    measureRow,
    pixelFwhm,
    profileFwhm,
)

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

    def testFwhmIncludesThePixelResponse(self):
        row, variance, centers = traceRow()
        result = measure(row, variance, centers)
        good = ~result["flag"]
        sigma = result["sigma"][good]
        np.testing.assert_allclose(result["fwhm"][good], FWHM_FACTOR * np.sqrt(sigma**2 + PIXEL_VARIANCE))
        np.testing.assert_allclose(result["fwhmIntrinsic"][good], FWHM_FACTOR * sigma)

    def testFwhmMatchesThePixelRecordedTruth(self):
        """And the intrinsic width, the wrong convention for ``fwhm``, does not.

        At sigma 1.1 px the two conventions differ by 3.4 %, so a 1 % check
        separates them.
        """
        sigma = 1.1
        row, variance, centers = traceRow(sigma=sigma)
        result = measure(row, variance, centers)
        measured = np.median(result["fwhm"][~result["flag"]])
        assert measured == pytest.approx(float(pixelFwhm(sigma)), rel=0.01)
        assert measured != pytest.approx(FWHM_FACTOR * sigma, rel=0.01)

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
        """Each fiber keeps its own result; only the fibers beside the gap can change.

        Those have an unmodelled lit neighbour, so their fits are contaminated.
        """
        row, variance, centers = traceRow()
        withNan = centers.copy()
        withNan[[3, 50]] = np.nan
        result = measure(row, variance, withNan)
        full = measure(row, variance, centers)
        assert len(result["flag"]) == len(centers)
        assert result["flag"][3] and result["flag"][50]
        # Fits use the fibers two either side, whose own centers the gap can move.
        away = np.setdiff1d(np.arange(len(centers)), [*range(7), *range(47, 54)])
        np.testing.assert_array_equal(result["flag"][away], full["flag"][away])
        np.testing.assert_allclose(result["sigma"][away], full["sigma"][away], rtol=1e-6, equal_nan=True)

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

    def testFitsOnlyTheFibersAskedFor(self):
        """The others are neighbours only: flagged, with no values."""
        row, variance, centers = traceRow()
        fit = np.zeros(len(centers), dtype=bool)
        fit[::10] = True
        some = measure(row, variance, centers, fit=fit)
        every = measure(row, variance, centers)
        assert some["flag"][~fit].all() and np.isnan(some["sigma"][~fit]).all()
        # Fitted neighbours' centers are refined against the noise; unfitted ones are not. The
        # difference is well inside the 2.2 % scatter of a single sample.
        np.testing.assert_allclose(some["sigma"][fit], every["sigma"][fit], rtol=0.01)

    def testFittingNoFiberGivesAllFlagged(self):
        row, variance, centers = traceRow()
        result = measure(row, variance, centers, fit=np.zeros(len(centers), dtype=bool))
        assert result["flag"].all()

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
        """A fiber's width barely changes when its neighbours are not fitted.

        They still enter its fit, at their predicted centers rather than their
        refined ones.
        """
        image, variance, bad, xCenters, fiberIds = traceImage(nRows=1)
        lam = np.ones_like(xCenters)
        every = measureImageWidths(image, variance, bad, [0], fiberIds, xCenters, lam)
        chosen = {10, 11, 300}
        subset = measureImageWidths(image, variance, bad, [0], fiberIds, xCenters, lam, select=chosen)
        assert set(subset["fiberId"]) == chosen
        expected = every.set_index("fiberId").loc[sorted(chosen), "fwhm"].to_numpy()
        np.testing.assert_allclose(subset.sort_values("fiberId")["fwhm"], expected, rtol=0.01)

    @pytest.mark.parametrize("shift", [-0.4, 0.3])
    def testDxCenterIsMeasuredMinusCalib(self, shift):
        """Traces displaced by +shift from the calib prediction give dxCenter = +shift, as drp_stella's residuals."""
        image, variance, bad, xCenters, fiberIds = traceImage(nRows=1, shift=shift)
        table = measureImageWidths(
            image, variance, bad, [0], fiberIds, xCenters, np.ones_like(xCenters), calibXCenters=xCenters
        )
        assert table["dxCenter"].median() == pytest.approx(shift, abs=0.02)
        # Negative control: not calibration minus measured, the sign before PIPE2D-1921.
        assert not np.isclose(table["dxCenter"].median(), -shift, atol=0.02)

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


def oversampledProfile(sigma=1.3, radius=10, oversample=3, background=0.0):
    """Return a ``FiberProfile``-like profile: a pixel-integrated Gaussian at sub-pixel offsets."""
    index = (np.arange(2 * (radius + 1) * oversample + 1) - (radius + 1) * oversample) / oversample
    scale = np.sqrt(2.0) * sigma
    profile = 0.5 * (erf((index + 0.5) / scale) - erf((index - 0.5) / scale))
    return index, profile + background * profile.max()


def flatTopped(index, radius=1.6, sigma=0.5):
    """Return a projected uniform disk convolved with a Gaussian, and its true FWHM."""
    fine = np.arange(-15.0, 15.0, 0.001)
    disk = np.sqrt(np.clip(radius**2 - fine**2, 0.0, None))
    profile = np.convolve(disk, np.exp(-0.5 * (fine / sigma) ** 2), mode="same")
    above = fine[profile >= profile.max() / 2]
    return np.interp(index, fine, profile), above.max() - above.min()


class TestProfileFwhm:
    """Reading the half-maximum width of an oversampled profile, with no model."""

    @pytest.mark.parametrize("sigma", [0.9, 1.3, 1.6])
    def testGaussianGivesThePixelRecordedFwhm(self, sigma):
        index, profile = oversampledProfile(sigma)
        assert profileFwhm(index, profile)[0] == pytest.approx(float(pixelFwhm(sigma)), rel=0.01)

    def testReadsTheTrueWidthOfANonGaussianProfile(self):
        """Where a Gaussian fitted to the profile would not."""
        index = np.arange(-33, 34) / 3
        profile, truth = flatTopped(index)
        assert profileFwhm(index, profile)[0] == pytest.approx(truth, rel=0.01)

    def testAgreesWithTheRowFitOnAGaussian(self):
        index, profile = oversampledProfile(1.3)
        row, variance, centers = traceRow(sigma=1.3)
        result = measure(row, variance, centers)
        assert profileFwhm(index, profile)[0] == pytest.approx(
            np.median(result["fwhm"][~result["flag"]]), rel=0.01
        )

    def testOneResultPerRowAndMaskedSamplesIgnored(self):
        index, narrow = oversampledProfile(1.1)
        _, wide = oversampledProfile(1.5)
        profiles = np.ma.masked_array(np.stack([narrow, wide]))
        profiles[0, 5:8] = np.ma.masked
        profiles.data[0, 5:8] = 1e6  # would be the peak if used
        np.testing.assert_allclose(profileFwhm(index, profiles), pixelFwhm([1.1, 1.5]), rtol=0.01)

    @pytest.mark.parametrize(
        "profile",
        [np.ones(3), np.arange(9.0), np.zeros(9), np.r_[np.full(4, 1.5), 2.0, np.full(4, 1.5)]],
        ids=["tooFewSamples", "peakAtTheEnd", "noPeak", "neverFallsToHalf"],
    )
    def testUnmeasurableGivesNan(self, profile):
        assert np.isnan(profileFwhm(np.arange(len(profile), dtype=float), profile)[0])


DATA = Path(__file__).parent / "data"


def loadQuartz(name):
    """Load a fixture written by ``data/makeQuartzFixtures.py``."""
    path = DATA / f"quartz-{name}.npz"
    if not path.exists():
        pytest.skip(f"{path.name} missing; make it with tests/imageQualityQa/data/makeQuartzFixtures.py")
    return dict(np.load(path))


def calibProfile(data):
    """Return the index, the median calibration profile and its FWHM, without placeholder fibers.

    Fibers the calibration had no light for carry drp_stella's placeholder
    Gaussian of FWHM 1.5 px.
    """
    index, profiles = data["calibProfileIndex"].astype(float), data["calibProfiles"].astype(float)
    real = np.abs(profileFwhm(index, profiles) - 1.5) > 0.1
    profile = np.nanmedian(profiles[real], axis=0)
    return index, profile, float(profileFwhm(index, profile)[0])


@functools.cache
def measureQuartz(name):
    """Measure a fixture once; return the joint fit's table and the fixture."""
    data = loadQuartz(name)
    table = measureImageWidths(
        data["image"],
        data["variance"],
        data["bad"],
        data["rows"],
        data["fiberIds"],
        data["xCenters"],
        data["wavelengths"],
    )
    return table, data


def medianFwhm(table):
    return float(table.loc[~table["flag"], "fwhm"].median())


KNOWN_GOOD = ["133040-b2", "133040-r2", "133040-r1"]


class TestRealQuartz:
    """Real quartz calexp rows and the calibration profiles their reduction used.

    Run25 visit 133040 is known-good. Run27 visit 140032 has SM1 defocused and
    is expected to fail ``medFwhm``; it was reduced with a Run26 calibration.
    See ``docs/trace-widths.ipynb`` for the comparison these numbers come from.
    """

    @pytest.mark.parametrize("name", KNOWN_GOOD)
    def testMostSamplesAreUsable(self, name):
        """84-96 % on these eight rows, against the task's 50 % acceptance."""
        table, _ = measureQuartz(name)
        assert (~table["flag"]).mean() > 0.8

    @pytest.mark.parametrize("name", KNOWN_GOOD)
    def testReadsTheCalibrationsHalfMaxWidthOrSlightlyMore(self, name):
        """5-10 % more: about 3 % from the fit, the rest the exposure itself."""
        table, data = measureQuartz(name)
        _, _, calibFwhm = calibProfile(data)
        assert 1.0 < medianFwhm(table) / calibFwhm < 1.15

    @pytest.mark.parametrize("name", KNOWN_GOOD)
    def testTheProfileIsNotGaussian(self, name):
        """Its half-maximum width is 8-10 % more than a Gaussian fitted to it.

        Why ``profileFwhm`` reads the width directly: a Gaussian fit to the
        calibration profile understates the FWHM by that much.
        """
        from scipy.optimize import curve_fit

        index, profile, calibFwhm = calibProfile(loadQuartz(name))
        (_, _, sigma), _ = curve_fit(
            lambda x, a, c, s: a * np.exp(-0.5 * ((x - c) / s) ** 2),
            index,
            profile,
            p0=(profile.max(), 0.0, 1.2),
        )
        assert calibFwhm / (FWHM_FACTOR * abs(sigma)) > 1.05

    def testRowsBuiltFromTheRealProfileReadItsHalfMaxWidth(self):
        """The joint fit, given the calibration's own profile shape, recovers its FWHM."""
        index, profile, calibFwhm = calibProfile(loadQuartz("133040-r2"))
        rng = np.random.default_rng(0)
        centers = 20.0 + PFS_PITCH * np.arange(700) + rng.uniform(-0.3, 0.3, 700)
        centers = centers[centers < N_COLS - 20]
        pixels = np.arange(N_COLS)
        model = np.full(N_COLS, 300.0)
        scale = 20000.0 / (np.sum(profile) * (index[1] - index[0]))
        for center in centers:
            model += scale * np.interp(pixels - center, index, profile, left=0.0, right=0.0)
        variance = model + 25.0
        result = measure(model + rng.normal(0.0, np.sqrt(variance)), variance, centers)
        assert np.median(result["fwhm"][~result["flag"]]) == pytest.approx(calibFwhm, rel=0.05)

    def testTheDefocusedSpectrographReadsWider(self):
        """The same detector, r1, in focus (Run25) and with SM1 defocused (Run27)."""
        good, _ = measureQuartz("133040-r1")
        defocused, _ = measureQuartz("140032-r1")
        assert medianFwhm(defocused) > 1.25 * medianFwhm(good)

    def testTheCalibrationDoesNotSeeTheDefocus(self):
        """Why the exposure is measured: its calibration profile reads like a focused one."""
        _, _, focused = calibProfile(loadQuartz("133040-r1"))
        _, _, defocused = calibProfile(loadQuartz("140032-r1"))
        assert defocused == pytest.approx(focused, rel=0.03)
