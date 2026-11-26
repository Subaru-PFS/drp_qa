"""Statistics of a ``fiberProfiles`` calibration, on synthetic profiles and on two real calibrations."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import erf

from pfs.drp.qa.crossDispersion import pixelFwhm
from pfs.drp.qa.metrics.fiberProfiles import (
    FIBER_COLUMNS,
    FIBER_TABLE_COLUMNS,
    METRIC_COLUMNS,
    METRICS,
    PLACEHOLDER_FWHM,
    SWATH_COLUMNS,
    compareFibers,
    coverage,
    detectorMetrics,
    fiberTable,
    findReplaced,
    iterProfiles,
    measureProfiles,
    neighbourResidual,
    profileQa,
    summarizeFibers,
    swathStats,
)

#: Intrinsic sigma of a PFS trace core, in pixels.
SIGMA = 1.25
RADIUS = 10
OVERSAMPLE = 5


def profileIndex(radius=RADIUS, oversample=OVERSAMPLE):
    """Return the sample offsets as drp_stella's ``FiberProfile`` builds them."""
    center = int((radius + 1) * oversample + 0.5)
    return (np.arange(2 * center + 1) - center) / oversample


def gaussian(index, sigma=SIGMA, shift=0.0):
    """Return a pixel-integrated Gaussian of unit flux at sub-pixel offsets."""
    scale = np.sqrt(2.0) * sigma
    return 0.5 * (erf((index - shift + 0.5) / scale) - erf((index - shift - 0.5) / scale))


def plateau(index, level, inner=4.0, outer=9.0):
    """Return light left by a neighbouring fiber: ``level`` between two offsets on one side."""
    return np.where((index >= inner) & (index <= outer), level, 0.0)


def secondMoment(index, profile):
    """Return the whole-window second moment drp_stella's ``calculateStatistics().width`` is."""
    centroid = (index * profile).sum() / profile.sum()
    return np.sqrt(((index - centroid) ** 2 * profile).sum() / profile.sum())


def one(stats, name):
    """Return the single swath's value of a statistic."""
    return float(stats[name][0])


class TestSwathStats:
    """One fiber's swaths."""

    def testCleanGaussian(self):
        index = profileIndex()
        stats = swathStats(index, gaussian(index))
        assert set(stats) == set(SWATH_COLUMNS) - {"fiberId", "row"}
        assert one(stats, "fwhm") == pytest.approx(float(pixelFwhm(SIGMA)), rel=0.01)
        assert one(stats, "centroid") == pytest.approx(0.0, abs=1e-6)
        assert 0 < one(stats, "wingFrac") < 0.003
        assert one(stats, "wingMax") < 0.01
        assert one(stats, "minimum") >= 0
        assert one(stats, "variation") == pytest.approx(0.0, abs=1e-12)
        assert one(stats, "maskedFrac") == 0

    def testStatisticsDoNotDependOnTheNormalisation(self):
        index = profileIndex()
        profile = gaussian(index) + plateau(index, 0.01)
        unit, scaled = swathStats(index, profile), swathStats(index, 37.0 * profile)
        for name in unit:
            np.testing.assert_allclose(scaled[name], unit[name], rtol=1e-12, atol=1e-15)

    def testNeighbourLightShowsInTheWingsAndNotInTheWidth(self):
        """A plateau 4-9 px out at 18 % of the peak, as a broken cobra's light leaves."""
        index = profileIndex()
        clean = gaussian(index)
        level = 0.18 * clean.max()
        left = swathStats(index, clean + plateau(index, level))
        core = one(swathStats(index, clean), "wingMax")  # the core's own light at the wing radius

        assert one(left, "wingMax") == pytest.approx(0.18 + core, rel=0.01)
        # Negative control: wingMax is relative to the peak, not to the profile's sum.
        assert not np.isclose(one(left, "wingMax"), level / (clean + plateau(index, level)).sum(), rtol=0.5)
        assert one(left, "wingFrac") > 50 * one(swathStats(index, clean), "wingFrac")
        assert one(left, "wingFrac") == pytest.approx(5 * level / (1 + 5 * level), rel=0.05)
        assert one(left, "fwhm") == pytest.approx(one(swathStats(index, clean), "fwhm"), rel=1e-6)

    def testAFloorMovesTheSecondMomentAndNotTheWidth(self):
        """Why the width is read at half maximum: a floor of 2 % of the peak."""
        index = profileIndex()
        clean = gaussian(index)
        floored = clean + 0.02 * clean.max()

        ratio = one(swathStats(index, floored), "fwhm") / one(swathStats(index, clean), "fwhm")
        assert ratio == pytest.approx(1.0, abs=0.02)
        # Negative control: the whole-window second moment doubles on the same floor.
        assert secondMoment(index, floored) / secondMoment(index, clean) > 1.5
        assert one(swathStats(index, floored), "wingFrac") > 0.05

    def testCentroidIsPositiveTowardHigherColumn(self):
        index = profileIndex()
        stats = swathStats(index, gaussian(index, shift=0.3))
        assert one(stats, "centroid") == pytest.approx(0.3, abs=1e-3)
        # Negative control: the opposite sign convention.
        assert not np.isclose(one(stats, "centroid"), -0.3, atol=0.1)

    def testCentroidIsOfTheCoreAndIgnoresTheWings(self):
        index = profileIndex()
        clean = gaussian(index, shift=0.3)
        withLight = clean + plateau(index, 0.18 * clean.max())
        assert one(swathStats(index, withLight), "centroid") == pytest.approx(0.3, abs=0.02)
        # Negative control: the mean offset over the whole profile follows the wing light.
        assert (index * withLight).sum() / withLight.sum() > 1.0

    def testNegativeValuesAreReportedAgainstThePeak(self):
        index = profileIndex()
        profile = gaussian(index)
        profile[np.abs(index - 6.0) < 0.5] -= 0.05 * profile.max()
        assert one(swathStats(index, profile), "minimum") == pytest.approx(-0.05, rel=0.01)

    def testMaskedSamplesAreInterpolatedOver(self):
        index = profileIndex()
        clean = gaussian(index)
        profile = np.ma.masked_array(clean.copy())
        profile[85:89] = np.ma.masked
        profile.data[85:89] = 1e6  # would be the wings' maximum if used
        profile.data[20] = np.nan  # not finite: masked as well
        stats, reference = swathStats(index, profile), swathStats(index, clean)
        assert one(stats, "maskedFrac") == pytest.approx(5 / len(index))
        for name in ("fwhm", "centroid", "wingFrac", "wingMax", "minimum"):
            assert one(stats, name) == pytest.approx(one(reference, name), rel=5e-3, abs=1e-6)

    def testUnmeasurableSwathIsNan(self):
        index = profileIndex()
        profiles = np.ma.masked_array(np.stack([gaussian(index), gaussian(index), np.zeros(len(index))]))
        profiles[1] = np.ma.masked
        stats = swathStats(index, profiles)
        for name in ("fwhm", "centroid", "wingFrac", "wingMax", "minimum", "variation"):
            assert np.isfinite(stats[name][0])
            assert np.isnan(stats[name][1:]).all()
        np.testing.assert_array_equal(stats["maskedFrac"], [0, 1, 0])

    def testVariationComparesShapesNotScales(self):
        index = profileIndex()
        same = swathStats(index, np.stack([gaussian(index), 3.0 * gaussian(index)]))
        np.testing.assert_allclose(same["variation"], 0.0, atol=1e-12)

        differing = swathStats(index, np.stack([gaussian(index), gaussian(index), gaussian(index, 1.5)]))
        assert differing["variation"][2] > 0.05
        assert differing["variation"][2] > 1.5 * differing["variation"][0]


def datamodelForm(fibers):
    """Return profiles as `pfs.datamodel.PfsFiberProfiles` holds them: parallel lists."""
    return SimpleNamespace(
        fiberId=np.array(list(fibers)),
        radius=np.full(len(fibers), RADIUS),
        oversample=np.full(len(fibers), float(OVERSAMPLE)),
        rows=[np.array(rows) for rows, _ in fibers.values()],
        profiles=[np.ma.masked_array(profiles) for _, profiles in fibers.values()],
    )


class StellaForm:
    """Profiles as `pfs.drp.stella.FiberProfileSet` holds them: a mapping of profile objects."""

    def __init__(self, fibers):
        self._fibers = {
            fiberId: SimpleNamespace(
                index=profileIndex(), rows=np.array(rows), profiles=np.ma.masked_invalid(profiles)
            )
            for fiberId, (rows, profiles) in fibers.items()
        }

    @property
    def fiberId(self):
        return np.array(sorted(self._fibers))

    def __getitem__(self, fiberId):
        return self._fibers[fiberId]

    def __len__(self):
        return len(self._fibers)


def makeFibers(nFibers=31, bad=None):
    """Return ``{fiberId: (rows, profiles)}`` of clean two-swath fibers, with ``bad`` ones replaced."""
    index = profileIndex()
    clean = np.stack([gaussian(index), gaussian(index, 1.02 * SIGMA)])
    fibers = {fiberId: ([1000.0, 3000.0], clean.copy()) for fiberId in range(1, nFibers + 1)}
    fibers.update(bad or {})
    return fibers


def placeholderProfile():
    """Return the sampled Gaussian drp_stella stores for a fiber it had no light for."""
    index = profileIndex()
    sigma = PLACEHOLDER_FWHM / (2 * np.sqrt(2 * np.log(2)))
    profile = np.exp(-0.5 * (index / sigma) ** 2)
    return ([1000.0, 3000.0], np.stack([profile, profile]))


class TestMeasureProfiles:
    """A whole calibration, in either form."""

    def testIndexIsRebuiltFromTheDatamodelForm(self):
        fibers = makeFibers(2)
        for _, index, rows, profiles in iterProfiles(datamodelForm(fibers)):
            np.testing.assert_allclose(index, profileIndex())
            assert index[len(index) // 2] == 0.0
            assert profiles.shape == (len(rows), len(index))

    def testBothFormsGiveTheSameTable(self):
        fibers = makeFibers(4)
        fromDatamodel = measureProfiles(datamodelForm(fibers))
        fromStella = measureProfiles(StellaForm(fibers))
        assert list(fromDatamodel.columns) == list(SWATH_COLUMNS)
        assert len(fromDatamodel) == 8
        pd.testing.assert_frame_equal(fromDatamodel, fromStella)

    def testEmptyCalibrationGivesAnEmptyTable(self):
        swaths = measureProfiles(datamodelForm({}))
        assert swaths.empty and list(swaths.columns) == list(SWATH_COLUMNS)
        fibers = summarizeFibers(swaths)
        assert fibers.empty and list(fibers.columns) == list(FIBER_COLUMNS)


class TestSummarizeFibers:
    """One row per fiber."""

    def testReducesTheSwaths(self):
        index = profileIndex()
        level = 0.18 * gaussian(index).max()
        withLight = np.stack([gaussian(index), gaussian(index) + plateau(index, level)])
        masked = np.ma.masked_array(np.stack([gaussian(index), gaussian(index)]))
        masked[1] = np.ma.masked
        fibers = summarizeFibers(
            measureProfiles(
                datamodelForm(makeFibers(3, {2: ([1000.0, 3000.0], withLight), 3: ([1.0, 2.0], masked)}))
            )
        )
        assert list(fibers.columns) == list(FIBER_COLUMNS)
        assert list(fibers["fiberId"]) == [1, 2, 3]
        clean, light, half = (fibers.iloc[ii] for ii in range(3))

        assert (clean["nSwath"], clean["nMeasured"]) == (2, 2)
        assert clean["fwhm"] == pytest.approx(float(np.mean(pixelFwhm([SIGMA, 1.02 * SIGMA]))), rel=0.01)
        assert clean["fwhmRange"] == pytest.approx(float(np.ptp(pixelFwhm([SIGMA, 1.02 * SIGMA]))), rel=0.1)
        # The worst swath decides wingMax.
        assert light["wingMax"] == pytest.approx(0.18 + clean["wingMax"], rel=0.01)
        assert light["variation"] > 10 * clean["variation"]
        assert (half["nSwath"], half["nMeasured"]) == (2, 1)
        assert half["maskedFrac"] == pytest.approx(0.5)
        assert not fibers["placeholder"].any()

    def testPlaceholderFibersAreMarked(self):
        fibers = summarizeFibers(measureProfiles(datamodelForm(makeFibers(3, {2: placeholderProfile()}))))
        assert list(fibers["placeholder"]) == [False, True, False]
        assert fibers["fwhm"].iloc[1] == pytest.approx(PLACEHOLDER_FWHM, abs=0.05)


class TestAgainstOtherFibers:
    """A fiber against the expected list, the previous calibration and its neighbours."""

    def testCoverage(self):
        assert coverage([1, 2, 5], [1, 2, 3, 4]) == {"missing": [3, 4], "unexpected": [5]}
        assert coverage(np.array([2, 1]), [1, 2]) == {"missing": [], "unexpected": []}

    def testCompareFibersIsNewMinusPrevious(self):
        index = profileIndex()
        wide = np.stack([gaussian(index, 1.1 * SIGMA), gaussian(index, 1.1 * SIGMA)])
        previous = summarizeFibers(measureProfiles(datamodelForm(makeFibers(3))))
        new = summarizeFibers(
            measureProfiles(datamodelForm(makeFibers(4, {2: ([1000.0, 3000.0], wide)}))).query("fiberId != 3")
        )
        compared = compareFibers(new, previous)
        assert list(compared["fiberId"]) == [1, 2, 3, 4]
        assert list(compared["inNew"]) == [True, True, False, True]
        assert list(compared["inPrevious"]) == [True, True, True, False]

        change = compared.set_index("fiberId")["fwhmChange"]
        assert change[1] == pytest.approx(0.0, abs=1e-9)
        assert change[2] > 0.1
        # Negative control: previous minus new would be negative for the fiber that widened.
        assert not change[2] < 0
        assert np.isnan(change[3]) and np.isnan(change[4])

    def testNeighbourResidualSinglesOutOneFiber(self):
        index = profileIndex()
        level = 0.18 * gaussian(index).max()
        withLight = np.stack([gaussian(index) + plateau(index, level)] * 2)
        fibers = summarizeFibers(
            measureProfiles(
                datamodelForm(makeFibers(31, {16: ([1000.0, 3000.0], withLight), 20: placeholderProfile()}))
            )
        )
        # Shuffled: the neighbourhood is by fiberId, not by the table's order.
        fibers = fibers.sample(frac=1.0, random_state=1)
        residual = neighbourResidual(fibers, "wingMax")
        assert residual.index.equals(fibers.index)
        byFiber = residual.groupby(fibers["fiberId"]).first()

        assert byFiber[16] == pytest.approx(0.18, rel=0.01)
        assert np.isnan(byFiber[20])
        np.testing.assert_allclose(byFiber.drop([16, 20]), 0.0, atol=1e-9)


def variedFibers(nFibers=31, seed=3, noise=1e-3):
    """Return clean fibers as a calibration measures them: widths that differ, and noise of ``noise`` of the peak.

    Without the noise, the mean of three Gaussians of nearby widths can equal a fourth to within the tolerance.
    """
    index = profileIndex()
    rng = np.random.default_rng(seed)
    fibers = {}
    for fiberId, sigma in zip(
        range(1, nFibers + 1), SIGMA * (1 + 0.02 * rng.standard_normal(nFibers)), strict=True
    ):
        profiles = np.stack([gaussian(index, sigma), gaussian(index, 1.02 * sigma)])
        fibers[fiberId] = (
            [1000.0, 3000.0],
            profiles + noise * profiles.max() * rng.standard_normal(profiles.shape),
        )
    return fibers


def replaceWithMean(fibers, fiberId, donors):
    """Give ``fiberId`` the mean of its donors' profiles, as drp_stella's merge step does."""
    fibers[fiberId] = ([1000.0, 3000.0], np.mean([fibers[donor][1] for donor in donors], axis=0))
    return fibers


class TestReplacedProfiles:
    """Profiles that are the mean of nearby fibers' profiles."""

    def testFindsAReplacedFiberAndItsDonors(self):
        fibers = replaceWithMean(variedFibers(), 16, (14, 15, 17))
        for form in (datamodelForm, StellaForm):
            assert findReplaced(form(fibers)) == {16: (14, 15, 17)}

    def testMeasuredProfilesAreNotReplaced(self):
        assert findReplaced(datamodelForm(variedFibers())) == {}

    @pytest.mark.parametrize("donors", [(15, 17), (14, 15, 17), (14, 15, 17, 18)])
    def testFindsReplacementsByTwoThreeOrFourDonors(self, donors):
        """drp_stella's two replacement steps default to different counts, and both can be configured."""
        fibers = replaceWithMean(variedFibers(), 16, donors)
        assert findReplaced(datamodelForm(fibers)) == {16: donors}
        # Negative control: a search for another count alone does not find it.
        other = 3 if len(donors) != 3 else 2
        assert findReplaced(datamodelForm(fibers), nDonors=other) == {}

    def testTwoCopiesOfTheSameDonorsDoNotNameEachOther(self):
        """They are identical, so each also equals the mean of the three donors and its twin."""
        fibers = replaceWithMean(replaceWithMean(variedFibers(), 15, (13, 14, 17)), 16, (13, 14, 17))
        assert findReplaced(datamodelForm(fibers)) == {15: (13, 14, 17), 16: (13, 14, 17)}
        # Negative control: a search for four donors alone does name the twin.
        assert findReplaced(datamodelForm(fibers), nDonors=4)[15] == (13, 14, 16, 17)

    def testDonorsAreLookedForWithinReach(self):
        fibers = replaceWithMean(variedFibers(), 16, (4, 15, 17))
        assert findReplaced(datamodelForm(fibers)) == {}
        assert findReplaced(datamodelForm(fibers), reach=12) == {16: (4, 15, 17)}

    def testFindsCopiesBesideAGroupOfFibersLitInTheDarks(self):
        """drp_stella takes no donor within three fibers of a fiber lit in the darks.

        With 14, 16 and 17 lit in the darks, no donor is nearer than 11-20, so the copies' donors are four to
        seven fibers away.
        """
        fibers = variedFibers()
        copies = {14: (8, 9, 10), 16: (10, 21, 22), 17: (21, 22, 23)}
        for fiberId, donors in copies.items():
            replaceWithMean(fibers, fiberId, donors)
        for form in (datamodelForm, StellaForm):
            assert findReplaced(form(fibers)) == copies
        # Negative control: a search five fibers either side finds none of the three.
        assert findReplaced(datamodelForm(fibers), reach=5) == {}

    def testReplacedProfilesAreLeftOutOfTheNeighbourReference(self):
        """Six copies of eleven neighbours inherit a donor's light: they must not become the reference."""
        index = profileIndex()
        fibers = variedFibers()
        level = 0.18 * fibers[16][1][0].max()
        fibers[16] = (fibers[16][0], fibers[16][1] + plateau(index, level))
        for copy, clean in (
            (11, (9, 10)),
            (12, (9, 10)),
            (13, (10, 14)),
            (15, (14, 17)),
            (18, (17, 19)),
            (20, (19, 21)),
        ):
            fibers = replaceWithMean(fibers, copy, (*clean, 16))
        table = fiberTable(datamodelForm(fibers)).set_index("fiberId")
        assert int(table["replaced"].sum()) == 6
        clean = table.drop(index=[11, 12, 13, 15, 16, 18, 20])
        assert clean["wingExcess"].abs().max() < 0.005
        assert table.loc[16, "wingExcess"] == pytest.approx(
            table.loc[16, "wingFrac"] - clean["wingFrac"].median(), abs=0.005
        )
        assert (
            table.loc[[11, 12, 13, 15, 18, 20], "wingExcess"].min() > 0.03
        )  # a copy still gets its own excess
        # Negative control: with the copies in the reference, the clean fibers among them read as deficient.
        biased = neighbourResidual(table.reset_index().drop(columns="replaced"), "wingFrac")
        assert biased.min() < -0.03

    def testAReplacedProfileInheritsADonorsLight(self):
        """One donor of three carries a shelf: the copy gets about a third of it."""
        index = profileIndex()
        fibers = variedFibers()
        level = 0.18 * fibers[15][1][0].max()
        fibers[15] = (fibers[15][0], fibers[15][1] + plateau(index, level))
        table = fiberTable(datamodelForm(replaceWithMean(fibers, 16, (14, 15, 17)))).set_index("fiberId")
        assert list(table.columns) == [name for name in FIBER_TABLE_COLUMNS if name != "fiberId"]
        assert bool(table.loc[16, "replaced"]) and table.loc[16, "donors"] == "14 15 17"
        assert not table.loc[15, "replaced"] and table.loc[15, "donors"] == ""
        assert table.loc[15, "wingExcess"] > 0.1
        assert table.loc[16, "wingExcess"] == pytest.approx(table.loc[15, "wingExcess"] / 3, rel=0.25)
        assert table.drop(index=[15, 16])["wingExcess"].abs().max() < 0.005


class TestDetectorMetrics:
    """One detector's numbers."""

    def table(self):
        index = profileIndex()
        fibers = variedFibers()
        fibers[15] = (fibers[15][0], fibers[15][1] + plateau(index, 0.18 * fibers[15][1][0].max()))
        fibers = replaceWithMean(fibers, 16, (14, 15, 17))
        fibers[20] = placeholderProfile()
        return fiberTable(datamodelForm(fibers))

    def testCountsAndExtremes(self):
        fibers = self.table()
        metrics = detectorMetrics(fibers, expected=range(1, 34))
        assert set(metrics) == set(METRICS)
        assert (metrics["nProfiles"], metrics["nMissing"]) == (31, 2)
        assert (metrics["nPlaceholder"], metrics["nReplaced"]) == (1, 1)
        byFiber = fibers.set_index("fiberId")
        assert metrics["maxWingExcess"] == pytest.approx(byFiber.loc[15, "wingExcess"])
        assert metrics["maxWingExcessReplaced"] == pytest.approx(byFiber.loc[16, "wingExcess"])
        # Negative control: the replaced profile's excess is kept out of the measured profiles' maximum.
        assert metrics["maxWingExcess"] > 2 * metrics["maxWingExcessReplaced"]
        assert metrics["medProfileFwhm"] == pytest.approx(float(pixelFwhm(SIGMA)), rel=0.03)
        assert -0.01 < metrics["minProfileValue"] < 0  # the noise of the wings

    def testAnUnmeasurableProfileIsCounted(self):
        """Every statistic of it is NaN, so it would otherwise leave no trace in the detector's metrics."""
        fibers = variedFibers()
        fibers[9] = (fibers[9][0], -np.abs(fibers[9][1]))  # no positive peak
        fibers[10] = (fibers[10][0], np.ma.masked_all(fibers[10][1].shape))
        table = fiberTable(datamodelForm(fibers))
        assert list(table.set_index("fiberId").loc[[9, 10], "nMeasured"]) == [0, 0]
        metrics = detectorMetrics(table)
        assert metrics["nUnmeasured"] == 2
        assert (metrics["nProfiles"], metrics["nPlaceholder"], metrics["nReplaced"]) == (31, 0, 0)
        # Negative control: without the count, the rest reads as a healthy detector.
        assert metrics["maxWingExcess"] < 0.005 and np.isfinite(metrics["medProfileFwhm"])
        assert detectorMetrics(fiberTable(datamodelForm(variedFibers())))["nUnmeasured"] == 0

    def testARunOfPlaceholdersIsNotReplaced(self):
        """Neighbouring placeholders are identical, so each equals the mean of three others."""
        fibers = variedFibers()
        for fiberId in range(12, 17):
            fibers[fiberId] = placeholderProfile()
        calibration = datamodelForm(fibers)
        assert set(findReplaced(calibration)) == set(range(12, 17))
        table = fiberTable(calibration).set_index("fiberId")
        assert table.loc[12:16, "placeholder"].all()
        assert not table["replaced"].any() and set(table["donors"]) == {""}
        metrics = detectorMetrics(table.reset_index())
        assert (metrics["nPlaceholder"], metrics["nReplaced"]) == (5, 0)

    def testUnknownAndEmptyAreNan(self):
        metrics = detectorMetrics(fiberTable(datamodelForm(variedFibers())))
        assert np.isnan(metrics["nMissing"])  # the expected fibers were not given
        assert np.isnan(metrics["maxWingExcessReplaced"])  # nothing was replaced
        assert metrics["nReplaced"] == 0

    def testProfileQaAddsWhatItMeasured(self):
        fibers, metrics = profileQa(
            datamodelForm(variedFibers(8)),
            {"visit": 156021, "arm": "r", "spectrograph": 4},
            expected=range(1, 9),
            calibVisit0=153013,
            calibObsDate="2026-11-02",
            calibRun="u/me/profilesGen/20261119T033500Z",
        )
        assert list(fibers.columns) == [
            "visit",
            "arm",
            "spectrograph",
            "calibVisit0",
            "calibObsDate",
            "calibRun",
            *FIBER_TABLE_COLUMNS,
        ]
        assert set(fibers["calibRun"]) == set(metrics["calibRun"]) == {"u/me/profilesGen/20261119T033500Z"}
        assert set(fibers["calibVisit0"]) == {153013} and len(fibers) == 8
        assert list(metrics.columns) == list(METRIC_COLUMNS)
        assert list(metrics["metric"]) == list(METRICS)
        assert metrics.set_index("metric").loc["nMissing", "value"] == 0
        assert set(metrics["visit"]) == {156021} and set(metrics["arm"]) == {"r"}


DATA = Path(__file__).parent / "data"

#: The fibers reported with a plateau of a neighbour's light on r4 in the Run31 calibration.
REPORTED = (2108, 2155, 2179, 2185, 2209, 2221, 2443, 2574)


def loadCalibration(run):
    """Load a fixture written by ``data/makeProfileFixtures.py`` in the datamodel's form, one swath per fiber."""
    path = DATA / f"profiles-{run}-r4.npz"
    if not path.exists():
        pytest.skip(f"{path.name} missing; make it with tests/fiberProfiles/data/makeProfileFixtures.py")
    data = np.load(path)
    count = len(data["fiberId"])
    return SimpleNamespace(
        fiberId=data["fiberId"],
        oversample=np.full(count, float(data["oversample"])),
        rows=[np.array([2000.0])] * count,
        profiles=[np.ma.masked_invalid(profile[np.newaxis, :].astype(float)) for profile in data["profile"]],
    )


class TestKnownFault:
    """The Run31 calibration of r4 against Run30's, on the real profiles (swaths averaged)."""

    def testRun31sFaultIsMeasured(self):
        fibers = fiberTable(loadCalibration("run31")).set_index("fiberId")
        assert len(fibers) == 600
        # Every reported fiber stands far above its neighbours, where their scatter is 0.0003.
        assert fibers.loc[list(REPORTED), "wingExcess"].min() > 0.05
        assert not fibers.loc[list(REPORTED), "replaced"].any()
        # 2208 cannot be hidden: its profile is the mean of three neighbours', and carries a third of 2209's light.
        assert fibers.loc[2208, "donors"] == "2206 2207 2209"
        assert fibers.loc[2208, "wingExcess"] == pytest.approx(fibers.loc[2209, "wingExcess"] / 3, rel=0.1)
        assert int(fibers["replaced"].sum()) == 21
        assert int((fibers["replaced"] & (fibers["wingExcess"] > 0.01)).sum()) == 12

        metrics = detectorMetrics(fibers.reset_index())
        assert metrics["maxWingExcess"] > 0.15
        assert metrics["maxWingExcessReplaced"] > 0.05

    def testRun30IsCleanOnTheSameFibers(self):
        fibers = fiberTable(loadCalibration("run30")).set_index("fiberId")
        metrics = detectorMetrics(fibers.reset_index())
        # Negative control: the same fibers, and the detector's worst, are an order of magnitude lower.
        assert fibers.loc[list(REPORTED), "wingExcess"].abs().max() < 0.005
        assert metrics["maxWingExcess"] < 0.02
        assert not metrics["maxWingExcessReplaced"] > 0.01
        assert fibers.loc[2208, "donors"] == "2206 2207 2209"


class FakePfsConfig:
    """A pfsConfig: fibers with their spectrograph and target type, selectable by spectrograph."""

    def __init__(self, fiberId, spectrograph, targetType):
        self.fiberId, self.spectrograph, self.targetType = (
            np.asarray(v) for v in (fiberId, spectrograph, targetType)
        )

    def select(self, spectrograph):
        keep = self.spectrograph == spectrograph
        return FakePfsConfig(self.fiberId[keep], self.spectrograph[keep], self.targetType[keep])


class TestTask:
    """The task's own code, loaded without the stack: the expected fibers and what ``run`` passes on."""

    def pfsConfig(self, task):
        science, engineering = int(task.TargetType.SCIENCE), int(task.TargetType.ENGINEERING)
        # Fibers 1-10 on spectrograph 4, two of them engineering fibers; 11-14 on spectrograph 3.
        return FakePfsConfig(
            np.arange(1, 15),
            np.r_[np.full(10, 4), np.full(4, 3)],
            np.r_[np.full(8, science), engineering, engineering, np.full(4, science)],
        )

    def testExpectedFibersAreTheSpectrographsLessItsEngineeringFibers(self, task):
        expected = task.expectedFibers(self.pfsConfig(task), 4)
        assert list(expected) == list(range(1, 9))
        # Negative control: the engineering fibers, and the other spectrograph's, are not expected.
        assert not {9, 10, 11} & set(expected.tolist())

    def testRunStoresBothTablesForTheCalibrationItWasGiven(self, task):
        calibration = StellaForm(variedFibers(8))
        calibration.identity = SimpleNamespace(visit0=153013, obsDate="2026-11-02")
        dataId = {"visit": 156021, "arm": "r", "spectrograph": 4}
        for given in (calibration, task.DeferredDatasetHandle(calibration)):
            outputs = task.FiberProfilesQaTask().run(
                fiberProfiles=given, dataId=dataId, pfsConfig=self.pfsConfig(task), calibRun="u/me/gen/run"
            )
            fibers, metrics = outputs.fiberProfilesQaFibers, outputs.fiberProfilesQaMetrics
            assert set(fibers["calibRun"]) == set(metrics["calibRun"]) == {"u/me/gen/run"}
            assert set(fibers["calibVisit0"]) == {153013} and set(fibers["calibObsDate"]) == {"2026-11-02"}
            assert list(fibers["fiberId"]) == list(range(1, 9))
            values = metrics.set_index("metric")["value"]
            assert (values["nProfiles"], values["nMissing"]) == (8, 0)
            assert set(metrics["calibVisit0"]) == {153013}

    def testRunWithoutAPfsConfigDoesNotKnowWhatIsMissing(self, task):
        calibration = StellaForm(variedFibers(8))
        outputs = task.FiberProfilesQaTask().run(
            fiberProfiles=calibration, dataId={"visit": 156021, "arm": "r", "spectrograph": 4}
        )
        values = outputs.fiberProfilesQaMetrics.set_index("metric")["value"]
        assert np.isnan(values["nMissing"])
        assert set(outputs.fiberProfilesQaFibers["calibVisit0"]) == {
            -1
        }  # the calibration carries no identity

    def testAnEmptyCalibrationIsAnError(self, task):
        calibration = StellaForm({})
        with pytest.raises(ValueError, match="holds no profile"):
            task.FiberProfilesQaTask().run(
                fiberProfiles=calibration, dataId={"visit": 1, "arm": "r", "spectrograph": 4}
            )
