"""Per-AG-exposure seeing and transparency from good-match spots (`pfs.drp.qa.guiders.conditions`).

The query and the qadb write need a database; the aggregation in `exposureConditions` is pure and
is what this file tests, against a direct `pfs.utils.agMetrics` computation on the same spots, with
negative controls on the sub-PSF drop, the transparency range cut and the ``"git"`` moment doubling.
"""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.guiders.conditions import exposureConditions
from pfs.utils import agMetrics

COLUMNS = [
    "agc_exposure_id",
    "version_actor",
    "altitude",
    "mxx",
    "myy",
    "estimated_magnitude",
    "guide_star_magnitude",
    "guide_star_color",
]


def makeSpots(rows):
    """Build a spots DataFrame from (agc_exposure_id, version_actor, altitude, mxx, myy, est, cat, color) rows."""
    spots = pd.DataFrame(rows, columns=COLUMNS)
    spots["pfs_visit_id"] = 42
    spots["taken_at"] = pd.Timestamp("2026-11-20 03:00:00")
    return spots


def perSpot(spots, variancePerMoment=1.0):
    """Return the per-spot seeing and (range-cut) transparency, straight from agMetrics, for the stats."""
    fwhm = agMetrics.momentsToFwhm(
        spots.mxx.to_numpy(float), spots.myy.to_numpy(float), variancePerMoment=variancePerMoment
    )
    seeing = agMetrics.calibratedSeeing(fwhm, spots.altitude.to_numpy(float))
    transparency = agMetrics.magnitudesToTransparency(
        spots.estimated_magnitude.to_numpy(float),
        spots.guide_star_magnitude.to_numpy(float),
        spots.guide_star_color.to_numpy(float),
    )
    transparency = np.where((transparency > 0.0) & (transparency < 2.0), transparency, np.nan)
    return np.asarray(seeing, float), np.asarray(transparency, float)


def test_aggregates_match_agMetrics():
    """One exposure's seeing/transparency mean, median and sigma match the agMetrics values directly."""
    spots = makeSpots(
        [
            (100, "2.4.0a", 60.0, 9.0, 8.0, 15.0, 15.1, 0.6),
            (100, "2.4.0a", 60.0, 12.0, 11.0, 14.6, 14.7, 0.5),
            (100, "2.4.0a", 60.0, 7.0, 8.0, 16.2, 16.0, 0.7),
        ]
    )
    out = exposureConditions(spots)
    assert len(out) == 1
    row = out.iloc[0]
    assert row.agc_exposure_id == 100 and row.pfs_visit_id == 42
    seeing, transparency = perSpot(spots)
    assert np.all(np.isfinite(seeing)) and np.all(np.isfinite(transparency))  # a clean exposure
    for name, values in (("seeing", seeing), ("transparency", transparency)):
        series = pd.Series(values)
        assert row[f"{name}_mean"] == pytest.approx(series.mean())
        assert row[f"{name}_median"] == pytest.approx(series.median())
        assert row[f"{name}_sigma"] == pytest.approx(series.std())  # pandas ddof=1, as the code uses


def test_sub_psf_spot_drops_from_seeing():
    """A spot the AG cannot resolve (NaN seeing) leaves the exposure's seeing but not its other spots."""
    spots = makeSpots(
        [
            (200, "2.4.0a", 90.0, 10.0, 10.0, 15.0, 15.0, 0.5),  # resolved
            (200, "2.4.0a", 90.0, 0.3, 0.3, 15.0, 15.0, 0.5),  # below the instrument PSF
        ]
    )
    seeing, _ = perSpot(spots)
    assert np.isfinite(seeing[0]) and np.isnan(seeing[1])  # the scenario: one resolved, one not
    row = exposureConditions(spots).iloc[0]
    # The NaN spot is dropped, not propagated or counted as zero: the mean is the one good spot's seeing.
    assert row.seeing_mean == pytest.approx(seeing[0])
    # Negative control: counting the sub-PSF spot as zero would halve the mean.
    assert row.seeing_mean > seeing[0] / 2 + 1e-6


def test_out_of_range_transparency_is_cut():
    """A star far from its catalogue magnitude (a mismatch, transparency outside 0-2) is left out."""
    spots = makeSpots(
        [
            (300, "2.4.0a", 90.0, 10.0, 10.0, 15.0, 15.0, 0.5),  # transparency ~1
            (300, "2.4.0a", 90.0, 10.0, 10.0, 13.0, 15.0, 0.5),  # ~2 mag bright -> transparency > 2
        ]
    )
    _, transparency = perSpot(spots)
    raw = agMetrics.magnitudesToTransparency(
        spots.estimated_magnitude.to_numpy(float),
        spots.guide_star_magnitude.to_numpy(float),
        spots.guide_star_color.to_numpy(float),
    )
    assert raw[1] > 2.0 and np.isnan(transparency[1])  # the scenario: second star out of range
    row = exposureConditions(spots).iloc[0]
    assert row.transparency_mean == pytest.approx(transparency[0])
    # Negative control: the out-of-range value would pull the mean above 2 if it were kept.
    assert row.transparency_mean < 2.0


def test_git_moments_are_doubled():
    """Identical moments give a larger seeing in the ``"git"`` era (halved moments corrected back)."""
    rows = [(1, "git", 90.0, 9.0, 9.0, 15.0, 15.0, 0.5), (2, "2.4.0a", 90.0, 9.0, 9.0, 15.0, 15.0, 0.5)]
    out = exposureConditions(makeSpots(rows)).set_index("agc_exposure_id")
    git, new = out.loc[1].seeing_mean, out.loc[2].seeing_mean
    assert np.isfinite(git) and np.isfinite(new)
    # git moments are doubled before the FWHM, so the FWHM and seeing are larger.
    assert git > new
    # Negative control: they must not be treated identically.
    assert not np.isclose(git, new)
    # The git seeing matches an agMetrics computation with the doubling applied.
    gitSeeing, _ = perSpot(makeSpots([rows[0]]), variancePerMoment=agMetrics.VARIANCE_PER_AGC_MOMENT)
    assert git == pytest.approx(gitSeeing[0])


def test_empty_input_gives_empty_frame():
    """No spots returns an empty frame with the output columns, so a write of it is a no-op."""
    out = exposureConditions(makeSpots([]))
    assert len(out) == 0
    for column in ("agc_exposure_id", "pfs_visit_id", "seeing_median", "transparency_sigma"):
        assert column in out.columns
