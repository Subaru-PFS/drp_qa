"""Shared fixtures for the ``pfs.drp.qa.guiders`` tests.

These tests need numpy, pandas, matplotlib and pfs_utils but not the LSST
stack. CI runs them in their own job (``guiders`` in
``.github/workflows/tests.yml``); the standard-library job ignores this
directory.
"""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.guiders.coordinates import AGC_CAMERA_CENTERS_MM
from pfs.utils.datamodel.ag import SourceCatalogFlags

GAIA_ISOLATED = SourceCatalogFlags.GAIA | SourceCatalogFlags.NON_BINARY
NOISE = 2e-3  # Centroiding noise (mm)
SEED = 12345


def makeAgcData(
    rng: np.random.RandomState,
    nVisit: int = 3,
    nExp: int = 8,
    nStar: int = 5,
    closedFrac: float = 0.2,
    stars: bool = False,
) -> pd.DataFrame:
    """Make synthetic AG data.

    Positions are in hardware coordinates, as the readers return them. Each
    agc_exposure has a random offset (~30 microns) and rotation (~1 arcmin)
    of the guide stars' centroids relative to their nominal positions, plus
    centroiding noise. The rows are shuffled (the opdb doesn't guarantee
    an order) and the per-visit frames are concatenated without resetting the
    index, as older versions of the readers did.

    Ported from ``makeAgcData`` in drp_stella's ``tests/test_guiders.py``
    (branch ``tickets/guiders-cleanup-reference``). The random draws are made
    in the same order, so the same seed gives the same data. Unlike the
    original it doesn't add the derived ``rms``, ``FWHM`` and ``left``
    columns when ``stars=True``; those come from the package once its readers
    move.

    Parameters
    ----------
    rng : `numpy.random.RandomState`
        Random number generator.
    nVisit, nExp, nStar : `int`
        Number of visits, agc_exposures per visit, and guide stars per camera.
    closedFrac : `float`
        Probability that the spectrograph shutters are closed for an exposure.
    stars : `bool`
        Add the columns of ``readAGCStarsForVisitSetByPfsVisitId`` (used for
        focus plots) to those of ``readAgcDataFromOpdb``.

    Returns
    -------
    agcData : `pandas.DataFrame`
        One row per (agc_exposure, guide star), with duplicate index labels
        when ``nVisit > 1``.
    """
    nominal = {}
    for cid in range(6):
        xc, yc = AGC_CAMERA_CENTERS_MM[cid]
        for ss in range(nStar):
            nominal[1000 * cid + ss] = (cid, xc + rng.uniform(-5, 5), yc + rng.uniform(-5, 5))

    frames = []
    aid = 1000
    t0 = pd.Timestamp("2025-05-01T10:00:00")
    for vv in range(nVisit):
        rows = []
        for ee in range(nExp):
            aid += 1
            theta = np.deg2rad(rng.normal(0, 1 / 60))
            x0, y0 = rng.normal(0, 0.03, 2)
            shutter = int(rng.uniform() > closedFrac)
            cc, ss = np.cos(theta), np.sin(theta)
            for gsid, (cid, xn, yn) in nominal.items():
                row = {
                    "pfs_visit_id": 120000 + vv,
                    "agc_exposure_id": aid,
                    "guide_star_id": gsid,
                    "taken_at": t0 + np.timedelta64(10 * aid, "s"),
                    "altitude": 60.0 + vv,
                    "azimuth": 10.0 + ee,
                    "insrot": -30.0 + vv,
                    "exptime": 900.0,
                    "m2_pos3": 0.1 * vv,
                    "agc_camera_id": cid,
                    "agc_nominal_x_mm": xn,
                    "agc_nominal_y_mm": yn,
                    "agc_center_x_mm": x0 + cc * xn - ss * yn + rng.normal(0, NOISE),
                    "agc_center_y_mm": y0 + ss * xn + cc * yn + rng.normal(0, NOISE),
                    "agc_match_flags": 1,
                    "agc_data_flags": int(rng.uniform() < 0.5),
                    "shutter_open": shutter,
                    "guide_delta_insrot": rng.normal(0, 3),
                    "guide_delta_azimuth": rng.normal(0, 0.5),
                    "guide_delta_altitude": rng.normal(0, 0.5),
                }
                if stars:
                    mxx = rng.uniform(2, 4)
                    row.update(
                        guide_star_flag=int(GAIA_ISOLATED) if gsid % 5 else int(SourceCatalogFlags.HSC),
                        m2_off3=-0.3 + 0.05 * (ee % 4),
                        centroid_y_pix=rng.uniform(0, 1000),
                        mxx=mxx,
                        myy=mxx * rng.uniform(0.9, 1.1),
                        mxy=0.1,
                        estimated_magnitude=15.0,
                        guide_delta_z=rng.normal(0, 0.02),
                        **{f"guide_delta_z{ii}": rng.normal(0, 0.02) for ii in range(1, 7)},
                    )
                rows.append(row)
        frames.append(pd.DataFrame(rows).sample(frac=1, random_state=rng))

    return pd.concat(frames)


@pytest.fixture
def rng() -> np.random.RandomState:
    """Return a freshly seeded random number generator."""
    return np.random.RandomState(SEED)


@pytest.fixture(name="makeAgcData")
def makeAgcDataFixture(rng):
    """Return `makeAgcData` bound to the test's ``rng``.

    Call it with `makeAgcData`'s keyword arguments, e.g.
    ``makeAgcData(nVisit=1, stars=True)``.
    """

    def make(**kwargs) -> pd.DataFrame:
        return makeAgcData(rng, **kwargs)

    return make
