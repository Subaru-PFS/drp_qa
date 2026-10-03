"""Placeholder tests for the ``pfs.drp.qa.guiders`` scaffold and its CI job.

They check that the package imports, that the CI environment provides the
pfs_utils modules the guider code uses, and that the shared fixtures work.
"""

import importlib

import pytest
from matplotlib.figure import Figure

from pfs.drp.qa.utils.plotting import opaqueColorbar


@pytest.mark.parametrize("name", ["coordinates", "queries", "compat", "analysis", "plotting"])
def testGuidersModulesImport(name):
    importlib.import_module(f"pfs.drp.qa.guiders.{name}")


@pytest.mark.parametrize(
    "name",
    [
        "pfs.utils.database.opdb",
        "pfs.utils.coordinates.CoordTransp",
        "pfs.utils.coordinates.transform",
        "pfs.utils.datamodel.ag",
    ],
)
def testPfsUtilsModulesImport(name):
    """The guider code needs these; pfs-utils is installed with --no-deps in CI."""
    importlib.import_module(name)


@pytest.mark.parametrize("alpha", [0.3, 0, None])
def testOpaqueColorbar(alpha):
    ax = Figure().subplots()
    S = ax.scatter([0, 1], [0, 1], c=[0, 1], alpha=alpha)
    with opaqueColorbar(S):
        assert S.get_alpha() == 1
        ax.figure.colorbar(S)
    assert S.get_alpha() == alpha


def testMakeAgcData(makeAgcData):
    nVisit, nExp, nStar = 3, 8, 5
    agcData = makeAgcData(nVisit=nVisit, nExp=nExp, nStar=nStar)

    assert len(agcData) == nVisit * nExp * 6 * nStar
    assert agcData.pfs_visit_id.nunique() == nVisit
    assert agcData.agc_exposure_id.nunique() == nVisit * nExp
    assert agcData.guide_star_id.nunique() == 6 * nStar
    assert not agcData.index.is_unique  # per-visit frames are concatenated as-is
    assert not agcData.agc_exposure_id.is_monotonic_increasing  # rows are shuffled
    assert set(agcData.shutter_open) == {0, 1}

    # Offsets from nominal are ~30 microns of pointing error, not mm.
    dx = agcData.agc_center_x_mm - agcData.agc_nominal_x_mm
    assert 0.01 < dx.std() < 0.1


def testMakeAgcDataStars(makeAgcData):
    agcData = makeAgcData(nVisit=1, nExp=2, stars=True)
    for column in ("guide_star_flag", "m2_off3", "mxx", "myy", "mxy", "guide_delta_z", "guide_delta_z6"):
        assert column in agcData.columns, column
    assert {"rms", "FWHM", "left"}.isdisjoint(agcData.columns)
