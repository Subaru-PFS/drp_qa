"""Tests for ``pfs.drp.qa.guiders.compat``, drp_stella's reader names."""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.guiders import compat
from pfs.drp.qa.guiders.coordinates import AGC_PIXEL_SIZE_UM, AGC_PLATE_SCALE_UM_PER_ARCSEC, opdbToHardware
from pfs.drp.qa.guiders.queries import AGC_DATA_COLUMNS, readAgcData

VISITS = [120000, 120001]

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.filterwarnings("default::DeprecationWarning")
@pytest.mark.parametrize(
    "name, args",
    [
        ("readAGCPositionsForVisitByAgcExposureId", (VISITS[0], True)),
        ("readAgcDataFromOpdb", (VISITS,)),
        ("readAGCStarsForVisitByPfsVisitId", (VISITS[0],)),
        ("readAGCStarsForVisitSetByPfsVisitId", (VISITS,)),
    ],
)
def testDeprecated(makeOpdb, name, args):
    opdb, _ = makeOpdb(nVisit=2)
    with pytest.warns(DeprecationWarning, match=f"{name} is deprecated; use .*queries.readAgcData") as record:
        getattr(compat, name)(opdb, *args)
    assert record[0].filename == __file__  # attributed to the caller


@pytest.mark.parametrize("flip", [True, False])
def testReadAGCPositionsForVisitByAgcExposureId(makeOpdb, flip):
    opdb, _ = makeOpdb(nVisit=1)
    data = compat.readAGCPositionsForVisitByAgcExposureId(opdb, VISITS[0], flip)
    hardware = readAgcData(opdb, VISITS[0])

    assert list(data.columns) == list(AGC_DATA_COLUMNS)
    np.testing.assert_allclose(data.agc_center_x_mm, hardware.agc_center_x_mm)
    sign = 1 if flip else -1
    np.testing.assert_allclose(data.agc_center_y_mm, sign * hardware.agc_center_y_mm)
    np.testing.assert_allclose(data.agc_nominal_y_mm, sign * hardware.agc_nominal_y_mm)
    if not flip:
        # In the opdb's frame, so it can still be converted.
        pd.testing.assert_frame_equal(opdbToHardware(data), hardware, check_like=True)


def testReadAgcDataFromOpdb(makeOpdb, StubButler):
    opdb, _ = makeOpdb(nVisit=2)
    rows = opdb.tables["agc_data"]
    rows.loc[rows.pfs_visit_id == VISITS[0], "exptime"] = np.nan  # no SpS exposure

    data = compat.readAgcDataFromOpdb(opdb, VISITS)
    assert "inst_pa" not in data
    assert (data[data.pfs_visit_id == VISITS[0]].exptime == 0).all()
    assert (data[data.pfs_visit_id == VISITS[1]].exptime == 900).all()
    pd.testing.assert_index_equal(data.index, pd.RangeIndex(len(data)))

    butler = StubButler([{"visit": v, "arm": "r", "spectrograph": 1} for v in VISITS])
    data = compat.readAgcDataFromOpdb(opdb, iter(VISITS), butler=butler)
    np.testing.assert_array_equal(data.inst_pa, 10.0 * data.pfs_visit_id + 1)

    with pytest.raises(RuntimeError, match="No AG data for visits 1, 2"):
        compat.readAgcDataFromOpdb(opdb, iter([1, 2]))


@pytest.mark.parametrize("flip", [True, False])
def testReadAGCStars(makeOpdb, flip):
    opdb, _ = makeOpdb(nVisit=2)
    rows = opdb.tables["agc_data"]
    rows.loc[rows.guide_star_id % 2 == 0, "agc_match_flags"] = 0

    stars = compat.readAGCStarsForVisitSetByPfsVisitId(opdb, VISITS, flip)
    single = compat.readAGCStarsForVisitByPfsVisitId(opdb, VISITS[0], flipToHardwareCoords=flip)
    hardware = readAgcData(opdb, VISITS)

    assert (stars.agc_match_flags == 1).all()
    assert len(stars) == (hardware.agc_match_flags == 1).sum()
    assert len(single) == (stars.pfs_visit_id == VISITS[0]).sum()
    assert set(AGC_DATA_COLUMNS) | {
        "flags",
        "guide_delta_az",
        "guide_delta_el",
        "rms",
        "FWHM",
        "left",
    } == set(stars.columns)
    pd.testing.assert_series_equal(stars["flags"], stars.agc_data_flags, check_names=False)
    pd.testing.assert_series_equal(stars.guide_delta_az, stars.guide_delta_azimuth, check_names=False)
    pd.testing.assert_series_equal(stars.guide_delta_el, stars.guide_delta_altitude, check_names=False)

    good = hardware[hardware.agc_match_flags == 1].reset_index(drop=True)
    sign = 1 if flip else -1
    np.testing.assert_allclose(stars.agc_center_y_mm, sign * good.agc_center_y_mm)


def testReadAGCStarsEmpty(makeOpdb):
    opdb, _ = makeOpdb(nVisit=1)
    opdb.tables["agc_data"]["agc_match_flags"] = 0

    assert compat.readAGCStarsForVisitByPfsVisitId(opdb, VISITS[0]).empty
    with pytest.raises(RuntimeError, match=f"No AG stars in visits {VISITS[0]}"):
        compat.readAGCStarsForVisitSetByPfsVisitId(opdb, [VISITS[0]])


def testImageSizes():
    """drp_stella's rms (pix), FWHM (arcsec) and left."""
    stars = pd.DataFrame(
        {
            "mxx": [4.0, 2.0, -1.0],
            "myy": [4.0, 8.0, 1.0],
            "mxy": [0.0, 1.0, 0.0],
            "agc_data_flags": [0, 1, 0],
        }
    )
    trace = compat._addImageSizes(stars.copy(), useTraceRadius=True)
    np.testing.assert_allclose(trace.rms, [2.0, np.sqrt(5.0), np.nan])
    fwhmPerPix = 2 * np.sqrt(2 * np.log(2)) * AGC_PIXEL_SIZE_UM / AGC_PLATE_SCALE_UM_PER_ARCSEC
    np.testing.assert_allclose(trace.FWHM, fwhmPerPix * trace.rms)
    np.testing.assert_allclose(trace.FWHM[0], 0.6466, atol=1e-4)  # 2 pixels rms, in arcsec
    assert list(trace.left) == [True, False, True]

    det = compat._addImageSizes(stars.copy(), useTraceRadius=False)
    np.testing.assert_allclose(det.rms, [2.0, 15.0**0.25, np.nan])
