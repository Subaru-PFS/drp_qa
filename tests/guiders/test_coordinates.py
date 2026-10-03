"""Tests for the frame, sign and unit conventions in ``pfs.drp.qa.guiders.coordinates``."""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.guiders import coordinates as coords
from pfs.utils.coordinates.coordinates import det2dp
from pfs.utils.coordinates.CoordTransp import DCoeff, ag_pixel_to_pfimm
from pfs.utils.coordinates.Subaru_POPT2_PFS import PFS

# Units


@pytest.mark.parametrize(
    ("func", "value", "expected"),
    [
        (coords.mmToUm, 1.5, 1500),
        (coords.umToMm, 1500, 1.5),
        (coords.pixToUm, 2, 26),
        (coords.pixToMm, 1000, 13),
        (coords.umToArcsec, 94.7, 1),
        (coords.pixToArcsec, 94.7 / 13, 1),
        (coords.radToArcsec, np.pi / 180, 3600),
        (coords.arcsecToRad, 3600, np.pi / 180),
        (coords.guiderFocusToM2Off3, 800, 1),
        (coords.m2Off3ToGuiderFocus, -0.25, -200),
    ],
)
def testUnitConversions(func, value, expected):
    assert func(value) == pytest.approx(expected)


def testUnitRoundTrips(rng):
    x = rng.normal(size=10)
    np.testing.assert_allclose(coords.umToMm(coords.mmToUm(x)), x)
    np.testing.assert_allclose(coords.arcsecToRad(coords.radToArcsec(x)), x)
    np.testing.assert_allclose(coords.m2Off3ToGuiderFocus(coords.guiderFocusToM2Off3(x)), x)


def testUnitConversionsKeepSeries():
    x = pd.Series([1.0, 2.0], index=[7, 7])
    pd.testing.assert_series_equal(coords.mmToUm(x), pd.Series([1000.0, 2000.0], index=[7, 7]))


def testConstantsMatchPfsUtils():
    """The AG pixel size and camera radius agree with pfs_utils's."""
    assert coords.mmToUm(DCoeff.agpixel) == pytest.approx(coords.AGC_PIXEL_SIZE_UM)
    assert DCoeff.agcent == pytest.approx(coords.AGC_RING_RADIUS_MM, abs=2)


# Frames


def testOpdbToHardware():
    """Only the opdb's y positions change sign, in a copy, once."""
    opdb = pd.DataFrame(
        {
            "agc_center_x_mm": [1.0, 2.0],
            "agc_center_y_mm": [3.0, -4.0],
            "agc_nominal_x_mm": [5.0, 6.0],
            "agc_nominal_y_mm": [7.0, -8.0],
            "centroid_y_pix": [9.0, 10.0],
        },
        index=[0, 0],
    )
    original = opdb.copy()

    hardware = coords.opdbToHardware(opdb)

    pd.testing.assert_frame_equal(opdb, original)
    assert opdb.attrs == {}
    np.testing.assert_array_equal(hardware.agc_center_y_mm, [-3.0, 4.0])
    np.testing.assert_array_equal(hardware.agc_nominal_y_mm, [-7.0, 8.0])
    for column in ("agc_center_x_mm", "agc_nominal_x_mm", "centroid_y_pix"):
        pd.testing.assert_series_equal(hardware[column], opdb[column])

    with pytest.raises(ValueError, match="already in hardware"):
        coords.opdbToHardware(hardware)
    with pytest.raises(ValueError, match="already in hardware"):
        coords.opdbToHardware(hardware[hardware.agc_center_x_mm > 1])


def testOpdbToHardwareMissingColumns():
    hardware = coords.opdbToHardware(pd.DataFrame({"agc_center_y_mm": [1.0]}))
    np.testing.assert_array_equal(hardware.agc_center_y_mm, [-1.0])


@pytest.mark.parametrize("agcCameraId", range(6))
def testOpdbConvention(agcCameraId):
    """The opdb's convention (ag_pixel_to_pfimm) becomes det2dp's, and only with the flip."""
    xPix, yPix = (grid.ravel() for grid in np.meshgrid(np.linspace(0, 1071, 7), np.linspace(0, 1041, 7)))
    x_mm, y_mm = ag_pixel_to_pfimm(agcCameraId, xPix, yPix)
    hardware = coords.opdbToHardware(pd.DataFrame({"agc_center_x_mm": x_mm, "agc_center_y_mm": y_mm}))

    xDet, yDet = det2dp(agcCameraId, xPix, yPix)
    assert np.max(np.hypot(hardware.agc_center_x_mm - xDet, hardware.agc_center_y_mm - yDet)) < 0.1
    # Negative control: unflipped, they disagree by 14 to 436 mm.
    assert np.max(np.hypot(x_mm - xDet, y_mm - yDet)) > 10


@pytest.mark.parametrize("agcCameraId", range(6))
def testCameraCentersAreHardware(agcCameraId):
    """AGC_CAMERA_CENTERS_MM are within 5 mm of det2dp's detector centers."""
    xc, yc = coords.AGC_CAMERA_CENTERS_MM[agcCameraId]
    x, y = det2dp(agcCameraId, 535.5, 520.5)
    assert np.hypot(x - xc, y - yc) < 5


@pytest.mark.parametrize(
    ("x_mm", "y_mm", "insrot_deg", "dz_mm", "dp_mm"),
    [
        (1, 0, 0, -1, 0),  # +x is Front at InR = 0: away from the zenith
        (0, 1, 0, 0, 1),  # +y is Opt
        (1, 0, 90, 0, 1),  # +x is Opt at InR = 90
        (0, 1, 90, 1, 0),  # +y is Rear: towards the zenith
    ],
)
def testPfiToZenithDirections(x_mm, y_mm, insrot_deg, dz_mm, dp_mm):
    """Hardware axes point where Subaru_POPT2_PFS's frame definitions say."""
    dz, dp = coords.pfiToZenith(x_mm, y_mm, insrot_deg)
    assert (dz, dp) == pytest.approx((dz_mm, dp_mm))


def testPfiToZenithMatchesDrpStella(rng):
    """pfiToZenith(x, y) is drp_stella's ag_pfimm_to_zenith_offset(x, -y)."""
    x, y = rng.uniform(-250, 250, (2, 20))
    insrot = rng.uniform(-180, 180, 20)
    xfp, yfp = PFS().pfi2fp(x, -y, insrot)
    np.testing.assert_allclose(coords.pfiToZenith(x, y, insrot), (yfp, xfp))
    # Negative control: drp_stella's function given hardware y, unnegated.
    xfp, yfp = PFS().pfi2fp(x, y, insrot)
    assert not np.allclose(coords.pfiToZenith(x, y, insrot), (yfp, xfp))


def testZenithRoundTrip(rng):
    x, y = rng.uniform(-250, 250, (2, 20))
    insrot = rng.uniform(-180, 180, 20)
    dz, dp = coords.pfiToZenith(x, y, insrot)

    assert np.allclose(np.hypot(dz, dp), np.hypot(x, y))
    np.testing.assert_allclose(coords.zenithToPfi(dz, dp, insrot), (x, y), atol=1e-12)


def testZenithIsLinear(rng):
    """Offsets convert like positions: the rotator axis is at the origin."""
    x, y = rng.uniform(-250, 250, (2, 20))
    dx, dy = rng.normal(0, 0.03, (2, 20))
    insrot = 37.0

    np.testing.assert_allclose(coords.pfiToZenith(0, 0, insrot), (0, 0), atol=1e-12)
    centers = np.array(coords.pfiToZenith(x + dx, y + dy, insrot))
    nominals = np.array(coords.pfiToZenith(x, y, insrot))
    np.testing.assert_allclose(centers - nominals, coords.pfiToZenith(dx, dy, insrot), atol=1e-12)


def testRotXY(rng):
    """Positive angles rotate +x towards +y."""
    assert coords.rotXY(np.pi / 2, 1, 0) == pytest.approx((0, 1))

    x, y = rng.normal(size=(2, 10))
    angle = rng.uniform(-np.pi, np.pi, 10)
    np.testing.assert_allclose(coords.rotXY(-angle, *coords.rotXY(angle, x, y)), (x, y))


# Signs


def testOffsetSign():
    """Offsets are center minus reference, in microns."""
    agcData = pd.DataFrame(
        {
            "pfs_visit_id": [1],
            "agc_exposure_id": [1],
            "guide_star_id": [1],
            "agc_center_x_mm": [100.010],
            "agc_center_y_mm": [-50.000],
            "agc_nominal_x_mm": [100.000],
            "agc_nominal_y_mm": [-49.980],
        }
    )
    agcData = coords.addOffsets(agcData, "nominal")
    assert agcData.dx_nominal_um.iloc[0] == pytest.approx(10)
    assert agcData.dy_nominal_um.iloc[0] == pytest.approx(-20)


@pytest.mark.parametrize("stat", ["mean", "median"])
@pytest.mark.parametrize("reference", coords.REFERENCES)
def testAddOffsets(makeAgcData, reference, stat):
    """Each reference is the per-star (and per-visit) stat of its source column."""
    agcData = makeAgcData()
    original = agcData.copy()

    result = coords.addOffsets(agcData, reference, stat)

    pd.testing.assert_frame_equal(agcData, original)
    pd.testing.assert_index_equal(result.index, agcData.index)  # duplicate labels and all
    pd.testing.assert_frame_equal(result[agcData.columns], agcData)

    for xy in "xy":
        refColumn = f"agc_{reference}_{xy}_mm"
        if reference == "nominal":
            expected = agcData[refColumn].to_numpy()
        else:
            source = f"agc_{'nominal' if reference.startswith('nominal') else 'center'}_{xy}_mm"
            keys = ["pfs_visit_id", "guide_star_id"] if reference.endswith("_visit") else ["guide_star_id"]
            stats = agcData.groupby(keys)[source].agg(stat).rename(refColumn).reset_index()
            expected = agcData[keys].merge(stats, on=keys, how="left")[refColumn].to_numpy()
        np.testing.assert_allclose(result[refColumn], expected)

        offset = result[f"d{xy}_{reference}_um"]
        np.testing.assert_allclose(offset, 1e3 * (agcData[f"agc_center_{xy}_mm"].to_numpy() - expected))


@pytest.mark.parametrize(("stat", "row"), [("first", 0), ("last", -1)])
def testFirstLastByExposure(makeAgcData, stat, row):
    """``first`` and ``last`` follow agc_exposure_id, not the (shuffled) row order."""
    result = coords.addReferencePositions(makeAgcData(), "center0_visit", stat)
    nUnordered = 0
    for _, group in result.groupby(["pfs_visit_id", "guide_star_id"]):
        expected = group.sort_values("agc_exposure_id").agc_center_x_mm.iloc[row]
        np.testing.assert_array_equal(group.agc_center0_visit_x_mm, expected)
        nUnordered += group.agc_center_x_mm.iloc[row] != expected
    # Negative control: taking the row order would give a different answer.
    assert nUnordered > 0


def testReferencePositionsRecomputed(makeAgcData):
    agcData = makeAgcData()
    means = coords.addReferencePositions(agcData, "center0", "mean")
    medians = coords.addReferencePositions(means, "center0", "median")
    expected = coords.addReferencePositions(agcData, "center0", "median")
    pd.testing.assert_series_equal(medians.agc_center0_x_mm, expected.agc_center0_x_mm)


def testColumnNames():
    assert coords.referenceColumns("nominal0_visit") == ("agc_nominal0_visit_x_mm", "agc_nominal0_visit_y_mm")
    assert coords.offsetColumns("center0") == ("dx_center0_um", "dy_center0_um")


def testUnknownReferenceOrStat(makeAgcData):
    agcData = makeAgcData(nVisit=1, nExp=2)
    with pytest.raises(ValueError, match="Unknown reference"):
        coords.referenceColumns("center")
    with pytest.raises(ValueError, match="Unknown reference"):
        coords.addOffsets(agcData, "boresight")
    with pytest.raises(ValueError, match="Unknown stat"):
        coords.addReferencePositions(agcData, "center0", "max")
