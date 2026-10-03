"""Regression tests of ``pfs.drp.qa.guiders`` on real AG data.

The data are engineering visits of Run 30 (2026-08-31), read with
`pfs.drp.qa.guiders.queries.readAgcData` and trimmed to a few guide stars per
camera: ``data/``, made by ``data/makeGuiderFixtures.py``. The `realAgcData`
and `realAgcStars` fixtures read them.

The bounds are set from the values these data give, quoted in each test, with
room for the trimming and for library versions. Each test has a negative
control showing that a wrong convention fails the same check.
"""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.guiders.analysis import (
    FOCUS_PISTON_OFFSET_MM,
    GUIDE_STAR_EPOCH,
    GuiderFitConfig,
    addImageSizes,
    agActorPositions,
    comparePfsUtilsPositions,
    estimateFocusErrors,
    fitDriftRate,
    fitGuiderModel,
    guideErrorsByExposure,
    selectValidMatches,
)
from pfs.drp.qa.guiders.coordinates import AGC_CAMERA_CENTERS_MM, addReferencePositions, opdbToHardware
from pfs.drp.qa.guiders.queries import AGC_DATA_COLUMNS
from pfs.utils.coordinates import CoordTransp
from pfs.utils.datamodel.ag import SourceDetectionFlags

NAMES = ["focusSweep", "raster", "allSky"]

# The raster scan's positions: 148284 and 148291 at the center, 148285-148290 on a hexagon.
RASTER_CENTER_VISITS = [148284, 148291]
RASTER_HEXAGON_VISITS = list(range(148285, 148291))

# M2_OFF3 near best focus (mm), and the range about it in which the focus error is linear.
BEST_FOCUS_MM = 3.29
LINEAR_FOCUS_RANGE_MM = 0.2


def rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def distanceFromNominal_um(agcData: pd.DataFrame) -> pd.Series:
    return 1e3 * np.hypot(
        agcData.agc_center_x_mm - agcData.agc_nominal_x_mm, agcData.agc_center_y_mm - agcData.agc_nominal_y_mm
    )


# The data as read


@pytest.mark.parametrize("name", NAMES)
def testRealDataColumns(realAgcData, name):
    agcData = realAgcData(name)

    assert list(agcData.columns) == list(AGC_DATA_COLUMNS)
    order = ["agc_exposure_id", "agc_camera_id", "spot_id"]
    pd.testing.assert_frame_equal(agcData, agcData.sort_values(order, ignore_index=True))
    with pytest.raises(ValueError, match="already in hardware coordinates"):
        opdbToHardware(agcData)

    valid = selectValidMatches(agcData)
    assert valid.any() and not valid.all()
    assert set(agcData.shutter_open) >= {0, 1}


@pytest.mark.parametrize("name", NAMES)
def testRealDataHardwareFrame(realAgcData, name):
    """Each camera's stars are at its center in hardware coordinates (within 10 mm here).

    The cameras off the x axis (AG2, AG3, AG5, AG6) give the sign of y.
    """
    agcData = realAgcData(name)
    nominal = agcData.groupby("agc_camera_id")[["agc_nominal_x_mm", "agc_nominal_y_mm"]].median()
    centers = pd.DataFrame.from_dict(AGC_CAMERA_CENTERS_MM, orient="index", columns=["x", "y"])

    assert list(nominal.index) == list(range(6))
    distance = np.hypot(nominal.agc_nominal_x_mm - centers.x, nominal.agc_nominal_y_mm - centers.y)
    assert distance.max() < 15

    # Negative control: in the opdb's frame (y negated) the cameras off the x axis are 420 mm away.
    flipped = np.hypot(nominal.agc_nominal_x_mm - centers.x, -nominal.agc_nominal_y_mm - centers.y)
    assert flipped[[1, 2, 4, 5]].min() > 400


@pytest.mark.parametrize("name", NAMES)
def testRealDataInvalidMatches(realAgcData, name):
    """Invalid matches are far from where the guider expected the star.

    Median distances from the nominal position, valid and invalid: 28 and
    204 µm (focusSweep), 16 and 768 µm (raster), 34 and 685 µm (allSky).
    """
    agcData = realAgcData(name)
    distance = distanceFromNominal_um(agcData)
    valid = selectValidMatches(agcData)

    assert distance[~valid].median() > 5 * distance[valid].median()


# Focus


def estimateFocusErrorsWithAgActor(agcData: pd.DataFrame) -> pd.DataFrame:
    """Return the focus error of each AG exposure, with the AG actor's."""
    focusErrors = estimateFocusErrors(agcData, byCamera=False)
    agActor = agcData.groupby("agc_exposure_id").guide_delta_z.first()
    # The AG actor subtracts FOCUS_PISTON_OFFSET_MM; estimateFocusErrors doesn't.
    focusErrors["ag_actor_focus_error_um"] = 1e3 * (
        focusErrors.agc_exposure_id.map(agActor) + FOCUS_PISTON_OFFSET_MM
    )

    return focusErrors


@pytest.fixture
def focusSweep(realAgcData) -> pd.DataFrame:
    """Return the focus sweep's AG data, with image sizes."""
    return addImageSizes(realAgcData("focusSweep"))


def fitFocusLine(focusErrors: pd.DataFrame, column: str = "focus_error_um") -> tuple[float, float]:
    """Fit a line to the focus errors near best focus; return its slope (µm/mm) and zero (mm)."""
    near = focusErrors[(focusErrors.focus_position_mm - BEST_FOCUS_MM).abs() <= LINEAR_FOCUS_RANGE_MM]
    near = near.dropna(subset=column)
    slope, intercept = np.polyfit(near.focus_position_mm, near[column], 1)

    return float(slope), float(-intercept / slope)


def testRealDataFocusSweep(focusSweep):
    """The focus error rises through focus, as the AG actor's does.

    The median focus error at each M2_OFF3 rises from -466 µm at 2.725 mm to
    277 µm at 3.55 mm, linearly within 0.2 mm of focus: 606 µm/mm, zero at
    3.278 mm. The AG actor's give 673 µm/mm and 3.291 mm.
    """
    focusErrors = estimateFocusErrorsWithAgActor(focusSweep)
    medians = focusErrors.groupby(focusErrors.focus_position_mm.round(3)).focus_error_um.median()
    assert len(medians) == 10
    assert medians.is_monotonic_increasing
    assert medians.iloc[0] < -300 and medians.iloc[-1] > 200

    slope, bestFocus = fitFocusLine(focusErrors)
    assert 500 < slope < 750
    assert bestFocus == pytest.approx(BEST_FOCUS_MM, abs=0.04)

    agActorSlope, agActorBestFocus = fitFocusLine(focusErrors, "ag_actor_focus_error_um")
    assert slope == pytest.approx(agActorSlope, rel=0.2)
    assert bestFocus == pytest.approx(agActorBestFocus, abs=0.03)

    # Negative control: with the detectors' halves swapped, the slope has the wrong sign.
    swapped = estimateFocusErrors(focusSweep.assign(left=~focusSweep.left), byCamera=False)
    swappedSlope, _ = fitFocusLine(swapped)
    assert swappedSlope < -500


def testRealDataFocusPairing(focusSweep):
    """M2_OFF3 changes in the same AG exposure as the focus error.

    In visit 148266 M2_OFF3 goes from 3.25 to 2.725 mm after eight AG
    exposures, and the focus error from -31 to 13 µm to -512 to -365 µm.
    ``m2_off3`` comes from tel_status, which `readAgcData` pairs with the AG
    exposures by order.
    """
    focusErrors = estimateFocusErrors(focusSweep, byCamera=False)
    visit = focusErrors[focusErrors.pfs_visit_id == 148266].sort_values("agc_exposure_id")
    error = visit.focus_error_um.to_numpy()

    moved = visit.focus_position_mm.to_numpy() < 3
    assert moved.sum() == 4
    assert error[~moved].min() > -100
    assert error[moved].max() < -300

    # Negative control: paired one exposure late, an exposure at 2.725 mm is said to be at 3.25 mm.
    late = visit.focus_position_mm.shift(1).to_numpy() < 3
    assert error[~late].min() < -300


# Boresight, guide errors and shutters


def testRealDataRasterBoresight(realAgcData):
    """The boresight reference follows the raster scan from visit to visit.

    With the shutters open, the scan's visits are at the center (148284,
    148291) or on a hexagon of radius 92-96 µm (148285-148290), at 90, 150,
    -150, -90, -30 and 30 degrees.
    """
    agcData = realAgcData("raster")
    # After its shutters close, each visit's AG exposures move on to the next position.
    agcData = agcData[(agcData.shutter_open == 1).to_numpy() & selectValidMatches(agcData)]
    agcData = addReferencePositions(addReferencePositions(agcData, "boresight"), "nominal0")

    shift = (
        pd.DataFrame(
            {
                "dx_um": 1e3 * (agcData.agc_boresight_x_mm - agcData.agc_nominal0_x_mm),
                "dy_um": 1e3 * (agcData.agc_boresight_y_mm - agcData.agc_nominal0_y_mm),
                "pfs_visit_id": agcData.pfs_visit_id,
            }
        )
        .groupby("pfs_visit_id")[["dx_um", "dy_um"]]
        .first()
    )
    radius = np.hypot(shift.dx_um, shift.dy_um)
    angle = np.rad2deg(np.arctan2(shift.dy_um, shift.dx_um))

    assert radius[RASTER_CENTER_VISITS].max() < 5
    np.testing.assert_allclose(radius[RASTER_HEXAGON_VISITS], 94, atol=6)
    expected = [90, 150, -150, -90, -30, 30]
    np.testing.assert_allclose(
        np.angle(np.exp(1j * np.deg2rad(angle[RASTER_HEXAGON_VISITS] - expected))), 0, atol=0.1
    )

    # Negative control: nominal0 doesn't follow the scan. Nominal positions minus each reference:
    # 3.8 µm rms for boresight, 72 µm for nominal0.
    def residual(reference):
        return rms(
            1e3
            * np.hypot(
                agcData.agc_nominal_x_mm - agcData[f"agc_{reference}_x_mm"],
                agcData.agc_nominal_y_mm - agcData[f"agc_{reference}_y_mm"],
            )
        )

    assert residual("boresight") < 6
    assert residual("nominal0") > 50


def testRealDataShutterOpen(realAgcData):
    """AG exposures taken with the shutters closed have the larger guide errors.

    In the raster scan the telescope moves to the next position after the
    shutters close, and the guider takes about 40 s to catch up. Median guide
    errors: 38 µm with the shutters closed, 12 µm open. ``shutter_open``
    compares agc_exposure and sps_exposure times in the opdb, so needs them
    on one clock.
    """
    agcData = realAgcData("raster")
    guideErrors = guideErrorsByExposure(agcData)
    exposures = agcData.groupby("agc_exposure_id").agg(
        shutter_open=("shutter_open", "first"), pfs_visit_id=("pfs_visit_id", "first")
    )

    medians = guideErrors.groupby(exposures.shutter_open).median()
    assert medians[0] > 2 * medians[1]

    # Negative control: with the flags three AG exposures (30 s) late, closed and open look alike.
    late = exposures.groupby("pfs_visit_id").shutter_open.shift(3)
    medians = guideErrors.groupby(late).median()
    assert medians[0] < 1.5 * medians[1]


def testRealDataFitGuiderModel(realAgcData):
    """The boresight and camera models take out most of the raster scan's guide errors.

    The rms of the mean guide error of each AG exposure and camera is 58 µm
    with no model, 8.1 µm with the boresight model, and 5.5 µm with both.
    """

    def guideErrorRms(agcData, **kwargs):
        config = GuiderFitConfig(maxGuideError_um=0, maxPosError_um=0, **kwargs)
        guideErrors = fitGuiderModel(agcData, config).guideErrorByCamera
        return rms(np.hypot(guideErrors.dx_model_um, guideErrors.dy_model_um))

    agcData = realAgcData("raster")
    raw = guideErrorRms(agcData, modelBoresightOffset=False, modelCCDOffset=False)
    boresight = guideErrorRms(agcData, modelCCDOffset=False)
    both = guideErrorRms(agcData)

    assert raw > 40
    assert boresight < 12
    assert both < 8
    assert both < boresight


def testRealDataFitGuiderModelValidMatches(realAgcData):
    """Only valid matches are fitted.

    In the all-sky exposure the valid matches' residuals from the models are
    20 µm rms; fitting the invalid ones too makes them 4 mm.
    """
    agcData = realAgcData("allSky")
    config = GuiderFitConfig(maxGuideError_um=0, maxPosError_um=0)
    fit = fitGuiderModel(agcData, config).agcData
    valid = fit.selected.to_numpy()

    assert rms(fit.dr_model_um[valid]) < 30

    # Negative control: every match counted as valid.
    fitAll = fitGuiderModel(agcData.assign(agc_match_flags=1), config).agcData
    assert rms(fitAll.dr_model_um[valid]) > 500


# Drift


def testRealDataDriftRate(realAgcData):
    """The guided stars of the 900 s all-sky exposure don't drift.

    Drift rates: -0.004 µm/min radial, 0.22 tangential; -0.09 in x, -0.09 in y.
    """
    agcData = realAgcData("allSky")

    rates = fitDriftRate(agcData).rates
    assert abs(rates.radial_rate_um_per_min) < 1
    assert abs(rates.tangential_rate_um_per_min) < 1

    rates = fitDriftRate(agcData, radialTangential=False).rates
    assert abs(rates.x_rate_um_per_min) < 1
    assert abs(rates.y_rate_um_per_min) < 1

    # Negative control: a drift of 2 µm/min in x added to the stars' centers is measured.
    minutes = (agcData.taken_at - agcData.taken_at.min()).dt.total_seconds() / 60
    drifting = agcData.assign(agc_center_x_mm=agcData.agc_center_x_mm + 2e-3 * minutes)
    driftRates = fitDriftRate(drifting, radialTangential=False).rates
    assert driftRates.x_rate_um_per_min - rates.x_rate_um_per_min == pytest.approx(2, abs=0.1)
    assert driftRates.y_rate_um_per_min == pytest.approx(rates.y_rate_um_per_min)


# Comparison with pfs_utils

# The visits with their guide stars in data/, at rotator angles of -165 and 103 degrees.
STAR_VISITS = [("allSky", 148258), ("raster", 148291)]


def withGuideStars(agcData: pd.DataFrame, agcStars: pd.DataFrame) -> pd.DataFrame:
    """Return the rows of AG data with the AG actor's inputs, and their guide stars' catalogue entries."""
    columns = [
        "guide_star_id",
        "guide_star_ra",
        "guide_star_dec",
        "guide_star_pm_ra",
        "guide_star_pm_dec",
        "guide_star_parallax",
    ]
    agcData = agcData.dropna(subset=["guide_ra", "guide_dec", "guide_pa", "adc_pa", "m2_pos3"])
    return agcData.merge(agcStars[columns], on="guide_star_id")


def distanceFromPfsUtils_um(stars: pd.DataFrame) -> pd.Series:
    return 1e3 * np.hypot(
        stars.pfs_utils_x_mm - stars.agc_nominal_x_mm, stars.pfs_utils_y_mm - stars.agc_nominal_y_mm
    )


@pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")  # pfs_utils's clamped parallaxes
@pytest.mark.parametrize("name, visit", STAR_VISITS)
def testRealDataAgActorPositions(realAgcData, realAgcStars, name, visit):
    """pfs_utils's model, with the AG actor's inputs, gives the guider's nominal positions.

    Every row: at most 0.14 µm from the opdb's position (allSky), 0.15 µm
    (raster, whose visits all have 148291's design, through the raster
    offsets).
    """
    agcData = withGuideStars(realAgcData(name), realAgcStars(visit))
    assert agcData.agc_exposure_id.nunique() > 90

    x_mm, y_mm = agActorPositions(agcData)

    distance = 1e3 * np.hypot(x_mm - agcData.agc_nominal_x_mm, y_mm - agcData.agc_nominal_y_mm)
    assert distance.max() < 1


@pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")
@pytest.mark.parametrize("name, visit", STAR_VISITS)
def testRealDataComparePfsUtils(realAgcData, realAgcStars, name, visit):
    """The comparison with pfs_utils agrees with the guider, and leaves its inputs alone.

    Each star's median positions over the first 10 AG exposures: at most
    0.02 µm apart.
    """
    agcData = realAgcData(name)
    agcData = agcData[agcData.pfs_visit_id == visit]
    agcStars = realAgcStars(visit)
    original = (agcData.copy(), agcStars.copy())

    stars = comparePfsUtilsPositions(agcStars, agcData).stars

    assert len(stars) >= 10
    assert distanceFromPfsUtils_um(stars).max() < 1
    pd.testing.assert_frame_equal(agcData, original[0])
    pd.testing.assert_frame_equal(agcStars, original[1])  # pfs_utils clamps parallaxes in place

    # Negative control: pfs_utils given the opdb's HST as if UTC, as drp_stella did. 108-692 µm
    # (148258) and 30-116 µm (148291) off.
    hst = agcData.assign(taken_at=agcData.taken_at - pd.Timedelta(hours=10))
    assert distanceFromPfsUtils_um(comparePfsUtilsPositions(agcStars, hst).stars).max() > 100

    # Negative control: the halves of the detectors swapped; 28 µm off.
    swapped = agcData.assign(agc_data_flags=agcData.agc_data_flags ^ int(SourceDetectionFlags.RIGHT))
    assert distanceFromPfsUtils_um(comparePfsUtilsPositions(agcStars, swapped).stars).min() > 20

    # Negative control: drp_stella's transform, pfs_utils's model for the cobras (sky_pfi), with the
    # same field center, position angle and time; 534-562 µm rms off. Its y has the opdb's sign.
    takenAt = stars.taken_at.iloc[0] + pd.Timedelta(hours=10)
    cobras = CoordTransp.CoordinateTransform(
        np.array([stars.guide_star_ra, stars.guide_star_dec], dtype=float),
        "sky_pfi",
        cent=np.array([[stars.guide_ra.iloc[0]], [stars.guide_dec.iloc[0]]]),
        pa=stars.guide_pa.iloc[0],
        time=takenAt.strftime("%Y-%m-%d %H:%M:%S"),
        pm=np.array([stars.guide_star_pm_ra, stars.guide_star_pm_dec], dtype=float),
        par=np.array(stars.guide_star_parallax, dtype=float),
        epoch=GUIDE_STAR_EPOCH,
    )
    cobraStars = stars.assign(pfs_utils_x_mm=cobras[0], pfs_utils_y_mm=-cobras[1])
    assert rms(distanceFromPfsUtils_um(cobraStars)) > 400
