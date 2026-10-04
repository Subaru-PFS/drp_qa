"""Tests for ``pfs.drp.qa.guiders.analysis``, on synthetic AG data.

Each fix of drp_stella's guider analysis has a test here, with a negative
control showing that drp_stella's version fails the same check.
"""

import dataclasses
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.guiders import analysis
from pfs.drp.qa.guiders.analysis import (
    FOCUS_PISTON_MM_PER_PIX2,
    GuiderFitConfig,
    MeasureXYRot,
    addImageSizes,
    averageByFocusPosition,
    comparePfsUtilsPositions,
    correctAgActorFocus,
    estimateFocusErrors,
    estimateGuideErrors,
    fitDriftRate,
    fitGlobalModel,
    fitGuiderModel,
    fitTransform,
    guideErrorsByExposure,
    momentDifferenceToPiston,
    selectGoodDetections,
    selectIsolatedGaiaStars,
    selectStars,
    selectValidMatches,
    smoothAgcData,
)
from pfs.drp.qa.guiders.coordinates import (
    AGC_PIXEL_SIZE_UM,
    AGC_PLATE_SCALE_UM_PER_ARCSEC,
    addOffsets,
    pfiToZenith,
    rotXY,
    zenithToPfi,
)
from pfs.utils.datamodel.ag import SourceCatalogFlags, SourceDetectionFlags, SourceMatchingFlags

VISITS = [120000, 120001, 120002]


def rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def fitConfig(**kwargs) -> GuiderFitConfig:
    """Return a GuiderFitConfig that uses every exposure and star."""
    kwargs.setdefault("maxGuideError_um", 0)
    kwargs.setdefault("maxPosError_um", 0)
    return GuiderFitConfig(**kwargs)


def guideErrorRms(fit) -> float:
    """Return the rms of the per-camera mean guide errors (microns)."""
    return rms(np.hypot(fit.guideErrorByCamera.dx_model_um, fit.guideErrorByCamera.dy_model_um))


# No shared state, and inputs left alone


def testNoMutableDefaults():
    """No function or config has a mutable default (drp_stella's GuiderConfig and plotDriftRate had)."""
    for name in analysis.__all__:
        obj = getattr(analysis, name)
        if dataclasses.is_dataclass(obj):
            for field in dataclasses.fields(obj):
                assert field.default_factory is dataclasses.MISSING, f"{name}.{field.name}"
                assert not isinstance(field.default, dict | list | set), f"{name}.{field.name}"
        if callable(obj):
            for parameter in inspect.signature(obj).parameters.values():
                default = parameter.default
                assert not isinstance(default, dict | list | set | pd.DataFrame), f"{name}({parameter.name})"


def testConfigIsFrozen():
    config = GuiderFitConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.maxGuideError_um = 10
    with pytest.raises(ValueError, match="agcExposureStride must be at least 1"):
        GuiderFitConfig(agcExposureStride=0)


CALLS = {
    "fitGuiderModel": lambda data: fitGuiderModel(data, fitConfig()),
    "fitGlobalModel": lambda data: fitGlobalModel(data, fitAgcOffsets=True, fitAgcRotation=True),
    "guideErrorsByExposure": lambda data: guideErrorsByExposure(data, "center0"),
    "selectStars": lambda data: selectStars(data, GuiderFitConfig(), 1, guideErrorsByExposure(data)),
    "smoothAgcData": lambda data: smoothAgcData(data, 3),
    "estimateGuideErrors": lambda data: estimateGuideErrors(
        data, "boresight", smoothing=3, subtractMedian=True
    ),
    "fitDriftRate": lambda data: fitDriftRate(data, "nominal0", smoothing=3),
    "addImageSizes": lambda data: addImageSizes(data),
    "correctAgActorFocus": lambda data: correctAgActorFocus(data),
    "estimateFocusErrors": lambda data: estimateFocusErrors(addImageSizes(data)),
}


@pytest.mark.parametrize("name", CALLS)
def testInputUnchanged(makeAgcData, name):
    """No function modifies its input; drp_stella's showGuiderErrors and plotDriftRate did."""
    agcData = makeAgcData(stars=True)
    original = agcData.copy()

    CALLS[name](agcData)

    pd.testing.assert_frame_equal(agcData, original)


# Selection


def testSelectStarsMatchesGuideErrorsByKey(makeAgcData):
    """The guide error cut hits the stars of the exposures that fail it, whatever the row order."""
    agcData = makeAgcData()
    guideErrors = guideErrorsByExposure(agcData)
    maxGuideError_um = guideErrors.median()  # cut about half of the exposures
    config = GuiderFitConfig(maxGuideError_um=maxGuideError_um, onlyShutterOpen=False)

    selected = selectStars(agcData, config, guideErrors=guideErrors)

    dr = np.hypot(
        agcData.agc_center_x_mm - agcData.agc_nominal_x_mm, agcData.agc_center_y_mm - agcData.agc_nominal_y_mm
    )
    expected = (1e3 * dr).groupby(agcData.agc_exposure_id.to_numpy()).transform("mean") < maxGuideError_um
    np.testing.assert_array_equal(selected, expected.to_numpy())
    assert 0 < selected.sum() < len(selected)

    # Negative control: drp_stella merged the guide errors onto the rows, which reorders them.
    merged = guideErrors.rename("guide_error_um").reset_index().merge(agcData, on="agc_exposure_id")
    assert ((merged.guide_error_um.to_numpy() < maxGuideError_um) != expected.to_numpy()).any()


def testSelectStarsCuts(makeAgcData):
    agcData = makeAgcData()
    exposures = np.sort(agcData.agc_exposure_id.unique())
    closed = agcData.groupby("agc_exposure_id").shutter_open.max().loc[exposures].to_numpy() == 0
    first = np.argmin(closed)  # the first open exposure starts the stride

    config = GuiderFitConfig(agcExposureStride=3)
    selected = selectStars(agcData, config, agc_camera_id=2)

    expected = set(exposures[first::3]) - set(exposures[closed])
    assert set(agcData.agc_exposure_id[selected]) == expected
    assert set(agcData.agc_camera_id[selected]) == {2}

    config = GuiderFitConfig(onlyShutterOpen=False, pfsVisitIdMin=VISITS[1], agcExposureIdMax=exposures[-2])
    selected = selectStars(agcData, config)
    assert set(agcData.pfs_visit_id[selected]) == set(VISITS[1:])
    assert agcData.agc_exposure_id[selected].max() == exposures[-2]


def invalidate(agcData, shift_mm=1.0):
    """Return a copy of AG data with every fifth guide star's matches invalid, and its centers moved."""
    agcData = agcData.copy()
    invalid = (agcData.guide_star_id % 5 == 1).to_numpy()
    agcData.loc[invalid, "agc_match_flags"] = 0
    agcData.loc[invalid, "agc_center_x_mm"] += shift_mm
    return agcData, invalid


def testInvalidMatchesNotFitted(makeAgcData):
    """Invalid matches are kept but neither fitted nor averaged; Copilot's review of drp_qa#84."""
    agcData, invalid = invalidate(makeAgcData())

    fit = fitGuiderModel(agcData, fitConfig())
    assert len(fit.agcData) == len(agcData)
    assert fit.agcData.agc_model_x_mm[invalid].notna().all()
    assert not fit.agcData.selected[invalid].any()
    assert not selectStars(agcData)[invalid].any()
    assert guideErrorRms(fit) < 3

    clean = guideErrorsByExposure(agcData[~invalid])
    pd.testing.assert_series_equal(guideErrorsByExposure(agcData), clean)

    # Negative control: counted as valid, the moved stars spoil the per-camera guide errors.
    asValid = agcData.assign(agc_match_flags=1)
    assert guideErrorRms(fitGuiderModel(asValid, fitConfig())) > 10
    assert not np.allclose(guideErrorsByExposure(asValid), clean)


def testSelectFlags():
    flags = pd.DataFrame(
        {
            "agc_match_flags": [1, 0, int(SourceMatchingFlags.GOOD_MATCH | SourceMatchingFlags.BAD_RESIDUAL)],
            "agc_data_flags": [0, int(SourceDetectionFlags.RIGHT), int(SourceDetectionFlags.EDGE)],
            "guide_star_flag": [
                int(SourceCatalogFlags.GAIA | SourceCatalogFlags.NON_BINARY),
                int(SourceCatalogFlags.GAIA),
                int(SourceCatalogFlags.GAIA | SourceCatalogFlags.NON_BINARY | SourceCatalogFlags.GALAXY),
            ],
        }
    )
    np.testing.assert_array_equal(selectGoodDetections(flags), [True, True, False])
    np.testing.assert_array_equal(selectIsolatedGaiaStars(flags), [True, False, False])
    np.testing.assert_array_equal(selectValidMatches(flags), [True, False, False])


# Boresight and per-camera transforms


def testFitTransform(rng):
    """The fitted transform takes measured positions to true ones, and back."""
    x, y = rng.uniform(-250, 250, (2, 50))
    xTrue, yTrue = rotXY(np.deg2rad(1 / 60), x, y)
    xTrue, yTrue = xTrue + 0.03, yTrue - 0.02

    transform = fitTransform(x, y, xTrue, yTrue, nsigma=0)  # without noise, clipping would be erratic

    np.testing.assert_allclose(transform.distort(x, y), (xTrue, yTrue), atol=1e-4)
    np.testing.assert_allclose(transform.distort(xTrue, yTrue, inverse=True), (x, y), atol=1e-4)
    np.testing.assert_allclose(transform.getArgs()[:3], [0.03, -0.02, -1 / 60], atol=1e-5)


def testFitGuiderModelUnits(makeAgcData):
    """The boresight model removes each exposure's pointing error.

    drp_stella gave MeasureXYRot positions (mm) where it expects offsets
    (microns), making the residuals worse.
    """
    agcData = makeAgcData()

    raw = guideErrorRms(fitGuiderModel(agcData, fitConfig(modelBoresightOffset=False, modelCCDOffset=False)))
    boresight = guideErrorRms(fitGuiderModel(agcData, fitConfig(modelCCDOffset=False)))
    both = guideErrorRms(fitGuiderModel(agcData, fitConfig()))

    assert raw > 50
    assert boresight < 3  # the per-camera mean of the noise is ~1 micron
    assert both < 3

    # Negative control: drp_stella's call, MeasureXYRot(center, nominal) with the nominal positions as offsets.
    data = agcData.reset_index(drop=True)
    xModel, yModel = data.agc_nominal_x_mm.to_numpy(copy=True), data.agc_nominal_y_mm.to_numpy(copy=True)
    for rows in data.groupby("agc_exposure_id").groups.values():
        rows = data.index.isin(rows)
        transform = MeasureXYRot(
            data.agc_center_x_mm[rows], data.agc_center_y_mm[rows], xModel[rows], yModel[rows], nsigma=3
        ).fit()
        xModel[rows], yModel[rows] = transform.distort(xModel[rows], yModel[rows], inverse=True)
    data = data.assign(agc_model_x_mm=xModel, agc_model_y_mm=yModel)
    mismatched = addOffsets(data, "model").groupby(["agc_exposure_id", "agc_camera_id"])
    assert rms(np.hypot(mismatched.dx_model_um.mean(), mismatched.dy_model_um.mean())) > raw


def testFitGuiderModelCameraOwnStars(makeAgcData):
    """Each camera's transform is fitted to its own stars.

    drp_stella fitted every camera's transform to all the stars.
    """
    agcData = makeAgcData()
    agcData.loc[(agcData.agc_camera_id == 2).to_numpy(), "agc_center_x_mm"] += 0.020

    fit = fitGuiderModel(agcData, fitConfig())
    camera2 = fit.guideErrorByCamera[fit.guideErrorByCamera.agc_camera_id == 2]
    assert abs(camera2.dx_model_um.mean()) < 1
    assert guideErrorRms(fit) < 3

    # Negative control: a transform fitted to all the cameras' stars leaves most of the shift.
    boresight = fitGuiderModel(agcData, fitConfig(modelCCDOffset=False)).agcData
    transform = fitTransform(
        boresight.agc_center_x_mm,
        boresight.agc_center_y_mm,
        boresight.agc_model_x_mm,
        boresight.agc_model_y_mm,
    )
    rows = (boresight.agc_camera_id == 2).to_numpy()
    xModel, _ = transform.distort(
        boresight.agc_model_x_mm[rows], boresight.agc_model_y_mm[rows], inverse=True
    )
    assert np.mean(1e3 * (boresight.agc_center_x_mm[rows] - xModel)) > 10


def testFitGuiderModelRepeatable(makeAgcData):
    """A second fit gives the same answer; drp_stella's re-applied the corrections to its own output."""
    agcData = makeAgcData()
    first = fitGuiderModel(agcData, fitConfig())
    second = fitGuiderModel(agcData, fitConfig())

    pd.testing.assert_frame_equal(first.agcData, second.agcData)
    pd.testing.assert_index_equal(first.agcData.index, agcData.index)
    assert list(first.agcData.agc_exposure_id) == list(agcData.agc_exposure_id)


def testFitGuiderModelTransforms(makeAgcData, rng):
    """Transforms are reused from a previous fit, and refitted with solveForAGTransforms.

    drp_stella shared one dict of transforms between GuiderConfigs, and never
    refitted a cached one.
    """
    previous = fitGuiderModel(makeAgcData(), fitConfig())
    agcData = makeAgcData()  # the same exposures and cameras, new pointing errors
    fresh = fitGuiderModel(agcData, fitConfig())

    assert set(fresh.exposureTransforms) == set(agcData.agc_exposure_id)
    assert set(fresh.cameraTransforms) == set(range(6))
    assert list(fresh.exposureParameters.columns) == [
        "x0_mm",
        "y0_mm",
        "theta_deg",
        "dscale",
        "scale2_per_mm2",
    ]
    assert not set(map(id, fresh.exposureTransforms.values())) & set(
        map(id, previous.exposureTransforms.values())
    )
    with pytest.raises(TypeError):
        fresh.exposureTransforms[0] = None

    reused = fitGuiderModel(agcData, fitConfig(), previous)
    assert all(
        reused.exposureTransforms[k] is previous.exposureTransforms[k] for k in previous.exposureTransforms
    )

    refitted = fitGuiderModel(agcData, fitConfig(solveForAGTransforms=True), previous)
    pd.testing.assert_frame_equal(refitted.exposureParameters, fresh.exposureParameters)
    pd.testing.assert_frame_equal(refitted.agcData, fresh.agcData)

    # Negative control: reusing the stale transforms doesn't fit the new data.
    assert guideErrorRms(reused) > 10 * guideErrorRms(fresh)


def testFitGuiderModelCuts(makeAgcData):
    """The per-camera guide errors average only the selected stars that pass the position cut."""
    agcData = makeAgcData()
    wild = (agcData.guide_star_id == agcData.guide_star_id.iloc[0]).to_numpy()
    agcData.loc[wild, "agc_center_y_mm"] += 0.1

    fit = fitGuiderModel(agcData, fitConfig(maxPosError_um=40))

    assert (
        fit.agcData.selected.to_numpy().tolist()
        == selectStars(
            agcData, fit.config, guideErrors=fit.agcData.groupby("agc_exposure_id").guide_error_um.first()
        ).tolist()
    )
    assert set(fit.agcData.shutter_open[fit.agcData.selected]) == {1}
    nStars = fit.agcData[fit.agcData.selected].groupby(["agc_exposure_id", "agc_camera_id"]).size()
    assert len(fit.guideErrorByCamera) == len(nStars)
    assert guideErrorRms(fit) < 3  # the wild star is cut


# Global model in the zenith frame


def makeZenithData(makeAgcData, rng, cameraOffsets_mm, rotationSigma_rad=1e-5):
    """Return AG data whose errors are an offset and rotation per exposure, and an offset per camera.

    The offsets are in the zenith frame, and the visits are at very different
    rotator angles.
    """
    agcData = makeAgcData(closedFrac=0).reset_index(drop=True)
    agcData["insrot"] = agcData.pfs_visit_id.map(dict(zip(VISITS, [-60.0, 0.0, 70.0], strict=True)))
    agcData["agc_match_flags"] = 1

    exposures = np.sort(agcData.agc_exposure_id.unique())
    terms = pd.DataFrame(
        {"dz": rng.normal(0, 0.03, len(exposures)), "dp": rng.normal(0, 0.03, len(exposures))},
        index=exposures,
    )
    terms["rotation"] = rng.normal(0, rotationSigma_rad, len(exposures))

    dz, dp = pfiToZenith(agcData.agc_nominal_x_mm, agcData.agc_nominal_y_mm, agcData.insrot.to_numpy())
    exposure = terms.loc[agcData.agc_exposure_id]
    errDz = exposure.dz.to_numpy() - exposure.rotation.to_numpy() * dp
    errDp = exposure.dp.to_numpy() + exposure.rotation.to_numpy() * dz
    camera = np.asarray(cameraOffsets_mm)[agcData.agc_camera_id]
    errDz += camera[:, 0]
    errDp += camera[:, 1]

    dx, dy = zenithToPfi(errDz, errDp, agcData.insrot.to_numpy())
    agcData["agc_center_x_mm"] = agcData.agc_nominal_x_mm + dx + rng.normal(0, 1e-4, len(agcData))
    agcData["agc_center_y_mm"] = agcData.agc_nominal_y_mm + dy + rng.normal(0, 1e-4, len(agcData))

    return agcData, terms


def testFitGlobalModelExposures(makeAgcData, rng):
    agcData, terms = makeZenithData(makeAgcData, rng, np.zeros((6, 2)))

    fit = fitGlobalModel(agcData)

    np.testing.assert_allclose(fit.exposures.dz_um, 1e3 * terms.dz, atol=0.1)
    np.testing.assert_allclose(fit.exposures.dp_um, 1e3 * terms.dp, atol=0.1)
    # Positive rotation turns +dz towards +dp.
    np.testing.assert_allclose(fit.exposures.rotation_arcsec, np.rad2deg(terms.rotation) * 3600, atol=0.1)
    np.testing.assert_array_equal(fit.exposures.scale, 0)
    assert fit.cameras is None
    assert rms(np.hypot(fit.agcData.dx_model_um, fit.agcData.dy_model_um)) < 0.2


def testFitGlobalModelHardwareFrame(makeAgcData, rng):
    """Camera offsets constant in the zenith frame are recovered from hardware coordinates.

    drp_stella's fit_global_model_pfimm took the opdb's sign of y.
    """
    cameraOffsets = rng.normal(0, 0.02, (6, 2))
    cameraOffsets -= cameraOffsets.mean(axis=0)  # the exposures' offsets absorb any mean
    # Without exposure rotations, which three alternations of the fits don't separate from the cameras'.
    agcData, _ = makeZenithData(makeAgcData, rng, cameraOffsets, rotationSigma_rad=0)

    fit = fitGlobalModel(agcData, fitPfiRotation=False, fitAgcOffsets=True)

    np.testing.assert_allclose(fit.cameras.dz_um, 1e3 * cameraOffsets[:, 0], atol=0.1)
    np.testing.assert_allclose(fit.cameras.dp_um, 1e3 * cameraOffsets[:, 1], atol=0.1)
    np.testing.assert_array_equal(fit.exposures.rotation_arcsec, 0)
    assert rms(np.hypot(fit.agcData.dx_model_um, fit.agcData.dy_model_um)) < 0.2

    # Negative control: in the opdb's frame the camera offsets turn with the rotator, and don't fit.
    opdbFrame = agcData.assign(
        agc_center_y_mm=-agcData.agc_center_y_mm, agc_nominal_y_mm=-agcData.agc_nominal_y_mm
    )
    wrong = fitGlobalModel(opdbFrame, fitPfiRotation=False, fitAgcOffsets=True)
    assert rms(np.hypot(wrong.agcData.dx_model_um, wrong.agcData.dy_model_um)) > 5


def testFitGlobalModelValid(makeAgcData, rng):
    """Only valid matches are fitted, but every star gets a model position."""
    agcData, _ = makeZenithData(makeAgcData, rng, np.zeros((6, 2)))
    bad = agcData.guide_star_id % 5 == 0
    agcData.loc[bad, "agc_center_x_mm"] += 1.0
    agcData.loc[bad, "agc_match_flags"] = 0

    fit = fitGlobalModel(agcData, fitAgcOffsets=True, fitAgcRotation=True)

    good = fit.agcData[~bad]
    assert rms(np.hypot(good.dx_model_um, good.dy_model_um)) < 0.5
    assert fit.agcData.agc_model_x_mm.notna().all()


def testFitGlobalModelLeavesPandasAlone(makeAgcData, rng):
    """drp_stella appended to the DataFrame class's _metadata list on every call."""
    metadata = list(pd.DataFrame._metadata)
    agcData, _ = makeZenithData(makeAgcData, rng, np.zeros((6, 2)))
    fitGlobalModel(agcData, fitAgcOffsets=True)
    assert pd.DataFrame._metadata == metadata


# Smoothing and guide errors


def testSmoothAgcData(makeAgcData):
    """Each star is smoothed along its own exposures, and IDs and flags are left alone.

    drp_stella smoothed every numeric column across all of a camera's rows.
    """
    agcData = makeAgcData(stars=True)
    agcData["spot_id"] = agcData.guide_star_id % 1000
    smoothed = smoothAgcData(agcData, 3)

    pd.testing.assert_index_equal(smoothed.index, agcData.index)
    for column in (
        "pfs_visit_id",
        "agc_exposure_id",
        "agc_camera_id",
        "spot_id",
        "guide_star_id",
        "shutter_open",
        "taken_at",
        "agc_match_flags",
        "agc_data_flags",
        "guide_star_flag",
    ):
        pd.testing.assert_series_equal(smoothed[column], agcData[column])

    data = agcData.reset_index(drop=True)
    star = data.guide_star_id == 2003
    expected = data[star].sort_values("agc_exposure_id").agc_center_x_mm.rolling(3, 1, center=True).mean()
    np.testing.assert_allclose(smoothed.reset_index(drop=True)[star].agc_center_x_mm, expected.sort_index())
    pd.testing.assert_frame_equal(smoothAgcData(agcData, 1), agcData)

    # Negative control: smoothing a camera's rows together averages the visit IDs.
    byCamera = (
        agcData.groupby("agc_camera_id")[["pfs_visit_id"]].rolling(3, min_periods=1, center=True).mean()
    )
    assert (byCamera.pfs_visit_id % 1 != 0).any()


def testSmoothAgcDataValidMatchesOnly(makeAgcData):
    """Only valid matches are smoothed, along the star's other valid matches; invalid ones are left alone.

    An invalid match 1 mm off would otherwise move its neighbours by 0.33 mm.
    """
    agcData = makeAgcData(nVisit=1, nExp=6).reset_index(drop=True)
    star = (agcData.guide_star_id == 2003).to_numpy()
    rows = agcData[star].sort_values("agc_exposure_id").index
    bad = rows[2]
    agcData.loc[bad, "agc_match_flags"] = 0
    agcData.loc[bad, "agc_center_x_mm"] += 1.0

    smoothed = smoothAgcData(agcData, 3)

    assert smoothed.agc_center_x_mm[bad] == agcData.agc_center_x_mm[bad]
    good = rows.drop(bad)
    expected = agcData.agc_center_x_mm[good].rolling(3, min_periods=1, center=True).mean()
    np.testing.assert_allclose(smoothed.agc_center_x_mm[good], expected)

    # Negative control: every row smoothed together, as before; the bad match drags its neighbours by 1/3 mm.
    everyRow = agcData.agc_center_x_mm[rows].rolling(3, min_periods=1, center=True).mean()
    neighbours = [rows[1], rows[3]]
    assert (everyRow[neighbours] - smoothed.agc_center_x_mm[neighbours]).abs().min() > 0.25


@pytest.mark.parametrize("reference", ["center0_visit", "nominal0_visit"])
def testEstimateGuideErrorsPerVisit(makeAgcData, reference):
    """The per-visit references work; drp_stella raised "Test me"."""
    agcData = makeAgcData()
    # Move the nominal positions from visit to visit, as the per-visit references then differ.
    agcData["agc_nominal_x_mm"] += 0.01 * (agcData.pfs_visit_id - VISITS[0])
    guideErrors = estimateGuideErrors(agcData, reference)

    source = "center" if reference.startswith("center") else "nominal"
    data = agcData[agcData.shutter_open > 0].copy()
    reference0 = agcData.groupby(["pfs_visit_id", "guide_star_id"])[f"agc_{source}_x_mm"].median()
    keys = pd.MultiIndex.from_frame(data[["pfs_visit_id", "guide_star_id"]])
    data["dx"] = 1e3 * (data.agc_center_x_mm - reference0.loc[keys].to_numpy())
    expected = data.groupby(["agc_exposure_id", "agc_camera_id"]).dx.mean()
    np.testing.assert_allclose(guideErrors.dx_um, expected.to_numpy())
    assert guideErrors.attrs["offset"] == f"center - {reference}"

    # Negative control: the reference over all visits gives different errors.
    allVisits = estimateGuideErrors(agcData, reference.removesuffix("_visit"))
    assert not np.allclose(allVisits.dx_um, guideErrors.dx_um, atol=1)


def testEstimateGuideErrorsClosedShutter(makeAgcData):
    """Closed-shutter exposures are kept on request; drp_stella always dropped them."""
    agcData = makeAgcData()
    assert set(agcData.shutter_open) == {0, 1}

    assert set(estimateGuideErrors(agcData).shutter_open) == {1}
    assert set(estimateGuideErrors(agcData, includeClosedShutter=True).shutter_open) == {0, 1}


def testEstimateGuideErrorsOptions(makeAgcData):
    agcData = makeAgcData()
    agcData.loc[(agcData.guide_star_id == 1001).to_numpy(), "agc_match_flags"] = 0

    guideErrors = estimateGuideErrors(agcData, byVisit=True)
    assert len(guideErrors) == 3 * 6
    assert list(guideErrors.columns[:3]) == ["pfs_visit_id", "agc_exposure_id", "agc_camera_id"]

    guideErrors = estimateGuideErrors(agcData, "nominal0", position="nominal", subtractMedian=True)
    np.testing.assert_allclose(guideErrors.groupby("agc_camera_id").dx_um.median(), 0, atol=1e-9)
    assert guideErrors.attrs["offset"] == "nominal - nominal0"

    guideErrors = estimateGuideErrors(agcData, "boresight", recenterPerVisit=True, smoothing=3)
    np.testing.assert_allclose(guideErrors.groupby("pfs_visit_id").dy_um.mean(), 0, atol=1e-9)

    with pytest.raises(ValueError, match="Unknown position"):
        estimateGuideErrors(agcData, position="centre")


# Drift rates


def makeDriftData(makeAgcData, rng, rates_um_per_min):
    """Return one visit's AG data, whose stars drift at the given (x, y) rates."""
    agcData = makeAgcData(nVisit=1, nExp=20, closedFrac=0).reset_index(drop=True)
    time_min = (agcData.taken_at - agcData.taken_at.min()).dt.total_seconds() / 60
    for xy, rate in zip("xy", rates_um_per_min, strict=True):
        noise = rng.normal(0, 1e-4, len(agcData))
        agcData[f"agc_center_{xy}_mm"] = agcData[f"agc_nominal_{xy}_mm"] + 1e-3 * rate * time_min + noise

    return agcData


@pytest.mark.parametrize("robust", [True, False])
def testFitDriftRateXY(makeAgcData, rng, robust):
    """The x and y rates are each fitted, under their own keys.

    drp_stella stored the y rate under the x key.
    """
    rx, ry = 0.5, -2.0
    agcData = makeDriftData(makeAgcData, rng, (rx, ry))

    fit = fitDriftRate(agcData, radialTangential=False, robust=robust)

    assert fit.rates.x_rate_um_per_min == pytest.approx(rx, abs=0.02)
    assert fit.rates.y_rate_um_per_min == pytest.approx(ry, abs=0.02)
    assert fit.rates.name == fit.rates.pfs_visit_id == VISITS[0]
    assert fit.offsets.time_min.is_monotonic_increasing
    assert len(fit.offsets) == 20 * 6
    # Negative control: the y rate doesn't pass for the x rate.
    assert fit.rates.y_rate_um_per_min != pytest.approx(rx, abs=0.02)


def testFitDriftRateRadialTangential(makeAgcData, rng):
    """A drift along +x is radial on AG1 (at +x) and tangential on AG3 (near +y)."""
    agcData = makeDriftData(makeAgcData, rng, (1.0, 0.0))

    for camera, component in [(0, "radial"), (2, "tangential")]:
        fit = fitDriftRate(agcData[agcData.agc_camera_id == camera])
        other = "tangential" if component == "radial" else "radial"
        assert abs(fit.rates[f"{component}_rate_um_per_min"]) > 0.8
        assert abs(fit.rates[f"{other}_rate_um_per_min"]) < 0.6


def testFitDriftRateInvalidMatches(makeAgcData, rng):
    """Invalid matches don't enter the drift rates."""
    agcData = makeDriftData(makeAgcData, rng, (1.0, 0.0))
    agcData, invalid = invalidate(agcData, shift_mm=0.0)
    time_min = (agcData.taken_at - agcData.taken_at.min()).dt.total_seconds() / 60
    agcData.loc[invalid, "agc_center_x_mm"] += 1e-3 * 50 * time_min[invalid]  # 50 microns/min more

    rates = fitDriftRate(agcData, radialTangential=False).rates
    assert rates.x_rate_um_per_min == pytest.approx(1.0, abs=0.05)
    # Negative control: counted as valid, they pull the rate up.
    wrong = fitDriftRate(agcData.assign(agc_match_flags=1), radialTangential=False).rates
    assert wrong.x_rate_um_per_min > 2


def testFitDriftRateIndependentCalls(makeAgcData, rng):
    """Each call returns its own rates; drp_stella's shared rates dict kept the first visit's."""
    first = makeDriftData(makeAgcData, rng, (1.0, 1.0))
    second = makeDriftData(makeAgcData, rng, (-1.0, -1.0))

    a = fitDriftRate(first, radialTangential=False).rates
    b = fitDriftRate(second, radialTangential=False).rates

    assert a.x_rate_um_per_min == pytest.approx(1.0, abs=0.02)
    assert b.x_rate_um_per_min == pytest.approx(-1.0, abs=0.02)


def testFitDriftRateOptions(makeAgcData, rng):
    agcData = makeDriftData(makeAgcData, rng, (1.0, 0.0))
    exposures = np.sort(agcData.agc_exposure_id.unique())[:10]

    fit = fitDriftRate(
        agcData, "nominal0", agcExposureIds=exposures, byCamera=False, subtractMeanOffset=False
    )
    assert set(fit.offsets.agc_exposure_id) == set(exposures)
    assert "agc_camera_id" not in fit.offsets
    assert fit.offsets.attrs["offset"] == "center - nominal0"

    fit = fitDriftRate(agcData, "nominal0", position="nominal")
    np.testing.assert_allclose(fit.offsets.dx_um, 0, atol=1e-9)  # the nominal positions don't move

    with pytest.raises(ValueError, match="No AG exposures with the shutters open"):
        fitDriftRate(agcData.assign(shutter_open=0))


# Comparison with pfs_utils
#
# These replace pfs_utils's model with a lookup. The model itself (its frame,
# time and inputs) is checked against the guider's positions in test_realData.


@pytest.fixture
def fakeAgActorPositions(monkeypatch):
    """Replace agActorPositions with a lookup of each star's position (hardware coordinates).

    Each call's AG exposures and ``instPaCorrection_arcsec`` are recorded in
    ``calls``.
    """

    class Fake:
        def __init__(self):
            self.positions = {}  # guide_star_id: (x_mm, y_mm)
            self.calls = []

        def __call__(self, agcData, instPaCorrection_arcsec=0.0):
            self.calls.append(
                SimpleNamespace(
                    exposures=sorted(agcData.agc_exposure_id.unique()),
                    instPaCorrection_arcsec=instPaCorrection_arcsec,
                )
            )
            xy = np.array([self.positions[gid] for gid in agcData.guide_star_id])
            return xy[:, 0], xy[:, 1]

    fake = Fake()
    monkeypatch.setattr(analysis, "agActorPositions", fake)

    return fake


def makeComparisonData(makeAgcData, fake, rotation_rad=0.0):
    """Return guide stars and AG data, and teach ``fake`` pfs_utils's positions of the stars.

    pfs_utils's positions are each star's nominal position in its first
    exposure, rotated by ``rotation_rad`` about the boresight.
    """
    agcData = makeAgcData(nVisit=1, nExp=12).reset_index(drop=True)
    agcData["agc_nominal_x_mm"] += 1e-3 * (agcData.agc_exposure_id % 7)  # vary between exposures
    agcData["guide_ra"], agcData["guide_dec"], agcData["guide_pa"] = 150.0, 2.0, -90.0
    agcData["adc_pa"] = 0.5
    stars = agcData.drop_duplicates("guide_star_id")[["guide_star_id", "agc_camera_id"]].copy()
    stars["pfs_design_id"] = 1
    stars["pfs_visit_id"] = VISITS[0]
    stars["guide_star_ra"] = stars.guide_star_id.astype(float)
    for column in ("guide_star_dec", "guide_star_pm_ra", "guide_star_pm_dec", "guide_star_parallax"):
        stars[column] = 0.0

    first = agcData[agcData.agc_exposure_id == agcData.agc_exposure_id.min()].set_index("guide_star_id")
    x, y = rotXY(-rotation_rad, first.agc_nominal_x_mm, first.agc_nominal_y_mm)
    fake.positions = {gid: (x[gid], y[gid]) for gid in first.index}

    return stars, agcData


def testComparePfsUtilsAverages(makeAgcData, fakeAgActorPositions):
    """The guider's positions are the median over the first nAgcExposures.

    drp_stella grouped by exposure and star, so each median was of one row,
    and returned each star nAgcExposures times.
    """
    stars, agcData = makeComparisonData(makeAgcData, fakeAgActorPositions)
    nAgcExposures = 5

    result = comparePfsUtilsPositions(stars, agcData, nAgcExposures=nAgcExposures)

    exposures = np.sort(agcData.agc_exposure_id.unique())[:nAgcExposures]
    first = agcData[agcData.agc_exposure_id.isin(exposures)]
    expected = first.groupby("guide_star_id").agc_nominal_x_mm.median()
    assert len(result.stars) == len(stars)
    np.testing.assert_allclose(result.stars.agc_nominal_x_mm, expected.loc[result.stars.guide_star_id])
    (call,) = fakeAgActorPositions.calls
    assert call.exposures == list(exposures)
    assert result.alignOffset_um is None

    # Negative control: drp_stella's medians, of one row each, vary between exposures.
    perExposure = first.groupby(["agc_exposure_id", "guide_star_id"]).agc_nominal_x_mm.median()
    assert len(perExposure) == nAgcExposures * len(stars)
    assert not np.allclose(perExposure.groupby("guide_star_id").first(), expected, rtol=0, atol=1e-6)


def testComparePfsUtilsInputs(makeAgcData, fakeAgActorPositions):
    """AG exposures without the AG actor's inputs (no agc_guide_offset row) are skipped."""
    stars, agcData = makeComparisonData(makeAgcData, fakeAgActorPositions)
    exposures = np.sort(agcData.agc_exposure_id.unique())
    agcData.loc[agcData.agc_exposure_id == exposures[0], "guide_ra"] = np.nan

    comparePfsUtilsPositions(stars, agcData, nAgcExposures=3)

    (call,) = fakeAgActorPositions.calls
    assert call.exposures == list(exposures[1:4])

    with pytest.raises(ValueError, match="No guide star is in both"):
        comparePfsUtilsPositions(stars, agcData.assign(guide_pa=np.nan))


def testComparePfsUtilsRotation(makeAgcData, fakeAgActorPositions):
    """delta_theta is the guider's angle minus pfs_utils's, wrapped across -x."""
    stars, agcData = makeComparisonData(makeAgcData, fakeAgActorPositions, rotation_rad=np.deg2rad(2 / 3600))
    # Put a star of AG4 just below -x, so that pfs_utils puts it just above.
    ag4 = agcData.guide_star_id == stars.guide_star_id[stars.agc_camera_id == 3].iloc[0]
    agcData.loc[ag4, "agc_nominal_y_mm"] = -1e-3
    star = agcData[ag4].sort_values("agc_exposure_id").iloc[0]
    x, y = rotXY(np.deg2rad(-2 / 3600), star.agc_nominal_x_mm, star.agc_nominal_y_mm)
    fakeAgActorPositions.positions[star.guide_star_id] = (x, y)

    result = comparePfsUtilsPositions(stars, agcData, nAgcExposures=1, instPaCorrection_arcsec=36).stars

    np.testing.assert_allclose(result.delta_theta_arcsec, 2, atol=1e-3)
    assert fakeAgActorPositions.calls[-1].instPaCorrection_arcsec == 36

    # Negative control: the unwrapped difference is a full turn out for the AG4 star.
    unwrapped = np.rad2deg(result.agc_nominal_theta_rad - result.pfs_utils_theta_rad) * 3600
    assert np.max(np.abs(unwrapped)) > 1e6


def testComparePfsUtilsAlign(makeAgcData, fakeAgActorPositions):
    stars, agcData = makeComparisonData(makeAgcData, fakeAgActorPositions)
    fakeAgActorPositions.positions = {
        gid: (x + 0.01, y) for gid, (x, y) in fakeAgActorPositions.positions.items()
    }

    result = comparePfsUtilsPositions(stars, agcData, nAgcExposures=1, alignCenterPosition=True)

    assert result.alignOffset_um == pytest.approx((10, 0), abs=1e-6)
    np.testing.assert_allclose(result.stars.pfs_utils_x_mm, result.stars.agc_nominal_x_mm)

    with pytest.raises(ValueError, match="No guide star is in both"):
        comparePfsUtilsPositions(stars.assign(guide_star_id=-1), agcData)


# Image sizes and focus


def testAddImageSizes():
    stars = pd.DataFrame(
        {
            "mxx": [4.0, 2.0, -1.0, -2.0],
            "myy": [4.0, 8.0, 1.0, -3.0],
            "mxy": [0.0, 1.0, 0.0, 0.0],
            "agc_data_flags": [0, int(SourceDetectionFlags.RIGHT), 0, 0],
        }
    )
    trace = addImageSizes(stars)
    np.testing.assert_allclose(trace.rms_pix, [2.0, np.sqrt(5.0), np.nan, np.nan])
    fwhmPerPix = 2 * np.sqrt(2 * np.log(2)) * AGC_PIXEL_SIZE_UM / AGC_PLATE_SCALE_UM_PER_ARCSEC
    np.testing.assert_allclose(trace.fwhm_arcsec, fwhmPerPix * trace.rms_pix)
    np.testing.assert_allclose(trace.fwhm_arcsec[0], 0.6466, atol=1e-4)  # 2 pixels rms, in arcsec
    assert list(trace.left) == [True, False, True, True]

    det = addImageSizes(stars, useTraceRadius=False)
    np.testing.assert_allclose(det.rms_pix, [2.0, 15.0**0.25, np.nan, np.nan])
    # Negative control: the last row's determinant is positive, though its moments are impossible.
    assert stars.mxx.iloc[-1] * stars.myy.iloc[-1] - stars.mxy.iloc[-1] ** 2 > 0
    assert "rms_pix" not in stars  # drp_stella's setImageSizes added it to its input


def testMomentDifferenceToPiston():
    """ics_agActor's calibration: 0.0086 mm per pix^2 of 2-D moment difference, less 0.026 mm."""
    assert momentDifferenceToPiston(1.0) == pytest.approx(2 * 0.0086)
    assert momentDifferenceToPiston(1.0, includePistonCorrection=True) == pytest.approx(2 * 0.0086 - 0.026)
    np.testing.assert_allclose(momentDifferenceToPiston([0.0, -1.0]), [0.0, -2 * FOCUS_PISTON_MM_PER_PIX2])


def testCorrectAgActorFocus(makeAgcData):
    """Rows before visit 122129 are corrected for INSTRM-2501, whatever the other visits.

    drp_stella corrected the rows only if every visit was before 122129.
    """
    agcData = makeAgcData(stars=True)
    agcData["pfs_visit_id"] = agcData.pfs_visit_id.map(
        dict(zip(VISITS, [122000, 122128, 122129], strict=True))
    )

    corrected = correctAgActorFocus(agcData)

    offset = -0.026
    early = agcData.pfs_visit_id < 122129
    for column in ("guide_delta_z", "guide_delta_z3"):
        expected = np.where(early, offset + (agcData[column] - offset) / 4, agcData[column])
        np.testing.assert_allclose(corrected[column], expected)

    # Negative control: drp_stella's test, on the latest visit, corrects nothing here.
    assert agcData.pfs_visit_id.max() >= 122129
    assert not np.allclose(agcData.guide_delta_z[early], corrected.guide_delta_z[early])


def makeFocusData(rms_pix):
    """Return stars whose rms sizes are given by (exposure, camera, left)."""
    rows = []
    for (aid, cid, left), size in rms_pix.items():
        for _ in range(3):
            rows.append(
                {
                    "pfs_visit_id": 120000,
                    "agc_exposure_id": aid,
                    "agc_camera_id": cid,
                    "left": left,
                    "rms_pix": size,
                    "altitude": 60.0,
                    "insrot": -30.0,
                    "m2_off3": 0.1 * aid,
                    "m2_pos3": 0.0,
                    "agc_match_flags": 1,
                }
            )

    return pd.DataFrame(rows).sample(frac=1, random_state=1)


def testEstimateFocusErrors():
    sizes = {
        (1, 0, True): 2.1,
        (1, 0, False): 2.0,
        (1, 1, True): 2.0,
        (1, 1, False): 2.2,
        (1, 2, True): 2.4,
        (1, 2, False): 2.0,
        (2, 0, True): 2.0,  # one half only
    }
    agcData = makeFocusData(sizes)

    byCamera = estimateFocusErrors(agcData)

    assert list(zip(byCamera.agc_exposure_id, byCamera.agc_camera_id, strict=True)) == [
        (1, 0),
        (1, 1),
        (1, 2),
    ]
    expected = [
        1e3 * 2 * 0.0086 * (left**2 - right**2) for left, right in [(2.1, 2.0), (2.0, 2.2), (2.4, 2.0)]
    ]
    np.testing.assert_allclose(byCamera.focus_error_um, expected)
    np.testing.assert_allclose(byCamera.focus_position_mm, 0.1)

    byExposure = estimateFocusErrors(agcData, byCamera=False)
    assert list(byExposure.agc_exposure_id) == [1]
    np.testing.assert_allclose(byExposure.rms_left_pix, 2.1)  # median of the cameras' medians
    np.testing.assert_allclose(byExposure.rms_right_pix, 2.0)
    np.testing.assert_allclose(byExposure.focus_error_um, expected[0])

    np.testing.assert_allclose(estimateFocusErrors(agcData, focusColumn="m2_pos3").focus_position_mm, 0)

    # Invalid matches don't count; counted as valid, these would change the first camera's error.
    bad = makeFocusData({(1, 0, True): 9.0}).assign(agc_match_flags=0)
    withBad = estimateFocusErrors(pd.concat([agcData, bad]))
    pd.testing.assert_frame_equal(withBad, byCamera)
    wrong = estimateFocusErrors(pd.concat([agcData, bad.assign(agc_match_flags=1)]))
    assert wrong.focus_error_um.iloc[0] != pytest.approx(byCamera.focus_error_um.iloc[0])


def testAverageByFocusPosition():
    """Points are grouped by their rounded focus positions.

    drp_stella compared unrounded positions with rounded ones, and rounded
    negative positions towards zero.
    """
    focusPosition = pd.Series([0.1000001, 0.0999999, 0.1, -0.2000001, -0.1999999, np.nan])
    values = np.arange(6.0)

    x, y = averageByFocusPosition(focusPosition, focusPosition, values)
    np.testing.assert_allclose(x, [-0.2, 0.1], atol=1e-6)
    np.testing.assert_array_equal(y, [3.5, 1.0])

    _, y = averageByFocusPosition(focusPosition, focusPosition, values, useMedian=False)
    np.testing.assert_array_equal(y, [3.5, 1.0])

    # Negative control: drp_stella's version.
    rounded = (1e3 * np.sort(list(set(focusPosition.dropna()))) + 0.5).astype(int) / 1e3
    assert -0.199 in rounded  # -0.2 rounded towards zero
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        medians = [np.median(values[focusPosition == fp]) for fp in rounded]
    assert np.isnan(medians).any()
