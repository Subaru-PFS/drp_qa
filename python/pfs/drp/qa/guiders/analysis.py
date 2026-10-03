"""Fits and statistics on AG data.

Functions take DataFrames from `pfs.drp.qa.guiders.queries`, never a database
connection, and don't modify their inputs; those that return AG data return a
copy, with rows in the same order and the same index. Positions are in
hardware coordinates, and offsets follow `pfs.drp.qa.guiders.coordinates`:
``dx_<reference>_um`` is a center minus a reference, in microns.

The fits that predict where each star should be, `fitGuiderModel` and
`fitGlobalModel`, put the prediction in the ``model`` reference,
``agc_model_[xy]_mm``, and the residuals in ``dx_model_um`` and
``dy_model_um``. Their parameters are in the result, not in shared state.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from numpy.typing import ArrayLike

from pfs.drp.qa.guiders.coordinates import (
    addOffsets,
    addReferencePositions,
    mmToUm,
    offsetColumns,
    pfiToZenith,
    pixToArcsec,
    radToArcsec,
    referenceColumns,
    umToMm,
    zenithToPfi,
)
from pfs.utils.coordinates import CoordTransp
from pfs.utils.coordinates.transform import MeasureDistortion
from pfs.utils.datamodel.ag import SourceCatalogFlags, SourceDetectionFlags, SourceMatchingFlags

__all__ = [
    "AGACTOR_FOCUS_FIX_VISIT",
    "FOCUS_PISTON_MM_PER_PIX2",
    "FOCUS_PISTON_OFFSET_MM",
    "GAUSSIAN_FWHM_PER_SIGMA",
    "DriftFit",
    "GlobalModelFit",
    "GuiderFit",
    "GuiderFitConfig",
    "MeasureXYRot",
    "PfsUtilsComparison",
    "addImageSizes",
    "averageByFocusPosition",
    "comparePfsUtilsPositions",
    "correctAgActorFocus",
    "estimateFocusErrors",
    "estimateGuideErrors",
    "fitDriftRate",
    "fitGlobalModel",
    "fitGuiderModel",
    "fitTransform",
    "guideErrorsByExposure",
    "momentDifferenceToPiston",
    "selectGoodDetections",
    "selectIsolatedGaiaStars",
    "selectStars",
    "selectValidMatches",
    "smoothAgcData",
]

GAUSSIAN_FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))

# ics_agActor's focus calibration (momentdifference2focuserror): the focus
# error is FOCUS_PISTON_MM_PER_PIX2 times the difference of the 2-D second
# moments of the two halves of a detector, less FOCUS_PISTON_OFFSET_MM.
FOCUS_PISTON_MM_PER_PIX2 = 0.0086
FOCUS_PISTON_OFFSET_MM = 0.026

# ics_agActor's guide_delta_z* were 4 times too large, less the offset, before
# this visit (INSTRM-2501, fixed on 2025-03-23).
AGACTOR_FOCUS_FIX_VISIT = 122129

_AGACTOR_FOCUS_COLUMNS = ("guide_delta_z", *(f"guide_delta_z{i}" for i in range(1, 7)))

# Detection flags other than RIGHT (the half of the detector) mark a bad measurement.
_BAD_DETECTION_FLAGS = ~int(SourceDetectionFlags.RIGHT)

# Columns that smoothAgcData leaves alone, as well as any ending in _flag or _flags.
_UNSMOOTHED = ("pfs_visit_id", "agc_exposure_id", "agc_camera_id", "spot_id", "guide_star_id", "shutter_open")

# The parameters of MeasureDistortion.distort, in order.
_TRANSFORM_PARAMETERS = ("x0_mm", "y0_mm", "theta_deg", "dscale", "scale2_per_mm2")

# HST is UTC-10 all year.
_HST_TO_UTC = pd.Timedelta(hours=10)


# Selection


def selectGoodDetections(agcData: pd.DataFrame) -> np.ndarray:
    """Select the stars whose detections have no flags but RIGHT.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data with ``agc_data_flags``.

    Returns
    -------
    good : `numpy.ndarray` of `bool`
        One element per row.
    """
    return (agcData.agc_data_flags.to_numpy() & _BAD_DETECTION_FLAGS) == 0


def selectValidMatches(agcData: pd.DataFrame) -> np.ndarray:
    """Select the spots validly matched to their guide stars.

    `pfs.drp.qa.guiders.queries.readAgcData` returns every match, whatever
    its flags; the fits and averages here use only the valid ones: those
    whose ``agc_match_flags`` (`SourceMatchingFlags`) are ``GOOD_MATCH`` and
    nothing else. ics_agActor writes 1 (``GOOD_MATCH``) or 0 so far.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data with ``agc_match_flags``.

    Returns
    -------
    valid : `numpy.ndarray` of `bool`
        One element per row.
    """
    return agcData.agc_match_flags.to_numpy() == int(SourceMatchingFlags.GOOD_MATCH)


def selectIsolatedGaiaStars(agcData: pd.DataFrame) -> np.ndarray:
    """Select the guide stars from GAIA that aren't binaries or galaxies.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data with ``guide_star_flag`` (`SourceCatalogFlags`).

    Returns
    -------
    isolated : `numpy.ndarray` of `bool`
        One element per row.
    """
    flag = agcData.guide_star_flag.to_numpy()

    return (
        ((flag & int(SourceCatalogFlags.GAIA)) != 0)
        & ((flag & int(SourceCatalogFlags.NON_BINARY)) != 0)
        & ((flag & int(SourceCatalogFlags.GALAXY)) == 0)
    )


@dataclass(frozen=True)
class GuiderFitConfig:
    """Which models `fitGuiderModel` fits, and which stars it uses.

    drp_stella's ``GuiderConfig`` mixed these options with plotting ones. Its
    ``maxGuideError``, ``maxPosError``, ``agc_exposure_idsStride``,
    ``pfs_visitIdMin`` (and so on) are renamed here with their units or in
    camelCase.

    Attributes
    ----------
    modelBoresightOffset : `bool`
        Fit and remove an offset, rotation and scale for each AG exposure.
    modelCCDOffset : `bool`
        Fit and remove an offset, rotation and scale for each AG camera.
    solveForAGTransforms : `bool`
        Refit the transforms of the ``previous`` fit given to
        `fitGuiderModel`, rather than reusing them.
    onlyShutterOpen : `bool`
        Only use AG exposures taken while the spectrograph shutters weren't
        closed (``shutter_open`` isn't 0).
    maxGuideError_um : `float`
        Only use AG exposures whose guide error is less than this (microns);
        see `guideErrorsByExposure`. No cut if <= 0.
    maxPosError_um : `float`
        Only average the guide stars whose mean distance from where the
        models put them is less than this (microns). No cut if <= 0.
    agcExposureStride : `int`
        Only use every ``agcExposureStride``'th AG exposure, starting with the
        first that passes the shutter and guide error cuts.
    pfsVisitIdMin, pfsVisitIdMax : `int`
        Only use these visits; no limit if <= 0.
    agcExposureIdMin, agcExposureIdMax : `int`
        Only use these AG exposures; no limit if <= 0.

    Raises
    ------
    ValueError
        If ``agcExposureStride`` is less than 1.
    """

    modelBoresightOffset: bool = True
    modelCCDOffset: bool = True
    solveForAGTransforms: bool = False
    onlyShutterOpen: bool = True
    maxGuideError_um: float = 25
    maxPosError_um: float = 40
    agcExposureStride: int = 1
    pfsVisitIdMin: int = 0
    pfsVisitIdMax: int = 0
    agcExposureIdMin: int = 0
    agcExposureIdMax: int = 0

    def __post_init__(self):
        if self.agcExposureStride < 1:
            raise ValueError(f"agcExposureStride must be at least 1, not {self.agcExposureStride}")


def guideErrorsByExposure(agcData: pd.DataFrame, reference: str = "nominal") -> pd.Series:
    """Return the guide error of each AG exposure.

    The guide error is the mean distance of the exposure's valid matches
    (`selectValidMatches`) from their reference positions.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data in hardware coordinates.
    reference : `str`
        One of `pfs.drp.qa.guiders.coordinates.REFERENCES`; its positions are
        computed from the valid matches, with the default statistic, if they
        aren't given.

    Returns
    -------
    guideErrors : `pandas.Series`
        Guide error (microns), indexed by ``agc_exposure_id``. An exposure
        with no valid match has none.
    """
    dx, dy = offsetColumns(reference)
    data = addOffsets(agcData[selectValidMatches(agcData)], reference).reset_index(drop=True)
    dr = pd.Series(np.hypot(data[dx], data[dy]), name="guide_error_um")

    return dr.groupby(data.agc_exposure_id).mean()


def selectStars(
    agcData: pd.DataFrame,
    config: GuiderFitConfig | None = None,
    agc_camera_id: int | None = None,
    guideErrors: pd.Series | None = None,
) -> np.ndarray:
    """Select the stars to use, as ``config`` says.

    Only valid matches (`selectValidMatches`) are selected.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data.
    config : `GuiderFitConfig`, optional
        Which stars to use; default ``GuiderFitConfig()``.
    agc_camera_id : `int`, optional
        Only select stars on this camera.
    guideErrors : `pandas.Series`, optional
        The guide error (microns) of each AG exposure, indexed by
        ``agc_exposure_id``, as `guideErrorsByExposure` returns. Without it,
        there is no guide error cut.

    Returns
    -------
    selected : `numpy.ndarray` of `bool`
        One element per row of ``agcData``.

    Notes
    -----
    Guide errors are matched to rows by ``agc_exposure_id``, so the rows may
    be in any order. The stride starts at the first AG exposure (whichever the
    camera) that passes the shutter and guide error cuts, and counts every AG
    exposure after it.
    """
    config = GuiderFitConfig() if config is None else config

    exposureId = agcData.agc_exposure_id.to_numpy()
    passes = np.ones(len(agcData), dtype=bool)
    if config.onlyShutterOpen:
        passes &= agcData.shutter_open.to_numpy() > 0
    if guideErrors is not None and config.maxGuideError_um > 0:
        guideError = agcData.agc_exposure_id.map(guideErrors).to_numpy(dtype=float)
        passes &= guideError < config.maxGuideError_um

    exposures = np.unique(exposureId)
    exposurePasses = pd.Series(passes).groupby(exposureId).all().to_numpy()
    start = int(np.argmax(exposurePasses)) if exposurePasses.any() else 0

    selected = passes & selectValidMatches(agcData)
    selected &= np.isin(exposureId, exposures[start :: config.agcExposureStride])
    if agc_camera_id is not None:
        selected &= agcData.agc_camera_id.to_numpy() == agc_camera_id
    for column, limit, compare in [
        ("pfs_visit_id", config.pfsVisitIdMin, np.greater_equal),
        ("pfs_visit_id", config.pfsVisitIdMax, np.less_equal),
        ("agc_exposure_id", config.agcExposureIdMin, np.greater_equal),
        ("agc_exposure_id", config.agcExposureIdMax, np.less_equal),
    ]:
        if limit > 0:
            selected &= compare(agcData[column].to_numpy(), limit)

    return selected


# Boresight and per-camera transforms


class MeasureXYRot(MeasureDistortion):
    """An offset, rotation and scale taking measured positions to true ones.

    The model is pfs_utils's `MeasureDistortion.distort`, whose parameters
    (``x0``, ``y0`` in mm, ``theta`` in degrees, ``dscale``, and ``scale2``
    per mm^2) all start at 0. Call `fit` to fit them.

    Parameters
    ----------
    x_mm, y_mm : array-like
        Measured positions (mm).
    dx_um, dy_um : array-like
        True minus measured positions (microns).
    nsigma : `float`, optional
        Clip the fit at this many standard deviations (default 5); no
        clipping if <= 0.
    alphaRot : `float`
        Weight of a prior on the rotation, ``alphaRot*theta**2``.
    """

    def __init__(self, x_mm, y_mm, dx_um, dy_um, nsigma: float | None = 5, alphaRot: float = 0.0):
        # MeasureDistortion.__init__ matches fiducials by ID, so isn't called.
        self.nsigma = 5 if nsigma is None else nsigma
        self.alphaRot = alphaRot

        x_mm, y_mm, dx_um, dy_um = (np.asarray(a, dtype=float) for a in (x_mm, y_mm, dx_um, dy_um))
        good = np.isfinite(x_mm + y_mm + dx_um + dy_um)

        self.x = x_mm[good]
        self.y = y_mm[good]
        # The true positions, to which MeasureDistortion.__call__ fits distort(x, y).
        self.x_mm = self.x + umToMm(dx_um[good])
        self.y_mm = self.y + umToMm(dy_um[good])

        self._args = np.zeros(len(_TRANSFORM_PARAMETERS))
        self.frozen = np.zeros(len(self._args), dtype=bool)

    def fit(self) -> "MeasureXYRot":
        """Fit the parameters, and return self."""
        result = scipy.optimize.minimize(self, self.getArgs().copy(), method="Powell")
        self.setArgs(result.x)

        return self


def fitTransform(x_mm, y_mm, xTrue_mm, yTrue_mm, nsigma: float | None = None) -> MeasureXYRot:
    """Fit the transform taking measured positions to true ones.

    Parameters
    ----------
    x_mm, y_mm : array-like
        Measured positions (mm).
    xTrue_mm, yTrue_mm : array-like
        True positions (mm).
    nsigma : `float`, optional
        Clip the fit at this many standard deviations; see `MeasureXYRot`.

    Returns
    -------
    transform : `MeasureXYRot`
        The fitted transform; ``transform.distort(x, y)`` maps measured
        positions to true ones, and ``inverse=True`` true to measured.
    """
    x_mm, y_mm, xTrue_mm, yTrue_mm = (np.asarray(a, dtype=float) for a in (x_mm, y_mm, xTrue_mm, yTrue_mm))

    return MeasureXYRot(x_mm, y_mm, mmToUm(xTrue_mm - x_mm), mmToUm(yTrue_mm - y_mm), nsigma=nsigma).fit()


def _transformParameters(transforms: Mapping[int, MeasureXYRot], name: str) -> pd.DataFrame:
    """Return the parameters of some transforms, one row each."""
    parameters = pd.DataFrame(
        [transform.getArgs() for transform in transforms.values()],
        index=pd.Index(list(transforms), name=name),
        columns=list(_TRANSFORM_PARAMETERS),
        dtype=float,
    )

    return parameters.sort_index()


@dataclass(frozen=True, eq=False)
class GuiderFit:
    """The result of `fitGuiderModel`.

    Attributes
    ----------
    agcData : `pandas.DataFrame`
        A copy of the AG data with ``agc_model_[xy]_mm``, where the models put
        each star (its nominal position, moved by its exposure's and its
        camera's transforms); ``dx_model_um``, ``dy_model_um`` and
        ``dr_model_um``, the star's offset from there; ``guide_error_um``, its
        exposure's guide error with only the boresight model; and
        ``selected``, `selectStars` with the fit's config and guide errors.
    guideErrorByCamera : `pandas.DataFrame`
        For each AG exposure and camera, the mean ``agc_model_[xy]_mm``,
        ``dx_model_um`` and ``dy_model_um`` of the selected stars that pass
        the ``maxPosError_um`` cut, sorted by exposure.
    exposureTransforms : `~collections.abc.Mapping` [`int`, `MeasureXYRot`]
        The boresight transform of each AG exposure, by ``agc_exposure_id``.
    cameraTransforms : `~collections.abc.Mapping` [`int`, `MeasureXYRot`]
        The transform of each AG camera, by ``agc_camera_id``.
    config : `GuiderFitConfig`
        The configuration of the fit.
    """

    agcData: pd.DataFrame
    guideErrorByCamera: pd.DataFrame
    exposureTransforms: Mapping[int, MeasureXYRot]
    cameraTransforms: Mapping[int, MeasureXYRot]
    config: GuiderFitConfig

    @property
    def exposureParameters(self) -> pd.DataFrame:
        """The parameters of `exposureTransforms`, indexed by ``agc_exposure_id``."""
        return _transformParameters(self.exposureTransforms, "agc_exposure_id")

    @property
    def cameraParameters(self) -> pd.DataFrame:
        """The parameters of `cameraTransforms`, indexed by ``agc_camera_id``."""
        return _transformParameters(self.cameraTransforms, "agc_camera_id")


def fitGuiderModel(
    agcData: pd.DataFrame, config: GuiderFitConfig | None = None, previous: GuiderFit | None = None
) -> GuiderFit:
    """Fit where the guide stars should be, per AG exposure and per camera.

    These are the fits of drp_stella's ``showGuiderErrors``. With
    ``modelBoresightOffset``, a transform is fitted to each AG exposure's
    valid matches without bad detection flags, taking their centers to their
    nominal positions. With ``modelCCDOffset``, a transform is then fitted to
    each camera's selected stars (`selectStars`), taking their centers to the
    boresight model. Each star's model position is its nominal position moved
    by the inverse of its exposure's and its camera's transforms. Invalid
    matches are kept in the result, with model positions, but are neither
    fitted nor averaged.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data in hardware coordinates, from
        `pfs.drp.qa.guiders.queries.readAgcData`.
    config : `GuiderFitConfig`, optional
        What to fit, and which stars to use; default ``GuiderFitConfig()``.
    previous : `GuiderFit`, optional
        A previous fit whose transforms are reused, unless
        ``config.solveForAGTransforms``.

    Returns
    -------
    fit : `GuiderFit`
        The model positions and transforms.

    Notes
    -----
    Unlike drp_stella, the camera transforms are fitted in hardware
    coordinates even when the plot is rotated, and move all of the camera's
    stars, not only the selected ones.
    """
    config = GuiderFitConfig() if config is None else config
    previousExposures = {} if previous is None else previous.exposureTransforms
    previousCameras = {} if previous is None else previous.cameraTransforms

    data = agcData.reset_index(drop=True)
    exposureId = data.agc_exposure_id.to_numpy()
    cameraId = data.agc_camera_id.to_numpy()
    xCenter = data.agc_center_x_mm.to_numpy(dtype=float)
    yCenter = data.agc_center_y_mm.to_numpy(dtype=float)
    xModel = data.agc_nominal_x_mm.to_numpy(dtype=float, copy=True)
    yModel = data.agc_nominal_y_mm.to_numpy(dtype=float, copy=True)
    good = selectGoodDetections(data) & selectValidMatches(data)

    exposureTransforms = {}
    if config.modelBoresightOffset:
        for aid in np.unique(exposureId):
            rows = exposureId == aid
            if aid in previousExposures and not config.solveForAGTransforms:
                transform = previousExposures[aid]
            else:
                fit = rows & good
                transform = fitTransform(xCenter[fit], yCenter[fit], xModel[fit], yModel[fit], nsigma=3)
            exposureTransforms[aid] = transform
            xModel[rows], yModel[rows] = transform.distort(xModel[rows], yModel[rows], inverse=True)

    # Copies, as xModel and yModel are about to change.
    guideErrors = guideErrorsByExposure(
        data.assign(agc_model_x_mm=xModel.copy(), agc_model_y_mm=yModel.copy()), "model"
    )

    cameraTransforms = {}
    if config.modelCCDOffset:
        for cid in np.unique(cameraId):
            fit = selectStars(data, config, cid, guideErrors)
            if not fit.any():
                continue
            if cid in previousCameras and not config.solveForAGTransforms:
                transform = previousCameras[cid]
            else:
                transform = fitTransform(xCenter[fit], yCenter[fit], xModel[fit], yModel[fit])
            cameraTransforms[cid] = transform
            rows = cameraId == cid
            xModel[rows], yModel[rows] = transform.distort(xModel[rows], yModel[rows], inverse=True)

    data["agc_model_x_mm"] = xModel
    data["agc_model_y_mm"] = yModel
    data = addOffsets(data, "model")
    data["dr_model_um"] = np.hypot(data.dx_model_um, data.dy_model_um)
    data["guide_error_um"] = data.agc_exposure_id.map(guideErrors)
    data["selected"] = selectStars(data, config, guideErrors=guideErrors)

    stars = data[data.selected]
    if config.maxPosError_um > 0:
        meanDr = stars.groupby("guide_star_id").dr_model_um.transform("mean")
        stars = stars[meanDr < config.maxPosError_um]
    guideErrorByCamera = stars.groupby(["agc_exposure_id", "agc_camera_id"], as_index=False).agg(
        agc_model_x_mm=("agc_model_x_mm", "mean"),
        agc_model_y_mm=("agc_model_y_mm", "mean"),
        dx_model_um=("dx_model_um", "mean"),
        dy_model_um=("dy_model_um", "mean"),
    )

    data.index = agcData.index

    return GuiderFit(
        agcData=data,
        guideErrorByCamera=guideErrorByCamera,
        exposureTransforms=MappingProxyType(exposureTransforms),
        cameraTransforms=MappingProxyType(cameraTransforms),
        config=config,
    )


# Global model in the zenith frame


@dataclass(frozen=True, eq=False)
class GlobalModelFit:
    """The result of `fitGlobalModel`.

    Attributes
    ----------
    agcData : `pandas.DataFrame`
        A copy of the AG data with ``agc_model_[xy]_mm``, each star's
        nominal position moved by the model, and ``dx_model_um`` and
        ``dy_model_um``, its offset from there.
    exposures : `pandas.DataFrame`
        The terms of each AG exposure, indexed by ``agc_exposure_id``:
        ``dz_um`` and ``dp_um``, the offset towards and perpendicular to the
        zenith; ``rotation_arcsec``, positive turning +dz towards +dp; and
        ``scale``. Terms that weren't fitted are 0.
    cameras : `pandas.DataFrame` or `None`
        The terms of each AG camera, indexed by ``agc_camera_id``:
        ``dz_um``, ``dp_um`` and ``rotation_arcsec`` (about the camera's mean
        nominal position). `None` unless ``fitAgcOffsets``.
    """

    agcData: pd.DataFrame
    exposures: pd.DataFrame
    cameras: pd.DataFrame | None


def _fitExposureTerms(errDz, errDp, dzNominal, dpNominal, exposureId, use, fitPfiRotation, fitPfiScale):
    """Fit an offset, and maybe a rotation and scale, to each exposure's errors (mm)."""
    modelDz, modelDp = np.zeros(len(errDz)), np.zeros(len(errDz))
    records = []
    for aid in np.unique(exposureId):
        rows = exposureId == aid
        fit = rows & use
        nFit = np.sum(fit)
        record = {"agc_exposure_id": aid, "dz_mm": 0.0, "dp_mm": 0.0, "rotation_rad": 0.0, "scale": 0.0}
        if nFit >= 2:
            dz, dp = dzNominal[fit], dpNominal[fit]
            columns = [np.r_[np.ones(nFit), np.zeros(nFit)], np.r_[np.zeros(nFit), np.ones(nFit)]]
            names = ["dz_mm", "dp_mm"]
            if fitPfiRotation:
                columns.append(np.r_[-dp, dz])
                names.append("rotation_rad")
            if fitPfiScale:
                columns.append(np.r_[dz, dp])
                names.append("scale")
            solution = np.linalg.lstsq(np.column_stack(columns), np.r_[errDz[fit], errDp[fit]], rcond=None)[0]
            record.update(zip(names, solution, strict=True))

            dz, dp = dzNominal[rows], dpNominal[rows]
            modelDz[rows] = record["dz_mm"] - record["rotation_rad"] * dp + record["scale"] * dz
            modelDp[rows] = record["dp_mm"] + record["rotation_rad"] * dz + record["scale"] * dp
        records.append(record)

    return modelDz, modelDp, records


def _fitCameraTerms(errDz, errDp, dzNominal, dpNominal, cameraId, use, fitAgcRotation):
    """Fit an offset, and maybe a rotation, to each camera's errors (mm)."""
    modelDz, modelDp = np.zeros(len(errDz)), np.zeros(len(errDz))
    records = []
    for cid in np.unique(cameraId):
        rows = cameraId == cid
        fit = rows & use
        nFit = np.sum(fit)
        record = {"agc_camera_id": int(cid), "dz_mm": 0.0, "dp_mm": 0.0, "rotation_rad": 0.0}
        if nFit >= 2:
            if fitAgcRotation:
                dz0, dp0 = np.mean(dzNominal[fit]), np.mean(dpNominal[fit])
                dz, dp = dzNominal[fit] - dz0, dpNominal[fit] - dp0
                design = np.column_stack(
                    [
                        np.r_[np.ones(nFit), np.zeros(nFit)],
                        np.r_[np.zeros(nFit), np.ones(nFit)],
                        np.r_[-dp, dz],
                    ]
                )
                solution = np.linalg.lstsq(design, np.r_[errDz[fit], errDp[fit]], rcond=None)[0]
                record.update(zip(["dz_mm", "dp_mm", "rotation_rad"], solution, strict=True))

                dz, dp = dzNominal[rows] - dz0, dpNominal[rows] - dp0
                modelDz[rows] = record["dz_mm"] - record["rotation_rad"] * dp
                modelDp[rows] = record["dp_mm"] + record["rotation_rad"] * dz
            else:
                record.update(dz_mm=np.mean(errDz[fit]), dp_mm=np.mean(errDp[fit]))
                modelDz[rows], modelDp[rows] = record["dz_mm"], record["dp_mm"]
        records.append(record)

    return modelDz, modelDp, records


def _termsTable(records: list[dict], index: str) -> pd.DataFrame:
    """Return fitted terms with their units: microns and arcsec rather than mm and radians."""
    terms = pd.DataFrame(records).set_index(index)
    terms["dz_mm"] = mmToUm(terms.dz_mm)
    terms["dp_mm"] = mmToUm(terms.dp_mm)
    terms["rotation_rad"] = radToArcsec(terms.rotation_rad)

    return terms.rename(columns={"dz_mm": "dz_um", "dp_mm": "dp_um", "rotation_rad": "rotation_arcsec"})


def fitGlobalModel(
    agcData: pd.DataFrame,
    valid: ArrayLike | None = None,
    fitPfiRotation: bool = True,
    fitPfiScale: bool = False,
    fitAgcOffsets: bool = False,
    fitAgcRotation: bool = False,
) -> GlobalModelFit:
    """Fit a model of the guide errors in the zenith frame.

    These are the fits of drp_stella's ``ag_to_zenith_offset``. The model
    has an offset of each AG exposure, and optionally a rotation and scale
    about the boresight, plus optionally a constant offset (and rotation) of
    each camera. It is fitted in the zenith frame
    (`pfs.drp.qa.guiders.coordinates.pfiToZenith`), where gravity, and so
    flexure, doesn't depend on the rotator angle. With camera terms, the
    exposure and camera terms are fitted in turn, three times, as in
    drp_stella; with exposure rotations too, that doesn't fully converge, and
    can leave a few microns in the terms.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data in hardware coordinates, from
        `pfs.drp.qa.guiders.queries.readAgcData`. Each exposure is converted
        to the zenith frame at its median ``insrot``.
    valid : array-like of `bool`, optional
        The stars to fit, one element per row; default those with
        ``agc_match_flags == 1``. The model is evaluated for every star.
    fitPfiRotation : `bool`
        Fit a rotation of each exposure about the boresight.
    fitPfiScale : `bool`
        Fit a scale of each exposure about the boresight.
    fitAgcOffsets : `bool`
        Fit an offset of each camera, the same in every exposure.
    fitAgcRotation : `bool`
        With ``fitAgcOffsets``, also fit a rotation of each camera.

    Returns
    -------
    fit : `GlobalModelFit`
        The model positions and the fitted terms.
    """
    data = agcData.reset_index(drop=True)
    exposureId = data.agc_exposure_id.to_numpy()
    cameraId = data.agc_camera_id.to_numpy()
    use = selectValidMatches(data) if valid is None else np.asarray(valid, dtype=bool)
    insrot = data.groupby("agc_exposure_id").insrot.transform("median").to_numpy(dtype=float)

    dzNominal, dpNominal = pfiToZenith(data.agc_nominal_x_mm, data.agc_nominal_y_mm, insrot)
    dzCenter, dpCenter = pfiToZenith(data.agc_center_x_mm, data.agc_center_y_mm, insrot)
    errDz, errDp = dzCenter - dzNominal, dpCenter - dpNominal

    cameraDz, cameraDp = np.zeros(len(data)), np.zeros(len(data))
    cameraRecords = []
    for _ in range(3 if fitAgcOffsets else 1):
        exposureDz, exposureDp, exposureRecords = _fitExposureTerms(
            errDz - cameraDz,
            errDp - cameraDp,
            dzNominal,
            dpNominal,
            exposureId,
            use,
            fitPfiRotation,
            fitPfiScale,
        )
        if fitAgcOffsets:
            cameraDz, cameraDp, cameraRecords = _fitCameraTerms(
                errDz - exposureDz, errDp - exposureDp, dzNominal, dpNominal, cameraId, use, fitAgcRotation
            )

    dx, dy = zenithToPfi(exposureDz + cameraDz, exposureDp + cameraDp, insrot)
    data["agc_model_x_mm"] = data.agc_nominal_x_mm + dx
    data["agc_model_y_mm"] = data.agc_nominal_y_mm + dy
    data = addOffsets(data, "model")
    data.index = agcData.index

    return GlobalModelFit(
        agcData=data,
        exposures=_termsTable(exposureRecords, "agc_exposure_id"),
        cameras=_termsTable(cameraRecords, "agc_camera_id") if fitAgcOffsets else None,
    )


# Smoothing and guide errors


def smoothAgcData(agcData: pd.DataFrame, nExposures: int) -> pd.DataFrame:
    """Return a copy of AG data smoothed along each guide star's exposures.

    Each numeric column is replaced by its running mean over ``nExposures``
    of the star's AG exposures, centered, in order of ``agc_exposure_id``.
    IDs, flags (columns ending in ``_flag`` or ``_flags``) and
    ``shutter_open`` aren't smoothed, nor are times and booleans.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data.
    nExposures : `int`
        Length of the boxcar; no smoothing if <= 1.

    Returns
    -------
    agcData : `pandas.DataFrame`
        A smoothed copy.
    """
    data = agcData.reset_index(drop=True)
    if nExposures > 1:
        columns = [
            column
            for column in data.select_dtypes("number").columns
            if column not in _UNSMOOTHED and not column.endswith(("_flag", "_flags"))
        ]
        ordered = data.sort_values(["guide_star_id", "agc_exposure_id"], kind="stable")
        rolling = ordered.groupby("guide_star_id", dropna=False)[columns].rolling(
            nExposures, min_periods=1, center=True
        )
        smoothed = rolling.mean().droplevel(0).sort_index()
        data[columns] = smoothed[columns].to_numpy()

    data.index = agcData.index

    return data


def estimateGuideErrors(
    agcData: pd.DataFrame,
    reference: str = "center0",
    stat: str = "median",
    position: str = "center",
    byVisit: bool = False,
    smoothing: int = 1,
    recenterPerVisit: bool = False,
    subtractMedian: bool = False,
    includeClosedShutter: bool = False,
) -> pd.DataFrame:
    """Estimate the mean guide error of each AG camera in each exposure or visit.

    The reference positions are computed from all the rows. Only the valid
    matches (``agc_match_flags == 1``) taken while the spectrograph shutters
    weren't closed are then averaged, unless ``includeClosedShutter``.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data in hardware coordinates, from
        `pfs.drp.qa.guiders.queries.readAgcData`.
    reference : `str`
        The reference position (one of
        `pfs.drp.qa.guiders.coordinates.REFERENCES`); drp_stella's
        ``guideStrategy``, whose ``center0PerVisit`` and ``nominal0PerVisit``
        are ``center0_visit`` and ``nominal0_visit``.
    stat : `str`
        The statistic for the reference positions; see
        `pfs.drp.qa.guiders.coordinates.addReferencePositions`.
    position : `str`
        ``center`` for the error of the stars' centers, or ``nominal`` for
        the movement of their nominal positions (drp_stella's
        ``showNominal``).
    byVisit : `bool`
        Average over each visit, rather than each AG exposure.
    smoothing : `int`
        Smooth each star's offsets over this many exposures first; see
        `smoothAgcData`.
    recenterPerVisit : `bool`
        Subtract each visit's mean.
    subtractMedian : `bool`
        Subtract each camera's median.
    includeClosedShutter : `bool`
        Include AG exposures taken while the shutters were closed.

    Returns
    -------
    guideErrors : `pandas.DataFrame`
        One row per AG exposure (or visit) and camera, with ``pfs_visit_id``,
        ``agc_exposure_id`` (a mean with ``byVisit``), ``agc_camera_id``, the
        means of ``agc_nominal_[xy]_mm``, ``altitude``, ``azimuth``,
        ``insrot``, ``taken_at`` and ``guide_delta_{altitude,azimuth,insrot}``,
        the max of ``shutter_open``, and ``dx_um`` and ``dy_um``, the mean
        ``position`` minus ``reference`` (microns). ``attrs["offset"]`` says
        which offset it is, e.g. ``"center - center0"``.
    """
    if position not in ("center", "nominal"):
        raise ValueError(f"Unknown position {position!r}; valid: center, nominal")

    data = addReferencePositions(agcData, reference, stat)
    for xy, column in zip("xy", referenceColumns(reference), strict=True):
        data[f"d{xy}_um"] = mmToUm(data[f"agc_{position}_{xy}_mm"] - data[column])

    keep = selectValidMatches(data)
    if not includeClosedShutter:
        keep &= data.shutter_open > 0
    data = smoothAgcData(data[keep], smoothing)

    aggregates = {
        "pfs_visit_id": ("pfs_visit_id", "first"),
        "agc_exposure_id": ("agc_exposure_id", "mean"),
        "agc_nominal_x_mm": ("agc_nominal_x_mm", "mean"),
        "agc_nominal_y_mm": ("agc_nominal_y_mm", "mean"),
        **{column: (column, "mean") for column in ("altitude", "azimuth", "insrot", "taken_at")},
        **{f"guide_delta_{c}": (f"guide_delta_{c}", "mean") for c in ("altitude", "azimuth", "insrot")},
        "shutter_open": ("shutter_open", "max"),
        "dx_um": ("dx_um", "mean"),
        "dy_um": ("dy_um", "mean"),
    }
    keys = ["pfs_visit_id" if byVisit else "agc_exposure_id", "agc_camera_id"]
    for key in keys:
        aggregates.pop(key, None)
    guideErrors = data.groupby(keys, as_index=False).agg(**aggregates)
    columns = ["pfs_visit_id", "agc_exposure_id", "agc_camera_id"]
    guideErrors = guideErrors[columns + [c for c in guideErrors.columns if c not in columns]]

    for d in ("dx_um", "dy_um"):
        if subtractMedian:
            guideErrors[d] -= guideErrors.groupby("agc_camera_id")[d].transform("median")
        if recenterPerVisit:
            guideErrors[d] -= guideErrors.groupby("pfs_visit_id")[d].transform("mean")

    guideErrors.attrs["offset"] = f"{position} - {reference}"

    return guideErrors


# Drift rates


@dataclass(frozen=True, eq=False)
class DriftFit:
    """The result of `fitDriftRate`.

    Attributes
    ----------
    offsets : `pandas.DataFrame`
        One row per AG exposure (and camera), in order of time: the
        exposure's ``taken_at`` and ``time_min`` (minutes since the first),
        the mean ``agc_nominal_[xy]_mm``, ``dx_um`` and ``dy_um`` (see
        ``attrs["offset"]``) and, with ``radialTangential``,
        ``dradial_um`` and ``dtangential_um``.
    rates : `pandas.Series`
        For each component (``x`` and ``y``, or ``radial`` and
        ``tangential``), ``<component>_rate_um_per_min`` and
        ``<component>_offset_um`` (at the first exposure); also
        ``pfs_visit_id`` (the first visit) and the mean ``altitude``,
        ``azimuth``, ``insrot`` and ``exptime``. Named by the first visit.
    """

    offsets: pd.DataFrame
    rates: pd.Series


def fitDriftRate(
    agcData: pd.DataFrame,
    reference: str = "nominal",
    stat: str = "mean",
    position: str = "center",
    agcExposureIds: Iterable[int] | None = None,
    radialTangential: bool = True,
    robust: bool = True,
    smoothing: int = 1,
    subtractMeanOffset: bool = True,
    byCamera: bool = True,
) -> DriftFit:
    """Fit the drift of the guide stars against time.

    Only valid matches (`selectValidMatches`) in AG exposures taken with the
    spectrograph shutters open (``shutter_open == 1``) are used.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data in hardware coordinates, normally for one visit.
    reference : `str`
        The reference position (one of
        `pfs.drp.qa.guiders.coordinates.REFERENCES`); drp_stella's
        ``guideStrategy``.
    stat : `str`
        The statistic for the reference positions; see
        `pfs.drp.qa.guiders.coordinates.addReferencePositions`.
    position : `str`
        ``center``, or ``nominal`` for the movement of the nominal positions
        (drp_stella's ``showNominal``, with ``reference="nominal0"``).
    agcExposureIds : iterable of `int`, optional
        Only use these AG exposures; the reference positions are computed
        from them alone.
    radialTangential : `bool`
        Fit the components along and across the line from the boresight to
        each camera, rather than x and y.
    robust : `bool`
        Fit with `scipy.stats.siegelslopes`, rather than least squares.
    smoothing : `int`
        Smooth each star's offsets over this many exposures first; see
        `smoothAgcData`.
    subtractMeanOffset : `bool`
        Subtract each camera's mean offset, or the overall mean if not
        ``byCamera``.
    byCamera : `bool`
        Average each camera separately, rather than all the stars of an
        exposure together.

    Returns
    -------
    fit : `DriftFit`
        The offsets and the fitted rates.

    Raises
    ------
    ValueError
        If no AG exposure is left to fit.
    """
    if position not in ("center", "nominal"):
        raise ValueError(f"Unknown position {position!r}; valid: center, nominal")

    visit = int(agcData.pfs_visit_id.min())
    data = agcData
    if agcExposureIds is not None:
        data = data[data.agc_exposure_id.isin(list(agcExposureIds))]
    data = addReferencePositions(data, reference, stat)
    for xy, column in zip("xy", referenceColumns(reference), strict=True):
        data[f"d{xy}_um"] = mmToUm(data[f"agc_{position}_{xy}_mm"] - data[column])
    data = data[(data.shutter_open == 1).to_numpy() & selectValidMatches(data)]
    if data.empty:
        raise ValueError(f"No AG exposures with the shutters open to fit, from visit {visit}")
    means = data[["altitude", "azimuth", "insrot", "exptime"]].mean()

    data = smoothAgcData(data, smoothing)
    keys = ["agc_exposure_id", "agc_camera_id"] if byCamera else ["agc_exposure_id"]
    offsets = data.groupby(keys, as_index=False).agg(
        taken_at=("taken_at", "first"),
        agc_nominal_x_mm=("agc_nominal_x_mm", "mean"),
        agc_nominal_y_mm=("agc_nominal_y_mm", "mean"),
        dx_um=("dx_um", "mean"),
        dy_um=("dy_um", "mean"),
    )
    if subtractMeanOffset:
        for d in ("dx_um", "dy_um"):
            if byCamera:
                offsets[d] -= offsets.groupby("agc_camera_id")[d].transform("mean")
            else:
                offsets[d] -= offsets[d].mean()

    offsets = offsets.sort_values("taken_at", kind="stable", ignore_index=True)
    offsets["time_min"] = (offsets.taken_at - offsets.taken_at.iloc[0]).dt.total_seconds() / 60
    if radialTangential:
        theta = np.arctan2(offsets.agc_nominal_y_mm, offsets.agc_nominal_x_mm)
        c, s = np.cos(theta), np.sin(theta)
        offsets["dradial_um"] = c * offsets.dx_um + s * offsets.dy_um
        offsets["dtangential_um"] = -s * offsets.dx_um + c * offsets.dy_um
        components = ("radial", "tangential")
    else:
        components = ("x", "y")
    offsets.attrs["offset"] = f"{position} - {reference}"

    rates = {}
    for component in components:
        z = offsets[f"d{component}_um"].to_numpy()
        if robust:
            slope, intercept = scipy.stats.siegelslopes(z, offsets.time_min.to_numpy())
        else:
            slope, intercept = np.polyfit(offsets.time_min.to_numpy(), z, 1)
        rates[f"{component}_rate_um_per_min"] = float(slope)
        rates[f"{component}_offset_um"] = float(intercept)
    rates["pfs_visit_id"] = visit
    rates.update(means.to_dict())

    return DriftFit(offsets=offsets, rates=pd.Series(rates, name=visit))


# Comparison with pfs_utils


@dataclass(frozen=True, eq=False)
class PfsUtilsComparison:
    """The result of `comparePfsUtilsPositions`.

    Attributes
    ----------
    stars : `pandas.DataFrame`
        One row per guide star: the columns of ``agcStars``, those of the
        star's first AG exposure, the median ``agc_nominal_[xy]_mm``,
        ``pfs_utils_[xy]_mm`` (hardware coordinates), their angles about the
        boresight ``agc_nominal_theta_rad`` and ``pfs_utils_theta_rad``, and
        ``delta_theta_arcsec``, the first minus the second.
    alignOffset_um : `tuple` [`float`, `float`] or `None`
        With ``alignCenterPosition``, the mean pfs_utils minus guider
        position (microns) subtracted from ``pfs_utils_[xy]_mm``.
    """

    stars: pd.DataFrame
    alignOffset_um: tuple[float, float] | None


def _hstToUtc(time) -> str:
    """Return an opdb time (HST) as a UTC string, as pfs_utils wants it."""
    time = pd.Timestamp(time)
    time = time.tz_convert("UTC").tz_localize(None) if time.tzinfo is not None else time + _HST_TO_UTC

    return time.strftime("%Y-%m-%d %H:%M:%S")


def comparePfsUtilsPositions(
    agcStars: pd.DataFrame,
    agcData: pd.DataFrame,
    nAgcExposures: int = 10,
    instPaCorrection_arcsec: float = 0.0,
    alignCenterPosition: bool = False,
) -> PfsUtilsComparison:
    """Compare the guider's nominal positions of a visit's guide stars with pfs_utils's.

    The guider's positions are the median of each star's
    ``agc_nominal_[xy]_mm`` over the visit's first ``nAgcExposures`` AG
    exposures; one exposure can miss a camera. pfs_utils's are from
    `pfs.utils.coordinates.CoordTransp.CoordinateTransform` (``sky_pfi``)
    at the config's field center and position angle, and the time of the
    first AG exposure.

    Parameters
    ----------
    agcStars : `pandas.DataFrame`
        The visit's guide stars, from
        `pfs.drp.qa.guiders.queries.readAGCStars` with ``pfs_visit_id``.
    agcData : `pandas.DataFrame`
        The visit's AG data, from `pfs.drp.qa.guiders.queries.readAgcData`.
    nAgcExposures : `int`
        How many AG exposures to average.
    instPaCorrection_arcsec : `float`
        Add this to the config's position angle (arcsec).
    alignCenterPosition : `bool`
        Shift pfs_utils's positions so that their mean matches the guider's.

    Returns
    -------
    comparison : `PfsUtilsComparison`
        The positions, and the shift applied.

    Raises
    ------
    ValueError
        If no guide star is in both ``agcStars`` and ``agcData``.

    Notes
    -----
    pfs_utils's PFI coordinates have the opdb's sign of y (see
    `pfs.drp.qa.guiders.coordinates`), so ``pfs_utils_y_mm`` is negated. In
    hardware coordinates angles run the other way, so
    ``delta_theta_arcsec`` has the opposite sign of drp_stella's
    ``delta_theta``, which was computed in the opdb's frame. The opdb's
    times are HST; pfs_utils is given UTC.
    """
    exposures = np.sort(agcData.agc_exposure_id.unique())[:nAgcExposures]
    data = agcData[agcData.agc_exposure_id.isin(exposures)]
    nominal = data.groupby("guide_star_id", as_index=False)[["agc_nominal_x_mm", "agc_nominal_y_mm"]].median()
    first = data.sort_values("agc_exposure_id", kind="stable").drop_duplicates("guide_star_id")
    first = first.drop(columns=["agc_nominal_x_mm", "agc_nominal_y_mm", "agc_camera_id", "pfs_visit_id"])
    stars = agcStars.merge(first.merge(nominal, on="guide_star_id"), on="guide_star_id")
    if stars.empty:
        raise ValueError("No guide star is in both agcStars and agcData")

    xy = CoordTransp.CoordinateTransform(
        np.array([stars.guide_star_ra, stars.guide_star_dec], dtype=float),
        "sky_pfi",
        za=90.0 - stars.altitude.iloc[0],
        cent=np.array([[stars.ra_center_config.mean()], [stars.dec_center_config.mean()]]),
        time=_hstToUtc(stars.taken_at.iloc[0]),
        pa=stars.pa_config.mean() + instPaCorrection_arcsec / 3600,
        pm=np.array([stars.guide_star_pm_ra, stars.guide_star_pm_dec], dtype=float),
        par=stars.guide_star_parallax.to_numpy(dtype=float),
    )
    stars["pfs_utils_x_mm"] = xy[0]
    stars["pfs_utils_y_mm"] = -xy[1]

    alignOffset_um = None
    if alignCenterPosition:
        dx = (stars.pfs_utils_x_mm - stars.agc_nominal_x_mm).mean()
        dy = (stars.pfs_utils_y_mm - stars.agc_nominal_y_mm).mean()
        stars["pfs_utils_x_mm"] -= dx
        stars["pfs_utils_y_mm"] -= dy
        alignOffset_um = (float(mmToUm(dx)), float(mmToUm(dy)))

    stars["agc_nominal_theta_rad"] = np.arctan2(stars.agc_nominal_y_mm, stars.agc_nominal_x_mm)
    stars["pfs_utils_theta_rad"] = np.arctan2(stars.pfs_utils_y_mm, stars.pfs_utils_x_mm)
    # Wrapped to (-pi, pi], as the angles jump by 2 pi across -x (AG4).
    dtheta = np.angle(np.exp(1j * (stars.agc_nominal_theta_rad - stars.pfs_utils_theta_rad)))
    stars["delta_theta_arcsec"] = radToArcsec(dtheta)

    return PfsUtilsComparison(stars=stars, alignOffset_um=alignOffset_um)


# Image sizes and focus


def addImageSizes(agcData: pd.DataFrame, useTraceRadius: bool = True) -> pd.DataFrame:
    """Return a copy of AG data with the stars' image sizes.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data with the second moments ``mxx``, ``myy`` and ``mxy`` (pix^2)
        and ``agc_data_flags``.
    useTraceRadius : `bool`
        Use the trace radius, sqrt((mxx + myy)/2), rather than the
        determinant radius, (mxx myy - mxy^2)^(1/4).

    Returns
    -------
    agcData : `pandas.DataFrame`
        A copy with ``rms_pix``, the 1-D rms size (NaN for impossible
        moments); ``fwhm_arcsec``, the FWHM of a Gaussian of that rms; and
        ``left``, whether the star is on the left half of its detector (its
        detection isn't flagged RIGHT).
    """
    agcData = agcData.copy()
    mxx, myy, mxy = (agcData[column].to_numpy(dtype=float) for column in ("mxx", "myy", "mxy"))
    if useTraceRadius:
        rms = np.sqrt(np.where((mxx < 0) | (myy < 0), np.nan, 0.5 * (mxx + myy)))
    else:
        det = mxx * myy - mxy**2
        rms = np.where((mxx < 0) | (myy < 0) | (det < 0), np.nan, det) ** 0.25

    agcData["rms_pix"] = rms
    agcData["fwhm_arcsec"] = pixToArcsec(GAUSSIAN_FWHM_PER_SIGMA * rms)
    agcData["left"] = (agcData.agc_data_flags & int(SourceDetectionFlags.RIGHT)) == 0

    return agcData


def momentDifferenceToPiston(deltaMxx_pix2: ArrayLike, includePistonCorrection: bool = False) -> ArrayLike:
    """Convert a difference of 1-D second moments to a focus error (mm).

    ics_agActor's calibration, which takes the difference of 2-D moments,
    twice the 1-D ones.

    Parameters
    ----------
    deltaMxx_pix2 : array-like
        The difference of a 1-D second moment (e.g. ``rms_pix**2``) between
        the left and right halves of a detector (pix^2).
    includePistonCorrection : `bool`
        Subtract `FOCUS_PISTON_OFFSET_MM`, as ics_agActor does.

    Returns
    -------
    piston : array-like
        The focus error (mm).
    """
    piston = np.multiply(2 * FOCUS_PISTON_MM_PER_PIX2, deltaMxx_pix2)
    if includePistonCorrection:
        piston = np.subtract(piston, FOCUS_PISTON_OFFSET_MM)

    return piston


def correctAgActorFocus(agcData: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of AG data with ics_agActor's early focus errors corrected.

    Before visit `AGACTOR_FOCUS_FIX_VISIT`, ics_agActor computed the focus
    error from 4(a^2 + b^2) rather than a^2 + b^2 (INSTRM-2501), so
    ``guide_delta_z`` and ``guide_delta_z1`` to ``guide_delta_z6`` are
    rescaled on those visits' rows.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data, from `pfs.drp.qa.guiders.queries.readAgcData`.

    Returns
    -------
    agcData : `pandas.DataFrame`
        A corrected copy.
    """
    agcData = agcData.copy()
    offset = momentDifferenceToPiston(0, includePistonCorrection=True)
    early = (agcData.pfs_visit_id < AGACTOR_FOCUS_FIX_VISIT).to_numpy()
    for column in _AGACTOR_FOCUS_COLUMNS:
        agcData[column] = np.where(early, offset + (agcData[column] - offset) / 4, agcData[column])

    return agcData


def estimateFocusErrors(
    agcData: pd.DataFrame, byCamera: bool = True, focusColumn: str = "m2_off3"
) -> pd.DataFrame:
    """Estimate the focus error of each AG exposure from its stars' sizes.

    As ics_agActor does, the focus error comes from the difference between
    the image sizes on the left and right halves of the detectors, which are
    focused differently: the median ``rms_pix`` of each half of each camera.
    Without ``byCamera``, the medians of each half are the medians over the
    cameras.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data with ``rms_pix`` and ``left`` (see `addImageSizes`), of the
        stars to use (e.g. `selectIsolatedGaiaStars`). Only the valid matches
        (`selectValidMatches`) are used.
    byCamera : `bool`
        Estimate the focus error of each camera, rather than of the
        exposure.
    focusColumn : `str`
        The focus position to average: ``m2_off3`` or ``m2_pos3``.

    Returns
    -------
    focusErrors : `pandas.DataFrame`
        One row per AG exposure (and camera) with stars on both halves:
        ``agc_exposure_id``, (``agc_camera_id``), ``pfs_visit_id``, the mean
        ``altitude`` and ``insrot``, ``focus_position_mm``, the mean of
        ``focusColumn``, ``rms_left_pix`` and ``rms_right_pix``, and
        ``focus_error_um``.
    """
    agcData = agcData[selectValidMatches(agcData)]
    halves = agcData.groupby(["agc_exposure_id", "agc_camera_id", "left"]).rms_pix.median()
    keys = ["agc_exposure_id", "agc_camera_id"] if byCamera else ["agc_exposure_id"]
    rms = halves.groupby(level=[*keys, "left"]).median().unstack("left").reindex(columns=[True, False])
    rms = rms.dropna().rename(columns={True: "rms_left_pix", False: "rms_right_pix"})
    rms.columns.name = None

    focusErrors = agcData.groupby(keys).agg(
        pfs_visit_id=("pfs_visit_id", "first"),
        altitude=("altitude", "mean"),
        insrot=("insrot", "mean"),
        focus_position_mm=(focusColumn, "mean"),
    )
    focusErrors = focusErrors.join(rms, how="inner")
    deltaMxx = focusErrors.rms_left_pix**2 - focusErrors.rms_right_pix**2
    focusErrors["focus_error_um"] = mmToUm(momentDifferenceToPiston(deltaMxx))

    return focusErrors.reset_index()


def averageByFocusPosition(
    focusPosition_mm: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    useMedian: bool = True,
    resolution_mm: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Average x and y over the points at each focus position.

    Parameters
    ----------
    focusPosition_mm : array-like
        Focus position of each point (mm); rounded to ``resolution_mm``.
        Points with no focus position are ignored.
    x, y : array-like
        Values to average, one per point.
    useMedian : `bool`
        Take the median, rather than the mean.
    resolution_mm : `float`
        Focus positions closer than this are the same (mm); default a micron.

    Returns
    -------
    x, y : `numpy.ndarray`
        The averages, in order of focus position. NaNs are ignored.
    """
    focus = np.round(np.asarray(focusPosition_mm, dtype=float) / resolution_mm)
    points = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)})
    grouped = points.groupby(focus)  # NaN keys are dropped
    averages = grouped.median() if useMedian else grouped.mean()

    return averages.x.to_numpy(), averages.y.to_numpy()
