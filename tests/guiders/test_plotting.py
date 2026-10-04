"""Tests for ``pfs.drp.qa.guiders.plotting``.

The smoke tests call every plot with each option combination used today:
its defaults, each of drp_stella's options on its own, and the settings of
ics_pfsPlotActor (master ``e430a58a``), on the real AG data in ``data/``.
They check that the plot draws something, makes no pyplot figure when given
one, and leaves its inputs alone. The other tests check the fixes to
drp_stella's plots, each with a negative control.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from pfs.drp.qa.guiders.analysis import (
    GuiderFitConfig,
    addImageSizes,
    comparePfsUtilsPositions,
    estimateFocusErrors,
    estimateGuideErrors,
    fitDriftRate,
    fitGuiderModel,
    selectIsolatedGaiaStars,
    selectValidMatches,
)
from pfs.drp.qa.guiders.coordinates import (
    AGC_CAMERA_CENTERS_MM,
    AGC_RING_RADIUS_MM,
    GUIDER_FOCUS_UM_PER_M2_OFF3_MM,
    addOffsets,
    rotXY,
)
from pfs.drp.qa.guiders.plotting import (
    FormatCoord,
    GuiderPlot,
    GuiderPlotConfig,
    plotDriftRate,
    plotFocus,
    plotFocusByAG,
    plotGuideErrors,
    plotPfsUtilsComparison,
    showAGCameraCartoon,
    showAgcErrorsForVisits,
    showAgcErrorsForVisitsByCamera,
    showGuiderErrors,
    showGuiderErrorsByParams,
    showTelescopeErrors,
)

# ics_pfsPlotActor's ShowGuiderErrors and ShowGuiderErrorsCombined defaults, split into the fit's
# options and the plot's.
ACTOR_FIT = GuiderFitConfig(
    modelBoresightOffset=False,
    modelCCDOffset=False,
    solveForAGTransforms=False,
    onlyShutterOpen=True,
    maxGuideError_um=100,
    maxPosError_um=40,
    agcExposureStride=1,
)
ACTOR_PLOT = GuiderPlotConfig(
    showAverageGuideStarPos=False,
    showAverageGuideStarPath=False,
    showGuideStars=True,
    showGuideStarsAsPoints=True,
    showGuideStarsAsArrows=False,
    showGuideStarPositions=False,
    rotateToAG1Down=False,
    guideErrorEstimate_um=50,
    pfiScaleReduction=1,
    gstarExpansion=10,
    guideStarFrac=0.3,
)
# ics_pfsPlotActor's FocusPlot (the AG actor's focus) and FocusSweepPlot (the stars' focus and FWHM).
ACTOR_FOCUS = {
    "showAGActorFocus": True,
    "showOpdbFocus": False,
    "showFWHM": False,
    "showFocusSets": True,
    "yLimits_um": 220,
    "minFWHM_arcsec": 0.3,
    "maxFWHM_arcsec": 1.59,
    "forceAlpha": 0.5,
    "connectMedian": False,
}
ACTOR_FOCUS_SWEEP = {
    "showAGActorFocus": False,
    "showOpdbFocus": True,
    "showFWHM": True,
    "showMedian": True,
    "connectMedian": False,
    "yLimits_um": 220,
    "minFWHM_arcsec": 0.3,
    "maxFWHM_arcsec": 1.59,
}


@pytest.fixture(autouse=True)
def closeFigures():
    """Close any pyplot figure a test makes."""
    yield
    plt.close("all")


def checkPlot(plot: GuiderPlot, fig: Figure | None = None, nAxes: int | None = None) -> None:
    """Check that a plot drew something, on ``fig`` if given, without pyplot."""
    assert isinstance(plot, GuiderPlot)
    assert plot.artists
    assert plot.axes.ndim == 2
    if nAxes is not None:
        assert plot.axes.size == nAxes
    if fig is not None:
        assert plot.fig is fig
        assert plt.get_fignums() == []


def nPoints(ax) -> int:
    """Return the number of points a panel's scatters show."""
    return sum(len(c.get_offsets()) for c in ax.collections if isinstance(c, PathCollection))


def optionId(kwargs: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in kwargs.items()) or "defaults"


# Smoke tests


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"byTime": True},
        {"yLimit_um": 100, "showLegend": False},
        {"pfsVisitIds": [148284, 148285]},
        {"agcExposureIds": range(1091800, 1091820)},
        {"reference": "boresight"},
    ],
    ids=optionId,
)
def testShowAgcErrorsForVisitsSmoke(realAgcData, kwargs):
    agcData = realAgcData("raster")
    original = agcData.copy()
    fig = Figure()

    checkPlot(showAgcErrorsForVisits(agcData, fig=fig, **kwargs), fig, nAxes=3)
    pd.testing.assert_frame_equal(agcData, original)


BY_CAMERA_OPTIONS = [
    {},
    {"agcCameraIds": [1, 2]},
    {"plotBy": "altitude"},
    {"plotBy": "insrot"},
    {"colorBy": "visit"},
    {"colorBy": "altitude"},
    {"colorBy": "insrot"},
    {"showCamerasAsLegend": False},
    {"showAltInsrot": True},
    {"plotXY": True},
    {"plotXY": True, "connectDxDy": True, "showCovariance": True},
    {"plotXY": True, "plotXYStride": 3},
    {"plotXY": True, "plotXYStride": None, "nVisitMin": 2},
    {"plotXY": True, "plotPerCamera": True},
    {"plotDzDfocus": True},
    {"plotDzDfocus": True, "plotPerCamera": True, "xLimit_um": 50},
    {"plotPerCamera": True},
    {"rotateToZenith": False},
    {"fitPfiModel": True},
    {"fitPfiModel": True, "fitPfiRotation": False, "fitPfiScale": False},
    {"fitPfiModel": True, "fitAgcOffsets": True, "fitAgcRotation": True},
    {"drawVisitBoundaries": True},
    {"yLimit_um": 30, "alpha": 1, "scatterMarkerSize": 4},
]


@pytest.mark.parametrize("kwargs", BY_CAMERA_OPTIONS, ids=optionId)
def testShowAgcErrorsForVisitsByCameraSmoke(realAgcData, kwargs):
    agcData = realAgcData("raster")
    original = agcData.copy()
    fig = Figure()

    checkPlot(showAgcErrorsForVisitsByCamera(agcData, fig=fig, **kwargs), fig)
    pd.testing.assert_frame_equal(agcData, original)


GUIDER_PLOT_OPTIONS = [
    {},
    {"showGuideStarsAsPoints": False, "showGuideStarsAsArrows": True},
    {"showGuideStarsAsArrows": True},  # points take precedence
    {"showGuideStarPositions": True},
    {"showAverageGuideStarPos": True},
    {"showAverageGuideStarPath": True},
    {"showGuideStars": False, "showAverageGuideStarPos": True, "showAverageGuideStarPath": True},
    {"showByVisit": False},
    {"rotateToAG1Down": True},
    {"rotateToAG1Down": True, "showAverageGuideStarPos": True},
    {"pfiScaleReduction": 2, "gstarExpansion": 5},
    {"guideStarFrac": 1, "markerSize": 2, "alpha": 0.5, "colormap": "plasma"},
]
GUIDER_FIT_OPTIONS = [
    GuiderFitConfig(),
    GuiderFitConfig(maxGuideError_um=0, maxPosError_um=0),
    GuiderFitConfig(onlyShutterOpen=False, agcExposureStride=3),
    GuiderFitConfig(pfsVisitIdMin=148285, pfsVisitIdMax=148290),
]


@pytest.mark.parametrize("kwargs", GUIDER_PLOT_OPTIONS, ids=optionId)
def testShowGuiderErrorsSmoke(realAgcData, kwargs):
    fit = fitGuiderModel(realAgcData("raster"), GuiderFitConfig(maxGuideError_um=0, maxPosError_um=0))
    original = fit.agcData.copy()
    fig = Figure()

    checkPlot(showGuiderErrors(fit, GuiderPlotConfig(**kwargs), name="raster", fig=fig), fig, nAxes=1)
    pd.testing.assert_frame_equal(fit.agcData, original)


@pytest.mark.parametrize("fitConfig", [*GUIDER_FIT_OPTIONS, ACTOR_FIT], ids=lambda c: repr(c)[15:60])
def testShowGuiderErrorsFitOptions(realAgcData, fitConfig):
    fit = fitGuiderModel(realAgcData("raster"), fitConfig)
    checkPlot(showGuiderErrors(fit, ACTOR_PLOT, fig=Figure()))


@pytest.mark.parametrize("params", [["altitude"], ["altitude", "insrot"], ["altitude", "insrot", "azimuth"]])
@pytest.mark.parametrize("rotateToAG1Down", [False, True])
def testShowGuiderErrorsByParamsSmoke(realAgcData, params, rotateToAG1Down):
    fit = fitGuiderModel(realAgcData("raster"))
    fig = Figure()

    plot = showGuiderErrorsByParams(fit, params, GuiderPlotConfig(rotateToAG1Down=rotateToAG1Down), fig=fig)

    checkPlot(plot, fig)
    assert len(plot.colorbars) == len(params)
    assert sum(ax.get_visible() for ax in plot.axes.flat) == len(params)


@pytest.mark.parametrize("showTheta", [False, True])
@pytest.mark.parametrize("name", ["raster", "allSky"])
def testShowTelescopeErrorsSmoke(realAgcData, name, showTheta):
    fig = Figure()
    checkPlot(showTelescopeErrors(realAgcData(name), showTheta=showTheta, fig=fig), fig, nAxes=4)


@pytest.mark.parametrize(
    "fitKwargs",
    [{}, {"radialTangential": False}, {"byCamera": False}, {"robust": False, "smoothing": 3}],
    ids=optionId,
)
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"byTime": False}, {"showCamera": False}, {"fitTrend": False, "name": "all sky"}],
    ids=optionId,
)
def testPlotDriftRateSmoke(realAgcData, fitKwargs, kwargs):
    fig = Figure()
    checkPlot(
        plotDriftRate(fitDriftRate(realAgcData("allSky"), **fitKwargs), fig=fig, **kwargs), fig, nAxes=2
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"colorBy": "pfs_visit_id"},
        {"colorBy": "time"},
        {"showAGMean": False},
        {"drawTrack": True},
        {"rotateToAG1Down": True},
        {"expand": 2},
        {"showClosedShutter": True},
        {"showCartoon": False, "name": "raster"},
    ],
    ids=optionId,
)
@pytest.mark.parametrize("byVisit", [False, True])
def testPlotGuideErrorsSmoke(realAgcData, kwargs, byVisit):
    guideErrors = estimateGuideErrors(realAgcData("raster"), byVisit=byVisit, includeClosedShutter=True)
    original = guideErrors.copy()
    fig = Figure()

    checkPlot(plotGuideErrors(guideErrors, fig=fig, **kwargs), fig, nAxes=1)
    pd.testing.assert_frame_equal(guideErrors, original)


@pytest.fixture
def pfsUtilsComparison(realAgcData, realAgcStars):
    """Return the comparison with pfs_utils of the all-sky exposure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pfs_utils's clamped parallaxes
        return comparePfsUtilsPositions(realAgcStars(148258), realAgcData("allSky"))


@pytest.mark.parametrize(
    "kwargs", [{}, {"compress": 5}, {"plotUsingScatter": True}, {"showCartoon": False}], ids=optionId
)
def testPlotPfsUtilsComparisonSmoke(pfsUtilsComparison, kwargs):
    fig = Figure()
    checkPlot(plotPfsUtilsComparison(pfsUtilsComparison, fig=fig, **kwargs), fig, nAxes=1)


FOCUS_OPTIONS = [
    {},
    {"agcCameraIds": [1, 3]},
    {"plotBy": "agc_exposure_id"},
    {"plotBy": "altitude"},
    {"plotBy": "insrot"},
    {"colorBy": "visit"},
    {"colorBy": "altitude"},
    {"colorBy": "insrot"},
    {"showAGActorFocus": False},
    {"showOpdbFocus": False},
    {"showFWHM": False},
    {"plotPerCamera": True},
    {"plotPerCamera": True, "colorBy": "visit", "plotBy": "agc_exposure_id"},
    {"showPfiFocusPosition": True},
    {"averageByFocusPosition": True},
    {"showMedian": True},
    {"showOnlyMedian": True},
    {"showMedian": True, "connectMedian": False},
    {"showCameraId": True},
    {"showCameraId": True, "showMedian": True},
    {"showFocusSets": True},
    {"onlyGuideStars": False},
    {"plotFrac": 0.5},
    {"ditherScale": 0},
    {"yLimits_um": (-300, 200)},
    {"yLimits_um": None},
    {"indicateFocusPosition": True},
    {"useTraceRadius": False},
    {"magMin": 10, "magMax": 20},
    {"minFWHM_arcsec": 0.3, "maxFWHM_arcsec": 1.59},
    {"useM2Off3": False},
    {"forceAlpha": 0.5, "scatterMarkerSize": 4},
    ACTOR_FOCUS_SWEEP,
    *({**ACTOR_FOCUS, "plotBy": plotBy} for plotBy in ["agc_exposure_id", "altitude", "insrot", "focus"]),
]


@pytest.mark.parametrize("kwargs", FOCUS_OPTIONS, ids=optionId)
def testPlotFocusSmoke(realAgcData, kwargs):
    agcData = realAgcData("focusSweep")
    original = agcData.copy()
    fig = Figure()

    checkPlot(plotFocus(agcData, fig=fig, **kwargs), fig)
    pd.testing.assert_frame_equal(agcData, original)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"onlyGuideStars": False},
        {"byExposureId": True},
        {"byCamera": False},
        {"maxFocusError_um": 300},
        {"agcExposureIdMin": 1091600, "agcExposureIdMax": 1091700},
        {"useTraceRadius": False, "showLegend": True},
    ],
    ids=optionId,
)
def testPlotFocusByAGSmoke(realAgcData, kwargs):
    fig = Figure()
    checkPlot(plotFocusByAG(realAgcData("focusSweep"), fig=fig, **kwargs), fig, nAxes=1)


@pytest.mark.parametrize(
    "plot, name, args",
    [
        (showAgcErrorsForVisits, "raster", ()),
        (showAgcErrorsForVisitsByCamera, "raster", ()),
        (showTelescopeErrors, "raster", ()),
        (plotFocus, "focusSweep", ()),
        (plotFocusByAG, "focusSweep", ()),
    ],
)
def testPlotsMakeAFigure(realAgcData, plot, name, args):
    """Given no figure or axes, a plot makes a pyplot figure, and titles it."""
    result = plot(realAgcData(name), *args)

    assert plt.get_fignums() == [result.fig.number]
    assert result.fig.get_suptitle()


# Fixes


def testShowAgcErrorsForVisitsSign(realAgcData):
    """The guide errors are center minus nominal, as in the other plots.

    drp_stella plotted nominal minus center.
    """
    agcData = realAgcData("raster")
    plot = showAgcErrorsForVisits(agcData, fig=Figure())

    valid = addOffsets(agcData[selectValidMatches(agcData)], "nominal")
    expected = valid.groupby("agc_exposure_id")[["dx_nominal_um", "dy_nominal_um"]].mean()
    errors = plot.data.set_index("agc_exposure_id")
    np.testing.assert_allclose(errors.dx_um, expected.dx_nominal_um)
    np.testing.assert_allclose(errors.dy_um, expected.dy_nominal_um)

    # The x panel shows them: each visit's line.
    xPanel = plot.axes[1, 0]
    shown = np.concatenate([line.get_ydata() for line in xPanel.lines if line.get_label().isdigit()])
    np.testing.assert_allclose(np.sort(shown), np.sort(expected.dx_nominal_um))

    # Negative control: drp_stella's nominal - center.
    assert not np.allclose(errors.dx_um, -expected.dx_nominal_um, atol=1)


def testShowAgcErrorsForVisitsPlotActorAxes(realAgcData):
    """ics_pfsPlotActor's three axes, given without the figure, are drawn on and titled at the top."""
    fig = Figure()
    ax1 = fig.add_subplot(311)
    axes = [ax1, fig.add_subplot(312, sharex=ax1), fig.add_subplot(313, sharex=ax1)]

    plot = showAgcErrorsForVisits(realAgcData("raster"), axes=axes)

    assert plot.fig is fig
    assert list(plot.axes[:, 0]) == axes
    assert len(fig.axes) == 3
    assert axes[0].get_title().startswith("pfs_visit_id 148284..148292")
    assert not fig.get_suptitle()


def testShowAgcErrorsForVisitsNoValidMatches(realAgcData):
    agcData = realAgcData("raster").assign(agc_match_flags=0)
    with pytest.raises(ValueError, match="No valid matches"):
        showAgcErrorsForVisits(agcData, fig=Figure())


def testShowAgcErrorsForVisitsByCameraPerVisitMeans(realAgcData):
    """With ``plotXYStride=None`` every camera's panel shows its per-visit means.

    drp_stella computed them for the first panel only.
    """
    agcData = realAgcData("raster")
    plot = showAgcErrorsForVisitsByCamera(
        agcData, plotXY=True, plotPerCamera=True, plotXYStride=None, fig=Figure()
    )
    nVisits = plot.data.groupby("agc_camera_id").pfs_visit_id.nunique()

    assert plot.axes.shape == (2, 3)
    for cid, ax in enumerate(plot.axes.flat):
        assert nPoints(ax) == nVisits[cid] <= 9

    # Negative control: each AG exposure, as drp_stella plotted in the panels after the first.
    perExposure = showAgcErrorsForVisitsByCamera(agcData, plotXY=True, plotPerCamera=True, fig=Figure())
    assert all(nPoints(ax) > 9 for ax in perExposure.axes.flat)


def testShowAgcErrorsForVisitsByCameraLabels(realAgcData):
    """Each camera's label has its points' colour, C0 for AG1.

    drp_stella's labels were two colours off.
    """
    plot = showAgcErrorsForVisitsByCamera(realAgcData("raster"), plotPerCamera=True, fig=Figure())

    for cid, ax in enumerate(plot.axes[0]):
        (label,) = ax.texts
        assert label.get_text() == f"AG{cid + 1}"
        assert label.get_color() == f"C{cid}"
        scatter = ax.collections[0]
        np.testing.assert_allclose(
            scatter.to_rgba(scatter.get_array())[:, :3], [to_rgb(f"C{cid}")] * nPoints(ax)
        )
        # Negative control: drp_stella's colour.
        assert label.get_color() != f"C{cid + 2}"


def testShowAgcErrorsForVisitsByCameraFloatCameraIds(realAgcData):
    """Float camera IDs (from a join with NaNs) work, and pick the camera's AG actor focus.

    drp_stella looked for ``guide_delta_z2.0``.
    """
    agcData = realAgcData("raster")
    plot = showAgcErrorsForVisitsByCamera(
        agcData.astype({"agc_camera_id": float}), plotDzDfocus=True, plotPerCamera=True, fig=Figure()
    )

    focus = agcData.groupby(["agc_exposure_id", "agc_camera_id"])[
        [f"guide_delta_z{i}" for i in range(1, 7)]
    ].first()
    for row in plot.data.sample(20, random_state=1).itertuples():
        expected = (
            1e3 * focus.loc[(row.agc_exposure_id, row.agc_camera_id), f"guide_delta_z{row.agc_camera_id + 1}"]
        )
        assert row.focus_error_um == pytest.approx(expected)
        # Negative control: another camera's.
        other = (
            1e3
            * focus.loc[
                (row.agc_exposure_id, row.agc_camera_id), f"guide_delta_z{(row.agc_camera_id + 1) % 6 + 1}"
            ]
        )
        assert row.focus_error_um != pytest.approx(other)


def testShowAgcErrorsForVisitsByCameraFiveCameras(realAgcData):
    """Per-camera XY panels are laid out for any number of cameras.

    drp_stella assumed six, and raised an IndexError otherwise.
    """
    plot = showAgcErrorsForVisitsByCamera(
        realAgcData("raster"), agcCameraIds=range(5), plotXY=True, plotPerCamera=True, fig=Figure()
    )

    assert plot.axes.shape == (2, 3)
    assert [ax.get_visible() for ax in plot.axes.flat] == [True] * 5 + [False]


def testShowAgcErrorsForVisitsByCameraOffsets(realAgcData):
    """Each camera's median, then each exposure's mean, is subtracted."""
    plot = showAgcErrorsForVisitsByCamera(realAgcData("raster"), fig=Figure())
    offsets = plot.data

    for d in ("dx_um", "dy_um"):
        np.testing.assert_allclose(offsets.groupby("agc_exposure_id")[d].mean(), 0, atol=1e-9)
    # A rotation of the camera's offsets to the zenith frame keeps their size.
    unrotated = showAgcErrorsForVisitsByCamera(realAgcData("raster"), rotateToZenith=False, fig=Figure()).data
    assert np.hypot(offsets.dx_um, offsets.dy_um).median() == pytest.approx(
        np.hypot(unrotated.dx_um, unrotated.dy_um).median(), rel=0.2
    )
    # Negative control: they differ in direction.
    assert not np.allclose(offsets.dx_um, unrotated.dx_um, atol=1)


def testShowAgcErrorsForVisitsByCameraHexbinScale(realAgcData):
    """With ``showAltInsrot`` every panel shares the colorbar's scale, symmetric about 0."""
    plot = showAgcErrorsForVisitsByCamera(realAgcData("raster"), showAltInsrot=True, fig=Figure())

    limits = {hexbin.get_clim() for hexbin in plot.artists}
    assert len(limits) == 1
    ((vmin, vmax),) = limits
    assert vmin == -vmax
    assert all(np.nanmax(np.abs(hexbin.get_array())) <= vmax for hexbin in plot.artists)
    assert plot.colorbars[0].norm.vmax == vmax

    # Negative control: each panel's own range, which each hexbin would take by itself.
    ranges = {(float(np.nanmin(h.get_array())), float(np.nanmax(h.get_array()))) for h in plot.artists}
    assert len(ranges) > 1


def testShowGuiderErrorsPlotActorAllSky(realAgcData):
    """With ics_pfsPlotActor's settings every AG exposure of the all-sky visit is plotted.

    The guide errors of its valid matches are 44 µm (median), under the actor's
    100 µm cut. drp_stella counted the invalid matches too, 693 µm, and plotted
    none.
    """
    agcData = realAgcData("allSky")
    fig = Figure()
    ax = fig.add_subplot(111)

    plot = showGuiderErrors(fitGuiderModel(agcData, ACTOR_FIT), ACTOR_PLOT, ax=ax)

    assert plot.axes[0, 0] is ax
    assert plot.data.agc_exposure_id.nunique() == agcData[agcData.shutter_open == 1].agc_exposure_id.nunique()

    # Negative control: every match counted as valid.
    allMatches = fitGuiderModel(agcData.assign(agc_match_flags=1), ACTOR_FIT)
    assert showGuiderErrors(allMatches, ACTOR_PLOT, fig=Figure()).data is None


def testShowGuiderErrorsColorbar(realAgcData):
    """Redrawn with its colorbars, the plot updates them rather than adding more (ics_pfsPlotActor's livePlot)."""
    fit = fitGuiderModel(realAgcData("allSky"), ACTOR_FIT)
    fig = Figure()
    ax = fig.add_subplot(111)

    first = showGuiderErrors(fit, ACTOR_PLOT, ax=ax)
    ax.cla()
    second = showGuiderErrors(fit, ACTOR_PLOT, ax=ax, colorbars=first.colorbars)

    assert second.colorbars[0] is first.colorbars[0]
    assert len(fig.axes) == 2  # the plot and its colorbar
    assert second.colorbars[0].mappable is second.artists[-1]

    # Negative control: without them, a second colorbar.
    ax.cla()
    showGuiderErrors(fit, ACTOR_PLOT, ax=ax)
    assert len(fig.axes) == 3


def testShowGuiderErrorsArrows(realAgcData):
    """The arrows' legend works with matplotlib >= 3.9 (drp_stella used ``legendHandles``)."""
    fit = fitGuiderModel(realAgcData("raster"))
    config = GuiderPlotConfig(showGuideStarsAsPoints=False, showGuideStarsAsArrows=True)
    plot = showGuiderErrors(fit, config, fig=Figure())

    legend = plot.axes[0, 0].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [f"AG{cid + 1}" for cid in range(6)]
    assert all(handle.get_alpha() == 1 for handle in legend.legend_handles)
    assert all(quiver.get_alpha() == 0.5 for quiver in plot.artists)


@pytest.mark.parametrize("pfiScaleReduction", [1, 2])
def testShowGuiderErrorsCameraPositions(realAgcData, pfiScaleReduction):
    """The stars are drawn about their camera's position, reduced by ``pfiScaleReduction``.

    drp_stella didn't reduce the cameras' positions, only the stars'.
    """
    fit = fitGuiderModel(realAgcData("raster"))
    config = GuiderPlotConfig(guideStarFrac=1, pfiScaleReduction=pfiScaleReduction)
    data = showGuiderErrors(fit, config, fig=Figure()).data

    for cid, camera in data.groupby("agc_camera_id"):
        x0, y0 = np.array(AGC_CAMERA_CENTERS_MM[cid]) / pfiScaleReduction
        offset = np.hypot(camera.xPlot - camera.xOff - x0, camera.yPlot - camera.yOff - y0)
        assert offset.max() < 10 / pfiScaleReduction
        # The points are the camera's position plus the offsets.
        np.testing.assert_allclose(camera.xPlot - camera.xOff, camera.xPos.mean())


def testShowGuiderErrorsRotated(realAgcData):
    """With ``rotateToAG1Down`` the offsets are rotated by minus the rotator angle, onto the ring."""
    fit = fitGuiderModel(realAgcData("raster"))
    data = showGuiderErrors(fit, GuiderPlotConfig(guideStarFrac=1, rotateToAG1Down=True), fig=Figure()).data

    xOff, yOff = rotXY(-np.deg2rad(data.insrot), data.dx_model_um, data.dy_model_um)
    np.testing.assert_allclose(data.xOff, xOff)
    np.testing.assert_allclose(data.yOff, yOff)
    np.testing.assert_allclose(np.hypot(data.xPlot - data.xOff, data.yPlot - data.yOff), AGC_RING_RADIUS_MM)


def testShowGuiderErrorsTitle(realAgcData):
    """The title gives the visits, clamped to the fit's limits, and says when closed shutters are included.

    drp_stella gave the visit range only with ``pfs_visitIdMax``, clamped it
    the wrong way, and said "including open shutter".
    """
    agcData = realAgcData("raster")
    title = showGuiderErrors(fitGuiderModel(agcData), fig=Figure()).fig.get_suptitle()
    assert title.startswith("148284..148292\n")

    fitConfig = GuiderFitConfig(pfsVisitIdMin=148286, pfsVisitIdMax=148289, onlyShutterOpen=False)
    title = showGuiderErrors(
        fitGuiderModel(agcData, fitConfig), name="raster", fig=Figure()
    ).fig.get_suptitle()
    assert title.startswith("raster  148286..148289\n")
    assert "(including closed shutter)" in title
    assert "Boresight and AGs offset and rotation/scale removed" in title


def testGuiderPlotConfigValidation():
    with pytest.raises(ValueError, match="guideStarFrac"):
        GuiderPlotConfig(guideStarFrac=0)
    with pytest.raises(ValueError, match="guideStarFrac"):
        GuiderPlotConfig(guideStarFrac=1.5)


def testShowTelescopeErrorsTheta(realAgcData):
    """``showTheta`` plots the rotator offset in arcsec; otherwise its motion at the AG cameras.

    drp_stella's ``showTheta`` drew an empty panel.
    """
    agcData = realAgcData("raster")
    shutterOpen = agcData[agcData.shutter_open > 0]
    theta = shutterOpen.groupby("agc_exposure_id").guide_delta_insrot.mean()

    thetaPanel = showTelescopeErrors(agcData, showTheta=True, fig=Figure()).axes[1, 0]
    np.testing.assert_allclose(thetaPanel.collections[0].get_offsets()[:, 1], theta)

    umPanel = showTelescopeErrors(agcData, fig=Figure()).axes[1, 0]
    expected = 1e3 * AGC_RING_RADIUS_MM * np.deg2rad(theta / 3600)
    np.testing.assert_allclose(umPanel.collections[0].get_offsets()[:, 1], expected)
    # Negative control: the two differ (by 1.17 µm per arcsec).
    assert not np.allclose(expected, theta)


def testShowTelescopeErrorsColorbars(realAgcData):
    """ics_pfsPlotActor-style redraws update the four colorbars."""
    agcData = realAgcData("raster")
    fig = Figure()
    axes = fig.subplots(2, 2)
    first = showTelescopeErrors(agcData, axes=axes)
    for ax in axes.flat:
        ax.cla()
    second = showTelescopeErrors(agcData, axes=axes, colorbars=first.colorbars)

    assert all(a is b for a, b in zip(first.colorbars, second.colorbars, strict=True))
    assert len(fig.axes) == 8


@pytest.mark.parametrize("radialTangential", [True, False])
def testPlotDriftRateLine(realAgcData, radialTangential):
    """The fitted lines have the fitted rates."""
    fit = fitDriftRate(realAgcData("allSky"), radialTangential=radialTangential)
    plot = plotDriftRate(fit, fig=Figure())
    components = ["radial", "tangential"] if radialTangential else ["y", "x"]

    for ax, component in zip(plot.axes[:, 0], components, strict=True):
        line = ax.lines[-1]
        (t0, t1), (y0, y1) = line.get_xdata(), line.get_ydata()
        assert (y1 - y0) / (t1 - t0) == pytest.approx(fit.rates[f"{component}_rate_um_per_min"])
        assert y0 == pytest.approx(fit.rates[f"{component}_offset_um"])
        assert f"{fit.rates[f'{component}_rate_um_per_min']:.3f} µm/min" in ax.texts[0].get_text()


def testPlotGuideErrorsPoints(realAgcData):
    """Each camera's guide error is drawn once, and the closed-shutter ones when asked.

    drp_stella drew the open-shutter points twice, and never the closed ones.
    """
    guideErrors = estimateGuideErrors(realAgcData("raster"), includeClosedShutter=True)
    nOpen = int((guideErrors.shutter_open > 0).sum())
    nClosed = int((guideErrors.shutter_open == 0).sum())
    assert nOpen and nClosed

    plot = plotGuideErrors(guideErrors, showAGMean=False, fig=Figure())
    assert nPoints(plot.axes[0, 0]) == nOpen

    plot = plotGuideErrors(guideErrors, showAGMean=False, showClosedShutter=True, fig=Figure())
    assert nPoints(plot.axes[0, 0]) == nOpen + nClosed
    # Negative control: drp_stella's count, with the open-shutter points twice.
    assert nPoints(plot.axes[0, 0]) != 2 * nOpen


def testPlotGuideErrorsRotatedCenters(realAgcData):
    """With ``rotateToAG1Down`` the cameras' centers are rotated with their points."""
    guideErrors = estimateGuideErrors(realAgcData("allSky"))
    plot = plotGuideErrors(guideErrors, rotateToAG1Down=True, fig=Figure())

    (centers,) = [
        line for line in plot.axes[0, 0].lines if line.get_marker() == "+" and len(line.get_xdata()) > 1
    ]
    medians = guideErrors.groupby("agc_camera_id")[["agc_nominal_x_mm", "agc_nominal_y_mm"]].median()
    x, y = rotXY(-np.deg2rad(guideErrors.insrot.mean()), medians.agc_nominal_x_mm, medians.agc_nominal_y_mm)
    np.testing.assert_allclose(centers.get_xdata(), x)
    np.testing.assert_allclose(centers.get_ydata(), y)

    # Negative control: the centers unrotated, 165 degrees away (insrot -165).
    assert (
        np.hypot(
            centers.get_xdata() - medians.agc_nominal_x_mm, centers.get_ydata() - medians.agc_nominal_y_mm
        ).min()
        > 100
    )


def testPlotGuideErrorsSharedColours(realAgcData):
    """The cameras' points and their means share one colour scale."""
    guideErrors = estimateGuideErrors(realAgcData("raster"), includeClosedShutter=True)
    plot = plotGuideErrors(guideErrors, showClosedShutter=True, fig=Figure())

    limits = {(artist.norm.vmin, artist.norm.vmax) for artist in plot.artists}
    assert limits == {(guideErrors.agc_exposure_id.min(), guideErrors.agc_exposure_id.max())}


def testPlotPfsUtilsComparisonAgrees(pfsUtilsComparison):
    """The guider's and pfs_utils's positions of each star are drawn at the same place."""
    plot = plotPfsUtilsComparison(pfsUtilsComparison, fig=Figure())
    pfsUtils, agc = (np.column_stack([line.get_xdata(), line.get_ydata()]) for line in plot.artists)

    assert 1e3 * np.abs(pfsUtils - agc).max() < 1
    assert "visit: 148258" in plot.fig.get_suptitle()


@pytest.fixture
def focusSweep(realAgcData) -> pd.DataFrame:
    return realAgcData("focusSweep")


def fwhmPanel(plot: GuiderPlot, column: int = 0):
    return plot.axes[-1, column]


def testPlotFocusFwhmOnce(focusSweep):
    """Each star's FWHM is drawn once.

    drp_stella drew them twice when plotting against focus.
    """
    plot = plotFocus(focusSweep, fig=Figure())
    assert nPoints(fwhmPanel(plot)) == len(plot.data)
    np.testing.assert_allclose(
        np.sort(fwhmPanel(plot).collections[0].get_offsets()[:, 1]), np.sort(plot.data.fwhm_arcsec)
    )


def testPlotFocusPlotFrac(focusSweep):
    """``plotFrac`` subsets the stars' positions, sizes and colours alike.

    drp_stella subset some and not others, so failed with a length mismatch.
    """
    plot = plotFocus(focusSweep, plotFrac=0.5, fig=Figure())
    scatter = fwhmPanel(plot).collections[0]

    assert 0.4 * len(plot.data) < len(scatter.get_offsets()) < 0.6 * len(plot.data)
    assert len(scatter.get_facecolors()) == len(scatter.get_offsets())


def testPlotFocusStarsFocusError(focusSweep):
    """The stars' focus errors are `estimateFocusErrors`, and rise through focus.

    606 µm per mm of M2_OFF3 near focus (see test_realData.py).
    """
    plot = plotFocus(focusSweep, showAGActorFocus=False, showFWHM=False, fig=Figure())
    focusErrors = estimateFocusErrors(plot.data, byCamera=True)

    for line in plot.axes[0, 0].lines:
        if not line.get_label().startswith("AG"):
            continue
        cid = int(line.get_label()[2:]) - 1
        camera = focusErrors[focusErrors.agc_camera_id == cid]
        np.testing.assert_allclose(line.get_xdata(), camera.focus_position_mm)
        np.testing.assert_allclose(line.get_ydata(), camera.focus_error_um)
    slope = np.polyfit(focusErrors.focus_position_mm, focusErrors.focus_error_um, 1)[0]
    assert slope > 300

    # Negative control: with the halves of the detectors swapped, the slope has the wrong sign.
    swapped = addImageSizes(plot.data).assign(left=lambda d: ~d.left)
    swappedErrors = estimateFocusErrors(swapped, byCamera=True)
    assert np.polyfit(swappedErrors.focus_position_mm, swappedErrors.focus_error_um, 1)[0] < -300


@pytest.mark.parametrize("connectMedian", [True, False])
def testPlotFocusMediansByVisit(focusSweep, connectMedian):
    """With ``colorBy`` other than camera, the median options apply to both rows of focus errors."""
    kwargs = {"colorBy": "visit", "showFWHM": False, "connectMedian": connectMedian}
    plot = plotFocus(focusSweep, showOnlyMedian=True, fig=Figure(), **kwargs)
    agActorPanel, starsPanel = plot.axes[:, 0]
    focusErrors = estimateFocusErrors(plot.data, byCamera=False)
    byExposure = plot.data.groupby("agc_exposure_id").first()

    for ax, x, y in [
        (starsPanel, focusErrors.focus_position_mm, focusErrors.focus_error_um),
        (agActorPanel, byExposure.focus_position_mm, 1e3 * byExposure.guide_delta_z),
    ]:
        assert nPoints(ax) == 0
        (median,) = [line for line in ax.lines if line.get_color() == "black" and len(line.get_xdata()) > 2]
        assert (median.get_linestyle() == "-") == connectMedian
        expected = y.groupby(x.round(3).to_numpy()).median()
        np.testing.assert_allclose(median.get_ydata(), expected)

    # Negative control: without the options, every AG exposure's point and no median.
    plain = plotFocus(focusSweep, fig=Figure(), **kwargs)
    assert nPoints(plain.axes[1, 0]) == len(focusErrors)
    assert nPoints(plain.axes[0, 0]) == len(byExposure)
    assert not [line for ax in plain.axes.flat for line in ax.lines if len(line.get_xdata()) > 2]


def testPlotFocusAgActorMedians(focusSweep):
    """The median options apply to the AG actor's focus errors: each camera's median at each M2_OFF3.

    drp_stella drew every AG exposure's point in this row whatever the options.
    """
    plot = plotFocus(focusSweep, showOpdbFocus=False, showFWHM=False, showOnlyMedian=True, fig=Figure())
    ax = plot.axes[0, 0]
    byCamera = plot.data.groupby(["agc_exposure_id", "agc_camera_id"]).first().reset_index()

    medians = [line for line in ax.lines if line.get_label().startswith("AG")]
    assert medians
    for line in medians:
        cid = int(line.get_label()[2:]) - 1
        camera = byCamera[byCamera.agc_camera_id == cid].dropna(subset=f"guide_delta_z{cid + 1}")
        expected = 1e3 * camera.groupby(camera.focus_position_mm.round(3))[f"guide_delta_z{cid + 1}"].median()
        np.testing.assert_allclose(line.get_xdata(), expected.index)
        np.testing.assert_allclose(line.get_ydata(), expected)
        assert line.get_linestyle() == "-"

    # Negative control: without the options, one point per AG exposure and camera (338, against 39 medians).
    plain = plotFocus(focusSweep, showOpdbFocus=False, showFWHM=False, fig=Figure()).axes[0, 0]
    nShown = sum(len(line.get_xdata()) for line in plain.lines if line.get_label().startswith("AG"))
    nValues = sum(
        byCamera[f"guide_delta_z{cid + 1}"][byCamera.agc_camera_id == cid].notna().sum() for cid in range(6)
    )
    assert nShown == nValues > 5 * sum(len(line.get_xdata()) for line in medians)


@pytest.mark.parametrize("connectMedian", [True, False])
def testPlotFocusMedianMarkers(focusSweep, connectMedian):
    """Each half keeps its symbol in the medians: circles for the left halves, stars for the right.

    The medians are joined in order of x with ``connectMedian``. drp_stella drew the left halves' medians as
    stars, the right halves' symbol, and never joined them.
    """
    plot = plotFocus(
        focusSweep, showCameraId=True, showMedian=True, connectMedian=connectMedian, fig=Figure()
    )
    data = plot.data
    medians = [line for line in fwhmPanel(plot).lines if len(line.get_xdata())]

    assert {line.get_marker() for line in medians} == {"o", "*"}
    for line in medians:
        assert line.get_linestyle() == ("-" if connectMedian else "None")
        camera = data[data.agc_camera_id == int(line.get_color()[1:])]
        isLeft = line.get_marker() == "o"
        half = (
            camera[camera.left == isLeft]
            .groupby("agc_exposure_id")
            .agg(x=("focus_position_mm", "mean"), y=("fwhm_arcsec", "median"))
            .sort_values("x", kind="stable")
        )
        np.testing.assert_allclose(line.get_xdata(), half.x)
        np.testing.assert_allclose(line.get_ydata(), half.y)
        half = half.y
        # Negative control: the other half's medians, in the same AG exposures.
        other = (
            camera[camera.left != isLeft].groupby("agc_exposure_id").fwhm_arcsec.median().reindex(half.index)
        )
        assert np.nanmax(np.abs(other.to_numpy() - half.to_numpy())) > 0.05


def testPlotFocusAxes2d(focusSweep):
    """Every helper takes the 2-D axes: visit boundaries, cursor readouts and focus sets on every panel.

    drp_stella iterated over the rows of the axes as if they were panels.
    """
    plot = plotFocus(
        focusSweep, plotBy="agc_exposure_id", plotPerCamera=True, showFocusSets=True, fig=Figure()
    )
    nVisits = plot.data.pfs_visit_id.nunique()

    assert plot.axes.shape == (3, 6)
    for ax in plot.axes.flat:
        assert isinstance(ax.format_coord, FormatCoord)
        boundaries = [line for line in ax.lines if line.get_alpha() == 0.25 and line.get_color() == "black"]
        assert len(boundaries) == nVisits - 1
        assert len(ax.patches) > 1  # the focus sets


def testPlotFocusPlotActorAxes(focusSweep):
    """ics_pfsPlotActor's flat list of axes, given without the figure, works without its AxesGrid."""
    fig = Figure()
    axes = fig.subplots(2, 1, sharex=True, height_ratios=[2, 3], squeeze=False).flatten()

    plot = plotFocus(focusSweep, axes=list(axes), **ACTOR_FOCUS_SWEEP)

    assert plot.fig is fig
    assert plot.axes.shape == (2, 1)
    assert len(fig.axes) == 2

    with pytest.raises(ValueError, match="needs 3x1 axes, not 2"):
        plotFocus(focusSweep, axes=list(axes))


def testPlotFocusColorbar(focusSweep):
    """With ``colorBy`` other than camera, one colorbar, updated when redrawn."""
    fig = Figure()
    axes = fig.subplots(3, 1, squeeze=False)
    first = plotFocus(focusSweep, colorBy="visit", axes=axes)
    for ax in axes.flat:
        ax.cla()
    second = plotFocus(focusSweep, colorBy="visit", axes=axes, colorbars=first.colorbars)

    assert len(first.colorbars) == 1
    assert second.colorbars[0] is first.colorbars[0]
    assert len(fig.axes) == 4


def clickAt(fig: Figure, ax, x: float, y: float) -> None:
    """Send a click at data coordinates (x, y) of ``ax``."""
    xPix, yPix = ax.transData.transform((x, y))
    event = MouseEvent("button_press_event", fig.canvas, xPix, yPix, button=1)
    fig.canvas.callbacks.process("button_press_event", event)


@pytest.mark.parametrize("plotPerCamera", [False, True])
def testShowFocusFit(focusSweep, plotPerCamera):
    """A click sets the M2_OFF3 of best focus; the top-left panel shows the focus error expected.

    drp_stella's ``ShowFocusFit`` failed with 2-D axes.
    """
    fig = Figure()
    plot = plotFocus(focusSweep, plotPerCamera=plotPerCamera, indicateFocusPosition=True, fig=fig)
    top = plot.axes[0, 0]
    nLines = len(top.lines)

    clickAt(fig, plot.axes[-1, -1], 3.2, 1)

    fitLine = top.lines[nLines]
    x, y = fitLine.get_xdata(), fitLine.get_ydata()
    assert (y[1] - y[0]) / (x[1] - x[0]) == pytest.approx(GUIDER_FOCUS_UM_PER_M2_OFF3_MM)
    assert np.interp(3.2, x, y) == pytest.approx(0, abs=1e-6)
    assert "M2_OFF3 = 3.20mm" in [text.get_text() for text in top.texts]
    for ax in plot.axes.flat:
        assert any(np.allclose(line.get_xdata(), 3.2) for line in ax.lines)

    # A second click replaces the line.
    clickAt(fig, plot.axes[-1, -1], 3.3, 1)
    assert len(top.lines) == nLines + 2  # the line and this panel's marker


def testShowFocusFitReplaced(focusSweep):
    """Redrawing a focus plot in the same figure replaces its click handler."""
    fig = Figure()
    axes = fig.subplots(3, 1, squeeze=False)
    nHandlers = len(fig.canvas.callbacks.callbacks["button_press_event"])  # the figure's own
    plotFocus(focusSweep, axes=axes)
    for ax in axes.flat:
        ax.cla()
    plotFocus(focusSweep, axes=axes)

    assert len(fig.canvas.callbacks.callbacks["button_press_event"]) == nHandlers + 1


def testPlotFocusOnlyGuideStars(makeAgcData):
    """``onlyGuideStars`` keeps only isolated GAIA stars."""
    agcData = makeAgcData(nVisit=1, stars=True)
    assert not selectIsolatedGaiaStars(agcData).all()

    plot = plotFocus(agcData, fig=Figure())
    assert selectIsolatedGaiaStars(plot.data).all()

    # Negative control: without it, the HSC stars are used too.
    assert not selectIsolatedGaiaStars(plotFocus(agcData, onlyGuideStars=False, fig=Figure()).data).all()


def testPlotFocusByAGOnlyGuideStars(makeAgcData):
    """``onlyGuideStars`` changes which stars give the focus errors.

    drp_stella's ``onlyGuideStars`` was never applied.
    """
    agcData = makeAgcData(nVisit=2, stars=True)
    # Make the HSC stars much larger on the left halves of the detectors.
    hsc = ~selectIsolatedGaiaStars(agcData) & ((agcData.agc_data_flags & 1) == 0).to_numpy()
    agcData.loc[hsc, ["mxx", "myy"]] *= 4

    gaia = plotFocusByAG(agcData, fig=Figure()).data
    every = plotFocusByAG(agcData, onlyGuideStars=False, fig=Figure()).data

    assert len(gaia) == len(every)
    assert not np.allclose(gaia.focus_error_um, every.focus_error_um, atol=1)


def testPlotFocusByAGRelative(realAgcData):
    """Each visit's mean of the cameras but AG1 is subtracted, then the overall mean; then negated."""
    relative = plotFocusByAG(realAgcData("focusSweep"), fig=Figure()).data

    assert relative.focus_error_um.mean() == pytest.approx(0, abs=1e-9)
    others = relative[relative.agc_camera_id != 0].groupby("pfs_visit_id").focus_error_um.mean()
    np.testing.assert_allclose(others - others.mean(), 0, atol=1e-9)


def testFormatCoord(realAgcData):
    """The cursor readout names the AG exposure's visit and design, from the data.

    drp_stella's read "(x, y) = (1001, y: 3.14)" and queried the opdb on every
    mouse move.
    """
    agcData = realAgcData("raster")
    aid = int(agcData.agc_exposure_id.iloc[0])
    visit = int(agcData.pfs_visit_id.iloc[0])

    formatter = FormatCoord("agc_exposure_id", agcData, {visit: "raster center"})
    assert formatter(aid + 0.3, 3.14159) == f"(x, y) = ({aid}, 3.14)  pfs_visit_id: {visit} (raster center)"
    assert FormatCoord("agc_exposure_id", agcData)(aid, 1) == f"(x, y) = ({aid}, 1.00)  pfs_visit_id: {visit}"
    assert formatter(0, 1) == "(x, y) = (0, 1.00)"
    # Not an AG exposure: no visit.
    assert FormatCoord("altitude", agcData)(aid, 1) == f"(x, y) = ({aid:.2f}, 1.00)"


@pytest.mark.parametrize("insrot_deg", [None, 0, 90])
def testShowAGCameraCartoon(insrot_deg):
    ax = Figure().add_subplot()
    cartoon = showAGCameraCartoon(ax, showInstrot=True, showUp=True, insrot_deg=insrot_deg)

    labels = {text.get_text(): text.get_position() for text in cartoon.texts if text.get_text().isdigit()}
    assert sorted(labels) == [str(cid + 1) for cid in range(6)]
    for cid, (x, y) in AGC_CAMERA_CENTERS_MM.items():
        if insrot_deg:
            x, y = rotXY(-np.deg2rad(insrot_deg), x, y)
        np.testing.assert_allclose(labels[str(cid + 1)], (x, y))
    # Negative control: AG2 is off the x axis, so the rotation moves it.
    if insrot_deg:
        assert not np.allclose(labels["2"], AGC_CAMERA_CENTERS_MM[1])


@pytest.mark.parametrize(
    "plot, kwargs",
    [(showAgcErrorsForVisitsByCamera, {"plotBy": "focus"}), (plotFocus, {"colorBy": "camera_id"})],
)
def testPlotsCheckChoices(realAgcData, plot, kwargs):
    with pytest.raises(ValueError, match="Unknown"):
        plot(realAgcData("focusSweep"), fig=Figure(), **kwargs)
