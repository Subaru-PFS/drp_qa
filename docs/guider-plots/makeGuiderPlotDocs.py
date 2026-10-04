#!/usr/bin/env python
"""Make the guider plots' documentation: README.md and a captioned sample of each plot.

Each plot in `pfs.drp.qa.guiders.plotting` is described once, in `PLOTS`: what
it shows, where its data come from, how to read it, and what to look for.
This script draws each plot from the real AG data of the tests (engineering
visits of Run 30, in ``tests/guiders/data``), overlays numbered callouts on
it, sets the description beside it, and writes ``<name>.png`` here; then it
writes ``README.md`` from the same descriptions.

Run from the top of drp_qa with pfs_utils on ``PYTHONPATH``::

    PYTHONPATH=python:/path/to/pfs_utils/python python docs/guider-plots/makeGuiderPlotDocs.py
"""

import textwrap
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure, SubFigure
from PIL import Image

from pfs.drp.qa.guiders import analysis, plotting

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "tests" / "guiders" / "data"
DPI = 72


def agcData(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"agcData-{name}.parquet")


def agcStars(visit: int) -> pd.DataFrame:
    stars = pd.read_parquet(DATA / "agcStars.parquet")
    return stars[stars.pfs_visit_id == visit].reset_index(drop=True)


@dataclass
class Callout:
    """A numbered marker on a panel of the sample, explained by ``text``.

    ``xy`` is in the panel's data coordinates, or a fraction of the panel if
    ``fraction``; ``panel`` is its (row, column).
    """

    text: str
    xy: tuple[float, float]
    panel: tuple[int, int] = (0, 0)
    fraction: bool = True


@dataclass
class PlotDoc:
    """The description of one plot, and how to draw its sample."""

    name: str
    title: str
    call: str
    sample: str
    what: str
    data: str
    lookFor: list[str]
    draw: Callable[[SubFigure], plotting.GuiderPlot]
    callouts: Callable[[plotting.GuiderPlot], list[Callout]] = lambda plot: []
    options: str = ""
    extra: list[str] = field(default_factory=list)


# The plots


def drawAgcErrors(fig):
    return plotting.showAgcErrorsForVisits(agcData("raster"), fig=fig)


def agcErrorsCallouts(plot):
    errors = plot.data
    first92 = errors[errors.pfs_visit_id == 148292].iloc[1]
    open85 = errors[(errors.pfs_visit_id == 148285) & (errors.shutter_open == 1)].iloc[3]
    jump = errors[errors.pfs_visit_id == 148286].dr_um.idxmax()
    return [
        Callout("Each visit has its own colour; its legend is on the top panel.", (0.05, 0.88)),
        Callout(
            "Large dots: AG exposures taken while the spectrograph shutters were open. These are the ones that "
            "matter for the science.",
            (open85.agc_exposure_id, open85.dr_um + 12),
            fraction=False,
        ),
        Callout(
            "A jump after the shutters close: in a raster scan the telescope moves to the next position, and "
            "the guider takes about 40 s to catch up.",
            (errors.agc_exposure_id[jump], errors.dr_um[jump] + 15),
            fraction=False,
        ),
        Callout(
            "148292, the start of the second scan, has no SpS exposure (shutter_open 2). Its first errors are "
            "hundreds of microns, as the guider starts on a new pointing.",
            (first92.agc_exposure_id - 6, 150),
            fraction=False,
        ),
        Callout(
            "x and y show the direction of the error, in hardware coordinates.", (0.03, 0.8), panel=(1, 0)
        ),
    ]


def drawByCamera(fig):
    return plotting.showAgcErrorsForVisitsByCamera(agcData("raster"), fig=fig)


def byCameraCallouts(plot):
    return [
        Callout("y points towards the zenith (rotateToZenith), x towards the Opt side.", (0.03, 0.85)),
        Callout(
            "Each point is one camera in one AG exposure, less the camera's median and the exposure's mean "
            "over the cameras: what is left is how the cameras move relative to each other.",
            (0.45, 0.62),
        ),
        Callout(
            "Outliers: a camera with one badly measured star, or a camera that moved.",
            (0.94, 0.12),
            panel=(1, 0),
        ),
    ]


def drawByCameraXY(fig):
    return plotting.showAgcErrorsForVisitsByCamera(
        agcData("raster"), plotXY=True, plotPerCamera=True, plotXYStride=None, showCovariance=True, fig=fig
    )


def byCameraXYCallouts(plot):
    return [
        Callout("One panel per camera; each point is the camera's mean offset in one visit.", (0.1, 0.12)),
        Callout(
            "The red ellipse holds 1 sigma of the points (clipped second moments). A long ellipse means the "
            "camera moves along one direction, e.g. flexure towards the zenith.",
            (0.82, 0.3),
            panel=(0, 2),
        ),
    ]


def drawGuiderErrors(fig):
    fitConfig = analysis.GuiderFitConfig(
        modelBoresightOffset=False, modelCCDOffset=False, maxGuideError_um=100
    )
    fit = analysis.fitGuiderModel(agcData("allSky"), fitConfig)
    return plotting.showGuiderErrors(fit, plotting.GuiderPlotConfig(guideStarFrac=0.3), fig=fig)


def guiderErrorsCallouts(plot):
    return [
        Callout(
            "+: the mean position of the camera's stars (mm). Each star is drawn at its offset from where "
            "the guider expected it (µm) from there.",
            (-120 - 25, 212 + 30),
            fraction=False,
        ),
        Callout(
            "A cloud off its +: the camera's stars are systematically away from where the guider expected "
            "them; here AG3 and AG5 by about 80 µm. Without the models these offsets mix the boresight's "
            "offset, rotation and scale with each camera's own.",
            (-120 + 45, 140),
            fraction=False,
        ),
        Callout("No AG1 stars: none of its matches in 148258 was valid.", (237, -40), fraction=False),
        Callout("The cartoon: where the cameras are, and the sense of the rotator.", (0.23, 0.12)),
        Callout("Colour: the AG exposure, so the time.", (0.9, 0.95)),
    ]


def drawGuiderErrorsByParams(fig):
    fit = analysis.fitGuiderModel(agcData("raster"))
    return plotting.showGuiderErrorsByParams(
        fit, ["altitude", "azimuth", "insrot", "agc_exposure_id"], fig=fig
    )


def byParamsCallouts(plot):
    return [
        Callout(
            "Each point is one camera's mean offset in one AG exposure, coloured by the quantity of the panel.",
            (0.62, 0.62),
        ),
        Callout(
            "A colour gradient across a cloud: the offsets follow that quantity.", (0.62, 0.62), panel=(0, 1)
        ),
    ]


def drawTelescopeErrors(fig):
    return plotting.showTelescopeErrors(agcData("allSky"), fig=fig)


def telescopeCallouts(plot):
    return [
        Callout("How often the AG actor sent each altitude and azimuth correction.", (0.1, 0.9)),
        Callout(
            "The same corrections, coloured by AG exposure: their order in time.", (0.1, 0.9), panel=(0, 1)
        ),
        Callout(
            "The rotator correction, as the motion it makes at the AG cameras (24 cm from the axis).",
            (0.1, 0.9),
            panel=(1, 0),
        ),
        Callout("The rotator correction by azimuth and altitude.", (0.1, 0.9), panel=(1, 1)),
    ]


def drawDriftRate(fig):
    return plotting.plotDriftRate(analysis.fitDriftRate(agcData("allSky")), fig=fig)


def driftCallouts(plot):
    return [
        Callout(
            "Each point: one camera's mean offset in one AG exposure, less the camera's mean.", (0.5, 0.85)
        ),
        Callout("The fitted line; its slope is the rate.", (0.92, 0.6)),
        Callout(
            "Radial: away from the boresight. Tangential: anticlockwise round it.", (0.03, 0.12), panel=(1, 0)
        ),
    ]


def drawGuideErrors(fig):
    guideErrors = analysis.estimateGuideErrors(agcData("allSky"))
    return plotting.plotGuideErrors(guideErrors, fig=fig)


def guideErrorsCallouts(plot):
    return [
        Callout(
            "Each camera's mean offset (µm) in each AG exposure, about the camera's median position (mm, red "
            "+). Here a tight cluster, a few microns across.",
            (120 + 45, 212 + 25),
            fraction=False,
        ),
        Callout("The mean over the cameras, about the boresight.", (35, -35), fraction=False),
        Callout(
            "No AG1 points: none of its matches in 148258 was valid. In a raster scan, each camera's points "
            "trace the scan instead.",
            (237, -45),
            fraction=False,
        ),
    ]


def drawPfsUtils(fig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pfs_utils's clamped parallaxes
        comparison = analysis.comparePfsUtilsPositions(agcStars(148258), agcData("allSky"))
    return plotting.plotPfsUtilsComparison(comparison, fig=fig)


def pfsUtilsCallouts(plot):
    return [
        Callout(
            "Each camera's stars, the cameras drawn 10 times closer to the boresight. The guider's positions "
            "(crosses) lie on pfs_utils's (circles).",
            (16, -16),
            fraction=False,
        ),
        Callout("d(theta): the median difference of the stars' angles about the boresight.", (0.5, 0.97)),
    ]


def drawFocus(fig):
    return plotting.plotFocus(agcData("focusSweep"), showMedian=True, yLimits_um=(-600, 400), fig=fig)


def focusCallouts(plot):
    return [
        Callout("The AG actor's focus error of each camera (guide_delta_z1-6).", (0.04, 0.78)),
        Callout(
            "The focus error from the stars' sizes on the two halves of each detector. It crosses 0 at best "
            "focus, here M2_OFF3 = 3.28 mm.",
            (3.28, 25),
            panel=(1, 0),
            fraction=False,
        ),
        Callout("Right axis: the same, as a change of M2_OFF3.", (0.97, 0.1), panel=(1, 0)),
        Callout(
            "Each star's FWHM: red on the left halves, green on the right, with their medians. Each half is "
            "sharpest on its own side of best focus.",
            (3.0, 2.3),
            panel=(2, 0),
            fraction=False,
        ),
        Callout(
            "Far from focus the focus error saturates, at about 450 µm. AG6's line is straight: in this "
            "trimmed sample it has stars on both halves at only three M2_OFF3.",
            (2.76, -330),
            panel=(1, 0),
            fraction=False,
        ),
    ]


def drawFocusByExposure(fig):
    return plotting.plotFocus(
        agcData("focusSweep"),
        plotBy="agc_exposure_id",
        showOpdbFocus=False,
        showFWHM=False,
        showFocusSets=True,
        yLimits_um=220,
        forceAlpha=0.5,
        fig=fig,
    )


def focusByExposureCallouts(plot):
    return [
        Callout("Shading: each run of AG exposures at one M2_OFF3 (showFocusSets).", (0.16, 0.93)),
        Callout(
            "Grey lines: the last AG exposure of each visit. The cursor readout names the visit.",
            (0.35, 0.08),
        ),
        Callout("The focus error steps with each move of M2_OFF3.", (0.75, 0.55)),
    ]


def drawFocusByAG(fig):
    return plotting.plotFocusByAG(agcData("focusSweep"), fig=fig)


def focusByAGCallouts(plot):
    return [
        Callout("One line per visit: each camera's focus relative to the others.", (0.25, 0.92)),
        Callout("Black stars, and the horizontal lines: each camera's mean over the visits.", (0.9, 0.35)),
    ]


PLOTS = [
    PlotDoc(
        name="showAgcErrorsForVisits",
        title="Guide error of each AG exposure",
        call="plotting.showAgcErrorsForVisits(agcData)",
        sample="Raster scan, visits 148284-148292.",
        what=(
            "The mean offset of the guide stars from where the guider expected them, center minus nominal, in "
            "each AG exposure: its length r, and its x and y in hardware coordinates (µm). This is what the "
            "guider tries to keep at 0."
        ),
        data=(
            "readAgcData: agc_match's agc_center_[xy]_mm and agc_nominal_[xy]_mm of the valid matches "
            "(agc_match.flags GOOD_MATCH); agc_exposure.taken_at; shutter_open from the sps_exposure times."
        ),
        lookFor=[
            "r within a few microns (1 AG pixel is 13 µm) while the shutters are open.",
            "Spikes, or errors that persist through a visit: the guider isn't correcting.",
            "A slow trend in x or y: a drift the guider follows late.",
        ],
        draw=drawAgcErrors,
        callouts=agcErrorsCallouts,
        options=(
            "byTime (HST on the x axis), yLimit_um, reference (e.g. boresight), pfsVisitIds and agcExposureIds. "
            "ics_pfsPlotActor draws it on its own three axes (axes=)."
        ),
    ),
    PlotDoc(
        name="showAgcErrorsForVisitsByCamera",
        title="Each camera's guide errors, relative to the others",
        call="plotting.showAgcErrorsForVisitsByCamera(agcData)",
        sample="Raster scan, visits 148284-148292.",
        what=(
            "Each camera's mean offset in each AG exposure (center minus nominal), less the camera's median, "
            "as the cameras' positions aren't known well, and less the exposure's mean over the cameras, "
            "which is the telescope's pointing error. What is left is how the cameras move relative to each "
            "other."
        ),
        data=(
            "readAgcData: valid matches without bad detection flags, taken while the shutters weren't closed. "
            "With fitPfiModel, offsets are from analysis.fitGlobalModel's model instead of the nominal positions."
        ),
        lookFor=[
            "Scatter of a few microns about 0.",
            "Trends with altitude or rotator angle (plotBy): flexure.",
            "Opposite cameras with opposite signs: a rotation or scale left in (try fitPfiModel).",
            "One camera apart from the others: that camera, or its stars.",
        ],
        draw=drawByCamera,
        callouts=byCameraCallouts,
        options=(
            "plotBy (agc_exposure_id, altitude, insrot), colorBy (camera, visit, altitude, insrot), "
            "plotPerCamera, showAltInsrot (mean offsets binned by altitude and rotator angle), plotDzDfocus "
            "(against the AG actor's focus error), drawVisitBoundaries, rotateToZenith."
        ),
    ),
    PlotDoc(
        name="showAgcErrorsForVisitsByCamera-plotXY",
        title="Each camera's guide errors, y against x",
        call=(
            "plotting.showAgcErrorsForVisitsByCamera(agcData, plotXY=True, plotPerCamera=True, "
            "plotXYStride=None, showCovariance=True)"
        ),
        sample="Raster scan, visits 148284-148292; one point per visit.",
        what="The same offsets as above, y against x: each AG exposure, or with plotXYStride=None each visit.",
        data="As above.",
        lookFor=[
            "Points clustered on 0.",
            "Long ellipses: motion along one direction, such as flexure towards the zenith.",
            "Clusters away from 0 for some visits: the cameras moved between visits.",
        ],
        draw=drawByCameraXY,
        callouts=byCameraXYCallouts,
        options="connectDxDy joins the points in order; nVisitMin drops visits with few AG exposures.",
    ),
    PlotDoc(
        name="showGuiderErrors",
        title="The guide stars on the PFI",
        call=(
            "fit = analysis.fitGuiderModel(agcData, analysis.GuiderFitConfig(...))\n"
            "plotting.showGuiderErrors(fit, plotting.GuiderPlotConfig(...))"
        ),
        sample=(
            "All-sky exposure 148258 with ics_pfsPlotActor's settings: no models, guide error under 100 µm, "
            "30% of the guide stars."
        ),
        what=(
            "Each AG camera at its position on the PFI (mm), and each guide star at its offset from where the "
            "guider's models put it (µm, so magnified 1000 times) about the camera. With the models off, as in "
            "ics_pfsPlotActor, that is the offset from the guider's nominal position; with "
            "modelBoresightOffset and modelCCDOffset, what is left after an offset, rotation and scale of "
            "each exposure and of each camera."
        ),
        data=(
            "analysis.fitGuiderModel on readAgcData. The stars plotted are those the fit selected: valid "
            "matches without bad detection flags, shutters open, in AG exposures whose guide error passes "
            "maxGuideError_um."
        ),
        lookFor=[
            "Clouds centred on their +, a few tens of microns across.",
            "Clouds off their + in a pattern round the ring: tangential is a rotation, radial a scale.",
            "Colour changing across a cloud: the stars drift during the visit.",
            "A missing camera: none of its matches is valid.",
        ],
        draw=drawGuiderErrors,
        callouts=guiderErrorsCallouts,
        options=(
            "GuiderPlotConfig: showGuideStarsAsArrows, showAverageGuideStarPos and Path (each camera's mean "
            "per exposure), rotateToAG1Down (minus the rotator angle), showGuideStarPositions, guideStarFrac. "
            "colorbars= updates the colorbar when redrawing (ics_pfsPlotActor)."
        ),
    ),
    PlotDoc(
        name="showGuiderErrorsByParams",
        title="The guide stars on the PFI, coloured by other quantities",
        call='plotting.showGuiderErrorsByParams(fit, ["altitude", "azimuth", "insrot", "agc_exposure_id"])',
        sample="Raster scan, visits 148284-148292, with the boresight and camera models.",
        what=(
            "Each camera's mean offset in each AG exposure, drawn as showGuiderErrors draws them, once per "
            "quantity and coloured by it."
        ),
        data="GuiderFit.guideErrorByCamera from analysis.fitGuiderModel, and the quantities from its agcData.",
        lookFor=["Colour gradients along a cloud: offsets that follow altitude, rotator angle, time..."],
        draw=drawGuiderErrorsByParams,
        callouts=byParamsCallouts,
    ),
    PlotDoc(
        name="showTelescopeErrors",
        title="The guide corrections sent to the telescope",
        call="plotting.showTelescopeErrors(agcData)",
        sample="All-sky exposure 148258.",
        what=(
            "The corrections the AG actor sent the telescope after each AG exposure with the shutters open: "
            "altitude and azimuth (arcsec) and rotator angle."
        ),
        data="readAgcData: agc_guide_offset's guide_delta_el, guide_delta_az and guide_delta_insrot (arcsec).",
        lookFor=[
            "Corrections centred on 0, of a few tenths of an arcsec.",
            "Corrections that keep one sign, or trend in time: tracking or pointing model errors.",
            "Large rotator corrections: a rotator or position angle error.",
        ],
        draw=drawTelescopeErrors,
        callouts=telescopeCallouts,
        options="showTheta: the rotator correction in arcsec, rather than microns at the AG cameras.",
    ),
    PlotDoc(
        name="plotDriftRate",
        title="The drift of the guide stars",
        call="plotting.plotDriftRate(analysis.fitDriftRate(agcData))",
        sample="All-sky exposure 148258: -0.004 µm/min radial, 0.22 µm/min tangential.",
        what=(
            "The guide stars' offsets against time, split into radial and tangential components, with the "
            "fitted drift rates."
        ),
        data=(
            "analysis.fitDriftRate on readAgcData: valid matches with the shutters open, center minus "
            "nominal by default, averaged per camera and AG exposure, less each camera's mean."
        ),
        lookFor=[
            "Rates within a fraction of a micron per minute.",
            "A radial drift with one sign on every camera: a scale change (focus, temperature).",
            "A tangential drift: a rotation.",
        ],
        draw=drawDriftRate,
        callouts=driftCallouts,
        options="byTime=False plots against agc_exposure_id; fitDriftRate(radialTangential=False) gives x and y.",
    ),
    PlotDoc(
        name="plotGuideErrors",
        title="Each camera's guide errors on the PFI",
        call=("guideErrors = analysis.estimateGuideErrors(agcData)\nplotting.plotGuideErrors(guideErrors)"),
        sample="All-sky exposure 148258, from each star's median position (center0).",
        what=(
            "Each camera's mean offset in each AG exposure (or visit) from a reference position, drawn about "
            "the camera's position, and their mean over the cameras about the boresight. drp_stella's "
            "estimateGuideErrors(plot=True)."
        ),
        data=(
            "analysis.estimateGuideErrors on readAgcData: valid matches, center minus a reference: center0 "
            "(each star's median position over the data), nominal, boresight..."
        ),
        lookFor=[
            "Each camera's points in a tight cluster.",
            "The same pattern on every camera: the telescope moved (in a raster scan, for instance).",
            "Patterns that turn round the ring: a rotation.",
        ],
        draw=drawGuideErrors,
        callouts=guideErrorsCallouts,
        options=(
            "colorBy (agc_exposure_id, pfs_visit_id, time), drawTrack, rotateToAG1Down, expand; "
            "showClosedShutter adds the closed-shutter AG exposures, as small dots, given "
            "estimateGuideErrors(includeClosedShutter=True)."
        ),
    ),
    PlotDoc(
        name="plotPfsUtilsComparison",
        title="The guide stars' positions: the guider's and pfs_utils's",
        call=(
            "comparison = analysis.comparePfsUtilsPositions(queries.readAGCStars(opdb, designId, visit), agcData)\n"
            "plotting.plotPfsUtilsComparison(comparison)"
        ),
        sample="All-sky exposure 148258.",
        what=(
            "Where the guider expected each guide star (agc_nominal), and where pfs_utils puts it with the "
            "AG actor's model and inputs, about each camera."
        ),
        data=(
            "readAGCStars (the design's guide stars) and readAgcData (the AG actor's field center, position "
            "angle, ADC and M2 positions, and detector half); medians over the visit's first 10 AG exposures."
        ),
        lookFor=[
            "The two on top of each other: they agree to well under a micron.",
            "d(theta) near 0.",
            "Any disagreement: the AG actor and pfs_utils no longer use the same model.",
        ],
        draw=drawPfsUtils,
        callouts=pfsUtilsCallouts,
        options="alignCenterPosition (in comparePfsUtilsPositions) removes the mean offset; plotUsingScatter.",
    ),
    PlotDoc(
        name="plotFocus",
        title="Focus from the AG cameras",
        call="plotting.plotFocus(agcData, showMedian=True)",
        sample="Focus sweep, visits 148266 and 148270-148277, M2_OFF3 from 2.725 to 3.55 mm.",
        what=(
            "The glass has been removed from one half of each AG detector, so its two halves focus at "
            "different M2_OFF3, and the difference of the stars' sizes on the two halves gives the focus "
            "error. Three rows: the AG actor's focus error; the same from the stars' sizes, measured here; and "
            "each star's FWHM."
        ),
        data=(
            "readAgcData: agc_guide_offset.guide_delta_z1-6 (the AG actor's), agc_data's second moments and "
            "detection flags (RIGHT marks the half), tel_status.m2_off3. Isolated GAIA stars, valid matches. "
            "analysis.estimateFocusErrors uses the AG actor's calibration."
        ),
        lookFor=[
            "Focus errors crossing 0 at best focus; the AG actor's and ours agreeing.",
            "Cameras disagreeing about best focus: the focal plane is tilted.",
            "Each half's FWHM smallest on its own side of best focus.",
        ],
        draw=drawFocus,
        callouts=focusCallouts,
        options=(
            "plotBy (focus, agc_exposure_id, altitude, insrot), colorBy, plotPerCamera, "
            "averageByFocusPosition, showCameraId, showPfiFocusPosition. Click a panel to set M2_OFF3: the "
            "top panel then shows the focus error expected about it (ShowFocusFit)."
        ),
    ),
    PlotDoc(
        name="plotFocus-byExposure",
        title="Focus from the AG cameras, in time",
        call=(
            'plotting.plotFocus(agcData, plotBy="agc_exposure_id", showOpdbFocus=False, showFWHM=False, '
            "showFocusSets=True)"
        ),
        sample="Focus sweep, as above; ics_pfsPlotActor's FocusPlot.",
        what="The AG actor's focus error of each camera against AG exposure.",
        data="As above.",
        lookFor=[
            "A step at each change of M2_OFF3: about 600-700 µm of focus error per mm near focus in Run 30.",
            "Drifts within a run at one M2_OFF3: the focus changing (temperature, altitude).",
        ],
        draw=drawFocusByExposure,
        callouts=focusByExposureCallouts,
    ),
    PlotDoc(
        name="plotFocusByAG",
        title="Each camera's focus relative to the others",
        call="plotting.plotFocusByAG(agcData)",
        sample="Focus sweep, as above.",
        what=(
            "Each camera's focus error in each visit, less the mean of the other cameras but AG1 (as drp_stella "
            "did) and the overall mean, negated to match Kawanomoto-san's plots."
        ),
        data="analysis.estimateFocusErrors per camera, from the stars' sizes; the median of each visit.",
        lookFor=[
            "The same pattern in every visit: a fixed tilt or piston of the cameras.",
            "A pattern that changes between visits: the focal plane moving.",
        ],
        draw=drawFocusByAG,
        callouts=focusByAGCallouts,
        options="byCamera=False plots each camera against the visit; byExposureId uses each AG exposure.",
    ),
]


# Drawing the captions


def drawCallouts(plot: plotting.GuiderPlot, callouts: list[Callout]) -> None:
    """Draw a numbered marker for each callout on the sample."""
    for n, callout in enumerate(callouts, 1):
        ax = plot.axes[callout.panel]
        ax.annotate(
            str(n),
            xy=callout.xy,
            xycoords="axes fraction" if callout.fraction else "data",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "circle,pad=0.25", "facecolor": "black", "edgecolor": "white", "alpha": 0.85},
            zorder=100,
            annotation_clip=False,
        )


class TextColumn:
    """Write wrapped paragraphs down a subfigure."""

    def __init__(self, fig: SubFigure, width: int = 72, size: float = 10, lineSpacing: float = 1.3):
        self.fig, self.width, self.size, self.lineSpacing = fig, width, size, lineSpacing
        self.y = 0.97
        # A line's height, as a fraction of the subfigure, which is as tall as the figure.
        self.lineHeight = lineSpacing * size / 72 / fig.figure.get_figheight()

    def write(self, text: str, bold: bool = False, indent: str = "", firstIndent: str | None = None) -> None:
        lines = textwrap.wrap(
            text,
            self.width,
            initial_indent=indent if firstIndent is None else firstIndent,
            subsequent_indent=indent,
        )
        self.fig.text(
            0.02,
            self.y,
            "\n".join(lines),
            va="top",
            fontsize=self.size,
            fontweight="bold" if bold else "normal",
            linespacing=self.lineSpacing,
        )
        self.y -= self.lineHeight * len(lines) + 0.25 * self.lineHeight


def makeCaption(doc: PlotDoc) -> tuple[Figure, list[Callout]]:
    """Draw the sample of a plot, with its callouts and description."""
    fig = Figure(figsize=(16, 8), dpi=DPI)
    plotFig, textFig = fig.subfigures(1, 2, width_ratios=[1.4, 1])
    plot = doc.draw(plotFig)
    callouts = doc.callouts(plot)
    drawCallouts(plot, callouts)

    text = TextColumn(textFig)
    text.write(doc.title, bold=True)
    text.write(doc.what)
    text.write("Data", bold=True)
    text.write(doc.data)
    if callouts:
        text.write("Reading it", bold=True)
        for n, callout in enumerate(callouts, 1):
            text.write(callout.text, indent="     ", firstIndent=f"({n}) ")
    text.write("Look for", bold=True)
    for item in doc.lookFor:
        text.write(item, indent="   ", firstIndent="-  ")
    text.write("Sample: " + doc.sample + " 1 AG pixel = 13 µm = 0.14 arcsec.")

    return fig, callouts


def markdown(docs: list[PlotDoc], callouts: dict[str, list[Callout]]) -> str:
    """Return README.md."""
    lines = [
        "# Guider plots",
        "",
        "What each plot in `pfs.drp.qa.guiders.plotting` shows, where its data come from, and what to look for.",
        "Each has a sample drawn from the real AG data of the tests (engineering visits of Run 30, in",
        "`tests/guiders/data`), with numbered callouts explained beside it.",
        "",
        "Offsets are a star's measured center minus a reference position, in microns; positions are in hardware",
        "coordinates (`pfs.drp.qa.guiders.coordinates`). 1 AG pixel is 13 µm, or 0.14 arcsec. The plots take",
        "AG data from `queries.readAgcData`, or the results of `analysis`; see the README's",
        '"Guider tools" section for an example.',
        "",
        "This file and the images are made by `makeGuiderPlotDocs.py`, from the descriptions in it; edit those",
        "and rerun it, rather than editing this file.",
        "",
    ]
    lines += [f"- [{doc.title}](#{doc.name.lower()})" for doc in docs]
    for doc in docs:
        lines += [
            "",
            f'<a id="{doc.name.lower()}"></a>',
            f"## {doc.title}",
            "",
            "```python",
            doc.call,
            "```",
            "",
        ]
        lines += [f"![{doc.name}]({doc.name}.png)", "", doc.what, "", f"**Data.** {doc.data}", ""]
        if callouts[doc.name]:
            lines += ["**Reading it.**", ""]
            lines += [f"{n}. {callout.text}" for n, callout in enumerate(callouts[doc.name], 1)]
            lines += [""]
        lines += ["**Look for.**", ""]
        lines += [f"- {item}" for item in doc.lookFor]
        lines += [""]
        if doc.options:
            lines += [f"**Options.** {doc.options}", ""]
        lines += [f"*Sample:* {doc.sample}"]

    return "\n".join(lines) + "\n"


def main() -> None:
    callouts = {}
    for doc in PLOTS:
        fig, callouts[doc.name] = makeCaption(doc)
        path = HERE / f"{doc.name}.png"
        fig.savefig(path, dpi=DPI)
        # 256 colours are plenty for these plots, and make the files a third of the size.
        with Image.open(path) as image:
            image = image.convert("RGB").quantize(256, dither=Image.Dither.NONE)
        image.save(path, optimize=True)
        print(f"{doc.name}.png")
    (HERE / "README.md").write_text(markdown(PLOTS, callouts))
    print("README.md")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
