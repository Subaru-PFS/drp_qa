"""Plots of AG data.

Functions take DataFrames from `pfs.drp.qa.guiders.queries`, or the results
of `pfs.drp.qa.guiders.analysis`, never a database connection, and don't
modify their inputs.

Each plot draws on the ``ax`` or ``axes`` it is given, or on new panels in
``fig``, or on a new figure, and returns a `GuiderPlot`: the figure, its
panels, the artists showing the data, and its colorbars. Given the
``colorbars`` of an earlier call, a plot updates them rather than adding new
ones, so that a figure can be redrawn in place, as ics_pfsPlotActor does.
Pyplot is only used to make a new figure, never to find the current axes.

A plot that makes its panels (given no ``ax`` or ``axes``) puts its title on
the figure; one drawn on given axes puts it on its top-left panel, so that
plots sharing a figure don't overwrite each other's.

Positions are in hardware coordinates and offsets are a center minus a
reference, in microns, as `pfs.drp.qa.guiders.coordinates` says. AG cameras
are identified by ``agc_camera_id``, 0-5 (AG1 is 0), and drawn in the colours
``C0``-``C5``. 1 AG pixel is 13 µm (0.14 arcsec).
"""

import itertools
import weakref
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import matplotlib.colors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, RegularPolygon
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from pfs.drp.qa.guiders import analysis
from pfs.drp.qa.guiders.analysis import (
    FOCUS_PISTON_OFFSET_MM,
    DriftFit,
    GuiderFit,
    PfsUtilsComparison,
    addImageSizes,
    correctAgActorFocus,
    estimateFocusErrors,
    fitGlobalModel,
    selectGoodDetections,
    selectIsolatedGaiaStars,
    selectValidMatches,
)
from pfs.drp.qa.guiders.coordinates import (
    AGC_CAMERA_CENTERS_MM,
    AGC_RING_RADIUS_MM,
    addOffsets,
    arcsecToRad,
    guiderFocusToM2Off3,
    m2Off3ToGuiderFocus,
    mmToUm,
    offsetColumns,
    pfiToZenith,
    rotXY,
    umToMm,
)
from pfs.drp.qa.utils.plotting import opaqueColorbar

__all__ = [
    "FormatCoord",
    "GuiderPlot",
    "GuiderPlotConfig",
    "ShowFocusFit",
    "plotDriftRate",
    "plotFocus",
    "plotFocusByAG",
    "plotGuideErrors",
    "plotPfsUtilsComparison",
    "showAGCameraCartoon",
    "showAgcErrorsForVisits",
    "showAgcErrorsForVisitsByCamera",
    "showGuiderErrors",
    "showGuiderErrorsByParams",
    "showTelescopeErrors",
]

# The colours of the AG cameras, by agc_camera_id.
_CAMERA_COLORS = [f"C{cid}" for cid in range(6)]

# Half the width of the PFI plots of each kind (mm).
_GUIDER_ERRORS_LIMIT_MM = 350
_GUIDE_ERRORS_LIMIT_MM = 280
_PFS_UTILS_LIMIT_MM = 300

_PLOT_BY = ("agc_exposure_id", "altitude", "insrot")


def _cameraColormap() -> tuple[matplotlib.colors.Colormap, matplotlib.colors.Normalize]:
    """Return a colormap and norm giving camera number n (1-6, agc_camera_id + 1) the colour C(n-1)."""
    return matplotlib.colors.ListedColormap(_CAMERA_COLORS), matplotlib.colors.Normalize(0.5, 6.5)


_COLOR_BY = {"camera": "agc_camera_id", "visit": "pfs_visit_id", "altitude": "altitude", "insrot": "insrot"}


# Figures, panels and colorbars


@dataclass(frozen=True, eq=False)
class GuiderPlot:
    """What a plot drew.

    Attributes
    ----------
    fig : `matplotlib.figure.Figure`
        The figure.
    axes : `numpy.ndarray` of `matplotlib.axes.Axes`
        The panels, rows by columns.
    artists : `list` of `matplotlib.artist.Artist`
        The artists showing the data (lines, scatters, quivers, hexbins), in
        the order they were drawn; not reference lines, labels or legends.
    colorbars : `tuple` of `matplotlib.colorbar.Colorbar`
        The colorbars, new or updated. Pass them back as ``colorbars`` to
        update them when redrawing the figure.
    data : `pandas.DataFrame` or `None`
        The values plotted, where they aren't simply the input; each
        function says what.
    """

    fig: Figure
    axes: np.ndarray
    artists: list[Artist]
    colorbars: tuple[Colorbar, ...] = ()
    data: pd.DataFrame | None = None


def _panels(
    nRows: int,
    nCols: int,
    fig: Figure | SubFigure | None,
    axes: Axes | Iterable[Axes] | None,
    **kwargs,
) -> tuple[Figure | SubFigure, np.ndarray, bool]:
    """Return the figure, the panels (rows by columns), and whether the plot made them.

    Given ``axes``, there must be ``nRows*nCols`` of them, in any shape; they
    are taken row by row. Otherwise the panels are made in ``fig``, or in a
    new pyplot figure, with ``kwargs`` passed to `Figure.subplots`.
    """
    if axes is not None:
        flat = [axes] if isinstance(axes, Axes) else list(np.ravel(np.asarray(axes, dtype=object)))
        if len(flat) != nRows * nCols:
            raise ValueError(f"This plot needs {nRows}x{nCols} axes, not {len(flat)}")
        grid = np.empty(len(flat), dtype=object)
        grid[:] = flat
        grid = grid.reshape(nRows, nCols)
        return grid[0, 0].figure, grid, False

    if fig is None:
        fig = plt.figure()
    grid = fig.subplots(nRows, nCols, squeeze=False, **kwargs)

    return fig, grid, True


def _plainTicks(axis, rotate: bool = False) -> None:
    """Label an axis of IDs in full, e.g. 1091800 rather than 800 + 1.091e6.

    The labels are long, so there are fewer of them, rotated if ``rotate``.
    """
    formatter = axis.get_major_formatter()
    if isinstance(formatter, ScalarFormatter) and max(np.abs(axis.get_view_interval())) >= 1e5:
        formatter.set_useOffset(False)
        formatter.set_scientific(False)
        axis.set_major_locator(MaxNLocator(nbins=3 if rotate else 4, integer=True))
        if rotate:
            axis.set_tick_params(labelrotation=30)


def _finish(fig: Figure | SubFigure, axes: np.ndarray, ownPanels: bool, title: str) -> None:
    """Title the figure if the plot made its panels, else its top-left panel; label ticks in full."""
    if ownPanels:
        fig.suptitle(title)
    else:
        axes[0, 0].set_title(title, fontsize="medium")
    for ax in axes.flat:
        _plainTicks(ax.xaxis, rotate=axes.shape[1] > 1)
        _plainTicks(ax.yaxis)


def _xLabel(fig: Figure | SubFigure, axes: np.ndarray, ownPanels: bool, label: str) -> None:
    """Label the x axes: once for the figure if the plot made several columns, else each bottom panel."""
    if ownPanels and axes.shape[1] > 1:
        fig.supxlabel(label)
    else:
        for ax in axes[-1, :]:
            ax.set_xlabel(label)


def _yLabel(fig: Figure | SubFigure, axes: np.ndarray, ownPanels: bool, label: str) -> None:
    """Label the y axes: once for the figure if the plot made several rows, else each left panel."""
    if ownPanels and axes.shape[0] > 1:
        fig.supylabel(label)
    else:
        for ax in axes[:, 0]:
            ax.set_ylabel(label)


def _colorbar(
    fig: Figure | SubFigure,
    mappable,
    label: str,
    colorbars: Sequence[Colorbar] | None,
    index: int,
    **kwargs,
) -> Colorbar:
    """Add an opaque colorbar for ``mappable``, or update ``colorbars[index]``."""
    old = colorbars[index] if colorbars is not None and index < len(colorbars) else None
    with opaqueColorbar(mappable):
        if old is None:
            colorbar = fig.colorbar(mappable, **kwargs)
        else:
            colorbar = old
            colorbar.update_normal(mappable)
    colorbar.set_label(label)
    _plainTicks(colorbar.long_axis)

    return colorbar


def _cameraLegend(ax: Axes, agcCameraIds: Iterable[int], **kwargs) -> None:
    """Add a legend naming the cameras by colour."""
    handles = [
        Line2D([], [], marker="o", ls="", color=_CAMERA_COLORS[int(cid)], label=f"AG{int(cid) + 1}")
        for cid in sorted(agcCameraIds)
    ]
    if handles:
        ax.legend(handles=handles, **kwargs)


def _markerAndAlpha(
    nPoints: float, forceAlpha: float | None = None, nPointsCrit: int = 100
) -> tuple[str, float]:
    """Return the marker and alpha for a scatter of ``nPoints`` points."""
    small = nPoints < nPointsCrit
    alpha = forceAlpha if forceAlpha is not None else 0.5 if small else 0.25

    return ("o" if small else "."), alpha


def _visitRange(visits: Iterable[int]) -> str:
    """Return a short description of some visits: a list of up to four, or a range."""
    visits = sorted({int(v) for v in visits})
    if not visits:
        return ""
    if len(visits) < 5:
        return ",".join(str(v) for v in visits)

    return f"{visits[0]}..{visits[-1]}"


def _checkChoice(name: str, value: str, choices: Iterable[str]) -> None:
    choices = list(choices)
    if value not in choices:
        raise ValueError(f"Unknown {name} {value!r}; valid: {', '.join(choices)}")


# Helpers shared by the plots


class FormatCoord:
    """Cursor readout naming the visit under the cursor.

    Set as an axes' ``format_coord``. When the x axis is ``agc_exposure_id``,
    the readout adds the AG exposure's visit, and its design name if
    ``designNames`` has it.

    Parameters
    ----------
    plotBy : `str`
        The quantity on the x axis.
    agcData : `pandas.DataFrame`
        AG data with ``agc_exposure_id`` and ``pfs_visit_id``.
    designNames : `~collections.abc.Mapping` [`int`, `str`], optional
        Design names, by ``pfs_visit_id``; e.g. from
        `pfs.drp.qa.guiders.queries.readPfsDesign`.
    """

    def __init__(self, plotBy: str, agcData: pd.DataFrame, designNames: Mapping[int, str] | None = None):
        self._plotBy = plotBy
        self._visits = (
            agcData.groupby("agc_exposure_id").pfs_visit_id.first().astype(int).to_dict()
            if plotBy == "agc_exposure_id"
            else {}
        )
        self._designNames = dict(designNames or {})

    def __call__(self, x: float, y: float) -> str:
        if self._plotBy != "agc_exposure_id":
            return f"(x, y) = ({x:.2f}, {y:.2f})"

        aid = int(np.floor(x + 0.5))
        label = f"(x, y) = ({aid}, {y:.2f})"
        visit = self._visits.get(aid)
        if visit is not None:
            label += f"  pfs_visit_id: {visit}"
            if visit in self._designNames:
                label += f" ({self._designNames[visit]})"

        return label


def _showVisitBoundaries(ax: Axes, agcData: pd.DataFrame) -> None:
    """Draw a line at the last AG exposure of each visit but the last."""
    lastExposure = agcData.groupby("pfs_visit_id").agc_exposure_id.max().sort_index()
    for aid in lastExposure.iloc[:-1]:
        ax.axvline(aid, color="black", zorder=-1, alpha=0.25)


def _drawCircularArrow(
    ax: Axes,
    radius: float,
    center: tuple[float, float],
    thetas_deg: tuple[float, float],
    clockwise: bool = True,
    angle_deg: float = 0,
    **kwargs,
) -> None:
    """Draw an arc of a circle of diameter ``radius`` with an arrow head.

    Modified from https://stackoverflow.com/questions/37512502.
    """
    theta1, theta2 = thetas_deg
    arc = Arc(
        center, radius, radius, angle=angle_deg, theta1=theta1, theta2=theta2, capstyle="round", **kwargs
    )
    ax.add_patch(arc)

    kwargs.update(color=arc.get_edgecolor())
    thetaEnd = np.deg2rad((theta1 if clockwise else theta2) + angle_deg)
    end = (center[0] + (radius / 2) * np.cos(thetaEnd), center[1] + (radius / 2) * np.sin(thetaEnd))
    orientation = np.deg2rad(angle_deg + theta2 + (180 if clockwise else 0))
    ax.add_patch(RegularPolygon(end, numVertices=3, radius=radius / 9, orientation=orientation, **kwargs))


def showAGCameraCartoon(
    ax: Axes,
    showInstrot: bool = False,
    showUp: bool = False,
    lookingAtHardware: bool = True,
    insrot_deg: float | None = None,
) -> Axes:
    """Draw a cartoon of the AG cameras in the bottom left corner of a plot.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The plot.
    showInstrot : `bool`
        Show the direction in which the rotator angle increases.
    showUp : `bool`
        Mark 0 degrees and up.
    lookingAtHardware : `bool`
        Draw the cameras in hardware coordinates, rather than with y flipped
        (the opdb's frame).
    insrot_deg : `float`, optional
        Rotate the cameras by minus this rotator angle (degrees), as the plots
        do with ``rotateToAG1Down``.

    Returns
    -------
    cartoon : `matplotlib.axes.Axes`
        The inset axes holding the cartoon.
    """
    cartoon = ax.inset_axes([0.01, 0.01, 0.2, 0.2])
    cartoon.set_aspect(1)
    cartoon.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    cartoon.set_zorder(-1)
    cartoon.set_xlim(cartoon.set_ylim(-300, 300))

    for cid, (x, y) in AGC_CAMERA_CENTERS_MM.items():
        if insrot_deg is not None:
            x, y = rotXY(-np.deg2rad(insrot_deg), x, y)
        if not lookingAtHardware:
            y = -y
        cartoon.text(x, y, f"{cid + 1}", ha="center", va="center", color=_CAMERA_COLORS[cid])

    if showInstrot:
        _drawCircularArrow(cartoon, 100, (0, 10), (0, 250), clockwise=lookingAtHardware)

    if showUp:
        cartoon.text(0, 5, r"0$^\circ$", ha="center", va="center", size=7)
        for sign in (-1, 1):
            cartoon.text(-260, sign * 150, r"$\uparrow$", va="center", rotation=90)

    return cartoon


# Guide errors by AG exposure


def showAgcErrorsForVisits(
    agcData: pd.DataFrame,
    pfsVisitIds: Iterable[int] | None = None,
    agcExposureIds: Iterable[int] | None = None,
    byTime: bool = False,
    yLimit_um: float | None = None,
    reference: str = "nominal",
    showLegend: bool = True,
    fig: Figure | None = None,
    axes: Iterable[Axes] | None = None,
) -> GuiderPlot:
    """Plot the mean guide error of each AG exposure: its length, x and y.

    The guide error is the mean offset of the exposure's valid matches
    (`pfs.drp.qa.guiders.analysis.selectValidMatches`) from their
    ``reference`` positions, center minus reference; ``r`` is its length.
    Each visit has its own colour, and AG exposures taken with the
    spectrograph shutters open (``shutter_open == 1``) are drawn larger.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data in hardware coordinates, from
        `pfs.drp.qa.guiders.queries.readAgcData`.
    pfsVisitIds, agcExposureIds : iterable of `int`, optional
        Only plot these visits and AG exposures.
    byTime : `bool`
        Plot against the time (HST), rather than ``agc_exposure_id``.
    yLimit_um : `float`, optional
        Show offsets up to this (microns): -0.1 to sqrt(2) times it for ``r``.
    reference : `str`
        The reference position; one of
        `pfs.drp.qa.guiders.coordinates.REFERENCES`.
    showLegend : `bool`
        Name the visits in a legend.
    fig : `matplotlib.figure.Figure`, optional
        Draw on new panels in this figure.
    axes : iterable of `matplotlib.axes.Axes`, optional
        Draw on these three panels (r, x, y), e.g. ics_pfsPlotActor's.

    Returns
    -------
    plot : `GuiderPlot`
        ``data`` has one row per AG exposure: ``agc_exposure_id``,
        ``pfs_visit_id``, ``taken_at``, ``shutter_open`` (the max) and the
        guide error, ``dx_um``, ``dy_um`` and ``dr_um``.

    Raises
    ------
    ValueError
        If there are no valid matches to plot.

    Notes
    -----
    drp_stella plotted the nominal position minus the center, the opposite
    sign of the other plots.
    """
    data = agcData
    if pfsVisitIds is not None:
        data = data[data.pfs_visit_id.isin(list(pfsVisitIds))]
    if agcExposureIds is not None:
        data = data[data.agc_exposure_id.isin(list(agcExposureIds))]
    data = data[selectValidMatches(data)]
    if data.empty:
        raise ValueError("No valid matches to plot")

    dx, dy = offsetColumns(reference)
    data = addOffsets(data, reference)
    errors = data.groupby("agc_exposure_id", as_index=False).agg(
        pfs_visit_id=("pfs_visit_id", "min"),
        taken_at=("taken_at", "mean"),
        shutter_open=("shutter_open", "max"),
        dx_um=(dx, "mean"),
        dy_um=(dy, "mean"),
    )
    errors["dr_um"] = np.hypot(errors.dx_um, errors.dy_um)

    fig, axes, ownPanels = _panels(3, 1, fig, axes, sharex=True)
    x = errors.taken_at if byTime else errors.agc_exposure_id
    visits = np.sort(errors.pfs_visit_id.unique())
    artists = []
    for ax, column, label in zip(axes[:, 0], ["dr_um", "dx_um", "dy_um"], ["r", "x", "y"], strict=True):
        for i, visit in enumerate(visits):
            color = f"C{i % 10}"
            rows = (errors.pfs_visit_id == visit).to_numpy()
            artists += ax.plot(x[rows], errors[column][rows], ".-", color=color, label=f"{visit}")
            rows = rows & (errors.shutter_open == 1).to_numpy()
            artists += ax.plot(x[rows], errors[column][rows], "o", color=color)
        ax.axhline(0, color="black")
        ax.set_ylabel(f"{label} error (µm)")
        if yLimit_um is not None:
            scale = np.array([-0.1, np.sqrt(2)]) if label == "r" else np.array([-1, 1])
            ax.set_ylim(yLimit_um * scale)
    axes[-1, 0].set_xlabel("HST" if byTime else "agc_exposure_id")
    if showLegend:
        axes[0, 0].legend(ncol=6)

    _finish(fig, axes, ownPanels, f"pfs_visit_id {_visitRange(visits)}  (center - {reference})")

    return GuiderPlot(fig=fig, axes=axes, artists=artists, data=errors)


def _cameraOffsets(
    agcData: pd.DataFrame,
    rotateToZenith: bool,
    fitPfiModel: bool,
    fitPfiRotation: bool,
    fitPfiScale: bool,
    fitAgcOffsets: bool,
    fitAgcRotation: bool,
) -> pd.DataFrame:
    """Return the mean offset of each camera in each AG exposure, for `showAgcErrorsForVisitsByCamera`.

    The valid matches without bad detection flags, taken while the shutters
    weren't closed, are averaged. Each camera's median is then subtracted,
    as the cameras' positions aren't known well, and then each exposure's
    mean, which is the telescope's pointing error.
    """
    use = (agcData.shutter_open > 0).to_numpy() & selectGoodDetections(agcData) & selectValidMatches(agcData)
    data = agcData[use].reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid matches with the shutters open to plot")

    reference = "nominal"
    if fitPfiModel:
        data = fitGlobalModel(
            data,
            fitPfiRotation=fitPfiRotation,
            fitPfiScale=fitPfiScale,
            fitAgcOffsets=fitAgcOffsets,
            fitAgcRotation=fitAgcRotation,
        ).agcData
        reference = "model"
    data = addOffsets(data, reference)
    dx, dy = (data[column].to_numpy(dtype=float) for column in offsetColumns(reference))
    if rotateToZenith:
        dz_mm, dp_mm = pfiToZenith(umToMm(dx), umToMm(dy), data.insrot.to_numpy(dtype=float))
        dx, dy = mmToUm(dp_mm), mmToUm(dz_mm)

    cameraId = data.agc_camera_id.to_numpy(dtype=int)
    agActorFocus = np.column_stack(
        [data[f"guide_delta_z{cid + 1}"].to_numpy(dtype=float) for cid in range(6)]
    )
    data = data.assign(
        agc_camera_id=cameraId,
        dx_um=dx,
        dy_um=dy,
        focus_error_um=mmToUm(agActorFocus[np.arange(len(data)), cameraId]),
    )

    offsets = data.groupby(["agc_exposure_id", "agc_camera_id"], as_index=False).agg(
        pfs_visit_id=("pfs_visit_id", "first"),
        altitude=("altitude", "first"),
        insrot=("insrot", "first"),
        focus_error_um=("focus_error_um", "first"),
        dx_um=("dx_um", "mean"),
        dy_um=("dy_um", "mean"),
    )
    for d in ("dx_um", "dy_um"):
        offsets[d] -= offsets.groupby("agc_camera_id")[d].transform("median")
        offsets[d] -= offsets.groupby("agc_exposure_id")[d].transform("mean")

    return offsets


def _covarianceEllipse(dx: np.ndarray, dy: np.ndarray, nIter: int = 3, nClip: float = 6) -> Ellipse | None:
    """Return the 1-sigma ellipse of some offsets, from clipped second moments."""
    good = np.isfinite(dx) & np.isfinite(dy)
    dx, dy = dx[good], dy[good]
    if len(dx) < 3:
        return None
    for _ in range(nIter):
        x0, y0 = np.median(dx), np.median(dy)
        cov = np.array(
            [
                [np.mean((dx - x0) ** 2), np.mean((dx - x0) * (dy - y0))],
                [np.mean((dx - x0) * (dy - y0)), np.mean((dy - y0) ** 2)],
            ]
        )
        diff = np.stack([dx - x0, dy - y0]).T
        distance2 = np.einsum("ij,jk,ik->i", diff, np.linalg.pinv(cov), diff)
        dx, dy = dx[distance2 < nClip**2], dy[distance2 < nClip**2]

    values, vectors = np.linalg.eigh(cov)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    width, height = 2 * np.sqrt(np.maximum(values, 0))

    return Ellipse(xy=(x0, y0), width=width, height=height, angle=angle, color="red", fill=False)


def showAgcErrorsForVisitsByCamera(
    agcData: pd.DataFrame,
    agcCameraIds: Iterable[int] = range(6),
    plotBy: str = "agc_exposure_id",
    colorBy: str = "camera",
    showCamerasAsLegend: bool = True,
    showAltInsrot: bool = False,
    plotXY: bool = False,
    connectDxDy: bool = False,
    plotXYStride: int | None = 1,
    plotDzDfocus: bool = False,
    plotPerCamera: bool = False,
    rotateToZenith: bool = True,
    fitPfiModel: bool = False,
    fitPfiRotation: bool = True,
    fitPfiScale: bool = True,
    fitAgcOffsets: bool = False,
    fitAgcRotation: bool = False,
    nVisitMin: int = 0,
    showCovariance: bool = False,
    drawVisitBoundaries: bool = False,
    xLimit_um: float = 0,
    yLimit_um: float = 0,
    alpha: float = 0.5,
    scatterMarkerSize: float | None = None,
    designNames: Mapping[int, str] | None = None,
    fig: Figure | None = None,
    axes: Iterable[Axes] | None = None,
    colorbars: Sequence[Colorbar] | None = None,
) -> GuiderPlot:
    """Plot each AG camera's guide errors, relative to its own median.

    Each camera's mean offset in each AG exposure (center minus nominal, or
    minus a fitted model) has the camera's median subtracted, as the
    cameras' positions aren't known well, and then the exposure's mean,
    which is the telescope's pointing error. Only valid matches without bad
    detection flags, taken while the shutters weren't closed, are used.

    There are three kinds of plot:

    - by default, the y and x offsets against ``plotBy`` (or, with
      ``showAltInsrot``, their means by altitude and rotator angle);
    - with ``plotXY``, y against x;
    - with ``plotDzDfocus``, x and y against the AG actor's focus error of
      the camera, less its median.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data in hardware coordinates, from
        `pfs.drp.qa.guiders.queries.readAgcData`.
    agcCameraIds : iterable of `int`
        The cameras to plot (0-5; AG1 is 0).
    plotBy : `str`
        The x axis of the default plot: ``agc_exposure_id``, ``altitude`` or
        ``insrot``.
    colorBy : `str`
        Colour the points by ``camera``, ``visit``, ``altitude`` or
        ``insrot``.
    showCamerasAsLegend : `bool`
        With ``colorBy="camera"``, name the cameras in a legend rather than
        a colorbar.
    showAltInsrot : `bool`
        In the default plot, show each camera's mean offsets in bins of
        altitude and rotator angle; implies ``plotPerCamera``, as the
        cameras' offsets average to 0 in each exposure.
    plotXY : `bool`
        Plot y against x.
    connectDxDy : `bool`
        With ``plotXY``, join each camera's points in order.
    plotXYStride : `int` or `None`
        With ``plotXY``, plot every ``plotXYStride``'th AG exposure; `None`
        for the mean of each visit.
    plotDzDfocus : `bool`
        Plot the offsets against the focus error; ignored with ``plotXY``.
    plotPerCamera : `bool`
        Give each camera its own panels. With ``plotXY`` the panels are in
        rows of three.
    rotateToZenith : `bool`
        Convert the offsets to the zenith frame
        (`pfs.drp.qa.guiders.coordinates.pfiToZenith`): y points to the
        zenith, x to the Opt side.
    fitPfiModel : `bool`
        Measure offsets from a model fitted to each AG exposure
        (`pfs.drp.qa.guiders.analysis.fitGlobalModel`) rather than from the
        nominal positions.
    fitPfiRotation, fitPfiScale, fitAgcOffsets, fitAgcRotation : `bool`
        The terms of that model; see
        `pfs.drp.qa.guiders.analysis.fitGlobalModel`.
    nVisitMin : `int`
        With ``plotXY`` and ``plotXYStride=None``, only plot the visits with
        more than this many AG exposures of a camera.
    showCovariance : `bool`
        With ``plotXY``, draw each camera's 1-sigma ellipse.
    drawVisitBoundaries : `bool`
        With ``plotBy="agc_exposure_id"``, mark the last AG exposure of
        each visit.
    xLimit_um, yLimit_um : `float`
        Limits of the focus error and of the offsets (microns); none if 0.
    alpha : `float`
        Alpha of the points.
    scatterMarkerSize : `float`, optional
        Size of the points.
    designNames : `~collections.abc.Mapping` [`int`, `str`], optional
        Design names by visit, for the cursor readout; see `FormatCoord`.
    fig : `matplotlib.figure.Figure`, optional
        Draw on new panels in this figure.
    axes : iterable of `matplotlib.axes.Axes`, optional
        Draw on these panels: rows (y and x, or x and y with
        ``plotDzDfocus``) by cameras (one unless ``plotPerCamera``); with
        ``plotXY``, one, or rows of three cameras.
    colorbars : sequence of `matplotlib.colorbar.Colorbar`, optional
        Update this colorbar rather than adding one.

    Returns
    -------
    plot : `GuiderPlot`
        ``data`` has the offsets plotted, one row per AG exposure and camera
        (or visit and camera): ``dx_um`` and ``dy_um``, and
        ``focus_error_um``, the AG actor's focus error of the camera.
    """
    _checkChoice("plotBy", plotBy, _PLOT_BY)
    _checkChoice("colorBy", colorBy, _COLOR_BY)
    colorColumn = _COLOR_BY[colorBy]
    agcCameraIds = [int(cid) for cid in agcCameraIds]
    if showAltInsrot and not (plotXY or plotDzDfocus):
        plotPerCamera = True

    offsets = _cameraOffsets(
        agcData, rotateToZenith, fitPfiModel, fitPfiRotation, fitPfiScale, fitAgcOffsets, fitAgcRotation
    )
    offsets = offsets[offsets.agc_camera_id.isin(agcCameraIds)].reset_index(drop=True)
    if plotXY and plotXYStride is None:
        offsets = offsets.groupby(["pfs_visit_id", "agc_camera_id"], as_index=False).agg(
            agc_exposure_id=("agc_exposure_id", "mean"),
            altitude=("altitude", "first"),
            insrot=("insrot", "first"),
            focus_error_um=("focus_error_um", "mean"),
            dx_um=("dx_um", "mean"),
            dy_um=("dy_um", "mean"),
            nAgcExposures=("agc_exposure_id", "count"),
        )
        offsets = offsets[offsets.nAgcExposures > nVisitMin].reset_index(drop=True)

    if colorBy == "camera":
        colors, vmin, vmax = offsets.agc_camera_id + 1, None, None
        cmap, norm = _cameraColormap()
    else:
        colors, cmap, norm = offsets[colorColumn], None, None
        vmin, vmax = colors.min(), colors.max()
    scatterKwargs = {
        "cmap": cmap,
        "norm": norm,
        "vmin": vmin,
        "vmax": vmax,
        "alpha": alpha,
        "s": scatterMarkerSize,
    }

    if plotXY:
        nCameras = len(agcCameraIds) if plotPerCamera else 1
        nCols = min(3, nCameras)
        nRows = -(-nCameras // nCols)
        fig, axes, ownPanels = _panels(nRows, nCols, fig, axes, sharex=True, sharey=True)
    else:
        components = ["x", "y"] if plotDzDfocus else ["y", "x"]
        nCols = len(agcCameraIds) if plotPerCamera else 1
        fig, axes, ownPanels = _panels(len(components), nCols, fig, axes, sharex=True, sharey=True)
    if ownPanels:
        fig.subplots_adjust(hspace=0.025, wspace=0.025)

    artists = []
    mappable = None
    colorbarLabel = colorBy
    if plotXY:
        stride = 1 if plotXYStride is None else plotXYStride
        for j, cid in enumerate(agcCameraIds):
            ax = axes.flat[j] if plotPerCamera else axes[0, 0]
            sel = (offsets.agc_camera_id == cid).to_numpy()
            dx, dy, c = offsets.dx_um[sel], offsets.dy_um[sel], colors[sel]
            if stride > 1:
                dx, dy, c = dx[::stride], dy[::stride], c[::stride]
            mappable = ax.scatter(dx, dy, c=c, **scatterKwargs)
            artists.append(mappable)
            if connectDxDy:
                artists += ax.plot(dx, dy, color=_CAMERA_COLORS[cid], alpha=alpha)
            if showCovariance:
                ellipse = _covarianceEllipse(dx.to_numpy(), dy.to_numpy())
                if ellipse is not None:
                    ax.add_patch(ellipse)
            if plotPerCamera:
                ax.text(0.03, 0.9, f"AG{cid + 1}", transform=ax.transAxes, color=_CAMERA_COLORS[cid])
        for ax in axes.flat[nCameras:]:
            ax.set_visible(False)
        for ax in axes.flat[:nCameras]:
            if yLimit_um > 0:
                ax.set_xlim(ax.set_ylim(-yLimit_um, yLimit_um))
            ax.set_aspect(1)
            ax.axhline(0, color="black", alpha=0.5)
            ax.axvline(0, color="black", alpha=0.5)
        _xLabel(fig, axes, ownPanels, "mean x offset (µm)")
        _yLabel(fig, axes, ownPanels, "mean y offset (µm)")
    else:
        panelCameras = [[cid] for cid in agcCameraIds] if plotPerCamera else [agcCameraIds]
        # Each camera's focus error relative to its median.
        dFocus = offsets.focus_error_um - offsets.groupby("agc_camera_id").focus_error_um.transform("median")
        for i, z in enumerate(components):
            for j, cameras in enumerate(panelCameras):
                ax = axes[i, j]
                sel = offsets.agc_camera_id.isin(cameras).to_numpy()
                if plotDzDfocus:
                    mappable = ax.scatter(
                        dFocus[sel], offsets[f"d{z}_um"][sel], c=colors[sel], **scatterKwargs
                    )
                elif showAltInsrot:
                    vlim = (-yLimit_um, yLimit_um) if yLimit_um > 0 else (None, None)
                    mappable = ax.hexbin(
                        offsets.altitude[sel],
                        offsets.insrot[sel],
                        C=offsets[f"d{z}_um"][sel],
                        vmin=vlim[0],
                        vmax=vlim[1],
                    )
                    colorbarLabel = "mean offset (µm)"
                    ax.text(0.03, 0.9, f"{z} offset", transform=ax.transAxes)
                else:
                    mappable = ax.scatter(
                        offsets[plotBy][sel], offsets[f"d{z}_um"][sel], c=colors[sel], **scatterKwargs
                    )
                    ax.axhline(0, color="black", alpha=0.5, zorder=10)
                artists.append(mappable)
                if plotPerCamera and i == 0:
                    cid = cameras[0]
                    ax.text(0.1, 1.025, f"AG{cid + 1}", transform=ax.transAxes, color=_CAMERA_COLORS[cid])
                if j == 0:
                    ax.set_ylabel("insrot" if showAltInsrot else f"{z} offset (µm)")
            if plotDzDfocus and xLimit_um > 0:
                axes[i, 0].set_xlim(-xLimit_um, xLimit_um)
            if yLimit_um > 0 and not showAltInsrot:
                axes[i, 0].set_ylim(-yLimit_um, yLimit_um)
        if showAltInsrot and yLimit_um <= 0:
            # One colour scale for every panel, as they share the colorbar.
            values = np.concatenate([np.ma.filled(hexbin.get_array(), np.nan) for hexbin in artists])
            limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
            for hexbin in artists:
                hexbin.set_clim(-limit, limit)
        xlabel = r"$\Delta$ focus (µm)" if plotDzDfocus else "altitude" if showAltInsrot else plotBy
        _xLabel(fig, axes, ownPanels, xlabel)

        if plotBy == "agc_exposure_id" and not (plotDzDfocus or showAltInsrot):
            for ax in axes.flat:
                if drawVisitBoundaries:
                    _showVisitBoundaries(ax, offsets)
                ax.format_coord = FormatCoord(plotBy, offsets, designNames)

    newColorbars = ()
    if mappable is not None:
        if colorBy == "camera" and showCamerasAsLegend and not showAltInsrot:
            if not plotPerCamera:
                _cameraLegend(axes[0, 0], offsets.agc_camera_id.unique(), ncol=6)
        else:
            colorbar = _colorbar(
                fig,
                mappable,
                colorbarLabel,
                colorbars,
                0,
                ax=list(axes.flat),
                orientation="vertical" if axes.shape[0] > 1 else "horizontal",
            )
            if colorBy == "camera" and not showAltInsrot:
                colorbar.set_ticks(range(1, 7), labels=[f"AG{cid + 1}" for cid in range(6)])
            newColorbars = (colorbar,)

    title = f"pfs_visit_id {_visitRange(offsets.pfs_visit_id)}"
    if rotateToZenith:
        title += "  zenith is up"
    if fitPfiModel:
        title += (
            "\nFit PFI offsets" + (" rotation" if fitPfiRotation else "") + (" scale" if fitPfiScale else "")
        )
        if fitAgcOffsets:
            title += " Fit AGC offsets" + (" rotations" if fitAgcRotation else "")
    title += "\n" + ", ".join(f"AG{cid + 1}" for cid in sorted(offsets.agc_camera_id.unique()))
    _finish(fig, axes, ownPanels, title)

    return GuiderPlot(fig=fig, axes=axes, artists=artists, colorbars=newColorbars, data=offsets)


# The guider model's residuals on the PFI


@dataclass(frozen=True)
class GuiderPlotConfig:
    """How `showGuiderErrors` and `showGuiderErrorsByParams` draw a fit.

    drp_stella's ``GuiderConfig`` held these options together with those of
    the fit, which are now `pfs.drp.qa.guiders.analysis.GuiderFitConfig`.
    Its ``guideErrorEstimate``, ``guide_star_frac``, ``agc_exposure_cm``,
    ``showByVisitSize`` and ``showByVisitAlpha`` are renamed here.

    The plots put each AG camera at its position on the PFI (mm) and each
    star at its offset from where the models put it (microns) from there.

    Attributes
    ----------
    showGuideStars : `bool`
        Plot the selected guide stars.
    showGuideStarsAsPoints : `bool`
        As points; this takes precedence over ``showGuideStarsAsArrows``.
    showGuideStarsAsArrows : `bool`
        As arrows from where the models put them.
    showGuideStarPositions : `bool`
        Start each star at its own position on the camera, ``gstarExpansion``
        times farther from the camera's mean position, rather than at the
        camera's mean position.
    showAverageGuideStarPos : `bool`
        Plot each camera's mean offset in each AG exposure
        (`pfs.drp.qa.guiders.analysis.GuiderFit.guideErrorByCamera`).
    showAverageGuideStarPath : `bool`
        Join those means in order.
    showByVisit : `bool`
        Colour the points by ``agc_exposure_id``, rather than by camera.
    rotateToAG1Down : `bool`
        Rotate each AG exposure by minus its rotator angle.
    guideErrorEstimate_um : `float`
        The length of the arrows' key (microns).
    pfiScaleReduction : `float`
        Bring the cameras this many times closer to the boresight.
    gstarExpansion : `float`
        See ``showGuideStarPositions``.
    guideStarFrac : `float`
        Plot this fraction of each camera's guide stars, chosen at random
        (but the same each time).
    colormap : `str`
        The colormap for ``agc_exposure_id``.
    markerSize : `float`
        The size of the points.
    alpha : `float`
        The alpha of the points.

    Raises
    ------
    ValueError
        If ``guideStarFrac`` isn't in (0, 1].
    """

    showGuideStars: bool = True
    showGuideStarsAsPoints: bool = True
    showGuideStarsAsArrows: bool = False
    showGuideStarPositions: bool = False
    showAverageGuideStarPos: bool = False
    showAverageGuideStarPath: bool = False
    showByVisit: bool = True
    rotateToAG1Down: bool = False
    guideErrorEstimate_um: float = 50
    pfiScaleReduction: float = 1
    gstarExpansion: float = 10
    guideStarFrac: float = 0.1
    colormap: str = "viridis"
    markerSize: float = 5
    alpha: float = 1

    def __post_init__(self):
        if not 0 < self.guideStarFrac <= 1:
            raise ValueError(f"guideStarFrac must be in (0, 1], not {self.guideStarFrac}")


# The seed for GuiderPlotConfig.guideStarFrac's choice of stars.
_GUIDE_STAR_SEED = 666


def _guiderErrorsTitle(data: pd.DataFrame, fit: GuiderFit, config: GuiderPlotConfig, name: str | None) -> str:
    """Return the title of `showGuiderErrors` and `showGuiderErrorsByParams`."""
    fitConfig = fit.config
    visits = data.pfs_visit_id
    v0, v1 = (int(visits.min()), int(visits.max())) if len(visits) else (0, 0)
    if fitConfig.pfsVisitIdMin > 0:
        v0 = max(v0, fitConfig.pfsVisitIdMin)
    if fitConfig.pfsVisitIdMax > 0:
        v1 = min(v1, fitConfig.pfsVisitIdMax)

    line = f"{name}  " if name else ""
    line += f"{v0}" if v0 == v1 else f"{v0}..{v1}"
    if config.rotateToAG1Down:
        line += "  Rotated"
        selected = data[data.selected]
        if len(selected) and np.std(selected.insrot) < 5:
            line += f" {-selected.insrot.mean():.1f}" + r"$^\circ$"
            line += f"  alt,az = {selected.altitude.mean():.1f},{selected.azimuth.mean():.1f}"
    lines = [line]
    removed = [
        what
        for what, on in [("Boresight", fitConfig.modelBoresightOffset), ("AGs", fitConfig.modelCCDOffset)]
        if on
    ]
    if removed:
        lines.append(f"{' and '.join(removed)} offset and rotation/scale removed")

    cuts = []
    if fitConfig.maxGuideError_um > 0:
        cuts.append(f"Max guide error: {fitConfig.maxGuideError_um} µm")
    if fitConfig.maxPosError_um > 0:
        cuts.append(f"Max per-star positional error: {fitConfig.maxPosError_um} µm")
    if cuts:
        lines.append("  ".join(cuts))

    stride = fitConfig.agcExposureStride
    line = (
        "Every AG exposure"
        if stride == 1
        else f"Every {stride}{ {2: 'nd', 3: 'rd'}.get(stride, 'th') } AG exposure"
    )
    if config.showGuideStars:
        line += f"; {100 * config.guideStarFrac:.0f}% of guide stars"
    if not fitConfig.onlyShutterOpen:
        line += " (including closed shutter)"
    if (config.showAverageGuideStarPos or config.showAverageGuideStarPath) and not (
        config.showGuideStarsAsArrows or config.showGuideStarsAsPoints
    ):
        line += "  Mean guide error"
    lines.append(line)

    return "\n".join(lines)


def _plotPositions(
    data: pd.DataFrame, x: str, y: str, dx: str, dy: str, config: GuiderPlotConfig
) -> tuple[np.ndarray, ...]:
    """Return positions (mm, reduced) and offsets (microns), rotated if ``config.rotateToAG1Down``."""
    xPos = data[x].to_numpy(dtype=float) / config.pfiScaleReduction
    yPos = data[y].to_numpy(dtype=float) / config.pfiScaleReduction
    xOff, yOff = data[dx].to_numpy(dtype=float), data[dy].to_numpy(dtype=float)
    if config.rotateToAG1Down:
        angle = -np.deg2rad(data.insrot.to_numpy(dtype=float))
        xPos, yPos = rotXY(angle, xPos, yPos)
        xOff, yOff = rotXY(angle, xOff, yOff)

    return xPos, yPos, xOff, yOff


def _cameraMeans(fit: GuiderFit, config: GuiderPlotConfig) -> pd.DataFrame:
    """Return `GuiderFit.guideErrorByCamera` with plot positions ``xPlot`` and ``yPlot``."""
    means = fit.guideErrorByCamera.copy()
    insrot = fit.agcData.groupby("agc_exposure_id").insrot.mean()
    means["insrot"] = means.agc_exposure_id.map(insrot)
    xPos, yPos, xOff, yOff = _plotPositions(
        means, "agc_model_x_mm", "agc_model_y_mm", "dx_model_um", "dy_model_um", config
    )
    means["xPlot"] = np.nan
    means["yPlot"] = np.nan
    for cid in means.agc_camera_id.unique():
        rows = (means.agc_camera_id == cid).to_numpy()
        means.loc[rows, "xPlot"] = np.mean(xPos[rows]) + xOff[rows]
        means.loc[rows, "yPlot"] = np.mean(yPos[rows]) + yOff[rows]

    return means


def _pfiPanel(ax: Axes, config: GuiderPlotConfig) -> None:
    """Draw the boresight, set the limits and the aspect ratio of a PFI plot."""
    if config.rotateToAG1Down:
        ax.add_patch(Circle((0, 0), AGC_RING_RADIUS_MM / config.pfiScaleReduction, fill=False, color="red"))
    ax.plot([0], [0], "+", color="red")
    limit = _GUIDER_ERRORS_LIMIT_MM / config.pfiScaleReduction
    ax.set_xlim(ax.set_ylim(-limit, limit))
    ax.set_aspect(1)


def showGuiderErrors(
    fit: GuiderFit,
    config: GuiderPlotConfig | None = None,
    agcCameraIds: Iterable[int] = range(6),
    name: str | None = None,
    fig: Figure | None = None,
    ax: Axes | None = None,
    colorbars: Sequence[Colorbar] | None = None,
) -> GuiderPlot:
    """Plot the guide stars' offsets from where the guider models put them.

    Each AG camera is drawn at its position on the PFI (mm, ``+``), and each
    selected star at its offset from where the models put it (microns) from
    there: ``dx_model_um`` and ``dy_model_um`` of
    `pfs.drp.qa.guiders.analysis.fitGuiderModel`. The stars are those the
    fit selected (``selected``: valid matches passing the shutter, guide
    error, stride and visit cuts) without bad detection flags.

    Parameters
    ----------
    fit : `pfs.drp.qa.guiders.analysis.GuiderFit`
        The fit, from `pfs.drp.qa.guiders.analysis.fitGuiderModel`.
    config : `GuiderPlotConfig`, optional
        What to draw; default ``GuiderPlotConfig()``.
    agcCameraIds : iterable of `int`
        The cameras to plot (0-5; AG1 is 0).
    name : `str`, optional
        Start the title with this.
    fig : `matplotlib.figure.Figure`, optional
        Draw in a new panel in this figure.
    ax : `matplotlib.axes.Axes`, optional
        Draw in this panel, e.g. ics_pfsPlotActor's.
    colorbars : sequence of `matplotlib.colorbar.Colorbar`, optional
        Update this colorbar (of ``agc_exposure_id``) rather than adding one.

    Returns
    -------
    plot : `GuiderPlot`
        ``data`` has the stars plotted, with their plot positions ``xPlot``
        and ``yPlot``.

    Notes
    -----
    Compared with drp_stella: the stars on the half of the detectors flagged
    RIGHT are plotted; the cameras' positions and their ring are scaled by
    ``pfiScaleReduction`` like the stars'; and the arrows' legend works with
    matplotlib 3.9 and later.
    """
    config = GuiderPlotConfig() if config is None else config
    fig, axes, ownPanels = _panels(1, 1, fig, ax)
    ax = axes[0, 0]

    data = fit.agcData.reset_index(drop=True)
    xPos, yPos, xOff, yOff = _plotPositions(
        data, "agc_model_x_mm", "agc_model_y_mm", "dx_model_um", "dy_model_um", config
    )
    data = data.assign(xPos=xPos, yPos=yPos, xOff=xOff, yOff=yOff)
    use = data.selected.to_numpy(dtype=bool) & selectGoodDetections(data)
    vmin, vmax = data.agc_exposure_id.min(), data.agc_exposure_id.max()
    means = (
        _cameraMeans(fit, config)
        if config.showAverageGuideStarPos or config.showAverageGuideStarPath
        else None
    )

    artists = []
    plotted = []
    scatter = quiver = None
    labelled = False
    crossColor = (
        "red"
        if config.showByVisit
        or (
            config.showAverageGuideStarPos
            and not (config.showGuideStarsAsArrows or config.showGuideStarsAsPoints)
        )
        else "black"
    )
    for cid in agcCameraIds:
        color = _CAMERA_COLORS[cid]
        stars = data[use & (data.agc_camera_id == cid).to_numpy()]
        if stars.empty:
            continue

        guideStars = np.sort(stars.guide_star_id.unique())
        nUsed = max(1, int(config.guideStarFrac * len(guideStars)))
        used = np.random.default_rng(_GUIDE_STAR_SEED).choice(guideStars, nUsed, replace=False)
        stars = stars[stars.guide_star_id.isin(used)].sort_values(["guide_star_id", "agc_exposure_id"])

        if config.rotateToAG1Down:
            means0 = stars.groupby("agc_exposure_id")[["xPos", "yPos"]].transform("mean")
            radius = np.hypot(means0.xPos, means0.yPos)
            ring = AGC_RING_RADIUS_MM / config.pfiScaleReduction
            xbar = (means0.xPos * ring / radius).to_numpy()
            ybar = (means0.yPos * ring / radius).to_numpy()
        else:
            xbar = np.full(len(stars), stars.xPos.mean())
            ybar = np.full(len(stars), stars.yPos.mean())
            ax.plot(xbar[:1], ybar[:1], "+", color=crossColor, zorder=10)

        if config.showGuideStarPositions:
            xStart = xbar + (stars.xPos.to_numpy() - xbar) * config.gstarExpansion
            yStart = ybar + (stars.yPos.to_numpy() - ybar) * config.gstarExpansion
        else:
            xStart, yStart = xbar, ybar
        xEnd, yEnd = xStart + stars.xOff.to_numpy(), yStart + stars.yOff.to_numpy()

        if config.showGuideStars:
            if config.showGuideStarsAsPoints:
                if config.showByVisit:
                    scatter = ax.scatter(
                        xEnd,
                        yEnd,
                        s=config.markerSize,
                        alpha=config.alpha,
                        vmin=vmin,
                        vmax=vmax,
                        c=stars.agc_exposure_id,
                        cmap=config.colormap,
                    )
                    artists.append(scatter)
                else:
                    artists += ax.plot(
                        xEnd,
                        yEnd,
                        ".",
                        color=color,
                        label=f"AG{cid + 1}",
                        alpha=config.alpha,
                        markersize=config.markerSize,
                    )
                    labelled = True
            elif config.showGuideStarsAsArrows:
                quiver = ax.quiver(
                    xStart, yStart, xEnd - xStart, yEnd - yStart, alpha=0.5, color=color, label=f"AG{cid + 1}"
                )
                artists.append(quiver)
                labelled = True
            plotted.append(stars.assign(xPlot=xEnd, yPlot=yEnd))

        if means is not None:
            camera = means[means.agc_camera_id == cid]
            if config.showAverageGuideStarPath and len(camera):
                artists += ax.plot(
                    camera.xPlot.iloc[:1], camera.yPlot.iloc[:1], ".", color="black", zorder=-10
                )
                artists += ax.plot(camera.xPlot, camera.yPlot, "-", color="black", alpha=0.25, zorder=-10)
            if config.showAverageGuideStarPos:
                scatter = ax.scatter(
                    camera.xPlot,
                    camera.yPlot,
                    s=config.markerSize,
                    alpha=config.alpha,
                    vmin=vmin,
                    vmax=vmax,
                    c=camera.agc_exposure_id,
                    cmap=config.colormap,
                )
                artists.append(scatter)

    newColorbars = ()
    if scatter is not None:
        newColorbars = (_colorbar(fig, scatter, "agc_exposure_id", colorbars, 0, ax=ax),)
    if quiver is not None:
        ax.quiverkey(
            quiver,
            0.1,
            0.9,
            config.guideErrorEstimate_um,
            f"{config.guideErrorEstimate_um} µm",
            color="black",
        )

    if not config.showGuideStarsAsArrows:
        showAGCameraCartoon(ax, showInstrot=True, showUp=config.rotateToAG1Down)
    elif labelled:
        legend = ax.legend(loc="lower right", markerscale=1)
        for handle in legend.legend_handles:
            handle.set_alpha(1)

    _pfiPanel(ax, config)
    if quiver is not None:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    else:
        ax.set_xlabel(r"$\delta$x (µm)")
        ax.set_ylabel(r"$\delta$y (µm)")

    _finish(fig, axes, ownPanels, _guiderErrorsTitle(data, fit, config, name))

    return GuiderPlot(
        fig=fig,
        axes=axes,
        artists=artists,
        colorbars=newColorbars,
        data=pd.concat(plotted, ignore_index=True) if plotted else None,
    )


def showGuiderErrorsByParams(
    fit: GuiderFit,
    params: Sequence[str],
    config: GuiderPlotConfig | None = None,
    name: str | None = None,
    fig: Figure | None = None,
    axes: Iterable[Axes] | None = None,
    colorbars: Sequence[Colorbar] | None = None,
) -> GuiderPlot:
    """Plot each camera's mean offset in each AG exposure, coloured by each of some quantities.

    The means are `pfs.drp.qa.guiders.analysis.GuiderFit.guideErrorByCamera`,
    drawn as `showGuiderErrors` draws them with ``showAverageGuideStarPos``,
    one panel per quantity.

    Parameters
    ----------
    fit : `pfs.drp.qa.guiders.analysis.GuiderFit`
        The fit, from `pfs.drp.qa.guiders.analysis.fitGuiderModel`.
    params : sequence of `str`
        Columns of ``fit.agcData`` to colour by (their first value in each
        AG exposure), e.g. ``["altitude", "insrot"]``.
    config : `GuiderPlotConfig`, optional
        How to draw; only its plotting options matter.
    name : `str`, optional
        Start the title with this.
    fig : `matplotlib.figure.Figure`, optional
        Draw on new panels in this figure.
    axes : iterable of `matplotlib.axes.Axes`, optional
        Draw on these panels; there are about sqrt(len(params)) rows of
        them.
    colorbars : sequence of `matplotlib.colorbar.Colorbar`, optional
        Update these colorbars, one per quantity, rather than adding them.

    Returns
    -------
    plot : `GuiderPlot`
        ``data`` has the means, with the quantities and the plot positions
        ``xPlot`` and ``yPlot``.
    """
    config = GuiderPlotConfig() if config is None else config
    params = list(params)
    nRows = max(1, int(np.sqrt(len(params))))
    nCols = len(params) // nRows
    if nRows * nCols < len(params):
        nRows += 1
    fig, axes, ownPanels = _panels(nRows, nCols, fig, axes, sharex=True, sharey=True)

    means = _cameraMeans(fit, config)
    values = fit.agcData.groupby("agc_exposure_id")[params].first()
    means = means.join(values, on="agc_exposure_id", rsuffix="_param")

    artists = []
    newColorbars = []
    for i, (ax, param) in enumerate(zip(axes.flat, params, strict=False)):
        column = param if param in means and f"{param}_param" not in means else f"{param}_param"
        vmin, vmax = np.nanpercentile(means[column], [1, 99])
        scatter = ax.scatter(
            means.xPlot,
            means.yPlot,
            s=config.markerSize,
            alpha=config.alpha,
            vmin=vmin,
            vmax=vmax,
            c=means[column],
            cmap=config.colormap,
        )
        artists.append(scatter)
        newColorbars.append(
            _colorbar(fig, scatter, param, colorbars, i, ax=ax, shrink=1 / nCols if nRows == 1 else 1)
        )
        _pfiPanel(ax, config)
        if not config.rotateToAG1Down:
            showAGCameraCartoon(ax, showInstrot=True)
    for ax in axes.flat[len(params) :]:
        ax.set_visible(False)
    _xLabel(fig, axes, ownPanels, r"$\delta$x (µm)")
    _yLabel(fig, axes, ownPanels, r"$\delta$y (µm)")

    _finish(fig, axes, ownPanels, _guiderErrorsTitle(fit.agcData, fit, config, name))

    return GuiderPlot(fig=fig, axes=axes, artists=artists, colorbars=tuple(newColorbars), data=means)


# Telescope guide offsets


def showTelescopeErrors(
    agcData: pd.DataFrame,
    showTheta: bool = False,
    fig: Figure | None = None,
    axes: Iterable[Axes] | None = None,
    colorbars: Sequence[Colorbar] | None = None,
) -> GuiderPlot:
    """Plot the guide offsets the AG actor sent the telescope.

    Four panels, of the mean ``guide_delta_altitude``,
    ``guide_delta_azimuth`` and ``guide_delta_insrot`` of each AG exposure
    taken while the shutters weren't closed: altitude against azimuth
    offsets, binned and coloured by ``agc_exposure_id``; the rotator offset
    against ``agc_exposure_id``, as an angle with ``showTheta`` or as the
    motion it causes at the AG cameras (microns); and the rotator offset by
    altitude and azimuth.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data, from `pfs.drp.qa.guiders.queries.readAgcData`.
    showTheta : `bool`
        Plot the rotator offset in arcsec rather than microns at the AG
        cameras.
    fig : `matplotlib.figure.Figure`, optional
        Draw on new panels in this figure.
    axes : iterable of `matplotlib.axes.Axes`, optional
        Draw on these four panels, two by two.
    colorbars : sequence of `matplotlib.colorbar.Colorbar`, optional
        Update these four colorbars rather than adding them.

    Returns
    -------
    plot : `GuiderPlot`
        ``data`` has one row per AG exposure plotted.
    """
    offsets = agcData.groupby("agc_exposure_id", as_index=False).agg(
        altitude=("altitude", "mean"),
        azimuth=("azimuth", "mean"),
        shutter_open=("shutter_open", "max"),
        dalt_arcsec=("guide_delta_altitude", "mean"),
        daz_arcsec=("guide_delta_azimuth", "mean"),
        theta_arcsec=("guide_delta_insrot", "mean"),
    )
    offsets = offsets[offsets.shutter_open > 0].reset_index(drop=True)
    if offsets.empty:
        raise ValueError("No AG exposures with the shutters open to plot")
    offsets["theta_um"] = mmToUm(AGC_RING_RADIUS_MM) * arcsecToRad(offsets.theta_arcsec)

    fig, axes, ownPanels = _panels(2, 2, fig, axes)
    if ownPanels:
        fig.subplots_adjust(wspace=0.5, hspace=0.35)
        axes[0, 0].sharex(axes[0, 1])
        axes[0, 0].sharey(axes[0, 1])
    thetaScale = 30  # Plot limit for theta (arcsec, or microns at the AG cameras)

    artists = []
    newColorbars = []
    ax = axes[0, 0]
    hexbin = ax.hexbin(
        offsets.dalt_arcsec, offsets.daz_arcsec, gridsize=max(1, min(10, int(np.sqrt(len(offsets)))))
    )
    artists.append(hexbin)
    newColorbars.append(_colorbar(fig, hexbin, "N", colorbars, 0, ax=ax))

    ax = axes[0, 1]
    scatter = ax.scatter(offsets.dalt_arcsec, offsets.daz_arcsec, s=10, c=offsets.agc_exposure_id)
    artists.append(scatter)
    newColorbars.append(_colorbar(fig, scatter, "agc_exposure_id", colorbars, 1, ax=ax))
    ax.set_facecolor("black")
    for ax in axes[0, :]:
        ax.set_aspect(1)
        ax.set_xlabel(r"$\delta$alt (arcsec)")
        ax.set_ylabel(r"$\delta$az (arcsec)")

    ax = axes[1, 0]
    if showTheta:
        y, ylabel = offsets.theta_arcsec, r"$\theta$ (arcsec)"
    else:
        y, ylabel = offsets.theta_um, f"guide error at {AGC_RING_RADIUS_MM / 10:.2f} cm (µm)"
    scatter = ax.scatter(offsets.agc_exposure_id, y, c=offsets.altitude)
    artists.append(scatter)
    newColorbars.append(_colorbar(fig, scatter, "altitude", colorbars, 2, ax=ax))
    ax.set_ylim(-thetaScale, thetaScale)
    ax.set_xlabel("agc_exposure_id")
    ax.set_ylabel(ylabel)

    ax = axes[1, 1]
    scatter = ax.scatter(
        offsets.azimuth,
        offsets.altitude,
        c=offsets.theta_arcsec,
        s=5,
        vmin=-thetaScale / 2,
        vmax=thetaScale / 2,
    )
    artists.append(scatter)
    newColorbars.append(_colorbar(fig, scatter, r"$\theta$ (arcsec)", colorbars, 3, ax=ax))
    ax.set_xlabel("azimuth")
    ax.set_ylabel("altitude")

    title = (
        rf"$\langle\delta$(alt, az)$\rangle$ = ({offsets.dalt_arcsec.mean():.1f}, "
        f"{offsets.daz_arcsec.mean():.2f}) arcsec"
        rf"   $\langle\theta\rangle$ = {offsets.theta_arcsec.mean():.1f} arcsec"
    )
    _finish(fig, axes, ownPanels, title)

    return GuiderPlot(fig=fig, axes=axes, artists=artists, colorbars=tuple(newColorbars), data=offsets)


# Drift


def plotDriftRate(
    fit: DriftFit,
    byTime: bool = True,
    showCamera: bool = True,
    fitTrend: bool = True,
    name: str | None = None,
    fig: Figure | None = None,
    axes: Iterable[Axes] | None = None,
) -> GuiderPlot:
    """Plot the drift of the guide stars, and the fitted rates.

    Parameters
    ----------
    fit : `pfs.drp.qa.guiders.analysis.DriftFit`
        The drift, from `pfs.drp.qa.guiders.analysis.fitDriftRate`.
    byTime : `bool`
        Plot against minutes since the first AG exposure, rather than
        ``agc_exposure_id``.
    showCamera : `bool`
        Colour the points by camera; needs a fit with ``byCamera``.
    fitTrend : `bool`
        Draw the fitted lines and rates.
    name : `str`, optional
        Add this to the title.
    fig : `matplotlib.figure.Figure`, optional
        Draw on new panels in this figure.
    axes : iterable of `matplotlib.axes.Axes`, optional
        Draw on these two panels (radial and tangential, or y and x).

    Returns
    -------
    plot : `GuiderPlot`
    """
    offsets, rates = fit.offsets, fit.rates
    showCamera &= "agc_camera_id" in offsets
    components = ["radial", "tangential"] if "dradial_um" in offsets else ["y", "x"]

    fig, axes, ownPanels = _panels(2, 1, fig, axes, sharex=True, sharey=True)
    x = offsets.time_min if byTime else offsets.agc_exposure_id
    artists = []
    for ax, component in zip(axes[:, 0], components, strict=True):
        y = offsets[f"d{component}_um"]
        if showCamera:
            for cid in range(6):
                rows = (offsets.agc_camera_id == cid).to_numpy()
                if rows.any():
                    artists += ax.plot(x[rows], y[rows], "o", color=_CAMERA_COLORS[cid], label=f"AG{cid + 1}")
            ax.legend(ncol=6, loc="upper right", fontsize="small")
        else:
            artists += ax.plot(x, y, "o")
        ax.set_ylabel(component)

        if fitTrend:
            rate, offset = rates[f"{component}_rate_um_per_min"], rates[f"{component}_offset_um"]
            t = np.array([offsets.time_min.iloc[0], offsets.time_min.iloc[-1]])
            xx = (
                t if byTime else np.array([offsets.agc_exposure_id.iloc[0], offsets.agc_exposure_id.iloc[-1]])
            )
            artists += ax.plot(xx, offset + rate * t, color="black")
            ax.text(0.02, 0.95, f"Rate: {rate:.3f} µm/min", transform=ax.transAxes, va="top")
    axes[-1, 0].set_xlabel("time (min)" if byTime else "agc_exposure_id")
    yLabel = f"{offsets.attrs.get('offset', 'offset')} (µm)"
    if ownPanels:
        fig.supylabel(yLabel)
    else:
        for ax, component in zip(axes[:, 0], components, strict=True):
            ax.set_ylabel(f"{component}: {yLabel}")

    title = f"visit {int(rates.pfs_visit_id)}" + (f" {name}" if name else "")
    title += f"\nalt, az ({rates.altitude:.1f}, {rates.azimuth:.1f}) insrot {rates.insrot:.1f}"
    _finish(fig, axes, ownPanels, title)

    return GuiderPlot(fig=fig, axes=axes, artists=artists)


# Guide errors by AG camera on the PFI


def plotGuideErrors(
    guideErrors: pd.DataFrame,
    colorBy: str = "agc_exposure_id",
    showAGMean: bool = True,
    drawTrack: bool = False,
    rotateToAG1Down: bool = False,
    expand: float = 1,
    showClosedShutter: bool = False,
    name: str | None = None,
    showCartoon: bool = True,
    fig: Figure | None = None,
    ax: Axes | None = None,
    colorbars: Sequence[Colorbar] | None = None,
) -> GuiderPlot:
    """Plot each AG camera's mean guide error on the PFI.

    drp_stella's ``estimateGuideErrors(plot=True)``. Each camera's guide
    errors (microns) are drawn about its median position (mm), and their
    mean over the cameras about the boresight.

    Parameters
    ----------
    guideErrors : `pandas.DataFrame`
        Guide errors, from `pfs.drp.qa.guiders.analysis.estimateGuideErrors`.
    colorBy : `str`
        Colour by ``agc_exposure_id``, ``pfs_visit_id`` or ``time``
        (seconds since the first).
    showAGMean : `bool`
        Plot the mean over the cameras of each AG exposure (or visit) at the
        boresight.
    drawTrack : `bool`
        Join each camera's points in order.
    rotateToAG1Down : `bool`
        Rotate by minus the rotator angle: the mean for the cameras, each
        exposure's for the means.
    expand : `float`
        Bring the cameras this many times closer to the boresight.
    showClosedShutter : `bool`
        Also plot, smaller, the AG exposures taken with the shutters closed;
        needs ``estimateGuideErrors(includeClosedShutter=True)``.
    name : `str`, optional
        Add this to the title.
    showCartoon : `bool`
        Show where the cameras are; see `showAGCameraCartoon`.
    fig : `matplotlib.figure.Figure`, optional
        Draw in a new panel in this figure.
    ax : `matplotlib.axes.Axes`, optional
        Draw in this panel.
    colorbars : sequence of `matplotlib.colorbar.Colorbar`, optional
        Update this colorbar rather than adding one.

    Returns
    -------
    plot : `GuiderPlot`

    Notes
    -----
    drp_stella drew the open-shutter points twice, and couldn't show the
    closed-shutter ones; the points and the means now share a colour scale.
    """
    _checkChoice("colorBy", colorBy, ("agc_exposure_id", "pfs_visit_id", "time"))
    fig, axes, ownPanels = _panels(1, 1, fig, ax)
    ax = axes[0, 0]

    data = guideErrors.reset_index(drop=True)
    data["time"] = (data.taken_at - data.taken_at.min()).dt.total_seconds()
    centers = data.groupby("agc_camera_id")[["agc_nominal_x_mm", "agc_nominal_y_mm"]].median()
    xCamera = data.agc_camera_id.map(centers.agc_nominal_x_mm).to_numpy() / expand
    yCamera = data.agc_camera_id.map(centers.agc_nominal_y_mm).to_numpy() / expand
    insrot = np.deg2rad(data.insrot.mean())

    means = data.groupby("agc_exposure_id", as_index=False).agg(
        pfs_visit_id=("pfs_visit_id", "first"),
        dx_um=("dx_um", "mean"),
        dy_um=("dy_um", "mean"),
        shutter_open=("shutter_open", "first"),
        time=("time", "mean"),
        insrot=("insrot", "mean"),
    )
    norm = matplotlib.colors.Normalize(data[colorBy].min(), data[colorBy].max())

    x, y = xCamera + data.dx_um.to_numpy(), yCamera + data.dy_um.to_numpy()
    xMean, yMean = means.dx_um.to_numpy(), means.dy_um.to_numpy()
    if rotateToAG1Down:
        x, y = rotXY(-insrot, x, y)
        xMean, yMean = rotXY(-np.deg2rad(means.insrot.to_numpy()), xMean, yMean)

    sets = [(data.shutter_open > 0, means.shutter_open > 0, None)]
    if showClosedShutter:
        sets.append((data.shutter_open == 0, means.shutter_open == 0, 5))
    artists = []
    scatter = None
    for rows, meanRows, size in sets:
        rows, meanRows = rows.to_numpy(), meanRows.to_numpy()
        scatter = ax.scatter(x[rows], y[rows], marker="o", s=size, c=data[colorBy][rows], norm=norm)
        artists.append(scatter)
        if drawTrack:
            for cid in range(6):
                track = rows & (data.agc_camera_id == cid).to_numpy()
                artists += ax.plot(x[track], y[track], color="black", alpha=0.25, zorder=-1)
        if showAGMean:
            artists.append(
                ax.scatter(
                    xMean[meanRows],
                    yMean[meanRows],
                    marker="o",
                    s=size,
                    alpha=0.5,
                    c=means[colorBy][meanRows],
                    norm=norm,
                )
            )
            if drawTrack:
                artists += ax.plot(xMean[meanRows], yMean[meanRows], color="black", alpha=0.25, zorder=-1)
    xCenter, yCenter = (
        centers.agc_nominal_x_mm.to_numpy() / expand,
        centers.agc_nominal_y_mm.to_numpy() / expand,
    )
    if rotateToAG1Down:
        xCenter, yCenter = rotXY(-insrot, xCenter, yCenter)
    ax.plot(xCenter, yCenter, "+", color="red", zorder=10)

    label = {"pfs_visit_id": "pfs_visit_id", "time": "time (s)", "agc_exposure_id": "agc_exposure_id"}[
        colorBy
    ]
    newColorbars = (_colorbar(fig, scatter, label, colorbars, 0, ax=ax),)

    ax.plot([0], [0], "+", color="red")
    ax.set_aspect(1)
    limit = _GUIDE_ERRORS_LIMIT_MM / expand
    ax.set_xlim(ax.set_ylim(-limit, limit))
    offset = guideErrors.attrs.get("offset", "center - reference")
    ax.set_xlabel(f"x (µm)  ({offset})")
    ax.set_ylabel(f"y (µm)  ({offset})")

    title = [_visitRange(data.pfs_visit_id) + (f" {name}" if name else "")]
    if rotateToAG1Down:
        title.append(f"rotated {-np.rad2deg(insrot):.1f}")
    title.append(
        f"alt, az ({data.altitude.mean():.1f}, {data.azimuth.mean():.1f}) insrot {np.rad2deg(insrot):.1f}"
    )
    if showCartoon:
        showAGCameraCartoon(
            ax,
            showInstrot=not rotateToAG1Down,
            showUp=True,
            insrot_deg=np.rad2deg(insrot) if rotateToAG1Down else None,
        )
    _finish(fig, axes, ownPanels, "\n".join(title))

    return GuiderPlot(fig=fig, axes=axes, artists=artists, colorbars=newColorbars)


# Comparison with pfs_utils


def plotPfsUtilsComparison(
    comparison: PfsUtilsComparison,
    compress: float = 10,
    plotUsingScatter: bool = False,
    showCartoon: bool = True,
    fig: Figure | None = None,
    ax: Axes | None = None,
    colorbars: Sequence[Colorbar] | None = None,
) -> GuiderPlot:
    """Plot the guider's and pfs_utils's positions of a visit's guide stars.

    Each camera's stars are drawn at their positions about the camera, with
    the cameras ``compress`` times closer to the boresight: pfs_utils's as
    circles, the guider's as crosses.

    Parameters
    ----------
    comparison : `pfs.drp.qa.guiders.analysis.PfsUtilsComparison`
        The positions, from
        `pfs.drp.qa.guiders.analysis.comparePfsUtilsPositions`.
    compress : `float`
        Bring the cameras this many times closer to the boresight.
    plotUsingScatter : `bool`
        Colour the stars by camera, with a colorbar.
    showCartoon : `bool`
        Show where the cameras are; see `showAGCameraCartoon`.
    fig : `matplotlib.figure.Figure`, optional
        Draw in a new panel in this figure.
    ax : `matplotlib.axes.Axes`, optional
        Draw in this panel.
    colorbars : sequence of `matplotlib.colorbar.Colorbar`, optional
        With ``plotUsingScatter``, update this colorbar rather than adding
        one.

    Returns
    -------
    plot : `GuiderPlot`
    """
    fig, axes, ownPanels = _panels(1, 1, fig, ax)
    ax = axes[0, 0]
    stars = comparison.stars
    centers = stars.groupby("agc_camera_id")[["pfs_utils_x_mm", "pfs_utils_y_mm"]].mean()
    xCamera = stars.agc_camera_id.map(centers.pfs_utils_x_mm)
    yCamera = stars.agc_camera_id.map(centers.pfs_utils_y_mm)

    artists = []
    newColorbars = ()
    for which, label, marker in [("pfs_utils", "pfs_utils", "o"), ("agc_nominal", "agc", "+")]:
        x = stars[f"{which}_x_mm"] - xCamera + xCamera / compress
        y = stars[f"{which}_y_mm"] - yCamera + yCamera / compress
        if plotUsingScatter:
            scatter = ax.scatter(x, y, c=stars.agc_camera_id, marker=marker, label=label)
            artists.append(scatter)
            if label == "agc":
                newColorbars = (_colorbar(fig, scatter, "agc_camera_id", colorbars, 0, ax=ax),)
        elif label == "pfs_utils":
            artists += ax.plot(x, y, "o", markerfacecolor="none", markersize=10, label=label)
        else:
            artists += ax.plot(x, y, "+", markersize=10, label=label)
    ax.legend()

    if comparison.alignOffset_um is not None:
        dx, dy = umToMm(comparison.alignOffset_um[0]), umToMm(comparison.alignOffset_um[1])
        ax.arrow(0, 0.75, dx, dy, length_includes_head=True, color="red")
    ax.plot([0], [0], "+", color="red")
    limit = _PFS_UTILS_LIMIT_MM / compress
    ax.set_xlim(ax.set_ylim(-limit, limit))
    ax.set_aspect(1)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    if showCartoon:
        showAGCameraCartoon(ax)

    title = [f"visit: {int(stars.pfs_visit_id.iloc[0])}"] if "pfs_visit_id" in stars and len(stars) else []
    title.append(
        f"alt, az ({stars.altitude.mean():.1f}, {stars.azimuth.mean():.1f}) insrot {stars.insrot.mean():.1f}"
    )
    title.append(f"d(theta) = {stars.delta_theta_arcsec.median():.1f} arcsec")
    if comparison.alignOffset_um is not None:
        title[-1] += (
            f"  (dx, dy) = ({comparison.alignOffset_um[0]:.0f}, {comparison.alignOffset_um[1]:.0f}) µm"
        )
    _finish(fig, axes, ownPanels, "\n".join(title))

    return GuiderPlot(fig=fig, axes=axes, artists=artists, colorbars=newColorbars)


# Focus


class ShowFocusFit:
    """Click handler drawing the guider focus error expected at an M2_OFF3.

    A click on any of the plot's panels sets the M2_OFF3 of best focus to
    the click's x; the top-left panel then shows the focus error expected at
    each M2_OFF3, with `pfs.drp.qa.guiders.coordinates.m2Off3ToGuiderFocus`.
    `plotFocus` connects one to its figure when plotting against focus.

    Parameters
    ----------
    axes : `numpy.ndarray` of `matplotlib.axes.Axes`
        The plot's panels, rows by columns.
    indicateFocusPosition : `bool`
        Also draw a vertical line at the M2_OFF3 on every panel.
    """

    def __init__(self, axes: np.ndarray, indicateFocusPosition: bool = False):
        self.indicateFocusPosition = indicateFocusPosition
        self._axes = np.asarray(axes, dtype=object)
        ax = self._axes[0, 0]
        self._text = ax.text(
            0.99, 0.02, "Click to set M2_OFF3", ha="right", va="bottom", transform=ax.transAxes
        )
        self.focus_mm = None
        self.lines = []

    def __call__(self, event) -> None:
        if event.inaxes is None or event.inaxes not in list(self._axes.flat):
            return
        self.focus_mm = event.xdata
        self._text.set_text(f"M2_OFF3 = {self.focus_mm:.2f}mm")

        for line in self.lines:
            line.remove()
        self.lines = []

        ax = self._axes[0, 0]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        m2Off3 = np.array(xlim)
        self.lines += ax.plot(m2Off3, m2Off3ToGuiderFocus(m2Off3 - self.focus_mm), color="black")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if self.indicateFocusPosition:
            self.lines += [
                panel.axvline(self.focus_mm, color="black", alpha=0.5) for panel in self._axes.flat
            ]


# The ShowFocusFit connected to each figure, so that redrawing a plot replaces it.
_focusFits: "weakref.WeakKeyDictionary[Figure, int]" = weakref.WeakKeyDictionary()


def _showFocusSets(axes: np.ndarray, byExposure: pd.DataFrame, what: str) -> None:
    """Shade the ranges of ``what`` over which the focus position didn't change."""
    byExposure = byExposure.sort_values("agc_exposure_id")
    focus = np.round(byExposure.focus_position_mm.to_numpy() / 1e-3)
    focus = np.where(np.isfinite(focus), focus, -1)
    values = byExposure[what].to_numpy()
    edges = [values.min(), *values[:-1][np.diff(focus) != 0], values.max()]
    for i, (start, end) in enumerate(itertools.pairwise(edges)):
        for ax in axes.flat:
            ax.axvspan(start, end, color="black" if i % 2 == 0 else "brown", alpha=0.1, zorder=-1)


def _medianByX(x: np.ndarray, y: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the median of y at each value of x, rounded to ``resolution``."""
    xRounded = np.round(np.asarray(x, dtype=float) / resolution) * resolution
    medians = pd.Series(np.asarray(y, dtype=float)).groupby(xRounded).median()

    return medians.index.to_numpy(), medians.to_numpy()


def plotFocus(
    agcData: pd.DataFrame,
    agcCameraIds: Iterable[int] = range(6),
    plotBy: str = "focus",
    colorBy: str = "camera",
    showAGActorFocus: bool = True,
    showOpdbFocus: bool = True,
    showFWHM: bool = True,
    plotPerCamera: bool = False,
    showPfiFocusPosition: bool = False,
    averageByFocusPosition: bool = False,
    showMedian: bool = False,
    showOnlyMedian: bool = False,
    connectMedian: bool = True,
    showCameraId: bool = False,
    showFocusSets: bool = False,
    onlyGuideStars: bool = True,
    plotFrac: float = 1,
    ditherScale: float = 5e-3,
    yLimits_um: float | tuple[float, float] | None = 180,
    indicateFocusPosition: bool = False,
    useTraceRadius: bool = True,
    magMin: float | None = None,
    magMax: float | None = None,
    minFWHM_arcsec: float | None = None,
    maxFWHM_arcsec: float | None = None,
    useM2Off3: bool = True,
    forceAlpha: float | None = None,
    scatterMarkerSize: float | None = None,
    designNames: Mapping[int, str] | None = None,
    fig: Figure | None = None,
    axes: Iterable[Axes] | None = None,
    colorbars: Sequence[Colorbar] | None = None,
) -> GuiderPlot:
    """Plot the AG cameras' focus errors and image sizes.

    Up to three rows of panels: the AG actor's focus error
    (``guide_delta_z1`` to ``guide_delta_z6``, corrected with
    `pfs.drp.qa.guiders.analysis.correctAgActorFocus`); the focus error from
    the stars' sizes on the two halves of the detectors
    (`pfs.drp.qa.guiders.analysis.estimateFocusErrors`); and the stars' FWHM,
    left half red and right half green unless ``showCameraId``. Only valid
    matches without bad detection flags are used.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data, from `pfs.drp.qa.guiders.queries.readAgcData`.
    agcCameraIds : iterable of `int`
        The cameras to plot (0-5; AG1 is 0).
    plotBy : `str`
        The x axis: ``focus`` (M2_OFF3, or M2_POS3), ``agc_exposure_id``,
        ``altitude`` or ``insrot``.
    colorBy : `str`
        Colour the focus errors by ``camera``, ``visit``, ``altitude`` or
        ``insrot``; with any but ``camera``, the focus errors are those of
        each AG exposure (of each camera's, with ``plotPerCamera``).
    showAGActorFocus, showOpdbFocus, showFWHM : `bool`
        Draw each row of panels.
    plotPerCamera : `bool`
        Give each camera its own column of panels.
    showPfiFocusPosition : `bool`
        Mark the focus of the PFI: 0 for the AG actor, which corrects for the
        offset between the AG and PFI focal planes, and
        `pfs.drp.qa.guiders.analysis.FOCUS_PISTON_OFFSET_MM` for the stars.
    averageByFocusPosition : `bool`
        Plot the median focus error of each camera at each focus position.
    showMedian : `bool`
        Join the medians of the focus errors at each x, the AG actor's and
        the stars' (each camera's, or in black with ``colorBy`` other than
        ``camera``), and of the stars' FWHM (each half's; with
        ``showCameraId``, each camera's and half's in each AG exposure, in
        the halves' symbols).
    showOnlyMedian : `bool`
        Plot only those medians.
    connectMedian : `bool`
        Draw the medians at each x as lines, rather than points.
    showCameraId : `bool`
        Colour the FWHM by camera, with the halves of the detectors as
        markers.
    showFocusSets : `bool`
        Shade the ranges over which the focus position didn't change.
    onlyGuideStars : `bool`
        Only use isolated GAIA stars
        (`pfs.drp.qa.guiders.analysis.selectIsolatedGaiaStars`).
    plotFrac : `float`
        Plot this fraction of the stars' FWHM, chosen at random (but the
        same each time).
    ditherScale : `float`
        Spread the FWHM points along x by this fraction of the x range,
        according to their row on the detector.
    yLimits_um : `float` or `tuple` [`float`, `float`], optional
        Limits of the focus errors (microns): ``(-y, y)`` for a number; none
        for `None` or a number <= 0.
    indicateFocusPosition : `bool`
        See `ShowFocusFit`.
    useTraceRadius : `bool`
        Measure sizes with the trace radius rather than the determinant
        radius; see `pfs.drp.qa.guiders.analysis.addImageSizes`.
    magMin, magMax : `float`, optional
        Only use stars of these estimated magnitudes.
    minFWHM_arcsec, maxFWHM_arcsec : `float`, optional
        Limits of the FWHM axis.
    useM2Off3 : `bool`
        Plot against M2_OFF3, rather than M2_POS3.
    forceAlpha : `float`, optional
        Alpha of the points; by default it depends on their number.
    scatterMarkerSize : `float`, optional
        Size of the points.
    designNames : `~collections.abc.Mapping` [`int`, `str`], optional
        Design names by visit, for the cursor readout; see `FormatCoord`.
    fig : `matplotlib.figure.Figure`, optional
        Draw on new panels in this figure.
    axes : iterable of `matplotlib.axes.Axes`, optional
        Draw on these panels, rows by columns, e.g. ics_pfsPlotActor's.
    colorbars : sequence of `matplotlib.colorbar.Colorbar`, optional
        Update this colorbar (with ``colorBy`` other than ``camera``)
        rather than adding one.

    Returns
    -------
    plot : `GuiderPlot`
        ``data`` has the stars used, with ``rms_pix``, ``fwhm_arcsec``,
        ``left`` and ``focus_position_mm``.

    Notes
    -----
    Compared with drp_stella: every helper takes 2-D axes; ``plotFrac``
    applies to every array; the FWHM are drawn once; the INSTRM-2501
    correction applies to each visit before the fix; the stars' focus errors
    are always in microns (drp_stella's ``mmToMicrons`` is gone); and with
    ``plotPerCamera`` the FWHM are split by camera too.
    """
    _checkChoice("plotBy", plotBy, ("focus", *_PLOT_BY))
    _checkChoice("colorBy", colorBy, _COLOR_BY)
    colorColumn = _COLOR_BY[colorBy]
    what = "focus_position_mm" if plotBy == "focus" else plotBy
    agcCameraIds = [int(cid) for cid in agcCameraIds]
    if showOnlyMedian:
        showMedian = True
    if yLimits_um is None or (np.ndim(yLimits_um) == 0 and yLimits_um <= 0):
        yLimits_um = None
    elif np.ndim(yLimits_um) == 0:
        yLimits_um = (-yLimits_um, yLimits_um)

    rows = [
        name
        for name, on in [("agActor", showAGActorFocus), ("opdb", showOpdbFocus), ("fwhm", showFWHM)]
        if on
    ]
    if not rows:
        raise ValueError("Nothing to plot: set showAGActorFocus, showOpdbFocus or showFWHM")

    data = addImageSizes(agcData, useTraceRadius)
    keep = selectValidMatches(data) & selectGoodDetections(data)
    keep &= data.agc_camera_id.isin(agcCameraIds).to_numpy()
    if magMin is not None:
        keep &= (data.estimated_magnitude >= magMin).to_numpy()
    if magMax is not None:
        keep &= (data.estimated_magnitude <= magMax).to_numpy()
    if onlyGuideStars:
        keep &= selectIsolatedGaiaStars(data)
    data = correctAgActorFocus(data[keep]).reset_index(drop=True)
    data["agc_camera_id"] = data.agc_camera_id.astype(int)
    data["focus_position_mm"] = data.m2_off3 if useM2Off3 else data.m2_pos3
    focusColumn = "m2_off3" if useM2Off3 else "m2_pos3"

    nCols = len(agcCameraIds) if plotPerCamera else 1
    heights = {"agActor": 2, "opdb": 2, "fwhm": 3}
    fig, axes, ownPanels = _panels(
        len(rows), nCols, fig, axes, sharex=True, sharey="row", height_ratios=[heights[row] for row in rows]
    )
    if ownPanels:
        fig.subplots_adjust(hspace=0.025, wspace=0.025)
    panelCameras = [[cid] for cid in agcCameraIds] if plotPerCamera else [agcCameraIds]

    byCamera = data.groupby(["agc_exposure_id", "agc_camera_id"], as_index=False).agg(
        pfs_visit_id=("pfs_visit_id", "first"),
        altitude=("altitude", "mean"),
        insrot=("insrot", "mean"),
        focus_position_mm=("focus_position_mm", "mean"),
        guide_delta_z=("guide_delta_z", "first"),
        **{f"guide_delta_z{cid + 1}": (f"guide_delta_z{cid + 1}", "first") for cid in range(6)},
    )
    agActor = np.column_stack([byCamera[f"guide_delta_z{cid + 1}"].to_numpy(dtype=float) for cid in range(6)])
    byCamera["ag_actor_focus_error_um"] = mmToUm(agActor[np.arange(len(byCamera)), byCamera.agc_camera_id])
    byCamera["exposure_focus_error_um"] = mmToUm(byCamera.guide_delta_z)

    colorValues = pd.concat([byCamera[colorColumn], data[colorColumn]])
    norm = matplotlib.colors.Normalize(colorValues.min(), colorValues.max()) if colorBy != "camera" else None
    resolution = 1e-3 if plotBy == "focus" else 1

    cameraFocusErrors = (
        estimateFocusErrors(data, byCamera=True, focusColumn=focusColumn) if "opdb" in rows else None
    )
    nPerCamera = len(byCamera) / max(1, byCamera.agc_camera_id.nunique())
    marker, alpha = ("o", 1) if averageByFocusPosition else _markerAndAlpha(nPerCamera, forceAlpha)

    def drawFocusErrors(ax, x, y, c, color, label) -> list[Artist]:
        """Draw focus errors, as points coloured by ``c`` or in ``color``, and their medians at each x."""
        drawn = []
        if not showOnlyMedian:
            if c is None:
                drawn += ax.plot(x, y, marker, alpha=alpha, color=color, label=label)
            else:
                drawn.append(
                    ax.scatter(x, y, c=c, norm=norm, marker=marker, s=scatterMarkerSize, alpha=alpha)
                )
        if showMedian and len(x):
            xm, ym = _medianByX(x, y, resolution)
            drawn += ax.plot(
                xm,
                ym,
                "-" if connectMedian else marker,
                color=color,
                alpha=1 if connectMedian else alpha,
                label=label if showOnlyMedian else None,
            )
        return drawn

    artists = []
    mappable = None
    for i, row in enumerate(rows):
        for j, cameras in enumerate(panelCameras):
            ax = axes[i, j]
            if row in ("agActor", "opdb"):
                if colorBy == "camera":
                    for cid in cameras:
                        if row == "agActor":
                            cam = byCamera[byCamera.agc_camera_id == cid].dropna(
                                subset="ag_actor_focus_error_um"
                            )
                            y = cam.ag_actor_focus_error_um.to_numpy()
                        else:
                            cam = cameraFocusErrors[cameraFocusErrors.agc_camera_id == cid]
                            y = cam.focus_error_um.to_numpy()
                        if cam.empty:
                            continue
                        x = cam[what].to_numpy()
                        if averageByFocusPosition:
                            x, y = analysis.averageByFocusPosition(cam.focus_position_mm, x, y)
                        artists += drawFocusErrors(ax, x, y, None, _CAMERA_COLORS[cid], f"AG{cid + 1}")
                else:
                    if row == "opdb":
                        stars = data[data.agc_camera_id.isin(cameras)]
                        points = estimateFocusErrors(stars, byCamera=False, focusColumn=focusColumn)
                        y = points.focus_error_um
                    elif plotPerCamera:
                        points = byCamera[byCamera.agc_camera_id == cameras[0]]
                        y = points.ag_actor_focus_error_um
                    else:
                        points = byCamera.groupby("agc_exposure_id", as_index=False).first()
                        y = points.exposure_focus_error_um
                    newArtists = drawFocusErrors(ax, points[what], y, points[colorColumn], "black", None)
                    mappable = next((a for a in newArtists if isinstance(a, PathCollection)), mappable)
                    artists += newArtists
                ylabel = "AG actor focus error" if row == "agActor" else r"$\Delta$ focus"
            else:
                stars = data[data.agc_camera_id.isin(cameras)]
                artists += _plotFwhm(
                    ax,
                    stars,
                    what,
                    showCameraId,
                    showMedian,
                    showOnlyMedian,
                    connectMedian,
                    plotFrac,
                    ditherScale,
                    forceAlpha,
                    scatterMarkerSize,
                    resolution,
                )
                ax.set_ylim(minFWHM_arcsec, maxFWHM_arcsec)
                if j == 0:
                    ax.set_ylabel("FWHM (arcsec)")
                continue

            if colorBy == "camera" and not plotPerCamera and ax.get_legend_handles_labels()[0]:
                ax.legend(ncols=6, fontsize="small")
            ax.axhline(0, color="black", alpha=0.5)
            if showPfiFocusPosition:
                pfiFocus_um = 0 if row == "agActor" else mmToUm(FOCUS_PISTON_OFFSET_MM)
                ax.axhline(pfiFocus_um, color="red", label="PFI")
                ax.legend(ncols=7)
            if yLimits_um is not None:
                ax.set_ylim(yLimits_um)
            if plotPerCamera and i == 0:
                ax.text(
                    0.1, 0.95, f"AG{cameras[0] + 1}", transform=ax.transAxes, color=_CAMERA_COLORS[cameras[0]]
                )
            if j == 0:
                ax.set_ylabel(f"{ylabel}\n(µm)")
                if colorBy == "camera":
                    secondary = ax.secondary_yaxis(
                        "right", functions=(guiderFocusToM2Off3, m2Off3ToGuiderFocus)
                    )
                    secondary.set_ylabel(r"$\Delta$ M2_OFF3 (mm)")

    newColorbars = ()
    if mappable is not None:
        newColorbars = (_colorbar(fig, mappable, colorBy, colorbars, 0, ax=list(axes.flat)),)

    if ownPanels and "agActor" in rows and "opdb" in rows:
        axes[rows.index("opdb"), 0].sharey(axes[rows.index("agActor"), 0])

    xlabel = ("M2_OFF3 (mm)" if useM2Off3 else "M2_POS3 (mm)") if plotBy == "focus" else plotBy
    _xLabel(fig, axes, ownPanels, xlabel)

    if showFocusSets:
        _showFocusSets(axes, byCamera.groupby("agc_exposure_id", as_index=False).first(), what)
    if plotBy == "agc_exposure_id":
        for ax in axes.flat:
            _showVisitBoundaries(ax, data)
            ax.format_coord = FormatCoord(plotBy, data, designNames)
    if plotBy == "focus":
        focusFit = ShowFocusFit(axes, indicateFocusPosition=indicateFocusPosition)
        canvas = fig.canvas
        if fig in _focusFits:
            canvas.mpl_disconnect(_focusFits[fig])
        _focusFits[fig] = canvas.mpl_connect("button_press_event", focusFit)

    magLimits = ""
    if magMin is not None:
        magLimits = f"{magMin} < mag" + (f" < {magMax}" if magMax is not None else "")
    elif magMax is not None:
        magLimits = f"mag < {magMax}"
    title = f"pfs_visit_id {_visitRange(data.pfs_visit_id)}" + (" only GAIA stars" if onlyGuideStars else "")
    title += "\n" + ", ".join(f"AG{cid + 1}" for cid in sorted(data.agc_camera_id.unique()))
    title += (
        (f"  {magLimits}" if magLimits else "") + "  " + ("traceRadius" if useTraceRadius else "detRadius")
    )
    _finish(fig, axes, ownPanels, title)

    return GuiderPlot(fig=fig, axes=axes, artists=artists, colorbars=newColorbars, data=data)


def _plotFwhm(
    ax: Axes,
    stars: pd.DataFrame,
    what: str,
    showCameraId: bool,
    showMedian: bool,
    showOnlyMedian: bool,
    connectMedian: bool,
    plotFrac: float,
    ditherScale: float,
    forceAlpha: float | None,
    scatterMarkerSize: float | None,
    resolution: float,
) -> list[Artist]:
    """Plot the stars' FWHM for `plotFocus`; return the artists."""
    artists = []
    if not showOnlyMedian and len(stars):
        if plotFrac < 1:
            use = np.random.default_rng(_GUIDE_STAR_SEED).uniform(size=len(stars)) < plotFrac
            stars = stars[use]
        x = stars[what].to_numpy(dtype=float)
        rowOnDetector = stars.centroid_y_pix.to_numpy(dtype=float)
        meanRow = np.nanmean(rowOnDetector)
        if len(x) and meanRow:
            x = x + ditherScale * (np.nanmax(x) - np.nanmin(x)) / meanRow * (rowOnDetector - meanRow)
        y = stars.fwhm_arcsec.to_numpy()
        left = stars.left.to_numpy(dtype=bool)
        marker, alpha = _markerAndAlpha(len(x), forceAlpha)

        if showCameraId:
            cameraCmap, cameraNorm = _cameraColormap()
            for isLeft in (True, False):
                rows = left == isLeft
                size = (6 if isLeft else 4.5) ** 2 if scatterMarkerSize is None else scatterMarkerSize
                artists.append(
                    ax.scatter(
                        x[rows],
                        y[rows],
                        c=stars.agc_camera_id[rows] + 1,
                        s=size,
                        marker=marker if isLeft else "*",
                        alpha=alpha,
                        cmap=cameraCmap,
                        norm=cameraNorm,
                    )
                )
        else:
            artists.append(
                ax.scatter(
                    x, y, c=np.where(left, "red", "green"), s=scatterMarkerSize, marker=marker, alpha=alpha
                )
            )

    if showMedian and len(stars):
        if showCameraId:
            for cid in np.sort(stars.agc_camera_id.unique()):
                camera = stars[stars.agc_camera_id == cid]
                # The halves' symbols, as for the stars, larger and outlined.
                for isLeft, marker, size in [(True, "o", 8), (False, "*", 11)]:
                    half = camera[camera.left == isLeft]
                    medians = half.groupby("agc_exposure_id").agg(
                        x=(what, "mean"), y=("fwhm_arcsec", "median")
                    )
                    artists += ax.plot(
                        medians.x,
                        medians.y,
                        marker,
                        markersize=size,
                        markeredgecolor="black",
                        color=_CAMERA_COLORS[cid],
                    )
        else:
            for isLeft, color in [(True, "red"), (False, "green")]:
                half = stars[stars.left == isLeft]
                xm, ym = _medianByX(half[what], half.fwhm_arcsec, resolution)
                artists += ax.plot(xm, ym, "-" if connectMedian else "o", color=color)

    handles = []
    if showCameraId:
        handles += [
            Line2D([], [], marker="o", ls="", color="black", label="left"),
            Line2D([], [], marker="*", ls="", color="black", label="right"),
        ]
        handles += [
            Line2D([], [], marker="o", ls="", color=_CAMERA_COLORS[cid], label=f"AG{cid + 1}")
            for cid in np.sort(stars.agc_camera_id.unique())
        ]
        ax.legend(handles=handles, ncol=8, columnspacing=1.3)
    elif showOnlyMedian:
        handles = [
            Line2D([], [], marker="o", ls="", color="red", label="left"),
            Line2D([], [], marker="o", ls="", color="green", label="right"),
        ]
        ax.legend(handles=handles)

    return artists


def plotFocusByAG(
    agcData: pd.DataFrame,
    onlyGuideStars: bool = True,
    byExposureId: bool = False,
    byCamera: bool = True,
    maxFocusError_um: float = 0,
    agcExposureIdMin: int = 0,
    agcExposureIdMax: int = 0,
    useTraceRadius: bool = True,
    showLegend: bool = False,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> GuiderPlot:
    """Plot each AG camera's focus error relative to the others.

    The focus error of each camera in each AG exposure is from its stars'
    sizes (`pfs.drp.qa.guiders.analysis.estimateFocusErrors`). Each camera's
    median in each visit (or AG exposure) has the mean of the other cameras
    but AG1 subtracted, and then the overall mean, and is negated to match
    Kawanomoto-san's plots.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data, from `pfs.drp.qa.guiders.queries.readAgcData`.
    onlyGuideStars : `bool`
        Only use isolated GAIA stars
        (`pfs.drp.qa.guiders.analysis.selectIsolatedGaiaStars`).
    byExposureId : `bool`
        Take each camera's median in each AG exposure, rather than each
        visit.
    byCamera : `bool`
        Plot against the camera, one line per visit (or AG exposure),
        rather than against the visit, one line per camera.
    maxFocusError_um : `float`
        Only use focus errors smaller than this (microns); no cut if <= 0.
    agcExposureIdMin, agcExposureIdMax : `int`
        Only use these AG exposures; no limit if <= 0.
    useTraceRadius : `bool`
        See `pfs.drp.qa.guiders.analysis.addImageSizes`.
    showLegend : `bool`
        Add a legend.
    fig : `matplotlib.figure.Figure`, optional
        Draw in a new panel in this figure.
    ax : `matplotlib.axes.Axes`, optional
        Draw in this panel.

    Returns
    -------
    plot : `GuiderPlot`
        ``data`` has the relative focus error, ``focus_error_um``, of each
        visit (or AG exposure) and camera.

    Notes
    -----
    drp_stella's ``onlyGuideStars`` didn't select anything.
    """
    data = addImageSizes(agcData, useTraceRadius)
    keep = selectGoodDetections(data)
    if onlyGuideStars:
        keep &= selectIsolatedGaiaStars(data)
    focusErrors = estimateFocusErrors(data[keep], byCamera=True)

    use = np.ones(len(focusErrors), dtype=bool)
    if maxFocusError_um > 0:
        use &= (focusErrors.focus_error_um.abs() < maxFocusError_um).to_numpy()
    if agcExposureIdMin > 0:
        use &= (focusErrors.agc_exposure_id >= agcExposureIdMin).to_numpy()
    if agcExposureIdMax > 0:
        use &= (focusErrors.agc_exposure_id <= agcExposureIdMax).to_numpy()
    focusErrors = focusErrors[use]

    xName = "agc_exposure_id" if byExposureId else "pfs_visit_id"
    relative = focusErrors.groupby([xName, "agc_camera_id"], as_index=False).agg(
        focus_error_um=("focus_error_um", "median"),
        focus_position_mm=("focus_position_mm", "median"),
        nAgcExposures=("agc_exposure_id", "count"),
    )
    # The mean of the cameras but AG1 at each x.
    others = relative[relative.agc_camera_id != 0].groupby(xName).focus_error_um.mean()
    relative["focus_error_um"] -= relative[xName].map(others)
    relative["focus_error_um"] -= relative.focus_error_um.mean()
    relative["focus_error_um"] *= -1

    fig, axes, ownPanels = _panels(1, 1, fig, ax)
    ax = axes[0, 0]
    artists = []
    if byCamera:
        for value in np.sort(relative[xName].unique()):
            rows = relative[relative[xName] == value]
            artists += ax.plot(rows.agc_camera_id + 1, rows.focus_error_um, "-o", label=f"{value}", alpha=0.5)

    perCamera = np.full(6, np.nan)
    for cid in range(6):
        rows = relative[relative.agc_camera_id == cid]
        if rows.empty:
            continue
        color = _CAMERA_COLORS[cid]
        if not byCamera:
            artists += ax.plot(rows[xName], rows.focus_error_um, "-o", color=color, label=f"AG{cid + 1}")
        perCamera[cid] = np.nanmean(rows.focus_error_um)
        ax.axhline(perCamera[cid], color=color, label=f"AG{cid + 1}" if byCamera else None)
    if byCamera:
        artists += ax.plot(range(1, 7), perCamera, marker="*", markersize=10, color="black")

    if showLegend:
        ax.legend(ncols=2)
    if byCamera:
        ax.set_xticks(range(1, 7))
    ax.set_xlabel("AG camera" if byCamera else xName)
    ax.set_ylabel(r"$\Delta$ focus AG (µm)")

    title = f"{xName} {_visitRange(relative[xName])}" + (" only GAIA" if onlyGuideStars else "")
    title += "\n" + ", ".join(f"AG{cid + 1}" for cid in sorted(relative.agc_camera_id.unique()))
    _finish(fig, axes, ownPanels, title)

    return GuiderPlot(fig=fig, axes=axes, artists=artists, data=relative)
