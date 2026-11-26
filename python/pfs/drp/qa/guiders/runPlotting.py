"""Plots of a whole run's AG data, for the end-of-run report.

Each function takes an `pfs.drp.qa.guiders.runAnalysis.AgRun` (and, for the
guiding plot, its visits' guide errors) and returns a new
`matplotlib.figure.Figure`, made without pyplot so that it is never shown or
left open. Nothing here touches a database.

The x axis is a strip of nights (`runAnalysis.nightPosition`): one slot per
night with at least ``RunConfig.minFramesNight`` AG exposures, labeled by
its evening's date and spanning 18:00 to 06:00 HST. Per-camera panels are in
two rows, AG1/3/5 above AG2/4/6, the two groups being wired and placed
separately. Problems are drawn in red, so that they stand out.

`captions` says what each figure shows, for whatever page it is put on.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.ticker import MaxNLocator

from pfs.drp.qa.guiders.queries import AG_GRID_CELL_PIX, AG_RIGHT_EDGE_PIX
from pfs.drp.qa.guiders.runAnalysis import (
    CAMERA_GROUPS,
    DETECTION_FLAG_COLUMNS,
    MODEL_PARTS,
    AgRun,
    RunConfig,
    _meanAngle,
    cameraFocus,
    detectionFlagShares,
    detectorCells,
    isRejectedOffset,
    magnitudeSummary,
    nightPosition,
    onSkyRa,
    stateShares,
    summariseModelErrors,
)
from pfs.utils.coordinates.coordinates import det2dp

__all__ = [
    "AG_PIXEL_UM",
    "CAMERA_FIGURE_IN",
    "CAMERA_ORDER",
    "FOCAL_PLANE_FIGURE_IN",
    "GROUPS_NOTE",
    "captions",
    "plotAcquisition",
    "plotCamera",
    "plotCameraPresence",
    "plotFocalPlane",
    "plotFocus",
    "plotGuiding",
]

#: The cameras in the order of the per-camera panels: AG1/3/5, then AG2/4/6.
CAMERA_ORDER = [*CAMERA_GROUPS["AG1/3/5"], *CAMERA_GROUPS["AG2/4/6"]]

#: The size of an AG pixel (µm).
AG_PIXEL_UM = 13.0

BAD = "#c8102e"  # a fault
WARN = "#eda100"  # worth a look
OFF = "#3d6fb4"  # switched off on purpose: a camera the guider left out
QUIET = "#b5b4ae"  # context: other populations, reference lines
INK = "#52514e"  # the normal case
EMPTY = "#ecebe8"  # an empty heatmap cell

#: The sizes of `plotCamera`'s and `plotFocalPlane`'s figures (inches).
CAMERA_FIGURE_IN = (12.5, 6.4)
FOCAL_PLANE_FIGURE_IN = (13.0, 8.6)

#: On the detector map of `plotFocalPlane`: the top of the color scale (% of paired detections rejected), the good
#: matches a cell needs for its residual to be drawn, and the length of an arrow (detector pixels per µm).
REJECTED_MAX = 30
MIN_CELL_MATCHES = 50
ARROW_PIX_PER_UM = 50

#: The scale of `plotFocalPlane`'s focus panel (µm either side of the cameras' median), and its color maps.
FOCUS_MAX = 40
FOCUS_CMAP = LinearSegmentedColormap.from_list("focus", ["#3d6fb4", "#ffffff", "#c8102e"])
FWHM_CMAP = LinearSegmentedColormap.from_list("fwhm", ["#e3ecf7", "#2f5a9a"])

#: An AG detector (pixels, x then y), as `readAgRun`'s ``grid`` divides it.
_DETECTOR_PIX = (4 * AG_GRID_CELL_PIX["columnCell"], 4 * AG_GRID_CELL_PIX["rowCell"])

#: Why the report keeps the cameras in two groups.
GROUPS_NOTE = (
    "The six cameras alternate around the field in two groups, AG1/3/5 and AG2/4/6, which are wired "
    "separately. The guider treats all six alike, and when all work the grouping makes no difference to "
    "guiding. It is kept because a wiring fault takes a whole group at once (AG1/3/5 in Run 30), and "
    "guiding on one group alone holds the rotation and the focus less well, so a loss shows as a pattern "
    "across a group, not in one camera."
)


def captions(config: RunConfig | None = None) -> dict[str, str]:
    """Return what each figure of the report shows, by figure (``camera`` for the per-camera one).

    Parameters
    ----------
    config : `pfs.drp.qa.guiders.runAnalysis.RunConfig`, optional
        The parameters the run was analyzed with, for the numbers quoted.

    Returns
    -------
    captions : `dict` [`str`, `str`]
        A paragraph each for ``cameraPresence``, ``guiding``, ``focus``,
        ``acquisition`` and ``camera``, in plain text.
    """
    config = config or RunConfig()
    return {
        "cameraPresence": (
            "What each camera gave, hour by hour, in the AG exposures in which at least one camera matched "
            "a guide star (so the sky was usable). A cell is gray when the camera matched a star in every "
            "exposure of that hour, and colored by the share in which it did not: red when it mostly "
            "detected nothing at all, amber when it detected spots but matched none of them to a guide "
            "star, blue when the guider had left it out of its fit (disabled). The opdb records a camera's "
            "detections, not whether it returned an image, so a dead camera and an image with nothing in it "
            "both read as no detections: a camera that is red for hours while the others match stars is "
            f"dead or not being read out. Hatched: the design put fewer than {config.minDesignStars} bright "
            "guide stars on the camera, so no match is expected. A camera with no detections in the whole "
            "run is marked not in use."
        ),
        "guiding": (
            "One point per science visit, from the guide offsets the guider accepted in AG exposures taken "
            "while the spectrograph shutters were open, for visits with at least "
            f"{config.minOpenFrames} of them. A guide offset is the pointing error the guider measured in one AG exposure (ΔRA on "
            "the sky, ΔDec, and the rotator angle ΔInR). Size is how large the error was: the RMS of the "
            "visit's offsets about zero. Jitter is how steady the measurement was: frame to frame, the "
            "standard deviation of consecutive differences / √2, which a slow drift barely changes. "
            "Dashed lines are the run's medians; red marks a visit in which stars were not matched on "
            "every camera in use; a visit beyond a panel's scale is marked at its edge. Last: the AG exposures "
            "each night whose offset the guider rejected (INVALID_OFFSET: not sent to the telescope) or "
            "never wrote."
        ),
        "focus": (
            "First, the guider's focus error dZ, the median over the cameras of the focus each measures "
            "from the difference between its detector's two halves, as the median of each science visit; "
            "red marks a visit in which stars were not matched on every camera in use. Then the telescope's "
            "focus offset (tel_status.m2_off3); the spikes at the start of a night are AG focus sweeps."
        ),
        "acquisition": (
            "The first guide offset of each convergence visit (the move to a new design): the pointing error "
            f"the guider found on arriving at a field. Dashed: the guider's max_correction, ±{config.maxCorrection:g}″; "
            "an offset beyond it is rejected."
        ),
        "focalPlane": (
            "The six detectors where they sit on the PFI, each enlarged, over the whole run. Left, the "
            "cameras' position models: where the guider expects each guide star to land on a detector. "
            "Each matched star has a residual, measured minus expected; a pointing error moves all of a "
            "camera's stars together, so each AG exposure's mean is taken off, and what is left should be "
            "noise. A cell's color is the share of its paired detections that were rejected, and its arrow "
            "the mean residual left there. Beside each camera are the three parts fitted to its residuals: "
            f"x and y, how far a star {config.modelErrorRadiusPix:g} px from the detector's center along "
            "that axis lands from where it is expected (positive: the stars are spread farther apart than "
            "the model has them; negative: closer together), and the step between the detector's two "
            "halves along x. Detector x runs around the field and y along its radius, so a part that all "
            "the cameras share belongs to what they share, not to a camera. Top right: each camera's focus "
            "against the median of the cameras in use, where a tilt of the focal plane shows as a gradient "
            "across the field. Bottom right: the FWHM of its matched stars on each half of its detector "
            "(the two halves focus differently by design: one has glass in front of it)."
        ),
        "camera": (
            "One camera over the run, night by night. Top row: the good matches per AG exposure, as the "
            "median of each visit; the share of AG exposures in which it matched a star; and the share of "
            "its designed guide stars that were matched, by their magnitude (science visits; the dotted "
            "line is all cameras). Bottom row: the share of its detections that ics_agccActor flagged as "
            "bad, by flag; the AG exposures in which its rejected matches clustered on one side of its "
            "detector; and the three parts of its position-model error each night, with the cameras' "
            "median of each dashed. A night's value moves with the guide stars it had, a step by 1-2 µm: "
            "read the level, not a night."
        ),
    }


def _nightAxis(ax: Axes, nights: pd.DatetimeIndex, label: bool = True, rotate: bool = False) -> None:
    """Label ``ax``'s x axis with the nights and separate them with lines."""
    ax.set_xlim(0, len(nights))
    ax.set_xticks(np.arange(len(nights)) + 0.5)
    ax.set_xticklabels(
        [night.strftime("%m-%d") for night in nights], rotation=90 if rotate else 0, fontsize=7
    )
    ax.tick_params(labelbottom=label)
    for i in range(1, len(nights)):
        ax.axvline(i, color=EMPTY, lw=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    if label and not rotate:
        ax.set_xlabel("night (HST date of the evening; each spans 18:00 to 06:00 HST)")


def _legend(ax: Axes, above: bool = False, **kwargs) -> None:
    """Draw a compact legend, inside ``ax`` or above its top right corner."""
    if above:
        kwargs.update(loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.legend(
        scatterpoints=1,
        handlelength=1.2,
        handletextpad=0.5,
        borderaxespad=0.2,
        fontsize=7,
        frameon=False,
        **kwargs,
    )


def _shortNumber(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:.2g}"


def _figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height))
    fig.set_layout_engine("tight")
    return fig


def plotCameraPresence(run: AgRun) -> Figure:
    """Map, per camera and hour, what the camera gave when it was expected to match a guide star.

    Only exposures in which some camera matched a star count
    (`runAnalysis.stateShares`). A cell is shaded by the share of its
    exposures in which the camera did not match, in the color of what it
    mostly did instead: nothing detected (red), detections but no match
    (amber) or left out by the guider (blue). It is hatched when the
    camera's design had fewer than ``RunConfig.minDesignStars`` bright
    guide stars there (no match is then no surprise). A camera not in use
    has a hatched row saying so.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        One row per camera.
    """
    nights = run.plottedNights
    shares = stateShares(run)
    index = {night: i for i, night in enumerate(nights)}
    colors = {"dark": BAD, "unmatched": WARN, "disabled": OFF}
    cmaps = {name: LinearSegmentedColormap.from_list(name, [EMPTY, color]) for name, color in colors.items()}
    fig = _figure(11, 3.3)
    ax = fig.add_subplot()
    rowY = {cid: row + (0.5 if row >= 3 else 0) for row, cid in enumerate(CAMERA_ORDER)}
    for cid, y in rowY.items():
        if cid not in run.inUse:
            ax.add_patch(
                Rectangle((0, y), len(nights), 0.9, facecolor="none", edgecolor=QUIET, hatch="....", lw=0)
            )
            ax.text(
                len(nights) / 2,
                y + 0.45,
                "not in use: no detections in the run",
                ha="center",
                va="center",
                fontsize=7.5,
                color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            )
            continue
        for _, cell in shares[shares.agc_camera_id == cid].iterrows():
            x = index[cell.night] + 0.05 + 0.9 * cell.slot / 12
            judged = cell.n - cell.few_stars
            if judged == 0:
                ax.add_patch(
                    Rectangle((x, y), 0.9 / 12, 0.9, facecolor="none", edgecolor=QUIET, hatch="////", lw=0)
                )
                continue
            worst = max(colors, key=lambda name: cell[name])
            ax.add_patch(
                Rectangle((x, y), 0.9 / 12, 0.9, color=cmaps[worst](1 - cell.matched / judged), lw=0)
            )
    ax.set_ylim(len(CAMERA_ORDER) + 0.5, 0)
    ax.set_yticks([y + 0.45 for y in rowY.values()])
    ax.set_yticklabels([f"AG{cid + 1}" for cid in rowY])
    for name, y in (("AG1/3/5", 1.5), ("AG2/4/6", 5.0)):
        ax.annotate(
            name,
            xy=(0, y),
            xycoords=("axes fraction", "data"),
            xytext=(-34, 0),
            textcoords="offset points",
            rotation=90,
            va="center",
            ha="center",
            fontsize=8,
            color=INK,
        )
    _nightAxis(ax, nights)
    ax.grid(False)
    handles = [
        Rectangle((0, 0), 1, 1, color=EMPTY, label="matched a star"),
        Rectangle((0, 0), 1, 1, color=BAD, label="no detections"),
        Rectangle((0, 0), 1, 1, color=WARN, label="detections, no match"),
        Rectangle((0, 0), 1, 1, color=OFF, label="disabled"),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="none",
            edgecolor=QUIET,
            hatch="////",
            lw=0,
            label=f"under {run.config.minDesignStars} bright guide stars designed",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1, -0.2),
        ncol=5,
        fontsize=7,
        frameon=False,
        handlelength=1.2,
        borderaxespad=0.2,
    )
    ax.set_title(
        "Camera states per hour: the share of AG exposures (with a star matched on some camera) in which "
        "this camera matched none, in the color of what it mostly did instead",
        loc="left",
        fontsize=8.5,
    )
    return fig


def plotCamera(run: AgRun, cid: int, clusters: pd.DataFrame, modelErrors: pd.DataFrame) -> Figure:
    """Plot one camera over the run, night by night: stars, matches, flags, rejected matches and its position model.

    What is better seen for the six cameras together (where on a detector
    matches are rejected, focus, image size) is in `plotFocalPlane`.

    Parameters
    ----------
    run : `AgRun`
        The run.
    cid : `int`
        The camera, ``agc_camera_id`` (AG1 is 0).
    clusters : `pandas.DataFrame`
        From `runAnalysis.findRejectionClusters`.
    modelErrors : `pandas.DataFrame`
        From `runAnalysis.fitModelErrors`.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Six panels (`CAMERA_FIGURE_IN`), five against the nights: matched
        stars per visit; the share of judged AG exposures in which it
        matched a star; the share of its designed guide stars matched, by
        magnitude; the share of its detections carrying each bad-detection
        flag; AG exposures with its rejected matches clustered on one side
        of the detector; and the parts of its position-model error.
    """
    nights = run.plottedNights
    position = {night: i + 0.5 for i, night in enumerate(nights)}
    frames = run.frames[run.frames.night.isin(nights)].assign(x=lambda df: nightPosition(df.taken_at, nights))
    fig = _figure(*CAMERA_FIGURE_IN)
    axes = fig.subplots(2, 3)

    ax = axes[0, 0]
    visits = frames.groupby(["pfs_visit_id", "population"]).agg(
        x=("x", "median"), good=(f"n_good{cid}", "median")
    )
    visits = visits.reset_index()
    science = visits.population.eq("science")
    ax.scatter(visits.x[~science], visits.good[~science], s=7, color=QUIET, lw=0, label="other visits")
    ax.scatter(visits.x[science], visits.good[science], s=7, color="C0", lw=0, label="science visits")
    ax.set_ylim(bottom=-0.5)
    ax.set_title("Matched stars per AG exposure (visit median)", loc="left", fontsize=8.5)
    _legend(ax, loc="upper right", markerscale=2)

    ax = axes[0, 1]
    shares = stateShares(run)
    shares = shares[shares.agc_camera_id == cid].groupby("night")[["n", "matched", "few_stars"]].sum()
    judged = shares.n - shares.few_stars
    share = (100 * shares.matched / judged).where(judged > 0)
    x = np.array([position[n] for n in share.index])
    ax.bar(x, share.fillna(0), width=0.6, color=np.where(share < 90, BAD, INK))
    ax.set_ylim(0, 125)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("%")
    ax.set_title("AG exposures in which it matched a star (%)", loc="left", fontsize=8.5)
    _legend(
        ax,
        loc="upper left",
        ncol=2,
        handles=[Patch(color=INK, label="90% or more"), Patch(color=BAD, label="under 90%")],
    )

    ax = axes[0, 2]
    magnitudes = magnitudeSummary(run)
    if len(magnitudes):
        everyCamera = magnitudes.groupby("magnitude")[["n_expected", "n_good"]].sum()
        ax.plot(
            everyCamera.index + 0.25,
            100 * everyCamera.n_good / everyCamera.n_expected,
            ":",
            color=INK,
            lw=1,
            label="all cameras",
        )
        own = magnitudes[magnitudes.agc_camera_id == cid]
        ax.plot(
            own.magnitude + 0.25, 100 * own.fraction, "-o", ms=3.5, lw=1.5, color="C0", label="this camera"
        )
        _legend(ax, loc="lower left")
    ax.set_ylim(0, 105)
    ax.set_xlabel("guide-star magnitude", fontsize=8)
    ax.set_title("Designed guide stars matched (%)", loc="left", fontsize=8.5)

    ax = axes[1, 0]
    flagged = detectionFlagShares(run)
    flagged = flagged[flagged.agc_camera_id == cid]
    for name, color in zip(DETECTION_FLAG_COLUMNS.values(), ("C0", BAD, "C2", "C4", "C5"), strict=True):
        if len(flagged):
            ax.plot(
                [position[n] for n in flagged.night],
                flagged[name],
                "-o",
                ms=3,
                lw=1.2,
                color=color,
                label=name,
            )
    ax.set_ylim(0, max(5, 1.7 * ax.get_ylim()[1]))
    ax.set_ylabel("%")
    ax.set_title("Detections flagged by ics_agccActor (%)", loc="left", fontsize=8.5)
    if len(flagged):
        _legend(ax, loc="upper left", ncol=2)

    ax = axes[1, 1]
    mine = clusters[(clusters.agc_camera_id == cid) & clusters.night.isin(nights)]
    for dx, chosen, color, label in (
        (-0.15, mine.alone.astype(bool), BAD, "on it alone"),
        (0.15, ~mine.alone.astype(bool), WARN, "with other cameras"),
    ):
        counts = mine[chosen].groupby("night").size().reindex(nights, fill_value=0)
        ax.bar(np.arange(len(nights)) + 0.5 + dx, counts.to_numpy(), width=0.28, color=color, label=label)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, max(5, ax.get_ylim()[1]))
    alone = mine[mine.alone.astype(bool)]
    direction = f" (alone: toward {_meanAngle(alone.angle_deg):.0f}°)" if len(alone) else ""
    ax.set_title(f"Rejected matches clustered{direction}", loc="left", fontsize=8.5)
    _legend(ax, loc="upper left")

    ax = axes[1, 2]
    errors = modelErrors[(modelErrors.agc_camera_id == cid) & modelErrors.night.isin(nights)]
    typical = summariseModelErrors(modelErrors)
    for (column, label), color in zip(MODEL_PARTS.items(), ("C0", "C1", "C2"), strict=True):
        ax.plot(
            [position[n] for n in errors.night],
            errors[column],
            "-o",
            ms=3.5,
            lw=1.5,
            color=color,
            label=label,
        )
        if len(typical):
            ax.axhline(typical[column].median(), color=color, lw=0.8, ls="--")
    ax.axhline(0, color=QUIET, lw=0.8)
    limit = max(10, 1.15 * np.nanmax(np.abs(errors[list(MODEL_PARTS)].to_numpy(float)), initial=0))
    ax.set_ylim(-limit, 1.35 * limit)
    radius = run.config.modelErrorRadiusPix
    ax.set_title(
        f"Position error (µm; slopes at {radius:g} px; dashed: all cameras)", loc="left", fontsize=8.5
    )
    _legend(ax, loc="upper left", ncol=2)
    pixels = ax.secondary_yaxis("right", functions=(lambda um: um / AG_PIXEL_UM, lambda px: px * AG_PIXEL_UM))
    pixels.set_ylabel("AG pixels", fontsize=7)
    pixels.tick_params(labelsize=7)

    for row, col in ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)):
        _nightAxis(axes[row, col], nights, label=(row == 1), rotate=True)
    group = next(name for name, cids in CAMERA_GROUPS.items() if cid in cids)
    unused = "" if cid in run.inUse else ": not in use (no detections in the run)"
    fig.suptitle(f"AG{cid + 1} ({group}){unused}", x=0.01, ha="left", fontsize=10, fontweight="bold")
    return fig


def _onFocalPlane(cid: int, x, y, zoom: float) -> tuple[np.ndarray, np.ndarray]:
    """Return where detector pixels of camera ``cid`` are drawn on the PFI (mm, hardware coordinates).

    The detector is enlarged ``zoom`` times about its center, where it
    stays: at its true size it would be a dot on the PFI.
    """
    centerX, centerY = det2dp(cid, AG_RIGHT_EDGE_PIX, _DETECTOR_PIX[1] / 2)
    px, py = det2dp(cid, np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return centerX + zoom * (px - centerX), centerY + zoom * (py - centerY)


def _focalPlaneAxes(ax: Axes, title: str, extent: float) -> None:
    ax.plot([0], [0], "+", color=INK, ms=10)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)
    ax.set_title(title, loc="left", fontsize=9)


def _outline(ax: Axes, cid: int, x0: float, x1: float, y0: float, y1: float, zoom: float, **kwargs) -> None:
    corners = np.column_stack(_onFocalPlane(cid, [x0, x1, x1, x0], [y0, y0, y1, y1], zoom))
    ax.add_patch(Polygon(corners, closed=True, **kwargs))


def plotFocalPlane(run: AgRun, modelErrors: pd.DataFrame) -> Figure:
    """Draw the six detectors where they sit on the PFI, with what is better compared across them.

    Parameters
    ----------
    run : `AgRun`
        The run.
    modelErrors : `pandas.DataFrame`
        From `runAnalysis.fitModelErrors`.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Three panels (`FOCAL_PLANE_FIGURE_IN`), each with every detector
        enlarged at its place on the PFI (hardware coordinates, the field's
        center marked), the boundary between its halves dashed and its
        pixel (0, 0) dotted: the share of paired detections rejected in each
        cell with the residual left there (`runAnalysis.detectorCells`) and
        the three parts of the camera's position-model error; each
        camera's focus minus the median of the cameras in use; and the FWHM
        of its matched stars on each half. Run medians, over the plotted
        nights.
    """
    nights = run.plottedNights
    frames = run.frames[run.frames.night.isin(nights)]
    cameras = run.cameras[run.cameras.night.isin(nights)]
    fig = _figure(*FOCAL_PLANE_FIGURE_IN)
    grid = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    xEdges = AG_GRID_CELL_PIX["columnCell"] * np.arange(5.0)
    yEdges = AG_GRID_CELL_PIX["rowCell"] * np.arange(5.0)
    width, height = _DETECTOR_PIX

    def decorate(ax: Axes, zoom: float, label: Callable[[int], str], distance: float) -> None:
        for cid in range(6):
            ax.plot(
                *_onFocalPlane(cid, [AG_RIGHT_EDGE_PIX] * 2, [0, height], zoom), color=INK, lw=0.9, ls="--"
            )
            ax.plot(*_onFocalPlane(cid, [0], [0], zoom), "o", ms=3, color=INK)
            x, y = _onFocalPlane(cid, [AG_RIGHT_EDGE_PIX], [height / 2], 1)
            ax.text(
                distance * x[0],
                distance * y[0],
                label(cid),
                ha="center",
                va="center",
                fontsize=8,
                linespacing=1.3,
            )

    # Where matches are rejected, and the residual left there.
    ax = fig.add_subplot(grid[:, 0])
    zoom = 9.0
    cells = detectorCells(run)
    parts = summariseModelErrors(modelErrors)
    rejectedMap = LinearSegmentedColormap.from_list("rejected", ["#ffffff", WARN, BAD])
    norm = Normalize(0, REJECTED_MAX)
    arrows = []
    for cid in range(6):
        mine = cells[cells.agc_camera_id == cid].set_index(["row_cell", "column_cell"])
        for row in range(4):
            for column in range(4):
                share = mine.rejected.get((row, column), np.nan) if cid in run.inUse else np.nan
                _outline(
                    ax,
                    cid,
                    xEdges[column],
                    xEdges[column + 1],
                    yEdges[row],
                    yEdges[row + 1],
                    zoom,
                    facecolor=rejectedMap(norm(share)) if np.isfinite(share) else EMPTY,
                    edgecolor="white",
                    lw=0.6,
                )
        enough = (
            mine[(mine.n_good >= MIN_CELL_MATCHES) & mine.dx_um.notna()]
            if cid in run.inUse
            else mine.iloc[:0]
        )
        for (row, column), cell in enough.iterrows():
            x, y = (xEdges[column] + xEdges[column + 1]) / 2, (yEdges[row] + yEdges[row + 1]) / 2
            # Drawn on the detector, then placed with it: the arrow turns with the detector.
            tail = _onFocalPlane(cid, [x], [y], zoom)
            head = _onFocalPlane(
                cid, [x + ARROW_PIX_PER_UM * cell.dx_um], [y + ARROW_PIX_PER_UM * cell.dy_um], zoom
            )
            arrows.append((tail[0][0], tail[1][0], head[0][0] - tail[0][0], head[1][0] - tail[1][0]))

    def partsLabel(cid: int) -> str:
        name = f"AG{cid + 1}"
        if cid not in run.inUse:
            return f"{name}\nnot in use"
        if cid not in parts.index:
            return name
        row = parts.loc[cid]
        return f"{name}\nx {row.tangential_um:+.1f}  y {row.radial_um:+.1f}\nstep {row.step_um:+.1f} µm"

    key = zoom * AG_PIXEL_UM / 1000 * ARROW_PIX_PER_UM * 5  # the length of a 5 µm arrow (mm on the PFI)
    for x, y, u, v in [*arrows, (-key / 2, -38, key, 0)]:
        ax.annotate(
            "",
            (x + u, y + v),
            (x, y),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#1b1b1a",
                "lw": 1.2,
                "mutation_scale": 8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    ax.text(0, -62, "residual, 5 µm", ha="center", fontsize=8)
    ax.text(0, 18, "field center", ha="center", fontsize=8, color=INK)
    decorate(ax, zoom, partsLabel, 1.52)
    _focalPlaneAxes(
        ax,
        "Paired detections rejected (%), and the residual left: x and y slopes at "
        f"{run.config.modelErrorRadiusPix:g} px, step between the halves",
        420,
    )
    bar = fig.colorbar(ScalarMappable(norm, rejectedMap), ax=ax, fraction=0.03, pad=0.01, extend="max")
    bar.set_label("paired detections rejected (%)", fontsize=8)
    bar.ax.tick_params(labelsize=7)

    # Focus: each camera against the median of the cameras in use.
    ax = fig.add_subplot(grid[0, 1])
    zoom = 12.0
    z = cameraFocus(run, frames)
    focus = (z.sub(z.median(axis=1), axis=0)).median() if len(z) else pd.Series(dtype=float)
    focusNorm = Normalize(-FOCUS_MAX, FOCUS_MAX)
    for cid in range(6):
        value = focus.get(cid, np.nan) if cid in run.inUse else np.nan
        _outline(
            ax,
            cid,
            0,
            width,
            0,
            height,
            zoom,
            facecolor=FOCUS_CMAP(focusNorm(value)) if np.isfinite(value) else EMPTY,
            edgecolor=INK,
            lw=0.6,
        )
    decorate(
        ax,
        zoom,
        lambda cid: f"AG{cid + 1}\n"
        + (f"{focus[cid]:+.0f}" if cid in run.inUse and cid in focus else "not in use"),
        1.7,
    )
    _focalPlaneAxes(ax, "Focus minus the cameras' median (µm)", 500)
    bar = fig.colorbar(ScalarMappable(focusNorm, FOCUS_CMAP), ax=ax, fraction=0.04, pad=0.02, extend="both")
    bar.ax.tick_params(labelsize=7)

    # Image size on each half.
    ax = fig.add_subplot(grid[1, 1])
    fwhm = cameras.groupby(["agc_camera_id", "right_half"]).fwhm.median()
    finite = fwhm[np.isfinite(fwhm)]
    fwhmNorm = Normalize(*((finite.min() - 0.02, finite.max() + 0.02) if len(finite) else (0, 1)))
    for cid in range(6):
        for half, (x0, x1) in enumerate(((0, AG_RIGHT_EDGE_PIX), (AG_RIGHT_EDGE_PIX, width))):
            value = fwhm.get((cid, half), np.nan) if cid in run.inUse else np.nan
            _outline(
                ax,
                cid,
                x0,
                x1,
                0,
                height,
                zoom,
                facecolor=FWHM_CMAP(fwhmNorm(value)) if np.isfinite(value) else EMPTY,
                edgecolor=INK,
                lw=0.6,
            )

    def fwhmLabel(cid: int) -> str:
        if cid not in run.inUse:
            return f"AG{cid + 1}\nnot in use"
        return f"AG{cid + 1}\n{fwhm.get((cid, 0), np.nan):.2f} | {fwhm.get((cid, 1), np.nan):.2f}"

    decorate(ax, zoom, fwhmLabel, 1.82)
    _focalPlaneAxes(ax, "FWHM of matched stars (″): without RIGHT | RIGHT half", 530)
    bar = fig.colorbar(ScalarMappable(fwhmNorm, FWHM_CMAP), ax=ax, fraction=0.04, pad=0.02)
    bar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "The six detectors on the PFI, enlarged where they sit (dashed: the boundary between a detector's halves; "
        "dot: its pixel (0, 0))",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    return fig


def _scatterByCameras(ax: Axes, x, y, full) -> None:
    ax.scatter(x[full], y[full], s=10, color=INK, lw=0, label="stars matched on every camera in use")
    ax.scatter(x[~full], y[~full], s=16, color=BAD, marker="x", lw=1, label="on fewer")


def plotGuiding(run: AgRun, jitter: pd.DataFrame) -> Figure:
    """Plot each science visit's guide error, its size and its jitter, and each night's guide-loop problems.

    Parameters
    ----------
    run : `AgRun`
        The run.
    jitter : `pandas.DataFrame`
        From `runAnalysis.visitGuideErrors`.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        For ΔRA (on the sky), ΔDec and ΔInR, the RMS of each visit's
        guide offsets (left) and their frame-to-frame jitter (right), a
        visit with stars matched on fewer cameras than are in use marked;
        and per night the AG exposures whose offset was rejected or not
        written.
    """
    nights = run.plottedNights
    jitter = jitter[jitter.night.isin(nights)].assign(x=lambda df: nightPosition(df.taken_at, nights))
    full = jitter.full.astype(bool)
    fig = _figure(14.5, 8.5)
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.8])
    first = None
    for row, (axis, name) in enumerate((("ra", "ΔRA on the sky"), ("dec", "ΔDec"), ("inr", "ΔInR"))):
        for col, (column, what, unit) in enumerate(
            ((f"{axis}_size", "size", "RMS offset (″)"), (axis, "jitter", "frame-to-frame jitter (″)"))
        ):
            ax = fig.add_subplot(grid[row, col])
            first = first or ax
            values = jitter[column]
            _scatterByCameras(ax, jitter.x, values, full)
            median = values.median()
            ax.axhline(median, color=INK, lw=0.8, ls="--")
            top = np.nanpercentile(values, 99) * 1.15 if values.notna().any() else np.nan
            top = top if np.isfinite(top) and top > 0 else 1.0
            ax.set_ylim(0, top)
            for _, visit in jitter[values > top].iterrows():
                ax.annotate(
                    _shortNumber(visit[column]), (visit.x, top), fontsize=6, color=BAD, ha="center", va="top"
                )
            ax.set_ylabel(unit)
            ax.set_title(f"{name}, {what} (dashed: median, {median:.3g}″)", loc="left", fontsize=9)
            _nightAxis(ax, nights, label=False)
    _legend(first, above=True, ncol=2)

    ax = fig.add_subplot(grid[3, :])
    frames = run.frames[run.frames.night.isin(nights)]
    perNight = frames.groupby("night")["mask"].agg(
        rejected=lambda mask: int(isRejectedOffset(mask).sum()),
        missing=lambda mask: int(mask.isna().sum()),
    )
    x = np.array([list(nights).index(night) + 0.5 for night in perNight.index])
    for dx, column, color, label in (
        (-0.15, "rejected", BAD, "rejected (INVALID_OFFSET)"),
        (0.15, "missing", WARN, "no guide offset written"),
    ):
        ax.bar(x + dx, perNight[column], width=0.28, color=color, label=label)
        for xx, value in zip(x, perNight[column], strict=True):
            if value:
                ax.text(xx + dx, value, str(value), ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("AG exposures")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Guide-loop problems per night", loc="left", fontsize=9)
    ax.set_ylim(0, max(5, perNight.to_numpy().max(initial=0) * 1.3))
    _legend(ax, above=True, ncol=2)
    _nightAxis(ax, nights)
    return fig


def plotFocus(run: AgRun) -> Figure:
    """Plot the guider's focus error and the telescope's focus offset over the run.

    Each camera's focus against the others' is in `plotCamera`.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        dZ per science visit, and ``m2_off3``.
    """
    nights = run.plottedNights
    frames = run.frames[run.frames.night.isin(nights)]
    fig = _figure(11, 4.6)
    grid = fig.add_gridspec(2, 1)

    ax = fig.add_subplot(grid[0])
    visits = (
        frames[frames.population.eq("science")]
        .groupby("pfs_visit_id")
        .agg(taken_at=("taken_at", "median"), dz=("guide_delta_z", "median"), n_cams=("n_cams", "median"))
    )
    visits["dz"] *= 1e3
    _scatterByCameras(ax, nightPosition(visits.taken_at, nights), visits.dz, visits.n_cams >= len(run.inUse))
    ax.axhline(0, color=QUIET, lw=0.8)
    offScale = 0
    if visits.dz.notna().any():
        low, high = np.nanpercentile(visits.dz, [2, 98])
        pad = 0.3 * max(high - low, 1)
        ax.set_ylim(low - pad, high + pad)
        offScale = int(((visits.dz < low - pad) | (visits.dz > high + pad)).sum())
    ax.set_ylabel("µm")
    ax.set_title(
        "Guider focus error dZ (median of the cameras in use), per science visit"
        + (f"; {offScale} visits off scale" if offScale else ""),
        loc="left",
        fontsize=9,
    )
    _legend(ax, above=True, ncol=2)
    _nightAxis(ax, nights, label=False)

    ax = fig.add_subplot(grid[1])
    focus = run.focus[run.focus.night.isin(nights) & (run.focus.m2_off3 > 0)]
    ax.step(nightPosition(focus.created_at, nights), focus.m2_off3, where="post", color="C2", lw=1.2)
    ax.set_ylabel("mm")
    ax.set_title(
        "Telescope focus offset (tel_status.m2_off3); the spikes at the start of a night are AG focus sweeps",
        loc="left",
        fontsize=9,
    )
    _nightAxis(ax, nights)
    return fig


def plotAcquisition(run: AgRun) -> Figure:
    """Plot the first guide offset of each convergence visit: the pointing error the guider saw.

    Parameters
    ----------
    run : `AgRun`
        The run.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        ΔRA (on the sky) and ΔDec against time, with the guider's
        ``max_correction``.
    """
    nights = run.plottedNights
    frames = run.frames[
        run.frames.night.isin(nights) & run.frames.population.eq("convergence") & run.frames["mask"].notna()
    ]
    first = frames.groupby("pfs_visit_id").head(1)
    x = nightPosition(first.taken_at, nights)
    fig = _figure(11, 2.8)
    ax = fig.add_subplot()
    ax.scatter(x, onSkyRa(first), s=10, color="C0", lw=0, label="ΔRA on the sky")
    ax.scatter(x, first.guide_delta_dec, s=10, color="C1", lw=0, label="ΔDec")
    limit = run.config.maxCorrection
    for y in (-limit, limit):
        ax.axhline(y, color=BAD, lw=0.8, ls="--")
    ax.set_ylabel("arcsec")
    ax.set_title(
        f"Acquisition: first guide offset of each convergence visit (dashed: ±{limit:g}″ max_correction, "
        "applied to the coordinate ΔRA)",
        loc="left",
        fontsize=9,
    )
    _legend(ax, above=True, ncol=2)
    _nightAxis(ax, nights)
    return fig
