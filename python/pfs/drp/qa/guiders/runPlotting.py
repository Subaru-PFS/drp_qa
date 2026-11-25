"""Plots of a whole run's AG data, for the end-of-run report.

Each function takes an `pfs.drp.qa.guiders.runAnalysis.AgRun` (and, for the
guiding plot, its settled jitter) and returns a new
`matplotlib.figure.Figure`, made without pyplot so that it is never shown or
left open. Nothing here touches a database.

The x axis is a strip of nights (`runAnalysis.nightPosition`): one slot per
night with at least ``RunConfig.minFramesNight`` AG exposures, labelled by
its evening's date and spanning 18:00 to 06:00 HST. Per-camera panels are in
two rows, AG1/3/5 above AG2/4/6, the two groups being wired and placed
separately. Problems are drawn in red, so that they stand out.

`captions` says what each figure shows, for whatever page it is put on.
"""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from pfs.drp.qa.guiders.runAnalysis import (
    CAMERA_GROUPS,
    DETECTION_FLAG_COLUMNS,
    AgRun,
    RunConfig,
    _meanAngle,
    cameraFocus,
    detectionFlagShares,
    isRejectedOffset,
    magnitudeSummary,
    nightPosition,
    onSkyRa,
    stateShares,
    summariseModelErrors,
)

__all__ = [
    "AG_PIXEL_UM",
    "CAMERA_ORDER",
    "GROUPS_NOTE",
    "captions",
    "plotAcquisition",
    "plotCamera",
    "plotCameraPresence",
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
        The parameters the run was analysed with, for the numbers quoted.

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
            "a guide star (so the sky was usable). A cell is grey when the camera matched a star in every "
            "exposure of that hour, and coloured by the share in which it did not: red when it mostly "
            "detected nothing at all, amber when it detected spots but matched none of them to a guide "
            "star, blue when the guider had left it out of its fit (disabled). The opdb records a camera's "
            "detections, not whether it returned an image, so a dead camera and an image with nothing in it "
            "both read as no detections: a camera that is red for hours while the others match stars is "
            f"dead or not being read out. Hatched: the design put fewer than {config.minDesignStars} bright "
            "guide stars on the camera, so no match is expected. A camera with no detections in the whole "
            "run is marked not in use."
        ),
        "guiding": (
            "One point per science visit, from its settled guide offsets: those the guider accepted after "
            f"the visit's first {config.nSettle} guide frames, for visits with at least {config.minSettled} "
            "of them. A guide offset is the pointing error the guider measured in one AG exposure (ΔRA on "
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
        "camera": (
            "One camera over the run, night by night. Top row: the good matches per AG exposure, as the "
            "median of each visit; the FWHM of its matched stars on each half of its detector (the two "
            "halves focus differently by design: one has glass in front of it); its focus against the "
            "median of the cameras in use; and the share of its designed guide stars that were matched, by "
            "their magnitude (science visits; the dotted line is all cameras). Bottom row: the share of AG "
            "exposures in which it matched a star; the AG exposures in which its rejected matches clustered "
            "on one side of its detector, which is what a position model that is off does; the size of that "
            "position-model error at the edge of the detector; and the share of its detections that "
            "ics_agccActor flagged as bad, by flag."
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
    exposures in which the camera did not match, in the colour of what it
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
    colours = {"dark": BAD, "unmatched": WARN, "disabled": OFF}
    cmaps = {
        name: LinearSegmentedColormap.from_list(name, [EMPTY, colour]) for name, colour in colours.items()
    }
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
            worst = max(colours, key=lambda name: cell[name])
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
        "this camera matched none, in the colour of what it mostly did instead",
        loc="left",
        fontsize=8.5,
    )
    return fig


def plotCamera(run: AgRun, cid: int, clusters: pd.DataFrame, modelErrors: pd.DataFrame) -> Figure:
    """Plot one camera over the run: stars, image size, focus, presence, rejected matches and its position model.

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
        Eight panels, all but one against the nights: matched stars per
        visit; FWHM per detector half; focus minus the median of the
        cameras in use; the share of its designed guide stars matched, by
        magnitude; the share of judged AG exposures in which it matched a
        star; AG exposures with its rejected matches clustered on one side
        of the detector; its position-model error; and the share of its
        detections carrying each bad-detection flag.
    """
    nights = run.plottedNights
    position = {night: i + 0.5 for i, night in enumerate(nights)}
    frames = run.frames[run.frames.night.isin(nights)].assign(x=lambda df: nightPosition(df.taken_at, nights))
    fig = _figure(14.5, 6.2)
    axes = fig.subplots(2, 4)

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
    cameras = run.cameras[run.cameras.night.isin(nights)]
    for half, color, name in ((0, "C0", "glass half"), (1, "C1", "no-glass half")):
        values = (
            cameras[(cameras.agc_camera_id == cid) & (cameras.right_half == half)]
            .groupby("night")
            .fwhm.median()
        )
        ax.plot(
            [position[n] for n in values.index],
            values.to_numpy(),
            "-o",
            ms=3.5,
            lw=1.5,
            color=color,
            label=name,
        )
    allCameras = cameras.groupby("night").fwhm.median()
    ax.plot(
        [position[n] for n in allCameras.index],
        allCameras.to_numpy(),
        ":",
        color=INK,
        lw=1,
        label="all cameras",
    )
    ax.set_ylim(0, max(1.0, np.nanpercentile(cameras.fwhm, 99)) if cameras.fwhm.notna().any() else 1.0)
    ax.set_title("FWHM of matched stars, nightly median (″)", loc="left", fontsize=8.5)
    _legend(ax, loc="lower left", ncol=3)

    ax = axes[0, 2]
    z = cameraFocus(run, frames)
    if len(z) and cid in z:
        byNight = (z[cid] - z.median(axis=1)).groupby(frames.night.loc[z.index]).median()
        ax.plot([position[n] for n in byNight.index], byNight.to_numpy(), "-o", ms=3.5, lw=1.5, color="C0")
    ax.axhline(0, color=QUIET, lw=0.8)
    ax.set_ylim(-60, 60)
    ax.set_title("Focus minus the cameras' median (µm), nightly", loc="left", fontsize=8.5)

    ax = axes[0, 3]
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
    ax.set_title("Designed guide stars matched (%), science visits", loc="left", fontsize=8.5)

    ax = axes[1, 0]
    shares = stateShares(run)
    shares = shares[shares.agc_camera_id == cid].groupby("night")[["n", "matched", "few_stars"]].sum()
    judged = shares.n - shares.few_stars
    share = (100 * shares.matched / judged).where(judged > 0)
    x = np.array([position[n] for n in share.index])
    ax.bar(x, share.fillna(0), width=0.6, color=np.where(share < 90, BAD, INK))
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("AG exposures in which it matched a star (red: under 90%)", loc="left", fontsize=8.5)

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
    ax.plot(
        [position[n] for n in errors.night],
        errors.error_um,
        "-o",
        ms=3.5,
        lw=1.5,
        color="C0",
        label="this camera",
    )
    if len(modelErrors):
        typical = modelErrors.groupby("agc_camera_id").error_um.median().median()
        ax.axhline(typical, color=QUIET, lw=0.8, ls="--", label="all cameras' median")
    ax.set_ylim(0, max(12, np.nanmax(errors.error_um.to_numpy(float), initial=0) * 1.1))
    if len(errors):
        angle = summariseModelErrors(run, errors).angle_deg.iloc[0]
        ax.text(
            0.98,
            0.04,
            f"growing toward {angle:.0f}° (mod 180)" if np.isfinite(angle) else "no one direction",
            transform=ax.transAxes,
            ha="right",
            fontsize=7,
        )
    radius = run.config.modelErrorRadiusPix
    ax.set_title(f"Model error {radius:g} px from centre (µm)", loc="left", fontsize=8.5)
    _legend(ax, loc="upper left")
    pixels = ax.secondary_yaxis("right", functions=(lambda um: um / AG_PIXEL_UM, lambda px: px * AG_PIXEL_UM))
    pixels.set_ylabel("AG pixels", fontsize=7)
    pixels.tick_params(labelsize=7)

    ax = axes[1, 3]
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
    ax.set_ylim(0, max(5, 1.5 * ax.get_ylim()[1]))
    ax.set_ylabel("%")
    ax.set_title("Detections flagged by ics_agccActor (%)", loc="left", fontsize=8.5)
    if len(flagged):
        _legend(ax, loc="upper left", ncol=3)

    for row in range(2):
        for col in range(4):
            if (row, col) != (0, 3):
                _nightAxis(axes[row, col], nights, label=(row == 1), rotate=True)
    group = next(name for name, cids in CAMERA_GROUPS.items() if cid in cids)
    unused = "" if cid in run.inUse else ": not in use (no detections in the run)"
    fig.suptitle(f"AG{cid + 1} ({group}){unused}", x=0.01, ha="left", fontsize=10, fontweight="bold")
    return fig


def _scatterByCameras(ax: Axes, x, y, full) -> None:
    ax.scatter(x[full], y[full], s=10, color=INK, lw=0, label="stars matched on every camera in use")
    ax.scatter(x[~full], y[~full], s=16, color=BAD, marker="x", lw=1, label="on fewer")


def plotGuiding(run: AgRun, jitter: pd.DataFrame) -> Figure:
    """Plot each science visit's settled guide error, its size and its jitter, and each night's guide-loop problems.

    Parameters
    ----------
    run : `AgRun`
        The run.
    jitter : `pandas.DataFrame`
        From `runAnalysis.settledJitter`.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        For ΔRA (on the sky), ΔDec and ΔInR, the RMS of each visit's
        settled offsets (left) and their frame-to-frame jitter (right), a
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
