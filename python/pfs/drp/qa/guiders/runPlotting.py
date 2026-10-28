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
"""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from pfs.drp.qa.guiders.runAnalysis import (
    CAMERA_GROUPS,
    AgRun,
    CameraState,
    _meanAngle,
    nightPosition,
    onSkyRa,
)

__all__ = [
    "plotAcquisition",
    "plotCamera",
    "plotCameraPresence",
    "plotFocus",
    "plotGuiding",
]

#: The cameras in the order of the per-camera panels: AG1/3/5, then AG2/4/6.
CAMERA_ORDER = [*CAMERA_GROUPS["AG1/3/5"], *CAMERA_GROUPS["AG2/4/6"]]

BAD = "#c8102e"  # a fault
WARN = "#eda100"  # worth a look
QUIET = "#b5b4ae"  # context: other populations, reference lines
INK = "#52514e"  # the normal case
EMPTY = "#ecebe8"  # an empty heatmap cell


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
    """Map, per camera and hour, the share of AG exposures in which it matched no guide star.

    Only exposures in which some camera matched a star count. A cell is
    hatched when the camera's design had fewer than
    ``RunConfig.minDesignStars`` bright guide stars there (no match is then
    no surprise), and crossed when most of its misses detected nothing at
    all, as a dead camera does.

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
    frames = run.frames[run.frames.night.isin(nights) & (run.frames.n_cams > 0)].copy()
    frames["slot"] = np.floor(nightPosition(frames.taken_at, nights) * 12) / 12
    cmap = LinearSegmentedColormap.from_list("missing", [EMPTY, BAD])
    fig = _figure(11, 3.1)
    ax = fig.add_subplot()
    rowY = {cid: row + (0.5 if row >= 3 else 0) for row, cid in enumerate(CAMERA_ORDER)}
    for cid, y in rowY.items():
        for slot, states in frames.groupby("slot")[f"state{cid}"]:
            judged = states[states != CameraState.FEW_STARS]
            if len(judged) == 0:
                ax.add_patch(
                    Rectangle((slot, y), 1 / 12, 0.9, facecolor="none", edgecolor=QUIET, hatch="////", lw=0)
                )
                continue
            ax.add_patch(
                Rectangle((slot, y), 1 / 12, 0.9, color=cmap((judged != CameraState.OK).mean()), lw=0)
            )
            misses = judged[judged != CameraState.OK]
            if len(misses) and (misses == CameraState.DARK).mean() > 0.5:
                ax.plot(slot + 1 / 24, y + 0.45, marker="x", ms=3.5, color="white", mew=0.9)
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
    colorbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=Normalize(0, 1)), ax=ax, pad=0.01, fraction=0.03)
    colorbar.set_label("exposures with no\nmatched star", fontsize=8)
    colorbar.outline.set_visible(False)
    ax.set_title(
        "Camera presence per hour: share of AG exposures (with stars on some camera) in which this camera matched none\n"
        f"×: mostly no detections at all.   Hatched: the design had fewer than {run.config.minDesignStars} "
        "bright guide stars on it.",
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
        Six panels against the nights: matched stars per visit; FWHM per
        detector half; focus minus the six cameras' median; the share of
        judged AG exposures in which it matched a star; AG exposures with its
        rejected matches clustered on one side of the detector; and its
        position-model error.
    """
    nights = run.plottedNights
    position = {night: i + 0.5 for i, night in enumerate(nights)}
    frames = run.frames[run.frames.night.isin(nights)].assign(x=lambda df: nightPosition(df.taken_at, nights))
    fig = _figure(11, 6.2)
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
    zColumns = [f"guide_delta_z{c + 1}" for c in range(6)]
    allSix = frames[zColumns].notna().all(axis=1)
    z = frames.loc[allSix, zColumns].to_numpy() * 1e3
    if len(z):
        byNight = (
            pd.Series(z[:, cid] - np.median(z, axis=1)).groupby(frames.night[allSix].to_numpy()).median()
        )
        ax.plot([position[n] for n in byNight.index], byNight.to_numpy(), "-o", ms=3.5, lw=1.5, color="C0")
    ax.axhline(0, color=QUIET, lw=0.8)
    ax.set_ylim(-60, 60)
    ax.set_title("Focus minus the 6 cameras' median (µm), nightly", loc="left", fontsize=8.5)

    ax = axes[1, 0]
    states = frames.groupby("night")[f"state{cid}"]
    judged = [CameraState.OK, CameraState.UNMATCHED, CameraState.DARK]
    share = states.apply(
        lambda st: 100 * (st[st.isin(judged)] == CameraState.OK).mean() if st.isin(judged).any() else np.nan
    )
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
        angle = _meanAngle(2 * errors.angle_deg) / 2 % 180
        ax.text(
            0.98,
            0.04,
            f"growing toward {angle:.0f}° (mod 180)",
            transform=ax.transAxes,
            ha="right",
            fontsize=7,
        )
    radius = run.config.modelErrorRadiusPix
    ax.set_title(f"Model error {radius:g} px from centre (µm)", loc="left", fontsize=8.5)
    _legend(ax, loc="upper left")

    for row in range(2):
        for col in range(3):
            _nightAxis(axes[row, col], nights, label=(row == 1), rotate=True)
    group = next(name for name, cids in CAMERA_GROUPS.items() if cid in cids)
    fig.suptitle(f"AG{cid + 1} ({group})", x=0.01, ha="left", fontsize=10, fontweight="bold")
    return fig


def _scatterByCameras(ax: Axes, x, y, full) -> None:
    ax.scatter(x[full], y[full], s=10, color=INK, lw=0, label="stars matched on all 6 cameras")
    ax.scatter(x[~full], y[~full], s=16, color=BAD, marker="x", lw=1, label="on fewer than 6 cameras")


def plotGuiding(run: AgRun, jitter: pd.DataFrame) -> Figure:
    """Plot each science visit's settled jitter, and each night's guide-loop problems.

    Parameters
    ----------
    run : `AgRun`
        The run.
    jitter : `pandas.DataFrame`
        From `runAnalysis.settledJitter`.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The ΔRA (on the sky), ΔDec and ΔInR jitter per visit, a visit guided
        on fewer than six cameras marked, and per night the AG exposures
        whose offset was rejected or not written.
    """
    nights = run.plottedNights
    jitter = jitter[jitter.night.isin(nights)].assign(x=lambda df: nightPosition(df.taken_at, nights))
    full = jitter.n_cams >= 6
    fig = _figure(11, 8.5)
    axes = fig.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 0.8]})
    for ax, axis, name in zip(axes, ("ra", "dec", "inr"), ("ΔRA on the sky", "ΔDec", "ΔInR"), strict=False):
        _scatterByCameras(ax, jitter.x, jitter[axis], full)
        median = (jitter[axis][full] if full.any() else jitter[axis]).median()
        ax.axhline(median, color=INK, lw=0.8, ls="--")
        top = np.nanpercentile(jitter[axis], 99) * 1.15 if jitter[axis].notna().any() else np.nan
        top = top if np.isfinite(top) and top > 0 else 1.0
        ax.set_ylim(0, top)
        for _, visit in jitter[jitter[axis] > top].iterrows():
            ax.annotate(
                _shortNumber(visit[axis]), (visit.x, top), fontsize=6, color=BAD, ha="center", va="top"
            )
        ax.set_ylabel("arcsec")
        ax.set_title(
            f"{name}: settled frame-to-frame jitter per science visit (dashed: median, {median:.3g}″)",
            loc="left",
            fontsize=9,
        )
        _nightAxis(ax, nights, label=False)
    _legend(axes[0], above=True, ncol=2)

    ax = axes[3]
    frames = run.frames[run.frames.night.isin(nights)]
    perNight = frames.groupby("night")["mask"].agg(
        rejected=lambda mask: int((mask.fillna(0).astype(int) & 1).sum()),
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
    _scatterByCameras(ax, nightPosition(visits.taken_at, nights), visits.dz, visits.n_cams >= 6)
    ax.axhline(0, color=QUIET, lw=0.8)
    offScale = 0
    if visits.dz.notna().any():
        low, high = np.nanpercentile(visits.dz, [2, 98])
        pad = 0.3 * max(high - low, 1)
        ax.set_ylim(low - pad, high + pad)
        offScale = int(((visits.dz < low - pad) | (visits.dz > high + pad)).sum())
    ax.set_ylabel("µm")
    ax.set_title(
        "Guider focus error dZ (median of the cameras), per science visit"
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
