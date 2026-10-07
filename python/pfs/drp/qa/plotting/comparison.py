"""Plots of comparison mode: one period's metrics against the reference run's.

DataFrames in, figures out. Arms are panels, not colors, wherever two runs share a panel, so
the two runs are told apart by ink and line style alone: the period in dark ink, the reference
in gray and dashed. Verdict colors are the status palette and always carry a letter.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from pfs.drp.qa.plotting.palettes import detector_palette, spectrograph_plot_markers

__all__ = [
    "EXPOSURE_COLORS",
    "STATUS_COLORS",
    "plotArmTimeline",
    "plotMetricComparison",
    "plotNightlySeries",
    "plotVerdictGrid",
]

#: Verdict colors; each cell also carries its letter, so color is never alone.
STATUS_COLORS = {"PASS": "#0ca30c", "WARN": "#fab219", "FAIL": "#d03b3b", "UNKNOWN": "#d9d9d6"}
_INK, _REFERENCE_INK, _GRID = "#222222", "#8c8c8c", "#e6e6e3"

#: What lit an exposure: three categorical hues for the lit kinds (validated together), grays for
#: the unlit ones.
EXPOSURE_COLORS = {
    "arc": "#2a78d6",
    "quartz": "#eb6834",
    "sky": "#1baf7a",
    "dark": "#4d4d4b",
    "other": "#c4c4c0",
}
_ARM_ORDER = ("b", "r", "n", "m")


def plotMetricComparison(
    current: pd.DataFrame,
    reference: pd.DataFrame | None,
    metric: str,
    *,
    labels: tuple[str, str] = ("period", "reference"),
    thresholds: dict[str, tuple[float, float]] | None = None,
    title: str | None = None,
    panelSize: tuple[float, float] = (3.4, 2.6),
) -> Figure:
    """Plot a metric's distribution per arm: the period against the reference run.

    Parameters
    ----------
    current, reference : `pandas.DataFrame`
        Metrics rows with ``arm`` and ``metric``; ``reference`` may be `None`.
    metric : `str`
        The column to plot.
    labels : `tuple` [`str`, `str`], optional
        Names of the period and the reference, for the legend.
    thresholds : `dict` [`str`, `tuple` [`float`, `float`]], optional
        ``(warn, fail)`` by arm, drawn as vertical lines.
    title : `str`, optional
        The figure title; defaults to ``metric``.
    panelSize : `tuple` [`float`, `float`], optional
        Size of one panel in inches.

    Returns
    -------
    `matplotlib.figure.Figure`
        One panel per arm with data: the cumulative distribution of each run,
        with the numbers of images in the panel's title and one legend for the
        figure.
    """
    frames = [current] if reference is None else [current, reference]
    arms = [arm for arm in _ARM_ORDER if any((frame["arm"] == arm).any() for frame in frames)]
    fig, axes = plt.subplots(
        1, max(1, len(arms)), figsize=(panelSize[0] * max(1, len(arms)), panelSize[1]), squeeze=False
    )
    styles = ((labels[0], _INK, "-"), (labels[1], _REFERENCE_INK, "--"))
    for ax, arm in zip(axes.flat, arms, strict=False):
        counts = []
        for frame, (label, color, style) in zip((current, reference), styles, strict=True):
            if frame is None:
                continue
            values = np.sort(frame.loc[frame["arm"] == arm, metric].astype(float).dropna().to_numpy())
            counts.append(f"{values.size} {label}")
            if values.size:
                fraction = np.arange(1, values.size + 1) / values.size
                ax.step(values, fraction, where="post", color=color, linestyle=style, linewidth=2)
        if thresholds and arm in thresholds:
            for level, value in zip(("WARN", "FAIL"), thresholds[arm], strict=True):
                if value is not None and np.isfinite(value):
                    ax.axvline(value, color=STATUS_COLORS[level], linewidth=1.5)
                    ax.annotate(
                        level,
                        (value, 1.0),
                        xycoords=("data", "axes fraction"),
                        rotation=90,
                        fontsize="x-small",
                        color=_INK,
                        ha="right" if level == "WARN" else "left",  # apart when the two are close
                        va="top",
                    )
        # The counts go in the title, so no legend sits on the curves.
        ax.set_title(f"{arm} arm: {', '.join(counts)}", fontsize="small")
        ax.set_xlabel(metric)
        ax.set_ylim(0, 1.02)
        ax.grid(color=_GRID, linewidth=0.8)
        _recessive(ax)
    handles = [
        Line2D([], [], color=color, linestyle=style, linewidth=2, label=label)
        for (label, color, style), frame in zip(styles, (current, reference), strict=True)
        if frame is not None
    ]
    fig.legend(handles=handles, loc="upper right", fontsize="x-small", frameon=False, ncols=len(handles))
    axes.flat[0].set_ylabel("fraction of images")
    if not arms:
        axes.flat[0].text(0.5, 0.5, "no data", ha="center", va="center", transform=axes.flat[0].transAxes)
    fig.suptitle(title or metric)
    fig.tight_layout()
    return fig


def plotNightlySeries(
    metrics: pd.DataFrame,
    metric: str,
    *,
    reference: pd.DataFrame | None = None,
    title: str | None = None,
    panelSize: tuple[float, float] = (4.2, 2.6),
) -> Figure:
    """Plot a metric night by night, per detector, one panel per arm.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        Rows with ``night``, ``arm``, ``spectrograph`` and ``metric``.
    metric : `str`
        The column to plot.
    reference : `pandas.DataFrame`, optional
        The reference run's rows, with ``arm`` and ``metric``: its 5-95
        percentile range is shaded per arm.
    title : `str`, optional
        The figure title; defaults to ``metric``.
    panelSize : `tuple` [`float`, `float`], optional
        Size of one panel in inches.

    Returns
    -------
    `matplotlib.figure.Figure`
        The median of each detector on each night, in the arm's color with
        the spectrograph's marker.
    """
    arms = [arm for arm in _ARM_ORDER if (metrics["arm"] == arm).any()]
    fig, axes = plt.subplots(
        1, max(1, len(arms)), figsize=(panelSize[0] * max(1, len(arms)), panelSize[1]), squeeze=False
    )
    nights = sorted(metrics["night"].dropna().unique())
    position = {night: i for i, night in enumerate(nights)}
    for ax, arm in zip(axes.flat, arms, strict=False):
        subset = metrics[metrics["arm"] == arm]
        if reference is not None:
            values = reference.loc[reference["arm"] == arm, metric].astype(float).dropna()
            if len(values):
                low, high = np.percentile(values, [5, 95])
                ax.axhspan(low, high, color=_GRID, zorder=0)
        medians = subset.groupby(["spectrograph", "night"])[metric].median().reset_index()
        for spectrograph, group in medians.groupby("spectrograph"):
            ax.plot(
                [position[night] for night in group["night"]],
                group[metric],
                color=detector_palette.get(arm, _INK),
                marker=spectrograph_plot_markers.get(int(spectrograph), "o"),
                markersize=6,
                linewidth=1,
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
        ax.set_title(f"{arm} arm", fontsize="medium")
        ax.set_xticks(range(len(nights)))
        ax.set_xticklabels(
            [pd.Timestamp(night).strftime("%m-%d") for night in nights], rotation=90, fontsize="x-small"
        )
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        _recessive(ax)
    axes.flat[0].set_ylabel(metric)
    handles = [
        Line2D([], [], color=_INK, marker=marker, linestyle="", label=f"SM{spectrograph}")
        for spectrograph, marker in spectrograph_plot_markers.items()
        if (metrics["spectrograph"] == spectrograph).any()
    ]
    if reference is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=_GRID, label="reference 5-95%"))
    fig.legend(handles=handles, loc="upper right", fontsize="x-small", frameon=False, ncols=len(handles))
    fig.suptitle(title or metric, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def plotVerdictGrid(verdicts: pd.DataFrame, *, title: str = "Verdicts by detector and night") -> Figure:
    """Show the worst verdict of each detector on each night.

    Parameters
    ----------
    verdicts : `pandas.DataFrame`
        One row per image: ``night``, ``arm``, ``spectrograph`` and
        ``status`` (``PASS``, ``WARN``, ``FAIL`` or ``UNKNOWN``).
    title : `str`, optional
        The figure title.

    Returns
    -------
    `matplotlib.figure.Figure`
        Detectors down, nights across; each cell colored and lettered by its
        worst verdict, blank where nothing was judged.
    """
    order = {status: i for i, status in enumerate(("UNKNOWN", "PASS", "WARN", "FAIL"))}
    frame = verdicts.assign(
        rank=verdicts["status"].map(order).fillna(0),
        detector=verdicts["arm"].astype(str) + verdicts["spectrograph"].astype(str),
    )
    grid = frame.pivot_table(index="detector", columns="night", values="rank", aggfunc="max")
    detectors = sorted(
        grid.index, key=lambda name: (_ARM_ORDER.index(name[0]) if name[0] in _ARM_ORDER else 9, name)
    )
    grid = grid.reindex(index=detectors)
    names = list(order)
    cmap = ListedColormap([STATUS_COLORS[name] for name in names]).with_extremes(bad="white")
    fig, ax = plt.subplots(
        figsize=(max(4.0, 0.32 * grid.shape[1] + 1.5), max(2.5, 0.28 * grid.shape[0] + 1.2))
    )
    ax.imshow(
        np.ma.masked_invalid(grid.to_numpy(dtype=float)),
        cmap=cmap,
        vmin=-0.5,
        vmax=len(names) - 0.5,
        aspect="auto",
        interpolation="nearest",
    )
    for (row, column), value in np.ndenumerate(grid.to_numpy(dtype=float)):
        if np.isfinite(value):
            ax.text(column, row, names[int(value)][0], ha="center", va="center", fontsize=6, color=_INK)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(
        [pd.Timestamp(night).strftime("%m-%d") for night in grid.columns], rotation=90, fontsize="x-small"
    )
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid.index, fontsize="x-small")
    ax.set_xticks(np.arange(-0.5, grid.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=STATUS_COLORS[name], label=f"{name[0]} {name}") for name in names
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="x-small", frameon=False)
    ax.set_title(title, loc="left", fontsize="medium")
    fig.tight_layout()
    return fig


def _recessive(ax) -> None:
    """Quiet an axes' frame: no top or right spine, gray ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_REFERENCE_INK)
    ax.tick_params(colors="#555555", labelsize="x-small")


def plotArmTimeline(
    timeline: pd.DataFrame,
    darkSequences: pd.DataFrame,
    *,
    arm: str = "n",
    minWidth: float = 120.0,
    title: str | None = None,
) -> Figure:
    """Show an arm's exposures night by night, and how soon each dark sequence followed a lit one.

    Parameters
    ----------
    timeline : `pandas.DataFrame`
        ``night``, ``start``, ``exptime`` and ``kind``, from
        `pfs.drp.qa.comparison.persistence.armTimeline`.
    darkSequences : `pandas.DataFrame`
        ``night``, ``minutesSince`` and ``litKind``, from
        `pfs.drp.qa.comparison.persistence.darkSequences`.
    arm : `str`, optional
        The arm, for the title.
    minWidth : `float`, optional
        Shortest bar (s), so that a 5 s arc is visible.
    title : `str`, optional
        The figure title.

    Returns
    -------
    `matplotlib.figure.Figure`
        Left: one row per night, noon to noon, each exposure a bar colored by
        what lit it, darks dark gray. Right, on the same rows: the gap from
        the last lit exposure to each dark sequence (minutes, log scale),
        colored by what that exposure was.
    """
    nights = sorted(timeline["night"].dropna().unique())
    row = {night: i for i, night in enumerate(nights)}
    height = max(2.5, 0.22 * len(nights) + 1.4)
    fig, (axTime, axGap) = plt.subplots(
        1, 2, figsize=(12, height), sharey=True, gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05}
    )
    noon = pd.to_datetime(pd.Series(nights)).dt.normalize() + pd.Timedelta(hours=12)
    noonOf = dict(zip(nights, noon, strict=True))
    for kind, color in EXPOSURE_COLORS.items():
        subset = timeline[timeline["kind"] == kind]
        for night, group in subset.groupby("night"):
            offset = (pd.to_datetime(group["start"]) - noonOf[night]).dt.total_seconds() / 3600 + 12
            widths = group["exptime"].clip(lower=minWidth) / 3600
            axTime.broken_barh(
                list(zip(offset, widths, strict=True)),
                (row[night] - 0.38, 0.76),
                facecolors=color,
                linewidth=0,
            )
    axTime.set_xlim(12, 36)
    axTime.set_xticks(range(12, 37, 3))
    axTime.set_xticklabels([f"{hour % 24:02d}:00" for hour in range(12, 37, 3)], fontsize="x-small")
    axTime.set_xlabel("HST (night runs noon to noon)")
    axTime.set_yticks(range(len(nights)))
    axTime.set_yticklabels([pd.Timestamp(night).strftime("%m-%d") for night in nights], fontsize="x-small")
    axTime.invert_yaxis()
    axTime.grid(axis="x", color=_GRID, linewidth=0.8)
    _recessive(axTime)

    gaps = darkSequences.dropna(subset=["minutesSince"])
    gaps = gaps[gaps["night"].isin(row)]
    for kind in ("arc", "quartz", "sky"):
        subset = gaps[gaps["litKind"] == kind]
        axGap.scatter(
            subset["minutesSince"].clip(lower=0.1),
            subset["night"].map(row),
            s=36,
            color=EXPOSURE_COLORS[kind],
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
    axGap.set_xscale("log")
    axGap.set_xlim(0.5, 2000)
    axGap.axvspan(0.5, 5, color=_GRID, zorder=0)
    axGap.set_xlabel("minutes from lit to dark")
    axGap.grid(axis="x", color=_GRID, linewidth=0.8)
    _recessive(axGap)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=kind if kind != "other" else "bias, test")
        for kind, color in EXPOSURE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper right", fontsize="x-small", frameon=False, ncols=len(handles))
    fig.suptitle(title or f"{arm} arm: exposures by night, and each dark sequence's gap", x=0.02, ha="left")
    return fig
