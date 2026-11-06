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
    "plotCoverage",
    "plotDriftTimeline",
    "plotMetricComparison",
    "plotNightlySeries",
    "plotSetAgainstReference",
    "plotVerdictGrid",
    "plotVerdictTimeline",
]

#: Verdict colors; each cell also carries its letter, so color is never alone.
STATUS_COLORS = {
    "PASS": "#0ca30c",
    "WARN": "#fab219",
    "FAIL": "#d03b3b",
    "UNKNOWN": "#d9d9d6",
    "NOT JUDGED": "#eeeeec",
}
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
            if metric not in frame:  # a reference measured before the metric existed
                values = np.empty(0)
            else:
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
    sharey: bool = False,
    zeroLine: bool = False,
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
    sharey : `bool`, optional
        One y scale for every arm's panel, for a metric whose size compares across arms (a drift).
    zeroLine : `bool`, optional
        Draw a line at zero, for a signed metric.

    Returns
    -------
    `matplotlib.figure.Figure`
        The median of each detector on each night, in the arm's color with
        the spectrograph's marker.
    """
    arms = [arm for arm in _ARM_ORDER if (metrics["arm"] == arm).any()]
    fig, axes = plt.subplots(
        1,
        max(1, len(arms)),
        figsize=(panelSize[0] * max(1, len(arms)), panelSize[1]),
        squeeze=False,
        sharey=sharey,
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
        if zeroLine:
            ax.axhline(0, color="black", linestyle="--", linewidth=1.4, alpha=0.75, zorder=-100)
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
        ``status`` (``PASS``, ``WARN``, ``FAIL``, ``UNKNOWN`` or ``NOT JUDGED``).
    title : `str`, optional
        The figure title.

    Returns
    -------
    `matplotlib.figure.Figure`
        Detectors down, nights across; each cell colored and lettered by its
        worst verdict, blank where nothing was judged.
    """
    order = {status: i for i, status in enumerate(("NOT JUDGED", "UNKNOWN", "PASS", "WARN", "FAIL"))}
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


def plotVerdictTimeline(
    verdicts: pd.DataFrame, *, title: str = "Calibration images: verdict per visit"
) -> Figure:
    """Show every image's verdict as a mark on its detector's lane, visits across in time order.

    Parameters
    ----------
    verdicts : `pandas.DataFrame`
        One row per image: ``visit``, ``arm``, ``spectrograph``, ``night`` and ``status``.
    title : `str`, optional
        The figure title.

    Returns
    -------
    `matplotlib.figure.Figure`
        Detectors down, the period's visits across in order; PASS a small gray dot, WARN amber and
        FAIL red with the visit number written above, UNKNOWN and NOT JUDGED pale; nights separated
        and named along the top. Says which visit is wrong on which detector, which a per-night
        summary cannot.
    """
    frame = verdicts.assign(detector=verdicts["arm"].astype(str) + verdicts["spectrograph"].astype(str))
    visits = sorted(frame["visit"].astype(int).unique())
    column = {v: i for i, v in enumerate(visits)}
    detectors = sorted(
        frame["detector"].unique(), key=lambda n: (_ARM_ORDER.index(n[0]) if n[0] in _ARM_ORDER else 9, n)
    )
    row = {d: i for i, d in enumerate(detectors)}
    fig, ax = plt.subplots(
        figsize=(max(8.0, 0.055 * len(visits) + 2.5), max(3.0, 0.3 * len(detectors) + 1.6))
    )
    style = {
        "PASS": {"color": "#8c8c8c", "s": 9, "zorder": 2},
        "WARN": {"color": STATUS_COLORS["WARN"], "s": 30, "zorder": 4},
        "FAIL": {"color": STATUS_COLORS["FAIL"], "s": 34, "zorder": 5},
        "UNKNOWN": {"color": STATUS_COLORS["UNKNOWN"], "s": 9, "zorder": 1},
        "NOT JUDGED": {"color": STATUS_COLORS["NOT JUDGED"], "s": 9, "zorder": 1},
    }
    for status, kw in style.items():
        subset = frame[frame["status"] == status]
        if subset.empty:
            continue
        ax.scatter(
            subset["visit"].astype(int).map(column),
            subset["detector"].map(row),
            label=status,
            linewidths=0,
            **kw,
        )
    flagged = frame[frame["status"].isin(["WARN", "FAIL"])]
    for (visit, det), g in flagged.groupby(["visit", "detector"]):
        worst = "FAIL" if (g["status"] == "FAIL").any() else "WARN"
        ax.text(
            column[int(visit)],
            row[det] - 0.45,
            str(int(visit)),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=5,
            color=STATUS_COLORS[worst],
        )
    # Night boundaries, named along the top.
    nights = frame.dropna(subset=["night"]).groupby("night")["visit"].agg(["min", "max"]).sort_index()
    for night, (first, last) in nights.iterrows():
        left, right = column[int(first)], column[int(last)]
        ax.axvline(left - 0.5, color=_GRID, linewidth=1)
        ax.text(
            (left + right) / 2,
            -0.9,
            pd.Timestamp(night).strftime("%m-%d"),
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax.set_yticks(range(len(detectors)))
    ax.set_yticklabels(detectors, fontsize="x-small")
    ax.set_ylim(len(detectors) - 0.5, -1.2)
    ax.set_xlim(-0.5, len(visits) - 0.5)
    step = max(1, len(visits) // 24)
    ax.set_xticks(range(0, len(visits), step))
    ax.set_xticklabels([str(visits[i]) for i in range(0, len(visits), step)], rotation=90, fontsize=5)
    ax.set_xlabel("visit", fontsize="small")
    ax.grid(axis="y", color=_GRID, linewidth=0.6)
    _recessive(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="x-small", frameon=False, markerscale=1.5)
    ax.set_title(title, loc="left", fontsize="medium", pad=16)
    fig.tight_layout()
    return fig


def _recessive(ax) -> None:
    """Quiet an axes' frame: no top or right spine, gray ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_REFERENCE_INK)
    ax.tick_params(colors="#555555", labelsize="x-small")


def _calibrationName(collection: str) -> str:
    """``detectorMapsGen.20261030b`` from ``PFS/calib/.../detectorMapsGen.20261030b/certify``."""
    parts = [part for part in str(collection).split("/") if part and part != "certify"]
    return parts[-1] if parts else str(collection)


def plotSetAgainstReference(
    current: pd.DataFrame,
    reference: pd.DataFrame | None,
    metric: str,
    *,
    labels: tuple[str, str] = ("period", "reference"),
    thresholds: dict[str, tuple[float, float]] | None = None,
    title: str | None = None,
    panelSize: tuple[float, float] = (3.0, 2.6),
) -> Figure:
    """Compare a calibration set's metric with the reference run's set, detector by detector.

    Parameters
    ----------
    current, reference : `pandas.DataFrame`
        Metrics rows of the two sets, with ``arm``, ``spectrograph`` and ``metric``; ``reference``
        may be `None`.
    metric : `str`
        The column to compare.
    labels : `tuple` [`str`, `str`], optional
        Names of the period and the reference, for the legend.
    thresholds : `dict` [`str`, `tuple` [`float`, `float`]], optional
        ``(warn, fail)`` by arm, drawn as horizontal lines.
    title : `str`, optional
        The figure title; defaults to ``metric``.
    panelSize : `tuple` [`float`, `float`], optional
        Size of one panel in inches.

    Returns
    -------
    `matplotlib.figure.Figure`
        One panel per arm, the spectrographs along the bottom: each detector's median over the
        set's images, hollow for the reference on the left and filled for the period on the right,
        with a bar from the 16th to the 84th percentile of its images. The y scale is shared, so
        the arms compare too.
    """
    frames = [current] if reference is None else [current, reference]
    arms = [arm for arm in _ARM_ORDER if any((frame["arm"] == arm).any() for frame in frames)]
    fig, axes = plt.subplots(
        1,
        max(1, len(arms)),
        figsize=(panelSize[0] * max(1, len(arms)), panelSize[1]),
        squeeze=False,
        sharey=True,
    )
    for ax, arm in zip(axes.flat, arms, strict=False):
        color = detector_palette.get(arm, _INK)
        # The reference first (left): it is earlier, and what the period is measured against.
        for offset, frame, hollow in ((-0.15, reference, True), (0.15, current, False)):
            if frame is None:
                continue
            subset = frame[frame["arm"] == arm].dropna(subset=[metric])
            if subset.empty:
                continue
            stats = subset.groupby("spectrograph")[metric].quantile([0.16, 0.5, 0.84]).unstack()
            for spectrograph, row in stats.iterrows():
                x = int(spectrograph) + offset
                ax.plot([x, x], [row[0.16], row[0.84]], color=color, linewidth=1.2, alpha=0.5, zorder=2)
                ax.plot(
                    x,
                    row[0.5],
                    marker=spectrograph_plot_markers.get(int(spectrograph), "o"),
                    markersize=7,
                    linestyle="none",
                    markerfacecolor="none" if hollow else color,
                    markeredgecolor=color,
                    markeredgewidth=1.2 if hollow else 0.8,
                    zorder=3,
                )
        if thresholds and arm in thresholds:
            warn, fail = thresholds[arm]
            ax.axhline(warn, color=STATUS_COLORS["WARN"], linestyle="--", linewidth=1, zorder=1)
            ax.axhline(fail, color=STATUS_COLORS["FAIL"], linestyle="--", linewidth=1, zorder=1)
        ax.set_title(f"{arm} arm", fontsize="medium")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([f"{arm}{sp}" for sp in (1, 2, 3, 4)], fontsize="x-small")
        ax.set_xlim(0.4, 4.6)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        _recessive(ax)
    axes.flat[0].set_ylabel(metric)
    handles = [
        Line2D([], [], color=_INK, marker="o", markerfacecolor="none", linestyle="", label=labels[1]),
        Line2D([], [], color=_INK, marker="o", linestyle="", label=labels[0]),
    ]
    if thresholds:
        handles += [
            Line2D([], [], color=STATUS_COLORS["WARN"], linestyle="--", label="WARN"),
            Line2D([], [], color=STATUS_COLORS["FAIL"], linestyle="--", label="FAIL"),
        ]
    fig.legend(handles=handles, loc="upper right", fontsize="x-small", frameon=False, ncols=len(handles))
    fig.suptitle(title or metric, x=0.02, ha="left")
    fig.tight_layout()
    return fig


def plotDriftTimeline(
    drift: pd.DataFrame,
    *,
    title: str = "Drift from detectorMap_calib, per visit",
    reference: pd.DataFrame | None = None,
    referenceLabel: str = "reference",
) -> Figure:
    """Show each image's drift from the calibration, visits across in time order, the arms in a 2x2 grid.

    Parameters
    ----------
    drift : `pandas.DataFrame`
        One row per image: ``visit``, ``arm``, ``spectrograph``, ``night``, ``medianDeltaX``,
        ``medianDeltaY`` (px), ``calibRun`` (the calibration it was measured against) and, optionally,
        ``cameraSet`` (``brn``, ``bmn``, ...).
    title : `str`, optional
        The figure title.
    reference : `pandas.DataFrame`, optional
        The reference run's drift rows (``arm``, ``medianDeltaX``, ``medianDeltaY``), measured
        against its own calibration: their 5-95 % range is shaded per arm and metric.
    referenceLabel : `str`, optional
        The reference's name, for the legend.

    Returns
    -------
    `matplotlib.figure.Figure`
        The arms in a 2x2 grid, b and r above m and n so that b sits over m, the arm it is read
        with, each with x above y; a b image read with m is hollow, since it sits about 0.5 px
        from the b calibration whatever the detector does; every x panel shares one
        scale and every y panel another. The spectrograph is the marker, the arm the colour; a
        dashed line at zero; nights separated and named along the top; a dashed vertical line,
        named, at the first visit measured against each new calibration, so a change of reference
        is not read as a drift.
    """
    frame = drift.dropna(subset=["visit"]).copy()
    frame["visit"] = frame["visit"].astype(int)
    arms = [arm for arm in ("b", "r", "m", "n") if (frame["arm"] == arm).any()]  # b above m: read together
    visits = sorted(frame["visit"].unique())
    column = {v: i for i, v in enumerate(visits)}
    metrics = ("medianDeltaX", "medianDeltaY")
    cells = max(1, (len(arms) + 1) // 2)  # grid rows of arms
    width = max(5.0, 0.08 * len(visits) + 2.0)
    fig, axes = plt.subplots(2 * cells, 2, figsize=(2 * width, 4.4 * cells), squeeze=False, sharex=True)
    for ax in axes.flat:
        ax.set_visible(False)
    nights = frame.dropna(subset=["night"]).groupby("night")["visit"].agg(["min", "max"]).sort_index()
    changes = []
    if "calibRun" in frame:
        first = frame.dropna(subset=["calibRun"]).groupby("calibRun")["visit"].min().sort_values()
        changes = list(first.items())[1:]  # the first calibration is the starting point, not a change
    panels: dict[str, list] = {metric: [] for metric in metrics}
    for index, arm in enumerate(arms):
        subset = frame[frame["arm"] == arm]
        cell, col = divmod(index, 2)
        for row, metric in enumerate(metrics):
            ax = axes[2 * cell + row, col]
            ax.set_visible(True)
            panels[metric].append(ax)
            ax.axhline(0, color="black", linestyle="--", linewidth=1.4, alpha=0.75, zorder=-100)
            if reference is not None:
                values = reference.loc[reference["arm"] == arm, metric].astype(float).dropna()
                if len(values) >= 5:
                    low, high = np.percentile(values, [5, 95])
                    ax.axhspan(low, high, color=_GRID, zorder=-200)
            withM = (
                subset["cameraSet"].fillna("").astype(str).str.contains("m") & (subset["arm"] == "b")
                if "cameraSet" in subset
                else pd.Series(False, index=subset.index)
            )
            for hollow in (False, True):
                part = subset[(withM == hollow)].dropna(subset=[metric])
                for spectrograph, group in part.groupby("spectrograph"):
                    ax.plot(
                        group["visit"].map(column),
                        group[metric],
                        color=detector_palette.get(arm, _INK),
                        marker=spectrograph_plot_markers.get(int(spectrograph), "o"),
                        markersize=4,
                        linestyle="none",  # no lines: b read with m sits 0.5 px from b read with r
                        markerfacecolor="none" if hollow else detector_palette.get(arm, _INK),
                        markeredgecolor=detector_palette.get(arm, _INK) if hollow else "white",
                        markeredgewidth=0.8 if hollow else 0.5,
                    )
            for left in nights["min"]:
                ax.axvline(column[int(left)] - 0.5, color=_GRID, linewidth=1, zorder=-50)
            for _, visit in changes:
                ax.axvline(column[int(visit)] - 0.5, color=_INK, linestyle="--", linewidth=1, zorder=-40)
            ax.grid(axis="y", color=_GRID, linewidth=0.8)
            _recessive(ax)
            ax.set_ylabel(f"{arm} arm, {'x: traces' if metric == metrics[0] else 'y: arc lines'} (px)")
            if row == 0:
                for night, (left, right) in nights.iterrows():
                    ax.text(
                        (column[int(left)] + column[int(right)]) / 2,
                        1.0,
                        pd.Timestamp(night).strftime("%m-%d"),
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )
                for name, visit in changes:
                    ax.text(
                        column[int(visit)] - 0.3,
                        0.98,
                        _calibrationName(name),
                        transform=ax.get_xaxis_transform(),
                        ha="left",
                        va="top",
                        fontsize=6,
                        rotation=90,
                        color=_INK,
                    )
            step = max(1, len(visits) // 16)
            ax.set_xticks(range(0, len(visits), step))
            ax.set_xticklabels([str(visits[i]) for i in range(0, len(visits), step)], rotation=90, fontsize=6)
            ax.tick_params(labelbottom=True)
    for metric in metrics:  # one scale per metric, over every arm
        values = frame[metric].dropna()
        if len(values) and panels[metric]:
            low, high = float(values.min()), float(values.max())
            pad = 0.08 * max(high - low, 0.1)
            for ax in panels[metric]:
                ax.set_ylim(min(low, 0) - pad, max(high, 0) + pad)
    axes[0, 0].set_xlim(-0.5, len(visits) - 0.5)
    handles = [
        Line2D([], [], color=_INK, marker=marker, linestyle="", label=f"SM{spectrograph}")
        for spectrograph, marker in spectrograph_plot_markers.items()
        if (frame["spectrograph"] == spectrograph).any()
    ]
    if changes:
        handles.append(Line2D([], [], color=_INK, linestyle="--", label="new calibration"))
    if reference is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=_GRID, label=f"{referenceLabel} 5-95%"))
    if (
        "cameraSet" in frame
        and (frame["cameraSet"].fillna("").astype(str).str.contains("m") & (frame["arm"] == "b")).any()
    ):
        handles.append(
            Line2D(
                [], [], color=_INK, marker="o", markerfacecolor="none", linestyle="", label="b read with m"
            )
        )
    fig.legend(handles=handles, loc="upper right", fontsize="x-small", frameon=False, ncols=len(handles))
    fig.suptitle(title, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


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


#: Coverage states of a judged image, darkest first; ``blocked`` (no raw, no pfsConfig, raw opdb didn't
#: list) is the one that needs attention, so it alone has a hue.
COVERAGE_COLORS = {
    "judged": "#4d4d4b",
    "to judge": "#8c8c8c",
    "to reduce": "#c4c4c0",
    "failed": "#d03b3b",
    "blocked": "#eb6834",
}


def plotCoverage(coverage: pd.DataFrame, *, title: str = "Coverage of judged images") -> Figure:
    """Show how far each judged sequence type is covered, as stacked horizontal bars.

    Parameters
    ----------
    coverage : `pandas.DataFrame`
        One row per bar: ``label`` and one image count per key of
        `COVERAGE_COLORS`.
    title : `str`, optional
        The figure title.

    Returns
    -------
    `matplotlib.figure.Figure`
        One bar per row, its segments in `COVERAGE_COLORS` order, the number
        of images judged written at the bar's end.
    """
    rows = coverage.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, max(1.6, 0.34 * len(rows) + 0.9)))
    left = np.zeros(len(rows))
    for state, color in COVERAGE_COLORS.items():
        values = rows[state].to_numpy(dtype=float) if state in rows else np.zeros(len(rows))
        ax.barh(range(len(rows)), values, left=left, color=color, height=0.7, edgecolor="white", linewidth=1)
        left += values
    for i, (judged, total) in enumerate(zip(rows.get("judged", left * 0), left, strict=True)):
        ax.text(total, i, f"  {int(judged)}/{int(total)}", va="center", fontsize="x-small", color=_INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows["label"], fontsize="x-small")
    ax.invert_yaxis()
    ax.set_xlabel("detector images")
    ax.set_xlim(0, max(1.0, left.max()) * 1.15)
    ax.grid(axis="x", color=_GRID, linewidth=0.8)
    _recessive(ax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=state) for state, color in COVERAGE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper right", fontsize="x-small", frameon=False, ncols=len(handles))
    fig.suptitle(title, x=0.02, ha="left", fontsize="medium")
    fig.tight_layout()
    return fig
