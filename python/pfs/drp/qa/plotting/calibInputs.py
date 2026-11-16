"""Plots of a calibration set's inputs (`pfs.drp.qa.comparison.calibInputs`).

DataFrames in, figures out. The verdict grid is the report's main picture: a detector that goes
wrong mid-set shows as a row turning red at one visit. Every verdict cell carries its letter, so
color is never alone; detectors are told apart by arm color and spectrograph marker, as elsewhere
in `pfs.drp.qa.plotting`.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from pfs.drp.qa.comparison.calibInputs import Verdict
from pfs.drp.qa.metrics.lineResiduals import robustRms
from pfs.drp.qa.plotting.palettes import description_palette, detector_palette, spectrograph_plot_markers

__all__ = [
    "VERDICT_COLORS",
    "plotCalibrationComparison",
    "plotDetectorDetail",
    "plotDetectorSeries",
    "plotGoNoGoGrid",
    "plotLampTrends",
    "plotLineResiduals",
    "plotVerdictGrid",
]

#: Verdict colors, the status palette's green, amber and red, with a pale green for *weak*; each
#: cell also carries the verdict's first letter.
VERDICT_COLORS = {
    Verdict.USE: "#0ca30c",
    Verdict.WEAK: "#9fd59f",
    Verdict.CHECK: "#fab219",
    Verdict.OUT: "#d03b3b",
}
_INK, _GRID, _NOT_EXPOSED = "#222222", "#e6e6e3", "#d9d9d6"
_ARM_ORDER = "brmn"


def _detectorOrder(names) -> list[str]:
    return sorted(names, key=lambda name: (_ARM_ORDER.index(name[0]) if name[0] in _ARM_ORDER else 9, name))


def plotVerdictGrid(
    judged: pd.DataFrame, visits: pd.DataFrame, *, title: str = "Verdict per detector image"
) -> Figure:
    """Show every detector image's verdict: detectors down, the set's visits across in time order.

    Parameters
    ----------
    judged : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.calibInputs.judge`.
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.calibInputs.setVisits`: every set visit is a column, blank where
        a detector has no metrics.
    title : `str`, optional
        The figure title.

    Returns
    -------
    `matplotlib.figure.Figure`
        Cells colored and lettered by verdict (u, w, c, o); sequences separated and named.
    """
    names = list(Verdict.ORDER)[::-1]  # use, weak, check, out: lowest rank is best
    rank = {name: i for i, name in enumerate(names)}
    frame = judged.assign(rank=judged["verdict"].astype(str).map(rank))
    grid = frame.pivot_table(index="detector", columns="visit", values="rank", aggfunc="max")
    grid = grid.reindex(index=_detectorOrder(grid.index), columns=list(visits["visit"]))
    cmap = ListedColormap([VERDICT_COLORS[name] for name in names]).with_extremes(bad="white")
    fig, ax = plt.subplots(
        figsize=(max(5.0, 0.22 * grid.shape[1] + 2.0), max(3.0, 0.28 * grid.shape[0] + 1.6))
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
            ax.text(column, row, names[int(value)][0], ha="center", va="center", fontsize=5, color=_INK)
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(grid.index, fontsize="x-small")
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels([str(v) for v in grid.columns], rotation=90, fontsize=5)
    # Sequence bands along the top, named by lamp.
    sequences = visits.groupby("iic_sequence_id", sort=False)
    for _, g in sequences:
        left, right = (
            list(grid.columns).index(g["visit"].iloc[0]),
            list(grid.columns).index(g["visit"].iloc[-1]),
        )
        ax.axvline(left - 0.5, color=_INK, linewidth=0.8)
        ax.text(
            (left + right) / 2,
            -0.9,
            g["lamp"].iloc[0] or "?",
            ha="center",
            va="bottom",
            fontsize=6,
            rotation=0 if right - left > 1 else 90,
        )
    ax.set_xticks(np.arange(-0.5, grid.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLORS[n], label=f"{n[0]} {n}") for n in names]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=_GRID, label="not measured"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize="x-small", frameon=False)
    ax.set_title(title, loc="left", fontsize="medium", pad=22)
    fig.tight_layout()
    return fig


def plotGoNoGoGrid(
    judged: pd.DataFrame, visits: pd.DataFrame, *, title: str = "Go / no-go per visit and detector"
) -> Figure:
    """Show the go/no-go of every detector on every set visit: the one picture the calibration builder needs.

    Parameters
    ----------
    judged : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.calibInputs.judge`.
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.calibInputs.setVisits`.
    title : `str`, optional
        The figure title.

    Returns
    -------
    `matplotlib.figure.Figure`
        Visits down in time order (the first at the top), detectors across; go green, no-go red
        with a cross, exposed but not measured white, never exposed grey. Sequences separated and
        named by lamp down the right.
    """
    from pfs.drp.qa.comparison.calibInputs import goNoGo

    table = goNoGo(judged, visits).T  # visits down, detectors across
    codes = {"go": 0.0, "no-go": 1.0, "missing": 2.0}
    values = np.array(
        [[codes.get(cell, np.nan) for cell in row] for row in table.to_numpy(dtype=object)], dtype=float
    ).reshape(table.shape)
    cmap = ListedColormap([VERDICT_COLORS[Verdict.USE], VERDICT_COLORS[Verdict.OUT], "white"]).with_extremes(
        bad=_NOT_EXPOSED
    )
    nVisits, nDetectors = table.shape
    fig, ax = plt.subplots(figsize=(max(5.0, 0.42 * nDetectors + 2.5), max(4.0, 0.2 * nVisits + 1.8)))
    ax.imshow(
        np.ma.masked_invalid(values), cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", interpolation="nearest"
    )
    for (row, column), value in np.ndenumerate(values):
        if value == 1:
            ax.text(column, row, "×", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.set_xticks(range(nDetectors))
    ax.set_xticklabels(table.columns, fontsize=9)
    ax.xaxis.tick_top()
    ax.set_yticks(range(nVisits))
    ax.set_yticklabels([str(v) for v in table.index], fontsize=7)
    rows = list(table.index)
    for _, g in visits.groupby("iic_sequence_id", sort=False):
        top, bottom = rows.index(g["visit"].iloc[0]), rows.index(g["visit"].iloc[-1])
        ax.axhline(top - 0.5, color=_INK, linewidth=1)
        ax.text(
            nDetectors - 0.3,
            (top + bottom) / 2,
            g["lamp"].iloc[0] or "?",
            ha="left",
            va="center",
            fontsize=8,
            clip_on=False,
        )
    ax.set_xticks(np.arange(-0.5, nDetectors), minor=True)
    ax.set_yticks(np.arange(-0.5, nVisits), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLORS[Verdict.USE], label="go"),
        plt.Rectangle((0, 0), 1, 1, color=VERDICT_COLORS[Verdict.OUT], label="× no-go"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=_INK, label="not measured (no data)"),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=_NOT_EXPOSED, edgecolor=_NOT_EXPOSED, label="detector not exposed"
        ),
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=9, frameon=False
    )
    ax.set_title(title, loc="left", fontsize="medium", pad=28)
    fig.tight_layout()
    return fig


def plotDetectorDetail(
    judged: pd.DataFrame,
    species: pd.DataFrame,
    visits: pd.DataFrame,
    detector: str,
    *,
    title: str | None = None,
) -> Figure:
    """Show one detector through the set: FWHM, reserved-line RMS and lines per fiber, visit by visit.

    Parameters
    ----------
    judged : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.calibInputs.judge`.
    species : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.calibInputs.speciesTable`.
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.calibInputs.setVisits`, for the sequence bands.
    detector : `str`
        E.g. ``n4``.
    title : `str`, optional
        The figure title; default names the detector.

    Returns
    -------
    `matplotlib.figure.Figure`
        Three panels over the set's visits in order: FWHM (px) with the gate's thresholds where its
        reasons name them, reserved RMS (px, log), and each species' lines per fiber. No-go images
        are ringed in red, check images in amber; sequences are separated and named.
    """
    from pfs.drp.qa.comparison.calibInputs import parseReason

    order = {v: i for i, v in enumerate(visits["visit"])}
    # One row per set visit, NaN where the detector has no image, so lines break over gaps.
    rows = (
        judged[judged["detector"] == detector]
        .set_index("visit")
        .reindex(list(order))
        .reset_index()
        .rename(columns={"index": "visit"})
    )
    x = rows["visit"].map(order)
    fig, axes = plt.subplots(3, 1, figsize=(max(6.0, 0.18 * len(order) + 2.0), 7.5), sharex=True)
    color = detector_palette.get(detector[0], _INK)
    ring = {Verdict.OUT: VERDICT_COLORS[Verdict.OUT], Verdict.CHECK: VERDICT_COLORS[Verdict.CHECK]}
    for ax, metric, label in zip(
        axes[:2], ("medFwhm", "reservedRms"), ("FWHM (px)", "reserved RMS (px)"), strict=True
    ):
        ax.plot(x, rows[metric], color=color, marker="o", markersize=4, linewidth=1)
        for verdict, c in ring.items():
            flagged = rows[rows["verdict"] == verdict]
            ax.plot(
                flagged["visit"].map(order),
                flagged[metric],
                linestyle="none",
                marker="o",
                markersize=10,
                markerfacecolor="none",
                markeredgecolor=c,
                markeredgewidth=1.5,
            )
        ax.set_ylabel(label, fontsize="small")
    axes[1].set_yscale("log")
    # The gate's FWHM thresholds, where a reason names them (each level once).
    drawn = set()
    for r in rows.dropna(subset=["gateReason"]).itertuples():
        for e in parseReason(r.gateReason):
            if e["metric"] == "medFWHM" and (e["level"], e["threshold"]) not in drawn:
                drawn.add((e["level"], e["threshold"]))
                axes[0].axhline(
                    e["threshold"],
                    color=VERDICT_COLORS[Verdict.OUT if e["level"] == "fail" else Verdict.CHECK],
                    linestyle="--",
                    linewidth=0.8,
                )
    sp = species[(species["arm"] == detector[0]) & (species["spectrograph"].astype(str) == detector[1:])]
    seen = set()
    for (name, _seq), g in sp.groupby(["species", "iic_sequence_id"]):
        g = g.sort_values("visit")
        axes[2].plot(
            g["visit"].map(order),
            g["linesPerFiber"],
            marker="o",
            markersize=3,
            linewidth=1,
            color=description_palette.get(name, _INK),
            label=None if name in seen else name,
        )
        seen.add(name)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("lines per fiber", fontsize="small")
    axes[2].legend(fontsize="x-small", frameon=False, ncol=4)
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels([str(v) for v in order], rotation=90, fontsize=6)
    for _, g in visits.groupby("iic_sequence_id", sort=False):
        left, right = order[g["visit"].iloc[0]], order[g["visit"].iloc[-1]]
        for ax in axes:
            ax.axvline(left - 0.5, color=_GRID, linewidth=1)
        axes[0].text(
            (left + right) / 2,
            1.02,
            g["lamp"].iloc[0] or "?",
            ha="center",
            va="bottom",
            fontsize=7,
            transform=axes[0].get_xaxis_transform(),
            rotation=0 if right - left >= 2 else 90,
        )
    for ax in axes:
        ax.grid(color=_GRID, linewidth=0.6)
        ax.tick_params(labelsize="x-small")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        title or f"{detector} through the set (red ring: no-go, amber: check)",
        x=0.01,
        ha="left",
        fontsize="medium",
    )
    fig.tight_layout()
    return fig


def plotLineResiduals(
    data: pd.DataFrame,
    *,
    title: str = "Line residuals against the visit's own detectorMap",
    thresholds: dict[str, dict[str, tuple[float | None, float | None]]] | None = None,
) -> Figure:
    """Show one detector image's line residuals against its own detectorMap, by wavelength and by fiber.

    Parameters
    ----------
    data : `pandas.DataFrame`
        ``dmLineResidualData`` rows of one detector image: ``fiberId``, ``wavelength``,
        ``description``, ``statusType`` (``USED``, ``RESERVED``), ``dxVisit``, ``dyVisit``.
    title : `str`, optional
        The figure title.
    thresholds : `dict`, optional
        From `pfs.drp.qa.comparison.calibInputs.lineThresholds`: the reserved-line RMS warn and
        fail limits per species and metric, drawn as bands about zero. A species with none is
        named as such in the legend. A key ``sequence`` holds the sequence rule's limits (the
        detector's median fit RMS over the sequence times the check and out factors), drawn the
        same way, for an image out on that rule.

    Returns
    -------
    `matplotlib.figure.Figure`
        Two rows (dx, dy in pixels) by two columns (against wavelength, against fiber); species by
        color, USED lines as faint dots, RESERVED lines as filled markers with a black edge, the
        warn (amber) and fail (red) RMS limits as dashed bands, each panel annotated per species with the reserved
        lines' median and robust RMS (0.741 times the interquartile range, as `dmLineResiduals`
        measures them, so the numbers are those of ``dmLineResidualMetrics``). Lamp lines only:
        trace rows (``Trace``) have no dy and are left out.
    """
    lines = data[data["description"] != "Trace"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col", sharey="row")
    for row, (column, label) in enumerate((("dxVisit", "dx (px)"), ("dyVisit", "dy (px)"))):
        for col, (against, xlabel) in enumerate((("wavelength", "wavelength (nm)"), ("fiberId", "fiberId"))):
            ax = axes[row, col]
            for name, g in lines.groupby("description"):
                color = description_palette.get(name, _INK)
                for statusType, g2 in g.groupby("statusType"):
                    reserved = statusType == "RESERVED"
                    ax.plot(
                        g2[against],
                        g2[column],
                        linestyle="none",
                        marker="o" if reserved else ".",
                        markersize=5 if reserved else 4,
                        markerfacecolor=color,
                        markeredgecolor=_INK if reserved else color,
                        markeredgewidth=0.8 if reserved else 0,
                        alpha=1.0 if reserved else 0.45,
                        zorder=3 if reserved else 2,
                        label=f"{name} {'reserved' if reserved else 'used'}"
                        if col == 0 and row == 0
                        else None,
                    )
            ax.axhline(0, color=_INK, linewidth=0.8)
            metric = "robustRmsDeltaX" if column == "dxVisit" else "robustRmsDeltaY"
            for name, limits in (thresholds or {}).items():
                warn, fail = limits.get(metric, (None, None))
                if warn is None and fail is None and col == 0 and row == 0:
                    ax.plot([], [], linestyle="none", label=f"{name}: no RMS threshold in the file")
                for value, level in ((warn, Verdict.CHECK), (fail, Verdict.OUT)):
                    if value is not None:
                        for sign in (-1, 1):
                            ax.axhline(
                                sign * value,
                                color=VERDICT_COLORS[level],
                                linestyle="--",
                                linewidth=0.8,
                                label=f"{name} {'warn' if level == Verdict.CHECK else 'fail'} RMS ±{value:.3f}"
                                if sign > 0 and col == 0 and row == (0 if metric.endswith("X") else 1)
                                else None,
                            )
            # The reserved lines' statistics per species, as dmLineResidualMetrics computes them.
            notes = []
            for name, g in lines[lines["statusType"] == "RESERVED"].groupby("description"):
                values = g[column].dropna()
                if len(values):
                    notes.append(
                        f"{name}: median {values.median():+.3f}, robust RMS {robustRms(values):.3f} px, n={len(values)}"
                    )
            if notes:
                ax.text(
                    0.99,
                    0.97,
                    "reserved lines\n" + "\n".join(notes),
                    ha="right",
                    va="top",
                    fontsize="x-small",
                    transform=ax.transAxes,
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "edgecolor": _GRID,
                        "alpha": 0.85,
                    },
                )
            ax.grid(color=_GRID, linewidth=0.6)
            ax.tick_params(labelsize="x-small")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 1:
                ax.set_xlabel(xlabel, fontsize="small")
            if col == 0:
                ax.set_ylabel(label, fontsize="small")
    axes[0, 0].legend(fontsize="x-small", frameon=False, ncol=3, loc="lower left")
    if thresholds:
        axes[1, 0].legend(fontsize="x-small", frameon=False, ncol=3, loc="lower left")
    fig.suptitle(title, x=0.01, ha="left", fontsize="medium")
    fig.tight_layout()
    return fig


def plotDetectorSeries(judged: pd.DataFrame, *, title: str = "Each detector through the set") -> Figure:
    """Show FWHM and reserved-line RMS of every detector image, in visit order, one column per arm.

    Parameters
    ----------
    judged : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.calibInputs.judge`.
    title : `str`, optional
        The figure title.

    Returns
    -------
    `matplotlib.figure.Figure`
        Two rows (FWHM in pixels; reserved RMS in pixels, log scale) by one column per arm; one
        line per detector, marked by spectrograph; *out* images ringed in red.
    """
    arms = [arm for arm in _ARM_ORDER if arm in set(judged["arm"])]
    fig, axes = plt.subplots(2, len(arms), figsize=(3.2 * len(arms) + 1, 5.2), sharex="col", squeeze=False)
    for column, arm in enumerate(arms):
        frame = judged[judged["arm"] == arm].sort_values("visit")
        color = detector_palette.get(arm, _INK)
        for row, (metric, label, log) in enumerate(
            (("medFwhm", "FWHM (px)", False), ("reservedRms", "reserved RMS (px)", True))
        ):
            ax = axes[row, column]
            for sp, g in frame.groupby("spectrograph"):
                ax.plot(
                    g["visit"],
                    g[metric],
                    color=color,
                    linewidth=1,
                    alpha=0.8,
                    marker=spectrograph_plot_markers.get(int(sp), "o"),
                    markersize=3.5,
                    label=f"{arm}{sp}",
                )
                bad = g[g["verdict"] == Verdict.OUT]
                ax.plot(
                    bad["visit"],
                    bad[metric],
                    linestyle="none",
                    marker="o",
                    markersize=9,
                    markerfacecolor="none",
                    markeredgecolor=VERDICT_COLORS[Verdict.OUT],
                    markeredgewidth=1.5,
                )
            if log:
                ax.set_yscale("log")
            if column == 0:
                ax.set_ylabel(label, fontsize="small")
            if row == 0:
                ax.set_title(f"arm {arm}", loc="left", fontsize="small")
                ax.legend(fontsize="x-small", frameon=False, ncol=2)
            ax.grid(color=_GRID, linewidth=0.6)
            ax.tick_params(labelsize="x-small")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[1, column].set_xlabel("visit", fontsize="small")
        axes[1, column].ticklabel_format(axis="x", style="plain", useOffset=False)
        axes[1, column].tick_params(axis="x", rotation=90)
    fig.suptitle(f"{title} (red ring: left out)", x=0.01, ha="left", fontsize="medium")
    fig.tight_layout()
    return fig


def plotLampTrends(trends: pd.DataFrame, *, title: str = "Lamps through their sequences") -> Figure:
    """Show each species' line count through its sequence, relative to the detector's best exposure.

    Parameters
    ----------
    trends : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.calibInputs.trends`.
    title : `str`, optional
        The figure title.

    Returns
    -------
    `matplotlib.figure.Figure`
        One panel per arm; one line per sequence and species (a lamp taken twice, full slit and
        b+m, gives two lines), over the exposure number ``k``; 1.0 is the best exposure. Warm-up
        rises to it, fading falls from it.
    """
    frame = trends[trends["species"] != "Trace"]
    arms = [arm for arm in _ARM_ORDER if arm in set(frame["arm"])] or ["-"]
    fig, axes = plt.subplots(1, len(arms), figsize=(3.0 * len(arms) + 1, 3.2), sharey=True, squeeze=False)
    for ax, arm in zip(axes[0], arms, strict=True):
        seen = set()
        for (_seq, lamp, species), g in frame[frame["arm"] == arm].groupby(
            ["iic_sequence_id", "lamp", "species"]
        ):
            g = g.sort_values("k")
            label = f"{species} ({lamp})"
            ax.plot(
                g["k"],
                g["linesRatio"],
                marker="o",
                markersize=4,
                linewidth=1.2,
                color=description_palette.get(species, _INK),
                label=None if label in seen else label,
            )
            seen.add(label)
        ax.axhline(1.0, color=_GRID, linewidth=1)
        ax.set_title(f"arm {arm}", loc="left", fontsize="small")
        ax.set_xlabel("exposure in sequence", fontsize="small")
        ax.set_xticks(sorted(frame["k"].unique()) or [1])
        ax.tick_params(labelsize="x-small")
        ax.legend(fontsize="x-small", frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].set_ylabel("lines, relative to the detector's best", fontsize="small")
    fig.suptitle(title, x=0.01, ha="left", fontsize="medium")
    fig.tight_layout()
    return fig


def plotCalibrationComparison(table: pd.DataFrame, detector: str, *, title: str | None = None) -> Figure:
    """Show one detector's line residuals against each candidate calibration, visit by visit.

    Parameters
    ----------
    table : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.calibValidate.compareCalibrations`, this detector's rows.
    detector : `str`
        E.g. ``b1``.
    title : `str`, optional
        The figure title; default names the detector.

    Returns
    -------
    `matplotlib.figure.Figure`
        One panel per compared metric (offsets and RMS against the calibration, reserved RMS
        against the visit's map), visits across in order; one color per candidate, one marker per
        species; the set's visits shaded.
    """
    from pfs.drp.qa.comparison.calibValidate import METRICS

    metrics = [m for m in METRICS.values() if m in set(table["metric"])]
    visits = sorted(table["visit"].unique())
    order = {v: i for i, v in enumerate(visits)}
    labels = list(dict.fromkeys(table["calibration"]))
    colors = ["#2a78d6", "#d03b3b", "#0ca30c", "#8c8c8c"]
    markers = dict(zip(sorted(table["species"].unique()), "osD^v<>Ph*", strict=False))
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(max(7.0, 0.22 * len(visits) + 2.5), 2.2 * len(metrics) + 1),
        sharex=True,
        squeeze=False,
    )
    inSet = sorted(table.loc[table["inSet"], "visit"].unique())
    for ax, metric in zip(axes[:, 0], metrics, strict=True):
        if inSet:
            ax.axvspan(order[inSet[0]] - 0.5, order[inSet[-1]] + 0.5, color=_GRID, alpha=0.5, linewidth=0)
        rows = table[table["metric"] == metric]
        for i, label in enumerate(labels):
            for species, g in rows[rows["calibration"] == label].groupby("species"):
                g = g.sort_values("visit")
                ax.plot(
                    g["visit"].map(order),
                    g["value"],
                    linestyle="none",
                    marker=markers.get(species, "o"),
                    markersize=4,
                    color=colors[i % len(colors)],
                    alpha=0.8,
                    label=f"{label} {species}" if ax is axes[0, 0] else None,
                )
        if metric.startswith("offset"):
            ax.axhline(0, color=_INK, linewidth=0.8)
        else:
            ax.set_yscale("log")
        ax.set_ylabel(f"{metric} (px)", fontsize="small")
        ax.grid(color=_GRID, linewidth=0.6)
        ax.tick_params(labelsize="x-small")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0, 0].legend(fontsize="x-small", frameon=False, ncol=4)
    axes[-1, 0].set_xticks(range(len(visits)))
    axes[-1, 0].set_xticklabels([str(v) for v in visits], rotation=90, fontsize=6)
    fig.suptitle(
        title or f"{detector}: line residuals against each candidate calibration (shaded: the set)",
        x=0.01,
        ha="left",
        fontsize="medium",
    )
    fig.tight_layout()
    return fig
