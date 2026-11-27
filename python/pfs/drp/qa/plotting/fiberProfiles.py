"""The wing light of a ``fiberProfiles`` calibration along the slits, drawn from stored numbers.

``plotWingExcess`` draws every fiber's wing excess (`pfs.drp.qa.metrics.fiberProfiles`) against its
fiber number, one panel per arm. Fibers are numbered along the slits, so a fault of a fiber shows as
a point standing above its neighbours, in the same place in each arm, and a calibration with none
is a flat line.

DataFrames in, a `matplotlib.figure.Figure` out: no Butler, stack or task import.
"""

import pandas as pd
from matplotlib.figure import Figure

from pfs.drp.qa.metrics.fiberProfiles import WING_EXCESS_LIMIT

__all__ = ["plotWingExcess"]

#: Arms in the order of the panels, with their names.
_ARMS = {"b": "blue", "r": "red", "n": "near-infrared", "m": "medium-resolution red"}

#: Colours: the fibers under the limit, measured fibers over it, copies over it, the calibration in force.
_UNDER, _MEASURED, _COPY, _PREVIOUS = "#a9a8a1", "#2a78d6", "#eb6834", "#0b0b0b"

#: Fibers per spectrograph, for the rules between the slits.
_FIBERS_PER_SPECTROGRAPH = 651


def plotWingExcess(
    fibers: pd.DataFrame,
    previous: pd.DataFrame | None = None,
    *,
    limit: float = WING_EXCESS_LIMIT,
    title: str = "Wing light above the neighbouring fibers",
    previousLabel: str = "in force",
) -> Figure:
    """Return each fiber's wing excess against its fiber number, one panel per arm.

    Parameters
    ----------
    fibers : `pandas.DataFrame`
        ``fiberProfilesQaFibers`` of the calibration: ``arm``, ``fiberId``, ``wingExcess`` and
        ``replaced``.
    previous : `pandas.DataFrame`, optional
        The same of the calibration in force: its fibers over the limit are drawn as open rings, so a
        fiber that was already over it is told from a new one.
    limit : `float`
        Wing excess above which a fiber is over the limit.
    title : `str`, optional
        The figure title.
    previousLabel : `str`, optional
        How the calibration in force is named in the legend.

    Returns
    -------
    `matplotlib.figure.Figure`
        A panel per arm the calibration holds, sharing both axes. Fibers under the limit are small
        grey points; over it, a measured profile is a filled circle and a copy a diamond.
    """
    arms = [arm for arm in _ARMS if not fibers.empty and arm in set(fibers["arm"].astype(str))]
    fig = Figure(figsize=(12, 2.4 * max(len(arms), 1) + 1.0))
    if not arms:
        ax = fig.subplots()
        ax.set_title(title, loc="left")
        ax.text(0.5, 0.5, "no profiles", ha="center", va="center", transform=ax.transAxes)
        return fig
    axes = fig.subplots(len(arms), 1, sharex=True, sharey=True, squeeze=False)[:, 0]
    top = max(0.3, float(fibers["wingExcess"].max()) * 1.3)
    for ax, arm in zip(axes, arms, strict=True):
        sub = fibers[fibers["arm"].astype(str) == arm]
        over = sub["wingExcess"] > limit
        copy = sub["replaced"].astype(bool)
        ax.scatter(
            sub["fiberId"][~over], sub["wingExcess"][~over], s=5, color=_UNDER, linewidths=0, rasterized=True
        )
        if previous is not None and not previous.empty:
            old = previous[(previous["arm"].astype(str) == arm) & (previous["wingExcess"] > limit)]
            ax.scatter(
                old["fiberId"],
                old["wingExcess"],
                s=70,
                facecolors="none",
                edgecolors=_PREVIOUS,
                linewidths=0.9,
                zorder=2,
                label=f"over the limit in {previousLabel}",
            )
        ax.scatter(
            sub["fiberId"][over & ~copy],
            sub["wingExcess"][over & ~copy],
            s=30,
            color=_MEASURED,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
            label="over the limit, measured",
        )
        ax.scatter(
            sub["fiberId"][over & copy],
            sub["wingExcess"][over & copy],
            s=30,
            color=_COPY,
            marker="D",
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
            label="over the limit, copy",
        )
        ax.axhline(limit, color="#52514e", linewidth=1)
        ax.set_yscale("symlog", linthresh=0.005, linscale=0.6)
        ax.set_ylim(-0.004, top)
        ticks = [tick for tick in (0, 0.01, 0.02, 0.05, 0.1, 0.2) if tick < top]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:g}" for tick in ticks])
        ax.set_xlim(0, 4 * _FIBERS_PER_SPECTROGRAPH)
        for edge in range(1, 4):
            ax.axvline(edge * _FIBERS_PER_SPECTROGRAPH, color="#bdbcb5", linewidth=1.2)
        nMeasured, nCopy = int((over & ~copy).sum()), int((over & copy).sum())
        ax.set_title(
            f"{_ARMS[arm]}: {nMeasured} measured and {nCopy} copies over {limit:g}", loc="left", fontsize=10
        )
        ax.grid(True, color="#e8e7e2", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for number in range(1, 5):
        axes[0].text(
            (number - 0.5) * _FIBERS_PER_SPECTROGRAPH,
            top * 0.8,
            f"spectrograph {number}",
            ha="center",
            va="top",
            color="#8a8982",
            fontsize=9,
        )
    axes[len(axes) // 2].set_ylabel("wing fraction above the neighbouring fibers")
    axes[-1].set_xlabel("fiber number, in order along the slits")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig
