"""Slit-offset plots, drawn from stored numbers after the fact.

``plotTrajectory`` draws the fiber-by-run matrix of cross-arm confirmed slit-offset outliers
(`pfs.drp.qa.metrics.slitOffsets.offsetTrajectory`): a heatmap whose rows are the fibers an
outlier in some run and whose columns are the runs, the cell the number of arms confirmed. Read
across a row it shows whether a fiber is chronic (confirmed every run), stepped (on and then stays)
or flickering (build-to-build fit wobble).

DataFrames in, a `matplotlib.figure.Figure` out: no Butler, stack or task import, so a figure is
drawn from a stored collection in a notebook or on the dashboard.
"""

import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

__all__ = ["plotTrajectory"]

#: Arms-confirmed to colour: 0 (not confirmed) to 4 (all arms).
_COLOURS = ["#f3f4f6", "#fde2b3", "#f6a94a", "#d9611f", "#7f2704"]


def plotTrajectory(trajectory: pd.DataFrame, *, title: str = "Slit-offset outliers across runs") -> Figure:
    """Return a heatmap of cross-arm confirmed slit-offset outliers, fiber by run.

    Parameters
    ----------
    trajectory : `pandas.DataFrame`
        `pfs.drp.qa.metrics.slitOffsets.offsetTrajectory`: ``spectrograph``, ``fiberId`` and one
        integer column per run (the arms confirmed, 0 to 4).
    title : `str`, optional
        The figure title.

    Returns
    -------
    `matplotlib.figure.Figure`
        One row per fiber, one column per run; the cell is shaded and labelled by the arms
        confirmed. Fibers are grouped by spectrograph, with a rule between groups.
    """
    runs = [c for c in trajectory.columns if c not in ("spectrograph", "fiberId")]
    if trajectory.empty or not runs:
        fig = Figure(figsize=(6, 2))
        fig.add_subplot(111).text(0.5, 0.5, "No confirmed slit-offset outliers", ha="center", va="center")
        return fig

    table = trajectory.sort_values(["spectrograph", "fiberId"]).reset_index(drop=True)
    data = table[runs].to_numpy(dtype=float)
    labels = [f"SM{int(s)} · {int(f)}" for s, f in zip(table["spectrograph"], table["fiberId"], strict=True)]

    cmap = ListedColormap(_COLOURS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    fig = Figure(figsize=(max(6.0, 0.7 * len(runs) + 3.0), max(2.5, 0.3 * len(labels) + 1.2)))
    ax = fig.add_subplot(111)
    ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("run (certified detectorMap_calib)")

    # Label each confirmed cell with the arm count, and rule between spectrographs.
    for i in range(len(labels)):
        for j in range(len(runs)):
            if data[i, j] > 0:
                ax.text(j, i, f"{int(data[i, j])}", ha="center", va="center", fontsize=7, color="black")
    boundaries = np.flatnonzero(np.diff(table["spectrograph"].to_numpy())) + 0.5
    for b in boundaries:
        ax.axhline(b, color="white", lw=2)

    ax.set_title(
        f"{title}\n(cell = arms confirmed; a fault at/before the slit shows in every arm)", fontsize=10
    )
    fig.tight_layout()
    return fig
