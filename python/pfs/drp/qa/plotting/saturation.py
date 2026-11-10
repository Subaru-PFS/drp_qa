"""The saturation of one detector image, drawn.

`plotSaturation` is the figure of the per-detector saturation report
(`pfs.drp.qa.saturationReport`): where on the detector the ``SAT`` pixels and
the saturated lines are, which lines they are, and the pixels about the worst
of them. Arrays and a DataFrame in, a figure out.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from pfs.drp.qa.metrics.saturation import saturatedLineSummary

__all__ = ["plotSaturation"]

_SAT_COLOR = "#D32F2F"
_LINE_COLOR = "#1565C0"


def plotSaturation(
    image: np.ndarray,
    saturated: np.ndarray,
    lines: pd.DataFrame,
    title: str = "",
    maxLines: int = 15,
    maxCutouts: int = 6,
    cutoutHalfSize: int = 12,
) -> Figure:
    """Draw one detector image's saturation.

    Parameters
    ----------
    image : `numpy.ndarray`, shape ``(height, width)``
        The calexp's image.
    saturated : `numpy.ndarray` [`bool`], shape ``(height, width)``
        Its ``SAT`` pixels.
    lines : `pandas.DataFrame`
        The lamp or sky lines on lit fibers: ``fiberId``, ``description``,
        ``wavelength``, ``x`` and ``y`` (the expected position) and
        ``saturated``.
    title : `str`, optional
        Figure title, e.g. the visit and camera.
    maxLines : `int`, optional
        Lines (species and wavelength) shown in the bar chart, the most
        saturated first.
    maxCutouts : `int`, optional
        Saturated lines shown as cutouts: those with the most saturated
        pixels about them.
    cutoutHalfSize : `int`, optional
        Half-size of a cutout, in pixels.

    Returns
    -------
    `matplotlib.figure.Figure`
        Top left, the detector: every ``SAT`` pixel, and a ring at each
        saturated line. Top right, the number of fibers each line is saturated
        on. Below, cutouts of the image about the worst lines, their ``SAT``
        pixels outlined and the expected position crossed.
    """
    saturated = np.asarray(saturated, dtype=bool)
    height, width = saturated.shape
    hit = lines[lines["saturated"].to_numpy(dtype=bool)]

    fig = plt.figure(figsize=(13, 10), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.7, 1.0])
    mapAxes, barAxes = top.subplots(1, 2, width_ratios=[1.0, 1.0])

    rows, columns = np.nonzero(saturated)
    mapAxes.scatter(columns, rows, s=2, color=_SAT_COLOR, marker="s", linewidths=0, label="SAT pixel")
    mapAxes.scatter(
        hit["x"],
        hit["y"],
        s=40,
        facecolors="none",
        edgecolors=_LINE_COLOR,
        linewidths=0.7,
        label="saturated line",
    )
    mapAxes.set(xlim=(0, width), ylim=(0, height), xlabel="x (pixels)", ylabel="y (pixels)", aspect="equal")
    mapAxes.set_title(f"{len(rows):,} SAT pixels, {len(hit):,} saturated lines", loc="left", fontsize=10)
    if len(rows) or len(hit):
        mapAxes.legend(loc="upper right", fontsize=8, markerscale=2)

    summary = saturatedLineSummary(lines).head(maxLines).iloc[::-1]
    if len(summary):
        labels = [f"{row.description} {row.wavelength:.2f}" for row in summary.itertuples()]
        positions = np.arange(len(summary))
        barAxes.barh(positions, summary["nFibers"], color="0.85", label="fibers with the line")
        barAxes.barh(positions, summary["nSaturated"], color=_SAT_COLOR, label="saturated")
        barAxes.set_yticks(positions, labels, fontsize=8)
        for position, row in zip(positions, summary.itertuples(), strict=True):
            barAxes.text(row.nSaturated, position, f" {row.nSaturated}", va="center", fontsize=8)
        barAxes.set_xlabel("fibers")
        barAxes.legend(loc="lower right", fontsize=8)
        barAxes.set_title("Saturated lines, by wavelength (nm)", loc="left", fontsize=10)
    else:
        barAxes.text(0.5, 0.5, "no saturated line", ha="center", va="center", transform=barAxes.transAxes)
        barAxes.set_axis_off()

    cutoutAxes = np.atleast_1d(bottom.subplots(1, maxCutouts))
    worst = _worstLines(saturated, hit, cutoutHalfSize, maxCutouts)
    for axes, line in zip(cutoutAxes, worst, strict=False):
        column, row = round(line.x), round(line.y)
        x0, x1 = max(column - cutoutHalfSize, 0), min(column + cutoutHalfSize + 1, width)
        y0, y1 = max(row - cutoutHalfSize, 0), min(row + cutoutHalfSize + 1, height)
        cutout = np.asarray(image[y0:y1, x0:x1], dtype=float)
        finite = cutout[np.isfinite(cutout)]
        scale = max(float(np.percentile(finite, 99.5)), 1.0) if finite.size else 1.0
        extent = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)
        axes.imshow(
            np.arcsinh(cutout / (0.02 * scale)), origin="lower", cmap="gray", extent=extent, aspect="equal"
        )
        mask = saturated[y0:y1, x0:x1]
        if mask.any() and not mask.all():
            axes.contour(
                np.arange(x0, x1), np.arange(y0, y1), mask.astype(float), levels=[0.5], colors=_SAT_COLOR
            )
        axes.plot(line.x, line.y, "+", color=_LINE_COLOR, markersize=10)
        axes.set_title(
            f"fiber {line.fiberId}\n{line.description} {line.wavelength:.2f}\n{int(mask.sum())} SAT pixels",
            fontsize=8,
        )
        axes.tick_params(labelsize=7)
    for axes in cutoutAxes[len(worst) :]:
        axes.set_axis_off()
    if len(worst):
        bottom.suptitle(
            "The saturated lines with most SAT pixels about them (asinh stretch; SAT outlined, + expected position)",
            fontsize=10,
        )
    if title:
        fig.suptitle(title, fontsize=12)
    return fig


def _worstLines(saturated: np.ndarray, hit: pd.DataFrame, halfSize: int, number: int) -> list:
    """Return the ``number`` saturated lines with the most saturated pixels within ``halfSize``, as tuples."""
    if len(hit) == 0:
        return []
    height, width = saturated.shape
    counts = []
    for line in hit.itertuples():
        column, row = round(line.x), round(line.y)
        counts.append(
            int(
                saturated[
                    max(row - halfSize, 0) : min(row + halfSize + 1, height),
                    max(column - halfSize, 0) : min(column + halfSize + 1, width),
                ].sum()
            )
        )
    order = np.argsort(counts, kind="stable")[::-1][:number]
    rows = list(hit.itertuples())
    return [rows[index] for index in order]
