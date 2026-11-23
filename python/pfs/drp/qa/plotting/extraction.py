"""Draw the per-detector extraction-QA summary from a stored ``QaDict``.

``ExtractionQaTask`` measures, per GOOD fiber, the chi of the extraction residual
(the calibrated image minus the image reconstructed from the extracted spectra,
over ``sqrt(variance)``) and the residual fraction, and stores the per-fiber
arrays in ``extQaImage_pickle`` (a ``QaDict``). `plotExtractionStats` draws the
four-panel summary of those arrays — chiMed, chiStd, chiAtPeak and resFrac
against fiberId, coloured by the fiber's mean extracted flux — so the figure can
be made both inside the task (for the stored ``extQaStats`` PDF) and from the
stored pickle alone, outside the stack (the QA dashboard's per-visit page).

Stack-free: numpy and matplotlib only, no Butler, stack or task import;
``tests/plotting/test_imports.py`` enforces that.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["ExtractionPlotRanges", "plotExtractionStats"]


@dataclasses.dataclass(frozen=True)
class ExtractionPlotRanges:
    """The y-axis clip range ``(min, max)`` of each extraction-QA summary panel.

    A point outside a panel's range is drawn as a ``^``/``v`` marker pinned to the
    top/bottom edge, not dropped, so an outlier stays visible without stretching
    the scale. The defaults match ``ExtractionQaConfig``'s plot limits.

    Parameters
    ----------
    chiMed, chiStd, chiAtPeak, resFrac : `tuple` of `float`
        ``(min, max)`` for the median-chi, chi-standard-deviation, chi-at-peak and
        residual-fraction panels.
    """

    chiMed: tuple[float, float] = (-1.5, 1.5)
    chiStd: tuple[float, float] = (0.0, 3.5)
    chiAtPeak: tuple[float, float] = (-1.5, 1.5)
    resFrac: tuple[float, float] = (-5.0, 5.0)


def _title(dataId: Mapping[str, Any]) -> str:
    """Return the figure title for ``dataId`` (visit/arm/spectrograph/run), or ``""``."""
    if not dataId:
        return ""
    try:
        return ("Chi distribution (visit={visit} arm={arm} spectrograph={spectrograph})\nRUN={run}").format(
            **dataId
        )
    except KeyError:
        return ""


def _xPixelAxis(ax: Axes, fiberIds: np.ndarray, xa: np.ndarray) -> None:
    """Add a top axis labelling the bottom fiberId ticks with their mean X pixel."""
    twin = ax.twiny()
    twin.set_xlim(ax.get_xlim())
    ticks = ax.get_xticks()
    order = np.argsort(fiberIds)
    uniqueIds, index = np.unique(fiberIds[order], return_index=True)
    uniqueXa = xa[order][index]
    labels = np.interp(ticks, uniqueIds, uniqueXa)
    twin.set_xticks(ticks)
    twin.set_xticklabels([f"{pixel:.2f}" for pixel in labels], rotation=60)
    twin.set_xlabel("X (pix)")


def _panel(
    fig: Figure,
    ax: Axes,
    *,
    fiberIds: np.ndarray,
    xa: np.ndarray,
    values: np.ndarray,
    pfsArmAve: np.ndarray,
    targetMask: Mapping[Any, np.ndarray],
    ymin: float,
    ymax: float,
    hline: float,
    ylabel: str,
    textY: float,
) -> None:
    """Draw one summary panel: ``values`` against fiberId, coloured by ``pfsArmAve``.

    Each target type is scattered separately (so the legend counts them); points
    beyond ``[ymin, ymax]`` are pinned to the edge with ``^``/``v`` markers. A top
    axis maps fiberId to X pixel, and a text box reports the median and robust
    sigma over all fibers.
    """
    ax.axhline(hline, color="gray", linestyle="dashed")
    ax.set_ylim(ymin, ymax)
    vmin, vmax = np.nanmin(pfsArmAve), np.nanmax(pfsArmAve)
    mappable = None
    for target, mask in targetMask.items():
        count = int(np.sum(mask))
        if count == 0:
            continue
        fid, value, colour = fiberIds[mask], values[mask], pfsArmAve[mask]
        under = value < ymin
        over = value > ymax
        inside = ~(under | over)
        mappable = ax.scatter(
            fid[inside],
            value[inside],
            10.0,
            c=colour[inside],
            vmin=vmin,
            vmax=vmax,
            label=f"{target}: {count} fibers",
        )
        if np.any(over):
            ax.scatter(
                fid[over],
                np.full(int(np.sum(over)), ymax),
                marker="^",
                s=50,
                c=colour[over],
                edgecolors="k",
                rasterized=True,
                clip_on=False,
                vmin=vmin,
                vmax=vmax,
            )
        if np.any(under):
            ax.scatter(
                fid[under],
                np.full(int(np.sum(under)), ymin),
                marker="v",
                s=50,
                c=colour[under],
                edgecolors="k",
                rasterized=True,
                clip_on=False,
                vmin=vmin,
                vmax=vmax,
            )
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, location="right", fraction=0.04, alpha=1.0, label="pfsArmAve")
    _xPixelAxis(ax, fiberIds, xa)
    median = round(float(np.nanmedian(values)), 3)
    q3, q1 = np.nanpercentile(values, [75, 25])
    robustSigma = 0.741 * (q3 - q1)
    ax.text(0.05, textY, f"median={median:.3}, robustSigma={robustSigma:.3}", transform=ax.transAxes)
    ax.grid(color="gray", linestyle=":", linewidth=0.5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("fiberId")
    ax.legend()


def plotExtractionStats(
    qaStats: Mapping[str, Any],
    *,
    dataId: Mapping[str, Any] | None = None,
    ranges: ExtractionPlotRanges | None = None,
    title: str | None = None,
) -> Figure:
    """Draw the four-panel per-detector extraction-QA summary.

    Parameters
    ----------
    qaStats : mapping
        A detector's ``extQaImage_pickle`` (`QaDict`): the per-fiber arrays
        ``fiberIds``, ``xa`` (mean X pixel per fiber), ``pfsArmAve`` (mean
        extracted flux per fiber), ``chiMed``, ``chiStd``, ``chiAtPeak`` and
        ``resFrac``; ``targetMask`` (``{targetType: boolean array}``); and
        ``dataId``.
    dataId : mapping, optional
        The detector's data id for the title; falls back to ``qaStats["dataId"]``.
    ranges : `ExtractionPlotRanges`, optional
        The panels' y-axis clip ranges; the task-config defaults otherwise.
    title : `str`, optional
        A title to use instead of the one built from ``dataId``.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The 2x2 summary, as a bare figure (no pyplot state), so the caller owns
        and disposes of it.
    """
    ranges = ranges or ExtractionPlotRanges()
    if dataId is None:
        dataId = qaStats.get("dataId", {})
    fiberIds = np.asarray(qaStats["fiberIds"])
    xa = np.asarray(qaStats["xa"])
    pfsArmAve = np.asarray(qaStats["pfsArmAve"])
    targetMask = {target: np.asarray(mask, dtype=bool) for target, mask in qaStats["targetMask"].items()}

    fig = Figure(figsize=(12, 8), layout="constrained")
    axes = fig.subplots(2, 2)
    suptitle = _title(dataId) if title is None else title
    if suptitle:
        fig.suptitle(suptitle, y=1.05)

    panels = [
        (axes[0][0], "chiMed", ranges.chiMed, 0.0, "Chi (median)", 0.1),
        (axes[0][1], "chiStd", ranges.chiStd, 1.0, "Chi (standard deviation)", ranges.chiStd[1] * 0.0075),
        (axes[1][0], "chiAtPeak", ranges.chiAtPeak, 0.0, "Chi at profile peak pixels", 0.1),
        (axes[1][1], "resFrac", ranges.resFrac, 0.0, "Residual fraction (percent)", 0.1),
    ]
    for ax, key, (ymin, ymax), hline, ylabel, textY in panels:
        _panel(
            fig,
            ax,
            fiberIds=fiberIds,
            xa=xa,
            values=np.asarray(qaStats[key]),
            pfsArmAve=pfsArmAve,
            targetMask=targetMask,
            ymin=ymin,
            ymax=ymax,
            hline=hline,
            ylabel=ylabel,
            textY=textY,
        )
    return fig
