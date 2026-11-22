"""Detector-map line-residual plots, drawn from ``dmLineResidualData`` after the fact.

One figure per detector image. ``dmLineResiduals`` stores numbers only; these plots
are the former ``dmResiduals`` task's 1D and 2D residual views, redrawn from the stored
rows so the pipeline renders nothing.

Each figure has two rows, the two residual components, and three columns:

* the residual against the calibration detectorMap (``*Calib``: drift) as a map over the
  detector face,
* the residual against the visit's own adjusted detectorMap (``*Visit``: the reserved-line
  scatter the gate judges) as a second face map, and
* a trend: the spatial residual per fiber, the wavelength residual per line, both
  references overlaid, annotated with each reference's reserved-line median and robust RMS.

Showing both references side by side separates a coherent drift (a whole-detector pattern
in the calib map) from random scatter (the visit map), which is the question an observer
asks of a detectorMap.

DataFrames in, a `matplotlib.figure.Figure` out: no Butler, stack or task import, so a
figure is drawn from a stored collection in a notebook or on the dashboard.
"""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from pfs.drp.qa.metrics.lineResiduals import robustRms
from pfs.drp.qa.plotting.palettes import description_palette, div_palette, opaqueColorbar

__all__ = ["plotDetectorResiduals"]

#: Gate band colours: warn amber, fail red (the status palette, as `plotting.calibInputs`).
_WARN, _FAIL = "#fab219", "#d03b3b"

#: Ink colour for the zero line, axis text and any species outside ``description_palette``.
_INK = "#222222"
#: The two references: column key in ``dmLineResidualData``, label, and the trend color.
_REFERENCES = (
    ("Calib", "vs calib (drift)", "#1f6fb4"),
    ("Visit", "vs visit (scatter)", "#c04a2b"),
)


def _symmetricLimit(values: np.ndarray, percentile: float = 98.0) -> float:
    """Return a symmetric color/axis limit from a robust percentile of the residuals.

    Outliers beyond it are shown clipped by the diverging map's over/under colors rather
    than flattening the scale. Falls back to a small non-zero limit when there is nothing
    finite to measure.
    """
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 0.05
    limit = float(np.nanpercentile(finite, percentile))
    return limit if limit > 0 else 0.05


def _residualMap(fig: Figure, ax, data: pd.DataFrame, column: str, limit: float, *, title: str):
    """Bin the detector face into a median-residual field, diverging about zero.

    A hexbin of the median residual per cell, rather than a scatter: a detector carries up
    to ~10⁵ trace rows, which overplot into an unreadable mass, and the map's job is the
    spatial structure, not the single line (the trend panel carries that). Empty cells (the
    gap between detector halves, unexposed corners) are left blank. Returns the mappable so
    a shared colorbar can be drawn.
    """
    good = data[np.isfinite(data[column])]
    mappable = ax.hexbin(
        good["x"],
        good["y"],
        C=good[column],
        reduce_C_function=np.median,
        gridsize=45,
        cmap=div_palette,
        vmin=-limit,
        vmax=limit,
        mincnt=1,
        linewidths=0.2,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_title(title, fontsize="small", weight="bold")
    ax.set_xlabel("x (pix)", fontsize="small")
    ax.set_ylabel("y (pix)", fontsize="small")
    ax.tick_params(labelsize="x-small")
    ax.set_aspect("equal", adjustable="box")
    return mappable


def _trend(
    ax,
    data: pd.DataFrame,
    against: str,
    columns: tuple[str, str],
    *,
    unit: str,
    title: str,
    limits: tuple[float | None, float | None] | None = None,
):
    """Per-``against`` median of each reference's residual, over the reserved lines by species.

    ``against`` is ``fiberId`` (spatial) or ``wavelength`` (wavelength); ``columns`` are the
    (calib, visit) residual columns. The faint points are the reserved lines' visit residual
    coloured by species (`description_palette`, the `plotting.calibInputs` convention); over
    them each reference's per-group median is a bold line. ``limits`` is the gate's (warn, fail)
    RMS in ``unit``, drawn as dashed bands either side of zero. A text box gives each reference's
    reserved median, robust RMS and outlier count (``dmLineResidualMetrics``' numbers).
    """
    _, visitColumn = columns
    # Reserved lines' visit residual, coloured by species: the per-line detail plotLineResiduals
    # shows. Traces are left to the map and the median line, so they do not overplot here.
    points = data[(data["description"] != "Trace") & (data["statusType"] == "RESERVED")]
    points = points[np.isfinite(points[visitColumn])]
    for name, group in points.groupby("description"):
        ax.plot(
            group[against],
            group[visitColumn],
            linestyle="none",
            marker=".",
            markersize=3,
            alpha=0.35,
            color=description_palette.get(name, _INK),
            zorder=2,
            label=name,
        )
    # Each reference's per-group median, bold over the points.
    stats, spread, fallback = [], 0.0, 0.0
    for (suffix, label, color), column in zip(_REFERENCES, columns, strict=True):
        good = data[np.isfinite(data[column])]
        if good.empty:
            continue
        grouped = good.groupby(against)[column].median().sort_index()
        ax.plot(
            grouped.index, grouped.to_numpy(), linewidth=1.2, alpha=0.95, color=color, zorder=4, label=label
        )
        # A robust spread for the y-scale, from all lines, so a frame with no reserved lines
        # (an UNFITTED fit) is still held; reserved lines set it when there are any.
        allVals = good[column].to_numpy()
        fallback = max(fallback, abs(float(np.median(allVals))) + robustRms(allVals))
        reserved = good.loc[good["statusType"] == "RESERVED", column].to_numpy()
        reserved = reserved[np.isfinite(reserved)]
        if reserved.size:
            median, rms = float(np.median(reserved)), robustRms(reserved)
            spread = max(spread, abs(median) + rms)
            # Outliers: reserved lines more than 2.5 robust RMS from the median.
            nOut = int(np.count_nonzero(np.abs(reserved - median) > 2.5 * rms)) if rms > 0 else 0
            stats.append(
                f"{suffix:<5} med {median:+.3f}  rms {rms:.3f}  >2.5σ {100 * nOut / reserved.size:.1f}%"
            )
    # Hold the y-scale to a robust range so one wild line cannot flatten the view; the gate
    # fail band stays in frame, and clipped outliers are still counted in the stats box.
    scale = spread or fallback
    yLimit = 4 * scale if scale > 0 else None
    if limits and limits[1] is not None:
        yLimit = max(yLimit or 0.0, 1.3 * limits[1])
    if yLimit:
        ax.set_ylim(-yLimit, yLimit)
    if limits:
        warn, fail = limits
        for value, color, name in ((warn, _WARN, "warn RMS"), (fail, _FAIL, "fail RMS")):
            if value is not None:
                ax.axhline(
                    value, color=color, linestyle="--", linewidth=0.9, zorder=3, label=f"{name} ±{value:.3g}"
                )
                ax.axhline(-value, color=color, linestyle="--", linewidth=0.9, zorder=3)
    ax.axhline(0, color=_INK, linewidth=0.8, zorder=1)
    if stats:
        ax.text(
            0.015,
            0.02,
            f"reserved ({unit}):\n" + "\n".join(stats),
            transform=ax.transAxes,
            fontsize="x-small",
            family="monospace",
            va="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
            zorder=5,
        )
    ax.set_title(title, fontsize="small", weight="bold")
    ax.set_xlabel(against, fontsize="small")
    ax.set_ylabel(f"residual ({unit})", fontsize="small")
    ax.tick_params(labelsize="x-small")
    ax.grid(True, alpha=0.15)
    if ax.get_legend_handles_labels()[0]:  # nothing to show on a trace-only (empty Δy) panel
        ax.legend(fontsize="x-small", loc="upper right", framealpha=0.9, ncol=2)


def plotDetectorResiduals(
    data: pd.DataFrame,
    *,
    title: str | None = None,
    limits: dict[str, tuple[float | None, float | None]] | None = None,
) -> Figure:
    """Draw one detector image's detector-map residuals from ``dmLineResidualData``.

    Parameters
    ----------
    data : `pandas.DataFrame`
        ``dmLineResidualData`` rows of one detector image: ``fiberId``, ``wavelength``,
        ``x``, ``y``, ``description``, ``statusType`` (``USED``/``RESERVED``), and the
        residuals ``dxCalib``, ``dyCalib``, ``dxVisit``, ``dyVisit`` in pixels and
        ``dyKmsCalib``, ``dyKmsVisit`` in km/s. The km/s columns are optional: without them
        the Δy trend falls back to pixels (an older collection, or a trace-only frame).
    title : `str`, optional
        The figure's suptitle; the detector and visit, e.g. ``"n4 152676"``.
    limits : `dict`, optional
        The gate's warn and fail RMS, as ``{"dx": (warn, fail), "dy": (warn, fail)}`` in each
        trend's unit (pixels for ``dx``; km/s or pixels for ``dy``, matching the trend). Drawn as
        dashed amber/red bands either side of zero. Either bound may be `None`; omit a key
        for no band.

    Returns
    -------
    `matplotlib.figure.Figure`
        Two rows (Δx spatial, from traces and line x; Δy wavelength, from lamp and sky
        lines) by three columns (calib-drift face map, visit-scatter face map, trend). The
        maps are a median-residual hexbin field diverging about zero on a symmetric robust
        scale; the trends show the reserved lines by species with each reference's median
        over them, the gate bands, and a reserved median/RMS/outlier box.
    """
    limits = limits or {}
    fig = Figure(figsize=(15, 8), layout="constrained")
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1.3], figure=fig)

    # Δy trend in km/s (the judged quantity) when the collection stored it; older collections,
    # and trace-only frames, have only the pixel residual, so fall back to it.
    if {"dyKmsCalib", "dyKmsVisit"} <= set(data.columns):
        yTrendColumns, yTrendUnit = ("dyKmsCalib", "dyKmsVisit"), "km/s"
    else:
        yTrendColumns, yTrendUnit = ("dyCalib", "dyVisit"), "pix"
    # Δx (spatial): trace centroids and line x, in pixels, against fiberId.
    # Δy (wavelength): lamp and sky lines only, against wavelength.
    rows = (
        ("dx", "Δx", "Δx (spatial, pix)", ("dxCalib", "dxVisit"), "fiberId", "pix", ("dxCalib", "dxVisit")),
        ("dy", "Δy", "Δy (wavelength, pix)", ("dyCalib", "dyVisit"), "wavelength", yTrendUnit, yTrendColumns),
    )
    for r, (key, short, rowLabel, mapColumns, against, trendUnit, trendColumns) in enumerate(rows):
        mapData = data if r == 0 else data[data["description"] != "Trace"]
        limit = _symmetricLimit(pd.concat([mapData[c] for c in mapColumns]).to_numpy())
        mapAxes, mappable = [], None
        for c, (_suffix, refLabel, _color) in enumerate(_REFERENCES):
            ax = fig.add_subplot(gs[r, c])
            mapAxes.append(ax)
            mappable = _residualMap(fig, ax, mapData, mapColumns[c], limit, title=f"{short} {refLabel}")
        with opaqueColorbar(mappable):
            fig.colorbar(mappable, ax=mapAxes, location="right", shrink=0.9, label=rowLabel, pad=0.01)
        _trend(
            fig.add_subplot(gs[r, 2]),
            data,
            against,
            trendColumns,
            unit=trendUnit,
            title=f"{short} by species, median per {against}",
            limits=limits.get(key),
        )

    if title:
        fig.suptitle(f"DetectorMap residuals — {title}", fontsize="large", weight="bold")
    return fig
