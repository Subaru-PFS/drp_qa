"""Plots of derived QA thresholds against the validation data.

`plotThresholds` draws one panel per percentile population: the empirical CDF
of the known-good values, the suggested WARN and FAIL thresholds with the
confidence interval on FAIL, and the known-bad and unconfirmed values as a rug
below. `plotTolerances` draws the tolerance populations, too many and too
small for a panel each: every value as its departure from its population's
known-good median. The inputs are the outputs of
`pfs.drp.qa.metrics.calibration`: `labelRows` for the values and `calibrate`
for the thresholds.
"""

import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pfs.drp.qa.metrics.calibration import selectGroup

__all__ = ["plotThresholds", "plotTolerances"]

_STATUS_COLORS = {"PASS": "#4CAF50", "WARN": "#FFC107", "FAIL": "#F44336"}
_UNCONFIRMED_COLOR = "#757575"

#: Rug rows below the CDF, by ``validation`` label: (y, marker, color, filled).
_RUG = {
    "bad:FAIL": (-0.06, "x", _STATUS_COLORS["FAIL"], True),
    "bad:WARN": (-0.12, "^", _STATUS_COLORS["WARN"], True),
    "unconfirmed": (-0.18, "o", _UNCONFIRMED_COLOR, False),
}


def plotThresholds(
    labelled: pd.DataFrame,
    table: pd.DataFrame,
    metric: str,
    absolute: bool = False,
    ncols: int = 4,
    panelSize: tuple[float, float] = (3.8, 3.0),
) -> Figure:
    """Plot one metric's suggested thresholds against the validation data.

    Parameters
    ----------
    labelled : `pandas.DataFrame`
        Metrics rows with a ``validation`` column, from
        `pfs.drp.qa.metrics.calibration.labelRows`, and the grouping columns
        named in ``table["groupBy"]``.
    table : `pandas.DataFrame`
        Suggestions from `pfs.drp.qa.metrics.calibration.calibrate`; only the
        percentile rows for ``metric`` are drawn (`plotTolerances` draws the
        others).
    metric : `str`
        The metric to plot.
    absolute : `bool`, optional
        Plot absolute values, as for a metric calibrated on them.
    ncols : `int`, optional
        Panels per row.
    panelSize : `tuple` [`float`, `float`], optional
        Size of one panel in inches.

    Returns
    -------
    `matplotlib.figure.Figure`
        One panel per population. A panel title ends in a warning when the
        suggestion is unreliable, unbounded or degenerate, or the known-bad
        check fails.

    Raises
    ------
    ValueError
        If ``table`` has no rows for ``metric``.
    """
    rows = table[table["metric"] == metric]
    if "method" in rows:
        rows = rows[rows["method"] != "tolerance"]
    if rows.empty:
        raise ValueError(f"No suggestions for {metric!r} in the table")

    ncols = max(1, min(ncols, len(rows)))
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panelSize[0] * ncols, panelSize[1] * nrows),
        squeeze=False,
        layout="constrained",
    )
    for ax, (_, row) in zip(axes.flat, rows.iterrows(), strict=False):
        subset = selectGroup(labelled, row)
        values = subset[metric].astype(float)
        subset = subset.assign(_value=values.abs() if absolute else values)
        _drawPanel(ax, subset, row)
    for ax in list(axes.flat)[len(rows) :]:
        ax.set_visible(False)

    fig.suptitle(f"{metric}{' (absolute)' if absolute else ''}: suggested thresholds")
    fig.legend(handles=_legendHandles(), loc="outside lower center", ncols=4, fontsize="small")
    return fig


def plotTolerances(
    labelled: pd.DataFrame,
    table: pd.DataFrame,
    metric: str,
    panelBy: tuple[str, ...] = ("arm", "obsType"),
    ncols: int = 2,
    panelSize: tuple[float, float] = (7.5, 3.2),
    limit: float = 12.0,
) -> Figure:
    """Plot one metric's tolerance thresholds against the validation data.

    Each value is drawn as its departure, in percent, from the known-good
    median of its population (`MetricSpec.toleranceBy`), so every population
    shares one axis and its WARN and FAIL are short bars at the same height.
    One panel per value of ``panelBy``; along x, the rest of the population
    (e.g. spectrograph and lamp).

    Parameters
    ----------
    labelled : `pandas.DataFrame`
        Metrics rows with a ``validation`` column, from
        `pfs.drp.qa.metrics.calibration.labelRows`, and the grouping columns
        named in ``table["groupBy"]``.
    table : `pandas.DataFrame`
        Suggestions from `pfs.drp.qa.metrics.calibration.calibrate`; only the
        tolerance rows for ``metric`` with known-good values are drawn.
    metric : `str`
        The metric to plot.
    panelBy : `tuple` [`str`], optional
        Population columns that get a panel each.
    ncols : `int`, optional
        Panels per row.
    panelSize : `tuple` [`float`, `float`], optional
        Size of one panel in inches.
    limit : `float`, optional
        Half-height of the y axis, in percent. Values beyond it (a defocused
        spectrograph reads +24 to +41 %) are drawn at the edge, hollow.

    Returns
    -------
    `matplotlib.figure.Figure`
        One panel per ``panelBy`` value. A tick label ends in ``!`` when its
        population's known-bad check fails.

    Raises
    ------
    ValueError
        If ``table`` has no tolerance rows with known-good values for ``metric``.
    """
    rows = table[(table["metric"] == metric) & (table.get("method") == "tolerance") & (table["nGood"] > 0)]
    if rows.empty:
        raise ValueError(f"No tolerance suggestions for {metric!r} in the table")
    groupBy = [column for column in str(rows["groupBy"].iloc[0]).split("/") if column]
    panelColumns = [column for column in panelBy if column in groupBy]
    rest = [column for column in groupBy if column not in panelColumns]
    panels = list(rows.groupby(panelColumns, sort=True)) if panelColumns else [((), rows)]

    ncols = max(1, min(ncols, len(panels)))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panelSize[0] * ncols, panelSize[1] * nrows),
        squeeze=False,
        layout="constrained",
    )
    jitter = np.random.default_rng(0)
    for ax, (key, panelRows) in zip(axes.flat, panels, strict=False):
        labels = []
        for position, (_, row) in enumerate(panelRows.iterrows()):
            subset = selectGroup(labelled, row)
            percent = 100.0 * (subset[metric].astype(float) / row["median"] - 1.0)
            for label, (marker, color, filled) in _TOLERANCE_MARKERS.items():
                values = percent[(subset["validation"] == label).to_numpy()].dropna().to_numpy()
                if values.size == 0:
                    continue
                x = position + jitter.uniform(-0.25, 0.25, values.size)
                inside = np.abs(values) <= limit
                y = np.clip(values, -limit, limit)
                for mask, hollow in ((inside, not filled), (~inside, True)):
                    if not mask.any():
                        continue
                    style = (
                        {"facecolors": "none", "edgecolors": color}
                        if hollow and marker != "x"
                        else {"color": color}
                    )
                    ax.scatter(x[mask], y[mask], marker=marker, s=14, linewidths=0.8, clip_on=False, **style)
            for level, name in ((row["warn"], "WARN"), (row["fail"], "FAIL")):
                height = 100.0 * (level / row["median"] - 1.0)
                ax.hlines(height, position - 0.4, position + 0.4, color=_STATUS_COLORS[name], lw=1.5)
            bad = pd.notna(row["badOk"]) and not row["badOk"]
            labels.append(
                " ".join(_format(row[column]) for column in rest if row[column] != "") + ("!" if bad else "")
            )
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize="x-small")
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylim(-limit, limit)
        ax.set_ylabel("from reference median (%)")
        ax.grid(axis="y", alpha=0.3)
        key = key if isinstance(key, tuple) else (key,)
        ax.set_title(" ".join(_format(value) for value in key), fontsize="small", loc="left")
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)

    tolerance = rows.iloc[0]
    fig.suptitle(
        f"{metric}: departure from the known-good median of each "
        f"{'/'.join(groupBy)} population, WARN/FAIL at "
        f"{100 * (tolerance['warn'] / tolerance['median'] - 1):.0f}/"
        f"{100 * (tolerance['fail'] / tolerance['median'] - 1):.0f} %"
    )
    fig.legend(handles=_toleranceHandles(limit), loc="outside lower center", ncols=4, fontsize="small")
    return fig


def _format(value) -> str:
    """Return a population value for a label: 2, not the 2.0 a column with nulls holds."""
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


#: Markers of `plotTolerances`, by ``validation`` label: (marker, color, filled).
_TOLERANCE_MARKERS = {
    "good": ("o", "0.25", True),
    "unconfirmed": ("o", _UNCONFIRMED_COLOR, False),
    "bad:WARN": ("^", _STATUS_COLORS["WARN"], True),
    "bad:FAIL": ("x", _STATUS_COLORS["FAIL"], True),
}


def _toleranceHandles(limit: float) -> list:
    """Return `plotTolerances`'s legend handles."""
    handles = [
        Line2D([], [], ls="", marker="o", color="0.25", markersize=4, label="known good"),
        Line2D([], [], color=_STATUS_COLORS["WARN"], lw=1.5, label="WARN"),
        Line2D([], [], color=_STATUS_COLORS["FAIL"], lw=1.5, label="FAIL"),
        Line2D([], [], ls="", marker="x", color=_STATUS_COLORS["FAIL"], label="known bad, expects FAIL"),
        Line2D([], [], ls="", marker="^", color=_STATUS_COLORS["WARN"], label="known bad, expects WARN"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=_UNCONFIRMED_COLOR,
            label="unconfirmed",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            markerfacecolor="none",
            markeredgecolor="0.25",
            label=f"beyond ±{limit:g} %",
        ),
    ]
    return handles


def _drawPanel(ax, subset: pd.DataFrame, row: pd.Series) -> None:
    """Draw one population's CDF, thresholds and rug."""
    good = np.sort(subset.loc[subset["validation"] == "good", "_value"].dropna().to_numpy())
    if good.size:
        ax.step(good, np.arange(1, good.size + 1) / good.size, where="post", color="0.2", lw=1.2)

    for label, (y, marker, color, filled) in _RUG.items():
        values = subset.loc[subset["validation"] == label, "_value"].dropna()
        if values.empty:
            continue
        style = {"color": color} if filled else {"facecolors": "none", "edgecolors": color}
        ax.scatter(
            values, np.full(len(values), y), marker=marker, s=24, linewidths=1.0, clip_on=False, **style
        )

    # After everything else, so that an unbounded interval spans the final x range.
    if row["nGood"] > 0:
        ax.axvline(row["warn"], color=_STATUS_COLORS["WARN"], lw=1.5)
        ax.axvline(row["fail"], color=_STATUS_COLORS["FAIL"], lw=1.5)
        low, high = row["failLow"], row["failHigh"]
        ax.axvspan(
            low if np.isfinite(low) else ax.get_xlim()[0],
            high if np.isfinite(high) else ax.get_xlim()[1],
            color=_STATUS_COLORS["FAIL"],
            alpha=0.12,
            lw=0,
        )

    ax.set_ylim(-0.22, 1.02)
    ax.set_yticks([0.0, 0.5, 0.95])
    ax.axhline(0.0, color="0.7", lw=0.5)
    ax.set_ylabel("CDF (known good)")
    ax.grid(alpha=0.3)
    ax.set_title(_panelTitle(row), fontsize="small", loc="left")


def _panelTitle(row: pd.Series) -> str:
    """Return a panel title: population, sample and any reason for doubt."""
    title = f"{row['group']}  n={row['nGood']} ({row['nGoodVisits']} visits)"
    if row["nGood"] == 0:
        return f"{title}\nno known-good data"
    title += f"\nwarn={row['warn']:.3g} fail={row['fail']:.3g}"
    flags = [
        name
        for name, bad in (
            ("UNRELIABLE", not row["reliable"]),
            ("UNBOUNDED", not row["failBounded"]),
            ("DEGENERATE", row["degenerate"]),
            ("BAD CHECK FAILED", pd.notna(row["badOk"]) and not row["badOk"]),
        )
        if bad
    ]
    return f"{title}\n{' '.join(flags)}" if flags else title


def _legendHandles() -> list:
    """Return the figure legend's handles."""
    return [
        Line2D([], [], color="0.2", lw=1.2, label="known good (CDF)"),
        Line2D([], [], color=_STATUS_COLORS["WARN"], lw=1.5, label="WARN"),
        Line2D([], [], color=_STATUS_COLORS["FAIL"], lw=1.5, label="FAIL"),
        Patch(color=_STATUS_COLORS["FAIL"], alpha=0.12, label="95% CI on FAIL percentile"),
        Line2D([], [], ls="", marker="x", color=_STATUS_COLORS["FAIL"], label="known bad, expects FAIL"),
        Line2D([], [], ls="", marker="^", color=_STATUS_COLORS["WARN"], label="known bad, expects WARN"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=_UNCONFIRMED_COLOR,
            label="unconfirmed",
        ),
    ]
