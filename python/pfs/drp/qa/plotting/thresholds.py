"""Plots of derived QA thresholds against the validation data.

One panel per population: the empirical CDF of the known-good values, the
suggested WARN and FAIL thresholds with the confidence interval on FAIL, and
the known-bad and unconfirmed values as a rug below. The inputs are the
outputs of `pfs.drp.qa.metrics.calibration`: `labelRows` for the values and
`calibrate` for the thresholds.
"""

import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

__all__ = ["plotThresholds"]

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
        rows for ``metric`` are drawn.
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
        subset = _groupRows(labelled, row)
        values = subset[metric].astype(float)
        subset = subset.assign(_value=values.abs() if absolute else values)
        _drawPanel(ax, subset, row)
    for ax in list(axes.flat)[len(rows) :]:
        ax.set_visible(False)

    fig.suptitle(f"{metric}{' (absolute)' if absolute else ''}: suggested thresholds")
    fig.legend(handles=_legendHandles(), loc="outside lower center", ncols=4, fontsize="small")
    return fig


def _groupRows(labelled: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Return the labelled rows in one population of the table."""
    columns = [column for column in str(row["groupBy"]).split("/") if column]
    mask = np.ones(len(labelled), dtype=bool)
    for column in columns:
        value = row[column]
        series = labelled[column]
        mask &= (series.isna() if pd.isna(value) else series == value).to_numpy()
    return labelled[mask]


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
