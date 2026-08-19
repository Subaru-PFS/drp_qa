"""Time-series plots for imageQualityQa metrics.

Reads concatenated ``iqQaMetrics`` DataFrames (one row per visit / arm /
spectrograph quantum) and produces multi-panel figures showing FWHM,
flexure (dxCenter), and pass/fail status across visits.

The layout is horizontal: visits on the x-axis with exposure-type labels,
each metric is a stacked row panel.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pfs.drp.qa.utils.plotting import detector_palette, spectrograph_plot_markers

__all__ = ["plotIqTimeSeries"]

_STATUS_COLORS = {"PASS": "#4CAF50", "WARN": "#FFC107", "FAIL": "#F44336"}
_STATUS_INT = {"PASS": 0, "WARN": 1, "FAIL": 2}

_ARM_ORDER = ["b", "r", "n", "m"]
_SPEC_ORDER = [1, 2, 3, 4]
_SPEC_LINE_STYLES = {1: "-", 2: "--", 3: ":", 4: "-."}

_FLAG_BITS = [
    ("pctLowSN", "Low S/N", "#795548"),
    ("pctMeasFail", "Meas fail", "#00BCD4"),
    ("pctBroad", "Broad", "#2196F3"),
    ("pctNotVisible", "NotVisible", "#9E9E9E"),
    ("pctRejected", "Rejected", "#F44336"),
    ("pctBlend", "Blend", "#FF9800"),
    ("pctSuspect", "Suspect", "#9C27B0"),
]


def _detectorLabel(row):
    try:
        spec = int(row["spectrograph"]) if pd.notna(row["spectrograph"]) else row["spectrograph"]
    except (ValueError, TypeError):
        spec = row["spectrograph"]
    return f"{row['arm']}{spec}"


def _detectorOrder():
    return [f"{a}{s}" for a in _ARM_ORDER for s in _SPEC_ORDER]


def _visitLabel(visit, visitInfo):
    """Build a compact x-axis label like '140123 (Hg)' when metadata is available."""
    obsType, seqName = visitInfo.get(visit, ("", ""))
    abbr = ""
    if obsType == "trace":
        abbr = "Tr"
    elif seqName:
        lamp = seqName.split(":", 1)[-1].strip() if ":" in seqName else seqName
        if lamp == "Argon":
            abbr = "Ar"
        elif lamp == "Xenon":
            abbr = "Xe"
        elif lamp == "Neon":
            abbr = "Ne"
        elif lamp == "Krypton":
            abbr = "Kr"
        elif lamp == "HgCd":
            abbr = "Hg"
        else:
            abbr = lamp
    elif obsType and obsType != "unknown":
        abbr = obsType

    # Clean visit name (remove trailing .0)
    try:
        v_str = str(int(visit))
    except (ValueError, TypeError):
        v_str = str(visit)

    if abbr:
        return f"{v_str} ({abbr})"
    return v_str


def plotIqTimeSeries(
    metrics: pd.DataFrame,
    fwhmWarnThreshold: float = 3.2,
    fwhmFailThreshold: float = 3.5,
    dxCenterWarnThreshold: float = 1.0,
    dxCenterFailThreshold: float = 2.0,
    excludeObsTypes: list[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Create a horizontal time-series summary of image quality metrics.

    Visits run left-to-right on the shared x-axis with exposure-type
    labels.  Each metric is a stacked row panel.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        Concatenated ``iqQaMetrics`` rows.  Required columns: ``visit``,
        ``arm``, ``spectrograph``, ``medFwhm``, ``qaStatus``.  Optional
        columns (used if present): ``medDxCenter``, ``dxCenterRms``,
        ``pctFlagged``, ``pctLowSN``, ``pctMeasFail``,
        ``pctNotVisible``, ``pctBlend``, ``pctSuspect``,
        ``pctRejected``, ``pctBroad``, ``obsType``, ``seqName``.
    fwhmWarnThreshold, fwhmFailThreshold : `float`
        Horizontal threshold lines for the FWHM panel.
    dxCenterWarnThreshold, dxCenterFailThreshold : `float`
        Horizontal threshold lines for the dxCenter panel (drawn at ±value).
    excludeObsTypes : `list` [`str`] or `None`, optional
        Observation types to exclude from the plot (e.g. ``["trace"]``
        to hide quartz/trace visits that produce no useful metrics).
        Matched against the ``obsType`` column when present.

    Returns
    -------
    `matplotlib.figure.Figure`
    """
    df = metrics.copy()

    # Filter out rows where critical columns are missing or NaN (e.g. trailing garbage rows)
    for col in ["visit", "spectrograph", "arm"]:
        if col in df.columns:
            df = df.dropna(subset=[col])
            try:
                # Convert numeric fields to int where possible to prevent float formatting
                if col in ("visit", "spectrograph"):
                    df[col] = df[col].astype(int)
            except (ValueError, TypeError):
                pass

    if excludeObsTypes and "obsType" in df.columns:
        df = df[~df["obsType"].isin(excludeObsTypes)].copy()
    df["detector"] = df.apply(_detectorLabel, axis=1)

    hasDxCenter = "medDxCenter" in df.columns
    hasDxCenterRms = "dxCenterRms" in df.columns
    hasPctFlagged = "pctFlagged" in df.columns

    hasFlagBreakdown = False
    flagCols = [col for col, _, _ in _FLAG_BITS if col in df.columns]
    if flagCols and df[flagCols].notna().any().any():
        hasFlagBreakdown = True

    hasFitXRms = "fitXRms" in df.columns and df["fitXRms"].notna().any()
    hasFitYRms = "fitYRms" in df.columns and df["fitYRms"].notna().any()
    hasFitTraceXRms = "fitTraceXRms" in df.columns and df["fitTraceXRms"].notna().any()
    hasFitTraceYRms = "fitTraceYRms" in df.columns and df["fitTraceYRms"].notna().any()

    # Avoid duplicate panels if fitXRms and fitTraceXRms are identical
    if hasFitTraceXRms and hasFitXRms:
        valid_mask = df["fitXRms"].notna() & df["fitTraceXRms"].notna()
        if valid_mask.any() and np.allclose(
            df.loc[valid_mask, "fitXRms"], df.loc[valid_mask, "fitTraceXRms"]
        ):
            hasFitTraceXRms = False

    # Avoid duplicate panels if fitYRms and fitTraceYRms are identical
    if hasFitTraceYRms and hasFitYRms:
        valid_mask = df["fitYRms"].notna() & df["fitTraceYRms"].notna()
        if valid_mask.any() and np.allclose(
            df.loc[valid_mask, "fitYRms"], df.loc[valid_mask, "fitTraceYRms"]
        ):
            hasFitTraceYRms = False

    visits = sorted(df["visit"].unique())
    visitIdx = {v: i for i, v in enumerate(visits)}
    df["visitIdx"] = df["visit"].map(visitIdx)

    visitInfo = {}
    if "obsType" in df.columns:
        hasSeqName = "seqName" in df.columns
        for v in visits:
            vRows = df.loc[df["visit"] == v]
            obsVals = vRows["obsType"].dropna().unique()
            obsType = str(obsVals[0]) if len(obsVals) > 0 else ""
            seqName = ""
            if hasSeqName:
                seqVals = vRows["seqName"].dropna().unique()
                seqName = str(seqVals[0]) if len(seqVals) > 0 else ""
            visitInfo[v] = (obsType, seqName)

    panels = []
    if "medFwhm" in df.columns and df["medFwhm"].notna().any():
        panels.append(("medFwhm", "median FWHM (px)", fwhmWarnThreshold, fwhmFailThreshold, False))
    if hasDxCenter and df["medDxCenter"].notna().any():
        panels.append(("medDxCenter", "dxCenter (px)", dxCenterWarnThreshold, dxCenterFailThreshold, True))
    elif hasDxCenterRms and df["dxCenterRms"].notna().any():
        panels.append(("dxCenterRms", "dxCenter RMS (px)", None, None, False))
    if hasPctFlagged and df["pctFlagged"].notna().any():
        panels.append(("pctFlagged", "flagged lines (%)", None, None, False))

    if hasFitXRms:
        panels.append(("fitXRms", "fit xRMS (px)", None, None, False))
    if hasFitYRms:
        panels.append(("fitYRms", "fit yRMS (px)", None, None, False))
    if hasFitTraceXRms:
        panels.append(("fitTraceXRms", "fit trace xRMS (px)", None, None, False))
    if hasFitTraceYRms:
        panels.append(("fitTraceYRms", "fit trace yRMS (px)", None, None, False))

    # Fallback to medFwhm panel if no metrics have any non-NaN data at all
    if not panels:
        panels.append(("medFwhm", "median FWHM (px)", fwhmWarnThreshold, fwhmFailThreshold, False))

    nExtra = int(hasFlagBreakdown)
    nRows = len(panels) + nExtra + 1  # +1 for status heatmap
    fig, axes = plt.subplots(
        nRows,
        1,
        figsize=(14, 3 * nRows),
        sharex=True,
        layout="constrained",
    )
    if nRows == 1:
        axes = [axes]

    visitLabels = [_visitLabel(v, visitInfo) for v in visits]

    axIdx = 0
    for column, ylabel, warnThresh, failThresh, symmetric in panels:
        _plotMetricPanel(
            axes[axIdx],
            df,
            column,
            visits,
            visitIdx,
            ylabel=ylabel,
            warnThreshold=warnThresh,
            failThreshold=failThresh,
            symmetric=symmetric,
        )
        axIdx += 1

    if hasFlagBreakdown:
        _plotFlagBreakdown(axes[axIdx], df, visits, visitIdx)
        axIdx += 1

    _plotStatusHeatmap(axes[axIdx], df, visits)

    axes[-1].set_xticks(range(len(visits)))
    tick_labels = axes[-1].set_xticklabels(visitLabels, fontsize=7, rotation=90, ha="center")
    axes[-1].set_xlabel("visit")

    # Color-code the x-axis tick labels to distinguish trace vs arc
    for label, v in zip(tick_labels, visits, strict=False):
        obsType, _ = visitInfo.get(v, ("unknown", ""))
        if obsType == "trace":
            label.set_color("#2E7D32")  # Forest Green
        elif obsType == "arc":
            label.set_color("#1565C0")  # Dark Blue

    # Add subtle background vertical guide stripes for observation types
    for i, v in enumerate(visits):
        obsType, _ = visitInfo.get(v, ("unknown", ""))
        color = "#E8F5E9" if obsType == "trace" else "#E3F2FD"
        for ax in axes:
            ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.3, zorder=0)

    if title:
        fig.suptitle(title, fontsize=12, weight="bold")

    return fig


def _plotMetricPanel(
    ax,
    df,
    column,
    visits,
    visitIdx,
    ylabel="",
    warnThreshold=None,
    failThreshold=None,
    symmetric=False,
):
    """Scatter plot of a metric vs visit, one series per detector."""
    detOrder = _detectorOrder()

    for det in detOrder:
        sub = df[df["detector"] == det]
        if sub.empty:
            continue
        arm = sub["arm"].iloc[0]
        spec = int(sub["spectrograph"].iloc[0])
        color = detector_palette.get(arm, "gray")
        marker = spectrograph_plot_markers.get(spec, "o")

        # Sort values by visit index to ensure lines connect chronologically
        sub = sub.sort_values("visitIdx")

        # Plot thin connecting line
        ls = _SPEC_LINE_STYLES.get(spec, "-")
        ax.plot(
            sub["visitIdx"],
            sub[column],
            color=color,
            alpha=0.5,
            ls=ls,
            lw=1.2,
            zorder=9,
        )

        if column == "medDxCenter" and "dxCenterRms" in sub.columns:
            ax.errorbar(
                sub["visitIdx"],
                sub[column],
                yerr=sub["dxCenterRms"],
                fmt="none",
                ecolor=color,
                elinewidth=1.0,
                capsize=2,
                alpha=0.6,
                zorder=8,
            )

        ax.scatter(
            sub["visitIdx"],
            sub[column],
            c=color,
            marker=marker,
            s=30,
            alpha=0.8,
            label=det,
            zorder=10,
        )

    if warnThreshold is not None:
        ax.axhline(warnThreshold, color=_STATUS_COLORS["WARN"], ls="--", lw=1, alpha=0.8, zorder=5)
        if symmetric:
            ax.axhline(-warnThreshold, color=_STATUS_COLORS["WARN"], ls="--", lw=1, alpha=0.8, zorder=5)
    if failThreshold is not None:
        ax.axhline(failThreshold, color=_STATUS_COLORS["FAIL"], ls="--", lw=1, alpha=0.8, zorder=5)
        if symmetric:
            ax.axhline(-failThreshold, color=_STATUS_COLORS["FAIL"], ls="--", lw=1, alpha=0.8, zorder=5)

    if symmetric:
        ax.axhline(0, color="k", ls="-", lw=0.5, alpha=0.5, zorder=4)

    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(visits)))
    ax.set_xticklabels([], fontsize=7)
    ax.grid(True, color="k", ls="--", alpha=0.1)

    _addDetectorLegend(ax, df)


def _plotFlagBreakdown(ax, df, visits, visitIdx):
    """Per-arm grouped bars of status-bit flag percentages.

    Each visit gets one group of bars per arm, showing the fraction of
    lines with each reference-line status bit set.  Arms are identified
    by a small colored tick at the base of each arm slot.

    The bars are drawn side-by-side rather than stacked: a line can carry
    several status bits at once, and ``pctLowSN``/``pctMeasFail`` overlap
    with those same bits, so the percentages are not parts of a whole and
    stacking them would produce totals above 100 %.
    """
    presentBits = [(col, label, color) for col, label, color in _FLAG_BITS if col in df.columns]
    if not presentBits:
        return

    presentArms = [a for a in _ARM_ORDER if a in df["arm"].values]
    nArms = len(presentArms)
    groupWidth = 0.8
    armWidth = groupWidth / max(nArms, 1)
    barWidth = armWidth / len(presentBits)

    for aIdx, arm in enumerate(presentArms):
        armDf = df[df["arm"] == arm]
        armLeft = -groupWidth / 2 + armWidth * aIdx

        armMean = armDf.groupby("visitIdx")[[col for col, _, _ in presentBits]].mean()

        for vIdx in range(len(visits)):
            if vIdx not in armMean.index:
                continue
            row = armMean.loc[vIdx]
            for bIdx, (col, _label, color) in enumerate(presentBits):
                val = row.get(col, 0.0)
                if pd.isna(val):
                    val = 0.0
                ax.bar(
                    vIdx + armLeft + barWidth * (bIdx + 0.5),
                    val,
                    barWidth,
                    color=color,
                    edgecolor="none",
                )

            # Arm-colored marker at the base of each arm slot.
            armColor = detector_palette.get(arm, "gray")
            ax.plot(
                vIdx + armLeft + armWidth / 2,
                0,
                marker="^",
                ms=4,
                color=armColor,
                zorder=15,
                clip_on=False,
            )

    legendHandles = [Patch(facecolor=color, edgecolor="none", label=label) for _, label, color in presentBits]
    for arm in presentArms:
        (h,) = ax.plot([], [], marker="^", ls="none", ms=5, color=detector_palette.get(arm, "gray"))
        legendHandles.append(h)
        legendHandles[-1].set_label(f"{arm} arm")
    ax.legend(
        handles=legendHandles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        frameon=False,
    )

    ax.set_ylabel("status bits (%)")
    ax.set_xticks(range(len(visits)))
    ax.set_xticklabels([], fontsize=7)
    ax.grid(True, axis="y", color="k", ls="--", alpha=0.1)


def _plotStatusHeatmap(ax, df, visits):
    """Heatmap of qaStatus (PASS/WARN/FAIL) by visit and detector."""
    detOrder = _detectorOrder()
    presentDets = [d for d in detOrder if d in df["detector"].values]

    statusMap = df.drop_duplicates(subset=["detector", "visit"], keep="first").pivot(
        index="detector",
        columns="visit",
        values="qaStatus",
    )
    statusMap = statusMap.reindex(index=presentDets, columns=sorted(visits))

    grid = np.full(statusMap.shape, np.nan)
    for i, _det in enumerate(statusMap.index):
        for j, _vis in enumerate(statusMap.columns):
            val = statusMap.iloc[i, j]
            if pd.notna(val):
                grid[i, j] = _STATUS_INT.get(str(val), np.nan)

    cmap = ListedColormap([_STATUS_COLORS["PASS"], _STATUS_COLORS["WARN"], _STATUS_COLORS["FAIL"]])
    cmap.set_bad("0.85")

    ax.imshow(
        grid,
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
    )

    ax.set_yticks(range(len(presentDets)))
    ax.set_yticklabels(presentDets, fontsize=8)
    ax.set_ylabel("detector")

    for status, color in _STATUS_COLORS.items():
        ax.scatter([], [], c=color, s=60, marker="s", label=status)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        frameon=False,
    )


def _addDetectorLegend(ax, df):
    """Add a compact detector legend grouped by arm."""
    handles = []
    labels = []
    seen = set()
    present_detectors = set(df["detector"].unique()) if "detector" in df.columns else set()
    for arm in _ARM_ORDER:
        for spec in _SPEC_ORDER:
            det = f"{arm}{spec}"
            if det not in present_detectors or det in seen:
                continue
            seen.add(det)
            color = detector_palette.get(arm, "gray")
            marker = spectrograph_plot_markers.get(spec, "o")
            ls = _SPEC_LINE_STYLES.get(spec, "-")
            h = Line2D(
                [],
                [],
                color=color,
                marker=marker,
                ls=ls,
                lw=1.2,
                ms=5,
            )
            handles.append(h)
            labels.append(det)
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        ncol=1,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.5,
    )
