"""Image quality QA summary pipeline task.

Aggregates per-quantum summary metrics produced by `ImageQualityQaTask`
across all visit/arm/spectrograph quanta and produces pass/fail heatmaps
using leave-one-out (LOO) z-scores for arc visits and an unsaturated-fraction
excess test for trace visits.
"""

from typing import Iterable

import numpy as np
import pandas as pd
from lsst.pex.config import Field
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)
from lsst.pipe.base.connectionTypes import (
    Input as InputConnection,
    Output as OutputConnection,
)
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from pfs.drp.qa.storageClasses import MultipagePdfFigure

__all__ = ["ImageQualityQaSummaryTask"]


class ImageQualityQaSummaryConnections(
    PipelineTaskConnections,
    dimensions=("instrument",),
):
    """Connections for ImageQualityQaSummaryTask."""

    iqQaMetrics = InputConnection(
        name="iqQaMetrics",
        doc=(
            "Per-quantum summary metrics from ImageQualityQaTask."
            " One row per visit/arm/spectrograph quantum."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    iqQaSummaryPlot = OutputConnection(
        name="iqQaSummaryPlot",
        doc=(
            "Image quality summary plot: pass/fail heatmap, FWHM heatmap,"
            " deviation metrics, and failure table.  One page per arm."
        ),
        storageClass="MultipagePdfFigure",
        dimensions=("instrument",),
    )


class ImageQualityQaSummaryConfig(
    PipelineTaskConfig, pipelineConnections=ImageQualityQaSummaryConnections
):
    """Configuration for ImageQualityQaSummaryTask."""

    fwhmLooZThreshold = Field(
        dtype=float,
        default=2.0,
        doc=(
            "LOO z-score threshold above which a spectrograph's median FWHM"
            " is considered a FWHM failure."
        ),
    )
    fwhmLooZFloor = Field(
        dtype=float,
        default=0.05,
        doc=(
            "Minimum peer standard deviation (pixels) used as the floor when"
            " computing FWHM LOO z-scores.  Prevents artificially large"
            " z-scores when all peers agree very closely."
        ),
    )
    flagRateLooZThreshold = Field(
        dtype=float,
        default=2.0,
        doc=(
            "LOO z-score threshold above which a spectrograph's arc-line flag"
            " rate (pctFlagged) is considered anomalous."
        ),
    )
    flagRateLooZFloor = Field(
        dtype=float,
        default=2.0,
        doc=(
            "Minimum peer standard deviation (percentage points) used as the"
            " floor when computing flag-rate LOO z-scores."
        ),
    )
    unsatExcessThreshold = Field(
        dtype=float,
        default=1.0,
        doc=(
            "Unsaturated-fraction excess (percentage points above the peer"
            " mean) above which a spectrograph is considered out-of-focus"
            " for trace visits."
        ),
    )


# ── Colours ──────────────────────────────────────────────────────────────────
_TRACE_BG = "#c8dff5"
_FAIL_FWHM = "#d62728"
_FAIL_FLAG = "#ff7f0e"
_FAIL_BOTH = "#8b0000"
_PASS_ARC = "#2ca02c"


class ImageQualityQaSummaryTask(PipelineTask):
    """Aggregate per-quantum IQ metrics into pass/fail summary plots.

    Reads ``iqQaMetrics`` DataFrames produced by `ImageQualityQaTask` for all
    visit/arm/spectrograph quanta.  For arc visits, applies leave-one-out (LOO)
    z-score tests on both median FWHM and arc-line flag rate.  For trace visits,
    flags spectrographs whose unsaturated-fraction exceeds the peer mean by more
    than a configurable threshold (an in-focus detector concentrates flux,
    saturating more pixels and leaving fewer valid measurements).

    Produces a ``MultipagePdfFigure`` with one page per arm, each containing:

    1. Pass/fail heatmap (colour-coded by failure type).
    2. Median FWHM heatmap (arc visits).
    3. FWHM LOO z-score heatmap (arc visits).
    4. Flag-rate LOO z-score heatmap (arc visits).
    5. Trace unsaturated-fraction excess heatmap (trace visits).
    6. Failure summary table.
    """

    ConfigClass = ImageQualityQaSummaryConfig
    _DefaultName = "imageQualityQaSummary"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        inputs = butlerQC.get(inputRefs)
        try:
            outputs = self.run(**inputs)
        except Exception as e:
            self.log.error("ImageQualityQaSummaryTask failed: %s", e)
            raise
        else:
            butlerQC.put(outputs.iqQaSummaryPlot, outputRefs.iqQaSummaryPlot)

    def run(self, iqQaMetrics: Iterable[pd.DataFrame]) -> Struct:
        """Build pass/fail summary plots from aggregated per-quantum metrics.

        Parameters
        ----------
        iqQaMetrics : iterable of `pandas.DataFrame`
            One-row DataFrames from `ImageQualityQaTask`, each containing
            ``visit``, ``arm``, ``spectrograph``, ``medFwhm``, ``pctFlagged``,
            ``nLines``, and ``traceOnly``.

        Returns
        -------
        iqQaSummaryPlot : `MultipagePdfFigure`
            One page per arm in the data.
        """
        metrics = pd.concat(list(iqQaMetrics), ignore_index=True)
        self.log.info(
            "Building IQ summary from %d quanta across arms %s",
            len(metrics),
            sorted(metrics["arm"].unique()),
        )

        metrics = self._computePassFail(metrics)
        pdf = MultipagePdfFigure()
        for arm in sorted(metrics["arm"].unique()):
            fig = self._makeSummaryPage(metrics[metrics["arm"] == arm].copy(), arm=arm)
            pdf.append(fig)
            plt.close(fig)
        return Struct(iqQaSummaryPlot=pdf)

    # ── Pass/fail logic ───────────────────────────────────────────────────────

    def _looZScore(self, group: pd.DataFrame, col: str, floor: float) -> pd.Series:
        """Return LOO z-score for ``col`` within a visit×arm group.

        For each spectrograph, computes ``(val - peer_mean) / peer_std`` where
        the peers are the *other* spectrographs in the same visit and arm.
        ``peer_std`` is floored at ``floor`` to prevent spuriously large
        z-scores when peers agree very closely.
        """
        z = pd.Series(np.nan, index=group.index)
        for idx, row in group.iterrows():
            val = row[col]
            peers = group.loc[group.index != idx, col].dropna()
            if len(peers) < 2 or pd.isna(val):
                continue
            pm = peers.mean()
            ps = max(peers.std(), floor)
            z[idx] = (val - pm) / ps
        return z

    def _computePassFail(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Annotate metrics with LOO z-scores and pass/fail flags.

        Adds columns: ``looFwhmZ``, ``looFlagZ``, ``unsatExcess``,
        ``failFwhm``, ``failFlag``, ``failTrace``, ``fail``.
        """
        cfg = self.config
        metrics["looFwhmZ"] = np.nan
        metrics["looFlagZ"] = np.nan
        metrics["unsatExcess"] = np.nan

        arc = metrics[~metrics["traceOnly"]]
        for (visit, arm), grp in arc.groupby(["visit", "arm"]):
            idx = grp.index
            metrics.loc[idx, "looFwhmZ"] = self._looZScore(
                grp, "medFwhm", cfg.fwhmLooZFloor
            ).values
            metrics.loc[idx, "looFlagZ"] = self._looZScore(
                grp, "pctFlagged", cfg.flagRateLooZFloor
            ).values

        trace = metrics[metrics["traceOnly"]]
        for (visit, arm), grp in trace.groupby(["visit", "arm"]):
            peer_mean = grp["pctFlagged"].mean()
            metrics.loc[grp.index, "unsatExcess"] = grp["pctFlagged"] - peer_mean

        metrics["failFwhm"] = metrics["looFwhmZ"] > cfg.fwhmLooZThreshold
        metrics["failFlag"] = metrics["looFlagZ"] > cfg.flagRateLooZThreshold
        # For trace: more valid (less saturated) = out of focus
        metrics["failTrace"] = metrics["unsatExcess"] > cfg.unsatExcessThreshold
        metrics["fail"] = metrics["failFwhm"] | metrics["failFlag"] | metrics["failTrace"]
        return metrics

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _makeSummaryPage(self, data: pd.DataFrame, arm: str) -> plt.Figure:
        """Build a 6-panel summary figure for one arm."""
        arc_visits = sorted(data.loc[~data["traceOnly"], "visit"].unique())
        trace_visits = sorted(data.loc[data["traceOnly"], "visit"].unique())
        all_visits = arc_visits + trace_visits
        is_trace = [False] * len(arc_visits) + [True] * len(trace_visits)
        sms = sorted(data["spectrograph"].unique())

        fig, axes = plt.subplots(
            3, 2,
            figsize=(max(12, len(all_visits) * 0.9 + 4), 14),
            gridspec_kw={"height_ratios": [3, 3, 3], "hspace": 0.55, "wspace": 0.3},
        )

        self._plotPassFail(axes[0, 0], data, all_visits, is_trace, sms)
        self._plotFwhmHeatmap(axes[0, 1], data, arc_visits, sms)
        self._plotLooHeatmap(axes[1, 0], data, arc_visits, sms, "looFwhmZ",
                             "Arc FWHM LOO z-score  (red = FAIL)")
        self._plotLooHeatmap(axes[1, 1], data, arc_visits, sms, "looFlagZ",
                             f"Arc flag-rate LOO z-score  (orange = FAIL, floor={self.config.flagRateLooZFloor:.0f} pp)",
                             fail_col="failFlag", fail_color=_FAIL_FLAG, clim=(-5, 5))
        self._plotTraceHeatmap(axes[2, 0], data, trace_visits, sms)
        self._plotFailTable(axes[2, 1], data)

        fig.suptitle(
            f"Image Quality QA — arm {arm}  |  {len(arc_visits)} arc, {len(trace_visits)} trace visits",
            fontsize=11,
            fontweight="bold",
            y=0.99,
        )
        return fig

    def _xticklabels(self, visits: list, is_trace: list | None = None) -> list[str]:
        return [
            f"{'T' if (is_trace and is_trace[i]) else 'A'}\n{str(v)[4:]}"
            for i, v in enumerate(visits)
        ]

    def _setup_ax(self, ax: plt.Axes, visits: list, sms: list,
                  is_trace: list | None = None) -> None:
        n = len(visits)
        ax.set_xlim(0, n)
        ax.set_ylim(0, len(sms))
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_xticklabels(self._xticklabels(visits, is_trace), fontsize=7)
        ax.set_yticks(np.arange(len(sms)) + 0.5)
        ax.set_yticklabels([f"SM{s}" for s in sms], fontsize=8)

    def _get(self, data: pd.DataFrame, visit: int, sm: int, col: str):
        rows = data[(data["visit"] == visit) & (data["spectrograph"] == sm)]
        return rows[col].iloc[0] if len(rows) else np.nan

    def _plotPassFail(self, ax: plt.Axes, data: pd.DataFrame,
                      all_visits: list, is_trace: list, sms: list) -> None:
        ax.set_title("Pass / Fail  (FWHM LOO-z > 2σ  OR  flag-rate LOO-z > 2σ  OR  unsat excess)",
                     fontsize=8, fontweight="bold")
        self._setup_ax(ax, all_visits, sms, is_trace)
        for i, (v, itr) in enumerate(zip(all_visits, is_trace)):
            for j, sm in enumerate(sms):
                fail = self._get(data, v, sm, "fail")
                ff = self._get(data, v, sm, "failFwhm")
                fl = self._get(data, v, sm, "failFlag")
                ft = self._get(data, v, sm, "failTrace")
                if pd.isna(fail):
                    color = "#f0f0f0"
                    label = ""
                elif fail:
                    if ff and fl:
                        color = _FAIL_BOTH
                    elif fl or ft:
                        color = _FAIL_FLAG
                    else:
                        color = _FAIL_FWHM
                    label = "FAIL"
                else:
                    color = _TRACE_BG if itr else _PASS_ARC
                    label = "" if itr else "PASS"
                ax.add_patch(plt.Rectangle([i, j], 1, 1, color=color, lw=0.5, ec="#999"))
                ax.text(i + 0.5, j + 0.5, label, ha="center", va="center",
                        fontsize=5.5, fontweight="bold" if fail else "normal",
                        color="white" if fail else "#555")
        ax.legend(
            handles=[
                mpatches.Patch(color=_FAIL_FWHM, label="FAIL (FWHM)"),
                mpatches.Patch(color=_FAIL_FLAG, label="FAIL (flag-rate / trace)"),
                mpatches.Patch(color=_FAIL_BOTH, label="FAIL (both)"),
                mpatches.Patch(color=_PASS_ARC, label="PASS"),
                mpatches.Patch(color=_TRACE_BG, label="Trace visit"),
            ],
            loc="upper right", fontsize=6, framealpha=0.9,
        )

    def _plotFwhmHeatmap(self, ax: plt.Axes, data: pd.DataFrame,
                         arc_visits: list, sms: list) -> None:
        ax.set_title("Arc Median FWHM (px)  [flag=False, status=0]",
                     fontsize=8, fontweight="bold")
        self._setup_ax(ax, arc_visits, sms)
        arc_data = data[~data["traceOnly"]]
        vals = arc_data["medFwhm"].dropna()
        if vals.empty:
            return
        norm = Normalize(vals.quantile(0.05), vals.quantile(0.95))
        sm_obj = ScalarMappable(norm=norm, cmap="RdYlGn_r")
        sm_obj.set_array([])
        for i, v in enumerate(arc_visits):
            for j, sm in enumerate(sms):
                val = self._get(data, v, sm, "medFwhm")
                color = plt.cm.RdYlGn_r(norm(val)) if np.isfinite(val) else "#eee"
                ax.add_patch(plt.Rectangle([i, j], 1, 1, color=color, lw=0.5, ec="#999"))
                ax.text(i + 0.5, j + 0.5, f"{val:.2f}" if np.isfinite(val) else "",
                        ha="center", va="center", fontsize=6.5)
        ax.figure.colorbar(sm_obj, ax=ax, fraction=0.03, pad=0.04, label="FWHM (px)")

    def _plotLooHeatmap(self, ax: plt.Axes, data: pd.DataFrame,
                        arc_visits: list, sms: list, z_col: str, title: str,
                        fail_col: str = "failFwhm",
                        fail_color: str = _FAIL_FWHM,
                        clim: tuple = (-3, 3)) -> None:
        ax.set_title(title, fontsize=8, fontweight="bold")
        self._setup_ax(ax, arc_visits, sms)
        norm = Normalize(*clim)
        sm_obj = ScalarMappable(norm=norm, cmap="RdBu_r")
        sm_obj.set_array([])
        for i, v in enumerate(arc_visits):
            for j, sm in enumerate(sms):
                z = self._get(data, v, sm, z_col)
                fail = self._get(data, v, sm, fail_col)
                color = (fail_color if fail
                         else plt.cm.RdBu_r(norm(np.clip(z, *clim))) if np.isfinite(z)
                         else "#eee")
                ax.add_patch(plt.Rectangle([i, j], 1, 1, color=color, lw=0.5, ec="#999"))
                ax.text(i + 0.5, j + 0.5, f"{z:.1f}" if np.isfinite(z) else "",
                        ha="center", va="center", fontsize=6,
                        color="white" if fail else "black")
        ax.figure.colorbar(sm_obj, ax=ax, fraction=0.03, pad=0.04, label="LOO z-score")

    def _plotTraceHeatmap(self, ax: plt.Axes, data: pd.DataFrame,
                          trace_visits: list, sms: list) -> None:
        ax.set_title(
            f"Trace unsaturated-fraction excess  (red = FAIL > {self.config.unsatExcessThreshold:.1f}%)",
            fontsize=8, fontweight="bold",
        )
        if not trace_visits:
            ax.axis("off")
            ax.text(0.5, 0.5, "No trace visits", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#aaa")
            return
        self._setup_ax(ax, trace_visits, sms,
                       is_trace=[True] * len(trace_visits))
        norm = Normalize(-2, 10)
        sm_obj = ScalarMappable(norm=norm, cmap="RdYlGn_r")
        sm_obj.set_array([])
        for i, v in enumerate(trace_visits):
            for j, sm in enumerate(sms):
                exc = self._get(data, v, sm, "unsatExcess")
                fail = self._get(data, v, sm, "failTrace")
                color = (_FAIL_FWHM if fail
                         else plt.cm.RdYlGn_r(norm(np.clip(exc, -2, 10))) if np.isfinite(exc)
                         else "#eee")
                ax.add_patch(plt.Rectangle([i, j], 1, 1, color=color, lw=0.5, ec="#999"))
                ax.text(i + 0.5, j + 0.5, f"{exc:+.1f}%" if np.isfinite(exc) else "",
                        ha="center", va="center", fontsize=6,
                        color="white" if fail else "black")
        ax.figure.colorbar(sm_obj, ax=ax, fraction=0.03, pad=0.04,
                           label="Unsat excess (%)")

    def _plotFailTable(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        ax.axis("off")
        ax.set_title("Failure Summary", fontsize=9, fontweight="bold")
        failures = data[data["fail"]].sort_values(["visit", "spectrograph"])
        if failures.empty:
            ax.text(0.5, 0.5, "No failures", ha="center", va="center",
                    fontsize=12, transform=ax.transAxes)
            return
        rows = []
        for _, r in failures.iterrows():
            reasons = []
            if r["failFwhm"]:
                reasons.append(f"FWHM z={r['looFwhmZ']:.1f}")
            if r["failFlag"]:
                reasons.append(f"flag-rate z={r['looFlagZ']:.1f}")
            if r["failTrace"]:
                reasons.append(f"unsat excess={r['unsatExcess']:+.1f}%")
            visit_type = "Trace" if r["traceOnly"] else "Arc"
            rows.append([f"SM{int(r['spectrograph'])}", str(int(r["visit"])),
                         visit_type, ", ".join(reasons)])
        table = ax.table(
            cellText=rows,
            colLabels=["SM", "Visit", "Type", "Reason"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.4)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#333")
                cell.set_text_props(color="white", fontweight="bold")
            elif rows[row - 1][2] == "Trace":
                cell.set_facecolor(_TRACE_BG)
