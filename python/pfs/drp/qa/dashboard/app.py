"""The QA dashboard prototype: a Panel app over a Tier 1 table (PIPE2D-1924).

Three tabs:

Live
    The latest visit: its arm × spectrograph verdict grid, the recent visits,
    and the sequences they belong to. Over qadb (`live.QadbTier1Store`) it
    follows the visits as qaActor loads them; over a file it replays the
    table in visit order.
Run
    A visit range (or HST date range, when the table has visit times), the
    verdicts of every image in it, and trends of image quality, line counts,
    line residuals and detectorMap drift, each value drawn with its own
    thresholds. Tap a point to open its visit.
Visit
    One visit: the verdicts and every metric behind them, the pipeline's PDFs,
    and the Tier 2 line residuals rasterised server-side with Datashader.

Everything shown comes from the Tier 1 table (`store.Tier1Store`, or
`live.QadbTier1Store` read from qadb) or, for the
Visit tab's detail, from files fetched per visit into ``detailDir``; the app
computes no metric and holds no threshold of its own.
"""

import html
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

from pfs.drp.qa.dashboard.schema import IMAGE_STATUS_ORDER, NOTE, STATUS_ORDER, VERDICT
from pfs.drp.qa.dashboard.store import Tier1Store, visitTimeline
from pfs.drp.qa.metrics.registry import specFor

__all__ = ["Dashboard", "createApp"]

ARMS = ("b", "r", "n", "m")
SPECTROGRAPHS = (1, 2, 3, 4)
CAMERAS = tuple(f"{arm}{sm}" for arm in ARMS for sm in SPECTROGRAPHS)

#: Status colours (Okabe-Ito-like, distinguishable with the common colour-vision deficiencies).
STATUS_COLORS = {"PASS": "#3a9a5b", "WARN": "#e6a100", "FAIL": "#d1495b", "UNKNOWN": "#9aa0a6", "": "#c7ccd1"}

#: Trend panels: title, source, metric choices, and the line-residual reference/statusType.
TRENDS = (
    ("Image quality", "iqQaMetrics", ("medFwhm", "medDxCenter", "dxCenterRms", "pctFlagged"), None, None),
    ("Line counts", "iqQaMetrics", ("linesPerFiber", "nLines", "nLitFibers"), None, None),
    (
        "Line residuals (against the visit's detectorMap, RESERVED lines)",
        "dmLineResidualMetrics",
        ("robustRmsDeltaX", "robustRmsDeltaY"),
        "visit",
        "RESERVED",
    ),
    (
        "DetectorMap drift (against detectorMap_calib)",
        "dmLineResidualMetrics",
        ("medianDeltaX", "medianDeltaY"),
        "calib",
        "BOTH",
    ),
)

#: The pipeline's figures, by dataset type, as fetched into ``detailDir/<visit>/``.
PDF_TYPES = ("extQaStats", "extQaImage", "dmQaResidualPlot", "dmQaCombinedResidualPlot")

#: What the 2026-10-20 search found (PIPE2D-1924), shown where a visit has no detail files.
DETAIL_SEARCH = (
    "Searched 2026-10-20 on pfsa-usr02 `/work/datastore`: `u/wtg/qa-thresholds/010` holds "
    "`dmLineResidualData` and `iqQaData` but no PDFs (extractionQa and dmResiduals were not run "
    "there); `drpActor/reductions`, `qaActor/reductions` and `PFS/defaults` hold none of them for "
    "visits 140005, 140141 and 134336. `dmLineResidualData` was fetched for those three visits only."
)

_STATUS_RANK = {status: rank for rank, status in enumerate(IMAGE_STATUS_ORDER)}


def _plain(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with object strings: HoloViews reshapes columns, which Arrow-backed strings refuse."""
    strings = frame.select_dtypes(include=["string", "str"]).columns
    return frame.astype(dict.fromkeys(strings, object))


def _camera(frame: pd.DataFrame) -> pd.Series:
    return frame["arm"].astype(str) + frame["spectrograph"].astype(str)


def _badge(status: str, text: str | None = None) -> str:
    color = STATUS_COLORS.get(status, STATUS_COLORS[""])
    return (
        f'<span style="background:{color};color:white;border-radius:4px;padding:1px 7px;'
        f'font-weight:600;font-size:0.85em">{text or status}</span>'
    )


def _sliderEnd(visitIds: Sequence[int]) -> int:
    """Return the end of a slider over ``visitIds``: past the start, which a slider needs, with one visit."""
    return max(visitIds[-1], visitIds[0] + 1)


def verdictGrid(images: pd.DataFrame, size: int = 46) -> str:
    """Return an HTML arm × spectrograph grid of image statuses; reasons on hover."""
    byCamera = {
        f"{row.arm}{row.spectrograph}": row for row in images.itertuples(index=False)
    }  # one visit's images
    cells = ['<table style="border-collapse:separate;border-spacing:3px;font-family:monospace">']
    cells.append("<tr><td></td>" + "".join(f"<th>SM{sm}</th>" for sm in SPECTROGRAPHS) + "</tr>")
    for arm in ARMS:
        cells.append(f"<tr><th style='padding-right:4px'>{arm}</th>")
        for sm in SPECTROGRAPHS:
            row = byCamera.get(f"{arm}{sm}")
            if row is None:
                cells.append(
                    f'<td style="width:{size}px;height:{size * 0.6}px;border:1px dashed #c7ccd1;'
                    f'color:#9aa0a6;text-align:center" title="no image">–</td>'
                )
                continue
            title = html.escape(row.reasons or row.status)
            cells.append(
                f'<td style="width:{size}px;height:{size * 0.6}px;background:{STATUS_COLORS[row.status]};'
                f'color:white;text-align:center;border-radius:4px;font-size:0.8em" title="{title}">'
                f"{row.status[0]}</td>"
            )
        cells.append("</tr>")
    cells.append("</table>")
    return "".join(cells)


def _renumber(timeline: pd.DataFrame, visits) -> pd.DataFrame:
    """Return the rows of ``visits``, positions counted from 0 among them."""
    chosen = timeline[timeline["visit"].isin(visits)].reset_index(drop=True)
    return chosen.assign(position=np.arange(len(chosen)))


#: A gap of this many days or more between consecutive visits is labelled: a new run, or a block of one.
GAP_DAYS = 3


def _gapLabel(days: float) -> str:
    """Return `` (+113 d)`` for a gap of at least `GAP_DAYS`, else nothing."""
    return f" (+{days:.0f} d)" if np.isfinite(days) and days >= GAP_DAYS else ""


def _gaps(timeline: pd.DataFrame) -> pd.Series:
    """Return the days since the previous visit in ``timeline``; NaN for the first or without times."""
    return timeline["time"].diff().dt.total_seconds().div(86400.0)


def _timeTicks(timeline: pd.DataFrame, maxTicks: int = 30) -> list[tuple[int, str]]:
    """Return x ticks for visits in order: each visit's HST time when few, else each night's date.

    The first visit after a gap of `GAP_DAYS` or more is always labelled, with the gap: in visit
    order, months between runs take no space.
    """
    if timeline["night"].eq("").all():
        step = max(1, len(timeline) // maxTicks)
        return [(int(p), str(v)) for p, v in zip(timeline["position"], timeline["visit"], strict=True)][
            ::step
        ]
    newNight = timeline["night"] != timeline["night"].shift()
    gaps = _gaps(timeline)
    if len(timeline) <= 40:
        # The calendar date, not the night's: a visit at 06:13 belongs to the previous evening's night.
        return [
            (
                int(row.position),
                f"{row.time:%m-%d %H:%M}{_gapLabel(gap)}" if first and pd.notna(row.time) else row.clock,
            )
            for row, first, gap in zip(timeline.itertuples(), newNight, gaps, strict=True)
        ]
    starts = timeline.assign(gap=gaps)[newNight]
    isGap = starts["gap"].ge(GAP_DAYS)
    step = max(1, (~isGap).sum() // max(1, maxTicks - isGap.sum()))
    keep = isGap | (np.arange(len(starts)) % step == 0)
    return [
        (int(row.position), f"{row.night}{_gapLabel(row.gap)}")
        for row in starts[keep.to_numpy()].itertuples()
    ]


def _nightBreaks(timeline: pd.DataFrame) -> list[hv.VLine]:
    """Return a line before each night's first visit: dashed, or solid after a gap of `GAP_DAYS` or more."""
    gaps = _gaps(timeline)
    newNight = (timeline["night"] != timeline["night"].shift()) & (timeline["position"] > 0)
    newNight &= timeline["night"].ne("")
    return [
        hv.VLine(float(p) - 0.5).opts(color="#3c4043", line_width=3)
        if gap >= GAP_DAYS
        else hv.VLine(float(p) - 0.5).opts(color="#5f6368", line_width=1.5, line_dash="dashed")
        for p, gap in zip(timeline.loc[newNight, "position"], gaps[newNight], strict=True)
    ]


def statusHeatmap(
    images: pd.DataFrame, timeline: pd.DataFrame, height: int = 260
) -> tuple[hv.Overlay, hv.HeatMap]:
    """Return a visit × camera heat map of image statuses, in time order, and its heat map layer.

    ``timeline`` holds the visits to show (`visitTimeline`, renumbered from 0): the x axis is their
    order, labelled with times, with a dashed line where a night begins.
    """
    frame = images[images["visit"].isin(timeline["visit"])].assign(camera=_camera)
    frame = frame.merge(timeline[["visit", "position", "night", "clock"]], on="visit")
    frame = frame.assign(
        rank=frame["status"].map(_STATUS_RANK).astype(float),
        when=(frame["night"] + " " + frame["clock"]).str.strip(),
    )
    frame = _plain(frame)
    cmap = [STATUS_COLORS[status] for status in IMAGE_STATUS_ORDER]
    heatmap = hv.HeatMap(
        frame, kdims=["position", "camera"], vdims=["rank", "status", "visit", "when", "reasons"]
    ).opts(
        cmap=cmap,
        clim=(-0.5, len(cmap) - 0.5),
        colorbar=False,
        tools=["hover", "tap"],
        xrotation=60,
        height=height,
        responsive=True,
        invert_yaxis=True,
        xlabel=(
            "HST (dashed line: a new night; solid: a gap of days, labelled)"
            if len(timeline) <= 40
            else "night (dashed line: a new night; solid: a gap of days, labelled)"
        ),
        xlim=(-0.5, len(timeline) - 0.5),
        ylabel="",
        xticks=_timeTicks(timeline),
        line_color="white",
        line_width=1,
        hover_tooltips=[("visit", "@visit"), ("HST", "@when"), ("camera", "@camera"), ("status", "@status")],
    )
    return hv.Overlay([heatmap, *_nightBreaks(timeline)]), heatmap


def trendPlot(frame: pd.DataFrame, metric: str, timeline: pd.DataFrame, byTime: bool = False) -> hv.Overlay:
    """Return a metric's values in time, coloured by verdict, with each value's thresholds.

    Thresholds differ by population (arm, spectrograph, lamp, ...), so each value's own WARN and
    FAIL are drawn as dashes at its position rather than one line across the plot. The x axis is
    the visits in order, labelled by night, or, with ``byTime``, HST itself (runs months apart
    then sit far apart).
    """
    if frame.empty:
        return hv.Text(0, 0, "No values in this selection").opts(height=260, responsive=True)
    timeline = _renumber(timeline, frame["visit"].unique())
    frame = frame.drop(columns=["timeExpStart"], errors="ignore").merge(
        timeline[["visit", "position", "time", "night", "clock"]], on="visit"
    )
    if byTime and frame["time"].notna().any():
        frame = frame.assign(x=frame["time"])
        xlabel, ticks, breaks = "HST", None, []
    else:
        frame = frame.assign(x=frame["position"] + (frame["spectrograph"] - 2.5) * 0.08)
        xlabel, ticks, breaks = (
            "visits in order, by night (dashed line: a new night)",
            _timeTicks(timeline),
            _nightBreaks(timeline),
        )
    frame = frame.assign(
        camera=_camera(frame),
        status=frame["status"].replace("", "not judged"),
        when=(frame["night"] + " " + frame["clock"]).str.strip(),
    )
    frame = _plain(frame)
    colors = STATUS_COLORS | {"not judged": STATUS_COLORS[""]}
    vdims = [
        "value",
        "visit",
        "when",
        "camera",
        "status",
        "description",
        "seqName",
        "population",
        "warn",
        "fail",
    ]
    points = hv.Scatter(frame, "x", vdims).opts(
        color="status",
        cmap=colors,
        size=5,
        tools=["hover", "tap"],
        hover_tooltips=[
            ("visit", "@visit"),
            ("HST", "@when"),
            ("camera", "@camera"),
            ("value", "@value"),
            ("status", "@status"),
            ("species", "@description"),
        ],
        responsive=True,
        height=280,
        xlabel=xlabel,
        ylabel=metric,
        show_legend=True,
        legend_position="top_right",
        nonselection_alpha=0.3,
        **({"xticks": ticks, "xrotation": 60} if ticks else {}),
    )
    layers = [points]
    for level, color in (("warn", STATUS_COLORS["WARN"]), ("fail", STATUS_COLORS["FAIL"])):
        limits = frame[np.isfinite(frame[level])]
        if not limits.empty:
            layers.append(
                hv.Scatter(limits, "x", level).opts(marker="dash", size=9, color=color, alpha=0.8, angle=0)
            )
    return hv.Overlay([*layers, *breaks]).opts(active_tools=["box_zoom"])


@lru_cache(maxsize=4)
def readLines(path: Path) -> pd.DataFrame:
    """Return a fetched ``dmLineResidualData`` file, with a ``camera`` column; cached."""
    lines = pd.read_parquet(path)
    return lines.assign(camera=_camera(lines))


def residualRaster(lines: pd.DataFrame, reference: str = "Calib", component: str = "y"):
    """Return one detector's Tier 2 line residuals rasterised server-side: position, mean residual.

    Parameters
    ----------
    lines : `pandas.DataFrame`
        ``dmLineResidualData`` rows of one detector
        (`pfs.drp.qa.metrics.lineResiduals.LINE_COLUMNS`).
    reference : `str`
        ``"Calib"`` or ``"Visit"``: the ``d<x|y><reference>`` column drawn.
    component : `str`
        ``"x"`` or ``"y"``.
    """
    import datashader as ds
    from holoviews.operation.datashader import rasterize

    column = f"d{component}{reference}"
    finite = lines[np.isfinite(lines[column])]
    if finite.empty:
        return hv.Text(0, 0, f"No {column} values").opts(height=420, responsive=True)
    # Symmetric about zero, wide enough for this detector: a fixed range saturates a drifted one.
    limit = max(0.05, float(np.nanpercentile(np.abs(finite[column]), 98)))
    points = hv.Points(finite, ["x", "y"], [column])
    return rasterize(points, aggregator=ds.mean(column), width=400, height=400).opts(
        cmap="RdBu_r",
        clim=(-limit, limit),
        colorbar=True,
        responsive=True,
        height=480,
        tools=["hover"],
        clabel=f"mean {column} (pixels)",
        xlabel="x (pixels)",
        ylabel="y (pixels)",
    )


def judgedChecks(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return one visit's judged metric rows, one per check and camera, with a ``check`` label.

    A check is a metric of ``iqQaMetrics``, or a metric of one species, reference and line set of
    ``dmLineResidualMetrics``. ``pastFail`` is how far the value is beyond its FAIL threshold, in
    percent of it and in the bad direction (the absolute value for an ``absolute`` metric), and
    ``warnAt`` the same for the WARN threshold; NaN where the row carries no thresholds.
    """
    rows = metrics[(metrics["metric"] != VERDICT) & metrics["status"].isin(STATUS_ORDER)].copy()
    isLine = rows["source"] == "dmLineResidualMetrics"
    rows["check"] = rows["metric"].where(
        ~isLine,
        rows["metric"] + " · " + rows["description"] + " · " + rows["reference"] + "/" + rows["statusType"],
    )
    absolute = rows["metric"].map(lambda name: specFor(name).absolute).astype(bool)
    sign = rows["metric"].map(lambda name: 1.0 if specFor(name).higherIsWorse else -1.0)
    judged = rows["value"].where(~absolute, rows["value"].abs())
    # Percent beyond the FAIL threshold, in the bad direction: 0 at FAIL, negative before it.
    scale = rows["fail"].abs().where(rows["fail"] != 0)
    # As float: on an empty selection (a camera the visit lacks) the arithmetic gives object columns.
    rows["pastFail"] = (100 * sign * (judged - rows["fail"]) / scale).astype(float)
    rows["warnAt"] = (100 * sign * (rows["warn"] - rows["fail"]) / scale).astype(float)
    rows["shown"] = [
        format(value, specFor(name).valueFormat) if np.isfinite(value) else ""
        for name, value in zip(rows["metric"], rows["value"], strict=True)
    ]
    return rows


def _problemOrder(checks: pd.DataFrame) -> list[str]:
    """Return the checks ordered worst first: by FAIL count, then WARN count."""
    counts = checks.pivot_table(index="check", columns="status", values="value", aggfunc="size", fill_value=0)
    counts = counts.reindex(columns=list(STATUS_ORDER), fill_value=0)
    return counts.sort_values(["FAIL", "WARN"], ascending=False).index.tolist()


def checkMatrix(checks: pd.DataFrame, cameras: list[str], showAll: bool = False) -> hv.Overlay:
    """Return a check × camera grid coloured by verdict, each cell labelled with its value.

    Only checks with a WARN or FAIL on some camera, unless ``showAll``.
    """
    problems = set(checks.loc[checks["status"].isin(["WARN", "FAIL"]), "check"])
    if not showAll:
        checks = checks[checks["check"].isin(problems)]
    if checks.empty:
        return hv.Text(0, 0, "Every judged check passes").opts(height=120, responsive=True)
    order = _problemOrder(checks)
    frame = _plain(
        checks.assign(rank=checks["status"].map(_STATUS_RANK).astype(float))[
            ["camera", "check", "rank", "status", "shown", "warn", "fail", "population", "reason"]
        ]
    )
    cmap = [STATUS_COLORS[status] for status in IMAGE_STATUS_ORDER]
    height = 70 + 26 * len(order)
    heatmap = hv.HeatMap(
        frame, ["camera", "check"], ["rank", "status", "shown", "warn", "fail", "population", "reason"]
    )
    labels = hv.Labels(frame, ["camera", "check"], "shown")
    return (
        (heatmap * labels)
        .opts(
            hv.opts.HeatMap(
                cmap=cmap,
                clim=(-0.5, len(cmap) - 0.5),
                colorbar=False,
                tools=["hover", "tap"],
                responsive=True,
                height=height,
                xlabel="",
                ylabel="",
                line_color="white",
                line_width=2,
                hover_tooltips=[
                    ("camera", "@camera"),
                    ("check", "@check"),
                    ("value", "@shown"),
                    ("status", "@status"),
                    ("warn / fail", "@warn / @fail"),
                    ("population", "@population"),
                ],
            ),
            hv.opts.Labels(text_color="white", text_font_size="8pt"),
        )
        .redim.values(camera=cameras, check=list(reversed(order)))
    )


def thresholdDistance(checks: pd.DataFrame) -> hv.Overlay | hv.Text:
    """Return, for each check with a WARN or FAIL, every camera's value against its own thresholds.

    The x axis is percent beyond the FAIL threshold in the bad direction, so checks in different
    units compare: 0 is FAIL, each row's dash is its WARN, and a value at +40 % is 40 % worse than
    FAIL. Checks whose rows carry no thresholds are left out (see `judgedChecks`).
    """
    problems = set(checks.loc[checks["status"].isin(["WARN", "FAIL"]), "check"])
    rows = checks[checks["check"].isin(problems) & np.isfinite(checks["pastFail"])]
    if rows.empty:
        return hv.Text(0, 0, "No WARN or FAIL with thresholds").opts(height=120, responsive=True)
    order = _problemOrder(rows)
    high = float(np.clip(rows["pastFail"].max() * 1.1 + 5, 20, 300))
    low = float(np.clip(min(rows["warnAt"].min(), rows["pastFail"].min()) * 1.1 - 5, -100, -10))
    frame = _plain(
        rows.assign(x=rows["pastFail"].clip(low, high))[
            ["x", "check", "camera", "status", "shown", "warn", "fail", "pastFail"]
        ]
    )
    warns = _plain(rows.drop_duplicates(["check", "warnAt"]).assign(x=lambda f: f["warnAt"].clip(low, high)))
    flagged = frame[frame["status"].isin(["WARN", "FAIL"])]
    layers = [
        hv.VSpan(0, high).opts(color=STATUS_COLORS["FAIL"], alpha=0.08),
        hv.VLine(0).opts(color=STATUS_COLORS["FAIL"], line_width=1.5),
        hv.Scatter(warns, ["x", "check"], ["warn"]).opts(
            marker="dash", angle=90, size=16, color=STATUS_COLORS["WARN"], line_width=3
        ),
        hv.Scatter(frame, ["x", "check"], ["camera", "status", "shown", "warn", "fail", "pastFail"]).opts(
            color="status",
            cmap=STATUS_COLORS,
            size=9,
            line_color="white",
            tools=["hover"],
            hover_tooltips=[
                ("check", "@check"),
                ("camera", "@camera"),
                ("value", "@shown"),
                ("warn / fail", "@warn / @fail"),
                ("past FAIL", "@pastFail{0.0}%"),
            ],
            show_legend=False,
        ),
        hv.Labels(flagged, ["x", "check"], "camera").opts(
            text_font_size="8pt", yoffset=0.3, text_color="#555"
        ),
    ]
    return (
        hv.Overlay(layers)
        .opts(
            responsive=True,
            height=80 + 40 * len(order),
            xlabel="% beyond the FAIL threshold, in the bad direction (red line = FAIL, amber dash = WARN)",
            ylabel="",
            xlim=(low, high),
        )
        .redim.values(check=list(reversed(order)))
    )


class Dashboard(param.Parameterized):
    """The app's state: the replay (or followed) position, the visit in the Visit tab, the data's version."""

    replayIndex = param.Integer(default=0, bounds=(0, None))
    selectedVisit = param.Integer(default=0)
    dataVersion = param.Integer(default=0)
    follow = param.Boolean(default=True, doc="Over qadb: keep the Live tab on the latest visit")

    def __init__(
        self,
        store: Tier1Store,
        detailDir: Path | None = None,
        startVisit: int | None = None,
        refreshSeconds: float = 30.0,
        **params,
    ):
        self.store = store
        self.detailDir = Path(detailDir) if detailDir else None
        self.refreshSeconds = refreshSeconds
        self._reload()
        super().__init__(**params)
        if store.live or not self.visitIds:
            self.replayIndex = max(0, len(self.visitIds) - 1)
        else:
            # The replay starts at startVisit (the first visit at or after it), else a little way in.
            start = int(np.searchsorted(self.visitIds, startVisit)) if startVisit is not None else 20
            self.replayIndex = min(start, len(self.visitIds) - 1)
        self.selectedVisit = self.visitIds[self.replayIndex] if self.visitIds else 0
        self.tabs = pn.Tabs(dynamic=True)
        self._timer = None

    def _reload(self) -> None:
        """Read the visits, images and notes from the store."""
        self._seenVersion = self.store.version
        self.visits = self.store.visits()
        self.images = self.store.images()
        self.notes = self.store.notes()
        self.timeline = visitTimeline(self.visits)
        self.visitIds = [int(v) for v in self.visits["visit"]]

    def refresh(self) -> None:
        """Take what the store has read since the last call; over qadb, move the Live tab to the latest visit."""
        self.store.refresh()
        if self.store.version == self._seenVersion:
            return
        first = not self.visitIds
        self._reload()
        if self.follow and self.visitIds:
            self.replayIndex = len(self.visitIds) - 1
            if first:
                self.selectedVisit = self.visitIds[-1]
        if first and self.visitIds:
            self._buildTabs()
        self.dataVersion += 1

    # ---- shared ----------------------------------------------------------------------------------

    def provenanceBar(self) -> pn.pane.HTML:
        prov = self.store.provenance
        thresholds = prov.get("thresholds", {})
        pipeline = thresholds.get("pipelineVersions") or {}
        parts = [
            f"<b>Collection</b> {prov.get('collection', '?')}",
            f"<b>Thresholds</b> {thresholds.get('file', '?')} v{thresholds.get('version', '?')} "
            f"(derived {thresholds.get('derivedOn', '?')} from runs "
            f"{', '.join(str(r) for r in thresholds.get('referenceRuns') or [])})",
            f"<b>Layers</b> {' + '.join(prov.get('thresholdLayers', []))}",
            "<b>Pipeline</b> " + ", ".join(f"{k} {v}" for k, v in pipeline.items()),
            f"<b>drp_qa</b> {prov.get('drpQaVersion', '?')}",
            f"<b>Built</b> {prov.get('builtAt', '?')}",
        ]
        warnings = []
        if prov.get("nVerdictDisagreements"):
            warnings.append(
                f"{prov['nVerdictDisagreements']} of {prov.get('nImages', '?')} stored image verdicts differ "
                "from these thresholds: the collection was gated with an earlier thresholds file, and does "
                "not record which. Stored verdicts are shown; per-metric verdicts are re-judged."
            )
        if self.store.live:
            warnings.append(
                "Live from qadb: times are qadb's pfs_visit.issued_at (HST), and sequences are inferred from "
                "runs of one seqName (qadb holds no IIC sequence)."
            )
        elif (self.visits["sequenceSource"] == "inferred").all():
            warnings.append(
                "No opdb visit listing: sequences are inferred from runs of one seqName, and there are no "
                "visit times (date picker disabled)."
            )
        html = '<div style="font-size:0.85em;line-height:1.5">' + " · ".join(parts) + "</div>"
        for warning in warnings:
            html += (
                '<div style="font-size:0.85em;margin-top:4px;padding:4px 8px;border-left:4px solid '
                f'{STATUS_COLORS["WARN"]};background:rgba(230,161,0,0.08)">⚠ {warning}</div>'
            )
        return pn.pane.HTML(html, sizing_mode="stretch_width")

    def _imagesOf(self, visit: int) -> pd.DataFrame:
        return self.images[self.images["visit"] == visit]

    def _visitStatus(self, visit: int) -> str:
        statuses = self._imagesOf(visit)["status"]
        return max(statuses, key=_STATUS_RANK.get) if len(statuses) else "UNKNOWN"

    def _when(self, visit: int, seconds: bool = False) -> str:
        """Return when a visit began, ``2026-03-09 19:42 HST``; empty without the opdb listing."""
        time = self.timeline.set_index("visit").loc[visit, "time"]
        return "" if pd.isna(time) else f"{pd.Timestamp(time):%Y-%m-%d %H:%M{':%S' if seconds else ''}} HST"

    def _span(self, visits) -> str:
        """Return when some visits ran, ``03-09 19:40–19:52 · ``; empty without times."""
        times = self.timeline.set_index("visit").loc[list(visits), "time"].dropna()
        if times.empty:
            return ""
        first, last = pd.Timestamp(times.min()), pd.Timestamp(times.max())
        end = f"{last:%H:%M}" if (last - first) < pd.Timedelta(hours=12) else f"{last:%m-%d %H:%M}"
        return f"{first:%m-%d %H:%M}–{end} · "

    def _visitSummary(self, visit: int) -> str:
        images = self._imagesOf(visit)
        counts = images["status"].value_counts()
        info = self.visits.set_index("visit").loc[visit]
        night = self.timeline.set_index("visit").loc[visit, "night"]
        when = (
            f"<div style='font-size:1.15em;margin:2px 0'>{self._when(visit, seconds=True)}"
            f" <span style='color:#777'>· night of {night}</span></div>"
            if night
            else ""
        )
        chips = " ".join(
            _badge(status, f"{counts[status]} {status}")
            for status in reversed(IMAGE_STATUS_ORDER)
            if status in counts
        )
        return (
            f"<div style='font-size:1.4em;font-weight:700'>Visit {visit} {_badge(self._visitStatus(visit))}</div>"
            f"{when}<div>{info['obsType']} · {info['seqName']} · sequence {info['sequenceKey']} "
            f"({info['sequenceSource']})</div><div style='margin-top:4px'>{chips}</div>"
        )

    def _notesHtml(self, visit: int) -> str:
        """Return a visit's opdb notes (its own, its sequence's, its cameras') as a box; empty if none."""
        notes = self.notes[self.notes["visit"] == visit]
        if notes.empty:
            return ""
        items = "".join(
            f"<li>{html.escape(row.note)} <span style='color:#888;font-size:0.85em'>"
            f"({row.source.removeprefix('opdb:').replace('_', ' ')}"
            + (f", {row.arm}{row.spectrograph}" if row.arm else "")
            + ")</span></li>"
            for row in notes.itertuples()
        )
        return (
            f"<div style='margin-top:6px;padding:4px 10px;border-left:4px solid {STATUS_COLORS['WARN']};"
            f"background:rgba(230,161,0,0.08)'><b>Notes</b><ul style='margin:2px 0'>{items}</ul></div>"
        )

    def _tapToVisit(self, heatmap: hv.HeatMap, timeline: pd.DataFrame) -> None:
        """Open the visit of the column tapped in a `statusHeatmap`."""
        visits = timeline["visit"].to_numpy()

        def onTap(event):
            if event.new is not None:
                position = round(float(event.new))
                if 0 <= position < len(visits):
                    self.openVisit(int(visits[position]))

        hv.streams.Tap(source=heatmap).param.watch(onTap, "x")

    def openVisit(self, visit: int) -> None:
        self.selectedVisit = int(visit)
        self.tabs.active = 2

    # ---- Live ------------------------------------------------------------------------------------

    def _step(self) -> None:
        if self.replayIndex < len(self.visitIds) - 1:
            self.replayIndex += 1
        elif self._timer is not None:
            self._timer.stop()

    def liveTab(self) -> pn.viewable.Viewable:
        nextButton = pn.widgets.Button(name="Next visit ▶", button_type="primary", width=130)
        backButton = pn.widgets.Button(name="◀ Back", width=90)
        play = pn.widgets.Toggle(name="Auto-play", button_type="success", width=110)
        speed = pn.widgets.IntSlider(name="Seconds per visit", start=1, end=10, value=3, width=180)
        nextButton.on_click(lambda _: self._step())
        backButton.on_click(lambda _: setattr(self, "replayIndex", max(0, self.replayIndex - 1)))
        self._timer = pn.state.add_periodic_callback(self._step, period=speed.value * 1000, start=False)

        def togglePlay(event):
            self._timer.start() if event.new else self._timer.stop()

        def setSpeed(event):
            self._timer.period = event.new * 1000

        play.param.watch(togglePlay, "value")
        speed.param.watch(setSpeed, "value")
        follow = pn.widgets.Toggle.from_param(self.param.follow, name="Follow the latest visit", width=190)

        def onFollow(event):
            if event.new and self.visitIds:
                self.replayIndex = len(self.visitIds) - 1

        self.param.watch(onFollow, "follow")

        @pn.depends(self.param.replayIndex, self.param.dataVersion)
        def position(*_):
            visit = self.visitIds[self.replayIndex]
            when = self._when(visit)
            if self.store.live:
                updated = getattr(self.store, "updatedAt", None)
                return pn.pane.HTML(
                    f"<div style='padding-top:8px'>Visit <b>{self.replayIndex + 1}</b> of "
                    f"{len(self.visitIds)}"
                    + (f" · <b>{when}</b>" if when else "")
                    + (f" · qadb read at {updated:%H:%M:%S} HST" if updated else "")
                    + f" (every {self.refreshSeconds:g} s)</div>"
                )
            return pn.pane.HTML(
                f"<div style='padding-top:8px'>Replaying <b>{self.replayIndex + 1}</b> of "
                f"{len(self.visitIds)} visits"
                + (f" · now <b>{when}</b>" if when else "")
                + " — standing in for qaActor's feed</div>"
            )

        @pn.depends(self.param.replayIndex, self.param.dataVersion)
        def latest(*_):
            visit = self.visitIds[self.replayIndex]
            images = self._imagesOf(visit)
            problems = images[images["status"].isin(["WARN", "FAIL"])].assign(camera=_camera)
            reasons = "".join(
                f"<li>{_badge(row.status, row.camera)} {html.escape(row.reasons)}</li>"
                for row in problems.itertuples()
            )
            openButton = pn.widgets.Button(name=f"Open visit {visit} ↗", width=160)
            openButton.on_click(lambda _: self.openVisit(visit))
            return pn.Card(
                pn.Row(
                    pn.pane.HTML(verdictGrid(images), width=260),
                    pn.Column(
                        pn.pane.HTML(self._visitSummary(visit) + self._notesHtml(visit)),
                        openButton,
                        pn.pane.HTML(
                            f"<ul style='font-size:0.85em;max-height:190px;overflow:auto'>{reasons}</ul>"
                            if reasons
                            else "<i>No WARN or FAIL.</i>"
                        ),
                    ),
                ),
                title="Latest visit",
                collapsible=False,
                sizing_mode="stretch_width",
            )

        @pn.depends(self.param.replayIndex, self.param.dataVersion)
        def recent(*_):
            visits = self.visitIds[max(0, self.replayIndex - 29) : self.replayIndex + 1]
            timeline = _renumber(self.timeline, visits)
            overlay, heatmap = statusHeatmap(self.images, timeline)
            self._tapToVisit(heatmap, timeline)
            nights = [n for n in dict.fromkeys(timeline["night"]) if n]
            span = f" · night{'s' if len(nights) > 1 else ''} of {', '.join(nights)}" if nights else ""
            return pn.Card(
                pn.pane.HoloViews(overlay, sizing_mode="stretch_width"),
                title=f"Last {len(visits)} visits{span} (tap a cell to open the visit)",
                sizing_mode="stretch_width",
            )

        @pn.depends(self.param.replayIndex, self.param.dataVersion)
        def sequences(*_):
            seen = self.visits.iloc[: self.replayIndex + 1]
            keys = list(dict.fromkeys(seen["sequenceKey"][::-1]))[:5]
            cards = []
            for i, key in enumerate(keys):
                members = seen[seen["sequenceKey"] == key]
                worst = max((self._visitStatus(v) for v in members["visit"]), key=_STATUS_RANK.get)
                chips = []
                for visit in members["visit"]:
                    images = self._imagesOf(visit)
                    counts = images["status"].value_counts()
                    label = f"{visit}"
                    if counts.get("FAIL") or counts.get("WARN"):
                        label += f" ({counts.get('FAIL', 0)}F {counts.get('WARN', 0)}W)"
                    chips.append(_badge(self._visitStatus(visit), label))
                first = members.iloc[0]
                cards.append(
                    pn.Card(
                        pn.pane.HTML(
                            "<div style='line-height:2'>"
                            + " ".join(chips)
                            + "</div>"
                            + "".join(dict.fromkeys(self._notesHtml(int(v)) for v in members["visit"]))
                            + (
                                verdictGrid(self._imagesOf(int(members["visit"].iloc[-1])), size=30)
                                if i == 0
                                else ""
                            )
                        ),
                        title=f"{self._span(members['visit'])}{first['seqName']} · {first['obsType']} · "
                        f"{len(members)} visits · {worst}"
                        + (" · has notes" if self.notes["visit"].isin(members["visit"]).any() else ""),
                        header_background=STATUS_COLORS[worst],
                        header_color="white",
                        collapsed=i > 0,
                        sizing_mode="stretch_width",
                    )
                )
            return pn.Column(*cards, sizing_mode="stretch_width")

        controls = (
            (follow, backButton, nextButton) if self.store.live else (backButton, nextButton, play, speed)
        )
        return pn.Column(
            pn.Row(*controls, position),
            pn.Row(
                pn.Column(latest, recent, sizing_mode="stretch_width"),
                pn.Column("### Sequence groups", sequences, width=430),
            ),
            sizing_mode="stretch_width",
        )

    # ---- Run -------------------------------------------------------------------------------------

    def runTab(self) -> pn.viewable.Viewable:
        visitRange = pn.widgets.IntRangeSlider(
            name="Visits",
            start=self.visitIds[0],
            end=_sliderEnd(self.visitIds),
            value=(self.visitIds[0], self.visitIds[-1]),
            step=1,
            sizing_mode="stretch_width",
        )
        hasTimes = self.visits["timeExpStart"].notna().any()
        dates = pn.widgets.DatetimeRangePicker(
            name="HST dates" + ("" if hasTimes else " (needs the opdb visit listing)"), disabled=not hasTimes
        )
        obsTypes = sorted(self.visits["obsType"].dropna().unique())
        obsType = pn.widgets.MultiChoice(name="Exposure type", options=obsTypes, value=obsTypes)
        arms = pn.widgets.MultiChoice(name="Arm", options=list(ARMS), value=list(ARMS))

        latestShown = [self.visitIds[-1]]

        def grow(_):
            # Keep the range's end on the latest visit if it was there.
            visitRange.end = _sliderEnd(self.visitIds)
            if visitRange.value[1] >= latestShown[0]:
                visitRange.value = (visitRange.value[0], self.visitIds[-1])
            latestShown[0] = self.visitIds[-1]
            # Exposure types first seen live are shown; ones the user left out stay out.
            types = sorted(self.visits["obsType"].dropna().unique())
            added = [t for t in types if t not in obsType.options]
            obsType.options = types
            obsType.value = [*obsType.value, *added]

        self.param.watch(grow, "dataVersion")

        def selectedVisits(rng, dateRange, types) -> list[int]:
            chosen = self.visits[self.visits["visit"].between(*rng) & self.visits["obsType"].isin(types)]
            if dateRange and hasTimes:
                chosen = chosen[chosen["timeExpStart"].between(*dateRange)]
            return [int(v) for v in chosen["visit"]]

        @pn.depends(visitRange, dates, obsType, arms)
        def summary(rng, dateRange, types, armList):
            visits = selectedVisits(rng, dateRange, types)
            images = self.images[self.images["visit"].isin(visits) & self.images["arm"].isin(armList)]
            counts = images["status"].value_counts()
            chips = " ".join(_badge(s, f"{counts.get(s, 0)} {s}") for s in reversed(IMAGE_STATUS_ORDER))
            timeline = _renumber(self.timeline, visits)
            overlay, heatmap = statusHeatmap(images, timeline, height=300)
            self._tapToVisit(heatmap, timeline)
            return pn.Column(
                pn.pane.HTML(f"<b>{len(visits)} visits, {len(images)} images</b> &nbsp; {chips}"),
                pn.pane.HoloViews(overlay, sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
            )

        xAxis = pn.widgets.RadioButtonGroup(
            name="Trend x axis",
            options=["Visits in order", "HST time"],
            value="Visits in order",
            button_type="light",
        )
        panels = []
        for title, source, metrics, reference, statusType in TRENDS:
            choice = pn.widgets.RadioButtonGroup(options=list(metrics), value=metrics[0], button_type="light")

            @pn.depends(choice, visitRange, dates, obsType, arms, xAxis)
            def plot(
                metric,
                rng,
                dateRange,
                types,
                armList,
                axis,
                source=source,
                reference=reference,
                statusType=statusType,
            ):
                visits = selectedVisits(rng, dateRange, types)
                frame = self.store.trend(source, metric, rng[0], rng[1], reference, statusType)
                frame = frame[frame["visit"].isin(visits) & frame["arm"].isin(armList)].reset_index(drop=True)
                overlay = trendPlot(frame, metric, self.timeline, byTime=axis == "HST time")
                if not frame.empty:
                    points = overlay.get(0)
                    selection = hv.streams.Selection1D(source=points)

                    def onSelect(event, points=points):
                        if event.new:
                            self.openVisit(int(points.data.iloc[event.new[0]]["visit"]))

                    selection.param.watch(onSelect, "index")
                note = f"{len(frame)} values" + (
                    f" · reference={reference}, statusType={statusType}" if reference else ""
                )
                return pn.Column(
                    pn.pane.HoloViews(overlay, sizing_mode="stretch_width"),
                    pn.pane.HTML(
                        f"<span style='font-size:0.8em;color:#777'>{note}; dashes are each value's "
                        "WARN (amber) and FAIL (red) thresholds</span>"
                    ),
                    sizing_mode="stretch_width",
                )

            panels.append(pn.Card(choice, plot, title=title, sizing_mode="stretch_width"))

        return pn.Column(
            pn.Row(visitRange, dates, obsType, arms, sizing_mode="stretch_width"),
            pn.Row(pn.pane.HTML("<div style='padding-top:6px'>Trend x axis:</div>"), xAxis),
            pn.Card(summary, title="Verdicts (tap a cell to open the visit)", sizing_mode="stretch_width"),
            pn.GridBox(*panels, ncols=2, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    # ---- Visit -----------------------------------------------------------------------------------

    def _detailFiles(self, visit: int) -> dict[str, list[Path]]:
        if self.detailDir is None or not (self.detailDir / str(visit)).is_dir():
            return {}
        directory = self.detailDir / str(visit)
        return {kind: sorted(directory.glob(f"{kind}*.pdf")) for kind in PDF_TYPES} | {
            "dmLineResidualData": sorted(directory.glob("dmLineResidualData*.parquet"))
        }

    def visitTab(self) -> pn.viewable.Viewable:
        def labels() -> dict[str, int]:
            return {
                (
                    f"{visit} · {self._when(visit).removesuffix(' HST')}" if self._when(visit) else str(visit)
                ): visit
                for visit in self.visitIds
            }

        picker = pn.widgets.Select(name="Visit (HST)", options=labels(), width=230)
        self.param.watch(lambda _: setattr(picker, "options", labels()), "dataVersion")
        cameraPicker = pn.widgets.Select(name="Camera", options=["all", *CAMERAS], value="all", width=120)
        picker.link(self, value="selectedVisit", bidirectional=True)
        picker.value = self.selectedVisit

        @pn.depends(self.param.selectedVisit, cameraPicker, self.param.dataVersion)
        def body(visit, camera, _version):
            images = self._imagesOf(visit).assign(camera=_camera)
            metrics = self.store.visitMetrics(visit).assign(camera=_camera)
            if camera != "all":
                metrics = metrics[metrics["camera"] == camera]
            files = self._detailFiles(visit)

            shown = metrics[~metrics["metric"].isin([VERDICT, NOTE])][
                [
                    "camera",
                    "source",
                    "metric",
                    "description",
                    "reference",
                    "statusType",
                    "value",
                    "status",
                    "warn",
                    "fail",
                    "population",
                ]
            ]
            metricTable = pn.widgets.Tabulator(
                shown,
                show_index=False,
                disabled=True,
                pagination="local",
                page_size=25,
                layout="fit_data_stretch",
                header_filters=True,
            )
            metricTable.style.map(
                lambda s: (
                    f"background-color:{STATUS_COLORS[s]};color:white" if s in STATUS_COLORS and s else ""
                ),
                subset=["status"],
            )

            checks = judgedChecks(metrics)
            cameras = [c for c in CAMERAS if c in set(images["camera"])]
            showAll = pn.widgets.Checkbox(name="Show every judged check, not only those with a WARN or FAIL")

            def matrix(everything):
                overlay = checkMatrix(checks, cameras, everything)
                if isinstance(overlay, hv.Overlay):
                    tap = hv.streams.Tap(source=overlay.get(0))
                    tap.param.watch(
                        lambda event: event.new and setattr(cameraPicker, "value", str(event.new)), "x"
                    )
                return pn.pane.HoloViews(overlay, sizing_mode="stretch_width")

            unthresholded = sorted(
                set(
                    checks.loc[
                        checks["status"].isin(["WARN", "FAIL"]) & ~np.isfinite(checks["pastFail"]), "check"
                    ]
                )
            )
            verdictPlots = pn.Column(
                "#### What failed, where (tap a cell to pick that camera)",
                showAll,
                pn.bind(matrix, showAll),
                "#### How far past its thresholds",
                pn.pane.HoloViews(thresholdDistance(checks), sizing_mode="stretch_width"),
                pn.pane.HTML(
                    "<span style='font-size:0.85em;color:#777'>Not placed: "
                    + ", ".join(unthresholded)
                    + " — their Tier 1 rows store the verdict but not the thresholds.</span>"
                    if unthresholded
                    else ""
                ),
                pn.Card(
                    metricTable, title="All metrics (table)", collapsed=True, sizing_mode="stretch_width"
                ),
                sizing_mode="stretch_width",
            )

            pdfs = [
                pn.Card(
                    pn.pane.PDF(path, height=700, sizing_mode="stretch_width"),
                    title=path.name,
                    collapsed=True,
                )
                for kind in PDF_TYPES
                for path in files.get(kind, [])
            ]
            pdfTab = pn.Column(
                *(
                    pdfs
                    or [
                        pn.pane.Markdown(
                            f"**No PDFs fetched for visit {visit}.** The pipeline writes "
                            + ", ".join(f"`{kind}`" for kind in PDF_TYPES)
                            + "; the dashboard would show them here, fetched from the Butler for this dataId "
                            "only when this tab opens and cached on the dataset UUID (plan 5.3).\n\n"
                            + DETAIL_SEARCH
                        )
                    ]
                ),
                sizing_mode="stretch_width",
            )

            lineFiles = files.get("dmLineResidualData", [])
            if lineFiles and camera != "all" and camera not in set(images["camera"]):
                tier2 = pn.pane.Markdown(f"**Visit {visit} has no {camera} image.**")
            elif lineFiles:
                lines = pd.concat([readLines(path) for path in lineFiles], ignore_index=True)
                # Detectors share pixel coordinates: one at a time, by default the worst.
                ranked = images.assign(rank=images["status"].map(_STATUS_RANK)).sort_values(
                    ["rank", "camera"], ascending=[False, True]
                )
                detector = camera if camera != "all" else ranked["camera"].iloc[0]
                lines = lines[lines["camera"] == detector]
                reference = pn.widgets.RadioButtonGroup(options=["Calib", "Visit"], value="Calib")
                component = pn.widgets.RadioButtonGroup(options=["y", "x"], value="y")
                which = pn.widgets.RadioButtonGroup(options=["Lines", "Trace centroids", "All"], value="All")

                def raster(r, c, w):
                    chosen = {
                        "Lines": lines[lines["description"] != "Trace"],
                        "Trace centroids": lines[lines["description"] == "Trace"],
                    }.get(w, lines)
                    return pn.Column(
                        pn.pane.HoloViews(residualRaster(chosen, r, c), sizing_mode="stretch_width"),
                        pn.pane.HTML(
                            f"{detector}: {len(chosen):,} of {len(lines):,} rows "
                            f"({(lines['description'] == 'Trace').mean():.0%} trace centroids), "
                            "rasterised on the server (Datashader); the browser receives an image."
                        ),
                    )

                tier2 = pn.Column(
                    pn.pane.HTML(
                        f"<b>{detector}</b> ({images.set_index('camera')['status'].get(detector, 'no verdict')})"
                        + (" — the worst detector; pick another with Camera above" if camera == "all" else "")
                    ),
                    pn.Row(reference, component, which),
                    pn.bind(raster, reference, component, which),
                )
            else:
                tier2 = pn.pane.Markdown(
                    f"**No Tier 2 fetched for visit {visit}.** Here: `dmLineResidualData` (one row per "
                    "line or trace centroid, ~5×10⁴–3×10⁵ per detector) drawn as detector position × "
                    "residual against `detectorMap_calib` or the visit's detectorMap, rasterised "
                    "server-side with Datashader (plan 5.2). Read from the Butler for this dataId only.\n\n"
                    + DETAIL_SEARCH
                )

            return pn.Column(
                pn.Row(
                    pn.pane.HTML(verdictGrid(images.drop(columns="camera")), width=260),
                    pn.pane.HTML(self._visitSummary(visit) + self._notesHtml(visit)),
                ),
                pn.Tabs(
                    ("Gate verdicts", verdictPlots),
                    ("PDFs", pdfTab),
                    ("Tier 2 (Datashader)", tier2),
                    dynamic=True,
                ),
                sizing_mode="stretch_width",
            )

        prevButton = pn.widgets.Button(name="◀", width=40)
        nextButton = pn.widgets.Button(name="▶", width=40)

        def move(step):
            i = self.visitIds.index(self.selectedVisit) + step
            if 0 <= i < len(self.visitIds):
                self.selectedVisit = self.visitIds[i]

        prevButton.on_click(lambda _: move(-1))
        nextButton.on_click(lambda _: move(1))
        return pn.Column(
            pn.Row(picker, prevButton, nextButton, cameraPicker), body, sizing_mode="stretch_width"
        )

    # ---- layout ----------------------------------------------------------------------------------

    def _buildTabs(self) -> None:
        self.tabs[:] = [("Live", self.liveTab()), ("Run", self.runTab()), ("Visit", self.visitTab())]

    def view(self) -> pn.template.FastListTemplate:
        if self.visitIds:
            self._buildTabs()
        else:
            self.tabs.append(
                ("Live", pn.pane.HTML("<i>No visit in qadb yet: waiting for qaActor to load the first.</i>"))
            )
        if self.store.live:
            pn.state.add_periodic_callback(self.refresh, period=int(self.refreshSeconds * 1000))
        return pn.template.FastListTemplate(
            title="PFS QA dashboard (PIPE2D-1924)",
            main=[self.provenanceBar(), self.tabs],
            accent="#3b6ea8",
            theme_toggle=True,
            main_max_width="1600px",
        )


def createApp(
    tier1: Path | str | Tier1Store,
    detailDir: Path | str | None = None,
    startVisit: int | None = None,
    refreshSeconds: float = 30.0,
) -> pn.template.FastListTemplate:
    """Return the dashboard for a Tier 1 file or store; ``detailDir/<visit>/`` holds fetched detail files.

    Over a file the Live replay starts at ``startVisit``, or the first visit after it. A store that
    is `~Tier1Store.live` (`live.QadbTier1Store`) is polled every ``refreshSeconds``, and one store
    may serve every session.
    """
    hv.extension("bokeh")
    pn.extension("tabulator", sizing_mode="stretch_width")
    store = tier1 if isinstance(tier1, Tier1Store) else Tier1Store(tier1)
    return Dashboard(store, detailDir, startVisit, refreshSeconds).view()
