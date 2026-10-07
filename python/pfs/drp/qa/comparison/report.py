"""The per-period report: one self-contained HTML page.

Built from DataFrames: the classified visits, the coverage, the stored metrics with the gate's
per-metric judgement, the findings, and the reference run's metrics. Figures come from
`pfs.drp.qa.plotting.comparison` and are inlined as SVG, so the page needs no other file.

The page shows science visits, so it stays local: science sequence names are never printed,
only visit IDs.
"""

import datetime
import html
import io
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd

from pfs.drp.qa.comparison.findings import lampsOf
from pfs.drp.qa.comparison.persistence import darkSequences, gapSummary
from pfs.drp.qa.metrics.gate import STATUS_ORDER
from pfs.drp.qa.plotting.comparison import (
    plotArmTimeline,
    plotMetricComparison,
    plotNightlySeries,
    plotVerdictGrid,
)

__all__ = [
    "COMPARED_METRICS",
    "ReportInputs",
    "armThresholds",
    "buildReport",
    "imageVerdicts",
    "populations",
    "recurringSequences",
    "verdictCounts",
]

#: The metrics compared with the reference, in order.
COMPARED_METRICS = ("medFwhm", "medDxCenter", "pctFlagged", "nLines")

#: Nights a calibration sequence must recur on to be shown as a series.
MIN_RECURRING_NIGHTS = 4


@dataclass
class ReportInputs:
    """What a report is built from.

    Parameters
    ----------
    period : `str`
        E.g. ``run30``.
    version : `str`
        The drp_qa version.
    collection : `str`
        The output collection.
    readUntil : `str`
        When the opdb listing ends.
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.classify.classifyVisits`, this period's.
    summary : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.plan.summarize`.
    metrics : `pandas.DataFrame`
        ``iqQaMetrics`` rows of the period.
    judged : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.findings.judgeImages`.
    findings : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.findings.findings`.
    reference : `pandas.DataFrame`, optional
        The reference run's ``iqQaMetrics`` rows, with the columns of
        `populations`.
    referenceName : `str`
        The reference run, e.g. ``run25``.
    lastLit : `pandas.DataFrame`, optional
        From `pfs.drp.qa.comparison.persistence.lastLitBefore`.
    timeline : `pandas.DataFrame`, optional
        From `pfs.drp.qa.comparison.persistence.armTimeline`.
    """

    period: str
    version: str
    collection: str
    readUntil: str
    visits: pd.DataFrame
    summary: pd.DataFrame
    metrics: pd.DataFrame
    judged: pd.DataFrame
    findings: pd.DataFrame
    reference: pd.DataFrame | None = None
    referenceName: str = "run25"
    lastLit: pd.DataFrame | None = None
    timeline: pd.DataFrame | None = None


def populations(metrics: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Add each image's visit information to its metrics.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        ``iqQaMetrics`` rows with ``visit``.
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.classify.classifyVisits`.

    Returns
    -------
    `pandas.DataFrame`
        ``metrics`` with ``night``, ``sequence_type``, ``cadence``, ``sequence_name``,
        ``group_name``, ``category``, ``validated``, ``focusSweep``, ``dithered`` and
        ``lamps`` (from the sequence's command, e.g. ``neon`` or ``krypton (IIS)``).
    """
    columns = [
        "night",
        "sequence_type",
        "cadence",
        "sequence_name",
        "group_name",
        "category",
        "validated",
        "focusSweep",
        "dithered",
    ]
    indexed = visits.set_index("pfs_visit_id")
    info = indexed[columns].assign(lamps=indexed["cmd_str"].map(lambda cmd: ", ".join(lampsOf(cmd))))
    columns = [*columns, "lamps"]
    return metrics.drop(columns=[c for c in columns if c in metrics], errors="ignore").join(info, on="visit")


def imageVerdicts(metrics: pd.DataFrame, judged: pd.DataFrame) -> pd.DataFrame:
    """Return each image's worst verdict.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        ``iqQaMetrics`` rows, with the columns of `populations`.
    judged : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.findings.judgeImages`, made from the same
        rows in the same order.

    Returns
    -------
    `pandas.DataFrame`
        ``metrics`` with ``status``: ``PASS``, ``WARN``, ``FAIL`` or
        ``UNKNOWN`` where no metric was judged.
    """
    rank = judged["status"].map({status: i for i, status in enumerate(STATUS_ORDER)})
    worst = rank.groupby(judged["row"]).max().reindex(range(len(metrics)))
    status = [STATUS_ORDER[int(r)] if not pd.isna(r) else "UNKNOWN" for r in worst]
    return metrics.reset_index(drop=True).assign(status=status)


def verdictCounts(verdicts: pd.DataFrame) -> pd.DataFrame:
    """Count verdicts by category, sequence type and arm.

    Parameters
    ----------
    verdicts : `pandas.DataFrame`
        From `imageVerdicts`.

    Returns
    -------
    `pandas.DataFrame`
        ``category``, ``sequence_type``, ``cadence``, ``arm``, one column per
        verdict and ``images``.
    """
    counts = verdicts.pivot_table(
        index=["category", "sequence_type", "cadence", "arm"],
        columns="status",
        values="visit",
        aggfunc="size",
        fill_value=0,
    )
    counts = counts.reindex(columns=[*STATUS_ORDER, "UNKNOWN"], fill_value=0)
    counts["images"] = counts.sum(axis=1)
    counts.columns.name = None
    return counts.reset_index()


def armThresholds(judged: pd.DataFrame, metric: str) -> dict[str, tuple[float, float]]:
    """Return the thresholds of a metric per arm, where one pair serves the whole arm.

    Parameters
    ----------
    judged : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.findings.judgeImages`.
    metric : `str`
        The metric.

    Returns
    -------
    `dict` [`str`, `tuple` [`float`, `float`]]
        ``(warn, fail)`` by arm; an arm whose populations have different
        thresholds (per lamp, say) is left out rather than drawn wrongly.
    """
    # Only entries that judged something: none for a metric skipped on a population.
    rows = judged[(judged["metric"] == metric) & (judged["layer"] >= 0) & (judged["status"] != "")]
    result = {}
    for arm, group in rows.groupby("arm"):
        pairs = group[["warn", "fail"]].drop_duplicates()
        if len(pairs) == 1:
            result[arm] = (float(pairs["warn"].iloc[0]), float(pairs["fail"].iloc[0]))
    return result


def recurringSequences(visits: pd.DataFrame, minNights: int = MIN_RECURRING_NIGHTS) -> pd.DataFrame:
    """Return the daily calibration sequences, by name.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.classify.classifyVisits`.
    minNights : `int`, optional
        Nights a sequence must recur on.

    Returns
    -------
    `pandas.DataFrame`
        ``sequence_type``, ``sequence_name``, ``group_name``, ``nights`` and
        ``visits``, most nights first, for those on at least ``minNights``.
    """
    calibrations = visits[visits["judged"] & (visits["cadence"] == "daily")]
    keys = ["sequence_type", "sequence_name", "group_name"]
    counts = (
        calibrations.fillna({"sequence_name": "", "group_name": ""})
        .groupby(keys)
        .agg(nights=("night", "nunique"), visits=("pfs_visit_id", "size"))
        .reset_index()
    )
    counts = counts[counts["nights"] >= minNights]
    return counts.sort_values(["nights", "visits"], ascending=False, ignore_index=True)


def buildReport(inputs: ReportInputs) -> str:
    """Return the report as one HTML page.

    Parameters
    ----------
    inputs : `ReportInputs`
        What to report.

    Returns
    -------
    `str`
        The page.
    """
    metrics = populations(inputs.metrics.reset_index(drop=True), inputs.visits)
    allVerdicts = imageVerdicts(metrics, inputs.judged)
    verdicts = allVerdicts[allVerdicts["validated"].astype(bool)]
    unvalidated = allVerdicts[~allVerdicts["validated"].astype(bool)]
    isGated = inputs.findings["validated"].astype(bool)
    gatedFindings, otherFindings = inputs.findings[isGated], inputs.findings[~isGated]
    reference = inputs.reference
    nights = inputs.visits["night"].dropna()
    parts = [
        f"<h1>{_e(inputs.period)}: image-quality gate against {_e(inputs.referenceName)}</h1>",
        "<p class='meta'>"
        f"drp_qa {_e(inputs.version)} · collection <code>{_e(inputs.collection)}</code> · "
        f"nights {_e(_date(nights.min()))} to {_e(_date(nights.max()))} · opdb read until {_e(inputs.readUntil)} · "
        f"built {datetime.datetime.now():%Y-%m-%d %H:%M}</p>",
        _headline(verdicts, gatedFindings, unvalidated),
        "<h2>Coverage</h2>",
        "<p>Every visit of the period, by what it is and what happened to it: <code>gated</code> visits are "
        "judged against validated thresholds, <code>unvalidated</code> ones are judged the same way but have none, "
        "the rest aren't measured. Detector images are counted for judged visits only.</p>",
        _table(inputs.summary),
        "<h2>Verdicts</h2>",
    ]
    for category in ("calibration", "science"):
        subset = verdicts[verdicts["category"] == category]
        if not subset.empty:
            parts.append(
                _figure(plotVerdictGrid(subset, title=f"{category.capitalize()}: worst verdict per night"))
            )
    if not verdicts.empty:
        parts.append(_table(verdictCounts(verdicts)))
    parts.append("<h2>Findings</h2>")
    if gatedFindings.empty:
        parts.append("<p>No gated image warns or fails.</p>")
    else:
        parts += [
            "<p>Each image that warns or fails: the metrics that crossed, how far the problem spreads within its "
            "visit (and whether the whole sequence shares it), the setup, and the opdb's notes. Telescope focus "
            "sweeps are not judged on flux-dependent metrics, nor daily arcs and traces on <code>nLines</code> and <code>pctFlagged</code> "
            "(PIPE2D-1935), so a verdict here can be milder than the stored one.</p>",
            _table(_ordered(gatedFindings).drop(columns="validated")),
        ]

    parts.append("<h2 id='unvalidated'>Unvalidated exposures</h2>")
    if unvalidated.empty:
        parts.append("<p>No unvalidated exposure was judged.</p>")
    else:
        parts += [
            "<p class='unvalidated'><strong>Unvalidated:</strong> engineering and test exposures (engineering arcs, "
            "flats, fiber profiles, focus sweeps, windowed readouts, new sequence types) judged with the same "
            "thresholds, which weren't derived or checked for them. Many are off-nominal on purpose, so a "
            "<code>FAIL</code> here may be what the test expected: read it as a measurement, not a verdict.</p>",
            _figure(plotVerdictGrid(unvalidated, title="Unvalidated: worst verdict per night")),
            _table(verdictCounts(unvalidated)),
        ]
        if not otherFindings.empty:
            parts += [
                "<h3>Unvalidated findings</h3>",
                _table(_ordered(otherFindings).drop(columns="validated")),
            ]

    parts.append(f"<h2>Against {_e(inputs.referenceName)}</h2>")
    if reference is None or reference.empty:
        parts.append(f"<p>No {_e(inputs.referenceName)} metrics were given: run its comparison first.</p>")
    parts.append(
        "<p>Daily arcs and traces are compared with the reference run's sets: the sets are what the thresholds "
        "were derived from. <code>nLines</code> and <code>pctFlagged</code> aren't judged on daily ones until they "
        "allow for the fibers a design lights (PIPE2D-1935).</p>"
    )
    for sequenceType, category, cadence in (
        ("scienceArc", "calibration", "set"),
        ("scienceTrace", "calibration", "set"),
        ("scienceArc", "calibration", "daily"),
        ("scienceTrace", "calibration", "daily"),
        ("scienceObject", "calibration", ""),
        ("scienceObject", "science", ""),
    ):
        selected = _select(metrics, sequenceType, category, cadence)
        # Arcs and traces are compared lamp by lamp: line counts and flag rates are the lamp's.
        byLamp = cadence != ""
        for lamps, current in selected.groupby("lamps", sort=True) if byLamp else [("", selected)]:
            previous = None
            if reference is not None:
                previous = _select(reference, sequenceType, category, cadence and "set")
                if byLamp:
                    previous = previous[previous["lamps"] == lamps]
            heading = ", ".join(part for part in (sequenceType, category, cadence, lamps) if part)
            parts.append(f"<h3>{_e(heading)}</h3>")
            if byLamp and (previous is None or previous.empty):
                parts.append(f"<p>No {_e(inputs.referenceName)} set with these lamps to compare with.</p>")
            for metric in COMPARED_METRICS:
                if metric not in current or current[metric].dropna().empty:
                    continue
                fig = plotMetricComparison(
                    current,
                    previous,
                    metric,
                    labels=(inputs.period, inputs.referenceName),
                    thresholds=armThresholds(inputs.judged[inputs.judged["row"].isin(current.index)], metric),
                    title=f"{metric}: {heading}",
                )
                parts.append(_figure(fig))

    recurring = recurringSequences(inputs.visits)
    parts.append("<h2>Daily calibrations</h2>")
    if recurring.empty:
        parts.append(f"<p>No daily calibration recurs on {MIN_RECURRING_NIGHTS} or more nights.</p>")
    else:
        parts.append(
            "<p>Single-exposure arcs and traces repeated night after night, night by night: the median of each "
            "detector, with the 5-95 % range of the reference run's sets of the same type and lamps shaded. "
            "<code>medDxCenter</code> is the offset from the calibration detectorMap: drift, shown, not gated.</p>"
        )
        parts.append(_table(recurring))
        for _, row in recurring.iterrows():
            subset = metrics[
                (metrics["sequence_type"] == row["sequence_type"])
                & (metrics["sequence_name"].fillna("") == row["sequence_name"])
                & (metrics["group_name"].fillna("") == row["group_name"])
            ]
            label = f"{row['sequence_type']} {row['sequence_name']!r}"
            previous = None
            if reference is not None:
                previous = _select(reference, row["sequence_type"], "calibration", "set")
                previous = previous[previous["lamps"].isin(subset["lamps"].unique())]
            for metric in ("medFwhm", "medDxCenter"):
                if metric in subset and not subset[metric].dropna().empty:
                    parts.append(
                        _figure(
                            plotNightlySeries(subset, metric, reference=previous, title=f"{metric}: {label}")
                        )
                    )
    if inputs.lastLit is not None:
        parts += _persistenceSection(inputs.lastLit, inputs.timeline)
    return _page(f"{inputs.period} QA comparison", "\n".join(parts))


def _persistenceSection(lastLit: pd.DataFrame, timeline: pd.DataFrame | None) -> list[str]:
    """Return the n-arm darks section: when darks were taken relative to lit exposures."""
    parts = [
        "<h2>n-arm darks and the exposures before them</h2>",
        "<p>The n-arm detectors keep an image of a bright exposure for a while, so a dark taken soon after an "
        "arc or a quartz can still show its traces. Darks aren't measured yet (PIPE2D-1925), so this shows only "
        "when they were taken: each night's n-arm exposures, colored by what lit them with the darks in gray, and "
        "beside it, for each dark sequence, the minutes from the last lit exposure to its first dark (shaded under "
        "5 minutes). Whether those darks are dark needs their signal on the trace positions against this gap.</p>",
    ]
    if lastLit.empty:
        return [*parts, "<p>No n-arm dark in the period.</p>"]
    if timeline is not None and not timeline.empty:
        parts.append(_figure(plotArmTimeline(timeline, darkSequences(lastLit))))
    parts.append(_table(gapSummary(lastLit)))
    return parts


def _select(metrics: pd.DataFrame, sequenceType: str, category: str, cadence: str = "") -> pd.DataFrame:
    """Return the rows of one sequence type, category and cadence, less focus sweeps for sky."""
    subset = metrics[(metrics["sequence_type"] == sequenceType) & (metrics["category"] == category)]
    if cadence:
        subset = subset[subset["cadence"] == cadence]
    if "focusSweep" in subset:
        subset = subset[~subset["focusSweep"].fillna(False).astype(bool)]
    return subset


def _headline(verdicts: pd.DataFrame, findings: pd.DataFrame, unvalidated: pd.DataFrame) -> str:
    """Return the counts at the top of the page: the gate's, then the unvalidated ones apart."""
    counts = verdicts["status"].value_counts()
    items = [
        f"<span class='pill {s.lower()}'>{s[0]}</span> {counts.get(s, 0)} {s}"
        for s in (*STATUS_ORDER, "UNKNOWN")
    ]
    extents = (
        findings["extent"].str.replace(", whole sequence", "", regex=False) if not findings.empty else []
    )
    wide = int(sum(1 for extent in extents if extent and extent != "detector"))
    other = unvalidated["status"].value_counts()
    otherItems = ", ".join(f"{other.get(s, 0)} {s}" for s in (*STATUS_ORDER, "UNKNOWN") if other.get(s, 0))
    return (
        f"<p class='headline'>{len(verdicts)} gated images: {' · '.join(items)}.<br>"
        f"{len(findings)} findings, {wide} spreading beyond one detector.</p>"
        f"<p class='unvalidated'>Also {len(unvalidated)} unvalidated images"
        + (f" ({otherItems})" if otherItems else "")
        + ": judged without validated thresholds; see <a href='#unvalidated'>Unvalidated exposures</a>.</p>"
    )


def _ordered(findings: pd.DataFrame) -> pd.DataFrame:
    """Return the findings worst first, then by visit."""
    order = {"FAIL": 0, "WARN": 1}
    return (
        findings.assign(_order=findings["status"].map(order))
        .sort_values(["_order", "visit", "arm", "spectrograph"])
        .drop(columns="_order")
    )


def _date(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _e(text) -> str:
    return html.escape(str(text))


def _table(frame: pd.DataFrame) -> str:
    """Return a DataFrame as an HTML table."""
    return frame.to_html(
        index=False, border=0, classes="data", na_rep="", escape=True, float_format=lambda v: f"{v:.3g}"
    )


def _figure(fig) -> str:
    """Return a figure as inline SVG, and close it."""
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buffer.getvalue()
    return f"<figure>{svg[svg.index('<svg') :]}</figure>"


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>
:root {{ --ink:#222; --muted:#666; --surface:#fcfcfb; --rule:#e6e6e3; --card:#ffffff; }}
@media (prefers-color-scheme: dark) {{ :root {{ --ink:#e8e8e6; --muted:#a0a09c; --surface:#1b1b1a; --rule:#3a3a38; --card:#ffffff; }} }}
body {{ margin:0 auto; max-width:1200px; padding:16px; background:var(--surface); color:var(--ink);
  font:14px/1.45 system-ui, -apple-system, sans-serif; }}
h1 {{ font-size:22px; }} h2 {{ font-size:18px; margin-top:32px; border-bottom:1px solid var(--rule); }} h3 {{ font-size:15px; }}
.meta {{ color:var(--muted); }} .headline {{ font-size:15px; }}
.pill {{ display:inline-block; width:1.4em; text-align:center; border-radius:4px; color:#222; font-weight:600; }}
.pass {{ background:#0ca30c; }} .warn {{ background:#fab219; }} .fail {{ background:#d03b3b; }} .unknown {{ background:#d9d9d6; }}
table.data {{ border-collapse:collapse; font-size:12px; margin:8px 0; display:block; overflow-x:auto; }}
table.data th, table.data td {{ padding:3px 8px; border-bottom:1px solid var(--rule); text-align:left; vertical-align:top; }}
figure {{ margin:12px 0; background:var(--card); border-radius:6px; padding:6px; overflow-x:auto; }}
figure svg {{ max-width:100%; height:auto; }}
code {{ font-size:12px; }}
.unvalidated {{ border-left:4px solid #8c8c8c; padding-left:8px; color:var(--muted); }}
</style></head><body>
{body}
</body></html>
"""
