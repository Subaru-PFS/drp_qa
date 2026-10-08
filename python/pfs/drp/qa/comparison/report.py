"""The per-period report: one self-contained HTML page.

Built from DataFrames: the classified visits, the coverage, the stored metrics with the gate's
per-metric judgement, the findings, and the reference run's metrics. Figures come from
`pfs.drp.qa.plotting.comparison` and are inlined as SVG, so the page needs no other file.

The page shows science visits, so it stays local: science sequence names are never printed,
only visit IDs.
"""

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
    plotCoverage,
    plotMetricComparison,
    plotNightlySeries,
    plotVerdictGrid,
)

__all__ = [
    "COMPARED_METRICS",
    "ReportInputs",
    "armThresholds",
    "buildReport",
    "coverageRows",
    "failedSummary",
    "imageVerdicts",
    "populations",
    "problems",
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
    failed : `pandas.DataFrame`, optional
        From `failedSummary`.
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
    failed: pd.DataFrame | None = None


def failedSummary(detectors: pd.DataFrame, failed: pd.DataFrame) -> pd.DataFrame:
    """Return the images that failed, one row per visit, task and error.

    Parameters
    ----------
    detectors : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.plan.coverage`.
    failed : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.plan.failedQuanta`.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``sequence_type``, ``cadence``, ``cameras`` (``n1,n2``),
        ``task`` and ``error``, for the images still ``failed``; sorted by
        visit.
    """
    columns = ["visit", "sequence_type", "cadence", "cameras", "task", "error"]
    stillFailed = detectors[detectors["status"] == "failed"]
    if stillFailed.empty or failed.empty:
        return pd.DataFrame(columns=columns)
    merged = stillFailed.merge(failed, on=["visit", "arm", "spectrograph"], how="left")
    merged["camera"] = merged["arm"] + merged["spectrograph"].astype(str)
    merged[["task", "error"]] = merged[["task", "error"]].fillna("")
    keys = ["visit", "sequence_type", "cadence", "task", "error"]
    grouped = merged.groupby(keys, sort=True, dropna=False)["camera"].agg(lambda c: ",".join(sorted(c)))
    return grouped.rename("cameras").reset_index()[columns]


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


def problems(findings: pd.DataFrame) -> pd.DataFrame:
    """Group findings into problems: the same metrics crossed, on the same arm, in the same setup.

    Parameters
    ----------
    findings : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.findings.findings`.

    Returns
    -------
    `pandas.DataFrame`
        One row per problem: ``status`` (the worst), ``metrics`` (the names
        crossed), ``arm``, ``setup`` (without the exposure time), ``extent``
        (the commonest, without a visit's counts), ``expected``, ``images``, ``visits``,
        ``nights`` (first to last) and ``examples`` (up to three visits);
        failures first, then by images.
    """
    columns = [
        "status",
        "metrics",
        "arm",
        "setup",
        "extent",
        "expected",
        "images",
        "visits",
        "nights",
        "examples",
    ]
    if findings.empty:
        return pd.DataFrame(columns=columns)
    frame = findings.assign(
        metricNames=findings["metrics"]
        .str.findall(r"(\w+)=")
        .map(lambda names: ", ".join(dict.fromkeys(names))),
        setupKind=findings["setup"].str.split(";").str[0],
        extentKind=findings["extent"].str.replace(r"^visit \(.*?\)", "most of the visit", regex=True),
        expected=findings["expected"].fillna("") if "expected" in findings else "",
        rank=findings["status"].map({"FAIL": 0, "WARN": 1}),
    )
    grouped = frame.groupby(["metricNames", "arm", "setupKind", "expected"], dropna=False)
    result = grouped.agg(
        rank=("rank", "min"),
        extentKind=("extentKind", lambda e: e.value_counts().index[0]),
        images=("visit", "size"),
        visits=("visit", "nunique"),
        first=("night", "min"),
        last=("night", "max"),
        examples=("visit", lambda v: ", ".join(str(x) for x in sorted(v.unique())[:3])),
    ).reset_index()
    result["status"] = result["rank"].map({0: "FAIL", 1: "WARN"})
    result["nights"] = [
        str(first) if first == last else f"{first} to {last}"
        for first, last in zip(result["first"], result["last"], strict=True)
    ]
    result = result.rename(columns={"metricNames": "metrics", "setupKind": "setup", "extentKind": "extent"})
    return result.sort_values(["rank", "images"], ascending=[True, False])[columns].reset_index(drop=True)


def coverageRows(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the judged rows of a coverage summary as bars for `plotCoverage`.

    Parameters
    ----------
    summary : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.plan.summarize`.

    Returns
    -------
    `pandas.DataFrame`
        ``label`` and image counts ``judged``, ``to judge``, ``to reduce``,
        ``failed`` and ``blocked``, one row per judged sequence type, cadence
        and category.
    """
    judged = summary[summary["reason"].isin(["gated", "unvalidated"])].copy()
    blocked = [column for column in ("no pfsConfig", "no raw", "raw not in opdb") if column in judged]
    judged["blocked"] = judged[blocked].sum(axis=1) if blocked else 0
    judged["label"] = [
        " ".join(part for part in (row.sequence_type, row.cadence, row.category) if part)
        + (" (unvalidated)" if row.reason == "unvalidated" else "")
        for row in judged.itertuples()
    ]
    if "failed" not in judged:
        judged["failed"] = 0
    return judged[["label", "judged", "to judge", "to reduce", "failed", "blocked"]].reset_index(drop=True)


def _incomplete(summary: pd.DataFrame) -> str:
    """Return a banner when some judged types' images aren't judged yet, else nothing."""
    rows = coverageRows(summary)
    pending = rows[(rows["to judge"] + rows["to reduce"]) > 0]
    if pending.empty:
        return ""
    total = int((pending["to judge"] + pending["to reduce"]).sum())
    listed = ", ".join(
        f"{label} ({int(row['to judge'] + row['to reduce'])})"
        for label, row in pending.set_index("label").iterrows()
    )
    return (
        f"<p class='alert'><b>Incomplete:</b> {total} detector images are not judged yet: {_e(listed)}. "
        "The verdicts below leave them out.</p>"
    )


def buildReport(inputs: ReportInputs) -> str:
    """Return the report as one HTML page: pictures and counts first, detail folded away.

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
    findings = inputs.findings
    if "expected" not in findings:
        findings = findings.assign(expected="")
    isGated = findings["validated"].astype(bool)
    gatedFindings, otherFindings = findings[isGated], findings[~isGated]
    reference = inputs.reference
    nights = inputs.visits["night"].dropna()

    parts = [
        f"<h1>{_e(inputs.period)} against {_e(inputs.referenceName)}</h1>",
        f"<p class='meta'>nights {_e(_date(nights.min()))} to {_e(_date(nights.max()))} · drp_qa {_e(inputs.version)} · "
        f"<code>{_e(inputs.collection)}</code> · opdb to {_e(inputs.readUntil)}</p>",
        _incomplete(inputs.summary),
        _tiles(verdicts, gatedFindings, unvalidated),
        "<h2>Coverage</h2>",
    ]
    rows = coverageRows(inputs.summary)
    if not rows.empty:
        parts.append(_figure(plotCoverage(rows)))
    notMeasured = inputs.summary[~inputs.summary["reason"].isin(["gated", "unvalidated"])]
    if not notMeasured.empty:
        byReason = []
        for reason, group in notMeasured.groupby("reason", sort=False):
            counts = group.groupby("sequence_type")["visits"].sum().sort_values(ascending=False)
            label = "no method" if reason.startswith("no method") else reason
            byReason.append((label, counts))
        merged = {}
        for label, counts in byReason:
            merged.setdefault(label, []).extend(f"{n} {kind}" for kind, n in counts.items())
        listed = "; ".join(f"{label}: {', '.join(items)}" for label, items in merged.items())
        parts.append(f"<p class='caption'>Not measured (visits): {_e(listed)}.</p>")
    failed = inputs.failed
    if failed is not None and not failed.empty:
        images = int(failed["cameras"].str.count(",").sum() + len(failed))
        parts.append(
            f"<p class='caption'>Failed (not retried): {images} detector images of {failed['visit'].nunique()}"
            " visits, whose reduction raised; they fail the same way each time.</p>"
        )
        parts.append(_details("Failed images", _table(failed)))
    parts.append(_details("Coverage table", _table(inputs.summary)))

    parts.append("<h2>Verdicts</h2>")
    for category in ("calibration", "science"):
        subset = verdicts[verdicts["category"] == category]
        if not subset.empty:
            parts.append(
                _figure(plotVerdictGrid(subset, title=f"{category.capitalize()}: worst verdict per night"))
            )
    if not verdicts.empty:
        parts.append(_details("Counts by type and arm", _table(verdictCounts(verdicts))))

    parts.append("<h2>Problems</h2>")
    new = gatedFindings[gatedFindings["expected"] == ""]
    known = gatedFindings[gatedFindings["expected"] != ""]
    if gatedFindings.empty:
        parts.append("<p>No gated image warns or fails.</p>")
    else:
        parts.append(_table(problems(new).drop(columns="expected")) if not new.empty else "<p>None new.</p>")
        if not known.empty:
            parts.append(
                _details(
                    f"Expected by the validation set: {len(known)} images",
                    _table(problems(known)),
                )
            )
        parts.append(
            _details(
                f"Every finding: {len(gatedFindings)} images",
                _table(_ordered(gatedFindings).drop(columns="validated")),
            )
        )

    if not unvalidated.empty:
        parts += [
            "<h2 id='unvalidated'>Unvalidated</h2>",
            "<p class='caption unvalidated'>Judged without validated thresholds: a FAIL may be what the test expected.</p>",
            _figure(plotVerdictGrid(unvalidated, title="Unvalidated: worst verdict per night")),
        ]
        if not otherFindings.empty:
            parts.append(
                _details(
                    f"Unvalidated problems: {len(otherFindings)} images", _table(problems(otherFindings))
                )
            )

    parts.append(f"<h2>Against {_e(inputs.referenceName)}</h2>")
    if reference is None or reference.empty:
        parts.append(f"<p class='caption'>No {_e(inputs.referenceName)} metrics: report it first.</p>")
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
            if current.empty:
                continue
            previous = None
            if reference is not None:
                previous = _select(reference, sequenceType, category, cadence and "set")
                if byLamp:
                    previous = previous[previous["lamps"] == lamps]
            heading = ", ".join(part for part in (sequenceType, category, cadence, lamps) if part)
            figures = []
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
                figures.append(_figure(fig))
            noReference = byLamp and reference is not None and (previous is None or previous.empty)
            summary = f"{heading}: {len(current)} images" + (
                f", no {inputs.referenceName} set" if noReference else ""
            )
            parts.append(_details(summary, "".join(figures)))

    recurring = recurringSequences(inputs.visits)
    if not recurring.empty:
        parts += [
            "<h2>Daily calibrations</h2>",
            "<p class='caption'>Median per detector and night; shaded: the reference sets' 5-95 %.</p>",
        ]
        for index, row in recurring.iterrows():
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
            figures = [
                _figure(plotNightlySeries(subset, metric, reference=previous, title=f"{metric}: {label}"))
                for metric in ("medFwhm", "medDxCenter")
                if metric in subset and not subset[metric].dropna().empty
            ]
            if figures:
                parts.append(
                    _details(f"{label}: {row['nights']} nights", "".join(figures), opened=index == 0)
                )
    if inputs.lastLit is not None:
        parts += _persistenceSection(inputs.lastLit, inputs.timeline)
    parts.append(_details("How to read this", _HOW_TO_READ))
    return _page(f"{inputs.period} QA comparison", "\n".join(parts))


_HOW_TO_READ = """<ul>
<li><b>Gated</b>: <code>scienceArc</code>, <code>scienceTrace</code>, <code>scienceObject</code>, judged against
thresholds derived from the reference run. <b>Unvalidated</b>: other measurable types, judged the same way without
validated thresholds. Biases, darks and test exposures aren't measured.</li>
<li><b>Problems</b> group the images that warn or fail by metric, arm, setup and extent: <i>detector</i>, an arm on
every spectrograph, a whole spectrograph (<i>SMn</i>), <i>most of the visit</i>, and <i>whole sequence</i> when every
visit of the sequence shares it. Problems the validation set records as known-bad are folded away as expected.</li>
<li>Not judged: flux-dependent metrics on telescope focus sweeps, and <code>nLines</code> and <code>pctFlagged</code>
on daily arcs and traces, which are often taken with one fiber group lit (PIPE2D-1935). A verdict here can therefore
be milder than the stored one.</li>
<li><b>Against the reference</b>: arcs and traces lamp by lamp, daily ones against the reference's sets.</li>
<li><code>medDxCenter</code> is the offset from the calibration detectorMap: drift, shown, not gated.</li>
<li><b>n-arm darks</b>: the n detectors keep an image of a bright exposure for a while. Darks aren't measured yet
(PIPE2D-1925), so the timeline shows only when they were taken relative to lit exposures.</li>
</ul>"""


def _persistenceSection(lastLit: pd.DataFrame, timeline: pd.DataFrame | None) -> list[str]:
    """Return the n-arm darks section: when darks were taken relative to lit exposures."""
    parts = ["<h2>n-arm darks</h2>"]
    if lastLit.empty:
        return [*parts, "<p class='caption'>No n-arm dark in the period.</p>"]
    if timeline is not None and not timeline.empty:
        parts.append(_figure(plotArmTimeline(timeline, darkSequences(lastLit))))
    parts.append(_details("Darks by gap after the last lit exposure", _table(gapSummary(lastLit))))
    return parts


def _select(metrics: pd.DataFrame, sequenceType: str, category: str, cadence: str = "") -> pd.DataFrame:
    """Return the rows of one sequence type, category and cadence, less focus sweeps for sky."""
    subset = metrics[(metrics["sequence_type"] == sequenceType) & (metrics["category"] == category)]
    if cadence:
        subset = subset[subset["cadence"] == cadence]
    if "focusSweep" in subset:
        subset = subset[~subset["focusSweep"].fillna(False).astype(bool)]
    return subset


def _tiles(verdicts: pd.DataFrame, findings: pd.DataFrame, unvalidated: pd.DataFrame) -> str:
    """Return the verdict tiles at the top of the page, and one line under them."""
    counts = verdicts["status"].value_counts()
    tiles = "".join(
        f"<div class='tile {status.lower()}'><span>{counts.get(status, 0)}</span>{status}</div>"
        for status in (*STATUS_ORDER, "UNKNOWN")
    )
    expected = int((findings["expected"] != "").sum()) if not findings.empty else 0
    line = f"{len(verdicts)} gated images · {len(findings) - expected} problem images"
    if expected:
        line += f" (+{expected} expected)"
    if len(unvalidated):
        line += f" · {len(unvalidated)} <a href='#unvalidated'>unvalidated</a>"
    return f"<div class='tiles'>{tiles}</div><p class='meta'>{line}</p>"


def _ordered(findings: pd.DataFrame) -> pd.DataFrame:
    """Return the findings worst first, then by visit."""
    order = {"FAIL": 0, "WARN": 1}
    return (
        findings.assign(_order=findings["status"].map(order))
        .sort_values(["_order", "visit", "arm", "spectrograph"])
        .drop(columns="_order")
    )


def _details(summary: str, body: str, opened: bool = False) -> str:
    """Return a collapsible block."""
    return f"<details{' open' if opened else ''}><summary>{_e(summary)}</summary>{body}</details>"


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
.caption {{ color:var(--muted); font-size:13px; margin:4px 0 8px; }}
.alert {{ border-left:4px solid #d03b3b; padding:6px 10px; background:rgba(208,59,59,0.08); }}
.tiles {{ display:flex; gap:12px; flex-wrap:wrap; margin:12px 0 4px; }}
.tile {{ flex:1 1 120px; border-radius:8px; padding:10px 14px; color:#222; font-weight:600; font-size:13px; }}
.tile span {{ display:block; font-size:30px; line-height:1.1; }}
details {{ margin:6px 0; }} summary {{ cursor:pointer; color:var(--ink); font-size:14px; padding:4px 0; }}
</style></head><body>
{body}
</body></html>
"""
