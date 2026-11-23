"""Compare candidate calibrations on the same visits: the verdict after a build.

Each candidate is a collection holding ``detectorMap_calib`` (a certified calibration, or a
rebuild from the go/no-go pick). The same visits are reduced against each (``drpQa.py calib validate
run``), and `dmLineResiduals` measures, per detector image and species, the offsets and scatter
against that calibration and the reserved-line scatter against the visit's own adjusted map. This
module lays the results side by side: a tidy table, a summary per detector and calibration, and a
Markdown report. Stack-free: DataFrames in, text out.
"""

from collections.abc import Mapping

import pandas as pd

__all__ = [
    "buildSlitOffsetSection",
    "buildValidationReport",
    "compareCalibrations",
    "slitOffsetHealth",
    "validationSummary",
]

#: Line sets that stand in for one another: against the calibration, a visit whose fit marked
#: no usable lines stores every good line as ``UNFITTED`` instead of ``BOTH``.
_STATUS_TYPES = {"BOTH": ("BOTH", "UNFITTED")}

#: The metrics compared, by reference and line set, and their short names.
METRICS = {
    ("calib", "BOTH", "medianDeltaX"): "offset x vs calib",
    ("calib", "BOTH", "medianDeltaY"): "offset y vs calib",
    ("calib", "BOTH", "robustRmsDeltaX"): "RMS x vs calib",
    ("calib", "BOTH", "robustRmsDeltaY"): "RMS y vs calib",
    ("visit", "RESERVED", "robustRmsDeltaX"): "reserved RMS x",
    ("visit", "RESERVED", "robustRmsDeltaY"): "reserved RMS y",
}


def compareCalibrations(metrics: Mapping[str, pd.DataFrame], setVisits=None) -> pd.DataFrame:
    """Lay the candidates' ``dmLineResidualMetrics`` side by side.

    Parameters
    ----------
    metrics : mapping of `str` to `pandas.DataFrame`
        ``dmLineResidualMetrics`` rows per candidate, by label (e.g. ``asis``, ``pick``).
    setVisits : iterable of `int`, optional
        The calibration set's visits: rows get ``inSet``.

    Returns
    -------
    `pandas.DataFrame`
        One row per candidate, detector image, species and compared metric: ``calibration``,
        ``visit``, ``arm``, ``spectrograph``, ``detector``, ``species``, ``metric`` (the short
        name), ``value`` (px), ``nLines``, ``inSet``.
    """
    frames = []
    inSet = {int(v) for v in setVisits} if setVisits is not None else set()
    for label, frame in metrics.items():
        counts = frame[frame["metric"] == "nLines"]
        counts = counts.set_index(["visit", "arm", "spectrograph", "description", "reference", "statusType"])[
            "value"
        ]
        for (reference, statusType, metric), name in METRICS.items():
            rows = frame[
                (frame["reference"] == reference)
                & (frame["statusType"].isin(_STATUS_TYPES.get(statusType, (statusType,))))
                & (frame["metric"] == metric)
            ]
            if rows.empty:
                continue
            key = list(
                zip(
                    rows["visit"],
                    rows["arm"],
                    rows["spectrograph"],
                    rows["description"],
                    rows["statusType"],
                    strict=True,
                )
            )
            n = [counts.get((*k[:4], reference, k[4]), float("nan")) for k in key]
            frames.append(
                pd.DataFrame(
                    {
                        "calibration": label,
                        "visit": rows["visit"].astype(int).to_numpy(),
                        "arm": rows["arm"].to_numpy(),
                        "spectrograph": rows["spectrograph"].astype(int).to_numpy(),
                        "species": rows["description"].to_numpy(),
                        "metric": name,
                        "value": rows["value"].astype(float).to_numpy(),
                        "nLines": n,
                    }
                )
            )
    columns = [
        "calibration",
        "visit",
        "arm",
        "spectrograph",
        "detector",
        "species",
        "metric",
        "value",
        "nLines",
        "inSet",
    ]
    if not frames:
        return pd.DataFrame(columns=columns)
    table = pd.concat(frames, ignore_index=True)
    table["detector"] = table["arm"].astype(str) + table["spectrograph"].astype(str)
    table["inSet"] = table["visit"].isin(inSet)
    return (
        table[columns]
        .sort_values(["detector", "visit", "species", "metric", "calibration"])
        .reset_index(drop=True)
    )


def validationSummary(table: pd.DataFrame) -> pd.DataFrame:
    """Summarise `compareCalibrations` per detector, calibration and metric: medians over images and species.

    Returns
    -------
    `pandas.DataFrame`
        ``detector``, ``metric``, ``calibration``, ``set`` (median over the set's images, absolute
        for offsets), ``later`` (likewise over the images after the set), ``nSet``, ``nLater``.
    """
    if table.empty:
        return pd.DataFrame(columns=["detector", "metric", "calibration", "set", "later", "nSet", "nLater"])
    t = table.copy()
    t["abs"] = t["value"].abs().where(t["metric"].str.startswith("offset"), t["value"])
    rows = []
    for (det, metric, calib), g in t.groupby(["detector", "metric", "calibration"], sort=False):
        inSet, later = g[g["inSet"]], g[~g["inSet"]]
        rows.append(
            {
                "detector": det,
                "metric": metric,
                "calibration": calib,
                "set": inSet["abs"].median() if len(inSet) else float("nan"),
                "later": later["abs"].median() if len(later) else float("nan"),
                "nSet": int(inSet["visit"].nunique()),
                "nLater": int(later["visit"].nunique()),
            }
        )
    order = list(METRICS.values())
    out = pd.DataFrame(rows)
    out["metric"] = pd.Categorical(out["metric"], categories=order, ordered=True)
    return out.sort_values(["detector", "metric", "calibration"]).reset_index(drop=True)


def buildValidationReport(
    table: pd.DataFrame,
    title: str,
    candidates: Mapping[str, str],
    figures: Mapping[str, str] | None = None,
    visitsNote: str = "",
) -> str:
    """Return the Markdown report: the candidates, a summary table per detector, and a figure per detector.

    Parameters
    ----------
    table : `pandas.DataFrame`
        From `compareCalibrations`.
    title : `str`
        E.g. ``Run31 b1 and r2: pick against the certified build``.
    candidates : mapping of `str` to `str`
        Label to collection; a value already carrying backticks is written as given.
        The candidates' collections by label, for the header.
    figures : mapping of `str` to `str`, optional
        A figure per detector (`pfs.drp.qa.plotting.calibInputs.plotCalibrationComparison`), by
        detector, as paths relative to the report.
    visitsNote : `str`, optional
        One line on which visits were reduced.
    """
    out = [f"# {title}", "", "| candidate | collection |", "|---|---|"]
    out += [f"| {label} | {coll if '`' in coll else f'`{coll}`'} |" for label, coll in candidates.items()]
    out.append("")
    if visitsNote:
        out += [visitsNote, ""]
    summary = validationSummary(table)
    labels = list(candidates)
    for det, g in summary.groupby("detector", sort=False):
        out += [f"## {det}", "", "Median over images and species, px; offsets as absolute values.", ""]
        head = "| metric | " + " | ".join(f"{lab} (set) | {lab} (later)" for lab in labels) + " |"
        out += [head, "|---|" + "---|---|" * len(labels)]
        for metric, gm in g.groupby("metric", sort=True, observed=True):
            cells = []
            for lab in labels:
                row = gm[gm["calibration"] == lab]
                if row.empty:
                    cells += ["-", "-"]
                else:
                    r = row.iloc[0]
                    cells += [
                        f"{r['set']:.3f} (n={r['nSet']})",
                        f"{r['later']:.3f} (n={r['nLater']})" if r["nLater"] else "-",
                    ]
            out.append(f"| {metric} | " + " | ".join(cells) + " |")
        out.append("")
        if figures and det in figures:
            out += [f"![{det}]({figures[det]})", ""]
    out.append(
        "*vs calib: every line against the candidate calibration before the visit's adjustment (the drift it "
        "leaves); reserved RMS: the lines held out of the visit's own fit, against that fit (how well the adjusted "
        "map predicts lines it was not fitted to). The set's own images read near zero against a calibration "
        "built from them; the later nights are the fair comparison.*"
    )
    out.append("")
    return "\n".join(out)


def slitOffsetHealth(offsets: pd.DataFrame, *, threshold: float = 5.0) -> pd.DataFrame:
    """Flag each detector's slit-offset outliers from a collection's offsets.

    Runs `pfs.drp.qa.metrics.slitOffsets` per ``(arm, spectrograph)`` detector: detrends each MTP
    ferrule and flags the fibers whose offset is an outlier. Stack-free: it takes the offsets
    `pfs.drp.qa.comparison.butlerQueries.readSlitOffsets` read, not a Butler.

    Parameters
    ----------
    offsets : `pandas.DataFrame`
        ``arm``, ``spectrograph``, ``fiberId``, ``spatial``, ``spectral``, ``group`` (the MTP
        ferrule, ``"?"`` where unmapped), as `readSlitOffsets` returns.
    threshold : `float`, optional
        Per-fiber outlier sigma, passed to `flagOutliers`.

    Returns
    -------
    `pandas.DataFrame`
        `flagOutliers`' per-fiber table, stacked over detectors, with ``arm`` and ``spectrograph``.
    """
    from pfs.drp.qa.metrics.slitOffsets import flagOutliers, slitOffsetResiduals

    frames = []
    for (arm, spectrograph), g in offsets.groupby(["arm", "spectrograph"], sort=True):
        select = (g["group"].astype(str) != "?").to_numpy()
        residuals = slitOffsetResiduals(g["fiberId"], g["spatial"], g["spectral"], g["group"], select=select)
        flagged = flagOutliers(residuals, threshold=threshold)
        flagged["arm"] = arm
        flagged["spectrograph"] = int(spectrograph)
        frames.append(flagged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _confirmedTable(confirmed: pd.DataFrame, withStatus: bool = False) -> list[str]:
    """Return Markdown table rows for a `crossArmConfirmed` (or `compareRuns`) frame."""
    head = "| detector | fiberId | arms | max spatial (px) | max spectral (px) |"
    if withStatus:
        head += " status |"
    lines = [head, "|---|---|---|---|---|" + ("---|" if withStatus else "")]
    for row in confirmed.itertuples():
        cells = [
            f"SM{row.spectrograph}",
            str(int(row.fiberId)),
            str(row.arms),
            f"{row.maxSpatialResid:.3f}",
            f"{row.maxSpectralResid:.3f}",
        ]
        if withStatus:
            cells.append(str(row.status))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def buildSlitOffsetSection(
    confirmed: pd.DataFrame,
    previous: pd.DataFrame | None = None,
    *,
    previousLabel: str = "the previous run",
) -> str:
    """Return the Markdown slit-offset section: the confirmed outliers and the change from last run.

    Parameters
    ----------
    confirmed : `pandas.DataFrame`
        This calibration's cross-arm confirmed outliers (`crossArmConfirmed`).
    previous : `pandas.DataFrame`, optional
        The previous run's confirmed outliers; given, each fiber is marked ``new``, ``cleared`` or
        ``persisted`` and a change summary is shown (`compareRuns`, `changeSummary`).
    previousLabel : `str`, optional
        How the previous run is named in the text.

    Returns
    -------
    `str`
        The Markdown section.
    """
    from pfs.drp.qa.metrics.slitOffsets import changeSummary, compareRuns

    out = [
        "## Slit offsets",
        "",
        "Fibers whose slit offset is an outlier within its MTP ferrule in two or more arms of a "
        "spectrograph: a fault at or before the slit (a bad fiber) biases every arm. 1 px ≈ 71 µm at the "
        "slit. Chronic bad fibers persist run to run; a *new* one is a regression.",
        "",
    ]
    if previous is not None:
        comparison = compareRuns(confirmed, previous)
        summary = changeSummary(comparison)
        out.append(
            f"**{summary['nNew']} new**, {summary['nPersisted']} persisted and {summary['nCleared']} cleared "
            f"against {previousLabel}."
        )
        out.append("")
        ordered = comparison.sort_values(
            "status", key=lambda s: s.map({"new": 0, "persisted": 1, "cleared": 2})
        )
        out += _confirmedTable(ordered, withStatus=True) if len(ordered) else ["No confirmed outliers."]
    elif confirmed.empty:
        out.append("No confirmed slit-offset outliers.")
    else:
        out += _confirmedTable(confirmed)
    out.append("")
    return "\n".join(out)
