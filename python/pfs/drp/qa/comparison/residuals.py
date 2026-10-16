"""Line residuals in comparison mode: ``dmLineResiduals``' verdicts beside ``imageQualityQa``'s.

``dmLineResiduals`` (PIPE2D-1921, PIPE2D-1956) judges the same detector images as the gate,
against ``detectorMap_calib`` (drift) and against the visit's own ``detectorMap`` (the scatter it
leaves on the lines the fit held back). Its verdict is stored in ``dmLineResidualSummary`` and
each crossing is a ``WARN`` or ``FAIL`` row of ``dmLineResidualMetrics``. This module turns them
into findings and tables for the report, as `pfs.drp.qa.comparison.findings` does for the gate's.
Stack-free.
"""

import pandas as pd

from pfs.drp.qa.comparison.findings import describeImages
from pfs.drp.qa.metrics.gate import STATUS_ORDER
from pfs.drp.qa.metrics.lineResiduals import RESERVED, VISIT

__all__ = ["describeCrossing", "residualFindings", "residualProblems", "scatterTable"]

#: The keys of a detector image.
KEYS = ["visit", "arm", "spectrograph"]

#: The robust RMS metrics, by the component they measure.
_SCATTER = {"robustRmsDeltaX": "x", "robustRmsDeltaY": "y"}


def describeCrossing(row) -> str:
    """Return one judged row of ``dmLineResidualMetrics`` as text.

    Parameters
    ----------
    row : `Any`
        A row with ``reference``, ``description``, ``metric``, ``value`` and
        ``status``.

    Returns
    -------
    `str`
        E.g. ``calib Trace medianDeltaX=+1.234 px WARN`` (offsets signed,
        scatter not).
    """
    value = f"{row.value:+.3f}" if str(row.metric).startswith("median") else f"{row.value:.3f}"
    return f"{row.reference} {row.description} {row.metric}={value} px {row.status}"


def residualFindings(
    metrics: pd.DataFrame, summary: pd.DataFrame, visits: pd.DataFrame, notes: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Write a finding for each image whose line residuals warn or fail.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        ``dmLineResidualMetrics`` rows of the images.
    summary : `pandas.DataFrame`
        ``dmLineResidualSummary`` rows, one per image.
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.classify.classifyVisits`.
    notes : `pandas.DataFrame`, optional
        From `pfs.drp.qa.comparison.queries.readNotes`.

    Returns
    -------
    `pandas.DataFrame`
        As `pfs.drp.qa.comparison.findings.findings`, with ``metrics`` the
        crossings (`describeCrossing`, joined by ``; ``), ``expected`` empty,
        and ``recommendedAction`` last.
    """
    crossings = metrics[metrics["status"].isin(["WARN", "FAIL"])]
    text = (
        crossings.assign(text=[describeCrossing(row) for row in crossings.itertuples()])
        .groupby(KEYS)["text"]
        .agg("; ".join)
        .rename("metrics")
        .reset_index()
    )
    images = (
        summary[[*KEYS, "qaStatus", "recommendedAction"]]
        .drop_duplicates(KEYS)
        .merge(text, on=KEYS, how="left")
    )
    images["status"] = images["qaStatus"].where(images["qaStatus"].isin(STATUS_ORDER), "")
    images["metrics"] = images["metrics"].fillna("")
    return describeImages(images[[*KEYS, "status", "metrics", "recommendedAction"]], visits, notes)


def residualProblems(findings: pd.DataFrame) -> pd.DataFrame:
    """Group line-residual findings into problems: the same crossings, on the same arm, in the same setup.

    Parameters
    ----------
    findings : `pandas.DataFrame`
        From `residualFindings`.

    Returns
    -------
    `pandas.DataFrame`
        One row per problem: ``status`` (the worst), ``crossed`` (the
        reference, species and metric of each crossing), ``arm``, ``setup``
        (without the exposure time), ``extent`` (the commonest, without a
        visit's counts), ``action`` (the severest recommended), ``images``,
        ``visits``, ``nights`` (first to last) and ``examples`` (up to three
        visits); failures first, then by images.
    """
    columns = [
        "status",
        "crossed",
        "arm",
        "setup",
        "extent",
        "action",
        "images",
        "visits",
        "nights",
        "examples",
    ]
    if findings.empty:
        return pd.DataFrame(columns=columns)
    actions = ["NOMINAL", "UNJUDGED", "INSUFFICIENT_DATA", "APPLY_SHIFT", "RECALIBRATE"]
    frame = findings.assign(
        crossed=findings["metrics"]
        .str.findall(r"(\S+ \S+ \w+)=")
        .map(lambda names: ", ".join(dict.fromkeys(names))),
        setupKind=findings["setup"].str.split(";").str[0],
        extentKind=findings["extent"].str.replace(r"^visit \(.*?\)", "most of the visit", regex=True),
        rank=findings["status"].map({"FAIL": 0, "WARN": 1}),
        actionRank=findings["recommendedAction"].map(lambda a: actions.index(a) if a in actions else -1),
    )
    result = (
        frame.groupby(["crossed", "arm", "setupKind"], dropna=False)
        .agg(
            rank=("rank", "min"),
            extentKind=("extentKind", lambda e: e.value_counts().index[0]),
            actionRank=("actionRank", "max"),
            images=("visit", "size"),
            visits=("visit", "nunique"),
            first=("night", "min"),
            last=("night", "max"),
            examples=("visit", lambda v: ", ".join(str(x) for x in sorted(v.unique())[:3])),
        )
        .reset_index()
    )
    result["status"] = result["rank"].map({0: "FAIL", 1: "WARN"})
    result["action"] = [actions[int(rank)] if rank >= 0 else "" for rank in result["actionRank"]]
    result["nights"] = [
        str(first) if first == last else f"{first} to {last}"
        for first, last in zip(result["first"], result["last"], strict=True)
    ]
    result = result.rename(columns={"setupKind": "setup", "extentKind": "extent"})
    return result.sort_values(["rank", "images"], ascending=[True, False])[columns].reset_index(drop=True)


def scatterTable(
    metrics: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    labels: tuple[str, str] = ("run", "reference"),
) -> pd.DataFrame:
    """Summarize the scatter the adjusted map leaves, by arm and species, beside a reference run's.

    The robust RMS of ``RESERVED`` lines against the visit's ``detectorMap``:
    what a threshold on it would be derived from (PIPE2D-1956).

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        ``dmLineResidualMetrics`` rows of this run's images.
    reference : `pandas.DataFrame`, optional
        The same for the reference run.
    labels : `tuple` [`str`, `str`], optional
        Names for the two, in the column headings.

    Returns
    -------
    `pandas.DataFrame`
        ``arm``, ``description``, ``component`` (``x`` or ``y``), and per run
        ``images <label>``, ``median <label>`` and ``p95 <label>`` (px,
        rounded to 3 decimals); empty when nothing was measured.
    """

    def stats(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        rows = frame[
            (frame["reference"] == VISIT)
            & (frame["statusType"] == RESERVED)
            & frame["metric"].isin(_SCATTER)
            & frame["value"].notna()
        ]
        rows = rows.assign(component=rows["metric"].map(_SCATTER))
        table = rows.groupby(["arm", "description", "component"])["value"].agg(
            ["size", "median", lambda v: v.quantile(0.95)]
        )
        table.columns = [f"images {label}", f"median {label}", f"p95 {label}"]
        return table

    table = stats(metrics, labels[0])
    if table.empty:
        return pd.DataFrame(columns=["arm", "description", "component", *table.columns])
    if reference is not None and not reference.empty:
        table = table.join(stats(reference, labels[1]), how="left")
    return table.round(3).reset_index()
