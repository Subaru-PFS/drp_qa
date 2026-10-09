"""Findings: a note for each image the gate warns about or fails, written from the data.

Comparison mode doesn't rely on anyone having noted a problem. For each ``WARN`` or ``FAIL``
image it says which metrics crossed which thresholds, how far the problem spreads, and what
the exposure's setup was, so a defocused spectrograph reads as "medFwhm FAIL on every arm of
SM1 for the whole sequence", and an arc taken with one fiber group lit shows that group. The
opdb's notes are shown beside it, never used to decide it.

Judging is `pfs.drp.qa.metrics.gate.judge`, the gate's own path, with the thresholds the task
ran with, less two cases it can't judge yet. Telescope focus sweeps change how much light enters
the fibers, not the spectrograph's line widths, so the flux-dependent metrics (`FLUX_METRICS`)
are not judged on them. Daily arcs and traces are often taken with one fiber group lit, which
cuts ``nLines`` to about a quarter and raises ``pctFlagged`` on the b and n arms; neither allows for
the fibers a design lights yet (PIPE2D-1935), so they aren't judged on them (`DAILY_UNJUDGED`).
"""

import re

import pandas as pd

from pfs.drp.qa.metrics.gate import STATUS_ORDER, configThresholds, judge, loadThresholds, thresholdsPath

__all__ = [
    "DAILY_UNJUDGED",
    "FLUX_METRICS",
    "describeSetup",
    "extentOf",
    "findings",
    "judgeImages",
    "lampsOf",
    "taskThresholds",
]

#: Metrics that depend on how much light reached the fibers.
FLUX_METRICS = ("nLines", "pctFlagged")

#: Metrics not judged on daily arcs and traces until they allow for the fibers lit (PIPE2D-1935).
DAILY_UNJUDGED = ("nLines", "pctFlagged")

# The ways an IIC command names a lamp: head='sps iis on=neon ...', iisNeon=30, argon=10.
_LAMP_RE = re.compile(
    r"\biis\s+on=(\w+)|\biis([A-Z][a-z]+)=\d|\b(hgcd|hgar|argon|xenon|neon|krypton|halogen|qth)=([\d.]+)"
)


def lampsOf(cmdStr: str | None) -> list[str]:
    """Return the lamps an IIC command turns on.

    Parameters
    ----------
    cmdStr : `str` or `None`
        ``iic_sequence.cmd_str``.

    Returns
    -------
    `list` [`str`]
        Lower-case lamp names in order of appearance, without repeats;
        ``(IIS)`` follows a lamp lit through the engineering fibers.
    """
    if not cmdStr or not isinstance(cmdStr, str):
        return []
    lamps: list[str] = []
    for match in _LAMP_RE.finditer(cmdStr):
        if match.group(1):
            lamp = f"{match.group(1).lower()} (IIS)"
        elif match.group(2):
            lamp = f"{match.group(2).lower()} (IIS)"
        elif float(match.group(4)) > 0:
            lamp = match.group(3)
        else:
            continue
        if lamp not in lamps:
            lamps.append(lamp)
    return lamps


def describeSetup(visit: pd.Series) -> str:
    """Return the setup of a visit in a few words.

    Parameters
    ----------
    visit : `pandas.Series`
        A row of `pfs.drp.qa.comparison.classify.classifyVisits`.

    Returns
    -------
    `str`
        E.g. ``"scienceArc 'Arc: Neon', neon; 60 s"`` or ``"scienceObject,
        telescope focus sweep; 60 s"``.
    """
    parts = [str(visit.get("sequence_type") or "no sequence")]
    name = visit.get("sequence_name")
    name = name.strip() if isinstance(name, str) else name
    if isinstance(name, str) and name and visit.get("category") != "science":
        parts[0] += f" {name!r}"  # science sequence names identify programs
    if visit.get("cadence") == "daily":
        parts[0] += " (daily)"
    group = visit.get("group_name")
    if isinstance(group, str) and group.strip() and visit.get("category") != "science":
        parts.append(f"group {group.strip()}")
    lamps = lampsOf(visit.get("cmd_str"))
    if lamps:
        parts.append(", ".join(lamps))
    if visit.get("focusSweep"):
        parts.append("telescope focus sweep")
    if visit.get("dithered"):
        parts.append("dithered")
    text = ", ".join(parts)
    exptime = visit.get("exptime")
    if exptime is not None and not pd.isna(exptime):
        text += f"; {float(exptime):.0f} s"
    return text


def taskThresholds(config) -> list[pd.DataFrame]:
    """Return the thresholds an ``imageQualityQa`` config judges with, as the task builds them.

    Parameters
    ----------
    config : `Any`
        An ``ImageQualityQaConfig``, e.g. the ``imageQualityQa_config`` dataset
        of the output collection; duck-typed.

    Returns
    -------
    `list` [`pandas.DataFrame`]
        The thresholds file's table, if any, then the config's.
    """
    files = [thresholdsPath(config.thresholdsFile)] if config.thresholdsFile else []
    return [*loadThresholds(files), configThresholds(config)]


def judgeImages(metrics: pd.DataFrame, visits: pd.DataFrame, thresholds) -> pd.DataFrame:
    """Judge every metric of every image as the gate does, less what it can't judge yet.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        ``iqQaMetrics`` rows, with ``visit``, ``arm`` and ``spectrograph``.
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.classify.classifyVisits`.
    thresholds : `pandas.DataFrame`, path, or a sequence of them
        The thresholds the task ran with, highest priority first.

    Returns
    -------
    `pandas.DataFrame`
        `pfs.drp.qa.metrics.gate.judge`'s table, with the status cleared for
        a `FLUX_METRICS` entry on a focus-sweep visit and a `DAILY_UNJUDGED`
        entry on a daily arc or trace.
    """
    judged = judge(metrics, thresholds)
    sweeps = set(visits.loc[visits["focusSweep"].astype(bool), "pfs_visit_id"])
    daily = set(visits.loc[visits["cadence"] == "daily", "pfs_visit_id"])
    skip = judged["visit"].isin(sweeps) & judged["metric"].isin(FLUX_METRICS)
    skip |= judged["visit"].isin(daily) & judged["metric"].isin(DAILY_UNJUDGED)
    judged.loc[skip, ["status", "reason"]] = ""
    return judged


def extentOf(verdicts: pd.DataFrame) -> pd.Series:
    """Return how far each image's problem spreads within its visit.

    Parameters
    ----------
    verdicts : `pandas.DataFrame`
        One row per judged image: ``visit``, ``arm``, ``spectrograph`` and
        ``status`` (``PASS``, ``WARN``, ``FAIL`` or empty).

    Returns
    -------
    `pandas.Series`
        For a ``WARN`` or ``FAIL`` image: ``visit`` (every judged image of the
        visit), ``visit (n of N)`` (most of them), ``SM<n>`` (every
        arm of its spectrograph, at least two), ``<arm> arm`` (its arm on every
        spectrograph, at least two) or ``detector``; empty for the others. A
        problem on most of a visit is one problem, so it gets one label.
    """
    bad = verdicts["status"].isin(["WARN", "FAIL"])
    extent = pd.Series("", index=verdicts.index, dtype=object)
    for _, group in verdicts.groupby("visit"):
        groupBad = bad[group.index]
        if not groupBad.any():
            continue
        if len(group) > 1 and 2 * groupBad.sum() > len(group):
            label = "visit" if groupBad.all() else f"visit ({groupBad.sum()} of {len(group)})"
            extent[group.index[groupBad]] = label
            continue
        for index in group.index[groupBad]:
            arm, spectrograph = verdicts.at[index, "arm"], verdicts.at[index, "spectrograph"]
            sameModule = group.index[group["spectrograph"] == spectrograph]
            sameArm = group.index[group["arm"] == arm]
            if len(sameModule) > 1 and bad[sameModule].all():
                extent[index] = f"SM{spectrograph}"
            elif len(sameArm) > 1 and bad[sameArm].all():
                extent[index] = f"{arm} arm"
            else:
                extent[index] = "detector"
    return extent


def findings(
    metrics: pd.DataFrame,
    judged: pd.DataFrame,
    visits: pd.DataFrame,
    notes: pd.DataFrame | None = None,
    visitSet=None,
) -> pd.DataFrame:
    """Write a finding for each image that warns or fails.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        ``iqQaMetrics`` rows, as given to `judgeImages`.
    judged : `pandas.DataFrame`
        From `judgeImages`.
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.classify.classifyVisits`.
    notes : `pandas.DataFrame`, optional
        From `pfs.drp.qa.comparison.queries.readNotes`.
    visitSet : `pfs.drp.qa.metrics.validationVisits.ValidationVisitSet`, optional
        The validation visit set: a finding it records as known-bad is
        expected.

    Returns
    -------
    `pandas.DataFrame`
        One row per ``WARN`` or ``FAIL`` image: ``visit``, ``arm``,
        ``spectrograph``, ``night``, ``sequence_type``, ``category``,
        ``validated`` (a gated type), ``iic_sequence_id``, ``status``, ``metrics`` (each crossing, e.g.
        ``medFWHM=3.20px >= fail threshold 2.8px``), ``extent`` (`extentOf`, with
        ``, whole sequence`` when every visit of the sequence shares it),
        ``expected`` (the metrics the validation set records the image as
        known-bad for, ``any`` for every metric, or empty), ``setup``
        (`describeSetup`) and ``notes`` (the opdb's, joined by `` | ``).
    """
    keys = ["visit", "arm", "spectrograph"]
    images = metrics.reset_index(drop=True)[keys].copy()
    rank = judged["status"].map({status: i for i, status in enumerate(STATUS_ORDER)})
    worst = rank.groupby(judged["row"]).max()
    images["status"] = [STATUS_ORDER[int(r)] if not pd.isna(r) else "" for r in worst.reindex(images.index)]
    crossings = judged[judged["status"].isin(["WARN", "FAIL"])]
    images["metrics"] = crossings.groupby("row")["reason"].agg("; ".join).reindex(images.index).fillna("")
    images["extent"] = extentOf(images)
    images["expected"] = _expected(metrics.reset_index(drop=True), visitSet)

    info = visits.rename(columns={"pfs_visit_id": "visit"}).set_index("visit")
    result = images[images["status"].isin(["WARN", "FAIL"])].copy()
    if result.empty:
        columns = [*keys, "night", "sequence_type", "category", "validated", "iic_sequence_id", "status"]
        return pd.DataFrame(columns=[*columns, "metrics", "extent", "expected", "setup", "notes"])
    for column in ("night", "sequence_type", "category", "validated", "iic_sequence_id"):
        result[column] = result["visit"].map(info[column])
    result["setup"] = [
        describeSetup(info.loc[visit]) if visit in info.index else "" for visit in result["visit"]
    ]
    result["extent"] = _wholeSequence(result, images, info)
    result["notes"] = _notesFor(result, info, notes)
    columns = [*keys, "night", "sequence_type", "category", "validated", "iic_sequence_id", "status"]
    columns += ["metrics", "extent", "expected", "setup", "notes"]
    return result[columns].sort_values(keys, ignore_index=True)


def _expected(metrics: pd.DataFrame, visitSet) -> list[str]:
    """Return, for each image, the metrics the validation set records it as known-bad for."""
    if visitSet is None:
        return [""] * len(metrics)
    seqNames = metrics["seqName"] if "seqName" in metrics else pd.Series([None] * len(metrics))
    result = []
    for visit, arm, spectrograph, seqName in zip(
        metrics["visit"], metrics["arm"], metrics["spectrograph"], seqNames, strict=True
    ):
        entries = visitSet.find(
            int(visit), arm, int(spectrograph), seqName if isinstance(seqName, str) else None
        )
        bad = {entry.metric or "any" for entry in entries if entry.expect in ("WARN", "FAIL")}
        result.append(",".join(sorted(bad)))
    return result


def _wholeSequence(result: pd.DataFrame, images: pd.DataFrame, info: pd.DataFrame) -> pd.Series:
    """Append ``, whole sequence`` where every judged visit of the sequence shares the extent."""
    extent = result["extent"].copy()
    sequenceOf = images["visit"].map(info["iic_sequence_id"])
    for (sequence, label), group in result.groupby(
        ["iic_sequence_id", _extentKind(result["extent"])], dropna=True
    ):
        sequenceVisits = set(images.loc[sequenceOf == sequence, "visit"])
        if len(sequenceVisits) < 2:
            continue
        if label == "detector":
            # Only the detectors that are bad in every visit of the sequence.
            visitsPerDetector = group.groupby(["arm", "spectrograph"])["visit"].transform("nunique")
            extent[group.index[(visitsPerDetector == len(sequenceVisits)).to_numpy()]] = (
                label + ", whole sequence"
            )
        elif set(group["visit"]) == sequenceVisits:
            extent[group.index] = group["extent"] + ", whole sequence"
    return extent


def _extentKind(extent: pd.Series) -> pd.Series:
    """Return extents without a visit's count, so ``visit (11 of 12)`` and ``visit`` match."""
    return extent.str.replace(r"^visit \(.*\)$", "visit", regex=True).rename("kind")


def _notesFor(result: pd.DataFrame, info: pd.DataFrame, notes: pd.DataFrame | None) -> pd.Series:
    """Return the opdb's notes on each finding's visit, sequence or camera."""
    if notes is None or notes.empty:
        return pd.Series("", index=result.index, dtype=object)
    visitNotes = notes[notes["source"] == "obslog"].groupby("pfs_visit_id")["note"].agg(list)
    sequenceNotes = notes[notes["source"] == "obslog_sequence"].groupby("iic_sequence_id")["note"].agg(list)
    cameraNotes = (
        notes[notes["source"] == "sps_annotation"].groupby(["pfs_visit_id", "camera"])["note"].agg(list)
    )
    texts = []
    for _, row in result.iterrows():
        found = list(visitNotes.get(row["visit"], []))
        if not pd.isna(row["iic_sequence_id"]):
            found += sequenceNotes.get(row["iic_sequence_id"], [])
        found += cameraNotes.get((row["visit"], f"{row['arm']}{row['spectrograph']}"), [])
        texts.append(" | ".join(dict.fromkeys(str(note) for note in found)))
    return pd.Series(texts, index=result.index, dtype=object)
