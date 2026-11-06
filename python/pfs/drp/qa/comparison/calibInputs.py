"""Judge a calibration set's exposures per detector, for whoever builds the calibrations (PIPE2D-1941).

A run's calibration set is a few sequences of arc and quartz exposures (three per arc lamp, ten
traces), taken within an hour. ``detectorMap_calib`` and the fiber profiles are built per detector
from them, so an exposure is judged per detector image, never per arm: a detector that fails on
one exposure is left out of that exposure's inputs and nothing else is.

Each detector image is judged on what comparison mode measured on it (`pfs.drp.qa.imageQualityQa`
and `pfs.drp.qa.dmLineResiduals`, read from their collection), with no reference but the other
exposures of the same sequence on the same detector:

* the gate's verdict and the line-residual verdict (reserved lines against the visit's own
  detectorMap), as stored;
* consistency within the sequence: FWHM and reserved-line RMS relative to the detector's median
  over the sequence, which catches a detector that goes wrong mid-sequence without knowing what
  "right" looks like;
* the lamp: each species' line count relative to the detector's best exposure of the sequence, which
  shows warm-up and fading (PIPE2D-1929) and marks weak exposures, usable but with fewer lines.

Stack-free: DataFrames in, a Markdown report out. `bin.src/calibInputsReport.py` reads the tables
from a Butler or from `bin.src/exportMetrics.py` exports and the visit listing from comparison mode's
``fetch``.
"""

import datetime
import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from pfs.drp.qa.comparison.findings import lampsOf

__all__ = [
    "LAMP_SPECIES",
    "Thresholds",
    "Verdict",
    "buildReport",
    "coverage",
    "detectorTable",
    "goNoGo",
    "judge",
    "lampOf",
    "lineThresholds",
    "lineVerdicts",
    "parseReason",
    "pick",
    "setVisits",
    "speciesTable",
    "summary",
    "trends",
]

#: The line species each lamp gives, by the lamp name `lampsOf` returns (``(IIS)`` stripped).
LAMP_SPECIES: dict[str, tuple[str, ...]] = {
    "argon": ("ArI",),
    "xenon": ("XeI",),
    "krypton": ("KrI",),
    "neon": ("NeI",),
    "hgcd": ("HgI", "CdI"),
    "hgar": ("HgI", "ArI"),
    "helium": ("HeI",),
    "hydrogen": ("HI",),
    "halogen": ("Trace",),
    "quartz": ("Trace",),
    "qth": ("Trace",),
}

#: The sequence types a calibration set is made of.
SET_TYPES = ("scienceArc", "scienceTrace")


class Verdict:
    """The verdicts on a detector image, worst first."""

    OUT = "out"
    CHECK = "check"
    WEAK = "weak"
    USE = "use"
    ORDER = (OUT, CHECK, WEAK, USE)


@dataclass(frozen=True)
class Thresholds:
    """What marks a detector image, relative to the same detector's other exposures of the sequence.

    Parameters
    ----------
    weakLines : `float`
        A species with fewer lines than this fraction of the detector's best exposure of the
        sequence makes the image *weak*: usable, with fewer lines.
    fwhmCheck, fwhmOut : `float`
        FWHM more than this factor from the detector's sequence median, either way: *check*, *out*.
    rmsCheck, rmsOut : `float`
        Reserved-line RMS (the larger of x and y) more than this many times the detector's sequence
        median: *check*, *out*.
    """

    weakLines: float = 0.8
    fwhmCheck: float = 1.10
    fwhmOut: float = 1.25
    rmsCheck: float = 1.5
    rmsOut: float = 2.0


def lampOf(cmdStr: str | None, sequenceType: str | None = None) -> str | None:
    """Return the lamp a set sequence used, e.g. ``neon``, ``hgcd``, ``neon (IIS)``, ``halogen``.

    From the IIC command (`lampsOf`); a trace with no lamp named is ``halogen``.
    """
    lamps = lampsOf(cmdStr)
    if lamps:
        return lamps[0]
    return "halogen" if sequenceType == "scienceTrace" else None


def speciesOf(lamp: str | None) -> tuple[str, ...]:
    """Return the species a lamp name from `lampOf` gives; empty if unknown."""
    if not lamp:
        return ()
    return LAMP_SPECIES.get(lamp.replace(" (IIS)", "").strip().lower(), ())


def setVisits(listing: pd.DataFrame, visits: Iterable[int] | None = None) -> pd.DataFrame:
    """Return the set's visits with their lamp and place in their sequence.

    Parameters
    ----------
    listing : `pandas.DataFrame`
        Comparison mode's visit listing (`pfs.drp.qa.comparison.queries.readVisitListing`), or its
        classified form; ``sequence_type``, ``iic_sequence_id``, ``cmd_str`` and ``time_exp_start``
        are used.
    visits : iterable of `int`, optional
        The set's visits; default every arc and trace in ``listing``.

    Returns
    -------
    `pandas.DataFrame`
        One row per visit, in time order: ``visit``, ``iic_sequence_id``, ``sequence_type``,
        ``sequence_name``, ``lamp``, ``species`` (tuple), ``k`` (1-based place in the sequence),
        ``n`` (the sequence's visits), ``exptime``, ``time_exp_start``, ``cameras``.
    """
    rows = listing[listing["sequence_type"].isin(SET_TYPES)]
    if visits is not None:
        rows = rows[rows["pfs_visit_id"].isin({int(v) for v in visits})]
    rows = rows.dropna(subset=["iic_sequence_id"]).sort_values("pfs_visit_id")
    result = pd.DataFrame(
        {
            "visit": rows["pfs_visit_id"].astype(int).to_numpy(),
            "iic_sequence_id": rows["iic_sequence_id"].astype(int).to_numpy(),
            "sequence_type": rows["sequence_type"].to_numpy(),
            "sequence_name": rows.get("sequence_name", pd.Series("", index=rows.index)).fillna("").to_numpy(),
            "lamp": [
                lampOf(c, t)
                for c, t in zip(
                    rows.get("cmd_str", pd.Series(None, index=rows.index)),
                    rows["sequence_type"],
                    strict=False,
                )
            ],
            "exptime": rows.get("exptime", pd.Series(float("nan"), index=rows.index)).to_numpy(),
            "time_exp_start": rows.get("time_exp_start", pd.Series(pd.NaT, index=rows.index)).to_numpy(),
            "cameras": rows.get("cameras", pd.Series("", index=rows.index)).fillna("").to_numpy(),
        }
    )
    result["species"] = result["lamp"].map(speciesOf)
    result["k"] = result.groupby("iic_sequence_id").cumcount() + 1
    result["n"] = result.groupby("iic_sequence_id")["visit"].transform("size")
    return result.reset_index(drop=True)


_DETECTOR = ["visit", "arm", "spectrograph"]


def speciesTable(visits: pd.DataFrame, lineMetrics: pd.DataFrame) -> pd.DataFrame:
    """Each lamp species on each detector image, relative to the detector's best exposure of the sequence.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        From `setVisits`.
    lineMetrics : `pandas.DataFrame`
        ``dmLineResidualMetrics`` rows (long format) of the set's images.

    Returns
    -------
    `pandas.DataFrame`
        One row per detector image and species of its lamp: ``visit``, ``arm``, ``spectrograph``,
        ``iic_sequence_id``, ``k``, ``species``, ``nLines``, ``linesPerFiber``, ``rmsX``, ``rmsY``
        (RESERVED lines against the visit's detectorMap, pixels; NaN when not measured) and
        ``linesRatio``, ``nLines`` over the detector's best exposure of the sequence.
    """
    columns = [
        *_DETECTOR,
        "iic_sequence_id",
        "k",
        "species",
        "nLines",
        "linesPerFiber",
        "rmsX",
        "rmsY",
        "linesRatio",
    ]
    if lineMetrics.empty or visits.empty:
        return pd.DataFrame(columns=columns)
    rows = lineMetrics[lineMetrics["reference"] == "visit"]
    both = rows[rows["statusType"].isin(["BOTH", "UNFITTED"])].pivot_table(
        index=[*_DETECTOR, "description"], columns="metric", values="value", aggfunc="first"
    )
    reserved = rows[rows["statusType"] == "RESERVED"].pivot_table(
        index=[*_DETECTOR, "description"], columns="metric", values="value", aggfunc="first"
    )
    table = both.reindex(columns=["nLines", "linesPerFiber"]).join(
        reserved.reindex(columns=["robustRmsDeltaX", "robustRmsDeltaY"]).rename(
            columns={"robustRmsDeltaX": "rmsX", "robustRmsDeltaY": "rmsY"}
        )
    )
    table = table.reset_index().rename(columns={"description": "species"})
    table = table.merge(
        visits[["visit", "iic_sequence_id", "k", "species"]].rename(columns={"species": "lampSpecies"}),
        on="visit",
    )
    table = table[[s in lamp for s, lamp in zip(table["species"], table["lampSpecies"], strict=False)]].drop(
        columns="lampSpecies"
    )
    best = table.groupby(["iic_sequence_id", "arm", "spectrograph", "species"])["nLines"].transform("max")
    table["linesRatio"] = table["nLines"] / best
    return table.reindex(columns=columns).sort_values([*_DETECTOR, "species"]).reset_index(drop=True)


_STATUS_RANK = {"": 0, "PASS": 1, "UNKNOWN": 1, "WARN": 2, "FAIL": 3}
#: Short names for the judged metrics, for the report.
_METRIC_NAMES = {
    "robustRmsDeltaX": "RMS x",
    "robustRmsDeltaY": "RMS y",
    "medianDeltaX": "offset x",
    "medianDeltaY": "offset y",
    "medFWHM": "FWHM",
    "linesPerFiber": "lines/fiber",
}


#: One judged metric in a stored ``qaReason``: ``visit NeI: robustRmsDeltaX=2.038px >= fail threshold 0.123px``
#: or the gate's ``medFWHM=3.06px >= fail threshold 2.951px``.
_REASON_ENTRY = re.compile(
    r"(?:(?P<reference>\w+) (?P<species>\w+): )?(?P<metric>\|?\w+\|?)=(?P<value>-?[\d.]+)(?P<unit>px|%)? "
    r"(?P<op>[<>]=?) (?P<level>warn|fail) threshold (?P<threshold>-?[\d.]+)"
)


def parseReason(reason: str) -> list[dict]:
    """Return the judged metrics a stored ``qaReason`` names: reference, species, metric, value, op, level, threshold."""
    out = []
    for m in _REASON_ENTRY.finditer(reason or ""):
        d = m.groupdict()
        d["value"], d["threshold"] = float(d["value"]), float(d["threshold"])
        d["metric"] = d["metric"].strip("|")
        out.append(d)
    return out


def lineVerdicts(
    visits: pd.DataFrame, lineMetrics: pd.DataFrame, lineSummary: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return each detector image's line-residual verdict on its lamp's own lines against its own detectorMap.

    The worst ``status`` among the ``dmLineResidualMetrics`` rows with ``reference`` ``visit``
    whose species the lamp gives (plus ``Trace``), with those rows' metric and value as the reason,
    and the threshold when ``lineSummary``'s ``qaReason`` names it.
    Columns ``visit``, ``arm``, ``spectrograph``, ``lineStatus``, ``lineReason``.
    """
    columns = [*_DETECTOR, "lineStatus", "lineReason"]
    if lineMetrics.empty or "status" not in lineMetrics:
        return pd.DataFrame(columns=columns)
    thresholds: dict[tuple, dict] = {}
    if lineSummary is not None and not lineSummary.empty and "qaReason" in lineSummary:
        for r in lineSummary.itertuples():
            for entry in parseReason(getattr(r, "qaReason", "")):
                if entry["reference"] == "visit":
                    thresholds[
                        (int(r.visit), r.arm, int(r.spectrograph), entry["species"], entry["metric"])
                    ] = entry
    rows = lineMetrics[(lineMetrics["reference"] == "visit") & (lineMetrics["status"].fillna("") != "")]
    rows = rows.merge(visits[["visit", "species"]].rename(columns={"species": "lampSpecies"}), on="visit")
    rows = rows[
        [d == "Trace" or d in lamp for d, lamp in zip(rows["description"], rows["lampSpecies"], strict=True)]
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows = rows.assign(rank=rows["status"].map(_STATUS_RANK).fillna(0))
    out = []
    for key, g in rows.groupby(_DETECTOR):
        worst = g["rank"].max()
        status = g.loc[g["rank"] == worst, "status"].iloc[0]
        bad = g[g["rank"] >= 2].sort_values("rank", ascending=False)
        parts = []
        for r in bad.itertuples():
            t = thresholds.get((int(r.visit), r.arm, int(r.spectrograph), r.description, r.metric))
            limit = f" ({t['level']} {t['op']} {t['threshold']:.3g})" if t else f" {r.status}"
            parts.append(f"{r.description} {_METRIC_NAMES.get(r.metric, r.metric)} {r.value:.3f} px{limit}")
        out.append((*key, status, "; ".join(parts)))
    return pd.DataFrame(out, columns=columns)


def lineThresholds(
    arm: str,
    spectrograph: int,
    species: Iterable[str],
    cameraSet: str | None,
    obsType: str | None,
    thresholdsFile: str | None = None,
) -> dict[str, dict[str, tuple[float | None, float | None]]]:
    """Return the reserved-line RMS thresholds `dmLineResiduals` applies to one image, per species.

    Looked up as the task does, through `pfs.drp.qa.metrics.gate.judge` on the population
    (``reference`` ``visit``, ``statusType`` ``RESERVED``, arm, species, ``cameraSet``,
    ``obsType``) in ``thresholdsFile`` (the name ``dmLineResidualSummary.thresholdsFiles``
    records; default the packaged ``dmLineResidualThresholds-run25.yaml``).

    Returns
    -------
    `dict`
        ``{species: {"robustRmsDeltaX": (warn, fail), "robustRmsDeltaY": (warn, fail)}}``;
        a limit is None where the file judges nothing (e.g. Hg and Cd).
    """
    from pfs.drp.qa.metrics.gate import judge as gateJudge
    from pfs.drp.qa.metrics.gate import thresholdsPath

    species = [sp for sp in species if sp != "Trace"]
    if not species:
        return {}
    path = thresholdsPath(thresholdsFile or "dmLineResidualThresholds-run25.yaml")
    frame = pd.DataFrame(
        {
            "visit": 0,
            "arm": arm,
            "spectrograph": int(spectrograph),
            "reference": "visit",
            "statusType": "RESERVED",
            "description": species,
            "cameraSet": cameraSet,
            "obsType": obsType,
            "robustRmsDeltaX": 0.0,
            "robustRmsDeltaY": 0.0,
        }
    )
    judged = gateJudge(frame, path)
    out: dict[str, dict[str, tuple[float | None, float | None]]] = {sp: {} for sp in species}
    for r in judged.itertuples():
        if r.metric in ("robustRmsDeltaX", "robustRmsDeltaY"):
            warn = None if pd.isna(r.warn) else float(r.warn)
            fail = None if pd.isna(r.fail) else float(r.fail)
            out[species[int(r.row)]][r.metric] = (warn, fail)
    return out


def detectorTable(
    visits: pd.DataFrame,
    iqMetrics: pd.DataFrame,
    lineMetrics: pd.DataFrame,
    lineSummary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per detector image of the set, with its verdicts and its place in the sequence.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        From `setVisits`.
    iqMetrics : `pandas.DataFrame`
        ``iqQaMetrics`` rows of the set's images: ``medFwhm``, ``fitReservedXRms``,
        ``fitReservedYRms``, ``qaStatus``, ``qaReason``, ``traceOnly`` are used. A ``traceOnly``
        image's FWHM is read from the fiberProfiles calibration, not measured on the image, so it
        takes no part in the sequence comparison. An image whose gate judged nothing (``UNKNOWN``)
        and that has no line species measured is left out: it is unassessed, and `goNoGo` reports
        it as missing rather than go.
    lineMetrics : `pandas.DataFrame`
        ``dmLineResidualMetrics`` rows of the set's images (`speciesTable`).
    lineSummary : `pandas.DataFrame`, optional
        ``dmLineResidualSummary`` rows: only their ``qaReason`` is used, for the thresholds. The
        line-residual verdict is taken from ``lineMetrics``: the worst ``status`` among the lamp's
        own species (and traces) against the visit's own detectorMap. The summary's ``qaStatus``
        also judges the drift against the calibration in use and species the lamp doesn't give,
        neither of which is the image's doing.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph``, ``detector`` (``n4``), ``iic_sequence_id``, ``lamp``,
        ``k``, ``n``, ``gateStatus``, ``gateReason``, ``lineStatus``, ``lineReason``, ``medFwhm``,
        ``fwhmRatio`` (to the detector's sequence median), ``reservedRms`` (the larger of x and y),
        ``rmsRatio`` (likewise), ``minLinesRatio`` (the lamp's weakest species, `speciesTable`),
        ``nSpecies`` (species measured).
    """
    keep = [
        "visit",
        "arm",
        "spectrograph",
        "medFwhm",
        "fitReservedXRms",
        "fitReservedYRms",
        "qaStatus",
        "qaReason",
        "traceOnly",
    ]
    iq = iqMetrics.reindex(columns=keep).rename(columns={"qaStatus": "gateStatus", "qaReason": "gateReason"})
    iq.loc[iq["traceOnly"].fillna(False).astype(bool), "medFwhm"] = float("nan")
    iq = iq.drop(columns="traceOnly")
    table = iq.merge(visits[["visit", "iic_sequence_id", "lamp", "k", "n"]], on="visit")
    lines = lineVerdicts(visits, lineMetrics, lineSummary)
    table = table.merge(lines, on=_DETECTOR, how="left")
    table["lineStatus"] = table["lineStatus"].fillna("")
    table["lineReason"] = table["lineReason"].fillna("")
    table["gateReason"] = table["gateReason"].fillna("")
    table["detector"] = table["arm"].astype(str) + table["spectrograph"].astype(str)
    group = ["iic_sequence_id", "arm", "spectrograph"]
    table["fwhmRatio"] = table["medFwhm"] / table.groupby(group)["medFwhm"].transform("median")
    table["reservedRms"] = table[["fitReservedXRms", "fitReservedYRms"]].max(axis=1)
    table["rmsRatio"] = table["reservedRms"] / table.groupby(group)["reservedRms"].transform("median")
    species = speciesTable(visits, lineMetrics)
    bySpecies = species.groupby(_DETECTOR).agg(
        minLinesRatio=("linesRatio", "min"), nSpecies=("species", "size")
    )
    table = table.merge(bySpecies, on=_DETECTOR, how="left")
    table["nSpecies"] = table["nSpecies"].fillna(0).astype(int)
    unassessed = (
        ~table["gateStatus"].isin(["PASS", "WARN", "FAIL"])
        & (table["nSpecies"] == 0)
        & table["medFwhm"].isna()
        & table["reservedRms"].isna()
    )
    table = table[~unassessed]
    columns = [
        *_DETECTOR,
        "detector",
        "iic_sequence_id",
        "lamp",
        "k",
        "n",
        "gateStatus",
        "gateReason",
        "lineStatus",
        "lineReason",
        "medFwhm",
        "fwhmRatio",
        "fitReservedXRms",
        "fitReservedYRms",
        "reservedRms",
        "rmsRatio",
        "minLinesRatio",
        "nSpecies",
    ]
    return table.reindex(columns=columns).sort_values(_DETECTOR).reset_index(drop=True)


def _gateText(reason: str) -> str:
    """``FWHM 3.06 px (fail >= 2.95)`` from the gate's ``medFWHM=3.06px >= fail threshold 2.951px``."""
    entries = parseReason(reason)
    if not entries:
        return reason
    parts = []
    for e in entries:
        unit = " px" if e["unit"] == "px" else ""
        parts.append(
            f"{_METRIC_NAMES.get(e['metric'], e['metric'])} {e['value']:.3g}{unit} ({e['level']} {e['op']} {e['threshold']:.3g})"
        )
    return ", ".join(parts)


def judge(detectors: pd.DataFrame, thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Add ``verdict`` (`Verdict`) and ``reason`` to a `detectorTable`.

    *out*: the gate or the line residuals FAILed the image, or its reserved RMS is beyond ``rmsOut``
    of the detector's sequence median, or its FWHM is beyond ``fwhmOut`` with the RMS also beyond
    ``rmsCheck`` (a detector gone wrong shows in both; FWHM alone can be the measurement, e.g. a
    faint lamp's few lines). *check*: a WARN, or FWHM or RMS beyond ``fwhmCheck`` / ``rmsCheck``.
    *weak*: a species below ``weakLines`` of the detector's best. Otherwise *use*. A sequence of
    one exposure has no median to compare with and gets no consistency verdict.
    """
    t = thresholds or Thresholds()
    table = detectors.copy()
    verdicts, reasons = [], []
    for row in table.itertuples(index=False):
        out, check, weak = [], [], []
        if row.gateStatus == "FAIL":
            out.append(f"gate FAIL: {_gateText(row.gateReason)}")
        elif row.gateStatus == "WARN":
            check.append(f"gate WARN: {_gateText(row.gateReason)}")
        if row.lineStatus == "FAIL":
            out.append(f"line residuals FAIL: {row.lineReason}")
        elif row.lineStatus == "WARN":
            check.append(f"line residuals WARN: {row.lineReason}")
        if row.n > 1:
            fwhmFar = pd.notna(row.fwhmRatio) and not (1 / t.fwhmOut <= row.fwhmRatio <= t.fwhmOut)
            fwhmOff = pd.notna(row.fwhmRatio) and not (1 / t.fwhmCheck <= row.fwhmRatio <= t.fwhmCheck)
            rmsFar = pd.notna(row.rmsRatio) and row.rmsRatio > t.rmsOut
            rmsOff = pd.notna(row.rmsRatio) and row.rmsRatio > t.rmsCheck
            fwhmText = f"FWHM {row.medFwhm:.2f} px ({row.fwhmRatio:.2f}x its sequence)"
            rmsText = f"reserved RMS {row.reservedRms:.2f} px ({row.rmsRatio:.1f}x its sequence)"
            if fwhmFar and rmsOff:
                out.append(fwhmText)
            elif fwhmOff:
                check.append(fwhmText)
            if rmsFar:
                out.append(rmsText)
            elif rmsOff:
                check.append(rmsText)
        if pd.notna(row.minLinesRatio) and row.minLinesRatio < t.weakLines:
            weak.append(f"lines {row.minLinesRatio:.2f} of the detector's best exposure")
        if out:
            verdicts.append(Verdict.OUT)
            reasons.append("; ".join(out + check))
        elif check:
            verdicts.append(Verdict.CHECK)
            reasons.append("; ".join(check))
        elif weak:
            verdicts.append(Verdict.WEAK)
            reasons.append("; ".join(weak))
        else:
            verdicts.append(Verdict.USE)
            reasons.append("")
    table["verdict"] = pd.Categorical(verdicts, categories=list(Verdict.ORDER), ordered=True)
    table["reason"] = reasons
    return table


def coverage(judged: pd.DataFrame, species: pd.DataFrame) -> pd.DataFrame:
    """How many exposures each detector keeps per species, after `judge`.

    Returns
    -------
    `pandas.DataFrame`
        ``detector``, ``species``, ``exposures`` (measured), ``usable`` (not *out*), ``weak``.
    """
    if species.empty:
        return pd.DataFrame(columns=["detector", "species", "exposures", "usable", "weak"])
    merged = species.merge(judged[[*_DETECTOR, "detector", "verdict"]], on=_DETECTOR)
    merged["usable"] = merged["verdict"] != Verdict.OUT
    merged["weakOne"] = merged["verdict"] == Verdict.WEAK
    result = merged.groupby(["detector", "species"]).agg(
        exposures=("visit", "size"), usable=("usable", "sum"), weak=("weakOne", "sum")
    )
    return result.reset_index()


def trends(judged: pd.DataFrame, species: pd.DataFrame) -> pd.DataFrame:
    """Return the lamps through their sequences: per arm, lamp, species and exposure, medians over detectors.

    Returns
    -------
    `pandas.DataFrame`
        ``arm``, ``iic_sequence_id``, ``lamp``, ``species``, ``k``, ``visit``, ``linesRatio``
        (median over the arm's detectors), ``nLines`` (median), ``medFwhm`` (median), ``detectors``.
    """
    if species.empty:
        return pd.DataFrame(
            columns=[
                "arm",
                "iic_sequence_id",
                "lamp",
                "species",
                "k",
                "visit",
                "linesRatio",
                "nLines",
                "medFwhm",
                "detectors",
            ]
        )
    merged = species.merge(judged[[*_DETECTOR, "lamp", "medFwhm"]], on=_DETECTOR)
    result = merged.groupby(["arm", "iic_sequence_id", "lamp", "species", "k", "visit"]).agg(
        linesRatio=("linesRatio", "median"),
        nLines=("nLines", "median"),
        medFwhm=("medFwhm", "median"),
        detectors=("spectrograph", "size"),
    )
    return result.reset_index().sort_values(["arm", "visit", "species"]).reset_index(drop=True)


def pick(judged: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Return the recommendation per detector: the set's visits it was measured on, less the *out* ones.

    Returns
    -------
    `pandas.DataFrame`
        ``detector``, ``arm``, ``spectrograph``, ``measured`` (visits), ``out`` (visits to leave
        out), ``check`` (visits to look at), ``weak`` (visits with fewer lines), ``missing``
        (set visits with no metrics on this detector).
    """
    rows = []
    allVisits = set(visits["visit"].astype(int))
    for (det, arm, sp), g in judged.groupby(["detector", "arm", "spectrograph"], sort=True):
        measured = sorted(g["visit"].astype(int))
        rows.append(
            {
                "detector": det,
                "arm": arm,
                "spectrograph": int(sp),
                "measured": measured,
                "out": sorted(g.loc[g["verdict"] == Verdict.OUT, "visit"].astype(int)),
                "check": sorted(g.loc[g["verdict"] == Verdict.CHECK, "visit"].astype(int)),
                "weak": sorted(g.loc[g["verdict"] == Verdict.WEAK, "visit"].astype(int)),
                "missing": sorted(allVisits - set(measured)),
            }
        )
    return pd.DataFrame(
        rows, columns=["detector", "arm", "spectrograph", "measured", "out", "check", "weak", "missing"]
    )


def goNoGo(judged: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Return the go/no-go of every detector on every set visit.

    ``go``, ``no-go`` (*out*), ``missing`` (the visit exposed the detector, by the listing's
    ``cameras``, but there are no metrics: not ingested, not reduced) or ``""`` (the visit never
    exposed the detector, e.g. r and n on a b+m sequence). Detectors down (b, r, m, n order), the
    set's visits across in time order. A listing without ``cameras`` counts every detector as exposed.
    A detector the listing names but no image was measured on is a row of ``missing``.
    """
    frame = judged.assign(
        go=judged["verdict"].astype(str).map(lambda v: "no-go" if v == Verdict.OUT else "go")
    )
    table = frame.pivot_table(index="detector", columns="visit", values="go", aggfunc="first")
    listed = {
        c.strip()
        for cameras in visits.get("cameras", pd.Series("", index=visits.index))
        for c in str(cameras or "").split(",")
        if c.strip()
    }
    order = sorted(
        set(table.index) | listed, key=lambda name: ("brmn".index(name[0]) if name[0] in "brmn" else 9, name)
    )
    table = table.reindex(index=order, columns=list(visits["visit"])).astype(object)
    for visit, cameras in zip(
        visits["visit"], visits.get("cameras", pd.Series("", index=visits.index)), strict=True
    ):
        exposed = {c.strip() for c in str(cameras or "").split(",") if c.strip()} or set(order)
        for det in order:
            if pd.isna(table.at[det, visit]):
                table.at[det, visit] = "missing" if det in exposed else ""
    return table


def _ranges(values: Iterable[int]) -> str:
    """``152652–152661, 152663`` for a sorted list of ints; ``-`` for none."""
    values = sorted({int(v) for v in values})
    if not values:
        return "-"
    parts, start, prev = [], values[0], values[0]
    for v in [*values[1:], None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        parts.append(f"{start}–{prev}" if prev != start else f"{start}")
        if v is not None:
            start = prev = v
    return ", ".join(parts)


def _hst(value) -> str:
    if value is None or pd.isna(value):
        return "?"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


#: The short name of each kind of reason `judge` gives, by the text that opens it.
_TAGS = (
    ("gate FAIL", "gate FAIL"),
    ("gate WARN", "gate WARN"),
    ("line residuals FAIL", "line residuals FAIL"),
    ("line residuals WARN", "line residuals WARN"),
    ("FWHM", "FWHM off"),
    ("reserved RMS", "reserved RMS off"),
    ("lines", "few lines"),
)


def _tags(reason: str) -> set[str]:
    """Return the kinds of reason in a `judge` reason string, as short tags."""
    return {tag for opening, tag in _TAGS if any(part.startswith(opening) for part in reason.split("; "))}


def summary(visits: pd.DataFrame, judged: pd.DataFrame, species: pd.DataFrame) -> list[str]:
    """Return the report's key points, one line each: what to leave out, what is weak, what is missing."""
    lines = []
    outs = judged[judged["verdict"] == Verdict.OUT]
    if outs.empty:
        lines.append("Nothing to leave out: every measured detector image is usable.")
    for det, g in outs.groupby("detector", sort=False):
        why = sorted({tag for reason in g["reason"] for tag in _tags(reason)})
        lines.append(f"**Leave out {det} on {_ranges(g['visit'])}**: " + ", ".join(why) + ".")
    cov = coverage(judged, species)
    none = cov[cov["usable"] == 0] if not cov.empty else cov
    for r in none.itertuples():
        lines.append(f"**{r.detector} has no usable {r.species} exposure.**")
    checks = judged[judged["verdict"] == Verdict.CHECK]
    if not checks.empty:
        lines.append(
            f"Look at {len(checks)} detector image(s) before deciding: "
            + "; ".join(f"{det} {_ranges(g['visit'])}" for det, g in checks.groupby("detector", sort=False))
            + "."
        )
    tr = trends(judged, species)
    tr = tr[tr["species"] != "Trace"]
    if not tr.empty:
        last = tr.sort_values("k").groupby(["arm", "iic_sequence_id", "species"]).last()
        first = tr.sort_values("k").groupby(["arm", "iic_sequence_id", "species"]).first()
        fading = last[(last["linesRatio"] < 0.9) & (first["linesRatio"] >= 0.95)]
        warming = first[(first["linesRatio"] < 0.9) & (last["linesRatio"] >= 0.95)]
        if not fading.empty:
            lines.append(
                "Fading through the sequence (last exposure's lines over the best): "
                + ", ".join(f"{sp} {r.linesRatio:.2f} on {arm}" for (arm, _seq, sp), r in fading.iterrows())
                + "."
            )
        if not warming.empty:
            lines.append(
                "Warming up (first exposure's lines over the best): "
                + ", ".join(f"{sp} {r.linesRatio:.2f} on {arm}" for (arm, _seq, sp), r in warming.iterrows())
                + "."
            )
    measured = set(judged["visit"].astype(int))
    unmeasured = visits[~visits["visit"].isin(measured)]
    if len(unmeasured):
        lines.append(
            "Not judged (no metrics: not in the Butler, or not reduced): "
            + ", ".join(f"{r.visit} ({r.lamp} {r.k}/{r.n})" for r in unmeasured.itertuples())
            + "."
        )
    return lines


def buildReport(
    visits: pd.DataFrame,
    judged: pd.DataFrame,
    species: pd.DataFrame,
    title: str,
    source: str = "",
    thresholds: Thresholds | None = None,
    now: datetime.datetime | None = None,
    figures: dict[str, str] | None = None,
    detail: bool = False,
    exports: bool = False,
    detectorFigures: dict[str, str] | None = None,
    imageFigures: dict[tuple[str, int], str] | None = None,
) -> str:
    """Return the Markdown report: the go/no-go per detector and visit, then the no-go detectors in detail.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        From `setVisits`.
    judged : `pandas.DataFrame`
        From `judge`.
    species : `pandas.DataFrame`
        From `speciesTable`.
    title : `str`
        The report's title, e.g. ``Run31 calibration set``.
    source : `str`, optional
        The collection the metrics came from (or the exports' collection), named in the header.
    thresholds : `Thresholds`, optional
        The thresholds `judge` used, stated in the footer; default `Thresholds()`.
    now : `datetime.datetime`, optional
        The report's time (HST); default now.
    figures : `dict` [`str`, `str`], optional
        Figure files to embed, by caption, as paths relative to the report (see
        `pfs.drp.qa.plotting.calibInputs`); the go/no-go grid first.
    detail : `bool`, optional
        Add the reasons for every no-go and check, the weak exposures and the lamp trends.
    exports : `bool`, optional
        Whether ``source`` was read from `exportMetrics.py` exports rather than the Butler.
    detectorFigures : `dict` [`str`, `str`], optional
        A figure per no-go detector (`pfs.drp.qa.plotting.calibInputs.plotDetectorDetail`), by
        detector name, as paths relative to the report: the closing section.
    imageFigures : `dict` [(`str`, `int`), `str`], optional
        A line-residual figure per no-go image (`pfs.drp.qa.plotting.calibInputs.plotLineResiduals`),
        by (detector, visit), shown under its detector's figure; the driver makes one only where line
        residuals or the reserved RMS are part of the reason.
    """
    t = thresholds or Thresholds()
    now = now or datetime.datetime.now()
    first, last = visits["visit"].min(), visits["visit"].max()
    out = [f"# {title}: go/no-go", ""]
    out += [
        "| | |",
        "|---|---|",
        f"| Set | {first}–{last}, {len(visits)} visits in {visits['iic_sequence_id'].nunique()} sequences |",
        f"| Taken | {_hst(visits['time_exp_start'].min())}–{_hst(visits['time_exp_start'].max())} HST |",
        f"| Judged | {now:%Y-%m-%d %H:%M} HST |",
        f"| From | {'exports of ' if exports else ''}`{source or 'comparison mode'}` |",
        f"| Detector images | {len(judged)} |",
    ]
    out.append("")
    outs = judged[judged["verdict"] == Verdict.OUT]
    if outs.empty:
        out.append("- **Every measured detector image is go.**")
    else:
        out.append("- **No-go:**")
        order = sorted(
            outs["detector"].unique(), key=lambda d: ("brmn".index(d[0]) if d[0] in "brmn" else 9, d)
        )
        for det in order:
            out.append(f"  - {det}: {_ranges(outs.loc[outs['detector'] == det, 'visit'])}")
    cov = coverage(judged, species)
    for r in (cov[cov["usable"] == 0] if not cov.empty else cov).itertuples():
        out.append(f"- **{r.detector} has no go {r.species} exposure.**")
    table = goNoGo(judged, visits)
    missing = table.columns[(table == "missing").any(axis=0)]
    if len(missing):
        out.append(f"- Not measured (exposed, but no data in the Butler): {_ranges(missing)}.")
    out.append("")
    figures = dict(figures or {})
    for caption, path in list(figures.items())[:1]:
        out += [f"![{caption}]({path})", ""]
    out += ["## Per detector", "", "| detector | go | no-go |", "|---|---|---|"]
    for det, row in table.iterrows():
        out.append(f"| {det} | {_ranges(row.index[row == 'go'])} | {_ranges(row.index[row == 'no-go'])} |")
    out.append("")
    if detail:
        out += ["## Reasons in full", ""]
        for r in (
            judged[judged["verdict"].isin([Verdict.OUT, Verdict.CHECK])]
            .sort_values(["verdict", "detector", "visit"])
            .itertuples()
        ):
            mark = "**" if r.verdict == Verdict.OUT else ""
            out.append(
                f"- {mark}{r.detector} {r.visit}{mark} ({r.lamp} {r.k}/{r.n}, {r.verdict}): {r.reason}"
            )
        out.append("")
        out += [f"- {line}" for line in summary(visits, judged, species)[1:]]
        out.append("")
        for caption, path in list(figures.items())[1:]:
            out += [f"![{caption}]({path})", ""]
    if detectorFigures:
        out += ["## No-go detectors in detail", ""]
        for det, path in detectorFigures.items():
            rows = judged[judged["detector"] == det].sort_values("visit")
            out += [
                f"### {det}",
                "",
                "| visit | lamp | verdict | FWHM px | fit reserved RMS px | why |",
                "|---|---|---|---|---|---|",
            ]
            for r in rows[rows["verdict"].isin([Verdict.OUT, Verdict.CHECK])].itertuples():
                out.append(
                    f"| {r.visit} | {r.lamp} {r.k}/{r.n} | {r.verdict} | {r.medFwhm:.2f} | {r.reservedRms:.3f} | {r.reason} |"
                )
            out += ["", f"![{det} through the set]({path})", ""]
            for (d, visit), lines in (imageFigures or {}).items():
                if d == det:
                    out += [f"![{det} {visit} line residuals]({lines})", ""]
    out.append(
        f"*No-go: a gate or line-residual FAIL, reserved-line RMS beyond {t.rmsOut}x the detector's median over "
        f"its sequence, or FWHM beyond {t.fwhmOut}x with the RMS also off; judged with no calibration built from the set. Per-image numbers and reasons are in the CSV beside this report.*"
    )
    out.append("")
    return "\n".join(out)
