"""Judge a calibration set's exposures per detector, for whoever builds the calibrations.

A run's calibration set is a few sequences of arc and quartz exposures (three per arc lamp, ten
traces), taken within an hour. ``detectorMap_calib`` and the fiber profiles are built per detector
from them, so an exposure is judged per detector image, never per arm: a detector that fails on
one exposure is left out of that exposure's inputs and nothing else is.

Each detector image is judged on what comparison mode measured on it (`pfs.drp.qa.imageQualityQa`
and `pfs.drp.qa.dmLineResiduals`, read from their collection). The tasks store numbers, not
verdicts, so the image is judged here, three ways:

* the gate (`pfs.drp.qa.metrics.gate`), on the image's numbers: its width, lines per lit fiber and
  held-back line scatter against the camera's normal values. It is given nothing measured against
  the calibration in use, which the set is taken to replace, so the offset from it is not judged.
  Every check left is relative to a normal value: without the table of them the gate gives no
  verdict (``UNKNOWN``, never a pass), and the report says so;
* consistency within the sequence: FWHM and reserved-line RMS relative to the detector's median
  over the sequence, which catches a detector that goes wrong mid-sequence without knowing what
  "right" looks like;
* the lamp: each species' line count relative to the detector's best exposure of the sequence, which
  shows warm-up and fading and marks weak exposures, usable but with fewer lines.

The last two need no reference but the other exposures of the same sequence on the same detector
(`Thresholds`); they choose exposures and are no part of the gate.

Stack-free: DataFrames in, a Markdown report out. ``drpQa.py calib inputs`` reads the tables
from a Butler or from parquet exports (`pfs.drp.qa.metrics.readers.exportMetrics`) and the visit
listing from comparison mode's ``fetch``.
"""

import datetime
import re
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from pfs.drp.qa.metrics import gate
from pfs.drp.qa.metrics.registry import ARC, CHECKS, FIXED, UNJUDGED_SPECIES, specFor

__all__ = [
    "LAMP_SPECIES",
    "Rule",
    "Thresholds",
    "Verdict",
    "buildReport",
    "coverage",
    "detectorTable",
    "gateNumbers",
    "gateVerdicts",
    "goNoGo",
    "judge",
    "lampOf",
    "lampsOf",
    "pick",
    "scatterLimits",
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


class Rule:
    """The rules `judge` applies, as its ``rules`` column names them."""

    GATE_FAIL = "gate FAIL"
    GATE_WARN = "gate WARN"
    FWHM = "FWHM off"
    RMS = "reserved RMS off"
    LINES = "few lines"


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


def _checkLimits(check: str, metric: str, normal: float | None) -> tuple[float, float]:
    """Return a check's WARN and FAIL in the metric's units; NaN for a relative check without a normal value."""
    spec = CHECKS[check]
    if spec.kind == FIXED:
        return spec.limits()
    if normal is None or pd.isna(normal):
        return float("nan"), float("nan")
    return spec.limits(float(normal), specFor(metric).higherIsWorse)


def _gateTables(
    iqMetrics: pd.DataFrame, lineMetrics: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return the tables the gate is given here: nothing measured against the calibration in use.

    The set is taken to replace that calibration, so how far the instrument has moved from it is
    not the image's doing. That leaves out the trace centre offset of ``iqQaMetrics`` and the line
    residuals against ``detectorMap_calib``, and with them the gate's ``traceCentreOffset`` and
    ``calibOffset``.
    """
    iq = iqMetrics.drop(columns="medDxCenter", errors="ignore")
    if lineMetrics is None or lineMetrics.empty:
        return iq, None
    return iq, lineMetrics[lineMetrics["reference"] == "visit"]


def gateNumbers(
    iqMetrics: pd.DataFrame,
    lineMetrics: pd.DataFrame | None = None,
    normalValues: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the gate's verdict on each number of the set's images, with its limits.

    `pfs.drp.qa.metrics.gate.judge` of `pfs.drp.qa.metrics.gate.measurements`, on the width, the
    lines per lit fiber and the line residuals against the visit's own detectorMap. Nothing
    measured against the calibration in use is judged (the gate's ``calibOffset`` and
    ``traceCentreOffset``): the set is taken to replace it.

    Parameters
    ----------
    iqMetrics : `pandas.DataFrame`
        ``iqQaMetrics`` rows of the set's images.
    lineMetrics : `pandas.DataFrame`, optional
        ``dmLineResidualMetrics`` rows of the set's images.
    normalValues : `pandas.DataFrame`, optional
        The normal values (`pfs.drp.qa.metrics.normalValues.NORMAL_COLUMNS`). Without them a
        relative check gives no verdict.

    Returns
    -------
    `pandas.DataFrame`
        One row per image, check and number, as `pfs.drp.qa.metrics.gate.judge` returns them
        (``value``, ``normal``, ``excess``, ``warn``, ``fail``, ``status``, ``reason``), with
        ``warnLimit`` and ``failLimit``: the limits in the number's own units
        (`pfs.drp.qa.metrics.registry.Check.limits`), on its absolute value for an offset; NaN
        where the gate gave no verdict.
    """
    iq, lines = _gateTables(iqMetrics, lineMetrics)
    numbers = gate.judge(gate.measurements(iq=iq, lines=lines), normalValues)
    limits = [
        _checkLimits(check, metric, normal) if status else (float("nan"), float("nan"))
        for check, metric, normal, status in zip(
            numbers["check"], numbers["metric"], numbers["normal"], numbers["status"], strict=True
        )
    ]
    numbers["warnLimit"] = [warn for warn, _ in limits]
    numbers["failLimit"] = [fail for _, fail in limits]
    return numbers


def gateVerdicts(
    iqMetrics: pd.DataFrame,
    lineMetrics: pd.DataFrame | None = None,
    normalValues: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the gate's verdict on each of the set's images.

    `pfs.drp.qa.metrics.gate.gate`, on the tables `gateNumbers` judges.

    Parameters
    ----------
    iqMetrics, lineMetrics, normalValues
        As for `gateNumbers`.

    Returns
    -------
    `pandas.DataFrame`
        One row per image: ``visit``, ``arm``, ``spectrograph``, ``gateStatus`` (``PASS``,
        ``WARN``, ``FAIL``, or ``UNKNOWN`` when no check was judged), ``gateDecidedBy`` (the check
        with that verdict), ``gateReason`` (why, or for ``UNKNOWN`` why nothing was judged),
        ``gateChecks`` (every check at ``WARN`` or ``FAIL``, comma-separated) and ``gateUnjudged``
        (how many of the image's numbers the gate gave no verdict on, e.g. for want of a normal
        value).
    """
    iq, lines = _gateTables(iqMetrics, lineMetrics)
    images = gate.gate(iq=iq, lines=lines, normalValues=normalValues)
    images = images.rename(
        columns={"qaStatus": "gateStatus", "qaDecidedBy": "gateDecidedBy", "qaReason": "gateReason"}
    ).drop(columns="frameType")
    numbers = gateNumbers(iqMetrics, lineMetrics, normalValues)
    crossed = numbers[numbers["status"].isin(["WARN", "FAIL"])]
    checks = crossed.groupby(_DETECTOR)["check"].agg(lambda names: ",".join(dict.fromkeys(names)))
    unjudged = numbers[numbers["status"] == ""].groupby(_DETECTOR).size()
    images = images.join(checks.rename("gateChecks"), on=_DETECTOR).join(
        unjudged.rename("gateUnjudged"), on=_DETECTOR
    )
    images["gateChecks"] = images["gateChecks"].fillna("")
    images["gateUnjudged"] = images["gateUnjudged"].fillna(0).astype(int)
    return images


#: The gate's line-scatter checks, by the metric each reads.
_SCATTER_CHECKS = {"robustRmsDeltaX": CHECKS["scatterX"], "robustRmsDeltaY": CHECKS["scatterY"]}


def scatterLimits(
    arm: str,
    spectrograph: int,
    species: Iterable[str],
    visit: int,
    normalValues: pd.DataFrame | None = None,
    frameType: str = ARC,
) -> dict[str, dict[str, tuple[float | None, float | None]]]:
    """Return the gate's limits on one image's held-back line scatter, in pixels, per species.

    The scatter checks (``scatterX`` and ``scatterY`` of `pfs.drp.qa.metrics.registry.CHECKS`) are
    relative: WARN and FAIL are a percentage above the camera's normal value for the species. The
    normal value is the one in force at ``visit``, looked up as the gate does
    (`pfs.drp.qa.metrics.gate.judge`), so there is no limit without a table of them.

    Parameters
    ----------
    arm : `str`
        The image's arm.
    spectrograph : `int`
        The image's spectrograph.
    species : iterable of `str`
        The line species to return limits for; ``Trace`` is left out.
    visit : `int`
        The image's visit.
    normalValues : `pandas.DataFrame`, optional
        The normal values (`pfs.drp.qa.metrics.normalValues.NORMAL_COLUMNS`).
    frameType : `str`, optional
        The image's frame type (`pfs.drp.qa.metrics.registry.FRAME_TYPES`); a check that doesn't
        apply to it gives no limit.

    Returns
    -------
    `dict`
        ``{species: {"robustRmsDeltaX": (warn, fail), "robustRmsDeltaY": (warn, fail)}}``; both
        limits are None where the gate judges nothing: no normal value in force, a species whose
        scatter isn't judged (`pfs.drp.qa.metrics.registry.UNJUDGED_SPECIES`) or a frame type the
        check doesn't cover. Empty without ``normalValues``.
    """
    species = [name for name in species if name != "Trace"]
    if normalValues is None or normalValues.empty or not species:
        return {}
    one = pd.Series([arm]), pd.Series([int(spectrograph)])
    rows = pd.DataFrame(
        [
            {
                "visit": int(visit),
                "arm": arm,
                "spectrograph": int(spectrograph),
                "frameType": frameType,
                "check": check.name,
                "metric": metric,
                "part": name,
                "camera": gate.normalKey(check, *one).iloc[0],
                "selector": name,
                "value": 0.0,
                "withM": False,
            }
            for name in species
            for metric, check in _SCATTER_CHECKS.items()
        ],
        columns=list(gate.MEASUREMENT_COLUMNS),
    )
    out: dict[str, dict[str, tuple[float | None, float | None]]] = {name: {} for name in species}
    for row in gate.judge(rows, normalValues).itertuples():
        check = _SCATTER_CHECKS[row.metric]
        judged = frameType in check.frames and row.part not in UNJUDGED_SPECIES and pd.notna(row.normal)
        out[row.part][row.metric] = (
            _checkLimits(check.name, row.metric, row.normal) if judged else (None, None)
        )
    return out


#: ``reductionStats`` metric names of the detectorMap fit's reserved-line RMS, and the
#: ``detectorTable`` columns they become (named for the ``iqQaMetrics`` columns they replace).
_RESERVED_RMS = {"reservedXRms": "fitReservedXRms", "reservedYRms": "fitReservedYRms"}


def _reservedRms(reductionStats: pd.DataFrame) -> pd.DataFrame:
    """Return per-detector ``fitReservedXRms``/``fitReservedYRms`` from the reductionStats table.

    The reserved-line RMS is a whole-detector statistic (empty ``amplifier``). An image whose
    adjustment did not apply recorded none, so it has no row and its RMS merges as null.
    """
    # Typed so the reserved columns stay float when there are no rows: merging an object-dtype
    # empty frame leaves them object, and the later median/max then raise.
    empty = pd.DataFrame(
        {
            "visit": pd.Series(dtype="int64"),
            "arm": pd.Series(dtype="object"),
            "spectrograph": pd.Series(dtype="int64"),
            **{column: pd.Series(dtype="float64") for column in _RESERVED_RMS.values()},
        }
    )
    if reductionStats is None or reductionStats.empty:
        return empty
    rows = reductionStats[
        (reductionStats["amplifier"].fillna("") == "") & reductionStats["metric"].isin(list(_RESERVED_RMS))
    ]
    if rows.empty:
        return empty
    wide = (
        rows.pivot_table(index=_DETECTOR, columns="metric", values="value", aggfunc="first")
        .reindex(columns=list(_RESERVED_RMS))
        .rename(columns=_RESERVED_RMS)
        .reset_index()
    )
    return wide


def detectorTable(
    visits: pd.DataFrame,
    iqMetrics: pd.DataFrame,
    lineMetrics: pd.DataFrame,
    reductionStats: pd.DataFrame,
    normalValues: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per detector image of the set, with the gate's verdict and its place in the sequence.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        From `setVisits`.
    iqMetrics : `pandas.DataFrame`
        ``iqQaMetrics`` rows of the set's images: ``medFwhm`` and ``traceOnly`` are used here, and
        what the gate reads (`pfs.drp.qa.metrics.gate.measurements`). A ``traceOnly`` image's FWHM
        is read from the fiberProfiles calibration, not measured on the image, so it takes no part
        in the sequence comparison. An image the gate judged nothing on (``UNKNOWN``) and that has
        no width, fit RMS or line species measured is left out: it is unassessed, and `goNoGo`
        reports it as missing rather than go.
    lineMetrics : `pandas.DataFrame`
        ``dmLineResidualMetrics`` rows of the set's images (`speciesTable`).
    reductionStats : `pandas.DataFrame`
        ``reductionStats`` rows of the set's images, long format (amplifier, metric, value). The
        detectorMap fit's reserved-line RMS (``reservedXRms``, ``reservedYRms``, whole-detector)
        is taken from here, as ``fitReservedXRms``/``fitReservedYRms``; it came from ``iqQaMetrics``
        before the reduction statistics moved to their own table. An image whose adjustment did not
        apply has no reserved RMS, so its ``reservedRms`` is null, as before.
    normalValues : `pandas.DataFrame`, optional
        The normal values the gate judges by
        (`pfs.drp.qa.metrics.normalValues.NORMAL_COLUMNS`). Without them the gate gives no
        verdict on a calibration frame, and the image is judged on the sequence rules and the
        lamp alone (`judge`).

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph``, ``detector`` (``n4``), ``iic_sequence_id``, ``lamp``,
        ``k``, ``n``, ``gateStatus``, ``gateDecidedBy``, ``gateReason``, ``gateChecks``,
        ``gateUnjudged`` (`gateVerdicts`), ``medFwhm``, ``fwhmRatio`` (to the detector's sequence
        median), ``fitReservedXRms``, ``fitReservedYRms``, ``reservedRms`` (the larger of x and
        y), ``rmsRatio`` (likewise), ``minLinesRatio`` (the lamp's weakest species,
        `speciesTable`), ``nSpecies`` (species measured).
    """
    keep = ["visit", "arm", "spectrograph", "medFwhm", "traceOnly"]
    iq = iqMetrics.reindex(columns=keep)
    iq.loc[iq["traceOnly"].fillna(False).astype(bool), "medFwhm"] = float("nan")
    iq = iq.drop(columns="traceOnly")
    table = iq.merge(visits[["visit", "iic_sequence_id", "lamp", "k", "n"]], on="visit")
    table = table.merge(_reservedRms(reductionStats), on=_DETECTOR, how="left")
    table = table.merge(gateVerdicts(iqMetrics, lineMetrics, normalValues), on=_DETECTOR, how="left")
    table["gateStatus"] = table["gateStatus"].fillna("UNKNOWN")
    for column in ("gateDecidedBy", "gateReason", "gateChecks"):
        table[column] = table[column].fillna("")
    table["gateUnjudged"] = table["gateUnjudged"].fillna(0).astype(int)
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
        "gateDecidedBy",
        "gateReason",
        "gateChecks",
        "gateUnjudged",
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


def judge(detectors: pd.DataFrame, thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Add ``verdict`` (`Verdict`), ``reason`` and ``rules`` to a `detectorTable`.

    *out*: the gate FAILed the image, or its reserved RMS is beyond ``rmsOut`` of the detector's
    sequence median, or its FWHM is beyond ``fwhmOut`` with the RMS also beyond ``rmsCheck`` (a
    detector gone wrong shows in both; FWHM alone can be the measurement, e.g. a faint lamp's few
    lines). *check*: a gate WARN, or FWHM or RMS beyond ``fwhmCheck`` / ``rmsCheck``. *weak*: a
    species below ``weakLines`` of the detector's best. Otherwise *use*. A sequence of one
    exposure has no median to compare with and gets no consistency verdict. An image the gate gave
    no verdict on (``UNKNOWN``) is judged on the sequence and the lamp alone: that is not a gate
    pass, and `buildReport` counts those images.

    ``reason`` is the text of every rule behind the verdict, joined by ``"; "``; ``rules`` names
    them (`Rule`), joined by ``", "``.
    """
    t = thresholds or Thresholds()
    table = detectors.copy()
    verdicts, reasons, rules = [], [], []
    for row in table.itertuples(index=False):
        # Each entry is (rule, text).
        out, check, weak = [], [], []
        if row.gateStatus == "FAIL":
            out.append((Rule.GATE_FAIL, f"gate FAIL: {row.gateReason}"))
        elif row.gateStatus == "WARN":
            check.append((Rule.GATE_WARN, f"gate WARN: {row.gateReason}"))
        if row.n > 1:
            fwhmFar = pd.notna(row.fwhmRatio) and not (1 / t.fwhmOut <= row.fwhmRatio <= t.fwhmOut)
            fwhmOff = pd.notna(row.fwhmRatio) and not (1 / t.fwhmCheck <= row.fwhmRatio <= t.fwhmCheck)
            rmsFar = pd.notna(row.rmsRatio) and row.rmsRatio > t.rmsOut
            rmsOff = pd.notna(row.rmsRatio) and row.rmsRatio > t.rmsCheck
            fwhmText = f"FWHM {row.medFwhm:.2f} px ({row.fwhmRatio:.2f}x its sequence)"
            rmsText = f"reserved RMS {row.reservedRms:.2f} px ({row.rmsRatio:.1f}x its sequence)"
            if fwhmFar and rmsOff:
                out.append((Rule.FWHM, fwhmText))
            elif fwhmOff:
                check.append((Rule.FWHM, fwhmText))
            if rmsFar:
                out.append((Rule.RMS, rmsText))
            elif rmsOff:
                check.append((Rule.RMS, rmsText))
        if pd.notna(row.minLinesRatio) and row.minLinesRatio < t.weakLines:
            weak.append((Rule.LINES, f"lines {row.minLinesRatio:.2f} of the detector's best exposure"))
        if out:
            verdict, why = Verdict.OUT, out + check
        elif check:
            verdict, why = Verdict.CHECK, check
        elif weak:
            verdict, why = Verdict.WEAK, weak
        else:
            verdict, why = Verdict.USE, []
        verdicts.append(verdict)
        reasons.append("; ".join(text for _, text in why))
        rules.append(", ".join(rule for rule, _ in why))
    table["verdict"] = pd.Categorical(verdicts, categories=list(Verdict.ORDER), ordered=True)
    table["reason"] = reasons
    table["rules"] = rules
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


def summary(visits: pd.DataFrame, judged: pd.DataFrame, species: pd.DataFrame) -> list[str]:
    """Return the report's key points, one line each: what to leave out, what is weak, what is missing."""
    lines = []
    outs = judged[judged["verdict"] == Verdict.OUT]
    if outs.empty:
        lines.append("Nothing to leave out: every measured detector image is usable.")
    for det, g in outs.groupby("detector", sort=False):
        why = sorted({rule for rules in g["rules"] for rule in rules.split(", ") if rule})
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


_JUDGED = ("PASS", "WARN", "FAIL")


def _gateNotes(judged: pd.DataFrame) -> list[str]:
    """Return the report's lines on what the gate gave no verdict on; none when it judged everything."""
    if judged.empty:
        return []
    hasVerdict = judged["gateStatus"].isin(_JUDGED)
    unknown = judged[~hasVerdict]
    notes = []
    if len(unknown):
        why = (
            "; ".join(dict.fromkeys(reason for reason in unknown["gateReason"] if reason)) or "nothing judged"
        )
        if hasVerdict.any():
            notes.append(
                f"- The gate gave no verdict on {len(unknown)} of {len(judged)} detector images ({why}): "
                "those are judged on the sequence rules alone, which is not a gate pass."
            )
        else:
            notes.append(
                f"- **The gate gave no verdict on any detector image** ({why}): the go/no-go below is by "
                "the sequence rules alone, which is not a gate pass."
            )
    partial = int((hasVerdict & (judged["gateUnjudged"] > 0)).sum())
    if partial:
        notes.append(
            f"- On {partial} detector image(s) with a gate verdict, the gate left some numbers without one "
            "(`gateUnjudged` in the CSV)."
        )
    return notes


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
    normalValues: str = "",
) -> str:
    """Return the Markdown report: the go/no-go per detector and visit, then the no-go detectors in detail.

    The report says how many detector images the gate gave no verdict on, and why (no normal
    value): those are go or no-go by the sequence rules alone.

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
        Whether ``source`` was read from parquet exports rather than the Butler.
    detectorFigures : `dict` [`str`, `str`], optional
        A figure per no-go detector (`pfs.drp.qa.plotting.calibInputs.plotDetectorDetail`), by
        detector name, as paths relative to the report: the closing section.
    imageFigures : `dict` [(`str`, `int`), `str`], optional
        A line-residual figure per no-go image (`pfs.drp.qa.plotting.calibInputs.plotLineResiduals`),
        by (detector, visit), shown under its detector's figure; the driver makes one only where the
        line scatter or the reserved RMS is part of the reason.
    normalValues : `str`, optional
        The table of normal values the gate judged by, named in the header.
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
    if normalValues:
        out.append(f"| Normal values | `{normalValues}` |")
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
    out += _gateNotes(judged)
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
        "*No-go: a gate FAIL (width, lines per lit fiber and held-back line scatter, each against the camera's "
        f"normal value), reserved-line RMS beyond {t.rmsOut}x the detector's median over its sequence, or FWHM "
        f"beyond {t.fwhmOut}x with the RMS also off; judged with no calibration built from the set, and not on "
        "the offset from the calibration in use. Per-image numbers and reasons are in the CSV beside this report.*"
    )
    out.append("")
    return "\n".join(out)
