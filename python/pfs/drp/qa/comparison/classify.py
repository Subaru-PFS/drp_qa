"""What each visit is, and whether comparison mode measures it.

Every visit is a calibration or science, by what the exposure is rather than when it was taken:
every sequence type but ``scienceObject`` and ``scienceObject_windowed`` is a calibration, and
those two are science when their design was made for a science proposal. Twilight frames,
dithers and the fields of telescope focus sweeps are on engineering designs, so they are
calibrations.

The QA tasks measure every exposure they can. The sequence types in `VALIDATED_TYPES`, the arcs
and traces of the calibration recipe, are *gated*: their passes are planned and run first. The
others (science frames, engineering arcs, flats, fiber profiles, focus sweeps, windowed readouts,
any new type) are *unvalidated*: measured the same way so their results can be seen, but often
off-nominal on purpose, so a result out of the ordinary there may be what the test expected. So are
the engineering sequences taken as those types (`ENGINEERING_SEQUENCES`, by sequence name): a bootstrap, an IIS
``ImageQuality`` frame or a health check lights a handful of fibers, and its trace centroids scatter
twice a trace set's; a ``LineCheck`` or ``DetectorMap test`` arc is taken off the calibration recipe,
and its lamp lines scatter like a fault's. `NOT_MEASURED_TYPES` have
nothing to measure. Every visit is listed, so a run's coverage is complete and a new sequence
type shows up rather than disappearing.

Arcs and traces are taken two ways: as *sets* (several exposures of a lamp: 3 per arc lamp and 10
traces for the calibrations) and, from Run28, as *daily* single exposures of one arc and one trace,
for drift. ``cadence`` tells them apart; they are reported apart. A set can include one-visit
sequences, so a one-visit sequence in the same sequence group (``group_id``) as a longer one of its
type belongs to the set; one whose name recurs on `DAILY_MIN_NIGHTS` nights is daily; any other is a one-off
``single`` (a positioner scan, a bootstrap exposure), measured like a set.

Sky visits also get sub-labels from the telescope status: `focusSweep` (a telescope focus sweep,
which changes how much light enters the fibers but not the spectrograph's line widths) and
``dithered``.

An arc or trace gets the `shaka` sub-label from its design name (`shakaOf`): a *shaka* exposure
lights one group of every few cobras (``cobraHome-MOD4_group2``, every fourth cobra), so it is
measured and judged like a fully lit arc or trace but on about a quarter of the fibers, and is
reported apart from them. ``shakaMod`` is the modulus (4 for every fourth) and ``shakaGroup`` the
group; a fully lit (``cobraHome``) or science design has neither.
"""

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from pfs.drp.qa.comparison.runs import Period, nightOf, periodOf

__all__ = [
    "CADENCE_TYPES",
    "DAILY_MIN_NIGHTS",
    "ENGINEERING_CATEGORIES",
    "ENGINEERING_GROUPS",
    "ENGINEERING_SEQUENCES",
    "FOCUS_SWEEP_MIN_RANGE",
    "FOCUS_SWEEP_MIN_STEPS",
    "NOT_MEASURED_TYPES",
    "SKY_TYPES",
    "VALIDATED_TYPES",
    "cadences",
    "classifyVisits",
    "designKinds",
    "engineeringGroups",
    "engineeringSequences",
    "focusSweeps",
    "shakaColumn",
    "shakaLabel",
    "shakaOf",
]

#: Sequence types of the calibration recipe, whose passes come first: the *gated* visits.
VALIDATED_TYPES = ("scienceArc", "scienceTrace")

#: Starts of sequence group names (case-insensitive) whose arcs and traces are engineering exposures.
ENGINEERING_GROUPS = ("fiberprofiles",)

#: Starts of sequence names (case-insensitive) that are engineering sequences whatever their type:
#: bootstraps, IIS image-quality frames and health checks light a handful of fibers, line checks and
#: detectorMap tests take lamps and exposures off the calibration recipe. Unvalidated.
ENGINEERING_SEQUENCES = (
    "bootstrap",
    "imagequality",
    "image quality",
    "health check",
    "linecheck",
    "detectormap test",
)

#: Sequence types with no arc lines or traces to measure.
NOT_MEASURED_TYPES = ("biases", "darks", "masterBiases", "masterDarks")

#: Sequence types taken on sky, whose design says whether they are science.
SKY_TYPES = ("scienceObject", "scienceObject_windowed")

#: Sequence types taken as a set of exposures or as a daily single one.
CADENCE_TYPES = ("scienceArc", "scienceTrace")

#: A one-visit arc or trace is daily when its sequence name recurs as one on this many nights.
DAILY_MIN_NIGHTS = 4

#: Proposal categories that don't make a design science: engineering (``EN``), a fiber with no
#: proposal, or an ID that doesn't follow Subaru's pattern.
ENGINEERING_CATEGORIES = frozenset({"EN", "none", "other"})  # not "uncoded": S26B-047 is a science program

#: A telescope focus sweep: at least this many distinct focus offsets on one field in one night...
FOCUS_SWEEP_MIN_STEPS = 10
#: ...spanning at least this much (mm). The focus correction during a night spans about 0.1 mm.
FOCUS_SWEEP_MIN_RANGE = 0.5

#: A shaka design name's modulus (``MOD4``) and group (``group2``), case-insensitive: ``MOD<k>`` names
#: every ``k``-th cobra and ``group<n>`` the group of them that is lit.
_MOD_RE = re.compile(r"MOD(\d+)", re.IGNORECASE)
_GROUP_RE = re.compile(r"group(\d+)", re.IGNORECASE)


def designKinds(categories: pd.DataFrame) -> pd.Series:
    """Return whether each design was made for science or engineering.

    Parameters
    ----------
    categories : `pandas.DataFrame`
        ``pfs_design_id`` and ``category``, from
        `pfs.drp.qa.comparison.queries.readDesignCategories`.

    Returns
    -------
    `pandas.Series`
        ``science`` or ``engineering``, indexed by ``pfs_design_id``. A design
        is science when any of its science fibers belongs to a proposal outside
        `ENGINEERING_CATEGORIES`, so a design shared by an engineering and a
        science proposal is science. Designs with no science fibers are absent:
        `classifyVisits` treats them as engineering.
    """
    if categories.empty:
        return pd.Series(dtype=object, name="design_kind")
    isScience = ~categories["category"].isin(ENGINEERING_CATEGORIES)
    kinds = isScience.groupby(categories["pfs_design_id"]).any()
    return kinds.map({True: "science", False: "engineering"}).rename("design_kind")


def shakaOf(name: object) -> tuple[int | None, int | None]:
    """Return the modulus and group of a one-fiber-group (shaka) design name.

    A *shaka* design lights one group of every few cobras; its name has a
    ``MOD<k>`` before ``group<n>`` (``cobraHome-MOD4_group2`` lights every
    fourth cobra, group 2). The modulus is the ``MOD<k>`` nearest before the
    group, so ``cobraHome-MOD32_before_MOD4_group1`` reads as modulus 4, group
    1.

    Parameters
    ----------
    name : object
        A design name (``pfs_design.design_name``), or anything else.

    Returns
    -------
    `tuple` [`int` or `None`, `int` or `None`]
        ``(modulus, group)`` for a shaka design; ``(None, None)`` for a fully
        lit (``cobraHome``) or science design, or a non-string.
    """
    if not isinstance(name, str):
        return (None, None)
    group = _GROUP_RE.search(name)
    if group is None:
        return (None, None)
    mods = [mod for mod in _MOD_RE.finditer(name) if mod.start() < group.start()]
    if not mods:
        return (None, None)
    return int(mods[-1].group(1)), int(group.group(1))


def shakaLabel(mod: object, group: object) -> str:
    """Return a shaka exposure's label (``shaka MOD4 group2``), or ``""`` when it isn't one.

    Parameters
    ----------
    mod, group : object
        A visit's ``shakaMod`` and ``shakaGroup`` (`classifyVisits`); NA when it
        is not a shaka exposure.

    Returns
    -------
    `str`
        ``shaka MOD<k> group<n>``, or ``""`` when ``group`` is NA.
    """
    if pd.isna(group):
        return ""
    return f"shaka MOD{int(mod)} group{int(group)}"


def shakaColumn(visits: pd.DataFrame) -> list[str]:
    """Return each visit's shaka label (`shakaLabel`), from its ``shakaMod`` and ``shakaGroup``.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        With ``shakaMod`` and ``shakaGroup`` columns (`classifyVisits`).

    Returns
    -------
    `list` [`str`]
        One label per row (``""`` for a fully lit exposure), in row order.
    """
    return [
        shakaLabel(mod, group) for mod, group in zip(visits["shakaMod"], visits["shakaGroup"], strict=True)
    ]


def focusSweeps(sky: pd.DataFrame) -> pd.Series:
    """Return which sky visits belong to a telescope focus sweep.

    A sweep steps the hexapod focus offset between short exposures of one
    field: at least `FOCUS_SWEEP_MIN_STEPS` distinct offsets spanning
    `FOCUS_SWEEP_MIN_RANGE` mm, within one night and sequence name. Visits with
    no sequence name are never a sweep.

    Parameters
    ----------
    sky : `pandas.DataFrame`
        ``night``, ``sequence_name`` and ``focus_offset_max`` (mm; NaN without
        telescope status).

    Returns
    -------
    `pandas.Series`
        `bool`, aligned with ``sky``.
    """
    if sky.empty:
        return pd.Series(dtype=bool, index=sky.index)
    focus = sky["focus_offset_max"].round(3)
    name = sky["sequence_name"].fillna("").str.strip()
    keys = [sky["night"], name]
    steps = focus.groupby(keys).transform("nunique")
    span = focus.groupby(keys).transform("max") - focus.groupby(keys).transform("min")
    # Unnamed visits share one key, though they need not share a field: never a sweep.
    sweep = (steps >= FOCUS_SWEEP_MIN_STEPS) & (span >= FOCUS_SWEEP_MIN_RANGE) & (name != "")
    return sweep.astype("boolean").fillna(False).astype(bool)


def engineeringSequences(names: pd.Series) -> pd.Series:
    """Return whether each sequence name is an engineering sequence (`ENGINEERING_SEQUENCES`)."""
    lowered = names.fillna("").astype(str).str.strip().str.lower()
    return lowered.str.startswith(ENGINEERING_SEQUENCES)


def engineeringGroups(groups: pd.Series) -> pd.Series:
    """Return whether each sequence group is an engineering group (`ENGINEERING_GROUPS`).

    An arc or trace taken inside a fiberProfiles group is exposed with most cobras hidden and the
    rest dithered across pixels: a scienceArc by type, but not the slit a calibration arc lights
    (Run31's 153004 had 65 Neon lines on a b detector against 9,900 on a nightly arc).
    """
    lowered = groups.fillna("").astype(str).str.strip().str.lower()
    return lowered.str.startswith(ENGINEERING_GROUPS)


def cadences(visits: pd.DataFrame) -> pd.Series:
    """Return whether each arc or trace is part of a set, daily, or a one-off single.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        ``pfs_visit_id``, ``iic_sequence_id``, ``group_id``,
        ``sequence_type``, ``sequence_name`` and ``night``.

    Returns
    -------
    `pandas.Series`
        ``set`` for a sequence of several visits, or a one-visit sequence in
        a sequence group with one of the same type; ``daily`` for another one-visit sequence
        whose type and name recur as one-visit sequences on
        `DAILY_MIN_NIGHTS` nights or more; ``single`` otherwise; empty for
        other sequence types. Aligned with ``visits``.
    """
    result = pd.Series("", index=visits.index, dtype=object)
    isCadence = visits["sequence_type"].isin(CADENCE_TYPES)
    if not isCadence.any():
        return result
    calib = visits[isCadence]
    size = calib.groupby("iic_sequence_id")["pfs_visit_id"].transform("size").fillna(1)
    isSingle = (size == 1) | calib["iic_sequence_id"].isna()
    group = calib["group_id"] if "group_id" in calib else pd.Series(pd.NA, index=calib.index)
    # Rows with no group drop out of the groupby, and come back from the reindex as missing.
    groupHasSet = (
        (~isSingle)
        .groupby([group, calib["sequence_type"]], dropna=True)
        .transform("any")
        .reindex(calib.index)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    singles = calib[isSingle]
    keys = [singles["sequence_type"], singles["sequence_name"].fillna("")]
    nights = singles.groupby(keys)["night"].transform("nunique").reindex(calib.index)

    cadence = pd.Series("set", index=calib.index, dtype=object)
    loose = isSingle & ~groupHasSet
    cadence[loose] = "single"
    cadence[loose & (nights >= DAILY_MIN_NIGHTS)] = "daily"
    result.loc[cadence.index] = cadence
    return result


def classifyVisits(
    listing: pd.DataFrame,
    periods: Iterable[Period],
    designs: pd.Series | None = None,
    telStatus: pd.DataFrame | None = None,
    designNames: pd.Series | None = None,
) -> pd.DataFrame:
    """Say what each visit is and whether comparison mode measures it.

    Parameters
    ----------
    listing : `pandas.DataFrame`
        Visits, from `pfs.drp.qa.comparison.queries.readVisitListing`.
    periods : iterable of `Period`
        The observing periods, from `pfs.drp.qa.comparison.runs.loadPeriods`.
    designs : `pandas.Series`, optional
        `designKinds` of the sky visits' designs. Without it every sky visit is
        science.
    telStatus : `pandas.DataFrame`, optional
        From `pfs.drp.qa.comparison.queries.readTelStatus`. Without it no visit
        is a focus sweep or dithered.
    designNames : `pandas.Series`, optional
        The shaka designs' names (``design_name`` indexed by
        ``pfs_design_id``), from
        `pfs.drp.qa.comparison.queries.readDesignNames`. Without it no visit is
        a shaka.

    Returns
    -------
    `pandas.DataFrame`
        ``listing`` with these columns added:

        ``night``
            The night (`datetime.date`), named by its evening's date.
        ``period``
            The period's name, or `None` outside every period.
        ``category``
            ``calibration`` or ``science``.
        ``cadence``
            For an arc or trace, ``set``, ``daily`` or ``single`` (`cadences`);
            empty otherwise.
        ``focusSweep``, ``dithered``
            Sky sub-labels (`bool`).
        ``shaka``
            Whether the design lights one group of every few cobras (`bool`;
            `shakaOf`).
        ``shakaMod``, ``shakaGroup``
            The modulus and group of a shaka exposure (`Int64`; NA otherwise).
        ``judged``
            Whether the QA tasks measure the visit (`bool`). The name is
            historical: the verdict is given later, from the stored numbers
            (`pfs.drp.qa.metrics.gate`).
        ``validated``
            Whether it is gated: a measured visit of a `VALIDATED_TYPES`
            type, of no engineering sequence or group (`bool`).
        ``reason``
            ``gated``, ``unvalidated``, or why it isn't measured: ``test
            exposure``, ``outside every period``, ``no sequence`` or ``no
            method for <type>``.
    """
    visits = listing.copy()
    visits["sequence_name"] = visits["sequence_name"].str.strip()  # names are typed, with stray spaces
    visits["night"] = nightOf(visits["time_exp_start"]).to_numpy()
    visits["period"] = periodOf(visits["time_exp_start"], periods).to_numpy()

    sequenceType = visits["sequence_type"].astype(object)
    isSky = sequenceType.isin(SKY_TYPES).to_numpy()
    if designs is None:
        designKind = pd.Series("science", index=visits.index)
    else:
        # As objects: no design matched leaves the design IDs' integer type, which takes no name.
        designKind = visits["pfs_design_id"].map(designs).astype(object).fillna("engineering")
    visits["category"] = np.where(isSky & (designKind == "science").to_numpy(), "science", "calibration")

    if telStatus is not None and not telStatus.empty:
        status = visits[["pfs_visit_id"]].merge(telStatus, on="pfs_visit_id", how="left")
        status.index = visits.index
    else:
        status = pd.DataFrame(
            np.nan, index=visits.index, columns=["focus_offset_max", "dither_ra_max", "dither_dec_max"]
        )
    sky = visits.loc[isSky, ["night", "sequence_name"]].assign(
        focus_offset_max=status.loc[isSky, "focus_offset_max"]
    )
    visits["focusSweep"] = False
    visits.loc[isSky, "focusSweep"] = focusSweeps(sky)
    dithered = (status["dither_ra_max"].fillna(0) > 0) | (status["dither_dec_max"].fillna(0) > 0)
    visits["dithered"] = isSky & dithered.to_numpy()

    # One (modulus, group) per design, so shakaOf runs once per design and not once per visit.
    shakaByDesign = {} if designNames is None else {int(d): shakaOf(n) for d, n in designNames.items()}
    parsed = visits["pfs_design_id"].map(lambda design: shakaByDesign.get(design, (None, None)))
    visits["shakaMod"] = parsed.map(lambda modGroup: modGroup[0]).astype("Int64")
    visits["shakaGroup"] = parsed.map(lambda modGroup: modGroup[1]).astype("Int64")
    visits["shaka"] = visits["shakaGroup"].notna()

    reason = pd.Series("gated", index=visits.index, dtype=object)
    reason[~sequenceType.isin(VALIDATED_TYPES)] = "unvalidated"
    reason[engineeringSequences(visits["sequence_name"]).to_numpy()] = "unvalidated"
    if "group_name" in visits:
        reason[engineeringGroups(visits["group_name"]).to_numpy()] = "unvalidated"
    notMeasured = sequenceType.isin(NOT_MEASURED_TYPES)
    reason[notMeasured] = "no method for " + sequenceType[notMeasured].astype(str)
    reason[sequenceType.isna()] = "no sequence"
    reason[visits["exp_type"] == "test"] = "test exposure"
    reason[visits["period"].isna()] = "outside every period"
    visits["cadence"] = cadences(visits)
    visits["reason"] = reason
    visits["judged"] = reason.isin(["gated", "unvalidated"])
    visits["validated"] = reason == "gated"
    return visits
