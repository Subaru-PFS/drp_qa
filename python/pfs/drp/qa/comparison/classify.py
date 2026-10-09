"""What each visit is, and how the gate treats it.

Every visit is a calibration or science, by what the exposure is rather than when it was taken:
every sequence type but ``scienceObject`` and ``scienceObject_windowed`` is a calibration, and
those two are science when their design was made for a science proposal. Twilight frames,
dithers and the fields of telescope focus sweeps are on engineering designs, so they are
calibrations.

``imageQualityQa`` measures and judges every exposure it can, but only the sequence types in
`VALIDATED_TYPES` are *gated*: their thresholds were derived from, and checked against, visits of
those types. The others (engineering arcs, flats, fiber profiles, focus sweeps, windowed readouts,
any new type) are *unvalidated*: judged the same way so their results can be seen, but often
off-nominal on purpose, so a FAIL there may be what the test expected. `NOT_MEASURED_TYPES` have
nothing to measure. Every visit is listed, so a run's coverage is complete and a new sequence
type shows up rather than disappearing.

Arcs and traces are taken two ways: as *sets* (several exposures of a lamp: 3 per arc lamp and 10
traces for the calibrations) and, from Run28, as *daily* single exposures of one arc and one trace,
for drift. ``cadence`` tells them apart; they are reported apart. A set can include one-visit
sequences, so a one-visit sequence in the same sequence group (``group_id``) as a longer one of its
type belongs to the set; one whose name recurs on `DAILY_MIN_NIGHTS` nights is daily; any other is a one-off
``single`` (a positioner scan, a bootstrap exposure), judged like a set.

Sky visits also get sub-labels from the telescope status: `focusSweep` (a telescope focus sweep,
which changes how much light enters the fibers but not the spectrograph's line widths) and
``dithered``.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from pfs.drp.qa.comparison.runs import Period, nightOf, periodOf

__all__ = [
    "CADENCE_TYPES",
    "DAILY_MIN_NIGHTS",
    "ENGINEERING_CATEGORIES",
    "FOCUS_SWEEP_MIN_RANGE",
    "FOCUS_SWEEP_MIN_STEPS",
    "NOT_MEASURED_TYPES",
    "SKY_TYPES",
    "VALIDATED_TYPES",
    "cadences",
    "classifyVisits",
    "designKinds",
    "focusSweeps",
]

#: Sequence types the thresholds are validated for: the gate.
VALIDATED_TYPES = ("scienceArc", "scienceTrace", "scienceObject")

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
ENGINEERING_CATEGORIES = frozenset({"EN", "none", "other"})

#: A telescope focus sweep: at least this many distinct focus offsets on one field in one night...
FOCUS_SWEEP_MIN_STEPS = 10
#: ...spanning at least this much (mm). The focus correction during a night spans about 0.1 mm.
FOCUS_SWEEP_MIN_RANGE = 0.5


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
) -> pd.DataFrame:
    """Say what each visit is and whether the gate judges it.

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
        ``judged``
            Whether ``imageQualityQa`` measures and judges the visit (`bool`).
        ``validated``
            Whether its verdict is the gate's: a judged visit of a
            `VALIDATED_TYPES` type (`bool`).
        ``reason``
            ``gated``, ``unvalidated``, or why it isn't judged: ``test
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
        designKind = visits["pfs_design_id"].map(designs).fillna("engineering")
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

    reason = pd.Series("gated", index=visits.index, dtype=object)
    reason[~sequenceType.isin(VALIDATED_TYPES)] = "unvalidated"
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
