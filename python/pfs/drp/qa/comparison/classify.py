"""What each visit is, and whether the gate judges it.

Every visit is a calibration or science, by what the exposure is rather than when it was taken:
every sequence type but ``scienceObject`` and ``scienceObject_windowed`` is a calibration, and
those two are science when their design was made for a science proposal. Twilight frames,
dithers and the fields of telescope focus sweeps are on engineering designs, so they are
calibrations.

The gate (``imageQualityQa``) judges the sequence types in `JUDGED_TYPES`. Every other visit is
listed with the reason it isn't judged, so a run's coverage is complete and a new sequence type
shows up rather than disappearing.

Sky visits also get sub-labels from the telescope status: `focusSweep` (a telescope focus sweep,
which changes how much light enters the fibers but not the spectrograph's line widths) and
``dithered``.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from pfs.drp.qa.comparison.runs import Period, nightOf, periodOf

__all__ = [
    "ENGINEERING_CATEGORIES",
    "FOCUS_SWEEP_MIN_RANGE",
    "FOCUS_SWEEP_MIN_STEPS",
    "JUDGED_TYPES",
    "SKY_TYPES",
    "classifyVisits",
    "designKinds",
    "focusSweeps",
]

#: Sequence types the gate judges.
JUDGED_TYPES = ("scienceArc", "scienceTrace", "scienceObject")

#: Sequence types taken on sky, whose design says whether they are science.
SKY_TYPES = ("scienceObject", "scienceObject_windowed")

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
    `FOCUS_SWEEP_MIN_RANGE` mm, within one night and sequence name.

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
    keys = [sky["night"], sky["sequence_name"].fillna("")]
    steps = focus.groupby(keys).transform("nunique")
    span = focus.groupby(keys).transform("max") - focus.groupby(keys).transform("min")
    return ((steps >= FOCUS_SWEEP_MIN_STEPS) & (span >= FOCUS_SWEEP_MIN_RANGE)).fillna(False).astype(bool)


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
        ``focusSweep``, ``dithered``
            Sky sub-labels (`bool`).
        ``judged``
            Whether the gate judges the visit (`bool`).
        ``reason``
            Why not, or ``judged``: ``test exposure``, ``outside every
            period``, ``no sequence`` or ``no method for <type>``.
    """
    visits = listing.copy()
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

    reason = pd.Series("judged", index=visits.index, dtype=object)
    notJudged = ~sequenceType.isin(JUDGED_TYPES)
    reason[notJudged] = "no method for " + sequenceType[notJudged].fillna("").astype(str)
    reason[sequenceType.isna()] = "no sequence"
    reason[visits["exp_type"] == "test"] = "test exposure"
    reason[visits["period"].isna()] = "outside every period"
    visits["reason"] = reason
    visits["judged"] = reason == "judged"
    return visits
