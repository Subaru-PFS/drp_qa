"""Statistics of a ``fiberProfiles`` calibration, for judging it as a product.

A ``fiberProfiles`` calibration holds, for each fiber, the cross-dispersion
shape of its trace in a few swaths up the detector: pixel values at sub-pixel
offsets from the trace center. Extraction weights every exposure by these
shapes, so a bad profile biases every spectrum taken with it. This module
measures each profile; it sets no limits.

Every statistic is a ratio within one swath's profile, so none depends on how
the profile is normalised.

``fwhm``
    Width at half maximum, read directly
    (`pfs.drp.qa.crossDispersion.profileFwhm`). The second moment over the
    whole profile is not used: a floor of 2 % of the peak doubles it and moves
    this width by 1.5 %.
``centroid``
    Offset of the core from the center, in pixels: the mean offset under a
    Gaussian weight of sigma `CORE_RADIUS`/2, moved to its own result until
    it settles, so the wings take no part. Positive is toward higher column.
``wingFrac``
    Fraction of the profile's light at `WING_RADIUS` or more from the center.
    A Gaussian of the PFS core width has under 0.3 % of its light there.
``wingMax``
    Highest value at `WING_RADIUS` or more from the center, as a fraction of
    the peak. Light left over from another fiber shows here as a plateau or a
    bump.
``minimum``
    Lowest value, as a fraction of the peak. A profile cannot be negative.
``variation``
    Largest departure of the swath's profile from the mean of the fiber's
    swaths, each scaled to unit sum, as a fraction of the mean's peak.
``maskedFrac``
    Fraction of the swath's samples that are masked or not finite. The other
    statistics interpolate over them.

`measureProfiles` returns these per swath for a whole calibration and
`summarizeFibers` reduces them to one row per fiber. `compareFibers` sets one
calibration's fibers against another's, `neighbourResidual` sets each fiber
against the fibers beside it on the slit, and `coverage` lists the fibers a
calibration should hold and does not.

**Replaced profiles.** drp_stella's ``MergeFiberProfilesTask`` gives a fiber it
has no trustworthy measurement for (one that cannot be hidden, one measured too
close to another lit fiber, one with no profile) the mean of the profiles of
its nearest good fibers. Nothing in the calibration marks them; `findReplaced`
finds them, as profiles that equal the mean of a few nearby profiles (any of
the counts in `REPLACE_DONORS`). A replaced profile is a copy, not a
measurement: it inherits whatever its donors carry, and it is left out of the
reference its neighbours are set against.

`profileQa` is what the ``fiberProfilesQa`` task stores: the per-fiber table
(`FIBER_TABLE_COLUMNS`), with each fiber's ``wingExcess`` and ``fwhmExcess``
against its neighbours and whether it was replaced, and the detector's metrics
in long format (`METRICS`). Numbers only: no limit is applied here.

No Butler, no stack: numpy and pandas. Tested in
``tests/fiberProfiles/test_fiberProfiles.py``.
"""

from collections.abc import Collection, Iterator, Mapping
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pfs.drp.qa.crossDispersion import profileFwhm

__all__ = [
    "CORE_RADIUS",
    "FIBER_COLUMNS",
    "FIBER_TABLE_COLUMNS",
    "METRICS",
    "METRIC_COLUMNS",
    "PLACEHOLDER_FWHM",
    "REPLACE_DONORS",
    "SWATH_COLUMNS",
    "WING_EXCESS_LIMIT",
    "WING_RADIUS",
    "compareFibers",
    "coverage",
    "detectorMetrics",
    "fiberTable",
    "findReplaced",
    "iterProfiles",
    "measureProfiles",
    "neighbourResidual",
    "profileQa",
    "summarizeFibers",
    "swathStats",
]

#: Offsets within this many pixels of the center are the core.
CORE_RADIUS = 3.0

#: Offsets at least this many pixels from the center are the wings.
WING_RADIUS = 4.0

#: FWHM, in pixels, of the Gaussian drp_stella stores for a fiber it had no light for.
PLACEHOLDER_FWHM = 1.5

#: How close to `PLACEHOLDER_FWHM` every swath must read for a fiber to count as a placeholder.
PLACEHOLDER_TOLERANCE = 0.1

#: Most times the centroid's weight is moved, and the step, in pixels, at which it has settled.
CENTROID_ITERATIONS = 50
CENTROID_TOLERANCE = 1e-5

#: Fewest usable samples for a swath to be measured.
MIN_SAMPLES = 5

#: Statistics measured per swath, in the order of the table's columns.
SWATH_STATS = ("fwhm", "centroid", "wingFrac", "wingMax", "minimum", "variation", "maskedFrac")

#: Columns of the per-swath table.
SWATH_COLUMNS = ("fiberId", "row", *SWATH_STATS)

#: Columns of the per-fiber table.
FIBER_COLUMNS = (
    "fiberId",
    "nSwath",
    "nMeasured",
    "fwhm",
    "fwhmRange",
    "centroid",
    "wingFrac",
    "wingMax",
    "minimum",
    "variation",
    "maskedFrac",
    "placeholder",
)

#: Per-fiber statistics `compareFibers` differences by default.
COMPARE_COLUMNS = ("fwhm", "centroid", "wingFrac", "wingMax", "minimum", "variation")

#: How many fibers' profiles may be averaged into a replaced one. drp_stella's two replacement steps default to
#: 3 (``MergeFiberProfilesConfig.replaceNearest``) and 2 (``ReduceProfilesConfig.replaceNearest``), and both
#: can be configured.
REPLACE_DONORS = (2, 3, 4)

#: Fibers either side among which a replaced profile's donors are looked for. drp_stella takes no donor
#: within ``fiberInfluence`` (3) fibers of a fiber that is lit in the darks, so beside a group of such fibers
#: the donors are six or seven fibers away.
REPLACE_REACH = 10

#: Samples of a profile on which the candidate sets of donors are ranked, and how many of the closest sets are
#: then compared sample by sample. A copy equals its donors' mean at every sample, so it is among the closest
#: on any of them; ranking first keeps a wide reach affordable.
REPLACE_RANK_SAMPLES = 64
REPLACE_SHORTLIST = 16

#: Largest difference, as a fraction of the profile's peak, at which a profile equals the mean of its donors.
#: Stored copies differ by the rounding of 32-bit floats, 1e-7; measured profiles differ from any such mean by
#: their noise, 1e-3 or more.
REPLACE_TOLERANCE = 1e-6

#: How much closer a mean of more donors must be than the match already found, to take its place.
REPLACE_CLOSER = 0.1

#: Fibers either side that make up the neighbourhood of ``wingExcess`` and ``fwhmExcess``.
NEIGHBOUR_HALF_WINDOW = 5

#: The ``wingExcess`` above which the calibration report (``drpQa.py calib profiles``) lists a profile. Chosen on
#: the Run24 to Run31 calibrations, b4 left out: it lists 19 to 31 fibers in an ordinary run, 62 in Run31 and 195
#: in Run24. A run-level listing, as the slit-offset outliers are; it is not a verdict on an image.
WING_EXCESS_LIMIT = 0.01

#: Columns of the per-fiber table the task stores, after the identifying ones.
FIBER_TABLE_COLUMNS = (*FIBER_COLUMNS, "wingExcess", "fwhmExcess", "replaced", "donors")

#: The detector metrics the task stores, with what each is.
METRICS = {
    "nProfiles": "Fibers the calibration holds a profile for.",
    "nMissing": "Expected fibers without a profile; NaN when the expected fibers are not known.",
    "nPlaceholder": "Profiles that are the placeholder Gaussian of a fiber with no light.",
    "nReplaced": "Profiles that are the mean of nearby fibers' profiles, not a measurement.",
    "nUnmeasured": "Profiles with no swath that could be measured: masked, or without a positive peak and sum.",
    "medProfileFwhm": "Median width at half maximum of the measured profiles, in pixels.",
    "medWingFrac": "Median wing fraction of the measured profiles.",
    "medVariation": "Median, over the measured profiles, of the largest swath-to-swath departure.",
    "maxWingExcess": "Largest wing fraction above the slit neighbours, among the measured profiles.",
    "maxWingExcessReplaced": "The same among the replaced profiles; NaN when there are none.",
    "maxAbsFwhmExcess": "Largest width difference from the slit neighbours, either way, among the measured profiles.",
    "minProfileValue": "Lowest profile value on the detector, as a fraction of that profile's peak.",
}

#: The columns of the long-format metrics table, in order, with their dtypes.
METRIC_COLUMNS = {
    "visit": "int64",
    "arm": "string",
    "spectrograph": "int64",
    "calibVisit0": "int64",
    "calibRun": "string",
    "metric": "string",
    "value": "float64",
}


def swathStats(
    index: ArrayLike,
    profiles: ArrayLike,
    wingRadius: float = WING_RADIUS,
    coreRadius: float = CORE_RADIUS,
) -> dict[str, np.ndarray]:
    """Measure each swath of one fiber's profile.

    Parameters
    ----------
    index : array-like, shape (nIndex,)
        Offset of each profile sample from the trace center, in pixels, evenly
        spaced and increasing (``FiberProfile.index``).
    profiles : array-like, shape (nSwath, nIndex) or (nIndex,)
        Profile values, one row per swath (``FiberProfile.profiles``). Masked
        or non-finite samples are interpolated over.
    wingRadius : `float`
        Offsets at least this far from the center are the wings.
    coreRadius : `float`
        Offsets within this far of the center are the core: the peak is looked
        for there, and half of it is the sigma of the centroid's weight.

    Returns
    -------
    `dict` [`str`, `numpy.ndarray`]
        One array of length ``nSwath`` per name in `SWATH_STATS`. A swath with
        fewer than `MIN_SAMPLES` usable samples, or with no positive peak or
        sum, is NaN in every statistic except ``maskedFrac``.
    """
    index = np.asarray(index, dtype=np.float64)
    masked = np.ma.masked_invalid(np.ma.atleast_2d(np.ma.asarray(profiles, dtype=np.float64)))
    mask = np.ma.getmaskarray(masked)

    filled = np.full(masked.shape, np.nan)
    for ii, good in enumerate(~mask):
        if good.sum() >= MIN_SAMPLES:
            filled[ii] = np.interp(index, index[good], masked.data[ii, good])

    core = np.abs(index) <= coreRadius
    wing = np.abs(index) >= wingRadius
    total = filled.sum(axis=1)
    peak = filled[:, core].max(axis=1)
    usable = (total > 0) & (peak > 0)
    total = np.where(usable, total, np.nan)
    peak = np.where(usable, peak, np.nan)
    filled[~usable] = np.nan

    centroid = np.where(usable, 0.0, np.nan)
    for _ in range(CENTROID_ITERATIONS):
        weight = filled * np.exp(-0.5 * ((index - centroid[:, np.newaxis]) / (0.5 * coreRadius)) ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            moved = (weight * index).sum(axis=1) / weight.sum(axis=1)
        settled = np.nanmax(np.abs(moved - centroid), initial=0.0) < CENTROID_TOLERANCE
        centroid = moved
        if settled:
            break
    fwhm = profileFwhm(index, masked)
    fwhm[~usable] = np.nan

    variation = np.full(len(filled), np.nan)
    if usable.any():
        unit = filled / total[:, np.newaxis]
        mean = unit[usable].mean(axis=0)
        variation = np.abs(unit - mean).max(axis=1) / mean[core].max()

    return {
        "fwhm": fwhm,
        "centroid": centroid,
        "wingFrac": filled[:, wing].sum(axis=1) / total,
        "wingMax": filled[:, wing].max(axis=1) / peak,
        "minimum": filled.min(axis=1) / peak,
        "variation": variation,
        "maskedFrac": mask.mean(axis=1),
    }


def iterProfiles(fiberProfiles: Any) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ma.MaskedArray]]:
    """Yield each fiber's profile from either form of the calibration.

    Parameters
    ----------
    fiberProfiles : `pfs.datamodel.PfsFiberProfiles` or `pfs.drp.stella.FiberProfileSet`
        The calibration: the datamodel form (parallel lists ``fiberId``,
        ``oversample``, ``rows``, ``profiles``) or the drp_stella form (a
        mapping from ``fiberId`` to a profile with ``index``, ``rows`` and
        ``profiles``). Only those attributes are used.

    Yields
    ------
    fiberId : `int`
        Fiber identifier.
    index : `numpy.ndarray`, shape (nIndex,)
        Offset of each sample from the trace center, in pixels.
    rows : `numpy.ndarray`, shape (nSwath,)
        Detector row at the middle of each swath.
    profiles : `numpy.ma.MaskedArray`, shape (nSwath, nIndex)
        Profile values.
    """
    if hasattr(fiberProfiles, "oversample"):
        for fiberId, oversample, rows, profiles in zip(
            fiberProfiles.fiberId,
            fiberProfiles.oversample,
            fiberProfiles.rows,
            fiberProfiles.profiles,
            strict=True,
        ):
            profiles = np.ma.atleast_2d(np.ma.asarray(profiles, dtype=np.float64))
            nIndex = profiles.shape[1]
            index = (np.arange(nIndex) - nIndex // 2) / float(oversample)
            yield int(fiberId), index, np.asarray(rows, dtype=np.float64), profiles
        return
    for fiberId in fiberProfiles.fiberId:
        profile = fiberProfiles[fiberId]
        yield (
            int(fiberId),
            np.asarray(profile.index, dtype=np.float64),
            np.asarray(profile.rows, dtype=np.float64),
            np.ma.atleast_2d(np.ma.asarray(profile.profiles, dtype=np.float64)),
        )


def measureProfiles(
    fiberProfiles: Any, wingRadius: float = WING_RADIUS, coreRadius: float = CORE_RADIUS
) -> pd.DataFrame:
    """Measure every swath of every fiber in a calibration.

    Parameters
    ----------
    fiberProfiles : `pfs.datamodel.PfsFiberProfiles` or `pfs.drp.stella.FiberProfileSet`
        The calibration, in either form `iterProfiles` reads.
    wingRadius, coreRadius : `float`
        As for `swathStats`.

    Returns
    -------
    `pandas.DataFrame`
        One row per fiber and swath, with the columns `SWATH_COLUMNS`.
    """
    frames = []
    for fiberId, index, rows, profiles in iterProfiles(fiberProfiles):
        if len(rows) == 0:
            continue
        stats = swathStats(index, profiles, wingRadius=wingRadius, coreRadius=coreRadius)
        frames.append(pd.DataFrame({"fiberId": fiberId, "row": rows, **stats}))
    if not frames:
        return pd.DataFrame(columns=list(SWATH_COLUMNS))
    return pd.concat(frames, ignore_index=True)[list(SWATH_COLUMNS)]


def summarizeFibers(swaths: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-swath table to one row per fiber.

    Parameters
    ----------
    swaths : `pandas.DataFrame`
        Table from `measureProfiles`.

    Returns
    -------
    `pandas.DataFrame`
        One row per fiber, ordered by ``fiberId``, with the columns
        `FIBER_COLUMNS`:

        ``nSwath``, ``nMeasured``
            Swaths the fiber has, and those with a width.
        ``fwhm``, ``centroid``, ``wingFrac``
            Median over the swaths.
        ``fwhmRange``
            Widest swath minus narrowest.
        ``wingMax``, ``variation``
            Highest over the swaths.
        ``minimum``
            Lowest over the swaths.
        ``maskedFrac``
            Mean over the swaths.
        ``placeholder``
            Every swath reads within `PLACEHOLDER_TOLERANCE` of
            `PLACEHOLDER_FWHM`: the Gaussian stored for a fiber with no light.
    """
    if swaths.empty:
        return pd.DataFrame(columns=list(FIBER_COLUMNS))
    grouped = swaths.groupby("fiberId", sort=True)
    summary = grouped.agg(
        nSwath=("row", "size"),
        nMeasured=("fwhm", "count"),
        fwhm=("fwhm", "median"),
        fwhmRange=("fwhm", lambda values: values.max() - values.min()),
        centroid=("centroid", "median"),
        wingFrac=("wingFrac", "median"),
        wingMax=("wingMax", "max"),
        minimum=("minimum", "min"),
        variation=("variation", "max"),
        maskedFrac=("maskedFrac", "mean"),
    )
    offPlaceholder = (swaths["fwhm"] - PLACEHOLDER_FWHM).abs() > PLACEHOLDER_TOLERANCE
    anyOff = offPlaceholder.groupby(swaths["fiberId"], sort=True).any()
    summary["placeholder"] = (summary["nMeasured"] == summary["nSwath"]) & ~anyOff
    return summary.reset_index()[list(FIBER_COLUMNS)]


def coverage(fiberId: Collection[int], expected: Collection[int]) -> dict[str, list[int]]:
    """Compare the fibers a calibration holds with the fibers it should hold.

    Parameters
    ----------
    fiberId : collection of `int`
        Fibers in the calibration.
    expected : collection of `int`
        Fibers the detector should have a profile for.

    Returns
    -------
    `dict` [`str`, `list` [`int`]]
        ``missing``: expected and absent. ``unexpected``: present and not
        expected. Both sorted.
    """
    have = {int(ff) for ff in fiberId}
    want = {int(ff) for ff in expected}
    return {"missing": sorted(want - have), "unexpected": sorted(have - want)}


def compareFibers(
    new: pd.DataFrame, previous: pd.DataFrame, columns: Collection[str] = COMPARE_COLUMNS
) -> pd.DataFrame:
    """Set one calibration's fibers against another's.

    Parameters
    ----------
    new, previous : `pandas.DataFrame`
        Per-fiber tables from `summarizeFibers`.
    columns : collection of `str`
        Statistics to difference.

    Returns
    -------
    `pandas.DataFrame`
        One row per fiber in either table, ordered by ``fiberId``. For each
        statistic: its value in ``new``, ``<name>Previous`` and
        ``<name>Change`` (new minus previous). ``inNew`` and ``inPrevious``
        say which table holds the fiber; a fiber in only one has NaN changes.
    """
    columns = list(columns)
    left = new[["fiberId", *columns]].assign(inNew=True)
    right = previous[["fiberId", *columns]].assign(inPrevious=True)
    merged = left.merge(right, on="fiberId", how="outer", suffixes=("", "Previous")).sort_values("fiberId")
    for flag in ("inNew", "inPrevious"):
        merged[flag] = merged[flag].eq(True)
    for name in columns:
        merged[f"{name}Change"] = merged[name] - merged[f"{name}Previous"]
    return merged.reset_index(drop=True)


def neighbourResidual(fibers: pd.DataFrame, column: str, halfWindow: int = 5) -> pd.Series:
    """Return a statistic minus its median over the fibers beside each fiber.

    Fibers of one spectrograph are numbered along the slit, so neighbours in
    ``fiberId`` are neighbours on the detector and share its optics. The
    median is over up to ``halfWindow`` fibers either side, the fiber itself
    included, and leaves out placeholders and fibers with no value, so one
    discrepant fiber does not move its own reference. Replaced profiles
    (a ``replaced`` column, when the table has one) are left out of the
    reference too: they are copies, and a group of them that inherited the
    same light would raise it. A replaced profile still gets a residual.

    Parameters
    ----------
    fibers : `pandas.DataFrame`
        Per-fiber table from `summarizeFibers`, for one detector.
    column : `str`
        Statistic to take the residual of.
    halfWindow : `int`
        Fibers either side that make up the neighbourhood.

    Returns
    -------
    `pandas.Series`
        Residual, indexed as ``fibers``; NaN for placeholders and for fibers
        with no value.
    """
    order = fibers["fiberId"].argsort(kind="stable").to_numpy()
    values = fibers[column].to_numpy(dtype=np.float64)[order]
    if "placeholder" in fibers:
        values = np.where(fibers["placeholder"].to_numpy(dtype=bool)[order], np.nan, values)
    reference = values
    if "replaced" in fibers:
        reference = np.where(fibers["replaced"].to_numpy(dtype=bool)[order], np.nan, values)
    local = pd.Series(reference).rolling(2 * halfWindow + 1, center=True, min_periods=1).median()
    residual = np.full(len(fibers), np.nan)
    residual[order] = values - local.to_numpy()
    return pd.Series(residual, index=fibers.index, name=f"{column}Residual")


def findReplaced(
    fiberProfiles: Any,
    nDonors: int | Collection[int] = REPLACE_DONORS,
    reach: int = REPLACE_REACH,
    tolerance: float = REPLACE_TOLERANCE,
) -> dict[int, tuple[int, ...]]:
    """Return the fibers whose stored profile is the mean of nearby fibers' profiles.

    Parameters
    ----------
    fiberProfiles : `pfs.datamodel.PfsFiberProfiles` or `pfs.drp.stella.FiberProfileSet`
        The calibration, in either form `iterProfiles` reads.
    nDonors : `int` or collection of `int`
        How many profiles may be averaged into a replaced one; with several
        counts, the closest mean decides, and the fewer donors when two are
        as close.
    reach : `int`
        Fibers either side, in order of ``fiberId``, among which the donors
        are looked for. With many candidate sets, they are ranked on a thin
        sample of the profile first and only the closest are compared in
        full.
    tolerance : `float`
        Largest difference from the donors' mean, as a fraction of the
        profile's peak, at which the profile counts as that mean.

    Returns
    -------
    `dict` [`int`, `tuple` [`int`]]
        The donors of each replaced fiber, by ``fiberId``. A profile with too
        few finite samples, or with too few neighbours of its shape, is not
        in it. Identical neighbouring profiles, such as a run of
        placeholders, each equal the mean of the others and are in it.
    """
    entries = sorted(
        (
            (fiberId, np.ma.filled(profiles, np.nan))
            for fiberId, _, _, profiles in iterProfiles(fiberProfiles)
        ),
        key=lambda entry: entry[0],
    )
    counts = sorted({nDonors} if isinstance(nDonors, int) else {int(count) for count in nDonors})
    replaced = {}
    for ii, (fiberId, values) in enumerate(entries):
        finite = np.isfinite(values)
        if finite.sum() < MIN_SAMPLES:
            continue
        near = [
            jj
            for jj in range(max(0, ii - reach), min(len(entries), ii + reach + 1))
            if jj != ii and entries[jj][1].shape == values.shape
        ]
        stack = np.array([entries[jj][1] for jj in near])
        samples = tuple(range(1, values.ndim + 1))
        # Samples every candidate has, thinned, to rank the sets of donors on.
        flat = stack.reshape(len(near), -1) if near else np.empty((0, values.size))
        shared = np.flatnonzero(finite.ravel() & np.isfinite(flat).all(axis=0))
        ranked = shared[np.linspace(0, shared.size - 1, min(shared.size, REPLACE_RANK_SAMPLES)).astype(int)]
        closest = tolerance * np.nanmax(np.abs(values))
        for count in counts:
            if len(near) < count:
                continue
            combos = np.array(list(combinations(range(len(near)), count)))
            if len(combos) > REPLACE_SHORTLIST and ranked.size >= MIN_SAMPLES:
                rough = np.abs(flat[:, ranked][combos].mean(axis=1) - values.ravel()[ranked]).max(axis=1)
                combos = combos[np.sort(np.argsort(rough, kind="stable")[:REPLACE_SHORTLIST])]
            with np.errstate(invalid="ignore"):
                difference = np.abs(stack[combos].mean(axis=1) - values)
            compared = np.isfinite(difference).sum(axis=samples)
            difference = np.where(np.isfinite(difference), difference, -np.inf).max(axis=samples)
            difference = np.where(compared >= MIN_SAMPLES, difference, np.inf)
            best = int(np.argmin(difference))
            # The fewest donors that match keep the fiber: two copies of the same donors are identical, so
            # each also equals the mean of those donors and its twin, to rounding. A larger count has to be
            # closer by a clear factor to take it.
            if fiberId in replaced:
                matches = difference[best] < REPLACE_CLOSER * closest
            else:
                matches = difference[best] <= closest
            if matches:
                closest = difference[best]
                replaced[fiberId] = tuple(entries[near[jj]][0] for jj in combos[best])
    return replaced


def fiberTable(
    fiberProfiles: Any,
    wingRadius: float = WING_RADIUS,
    coreRadius: float = CORE_RADIUS,
    halfWindow: int = NEIGHBOUR_HALF_WINDOW,
) -> pd.DataFrame:
    """Return one row per fiber of a calibration, with its standing among its neighbours.

    Parameters
    ----------
    fiberProfiles : `pfs.datamodel.PfsFiberProfiles` or `pfs.drp.stella.FiberProfileSet`
        The calibration of one detector, in either form `iterProfiles` reads.
    wingRadius, coreRadius : `float`
        As for `swathStats`.
    halfWindow : `int`
        Fibers either side that make up the neighbourhood.

    Returns
    -------
    `pandas.DataFrame`
        `FIBER_TABLE_COLUMNS`, ordered by ``fiberId``: the columns of
        `summarizeFibers`, then

        ``wingExcess``, ``fwhmExcess``
            ``wingFrac`` and ``fwhm`` minus their median over the measured
            slit neighbours (`neighbourResidual`).
        ``replaced``
            The profile is the mean of nearby fibers' profiles (`findReplaced`)
            and not a placeholder.
        ``donors``
            Those fibers, space-separated; empty for a measured profile.
    """
    fibers = summarizeFibers(measureProfiles(fiberProfiles, wingRadius=wingRadius, coreRadius=coreRadius))
    if fibers.empty:
        return pd.DataFrame(columns=list(FIBER_TABLE_COLUMNS))
    replaced = findReplaced(fiberProfiles)
    # Neighbouring placeholders are the same Gaussian, so each equals the mean of the others: not a replacement.
    fibers["replaced"] = fibers["fiberId"].isin(replaced) & ~fibers["placeholder"].astype(bool)
    fibers["donors"] = [
        " ".join(str(donor) for donor in replaced[int(ff)]) if isReplaced else ""
        for ff, isReplaced in zip(fibers["fiberId"], fibers["replaced"], strict=True)
    ]
    fibers["wingExcess"] = neighbourResidual(fibers, "wingFrac", halfWindow)
    fibers["fwhmExcess"] = neighbourResidual(fibers, "fwhm", halfWindow)
    return fibers[list(FIBER_TABLE_COLUMNS)]


def detectorMetrics(fibers: pd.DataFrame, expected: Collection[int] | None = None) -> dict[str, float]:
    """Reduce a detector's per-fiber table to its metrics.

    Parameters
    ----------
    fibers : `pandas.DataFrame`
        Table from `fiberTable`.
    expected : collection of `int`, optional
        Fibers the detector should have a profile for.

    Returns
    -------
    `dict` [`str`, `float`]
        One value per name in `METRICS`. "Measured" leaves out placeholders
        and replaced profiles; a statistic over no profiles is NaN. A profile
        with no measurable swath is NaN in every statistic, so it enters none
        of them: ``nUnmeasured`` counts those.
    """
    measured = fibers[~fibers["placeholder"].astype(bool) & ~fibers["replaced"].astype(bool)]
    copies = fibers[fibers["replaced"].astype(bool)]

    def stat(values: pd.Series, how: str) -> float:
        values = values.dropna()
        return float(getattr(values, how)()) if len(values) else float("nan")

    return {
        "nProfiles": float(len(fibers)),
        "nMissing": float(len(coverage(fibers["fiberId"], expected)["missing"]))
        if expected is not None
        else float("nan"),
        "nPlaceholder": float(fibers["placeholder"].astype(bool).sum()),
        "nReplaced": float(len(copies)),
        "nUnmeasured": float((fibers["nMeasured"] == 0).sum()),
        "medProfileFwhm": stat(measured["fwhm"], "median"),
        "medWingFrac": stat(measured["wingFrac"], "median"),
        "medVariation": stat(measured["variation"], "median"),
        "maxWingExcess": stat(measured["wingExcess"], "max"),
        "maxWingExcessReplaced": stat(copies["wingExcess"], "max"),
        "maxAbsFwhmExcess": stat(measured["fwhmExcess"].abs(), "max"),
        "minProfileValue": stat(fibers["minimum"], "min"),
    }


def profileQa(
    fiberProfiles: Any,
    dataId: Mapping[str, Any],
    expected: Collection[int] | None = None,
    calibVisit0: int = -1,
    calibObsDate: str = "",
    calibRun: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure one detector's ``fiberProfiles`` calibration, as the task stores it.

    Parameters
    ----------
    fiberProfiles : `pfs.datamodel.PfsFiberProfiles` or `pfs.drp.stella.FiberProfileSet`
        The calibration of one detector.
    dataId : mapping
        The image the calibration was resolved for: ``visit``, ``arm``,
        ``spectrograph``.
    expected : collection of `int`, optional
        Fibers the detector should have a profile for.
    calibVisit0 : `int`
        First visit the calibration was built from; -1 when not known. It
        tells the tables of one calibration, stored for several visits, apart
        from those of another.
    calibObsDate : `str`
        The calibration's observation date.
    calibRun : `str`
        The Butler RUN collection the calibration was read from; empty when not
        known. It says whether a candidate's own calibration was measured, or
        one found further down the inputs.

    Returns
    -------
    fibers : `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph``, ``calibVisit0``,
        ``calibObsDate``, ``calibRun``, then `FIBER_TABLE_COLUMNS`: one row per
        fiber.
    metrics : `pandas.DataFrame`
        `METRIC_COLUMNS`: one row per name in `METRICS`.
    """
    visit, arm, spectrograph = int(dataId["visit"]), str(dataId["arm"]), int(dataId["spectrograph"])
    fibers = fiberTable(fiberProfiles)
    values = detectorMetrics(fibers, expected)
    fibers.insert(0, "visit", visit)
    fibers.insert(1, "arm", arm)
    fibers.insert(2, "spectrograph", spectrograph)
    fibers.insert(3, "calibVisit0", int(calibVisit0))
    fibers.insert(4, "calibObsDate", str(calibObsDate))
    fibers.insert(5, "calibRun", str(calibRun))
    metrics = pd.DataFrame(
        {
            "visit": visit,
            "arm": arm,
            "spectrograph": spectrograph,
            "calibVisit0": int(calibVisit0),
            "calibRun": str(calibRun),
            "metric": list(values),
            "value": list(values.values()),
        }
    ).astype(METRIC_COLUMNS)
    return fibers, metrics
