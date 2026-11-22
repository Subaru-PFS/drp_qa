"""Per-AG-exposure seeing and sky transparency for qadb, from good matches via pfs.utils.agMetrics.

The `seeing_agc_exposure` and `transparency_agc_exposure` values the observatory tool
(pfs_obsproc_qa_tools ``obsCondition``) fills today, recomputed with the shared `pfs.utils.agMetrics`
method — the airmass-corrected, PSF-floored seeing and the magnitude-based transparency — over the
good matches only, as the real-time ag actor log reports them. The spots come from
`pfs.drp.qa.guiders.queries.readConditionSpots`; `exposureConditions` aggregates per exposure and
`writeConditions` upserts the rows. Not yet wired into the QA pipeline.

The aggregation follows the ag actor log, not the tool (which it is meant to replace): seeing is the
median / mean / sigma of every good match's calibrated seeing, while transparency, like
`agMetrics.summarizeTransparency`, is taken over the brighter stars of each exposure (the faintest
quarter left out), so the stored transparency matches the log's. The seeing median is unchanged by
this (`agMetrics.calibratedSeeing` is monotonic, so the median commutes with it).
"""

import numpy as np
import pandas as pd

from pfs.drp.qa.guiders.queries import readConditionSpots
from pfs.drp.qa.qadb.writer import upsert
from pfs.utils import agMetrics

#: The faintest spots this fraction leaves out of the transparency, as `agMetrics.summarizeTransparency`.
_TRANSPARENCY_BRIGHT_FRACTION = 0.75
#: A transparency outside this is a mismatch, not sky (`agMetrics.summarizeTransparency`).
_TRANSPARENCY_RANGE = (0.0, 2.0)

_COLUMNS = [
    "agc_exposure_id",
    "pfs_visit_id",
    "taken_at",
    "seeing_mean",
    "seeing_median",
    "seeing_sigma",
    "transparency_mean",
    "transparency_median",
    "transparency_sigma",
]


def exposureConditions(spots: pd.DataFrame) -> pd.DataFrame:
    """Per-AG-exposure seeing (zenith, arcsec) and sky transparency from its good-match spots.

    Seeing is `agMetrics.calibratedSeeing` of each spot's FWHM, then the mean / median / sigma over
    all the exposure's good matches; a spot the AG cannot resolve (NaN seeing) drops out.
    Transparency is `agMetrics.magnitudesToTransparency` per spot, aggregated the same way but over
    the brighter `_TRANSPARENCY_BRIGHT_FRACTION` of each exposure's spots and inside
    `_TRANSPARENCY_RANGE`, matching the ag actor log's `agMetrics.summarizeTransparency`. The second
    moments are already corrected for the ics_agccActor ``"git"`` halving (`readConditionSpots`).

    Parameters
    ----------
    spots : `pandas.DataFrame`
        The good-match spots, from `pfs.drp.qa.guiders.queries.readConditionSpots`: one row per spot
        with ``agc_exposure_id``, ``pfs_visit_id``, ``taken_at``, ``altitude`` (degrees), ``mxx`` /
        ``myy``, ``estimated_magnitude`` and ``guide_star_magnitude`` / ``guide_star_color``.

    Returns
    -------
    conditions : `pandas.DataFrame`
        One row per ``agc_exposure_id`` with ``pfs_visit_id``, ``taken_at`` and the ``seeing_*`` /
        ``transparency_*`` columns of the qadb tables (``wavelength_ref`` is added by
        `writeConditions`). Empty (with those columns) for no spots.
    """
    if len(spots) == 0:
        return pd.DataFrame(columns=_COLUMNS)
    spots = spots.copy()
    fwhm = agMetrics.momentsToFwhm(spots.mxx.to_numpy(float), spots.myy.to_numpy(float))
    spots["seeing"] = agMetrics.calibratedSeeing(fwhm, spots.altitude.to_numpy(float))

    transparency = agMetrics.magnitudesToTransparency(
        spots.estimated_magnitude.to_numpy(float),
        spots.guide_star_magnitude.to_numpy(float),
        spots.guide_star_color.to_numpy(float),
    )
    lo, hi = _TRANSPARENCY_RANGE
    # Keep the brighter spots of each exposure (by estimated magnitude) whose transparency is sky,
    # the selection agMetrics.summarizeTransparency makes for the log.
    faintest = spots.groupby("agc_exposure_id").estimated_magnitude.transform(
        lambda m: np.nanquantile(m, _TRANSPARENCY_BRIGHT_FRACTION) if np.isfinite(m).any() else np.nan
    )
    bright = spots.estimated_magnitude.to_numpy(float) <= faintest.to_numpy(float)
    inRange = (transparency > lo) & (transparency < hi)
    spots["transparency"] = np.where(bright & inRange, transparency, np.nan)

    # mean / median / std skip NaN and std is ddof=1, so an unresolved or faint/out-of-range spot
    # (NaN) drops and a lone spot gives a NaN sigma; the exposure columns are constant within a group.
    out = (
        spots.groupby("agc_exposure_id")
        .agg(
            pfs_visit_id=("pfs_visit_id", "first"),
            taken_at=("taken_at", "first"),
            seeing_mean=("seeing", "mean"),
            seeing_median=("seeing", "median"),
            seeing_sigma=("seeing", "std"),
            transparency_mean=("transparency", "mean"),
            transparency_median=("transparency", "median"),
            transparency_sigma=("transparency", "std"),
        )
        .reset_index()
    )
    out["pfs_visit_id"] = out.pfs_visit_id.astype(int)
    return out[_COLUMNS]


def writeConditions(visits, opdb, qadb, wavelengthRef=None):
    """Compute the visits' conditions from opdb and upsert them to qadb.

    Parameters
    ----------
    visits : iterable of `int`
        The visits.
    opdb, qadb : `pfs.utils.database.opdb.OpDB`, `pfs.utils.database.qadb.QaDB`
        The operations and QA databases.
    wavelengthRef : `float`, optional
        Stored in ``wavelength_ref``. The agMetrics method references no single wavelength (the
        colour correction is per star), so the default is NULL rather than a placeholder.

    Returns
    -------
    counts : `tuple` of `int`
        The seeing and transparency rows written.
    """
    cond = exposureConditions(readConditionSpots(opdb, visits))
    if len(cond) == 0:
        return 0, 0
    keys = ["pfs_visit_id", "agc_exposure_id"]
    seeing = cond[[*keys, "seeing_mean", "seeing_median", "seeing_sigma", "taken_at"]].assign(
        wavelength_ref=wavelengthRef
    )
    transp = cond[
        [*keys, "transparency_mean", "transparency_median", "transparency_sigma", "taken_at"]
    ].assign(wavelength_ref=wavelengthRef)
    return upsert(qadb, "seeing_agc_exposure", seeing, keys), upsert(
        qadb, "transparency_agc_exposure", transp, keys
    )
