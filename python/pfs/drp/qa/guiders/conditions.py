"""Per-AG-exposure seeing and sky transparency for qadb, from good matches via pfs.utils.agMetrics.

The `seeing_agc_exposure` and `transparency_agc_exposure` values the observatory tool
(pfs_obsproc_qa_tools ``obsCondition``) fills today, recomputed from the **good matches** only
(``agc_match.flags = GOOD_MATCH`` and a clean detection, as the tool reads them) with the shared
`pfs.utils.agMetrics` calc: the calibrated seeing (`agMetrics.calibratedSeeing`) and the
magnitude-based transparency (`agMetrics.magnitudesToTransparency`). A driver reads the spots from
opdb, aggregates per exposure and upserts the rows. Not yet wired into the QA pipeline.
"""

import numpy as np
import pandas as pd

from pfs.drp.qa.qadb.writer import upsert
from pfs.utils import agMetrics
from pfs.utils.datamodel.ag import SourceMatchingFlags

_GOOD_MATCH = int(SourceMatchingFlags.GOOD_MATCH)
_TRANSPARENCY_RANGE = (
    0.0,
    2.0,
)  # a star outside this is a mismatch, not sky (agMetrics.summarizeTransparency)

#: One row per good-match, clean-detection spot of each AG exposure: the raw second moments (scaled
#: per exposure, corrected below), the spot and catalogue magnitudes and colour, the telescope
#: altitude and the exposure version. The filter is the tool's: GOOD_MATCH and ``agc_data.flags <= 1``.
CONDITIONS_SQL = """
SELECT agc_exposure.pfs_visit_id, agc_exposure.agc_exposure_id, agc_exposure.taken_at,
       agc_exposure.altitude, agc_exposure.version_actor,
       agc_data.central_image_moment_20_pix AS mxx, agc_data.central_image_moment_02_pix AS myy,
       agc_data.estimated_magnitude,
       pfs_design_agc.guide_star_magnitude, pfs_design_agc.guide_star_color
FROM agc_exposure
JOIN agc_data ON agc_data.agc_exposure_id = agc_exposure.agc_exposure_id
JOIN agc_match ON agc_match.agc_exposure_id = agc_data.agc_exposure_id
   AND agc_match.agc_camera_id = agc_data.agc_camera_id AND agc_match.spot_id = agc_data.spot_id
LEFT JOIN pfs_visit ON pfs_visit.pfs_visit_id = agc_exposure.pfs_visit_id
LEFT JOIN pfs_design_agc ON pfs_design_agc.pfs_design_id = pfs_visit.pfs_design_id
   AND pfs_design_agc.guide_star_id = agc_match.guide_star_id
WHERE agc_exposure.pfs_visit_id = ANY(:visits)
  AND agc_match.flags = :goodMatch AND COALESCE(agc_data.flags, 0) <= 1
"""


def readConditionsSpots(visits, opdb):
    """Return the good-match spots of the visits' AG exposures (`CONDITIONS_SQL`)."""
    return opdb.query_dataframe(CONDITIONS_SQL, params={"visits": list(visits), "goodMatch": _GOOD_MATCH})


def exposureConditions(spots: pd.DataFrame) -> pd.DataFrame:
    """Per-AG-exposure seeing (zenith, arcsec) and sky transparency from its good-match spots.

    Seeing is `agMetrics.calibratedSeeing` of each spot's FWHM, then the mean / median / sigma over
    the exposure's spots; a spot the AG cannot resolve (NaN seeing) drops out. Transparency is
    `agMetrics.magnitudesToTransparency` per spot, over the stars between `_TRANSPARENCY_RANGE`.
    The second moments are corrected for the ics_agccActor ``"git"`` halving before the FWHM.

    Returns one row per ``agc_exposure_id`` with ``pfs_visit_id``, ``taken_at`` and the
    ``seeing_*`` / ``transparency_*`` columns of the qadb tables (``wavelength_ref`` is added by
    `writeConditions`). Empty (with those columns) for no spots.
    """
    columns = [
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
    if len(spots) == 0:
        return pd.DataFrame(columns=columns)
    spots = spots.copy()
    vpm = np.where(spots.version_actor.to_numpy() == "git", agMetrics.VARIANCE_PER_AGC_MOMENT, 1.0)
    fwhm = agMetrics.momentsToFwhm(
        spots.mxx.to_numpy(float), spots.myy.to_numpy(float), variancePerMoment=vpm
    )
    spots["seeing"] = agMetrics.calibratedSeeing(fwhm, spots.altitude.to_numpy(float))
    transparency = agMetrics.magnitudesToTransparency(
        spots.estimated_magnitude.to_numpy(float),
        spots.guide_star_magnitude.to_numpy(float),
        spots.guide_star_color.to_numpy(float),
    )
    lo, hi = _TRANSPARENCY_RANGE
    spots["transparency"] = np.where((transparency > lo) & (transparency < hi), transparency, np.nan)

    def stats(values):
        v = values[np.isfinite(values)]
        return (v.mean(), v.median(), v.std()) if len(v) else (np.nan, np.nan, np.nan)

    def one(group):
        s_mean, s_med, s_sig = stats(group.seeing)
        t_mean, t_med, t_sig = stats(group.transparency)
        return pd.Series(
            {
                "pfs_visit_id": group.pfs_visit_id.iloc[0],
                "taken_at": group.taken_at.iloc[0],
                "seeing_mean": s_mean,
                "seeing_median": s_med,
                "seeing_sigma": s_sig,
                "transparency_mean": t_mean,
                "transparency_median": t_med,
                "transparency_sigma": t_sig,
            }
        )

    out = spots.groupby("agc_exposure_id").apply(one, include_groups=False).reset_index()
    out["pfs_visit_id"] = out.pfs_visit_id.astype(int)
    return out[columns]


def writeConditions(visits, opdb, qadb, wavelengthRef=float("nan")):
    """Compute the visits' conditions from opdb and upsert them to qadb. Returns (n_seeing, n_transparency)."""
    cond = exposureConditions(readConditionsSpots(visits, opdb))
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
