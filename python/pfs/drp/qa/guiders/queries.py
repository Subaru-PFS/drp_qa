"""Readers for AG data from the opdb and the butler.

This is the only module that touches a database or a butler.

Each opdb reader takes a `pfs.utils.database.opdb.OpDB` as its first argument
and binds its parameters rather than formatting them into the SQL. Readers
return a DataFrame with a fresh index. `readAgcData` returns positions in
hardware coordinates (see `pfs.drp.qa.guiders.coordinates`), converted once as
they are read.

Butler readers take a Gen3 butler whose default collections include the raws.
"""

import logging
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from pfs.drp.qa.guiders.coordinates import opdbToHardware

__all__ = [
    "AGC_DATA_COLUMNS",
    "find_W_M2OFF3",
    "readAGCStars",
    "readAgcData",
    "readInstPa",
    "readPfsDesign",
    "readRawMetadata",
    "readSpSInfo",
    "readTelStatus",
]

_log = logging.getLogger(__name__)

# The columns of readAgcData, in order, with their units and sources.
AGC_DATA_COLUMNS = {
    "pfs_visit_id": "PFS visit",
    "agc_exposure_id": "AG exposure",
    "agc_camera_id": "AG camera, 0-5 (AG1 is 0)",
    "spot_id": "spot in the AG exposure and camera",
    "guide_star_id": "guide star the spot was matched to",
    "taken_at": "time of the AG exposure (HST), written by agcc as it starts the exposure",
    "agc_exptime": "AG exposure time (s)",
    "altitude": "telescope altitude (deg)",
    "azimuth": "telescope azimuth (deg)",
    "insrot": "instrument rotator angle (deg)",
    "adc_pa": "ADC position angle (deg)",
    "m2_pos3": "hexapod position (mm)",
    "m2_off3": "hexapod focus offset (mm), from tel_status, or from W_M2OFF3 with a butler",
    "tel_ra": "telescope target RA (deg), from tel_status",
    "tel_dec": "telescope target Dec (deg), from tel_status",
    "inst_pa": "INST-PA (deg) from the raw headers; NaN without a butler",
    "exptime": "spectrograph exposure time (s), averaged over cameras; NaN with no SpS exposure",
    "shutter_open": "1 if the spectrograph shutters were open, 0 if closed, 2 with no SpS exposure",
    "agc_nominal_x_mm": "where the guider expected the star (mm, hardware coordinates)",
    "agc_nominal_y_mm": "where the guider expected the star (mm, hardware coordinates)",
    "agc_center_x_mm": "measured position of the star (mm, hardware coordinates)",
    "agc_center_y_mm": "measured position of the star (mm, hardware coordinates)",
    "agc_match_flags": "agc_match.flags (SourceMatchingFlags); GOOD_MATCH (1) alone for a valid match",
    "agc_data_flags": "agc_data.flags (SourceDetectionFlags); if NULL, RIGHT from centroid_x_pix, as agActor",
    "guide_star_flag": "pfs_design_agc.guide_star_flag (SourceCatalogFlags); 0 if missing",
    "image_moment_00_pix": "zeroth image moment (flux)",
    "centroid_x_pix": "centroid on the AG detector (pix)",
    "centroid_y_pix": "centroid on the AG detector (pix)",
    "mxx": "central second moment, xx (pix^2)",
    "myy": "central second moment, yy (pix^2)",
    "mxy": "central second moment, xy (pix^2)",
    "peak_pixel_x_pix": "peak pixel (pix)",
    "peak_pixel_y_pix": "peak pixel (pix)",
    "peak_intensity": "peak pixel value",
    "background": "background level",
    "estimated_magnitude": "estimated magnitude (mag)",
    "guide_ra": "designed field center RA (deg)",
    "guide_dec": "designed field center Dec (deg)",
    "guide_pa": "designed field position angle (deg)",
    "guide_delta_ra": "guide offset in RA (arcsec)",
    "guide_delta_dec": "guide offset in Dec (arcsec)",
    "guide_delta_insrot": "guide offset in rotator angle (arcsec)",
    "guide_delta_scale": "guide offset in scale",
    "guide_delta_azimuth": "guide offset in azimuth (arcsec); the opdb's guide_delta_az",
    "guide_delta_altitude": "guide offset in altitude (arcsec); the opdb's guide_delta_el",
    "guide_delta_z": "focus offset (mm)",
    **{f"guide_delta_z{i}": f"focus offset for AG{i} (mm)" for i in range(1, 7)},
}

# One row per matched spot. The SpS exposures are aggregated per visit before
# the join, so the join doesn't repeat rows per camera.
_AGC_DATA_SQL = """
SELECT
    agc_exposure.pfs_visit_id,
    agc_exposure.agc_exposure_id,
    agc_data.agc_camera_id,
    agc_data.spot_id,
    agc_match.guide_star_id,
    agc_exposure.taken_at,
    agc_exposure.agc_exptime,
    agc_exposure.altitude,
    agc_exposure.azimuth,
    agc_exposure.insrot,
    agc_exposure.adc_pa,
    agc_exposure.m2_pos3,
    sps.exptime,
    CASE
        WHEN sps.pfs_visit_id IS NULL THEN 2
        WHEN agc_exposure.taken_at BETWEEN sps.time_exp_start AND sps.time_exp_end THEN 1
        ELSE 0
    END AS shutter_open,
    agc_match.agc_nominal_x_mm,
    agc_match.agc_nominal_y_mm,
    agc_match.agc_center_x_mm,
    agc_match.agc_center_y_mm,
    agc_match.flags AS agc_match_flags,
    -- Older rows have no flags; infer RIGHT from the centroid, as agActor's query_agc_data does.
    COALESCE(agc_data.flags, CAST(agc_data.centroid_x_pix >= 511.5 + 24 AS INTEGER)) AS agc_data_flags,
    pfs_design_agc.guide_star_flag,
    agc_data.image_moment_00_pix,
    agc_data.centroid_x_pix,
    agc_data.centroid_y_pix,
    agc_data.central_image_moment_20_pix AS mxx,
    agc_data.central_image_moment_02_pix AS myy,
    agc_data.central_image_moment_11_pix AS mxy,
    agc_data.peak_pixel_x_pix,
    agc_data.peak_pixel_y_pix,
    agc_data.peak_intensity,
    agc_data.background,
    agc_data.estimated_magnitude,
    agc_guide_offset.guide_ra,
    agc_guide_offset.guide_dec,
    agc_guide_offset.guide_pa,
    agc_guide_offset.guide_delta_ra,
    agc_guide_offset.guide_delta_dec,
    agc_guide_offset.guide_delta_insrot,
    agc_guide_offset.guide_delta_scale,
    agc_guide_offset.guide_delta_az AS guide_delta_azimuth,
    agc_guide_offset.guide_delta_el AS guide_delta_altitude,
    agc_guide_offset.guide_delta_z,
    agc_guide_offset.guide_delta_z1,
    agc_guide_offset.guide_delta_z2,
    agc_guide_offset.guide_delta_z3,
    agc_guide_offset.guide_delta_z4,
    agc_guide_offset.guide_delta_z5,
    agc_guide_offset.guide_delta_z6
FROM agc_exposure
JOIN agc_data ON agc_data.agc_exposure_id = agc_exposure.agc_exposure_id
JOIN agc_match ON agc_match.agc_exposure_id = agc_data.agc_exposure_id
    AND agc_match.agc_camera_id = agc_data.agc_camera_id
    AND agc_match.spot_id = agc_data.spot_id
LEFT JOIN agc_guide_offset ON agc_guide_offset.agc_exposure_id = agc_exposure.agc_exposure_id
LEFT JOIN (
    SELECT
        pfs_visit_id,
        avg(exptime) AS exptime,
        min(time_exp_start) AS time_exp_start,
        min(time_exp_end) AS time_exp_end
    FROM sps_exposure
    WHERE pfs_visit_id = ANY(:visits)
    GROUP BY pfs_visit_id
) AS sps ON sps.pfs_visit_id = agc_exposure.pfs_visit_id
LEFT JOIN pfs_visit ON pfs_visit.pfs_visit_id = agc_exposure.pfs_visit_id
LEFT JOIN pfs_design_agc ON pfs_design_agc.pfs_design_id = pfs_visit.pfs_design_id
    AND pfs_design_agc.guide_star_id = agc_match.guide_star_id
WHERE agc_exposure.pfs_visit_id = ANY(:visits)
"""

# Every AG exposure, including those with no matched spots.
_AGC_EXPOSURES_SQL = """
SELECT pfs_visit_id, agc_exposure_id, m2_pos3
FROM agc_exposure
WHERE pfs_visit_id = ANY(:visits)
"""

# The AG actor (caller agcc) writes one tel_status row per AG exposure.
_AGC_TEL_STATUS_SQL = """
SELECT pfs_visit_id, status_sequence_id, m2_off3, tel_ra, tel_dec
FROM tel_status
WHERE pfs_visit_id = ANY(:visits) AND caller = 'agcc'
"""

_TEL_STATUS_COLUMNS = ("m2_off3", "tel_ra", "tel_dec")
_AGC_DATA_ORDER = ["agc_exposure_id", "agc_camera_id", "spot_id"]


def _checkOpdb(opdb) -> None:
    """Raise a helpful TypeError unless ``opdb`` looks like an `OpDB`."""
    if not callable(getattr(opdb, "query_dataframe", None)):
        raise TypeError(
            "opdb must be a pfs.utils.database.opdb.OpDB (e.g. OpDB(host=..., user=...)), "
            f"not {type(opdb).__name__}"
        )


def _visitList(visits: int | Iterable[int]) -> list[int]:
    """Return ``visits`` as a sorted list of unique Python ints.

    Python ints because psycopg can't bind NumPy integers.
    """
    if isinstance(visits, int | np.integer):
        visits = [visits]
    visits = sorted({int(v) for v in visits})
    if not visits:
        raise ValueError("No visits given")

    return visits


def _pairTelStatus(agcExposures: pd.DataFrame, telStatus: pd.DataFrame) -> pd.DataFrame:
    """Pair each visit's AG exposures with its AG tel_status rows, by order.

    The opdb has no key linking tel_status to agc_exposure. Within a visit
    both agc_exposure_id and status_sequence_id increase, and the AG actor
    writes one tel_status row per exposure, so the n-th of each belong
    together. Where a visit has more of one than the other, the extra rows at
    the end are dropped, with a warning.

    Parameters
    ----------
    agcExposures : `pandas.DataFrame`
        ``pfs_visit_id`` and ``agc_exposure_id`` of every AG exposure.
    telStatus : `pandas.DataFrame`
        ``pfs_visit_id``, ``status_sequence_id`` and `_TEL_STATUS_COLUMNS`
        of the AG actor's tel_status rows.

    Returns
    -------
    paired : `pandas.DataFrame`
        ``agc_exposure_id`` and `_TEL_STATUS_COLUMNS` (float).
    """
    agcExposures = agcExposures.sort_values("agc_exposure_id")
    telStatus = telStatus.sort_values(["pfs_visit_id", "status_sequence_id"])

    pairs = []
    for visit, exposures in agcExposures.groupby("pfs_visit_id", sort=True):
        status = telStatus[telStatus.pfs_visit_id == visit]
        nExp, nStatus = len(exposures), len(status)
        if nExp != nStatus:
            _log.warning(
                "pfs_visit_id %d: %d AG exposures but %d AG rows in tel_status; dropping the last %d %s",
                visit,
                nExp,
                nStatus,
                abs(nExp - nStatus),
                "AG exposures" if nExp > nStatus else "tel_status rows",
            )
        n = min(nExp, nStatus)
        pair = status[list(_TEL_STATUS_COLUMNS)].iloc[:n].reset_index(drop=True)
        pair.insert(0, "agc_exposure_id", exposures.agc_exposure_id.iloc[:n].to_numpy())
        pairs.append(pair)

    columns = ["agc_exposure_id", *_TEL_STATUS_COLUMNS]
    paired = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=columns)
    # A column that is all NULL comes back as object (None).
    return paired.astype({"agc_exposure_id": int, **dict.fromkeys(_TEL_STATUS_COLUMNS, float)})


def readAgcData(
    opdb,
    visits: int | Iterable[int],
    *,
    butler=None,
    rawDataId: Mapping | None = None,
) -> pd.DataFrame:
    """Read the AG measurements of the guide stars in some visits.

    This is the one star-level reader: one row per spot that the guider
    matched to a guide star, in every AG exposure of the visits, whatever the
    match's flags.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The operational database.
    visits : `int` or iterable of `int`
        The ``pfs_visit_id`` values to read.
    butler : `lsst.daf.butler.Butler`, optional
        A butler with the visits' raws. With it, ``inst_pa`` is read from the
        raw headers, and ``m2_off3`` is filled for visits with none in
        tel_status (before 2025-03-21) from the header's ``W_M2OFF3`` plus
        the change in ``m2_pos3`` since the visit's first AG exposure.
    rawDataId : `dict`, optional
        Which raw of each visit to read INST-PA from; see `readInstPa`.

    Returns
    -------
    agcData : `pandas.DataFrame`
        The columns of `AGC_DATA_COLUMNS`, in order, sorted by
        ``agc_exposure_id``, ``agc_camera_id`` and ``spot_id``, with a fresh
        index. Positions are in hardware coordinates. With no AG data the
        result is empty, with the same columns.

    Raises
    ------
    TypeError
        If ``opdb`` isn't an `OpDB`.
    ValueError
        If ``visits`` is empty.

    Notes
    -----
    ``m2_off3``, ``tel_ra`` and ``tel_dec`` come from tel_status, which has
    no key linking it to agc_exposure; see `_pairTelStatus`.
    """
    _checkOpdb(opdb)
    visits = _visitList(visits)
    params = {"visits": visits}

    agcData = opdb.query_dataframe(_AGC_DATA_SQL, params=params)
    missing = sorted(set(visits) - set(agcData.pfs_visit_id))
    if missing:
        _log.warning("No AG data for pfs_visit_id %s", ", ".join(str(v) for v in missing))
    if agcData.empty:
        return opdbToHardware(pd.DataFrame(columns=list(AGC_DATA_COLUMNS)))

    agcExposures = opdb.query_dataframe(_AGC_EXPOSURES_SQL, params=params)
    telStatus = _pairTelStatus(agcExposures, opdb.query_dataframe(_AGC_TEL_STATUS_SQL, params=params))
    # A left merge, so a star isn't dropped when its exposure has no tel_status row.
    agcData = agcData.merge(telStatus, on="agc_exposure_id", how="left", validate="many_to_one")

    agcData["guide_star_flag"] = agcData.guide_star_flag.fillna(0).astype(int)
    agcData["inst_pa"] = np.nan
    if butler is not None:
        agcData = _addButlerColumns(agcData, agcExposures, butler, rawDataId)

    agcData = agcData[list(AGC_DATA_COLUMNS)].sort_values(_AGC_DATA_ORDER, ignore_index=True)

    return opdbToHardware(agcData)


def _addButlerColumns(
    agcData: pd.DataFrame, agcExposures: pd.DataFrame, butler, rawDataId: Mapping | None
) -> pd.DataFrame:
    """Return a copy of ``agcData`` with ``inst_pa``, and missing ``m2_off3``, from the raws.

    ``agcExposures`` has the ``pfs_visit_id``, ``agc_exposure_id`` and
    ``m2_pos3`` of every AG exposure, as M2 may move before the first
    exposure with a matched spot.
    """
    agcData = agcData.copy()
    instPa = readInstPa(butler, agcData.pfs_visit_id.unique(), rawDataId)
    agcData["inst_pa"] = agcData.pfs_visit_id.map(instPa).astype(float)

    # m2_pos3 at each visit's first AG exposure (with an m2_pos3).
    m2Pos3First = agcExposures.sort_values("agc_exposure_id").groupby("pfs_visit_id").m2_pos3.first()
    for visit, rows in agcData.groupby("pfs_visit_id"):
        if rows.m2_off3.notna().any():
            continue
        moved = rows.m2_pos3 - m2Pos3First[visit]
        agcData.loc[rows.index, "m2_off3"] = find_W_M2OFF3(butler, visit) + moved

    return agcData


# Butler


def readRawMetadata(butler, visit: int, **dataId):
    """Return the header of a raw from a visit, or None if it has none.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A butler whose default collections include the raws.
    visit : `int`
        The visit.
    **dataId
        Further data ID keys to match, e.g. ``arm="r", spectrograph=1``.

    Returns
    -------
    metadata : `lsst.daf.base.PropertyList` or `None`
        The header of the first matching raw.
    """
    refs = list(butler.registry.queryDatasets("raw", visit=int(visit), **dataId))
    if not refs:
        return None

    return butler.get("raw.metadata", refs[0].dataId)


def readInstPa(butler, visits: Iterable[int], dataId: Mapping | None = None) -> pd.Series:
    """Read INST-PA from the raw headers of some visits.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A butler whose default collections include the raws.
    visits : iterable of `int`
        The visits.
    dataId : `dict`, optional
        Which raw of each visit to read; default ``spectrograph=1, arm="r"``.
        If a visit has no such raw, any of its raws is used.

    Returns
    -------
    instPa : `pandas.Series`
        INST-PA (deg), indexed by visit; NaN for a visit with no raws.
    """
    if dataId is None:
        dataId = {"spectrograph": 1, "arm": "r"}

    instPa = {}
    for visit in _visitList(visits):
        md = readRawMetadata(butler, visit, **dataId)
        if md is None:  # e.g. that camera wasn't taking data
            md = readRawMetadata(butler, visit)
        if md is None:
            _log.warning("pfs_visit_id %d: no raws; INST-PA is NaN", visit)
        instPa[visit] = np.nan if md is None else float(md["INST-PA"])

    return pd.Series(instPa, name="inst_pa", dtype=float)


def find_W_M2OFF3(butler, visit: int, nTry: int = 100, key: str = "W_M2OFF3") -> float:
    """Return a header value from the latest raw at or before a visit.

    Visits that only used the AGs have no raws, so this searches back from
    ``visit`` for one that has.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        A butler whose default collections include the raws.
    visit : `int`
        The first visit to try.
    nTry : `int`
        How many visits to try, counting down from ``visit``.
    key : `str`
        The header keyword.

    Returns
    -------
    value : `float`
        The value of ``key`` (for ``W_M2OFF3``, the M2 focus offset in mm).

    Raises
    ------
    RuntimeError
        If none of the visits has a raw.
    """
    if butler is None:
        raise RuntimeError(f"A butler is needed to find {key}")

    visit = int(visit)
    for v in range(visit, visit - nTry, -1):
        md = readRawMetadata(butler, v)
        if md is not None:
            return md[key]

    raise RuntimeError(f"No raw to read {key} from in visit {visit} or the {nTry - 1} visits before it")


# Other opdb readers


def readAGCStars(opdb, pfs_design_id: int, pfs_visit_id: int = 0) -> pd.DataFrame:
    """Read the guide stars of a design, and optionally of a visit's config.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The operational database.
    pfs_design_id : `int`
        The design.
    pfs_visit_id : `int`
        If positive, also read this visit's config, if it converged: its
        field center and the stars' final positions on the AG detectors.

    Returns
    -------
    stars : `pandas.DataFrame`
        One row per guide star, with ``pfs_design_id``,
        ``agc_camera_id``, ``guide_star_id``, ``guide_star_ra``,
        ``guide_star_dec`` (deg, the design's), ``guide_star_pm_ra``,
        ``guide_star_pm_dec`` (mas/yr) and ``guide_star_parallax`` (mas). Without a visit, also the
        design's field center ``ra_center_designed``, ``dec_center_designed``
        and ``pa_designed`` (deg). With one, ``pfs_visit_id``,
        ``ra_center_config``, ``dec_center_config``, ``pa_config`` (deg) and
        ``agc_final_x_pix``, ``agc_final_y_pix``.
    """
    _checkOpdb(opdb)
    params = {"pfs_design_id": int(pfs_design_id)}
    if pfs_visit_id <= 0:
        sql = """
        SELECT
            pfs_design.pfs_design_id, pfs_design_agc.agc_camera_id,
            pfs_design_agc.guide_star_id, guide_star_ra, guide_star_dec,
            guide_star_pm_ra, guide_star_pm_dec, guide_star_parallax,
            ra_center_designed, dec_center_designed, pa_designed
        FROM pfs_design
        JOIN pfs_design_agc ON pfs_design_agc.pfs_design_id = pfs_design.pfs_design_id
        WHERE pfs_design.pfs_design_id = :pfs_design_id
        ORDER BY pfs_design_agc.guide_star_id
        """
    else:
        params["pfs_visit_id"] = int(pfs_visit_id)
        sql = """
        SELECT DISTINCT
            pfs_design.pfs_design_id, pfs_config_sps.pfs_visit_id, pfs_design_agc.agc_camera_id,
            pfs_design_agc.guide_star_id, pfs_design_agc.guide_star_ra, pfs_design_agc.guide_star_dec,
            guide_star_pm_ra, guide_star_pm_dec, guide_star_parallax,
            ra_center_config, dec_center_config, pa_config,
            agc_final_x_pix, agc_final_y_pix
        FROM pfs_design
        JOIN pfs_design_agc ON pfs_design_agc.pfs_design_id = pfs_design.pfs_design_id
        JOIN pfs_config_sps ON pfs_config_sps.pfs_visit_id = :pfs_visit_id
        JOIN pfs_config ON pfs_config.visit0 = pfs_config_sps.visit0
            AND pfs_config.pfs_design_id = :pfs_design_id
        JOIN pfs_config_agc ON pfs_config_agc.pfs_design_id = pfs_design.pfs_design_id
            AND pfs_config_agc.guide_star_id = pfs_design_agc.guide_star_id
            AND pfs_config_agc.visit0 = pfs_config_sps.visit0
        WHERE pfs_design.pfs_design_id = :pfs_design_id
            AND converg_num_iter IS NOT NULL
        ORDER BY pfs_design_agc.guide_star_id
        """

    return opdb.query_dataframe(sql, params=params).reset_index(drop=True)


def readPfsDesign(opdb, pfs_visit_id: int) -> pd.DataFrame:
    """Read the design of a visit.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The operational database.
    pfs_visit_id : `int`
        The visit.

    Returns
    -------
    design : `pandas.DataFrame`
        ``pfs_design_id`` and ``design_name``; one row, or none for an
        unknown visit.
    """
    _checkOpdb(opdb)
    sql = """
    SELECT pfs_visit.pfs_design_id, pfs_design.design_name
    FROM pfs_visit
    JOIN pfs_design ON pfs_design.pfs_design_id = pfs_visit.pfs_design_id
    WHERE pfs_visit.pfs_visit_id = :pfs_visit_id
    """

    return opdb.query_dataframe(sql, params={"pfs_visit_id": int(pfs_visit_id)}).reset_index(drop=True)


def readSpSInfo(
    opdb,
    taken_after=None,
    min_exptime: float = 0,
    exp_type: str | None = "object",
    limit: int = 0,
    windowed: bool = False,
    showQuery: bool = False,
) -> pd.DataFrame:
    """Read a summary of spectrograph visits.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The operational database.
    taken_after : `str` or `datetime.datetime`, optional
        Only visits whose exposures started after this time (HST, like every
        opdb time except the planned times in pfs_design and pfs_config).
    min_exptime : `float`
        Only exposures longer than this (s), if positive.
    exp_type : `str` or `None`
        Only visits of this type; any if `None`.
    limit : `int`
        If positive, read at most this many rows. The limit applies to the
        rows of the query, of which there are several per visit, so you may
        get fewer visits.
    windowed : `bool`
        Only visits taken with windowed reads.
    showQuery : `bool`
        Print the SQL and its parameters.

    Returns
    -------
    visits : `pandas.DataFrame`
        One row per visit: ``pfs_visit_id``, ``taken_at`` (HST), ``exptime`` (s),
        ``exp_type``, ``altitude``, ``azimuth``, ``insrot`` (deg),
        ``group_id``, ``group_name`` and ``design_name``. Times and angles are
        averaged over the cameras and tel_status rows. ``group_id`` and
        ``group_name`` are missing for a visit in no sequence.
    """
    _checkOpdb(opdb)
    where = []
    params = {}
    if taken_after is not None:
        where.append("sps_exposure.time_exp_start > CAST(:taken_after AS timestamp)")
        params["taken_after"] = taken_after
    if min_exptime > 0:
        where.append("sps_exposure.exptime > :min_exptime")
        params["min_exptime"] = float(min_exptime)
    if exp_type is not None:
        where.append("sps_visit.exp_type = :exp_type")
        params["exp_type"] = exp_type
    if windowed:
        where.append("iic_sequence.sequence_type LIKE '%windowed'")

    sql = f"""
    SELECT DISTINCT
        sps_exposure.pfs_visit_id AS pfs_visit_id, sps_exposure.time_exp_start AS taken_at, exptime,
        exp_type, altitude, azimuth, insrot, design_name,
        sequence_group.group_id, sequence_group.group_name
    FROM sps_exposure
    JOIN pfs_visit ON pfs_visit.pfs_visit_id = sps_exposure.pfs_visit_id
    JOIN pfs_design ON pfs_design.pfs_design_id = pfs_visit.pfs_design_id
    JOIN sps_visit ON sps_visit.pfs_visit_id = sps_exposure.pfs_visit_id
    LEFT JOIN tel_status ON tel_status.pfs_visit_id = sps_exposure.pfs_visit_id
    LEFT JOIN visit_set ON visit_set.pfs_visit_id = sps_exposure.pfs_visit_id
    LEFT JOIN iic_sequence ON iic_sequence.iic_sequence_id = visit_set.iic_sequence_id
    LEFT JOIN sequence_group ON sequence_group.group_id = iic_sequence.group_id
    {"WHERE " + " AND ".join(where) if where else ""}
    {"LIMIT :limit" if limit > 0 else ""}
    """
    if limit > 0:
        params["limit"] = int(limit)
    if showQuery:
        print(sql, params)

    visits = opdb.query_dataframe(sql, params=params)

    # The cameras' start times differ slightly, so group by visit.
    return visits.groupby("pfs_visit_id", as_index=False).agg(
        taken_at=("taken_at", "mean"),
        exptime=("exptime", "mean"),
        exp_type=("exp_type", "first"),
        altitude=("altitude", "mean"),
        azimuth=("azimuth", "mean"),
        insrot=("insrot", "mean"),
        group_id=("group_id", "first"),
        group_name=("group_name", "first"),
        design_name=("design_name", "first"),
    )


def readTelStatus(opdb, pfs_visit_id: int | Iterable[int]) -> pd.DataFrame:
    """Read the telescope status rows of one or more visits.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The operational database.
    pfs_visit_id : `int` or iterable of `int`
        The visits.

    Returns
    -------
    telStatus : `pandas.DataFrame`
        Every column of tel_status, sorted by ``pfs_visit_id`` and
        ``status_sequence_id``. Positions are as the opdb has them (it has
        none in the AG frame).
    """
    _checkOpdb(opdb)
    sql = """
    SELECT *
    FROM tel_status
    WHERE pfs_visit_id = ANY(:visits)
    ORDER BY pfs_visit_id, status_sequence_id
    """

    return opdb.query_dataframe(sql, params={"visits": _visitList(pfs_visit_id)}).reset_index(drop=True)
