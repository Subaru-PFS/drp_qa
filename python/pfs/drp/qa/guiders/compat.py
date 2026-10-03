"""drp_stella's AG readers, as wrappers around `queries.readAgcData`.

``pfs.drp.stella.utils.guiders`` had four readers of the same AG measurements:
``readAGCPositionsForVisitByAgcExposureId``, ``readAgcDataFromOpdb``,
``readAGCStarsForVisitByPfsVisitId`` and
``readAGCStarsForVisitSetByPfsVisitId``. `pfs.drp.qa.guiders.queries.readAgcData`
replaces them. These wrappers keep their names, arguments and columns while
notebooks move to it, and each call raises a `DeprecationWarning`. Nothing in
`pfs.drp.qa.guiders` uses them.

They differ from the originals in that:

- ``opdb`` is a `pfs.utils.database.opdb.OpDB`, not a psycopg2 connection.
- Each row is one matched spot, sorted by exposure, camera and spot, with a
  fresh index. The positions readers averaged spots matched to the same star
  in an exposure.
- ``shutter_open`` uses the earliest start and end of all the spectrograph
  cameras. The star readers used camera 1 only, so without camera 1 they
  reported 2 (no exposure).
- A star whose exposure has no tel_status row is kept, with NaN ``m2_off3``.
- The columns are those of `pfs.drp.qa.guiders.queries.AGC_DATA_COLUMNS`
  plus each reader's own, so there are more of them.
"""

import warnings
from collections.abc import Iterable

import pandas as pd

from pfs.drp.qa.guiders.analysis import addImageSizes
from pfs.drp.qa.guiders.coordinates import _FRAME_ATTR, OPDB_Y_COLUMNS
from pfs.drp.qa.guiders.queries import readAgcData

__all__ = [
    "readAGCPositionsForVisitByAgcExposureId",
    "readAGCStarsForVisitByPfsVisitId",
    "readAGCStarsForVisitSetByPfsVisitId",
    "readAgcDataFromOpdb",
]


def _warn(name: str) -> None:
    warnings.warn(
        f"{name} is deprecated; use pfs.drp.qa.guiders.queries.readAgcData",
        DeprecationWarning,
        stacklevel=3,
    )


def _toOpdbFrame(agcData: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``agcData`` with the opdb's sign of y."""
    agcData = agcData.copy()
    for column in OPDB_Y_COLUMNS:
        agcData[column] = -agcData[column]
    agcData.attrs.pop(_FRAME_ATTR, None)

    return agcData


def _readStars(opdb, visits, flipToHardwareCoords: bool, useTraceRadius: bool, butler) -> pd.DataFrame:
    """Return the valid matches, with the star readers' extra columns."""
    stars = readAgcData(opdb, visits, butler=butler)
    stars = stars[stars.agc_match_flags == 1].reset_index(drop=True)
    stars["flags"] = stars.agc_data_flags
    stars["guide_delta_az"] = stars.guide_delta_azimuth
    stars["guide_delta_el"] = stars.guide_delta_altitude
    stars = addImageSizes(stars, useTraceRadius).rename(columns={"rms_pix": "rms", "fwhm_arcsec": "FWHM"})

    return stars if flipToHardwareCoords else _toOpdbFrame(stars)


def readAGCPositionsForVisitByAgcExposureId(opdb, pfs_visit_id: int, flipToHardwareCoords: bool):
    """Read the AG measurements of a visit, whatever their match flags.

    Deprecated: use `pfs.drp.qa.guiders.queries.readAgcData`.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The operational database.
    pfs_visit_id : `int`
        The visit.
    flipToHardwareCoords : `bool`
        Return positions in hardware coordinates, else the opdb's.

    Returns
    -------
    agcData : `pandas.DataFrame`
        As `readAgcData`; empty if the visit has no AG data.
    """
    _warn("readAGCPositionsForVisitByAgcExposureId")
    agcData = readAgcData(opdb, pfs_visit_id)

    return agcData if flipToHardwareCoords else _toOpdbFrame(agcData)


def readAgcDataFromOpdb(opdb, visits: Iterable[int], butler=None, dataId=None) -> pd.DataFrame:
    """Read the AG measurements of some visits, whatever their match flags.

    Deprecated: use `pfs.drp.qa.guiders.queries.readAgcData`.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The operational database.
    visits : iterable of `int`
        The visits.
    butler : `lsst.daf.butler.Butler`, optional
        A Gen3 butler, to read ``inst_pa`` from the raw headers.
    dataId : `dict`, optional
        Which raw to read INST-PA from; see `queries.readInstPa`.

    Returns
    -------
    agcData : `pandas.DataFrame`
        As `readAgcData`, but ``exptime`` is 0 rather than NaN with no SpS
        exposure, and there's no ``inst_pa`` without a butler.

    Raises
    ------
    RuntimeError
        If none of the visits has AG data.
    """
    _warn("readAgcDataFromOpdb")
    visits = list(visits)
    agcData = readAgcData(opdb, visits, butler=butler, rawDataId=dataId)
    if agcData.empty:
        raise RuntimeError(f"No AG data for visits {', '.join(str(v) for v in visits)}")

    agcData["exptime"] = agcData.exptime.fillna(0)
    if butler is None:
        agcData = agcData.drop(columns="inst_pa")

    return agcData


def readAGCStarsForVisitByPfsVisitId(
    opdb, pfs_visit_id: int, flipToHardwareCoords: bool = True, useTraceRadius: bool = True, butler=None
) -> pd.DataFrame:
    """Read the valid AG matches of a visit, with image sizes.

    Deprecated: use `pfs.drp.qa.guiders.queries.readAgcData`.

    Parameters
    ----------
    opdb : `pfs.utils.database.opdb.OpDB`
        The operational database.
    pfs_visit_id : `int`
        The visit.
    flipToHardwareCoords : `bool`
        Return positions in hardware coordinates, else the opdb's.
    useTraceRadius : `bool`
        Compute ``rms`` from the trace of the moments, sqrt((mxx + myy)/2),
        rather than their determinant, (mxx myy - mxy^2)^(1/4).
    butler : `lsst.daf.butler.Butler`, optional
        A Gen3 butler, to fill ``m2_off3`` before 2025-03-21 from the raws.

    Returns
    -------
    stars : `pandas.DataFrame`
        As `readAgcData`, only the rows with ``agc_match_flags == 1``, plus
        ``flags`` (``agc_data_flags``), ``guide_delta_az``,
        ``guide_delta_el``, ``rms`` (pix), ``FWHM`` (arcsec) and ``left``
        (not on the RIGHT half of the detector). Empty if the visit has
        none.
    """
    _warn("readAGCStarsForVisitByPfsVisitId")
    return _readStars(opdb, pfs_visit_id, flipToHardwareCoords, useTraceRadius, butler)


def readAGCStarsForVisitSetByPfsVisitId(
    opdb,
    visits: Iterable[int],
    flipToHardwareCoords: bool = True,
    useTraceRadius: bool = True,
    butler=None,
) -> pd.DataFrame:
    """Read the valid AG matches of some visits, with image sizes.

    Deprecated: use `pfs.drp.qa.guiders.queries.readAgcData`.

    Parameters are as `readAGCStarsForVisitByPfsVisitId`, but for a list of
    visits.

    Raises
    ------
    RuntimeError
        If none of the visits has a valid match.
    """
    _warn("readAGCStarsForVisitSetByPfsVisitId")
    visits = list(visits)
    stars = _readStars(opdb, visits, flipToHardwareCoords, useTraceRadius, butler)
    if stars.empty:
        raise RuntimeError(f"No AG stars in visits {', '.join(str(v) for v in visits)}")

    return stars
