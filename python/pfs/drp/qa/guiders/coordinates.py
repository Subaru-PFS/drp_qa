"""Frame, sign and unit conventions for AG data.

Frames
------
Positions are in hardware coordinates. At an instrument rotator angle of 0, +x
points to the telescope's Front and +y to its Opt side: the PFI frame that
pfs_utils's ``Subaru_POPT2_PFS`` documents, and the frame of
`pfs.utils.coordinates.coordinates.det2dp`. `AGC_CAMERA_CENTERS_MM` are in this
frame. The opdb's ``agc_center_y_mm`` and ``agc_nominal_y_mm`` follow
`pfs.utils.coordinates.CoordTransp.ag_pixel_to_pfimm` and have the opposite
sign of y, as do the PFI coordinates that pfs_utils's ``PFS.pfi2fp`` and
``PFS.fp2pfi`` take and return. The readers convert with `opdbToHardware` as
they read, and nothing downstream converts again.

`pfiToZenith` and `zenithToPfi` convert to and from the zenith frame, in which
gravity, and so flexure, doesn't depend on the rotator angle.

Signs
-----
An offset is always a center minus a reference:
``dx_<reference>_um = agc_center_x_mm - agc_<reference>_x_mm`` (in microns).
The references are listed in `REFERENCES`, and each has its own columns, so an
offset's name says what it is measured from.

Units
-----
Column names end in their unit: ``_mm``, ``_um``, ``_pix`` or ``_arcsec``.
Positions are in mm and offsets in microns. The opdb's angles (``altitude``,
``azimuth``, ``insrot``) are in degrees. Convert with the helpers here
(`mmToUm`, `pixToArcsec`, ...) rather than with bare factors.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from pfs.utils.coordinates.Subaru_POPT2_PFS import PFS

__all__ = [
    "AGC_CAMERA_CENTERS_MM",
    "AGC_PIXEL_SIZE_UM",
    "AGC_PLATE_SCALE_UM_PER_ARCSEC",
    "AGC_RING_RADIUS_MM",
    "GUIDER_FOCUS_UM_PER_M2_OFF3_MM",
    "OPDB_Y_COLUMNS",
    "REFERENCES",
    "addOffsets",
    "addReferencePositions",
    "arcsecToRad",
    "guiderFocusToM2Off3",
    "m2Off3ToGuiderFocus",
    "mmToUm",
    "offsetColumns",
    "opdbToHardware",
    "pfiToZenith",
    "pixToArcsec",
    "pixToMm",
    "pixToUm",
    "radToArcsec",
    "referenceColumns",
    "rotXY",
    "umToArcsec",
    "umToMm",
    "zenithToPfi",
]

# Approximate centers of the AG cameras (mm, hardware coordinates), indexed by
# agc_camera_id. agc_camera_id 0 is AG1.
AGC_CAMERA_CENTERS_MM = {
    0: (237.58, -0.50),
    1: (120.19, 212.49),
    2: (-120.02, 212.10),
    3: (-242.08, 2.00),
    4: (-122.58, -211.67),
    5: (119.23, -209.79),
}
AGC_RING_RADIUS_MM = float(np.mean(np.hypot(*np.array(list(AGC_CAMERA_CENTERS_MM.values())).T)))

# AG detector pixel size and plate scale, as per JEG.
AGC_PIXEL_SIZE_UM = 13.0
AGC_PLATE_SCALE_UM_PER_ARCSEC = 94.7

# Guider focus offset (microns) per mm of M2_OFF3. Origin unrecorded; it
# should come from pfs_utils.
GUIDER_FOCUS_UM_PER_M2_OFF3_MM = 800.0

# The opdb columns whose sign differs from hardware coordinates.
OPDB_Y_COLUMNS = ("agc_center_y_mm", "agc_nominal_y_mm")

# Marks a DataFrame that `opdbToHardware` has converted.
_FRAME_ATTR = "agcFrame"

REFERENCES = ("nominal", "nominal0", "nominal0_visit", "center0", "center0_visit")
_REFERENCE_STATS = ("mean", "median", "first", "last")


# Units
#
# These use NumPy ufuncs rather than * and /, so that they accept any
# array-like (a list too) and return a Series for a Series.


def mmToUm(x_mm: ArrayLike) -> ArrayLike:
    """Convert mm to microns."""
    return np.multiply(x_mm, 1e3)


def umToMm(x_um: ArrayLike) -> ArrayLike:
    """Convert microns to mm."""
    return np.multiply(x_um, 1e-3)


def pixToUm(x_pix: ArrayLike) -> ArrayLike:
    """Convert AG detector pixels to microns on the focal plane."""
    return np.multiply(x_pix, AGC_PIXEL_SIZE_UM)


def pixToMm(x_pix: ArrayLike) -> ArrayLike:
    """Convert AG detector pixels to mm on the focal plane."""
    return umToMm(pixToUm(x_pix))


def umToArcsec(x_um: ArrayLike) -> ArrayLike:
    """Convert microns on the focal plane to arcsec on the sky."""
    return np.divide(x_um, AGC_PLATE_SCALE_UM_PER_ARCSEC)


def pixToArcsec(x_pix: ArrayLike) -> ArrayLike:
    """Convert AG detector pixels to arcsec on the sky."""
    return umToArcsec(pixToUm(x_pix))


def radToArcsec(angle_rad: ArrayLike) -> ArrayLike:
    """Convert radians to arcsec."""
    return np.multiply(np.rad2deg(angle_rad), 3600)


def arcsecToRad(angle_arcsec: ArrayLike) -> ArrayLike:
    """Convert arcsec to radians."""
    return np.deg2rad(np.divide(angle_arcsec, 3600))


def guiderFocusToM2Off3(focus_um: ArrayLike) -> ArrayLike:
    """Convert a guider focus offset (microns) to M2_OFF3 (mm)."""
    return np.divide(focus_um, GUIDER_FOCUS_UM_PER_M2_OFF3_MM)


def m2Off3ToGuiderFocus(m2Off3_mm: ArrayLike) -> ArrayLike:
    """Convert M2_OFF3 (mm) to a guider focus offset (microns)."""
    return np.multiply(m2Off3_mm, GUIDER_FOCUS_UM_PER_M2_OFF3_MM)


# Frames


def opdbToHardware(agcData: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of AG data from the opdb in hardware coordinates.

    Negates whichever of `OPDB_Y_COLUMNS` are present. The result is marked
    as converted, so converting it again raises rather than flipping it back.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data as read from the opdb.

    Returns
    -------
    agcData : `pandas.DataFrame`
        A copy, in hardware coordinates.

    Raises
    ------
    ValueError
        If ``agcData`` has already been converted.
    """
    if agcData.attrs.get(_FRAME_ATTR) == "hardware":
        raise ValueError("agcData is already in hardware coordinates")

    agcData = agcData.copy()
    for column in OPDB_Y_COLUMNS:
        if column in agcData:
            agcData[column] = -agcData[column]
    agcData.attrs[_FRAME_ATTR] = "hardware"

    return agcData


def pfiToZenith(x_mm: ArrayLike, y_mm: ArrayLike, insrot_deg: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert positions or offsets on the PFI to the zenith frame.

    Parameters
    ----------
    x_mm, y_mm : array-like
        Positions or offsets in hardware coordinates (mm).
    insrot_deg : `float` or array-like
        Instrument rotator angle (degrees); an array is matched element by
        element.

    Returns
    -------
    dz_mm : `numpy.ndarray`
        Component towards the zenith (mm); the telescope focal plane's +y,
        "Rear".
    dp_mm : `numpy.ndarray`
        Component perpendicular to that (mm), positive towards the Opt
        Nasmyth platform; the telescope focal plane's +x.

    Notes
    -----
    This is pfs_utils's ``PFS.pfi2fp``, which expects y in the opdb
    convention. pfs_utils puts the rotator axis at the origin, so the
    conversion is a rotation (and a flip) that applies to offsets as well as
    to positions.
    """
    x_mm = np.asarray(x_mm, dtype=float)
    y_mm = np.asarray(y_mm, dtype=float)
    xfp, yfp = PFS().pfi2fp(x_mm, -y_mm, insrot_deg)

    return yfp, xfp


def zenithToPfi(dz_mm: ArrayLike, dp_mm: ArrayLike, insrot_deg: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert positions or offsets in the zenith frame to the PFI.

    The inverse of `pfiToZenith`.

    Parameters
    ----------
    dz_mm, dp_mm : array-like
        Components towards and perpendicular to the zenith (mm).
    insrot_deg : `float` or array-like
        Instrument rotator angle (degrees).

    Returns
    -------
    x_mm, y_mm : `numpy.ndarray`
        Positions or offsets in hardware coordinates (mm).
    """
    dz_mm = np.asarray(dz_mm, dtype=float)
    dp_mm = np.asarray(dp_mm, dtype=float)
    x_mm, y_mm = PFS().fp2pfi(dp_mm, dz_mm, insrot_deg)

    return x_mm, -y_mm


def rotXY(angle_rad: ArrayLike, x: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
    """Rotate (x, y) anticlockwise about the origin.

    Parameters
    ----------
    angle_rad : `float` or array-like
        Rotation angle (radians); positive takes +x towards +y.
    x, y : array-like
        Coordinates to rotate, in any unit.

    Returns
    -------
    x, y : array-like
        Rotated coordinates, in the same unit.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    return np.multiply(c, x) - np.multiply(s, y), np.multiply(s, x) + np.multiply(c, y)


# Signs


def _checkReference(reference: str) -> None:
    if reference not in REFERENCES:
        raise ValueError(f"Unknown reference {reference!r}; valid: {', '.join(REFERENCES)}")


def _positionColumns(name: str) -> tuple[str, str]:
    return f"agc_{name}_x_mm", f"agc_{name}_y_mm"


def referenceColumns(reference: str) -> tuple[str, str]:
    """Return the names of a reference's x and y position columns.

    Parameters
    ----------
    reference : `str`
        One of `REFERENCES`.

    Returns
    -------
    x, y : `str`
        ``agc_<reference>_x_mm`` and ``agc_<reference>_y_mm``.
    """
    _checkReference(reference)
    return _positionColumns(reference)


def offsetColumns(reference: str) -> tuple[str, str]:
    """Return the names of the x and y offset columns from a reference.

    Parameters
    ----------
    reference : `str`
        One of `REFERENCES`.

    Returns
    -------
    dx, dy : `str`
        ``dx_<reference>_um`` and ``dy_<reference>_um``.
    """
    _checkReference(reference)
    return f"dx_{reference}_um", f"dy_{reference}_um"


def addReferencePositions(agcData: pd.DataFrame, reference: str, stat: str = "median") -> pd.DataFrame:
    """Return a copy of AG data with a reference position for each row.

    The references are:

    ``nominal``
        ``agc_nominal_[xy]_mm``, where the guider expects the star; the opdb
        provides it, so nothing is added.
    ``nominal0``, ``center0``
        The ``stat`` of each guide star's ``agc_nominal_[xy]_mm`` or
        ``agc_center_[xy]_mm`` over all the rows.
    ``nominal0_visit``, ``center0_visit``
        The same, over each ``pfs_visit_id``.

    Columns that are already present are recomputed.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data in hardware coordinates, with ``guide_star_id``,
        ``pfs_visit_id`` and ``agc_exposure_id`` columns.
    reference : `str`
        One of `REFERENCES`.
    stat : `str`
        ``mean``, ``median``, ``first`` or ``last``, the last two in order of
        ``agc_exposure_id``.

    Returns
    -------
    agcData : `pandas.DataFrame`
        A copy, with the `referenceColumns` of ``reference``. Rows keep
        their order and index labels.
    """
    _checkReference(reference)
    if stat not in _REFERENCE_STATS:
        raise ValueError(f"Unknown stat {stat!r}; valid: {', '.join(_REFERENCE_STATS)}")

    agcData = agcData.copy()
    if reference == "nominal":
        return agcData

    source = "nominal" if reference.startswith("nominal") else "center"
    keys = ["pfs_visit_id", "guide_star_id"] if reference.endswith("_visit") else ["guide_star_id"]

    # A fresh index makes the result positional, whatever agcData's index.
    data = agcData.reset_index(drop=True)
    if stat in ("first", "last"):
        data = data.sort_values("agc_exposure_id", kind="stable")
    grouped = data.groupby(keys)
    for sourceColumn, column in zip(_positionColumns(source), referenceColumns(reference), strict=True):
        agcData[column] = grouped[sourceColumn].transform(stat).sort_index().to_numpy()

    return agcData


def addOffsets(agcData: pd.DataFrame, reference: str = "nominal", stat: str = "median") -> pd.DataFrame:
    """Return a copy of AG data with each star's offset from a reference.

    The offset is ``agc_center_[xy]_mm`` minus the reference position, in
    microns.

    Parameters
    ----------
    agcData : `pandas.DataFrame`
        AG data in hardware coordinates; see `addReferencePositions`.
    reference : `str`
        One of `REFERENCES`.
    stat : `str`
        Statistic for the reference positions; see `addReferencePositions`.

    Returns
    -------
    agcData : `pandas.DataFrame`
        A copy, with the `referenceColumns` and `offsetColumns` of
        ``reference``.
    """
    agcData = addReferencePositions(agcData, reference, stat)
    for center, ref, offset in zip(
        _positionColumns("center"), referenceColumns(reference), offsetColumns(reference), strict=True
    ):
        agcData[offset] = mmToUm(agcData[center] - agcData[ref])

    return agcData
