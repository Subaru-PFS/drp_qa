"""Saturation, from the ``SAT`` mask plane of the calexp.

The reference is the detector's own saturation level (rule R1 of
``docs/qa-principles.md``): ISR sets ``SAT`` on the pixels at or above it, and
on an H4RG those whose ramp left the valid range. A percentile of the frame's
own flux says nothing about full well.

Stack-free: the task hands over the plane as a boolean image, and positions
from the detectorMap.

- `saturatedAt`: whether a line is saturated, i.e. a pixel at its expected
  position is. The expected position, not the measured centroid: drp_stella's
  line finder skips ``SAT`` pixels, so a saturated line's centroid is
  elsewhere or missing.
- `apertureSaturation`: the saturated pixels among those the lit fibers'
  traces cover.
"""

import numpy as np

__all__ = ["APERTURE_HALF_WIDTH", "PEAK_HALF_SIZE", "apertureSaturation", "saturatedAt"]

#: Half-size (pixels) of the box about a line's expected position searched for a
#: saturated pixel: the peak pixel and its neighbours, which covers the
#: detectorMap's error in placing the line.
PEAK_HALF_SIZE = 1

#: Half-width (pixels) of a fiber's aperture about its trace centre: 7 columns of
#: the 6.17 a fiber has to itself, so neighbouring apertures just meet.
APERTURE_HALF_WIDTH = 3


def saturatedAt(
    saturated: np.ndarray, x: np.ndarray, y: np.ndarray, halfSize: int = PEAK_HALF_SIZE
) -> np.ndarray:
    """Return, per position, whether a pixel within ``halfSize`` of it is saturated.

    Parameters
    ----------
    saturated : `numpy.ndarray` [`bool`], shape ``(height, width)``
        The image's saturated pixels.
    x, y : `numpy.ndarray` [`float`]
        Positions, in pixels (``y`` the row).
    halfSize : `int`, optional
        Half-size of the box searched: 0 for the nearest pixel alone.

    Returns
    -------
    `numpy.ndarray` [`bool`]
        `False` for a position that is not finite; pixels of the box off the
        image are not saturated.
    """
    saturated = np.asarray(saturated, dtype=bool)
    height, width = saturated.shape
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    column = np.rint(np.where(finite, x, -1.0e6)).astype(np.int64)
    row = np.rint(np.where(finite, y, -1.0e6)).astype(np.int64)
    result = np.zeros(x.shape, dtype=bool)
    for dRow in range(-halfSize, halfSize + 1):
        for dColumn in range(-halfSize, halfSize + 1):
            rr, cc = row + dRow, column + dColumn
            inside = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            result[inside] |= saturated[rr[inside], cc[inside]]
    return result


def apertureSaturation(
    saturated: np.ndarray, xCenters: np.ndarray, halfWidth: int = APERTURE_HALF_WIDTH
) -> tuple[int, int]:
    """Count the saturated pixels within the fibers' apertures.

    Parameters
    ----------
    saturated : `numpy.ndarray` [`bool`], shape ``(height, width)``
        The image's saturated pixels.
    xCenters : `numpy.ndarray` [`float`], shape ``(numFibers, height)``
        Each fiber's trace centre on every row; not finite where it has none.
    halfWidth : `int`, optional
        Half-width of an aperture about the trace centre.

    Returns
    -------
    numSaturated : `int`
        Saturated pixels within an aperture.
    numPixels : `int`
        Pixels within an aperture, one in two apertures counted once.
    """
    saturated = np.asarray(saturated, dtype=bool)
    height, width = saturated.shape
    xCenters = np.asarray(xCenters, dtype=float).reshape(-1, height)
    finite = np.isfinite(xCenters)
    column = np.rint(np.where(finite, xCenters, -1.0e6)).astype(np.int64)
    row = np.broadcast_to(np.arange(height), column.shape)
    aperture = np.zeros(saturated.shape, dtype=bool)
    for offset in range(-halfWidth, halfWidth + 1):
        cc = column + offset
        inside = (cc >= 0) & (cc < width)
        aperture[row[inside], cc[inside]] = True
    return int((aperture & saturated).sum()), int(aperture.sum())
