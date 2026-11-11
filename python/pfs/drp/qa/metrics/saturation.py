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

`linePositions` and `traceCentres` ask a detectorMap (any object with its
``findPoint`` and ``getXCenter``) where those are, and `measureSaturation`
puts the four together for the task and for the per-detector report
(`pfs.drp.qa.saturationReport`). `saturatedLineSummary` counts the saturated
fibers of each line.
"""

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "APERTURE_HALF_WIDTH",
    "PEAK_HALF_SIZE",
    "apertureSaturation",
    "linePositions",
    "measureSaturation",
    "saturatedAt",
    "saturatedLineSummary",
    "traceCentres",
]

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


def linePositions(detectorMap: Any, fiberId: Any, wavelength: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return where the detectorMap expects each line.

    Parameters
    ----------
    detectorMap : `pfs.drp.stella.DetectorMap`
        Mapping from fiberId and wavelength to x, y (``findPoint``).
    fiberId : array of `int`
        Each line's fiber.
    wavelength : array of `float`
        Each row's wavelength. A trace centroid has one too (its row's), so it is
        placed like a line: the caller leaves those rows out of its counts.

    Returns
    -------
    x, y : `numpy.ndarray` [`float`]
        The expected positions; NaN for a row whose wavelength is not finite, or off the
        detector.
    """
    fiberId = np.asarray(fiberId, dtype=np.int32)
    wavelength = np.asarray(wavelength, dtype=np.float64)
    x = np.full(len(wavelength), np.nan)
    y = np.full(len(wavelength), np.nan)
    known = np.isfinite(wavelength)
    if known.any():
        points = np.asarray(detectorMap.findPoint(fiberId[known], wavelength[known]), dtype=np.float64)
        points = points.reshape(-1, 2)
        x[known], y[known] = points[:, 0], points[:, 1]
    return x, y


def traceCentres(detectorMap: Any, fiberIds: Any, height: int) -> np.ndarray:
    """Return each fiber's trace centre on every row.

    Parameters
    ----------
    detectorMap : `pfs.drp.stella.DetectorMap`
        Mapping from fiberId and row to x (``getXCenter``).
    fiberIds : array of `int`
        The fibers.
    height : `int`
        Number of rows of the image.

    Returns
    -------
    `numpy.ndarray` [`float`], shape ``(len(fiberIds), height)``
        The x of each trace on each row.
    """
    fiberIds = np.asarray(fiberIds, dtype=np.int32)
    fibers = np.repeat(fiberIds, height)
    rows = np.tile(np.arange(height, dtype=np.float64), len(fiberIds))
    return np.asarray(detectorMap.getXCenter(fibers, rows), dtype=np.float64).reshape(len(fiberIds), height)


def measureSaturation(
    saturated: np.ndarray, detectorMap: Any, lines: pd.DataFrame, litFiberIds: Any = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Measure one detector image's saturation.

    Parameters
    ----------
    saturated : `numpy.ndarray` [`bool`], shape ``(height, width)``
        The calexp's ``SAT`` pixels.
    detectorMap : `pfs.drp.stella.DetectorMap`
        The visit's detector map; ``fiberId`` gives the detector's fibers.
    lines : `pandas.DataFrame`
        The ``lines`` table, with ``fiberId`` and ``wavelength``.
    litFiberIds : collection of `int`, optional
        The fibers whose apertures are counted; every fiber of the
        detectorMap when `None`.

    Returns
    -------
    lineSaturated : `numpy.ndarray` [`bool`]
        Per row of ``lines``, whether the line is saturated (`saturatedAt` its
        expected position). Trace centroids are rows like any other here, and on a
        saturated trace are `True`: count the lamp or sky lines alone.
    x, y : `numpy.ndarray` [`float`]
        Each line's expected position (`linePositions`).
    pctSaturatedPixels : `float`
        Percentage of the pixels within the lit fibers' apertures that are
        saturated (`apertureSaturation`); NaN when no fiber is lit.
    """
    saturated = np.asarray(saturated, dtype=bool)
    if len(lines) and "wavelength" in lines.columns:
        x, y = linePositions(detectorMap, lines["fiberId"].to_numpy(), lines["wavelength"].to_numpy())
    else:
        x = np.full(len(lines), np.nan)
        y = np.full(len(lines), np.nan)
    lineSaturated = saturatedAt(saturated, x, y)

    fiberIds = np.asarray(detectorMap.fiberId, dtype=np.int32)
    if litFiberIds is not None:
        fiberIds = fiberIds[np.isin(fiberIds, list(litFiberIds))]
    pctSaturatedPixels = np.nan
    if len(fiberIds):
        numSaturated, numPixels = apertureSaturation(
            saturated, traceCentres(detectorMap, fiberIds, saturated.shape[0])
        )
        if numPixels:
            pctSaturatedPixels = 100.0 * numSaturated / numPixels
    return lineSaturated, x, y, pctSaturatedPixels


def saturatedLineSummary(lines: pd.DataFrame) -> pd.DataFrame:
    """Return, for each line with a saturated fiber, how many fibers are.

    Parameters
    ----------
    lines : `pandas.DataFrame`
        One row per line and fiber: ``description``, ``wavelength``,
        ``fiberId`` and ``saturated``.

    Returns
    -------
    `pandas.DataFrame`
        ``description``, ``wavelength``, ``nFibers`` (fibers with the line)
        and ``nSaturated`` (those saturated), for the lines with any, in order
        of wavelength; empty, with the same columns, when ``lines`` is.
    """
    columns = ["description", "wavelength", "nFibers", "nSaturated"]
    if len(lines) == 0:
        # Typed, so that an image with no lamp or sky line (a quartz) sorts and selects like any other.
        dtypes = {"description": object, "wavelength": float, "nFibers": int, "nSaturated": int}
        return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in dtypes.items()})
    summary = (
        lines.groupby(["description", "wavelength"], sort=True)
        .agg(nFibers=("fiberId", "nunique"), nSaturated=("saturated", "sum"))
        .reset_index()
    )
    summary["nSaturated"] = summary["nSaturated"].astype(int)
    summary = summary[summary["nSaturated"] > 0]
    return summary.sort_values(["wavelength", "description"], ignore_index=True)[columns]
