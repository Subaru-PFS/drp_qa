"""Cross-dispersion trace width measured directly from image pixels.

Each fiber is fitted jointly with its neighbours, with no aperture, so the
measurement uses no pixel without modelling the light on it. PFS fibers are
6.17 px apart on every arm (Run25) while traces are FWHM 2.6-3.7 px, so on a
quartz frame a neighbour's wing is most of the light between two traces.

* **Model.** The fiber, its two neighbours and the two beyond those are
  Gaussians sharing one sigma, each integrated over its pixels, plus a constant
  background, fitted over the pixels from one neighbour's peak to the other's.
* **Centers** start at the detector map's prediction and are refined inside the
  fit by a derivative term per fiber, which makes a small shift linear; the fit
  is then repeated once at the moved centers.
* **Sigma.** The fit is linear in everything except sigma. A coarse logarithmic
  grid brackets the least-squares minimum and parabolas through closely spaced
  points refine it per fiber. A best fit on the edge of the grid is flagged.
* **Significance** is the fitted flux over its standard error, from the fit's
  covariance with inverse-variance weights from the variance plane.

**FWHM includes the pixel response.** The fit returns the intrinsic sigma of the
light falling on the detector. ``fwhm`` adds the variance of a 1 px box
(`PIXEL_VARIANCE`, 1/12 px²) before scaling, so it measures the same thing as
``SdssShape`` moments of arc lines and the widths of ``fiberProfiles``: the
profile as the pixels record it. ``fwhmIntrinsic`` leaves it out; the two differ
by 3.4 % at an intrinsic FWHM of 2.6 px and 1.7 % at 3.7 px.

Characterised on synthetic quartz rows (all fibers lit, 20,000 e- per trace per
row) at the PFS pitch: median bias of ``fwhm`` against the true pixel-recorded
FWHM, and the scatter of single samples.

============== ============ ======== ========= ===================
intrinsic FWHM ``fwhm``     bias     scatter   at a 10 px pitch
============== ============ ======== ========= ===================
2.1 px         2.21 px      +0.0 %   0.6 %     +0.0 %, 0.4 %
2.6 px         2.69 px      +0.1 %   1.0 %     -0.0 %, 0.5 %
3.1 px         3.17 px      +0.4 %   2.2 %     +0.0 %, 0.6 %
3.7 px         3.76 px      +1.0 %   5.8 %     -0.0 %, 0.8 %
4.7 px         4.75 px      -7.3 %   7.3 %     -0.0 %, 1.7 %
6.1 px         6.14 px      -20.9 %  11.7 %    +0.4 %, 8.8 %
============== ============ ======== ========= ===================

PFS traces are well inside the accurate range. Past it a trace approaches the
pitch, its width becomes degenerate with its neighbours' amplitudes and is
underestimated; it stays monotonic, so a defocused trace still measures wide.
At a 10 px pitch the fit is unbiased throughout, which places the bias in the
overlap rather than the method. No sample of a pure-noise row is accepted. A
row of 4096 px takes about 0.2 s.

Measurement uses every fiber in the row, not only the requested ones, because a
fiber's fit depends on its *physical* neighbours. Callers select the fibers they
want afterwards.

`measureImageWidths` applies `measureRow` to the sampled rows of a whole
detector and returns the per-sample table ``imageQualityQa`` writes to
``iqQaData``. It takes the detector-map predictions as arrays, so the task does
the stack calls and everything after them is testable here. `profileWidth`
reads a width from an oversampled fiber profile, such as drp_stella's
``FiberProfile``, on the same convention.

No Butler, no stack: numpy, scipy and pandas. Tested in
``tests/imageQualityQa/test_crossDispersion.py``.
"""

from collections.abc import Collection

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import erf

__all__ = [
    "FWHM_FACTOR",
    "IMAGE_WIDTH_COLUMNS",
    "PIXEL_VARIANCE",
    "measureImageWidths",
    "measureRow",
    "pixelFwhm",
    "profileWidth",
]

#: Gaussian sigma to FWHM.
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))

#: Variance of a uniform 1 px box, in px²: what pixel integration adds to a profile's variance.
PIXEL_VARIANCE = 1.0 / 12.0


def pixelFwhm(sigma: ArrayLike) -> np.ndarray:
    """Return the FWHM of a Gaussian of intrinsic ``sigma`` as pixels record it.

    Parameters
    ----------
    sigma : array-like
        Intrinsic Gaussian sigma, in pixels.

    Returns
    -------
    `numpy.ndarray`
        ``FWHM_FACTOR * sqrt(sigma**2 + PIXEL_VARIANCE)``.
    """
    return FWHM_FACTOR * np.sqrt(np.asarray(sigma, dtype=np.float64) ** 2 + PIXEL_VARIANCE)


def _pixelFraction(offset: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return the fraction of a unit Gaussian's flux in a pixel at ``offset``.

    Parameters
    ----------
    offset : `numpy.ndarray`
        Distance from the Gaussian's center to the pixel center, in pixels.
    sigma : `numpy.ndarray`
        Gaussian sigma, in pixels.

    Returns
    -------
    `numpy.ndarray`
        Integrated flux fraction over ``[offset - 0.5, offset + 0.5]``.
    """
    scale = np.sqrt(2.0) * sigma
    return 0.5 * (erf((offset + 0.5) / scale) - erf((offset - 0.5) / scale))


def _pixelDerivative(offset: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return d/d(center) of `_pixelFraction` at ``offset``.

    Parameters
    ----------
    offset : `numpy.ndarray`
        Distance from the Gaussian's center to the pixel center, in pixels.
    sigma : `numpy.ndarray`
        Gaussian sigma, in pixels.

    Returns
    -------
    `numpy.ndarray`
        How the pixel's flux fraction changes as the center moves right.
    """
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    upper = np.exp(-0.5 * ((offset + 0.5) / sigma) ** 2)
    lower = np.exp(-0.5 * ((offset - 0.5) / sigma) ** 2)
    return -norm * (upper - lower)


def _solve(
    data: np.ndarray,
    weight: np.ndarray,
    offsets: np.ndarray,
    sigma: np.ndarray,
    covariance: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Solve the linear fit for every fiber at a given sigma.

    Parameters
    ----------
    data, weight : `numpy.ndarray`, shape (m, L)
        Window values and inverse-variance weights (0 = ignore).
    offsets : `numpy.ndarray`, shape (m, L, 5)
        Pixel minus center, for fibers i-2 .. i+2.
    sigma : `numpy.ndarray`, shape (m,) or scalar
        Sigma for each fiber's fit.
    covariance : `bool`, optional
        Also return the variance of the target fiber's flux.

    Returns
    -------
    chi2 : `numpy.ndarray`, shape (m,)
    coef : `numpy.ndarray`, shape (m, 9)
        Background; amplitude and amplitude*shift for fibers i-1, i, i+1; then
        amplitude for fibers i-2 and i+2.
    fluxVariance : `numpy.ndarray`, shape (m,), or `None`
    """
    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float64), data.shape[:1])[:, None, None]
    frac = _pixelFraction(offsets, sig)
    deriv = _pixelDerivative(offsets[:, :, 1:4], sig)
    design = np.concatenate(
        [
            np.ones((*data.shape, 1)),
            frac[:, :, 1:2],
            deriv[:, :, 0:1],  # i-1
            frac[:, :, 2:3],
            deriv[:, :, 1:2],  # i
            frac[:, :, 3:4],
            deriv[:, :, 2:3],  # i+1
            frac[:, :, 0:1],
            frac[:, :, 4:5],  # i-2, i+2
        ],
        axis=2,
    )
    wd = design * weight[:, :, None]
    normal = np.einsum("mlk,mlj->mkj", wd, design) + 1e-8 * np.eye(9)
    rhs = np.einsum("mlk,ml->mk", wd, data)
    coef = np.linalg.solve(normal, rhs[:, :, None])[:, :, 0]
    resid = data - np.einsum("mlk,mk->ml", design, coef)
    chi2 = np.sum(weight * resid**2, axis=1)
    fluxVariance = None
    if covariance:
        unit = np.zeros((data.shape[0], 9, 1))
        unit[:, 3, 0] = 1.0
        fluxVariance = np.linalg.solve(normal, unit)[:, 3, 0]
    return chi2, coef, fluxVariance


def _refine(
    data: np.ndarray, weight: np.ndarray, offsets: np.ndarray, sigma: np.ndarray, step: float
) -> np.ndarray:
    """One parabolic refinement of each fiber's sigma, in log space.

    Parameters
    ----------
    data, weight : `numpy.ndarray`, shape (m, L)
    offsets : `numpy.ndarray`, shape (m, L, 5)
    sigma : `numpy.ndarray`, shape (m,)
        Current estimate.
    step : `float`
        Log-space step between the three evaluation points.

    Returns
    -------
    `numpy.ndarray`
        Refined sigma, moved by at most one step.
    """
    lower, _, _ = _solve(data, weight, offsets, sigma * np.exp(-step))
    middle, _, _ = _solve(data, weight, offsets, sigma)
    upper, _, _ = _solve(data, weight, offsets, sigma * np.exp(step))
    curvature = lower - 2.0 * middle + upper
    move = np.where((curvature > 0) & np.isfinite(curvature), 0.5 * (lower - upper) / curvature, 0.0)
    return sigma * np.exp(np.clip(move, -1.0, 1.0) * step)


def measureRow(
    row: ArrayLike,
    bad: ArrayLike,
    variance: ArrayLike,
    xCenters: ArrayLike,
    *,
    minPeakSN: float = 5.0,
    sigmaRange: tuple[float, float] = (0.4, 4.0),
    sigmaSteps: int = 24,
    refinements: int = 1,
) -> dict[str, np.ndarray]:
    """Measure the cross-dispersion width of every fiber in one image row.

    Parameters
    ----------
    row : array-like
        Pixel values along the row.
    bad : array-like of `bool`
        True for pixels to ignore (masked BAD, SAT, CR, NO_DATA).
    variance : array-like
        Per-pixel variance, from the image's variance plane.
    xCenters : array-like
        Predicted x-center of every fiber in the row, from the detector map, in
        any order. Non-finite entries are flagged and otherwise ignored.
    minPeakSN : `float`, optional
        Minimum significance of the fitted flux for a measurement to count.
    sigmaRange : `tuple` [`float`, `float`], optional
        Bounds of the sigma search, in pixels. A best fit on either bound is
        flagged: the true width lies outside what was searched.
    sigmaSteps : `int`, optional
        Points in the coarse grid across ``sigmaRange``, spaced logarithmically.
        The grid only brackets the minimum; the parabolic refinement sets the
        precision.
    refinements : `int`, optional
        Extra passes after moving each center by its fitted shift.

    Returns
    -------
    `dict` [`str`, `numpy.ndarray`]
        One entry per input fiber, in input order:

        ``sigma``
            Intrinsic Gaussian sigma: pixel integration is modelled.
        ``fwhm``
            FWHM including the pixel response (`pixelFwhm`).
        ``fwhmIntrinsic``
            ``FWHM_FACTOR * sigma``.
        ``centroid``
            Fitted x-center, in pixels. Compare with the detector map's
            prediction to get a spatial offset.
        ``amplitude``
            Signal in the pixel nearest the center, above background.
        ``background``
            Fitted constant background under the trace.
        ``flux``, ``fluxErr``
            Total trace flux from the fitted Gaussian, and its standard error.
        ``peakRatio``
            Fraction of the flux in the pixel nearest the center. For a
            Gaussian this is set by ``sigma``; kept for the output schema.
        ``snr``
            ``flux / fluxErr``.
        ``flag``
            True when the fiber could not be measured, in which case the other
            entries are NaN.
    """
    row = np.asarray(row, dtype=np.float64)
    bad = np.asarray(bad, dtype=bool)
    variance = np.asarray(variance, dtype=np.float64)
    x = np.asarray(xCenters, dtype=np.float64)
    nFibers, nCols = len(x), len(row)

    keys = (
        "sigma",
        "fwhm",
        "fwhmIntrinsic",
        "centroid",
        "amplitude",
        "background",
        "flux",
        "fluxErr",
        "peakRatio",
        "snr",
    )
    out = {key: np.full(nFibers, np.nan) for key in keys}
    out["flag"] = np.ones(nFibers, dtype=bool)

    finite = np.flatnonzero(np.isfinite(x))
    if finite.size == 0:
        return out
    order = finite[np.argsort(x[finite])]
    predicted = x[order]
    centers = predicted.copy()
    m = len(centers)

    # Absent neighbours at the ends of the row are parked far away, where they
    # contribute nothing to any window.
    far = 1e6

    def neighbours(c):
        pad = np.r_[c[0] - 2 * far, c[0] - far, c, c[-1] + far, c[-1] + 2 * far]
        return np.stack([pad[k : k + m] for k in range(5)], axis=1)

    grid = np.geomspace(sigmaRange[0], sigmaRange[1], sigmaSteps)
    gridStep = np.log(grid[1] / grid[0])

    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        for _ in range(max(refinements, 0) + 1):
            around = neighbours(centers)
            # Window: from one neighbour's peak to the other's, plus a pixel.
            left = np.where(around[:, 1] > -far / 2, around[:, 1], centers - 4.0)
            right = np.where(around[:, 3] < nCols + far / 2, around[:, 3], centers + 4.0)
            lo = np.floor(left).astype(int) - 1
            hi = np.ceil(right).astype(int) + 1
            width = int(np.clip(np.max(hi - lo), 5, 40))
            pixels = lo[:, None] + np.arange(width + 1)[None, :]
            clipped = np.clip(pixels, 0, nCols - 1)
            usable = (pixels <= hi[:, None]) & (pixels >= 0) & (pixels < nCols) & ~bad[clipped]
            weight = np.where(usable, 1.0 / np.clip(variance[clipped], 1e-12, None), 0.0)
            data = row[clipped]
            offsets = pixels.astype(np.float64)[:, :, None] - around[:, None, :]

            # Coarse grid to bracket the minimum.
            chi2 = np.full((sigmaSteps, m), np.inf)
            for s, sig in enumerate(grid):
                try:
                    chi2[s], _, _ = _solve(data, weight, offsets, sig)
                except np.linalg.LinAlgError:
                    continue
            best = np.argmin(chi2, axis=0)
            onEdge = (best == 0) | (best == sigmaSteps - 1)
            sigma = grid[best]

            # Parabolic refinement with shrinking steps, per fiber.
            for step in (gridStep, gridStep / 4.0, gridStep / 16.0):
                sigma = np.clip(_refine(data, weight, offsets, sigma, step), grid[0], grid[-1])

            _, coef, _ = _solve(data, weight, offsets, sigma)
            amp = coef[:, 3]
            shift = np.where(amp > 0, coef[:, 4] / amp, 0.0)
            centers = centers + np.clip(np.nan_to_num(shift), -1.0, 1.0)

        # Final solve at each fiber's own sigma and refined center.
        offsets = pixels.astype(np.float64)[:, :, None] - neighbours(centers)[:, None, :]
        chi2Final, coef, fluxVariance = _solve(data, weight, offsets, sigma, covariance=True)

        background, flux = coef[:, 0], coef[:, 3]
        fluxErr = np.sqrt(np.clip(fluxVariance, 0.0, None))
        snr = flux / fluxErr
        nearest = np.clip(np.rint(centers).astype(int), 0, nCols - 1)
        peakFraction = _pixelFraction(nearest - centers, sigma)
        amplitude = flux * peakFraction

        good = (
            np.isfinite(sigma)
            & ~onEdge
            & np.isfinite(chi2Final)
            & (flux > 0)
            & ~bad[nearest]
            & (snr >= minPeakSN)
            & (np.abs(centers - predicted) < 2.0)
        )

    results = {
        "sigma": sigma,
        "fwhm": pixelFwhm(sigma),
        "fwhmIntrinsic": FWHM_FACTOR * sigma,
        "centroid": centers,
        "amplitude": amplitude,
        "background": background,
        "flux": flux,
        "fluxErr": fluxErr,
        "peakRatio": peakFraction,
        "snr": snr,
    }
    for key, value in results.items():
        out[key][order] = np.where(good, value, np.nan)
    out["flag"][order] = ~good
    return out


#: Columns of the table `measureImageWidths` returns, in order.
IMAGE_WIDTH_COLUMNS = (
    "fiberId", "x", "y", "lam", "fwhm", "fwhmIntrinsic", "theta", "flux", "fluxErr",
    "flag", "traceOnly", "peakRatio", "dxCenter", "snr",
)  # fmt: skip


def measureImageWidths(
    image: ArrayLike,
    variance: ArrayLike,
    bad: ArrayLike,
    rows: ArrayLike,
    fiberIds: ArrayLike,
    xCenters: ArrayLike,
    wavelengths: ArrayLike,
    calibXCenters: ArrayLike | None = None,
    *,
    select: Collection[int] | None = None,
    minPeakSN: float = 5.0,
) -> pd.DataFrame:
    """Measure trace widths on sampled rows of a detector image.

    Every fiber in ``fiberIds`` is fitted on every row, because a fiber's fit
    depends on its physical neighbours; ``select`` restricts only which fibers
    are *reported*.

    Parameters
    ----------
    image, variance : array-like, shape (nRows, nCols)
        Image and variance planes of the sampled rows.
    bad : array-like of `bool`, shape (nRows, nCols)
        True for pixels to ignore (masked BAD, SAT, CR, NO_DATA).
    rows : array-like, shape (nRows,)
        Detector row (y) of each sampled row.
    fiberIds : array-like of `int`, shape (nFibers,)
        Every fiber the detector map knows on this detector.
    xCenters, wavelengths : array-like, shape (nRows, nFibers)
        Detector-map x-center and wavelength of each fiber on each row.
    calibXCenters : array-like, shape (nRows, nFibers), optional
        x-center predicted by the static calibration detector map. When given,
        ``dxCenter`` is that prediction minus the measured centroid.
    select : collection of `int`, optional
        Fiber IDs to report. `None` reports every fiber.
    minPeakSN : `float`, optional
        Passed to `measureRow`.

    Returns
    -------
    `pandas.DataFrame`
        One row per reported (fiber, row) sample with a finite x-center and
        wavelength, with columns `IMAGE_WIDTH_COLUMNS`. ``fwhm`` includes the
        pixel response and ``fwhmIntrinsic`` does not. ``x`` is the
        detector-map prediction, not the fitted centroid. ``fluxErr`` and
        ``snr`` come from the fit's covariance. ``traceOnly`` is `False`:
        these are measurements of the exposure, not widths read back from a
        calibration.
    """
    image = np.asarray(image, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    bad = np.asarray(bad, dtype=bool)
    rows = np.asarray(rows, dtype=np.float64)
    fiberIds = np.asarray(fiberIds, dtype=np.int32)
    xCenters = np.asarray(xCenters, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    calib = None if calibXCenters is None else np.asarray(calibXCenters, dtype=np.float64)

    report = np.ones(len(fiberIds), dtype=bool) if select is None else np.isin(fiberIds, list(select))

    frames = []
    for ii, y in enumerate(rows):
        result = measureRow(image[ii], bad[ii], variance[ii], xCenters[ii], minPeakSN=minPeakSN)
        keep = report & np.isfinite(xCenters[ii]) & np.isfinite(wavelengths[ii])
        n = int(keep.sum())
        dxCenter = calib[ii][keep] - result["centroid"][keep] if calib is not None else np.full(n, np.nan)
        frames.append(
            pd.DataFrame(
                {
                    "fiberId": fiberIds[keep],
                    "x": xCenters[ii][keep],
                    "y": np.full(n, y),
                    "lam": wavelengths[ii][keep],
                    "fwhm": result["fwhm"][keep],
                    "fwhmIntrinsic": result["fwhmIntrinsic"][keep],
                    "theta": np.zeros(n),
                    "flux": result["flux"][keep],
                    "fluxErr": result["fluxErr"][keep],
                    "flag": result["flag"][keep],
                    "traceOnly": np.zeros(n, dtype=bool),
                    "peakRatio": result["peakRatio"][keep],
                    "dxCenter": dxCenter,
                    "snr": result["snr"][keep],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=list(IMAGE_WIDTH_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def profileWidth(index: ArrayLike, profiles: ArrayLike) -> dict[str, np.ndarray]:
    """Fit a Gaussian plus a constant to each row of an oversampled fiber profile.

    For profiles that already include the pixel response, such as drp_stella's
    ``FiberProfile``: its values are pixel values at sub-pixel offsets from the
    trace center, so the fitted sigma is on the convention of ``fwhm`` in
    `measureRow`, and no `PIXEL_VARIANCE` is added. The constant absorbs a
    residual background, to which a second moment is sensitive.

    Parameters
    ----------
    index : array-like, shape (nIndex,)
        Offset of each profile sample from the trace center, in pixels
        (``FiberProfile.index``).
    profiles : array-like, shape (nRows, nIndex) or (nIndex,)
        Profile values (``FiberProfile.profiles``), one row per swath. Masked
        or non-finite samples are ignored.

    Returns
    -------
    `dict` [`str`, `numpy.ndarray`]
        One entry per row: ``sigma``, ``fwhm`` (``FWHM_FACTOR * sigma``),
        ``center`` and ``background``, NaN where the fit failed or fewer than
        five samples were usable.
    """
    from scipy.optimize import curve_fit

    index = np.asarray(index, dtype=np.float64)
    masked = np.ma.masked_invalid(np.ma.atleast_2d(np.ma.asarray(profiles, dtype=np.float64)))
    keys = ("sigma", "fwhm", "center", "background")
    out = {key: np.full(masked.shape[0], np.nan) for key in keys}

    def gaussian(x, amplitude, center, sigma, background):
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + background

    for ii, row in enumerate(masked):
        use = ~np.ma.getmaskarray(row)
        if use.sum() < 5:
            continue
        x, y = index[use], row.data[use]
        weight = np.clip(y - np.min(y), 0.0, None)
        if weight.sum() <= 0:
            continue
        center = float(np.sum(x * weight) / weight.sum())
        sigma = float(np.sqrt(np.sum((x - center) ** 2 * weight) / weight.sum()))
        try:
            params, _ = curve_fit(
                gaussian,
                x,
                y,
                p0=(float(np.max(y) - np.min(y)), center, max(sigma, 0.3), float(np.min(y))),
                bounds=((0.0, x.min(), 0.1, -np.inf), (np.inf, x.max(), x.max() - x.min(), np.inf)),
            )
        except (RuntimeError, ValueError):
            continue
        out["sigma"][ii] = params[2]
        out["center"][ii] = params[1]
        out["background"][ii] = params[3]
    out["fwhm"] = FWHM_FACTOR * out["sigma"]
    return out
