"""Cross-dispersion trace width measured directly from image pixels.

The estimator this replaces cut a fixed ``2 * profileHalfWidth + 1`` pixel strip
around each fiber and took its background from the outermost two pixels on
each side. PFS fibers sit roughly 6-7 px apart on the detector and the default
half-width was 7, so those "background" pixels landed on the neighbouring
fibers. On a quartz frame, where every fiber is lit, that inflated both the
background and the noise estimate until the S/N gate rejected essentially every
sample: 0.004-0.23 % usable across all four arms of the Run25 trace visits, with
every quantum falling back to the fiber-profile calibration. Reproduced on
synthetic data, the old estimator returns 0 % usable at any pitch of 7 px or
less and 100 % at 10 px or more.

This estimator uses no pixel that belongs to another fiber:

* **Width, position, amplitude and background** come from one joint least-squares
  fit per fiber. The fiber, its two neighbours and the two beyond those are
  modelled as pixel-integrated Gaussians sharing one sigma, plus a constant
  background, over the pixels from one neighbour's peak to the other's. On a
  quartz frame the neighbours' wings are most of the light between traces, so
  they are modelled rather than subtracted as "background"; the outer pair
  matters once a trace broadens toward the pitch.
* **Centers start at the detector map's prediction** and are refined inside the
  fit through a derivative term per fiber, which makes a small shift linear. It
  is iterated once. Fixing the centers at a separately-measured centroid instead
  lets every centroid error leak into sigma: that version showed 22 % scatter at
  sigma 1.3 and a 6.5 px pitch, against under 4 % here.
* The fit is linear in everything except sigma. A coarse logarithmic grid
  brackets the minimum, then a parabola through closely spaced points refines it
  per fiber, twice. Refining on the coarse grid alone biases sigma by a few
  percent wherever the truth falls between grid points. A least-squares minimum
  has no feedback loop to run away, unlike the saddle-correction iteration first
  tried here, which diverged for broad traces -- exactly the defocused case QA
  exists to catch.
* **Significance** is the fitted flux over its standard error from the fit's own
  covariance, with inverse-variance weights from the image's variance plane.
  Dividing by the noise in one pixel instead ignores the fit's degrees of
  freedom and the choice of the best of many sigmas, and let about a sixth of a
  pure-noise row through a 5-sigma cut.
* **Pixel integration** is modelled exactly: every Gaussian in the fit is
  integrated over its pixel, so sigma needs no after-the-fact correction.

Characterised on synthetic rows at the measured PFS pitch of 6.17 px:

======== ========= ======== =========
sigma    FWHM      bias     scatter
======== ========= ======== =========
0.9 px   2.1 px    +0.0 %   0.7 %
1.3 px   3.1 px    +0.4 %   2.2 %
1.6 px   3.8 px    +0.9 %   6.2 %
2.0 px   4.7 px    -7.8 %   7.6 %
2.6 px   6.1 px    -21.5 %  12.0 %
======== ========= ======== =========

PFS traces are FWHM 2.6-3.7 px, well inside the accurate range. Past it, traces
approach the pitch and a fiber's width becomes degenerate with its neighbours'
amplitudes, so sigma is underestimated -- but it stays **monotonic**, so a
defocused trace still measures wide and still fails a threshold. At a 10 px pitch
the same fit is unbiased throughout, which places the bias in the overlap rather
than the method. A pure-noise row is accepted 0 % of the time.

Measurement uses every fiber in the row, not only the requested ones, because a
fiber's fit depends on its *physical* neighbours. Callers select the fibers they
want afterwards. Measured PFS pitch is 6.17 px on every arm (Run25), against the
7 px half-width of the estimator this replaces.

`measureImageWidths` applies `measureRow` to the sampled rows of a whole
detector and returns the per-sample table ``imageQualityQa`` writes to
``iqQaData``. It takes the detector-map predictions as arrays, so the task does
the stack calls and everything after them is testable here.

No Butler, no stack -- numpy, scipy and pandas -- unit-tested in
``tests/test_crossDispersion.py``, including on real quartz rows.
"""

from collections.abc import Collection

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.special import erf

__all__ = ["FWHM_FACTOR", "IMAGE_WIDTH_COLUMNS", "measureImageWidths", "measureRow"]

#: Gaussian sigma to FWHM.
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


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

        ``sigma``, ``fwhm``
            Gaussian width. Pixel integration is modelled, not corrected for.
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

    keys = ("sigma", "fwhm", "centroid", "amplitude", "background", "flux", "fluxErr", "peakRatio", "snr")
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
        "fwhm": FWHM_FACTOR * sigma,
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
    "fiberId", "x", "y", "lam", "fwhm", "theta", "flux", "fluxErr",
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
        wavelength, with columns `IMAGE_WIDTH_COLUMNS`. ``x`` is the
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
