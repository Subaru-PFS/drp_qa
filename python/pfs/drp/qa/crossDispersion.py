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
6.1 px         6.14 px      -20.9 %  11.3 %    +0.4 %, 8.8 %
============== ============ ======== ========= ===================

PFS traces are well inside the accurate range. Past it a trace approaches the
pitch, its width becomes degenerate with its neighbours' amplitudes and is
underestimated; it stays monotonic, so a defocused trace still measures wide.
At a 10 px pitch the fit is unbiased throughout, which places the bias in the
overlap rather than the method. No sample of a pure-noise row is accepted.
Fitting every fiber of a 4096-pixel row takes about 0.16 s on one laptop core;
the cost is proportional to the rows and to the fibers fitted.

Every fiber in the row is a neighbour in the fits, because a fiber's fit depends
on its *physical* neighbours; only the fibers asked for (``fit``) are fitted, so
a subset costs proportionally less. A neighbour that is not fitted stays at its
predicted center.

`measureImageWidths` applies `measureRow` to the sampled rows of a whole
detector and returns the per-sample table ``imageQualityQa`` writes to
``iqQaData``. It takes the detector-map predictions as arrays, so the task does
the stack calls and everything after them is testable here. `profileFwhm`
reads the half-maximum width of an oversampled fiber profile, such as
drp_stella's ``FiberProfile``, on the same convention.

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
    "profileFwhm",
]

#: Gaussian sigma to FWHM.
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))

#: Grid steps either side of the first pass's best sigma that later passes search.
LOCAL_STEPS = 3

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


def _solve(
    data: np.ndarray,
    weight: np.ndarray,
    edges: np.ndarray,
    sigma: np.ndarray,
    covariance: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Solve the linear fit for every fiber at a given sigma.

    Parameters
    ----------
    data, weight : `numpy.ndarray`, shape (m, L)
        Window values and inverse-variance weights (0 = ignore).
    edges : `numpy.ndarray`, shape (m, L + 1, 5)
        Pixel edges of the window minus each center, for fibers i-2 .. i+2.
        Adjacent pixels share an edge, so each Gaussian integral is evaluated
        once per edge rather than twice per pixel.
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
    frac = 0.5 * np.diff(erf(edges / (np.sqrt(2.0) * sig)), axis=1)
    gauss = np.exp(-0.5 * (edges[:, :, 1:4] / sig) ** 2)
    deriv = -np.diff(gauss, axis=1) / (np.sqrt(2.0 * np.pi) * sig)
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
    # The residual explicitly: d'Wd - coef.rhs is equal in exact arithmetic but subtracts two numbers
    # of ~1e6 to leave one of ~L, and the normal matrix is ill-conditioned enough to lose it.
    resid = data - np.einsum("mlk,mk->ml", design, coef)
    chi2 = np.sum(weight * resid**2, axis=1)
    fluxVariance = None
    if covariance:
        unit = np.zeros((data.shape[0], 9, 1))
        unit[:, 3, 0] = 1.0
        fluxVariance = np.linalg.solve(normal, unit)[:, 3, 0]
    return chi2, coef, fluxVariance


def _refine(
    data: np.ndarray, weight: np.ndarray, edges: np.ndarray, sigma: np.ndarray, step: float
) -> np.ndarray:
    """One parabolic refinement of each fiber's sigma, in log space.

    Parameters
    ----------
    data, weight : `numpy.ndarray`, shape (m, L)
    edges : `numpy.ndarray`, shape (m, L + 1, 5)
    sigma : `numpy.ndarray`, shape (m,)
        Current estimate.
    step : `float`
        Log-space step between the three evaluation points.

    Returns
    -------
    `numpy.ndarray`
        Refined sigma, moved by at most one step.
    """
    lower, _, _ = _solve(data, weight, edges, sigma * np.exp(-step))
    middle, _, _ = _solve(data, weight, edges, sigma)
    upper, _, _ = _solve(data, weight, edges, sigma * np.exp(step))
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
    fit: ArrayLike | None = None,
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
        Extra passes after moving each center by its fitted shift. They search
        the grid only within `LOCAL_STEPS` of the previous pass's best sigma.
    fit : array-like of `bool`, optional
        Which fibers to fit, in input order; the others are only neighbours,
        at their predicted centers, and are returned flagged. `None` fits
        every fiber.

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
    wanted = np.ones(nFibers, dtype=bool) if fit is None else np.asarray(fit, dtype=bool)
    sel = np.flatnonzero(wanted[order])  # positions, in center order, of the fibers to fit
    if sel.size == 0:
        return out

    # Absent neighbours at the ends of the row are parked far away, where they
    # contribute nothing to any window.
    far = 1e6

    def neighbours(c):
        pad = np.r_[c[0] - 2 * far, c[0] - far, c, c[-1] + far, c[-1] + 2 * far]
        return np.stack([pad[k : k + m] for k in range(5)], axis=1)[sel]

    grid = np.geomspace(sigmaRange[0], sigmaRange[1], sigmaSteps)
    gridStep = np.log(grid[1] / grid[0])

    with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
        for iteration in range(max(refinements, 0) + 1):
            around = neighbours(centers)
            here = centers[sel]
            # Window: from one neighbour's peak to the other's, plus a pixel.
            left = np.where(around[:, 1] > -far / 2, around[:, 1], here - 4.0)
            right = np.where(around[:, 3] < nCols + far / 2, around[:, 3], here + 4.0)
            lo = np.floor(left).astype(int) - 1
            hi = np.ceil(right).astype(int) + 1
            width = int(np.clip(np.max(hi - lo), 5, 40))
            pixels = lo[:, None] + np.arange(width + 1)[None, :]
            clipped = np.clip(pixels, 0, nCols - 1)
            usable = (pixels <= hi[:, None]) & (pixels >= 0) & (pixels < nCols) & ~bad[clipped]
            weight = np.where(usable, 1.0 / np.clip(variance[clipped], 1e-12, None), 0.0)
            data = np.where(usable, row[clipped], 0.0)
            # Pixel edges minus each center: pixel j spans edges j and j + 1.
            edgePixels = np.concatenate([pixels - 0.5, pixels[:, -1:] + 0.5], axis=1).astype(np.float64)
            edges = edgePixels[:, :, None] - around[:, None, :]

            if iteration == 0:
                # Coarse grid to bracket the minimum, then parabolic refinement
                # with shrinking steps, per fiber.
                chi2 = np.full((sigmaSteps, len(sel)), np.inf)
                for s, sig in enumerate(grid):
                    try:
                        chi2[s], _, _ = _solve(data, weight, edges, sig)
                    except np.linalg.LinAlgError:
                        continue
                best = np.argmin(chi2, axis=0)
                onEdge = (best == 0) | (best == sigmaSteps - 1)
                sigma = grid[best]
                steps = (gridStep, gridStep / 4.0, gridStep / 16.0)
            else:
                # The centers moved by a fraction of a pixel, so the minimum is
                # near the previous pass's: search the grid around it only.
                near = np.clip(
                    best[None, :] + np.arange(-LOCAL_STEPS, LOCAL_STEPS + 1)[:, None], 0, sigmaSteps - 1
                )
                chi2 = np.full(near.shape, np.inf)
                for s, sig in enumerate(grid[near]):
                    try:
                        chi2[s], _, _ = _solve(data, weight, edges, sig)
                    except np.linalg.LinAlgError:
                        continue
                best = near[np.argmin(chi2, axis=0), np.arange(len(sel))]
                onEdge = (best == 0) | (best == sigmaSteps - 1)
                sigma = grid[best]
                steps = (gridStep, gridStep / 4.0, gridStep / 16.0)
            for step in steps:
                sigma = np.clip(_refine(data, weight, edges, sigma, step), grid[0], grid[-1])

            _, coef, _ = _solve(data, weight, edges, sigma)
            amp = coef[:, 3]
            shift = np.where(amp > 0, coef[:, 4] / amp, 0.0)
            centers[sel] = here + np.clip(np.nan_to_num(shift), -1.0, 1.0)

        # Final solve at each fiber's own sigma and refined center.
        edges = edgePixels[:, :, None] - neighbours(centers)[:, None, :]
        chi2Final, coef, fluxVariance = _solve(data, weight, edges, sigma, covariance=True)

        here = centers[sel]
        background, flux = coef[:, 0], coef[:, 3]
        fluxErr = np.sqrt(np.clip(fluxVariance, 0.0, None))
        snr = flux / fluxErr
        nearest = np.clip(np.rint(here).astype(int), 0, nCols - 1)
        peakFraction = _pixelFraction(nearest - here, sigma)
        amplitude = flux * peakFraction

        good = (
            np.isfinite(sigma)
            & ~onEdge
            & np.isfinite(chi2Final)
            & (flux > 0)
            & ~bad[nearest]
            & (snr >= minPeakSN)
            & (np.abs(here - predicted[sel]) < 2.0)
        )

    results = {
        "sigma": sigma,
        "fwhm": pixelFwhm(sigma),
        "fwhmIntrinsic": FWHM_FACTOR * sigma,
        "centroid": here,
        "amplitude": amplitude,
        "background": background,
        "flux": flux,
        "fluxErr": fluxErr,
        "peakRatio": peakFraction,
        "snr": snr,
    }
    for key, value in results.items():
        out[key][order[sel]] = np.where(good, value, np.nan)
    out["flag"][order[sel]] = ~good
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

    Only the fibers reported are fitted, but every fiber in ``fiberIds`` is a
    neighbour in their fits, at its predicted center: a fiber's fit depends on
    its physical neighbours. Selecting the ~10 % of fibers that are flux
    standards on a science frame so costs ~10 % of fitting every fiber.

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
        ``dxCenter`` is the measured centroid minus that prediction, as drp_stella's residuals.
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
        result = measureRow(image[ii], bad[ii], variance[ii], xCenters[ii], minPeakSN=minPeakSN, fit=report)
        keep = report & np.isfinite(xCenters[ii]) & np.isfinite(wavelengths[ii])
        n = int(keep.sum())
        dxCenter = result["centroid"][keep] - calib[ii][keep] if calib is not None else np.full(n, np.nan)
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


def profileFwhm(index: ArrayLike, profiles: ArrayLike) -> np.ndarray:
    """Return the full width at half maximum of each row of an oversampled profile.

    Read directly, with no model: the peak from a parabola through the three
    highest samples, and each side from linear interpolation where the profile
    crosses half of it. For profiles that already include the pixel response,
    such as drp_stella's ``FiberProfile`` (pixel values at sub-pixel offsets
    from the trace center), the result is on the convention of ``fwhm`` in
    `measureRow`.

    The PFS trace profile is not Gaussian: its half-maximum width is 8-10 %
    more than a Gaussian fitted to it (Run25 ``fiberProfiles``), and its
    second moment depends on the wings. This reading is the one the joint fit
    of `measureRow` agrees with (``docs/trace-widths.ipynb``).

    Parameters
    ----------
    index : array-like, shape (nIndex,)
        Offset of each profile sample from the trace center, in pixels, in
        increasing order (``FiberProfile.index``).
    profiles : array-like, shape (nRows, nIndex) or (nIndex,)
        Profile values (``FiberProfile.profiles``), one row per swath. Masked
        or non-finite samples are ignored.

    Returns
    -------
    `numpy.ndarray`, shape (nRows,)
        FWHM in pixels; NaN where the profile has no positive peak away from
        its ends, or does not fall to half of it on both sides.
    """
    index = np.asarray(index, dtype=np.float64)
    masked = np.ma.masked_invalid(np.ma.atleast_2d(np.ma.asarray(profiles, dtype=np.float64)))
    fwhm = np.full(masked.shape[0], np.nan)
    for ii, row in enumerate(masked):
        use = ~np.ma.getmaskarray(row)
        x, y = index[use], row.data[use]
        if len(x) < 5:
            continue
        top = int(np.argmax(y))
        if top == 0 or top == len(y) - 1 or y[top] <= 0:
            continue
        # Parabola through the three highest samples, for the peak value.
        curve = np.polyfit(x[top - 1 : top + 2], y[top - 1 : top + 2], 2)
        vertex = -curve[1] / (2 * curve[0]) if curve[0] < 0 else x[top]
        peak = max(y[top], float(np.polyval(curve, np.clip(vertex, x[top - 1], x[top + 1]))))
        half = 0.5 * peak
        below = np.flatnonzero(y[:top] < half)
        above = np.flatnonzero(y[top + 1 :] < half)
        if below.size == 0 or above.size == 0:
            continue
        lo = below[-1]
        hi = top + 1 + above[0]
        left = x[lo] + (half - y[lo]) * (x[lo + 1] - x[lo]) / (y[lo + 1] - y[lo])
        right = x[hi - 1] + (half - y[hi - 1]) * (x[hi] - x[hi - 1]) / (y[hi] - y[hi - 1])
        fwhm[ii] = right - left
    return fwhm
