import numpy as np
from numpy.typing import ArrayLike


def getWeightedRMS(resid: ArrayLike, err: ArrayLike, soften: float = 0) -> float:
    """Small helper function to get the weighted RMS with optional softening.

    Parameters
    ----------
    resid : `numpy.ndarray`
        The residuals.
    err : `numpy.ndarray`
        The errors.
    soften : `float`, optional
        The softening parameter. Default is 0.

    Returns
    -------
    rms : `float`
    """
    weight = 1 / (err**2 + soften**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.sum(weight * resid**2) / np.sum(weight))


def getChi2(resid: ArrayLike, err: ArrayLike, soften: float = 0) -> float:
    """Small helper function to get the chi2 with optional softening.

    Parameters
    ----------
    resid : `numpy.ndarray`
        The residuals.
    err : `numpy.ndarray`
        The errors.
    soften : `float`, optional
        The softening parameter. Default is 0.

    Returns
    -------
    chi2 : `float`
    """
    with np.errstate(invalid="ignore"):
        resids = (resid**2) / (err**2 + soften**2)
        return np.sum(resids)


def gaussian_func(x: ArrayLike, a: float, mu: float, sigma: float) -> np.ndarray:
    """Gaussian function.

    Parameters
    ----------
    x : `numpy.ndarray`
        The x values.
    a : `float`
        The amplitude.
    mu : `float`
        The mean.
    sigma : `float`
        The standard deviation.

    Returns
    -------
    y : `numpy.ndarray`
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def gaussianFixedWidth(x: ArrayLike, a: float, mu: float, sigma: float = 1.5) -> ArrayLike:
    """Gaussian function with fixed width.

    Parameters
    ----------
    x : `numpy.ndarray`
        The x values.
    a : `float`
        The amplitude.
    mu : `float`
        The mean.
    sigma : `float`, optional
        The standard deviation. Default is 1.5.

    Returns
    -------
    y : `numpy.ndarray`
    """
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
