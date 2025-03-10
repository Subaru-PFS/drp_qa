import numpy as np
from pfs.drp.stella.datamodel.drp import PfsArm
from pfs.drp.stella.fitFocalPlane import FitBlockedOversampledSplineTask
from pfs.drp.stella.selectFibers import SelectFibersTask
from pfs.drp.stella.subtractSky1d import subtractSky1d

arm_colors = ['steelblue', 'firebrick', 'darkgoldenrod']


def subtractSkyWithExclusion(butler, dataId, excludeFiberId, fitSkyModelConfig):
    """Perform sky subtraction on PFS spectra while excluding a specific fiber.

    This function selects sky fibers, fits a sky model, subtracts it,
    and returns only the spectrum of the excluded fiber.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Data butler used to retrieve `pfsArm` and `pfsConfig` data.
    dataId : `dict`
        Identifier for the data to retrieve (e.g., visit, spectrograph, arm).
    excludeFiberId : `int`
        Fiber ID to exclude from the sky selection process.
    fitSkyModelConfig : `FitBlockedOversampledSplineConfig`
        Configuration for sky model fitting.

    Returns
    -------
    pfsArm : `pfs.datamodel.PfsArm`
        Sky-subtracted spectra, returning only the spectrum corresponding to `excludeFiberId`.

    Raises
    ------
    RuntimeError
        If no valid sky spectra are available for subtraction.

    Notes
    -----
    - Uses `SelectFibersTask` to select sky fibers while excluding `excludeFiberId`.
    - Fits a sky model using `FitBlockedOversampledSplineTask`.
    - Applies sky subtraction using `subtractSky1d`.
    """
    # Retrieve spectra and configuration from Butler
    spectra = butler.get('pfsArm', dataId)
    pfsConfig = butler.get('pfsConfig', dataId)

    # Select sky fibers, excluding the specified fiber
    selectSky = SelectFibersTask()
    selectSky.config.targetType = ("SKY",)  # Selecting only sky fibers
    skyConfig = selectSky.run(pfsConfig.select(fiberId=spectra.fiberId))

    # Remove the excluded fiber from the sky selection
    skyConfig = skyConfig[skyConfig.fiberId != excludeFiberId]

    # Extract spectra for the selected sky fibers
    skySpectra = spectra.select(pfsConfig, fiberId=skyConfig.fiberId)

    # Ensure excluded fiber is not part of the selected sky spectra
    assert excludeFiberId not in skySpectra.fiberId

    # If no sky spectra are available, raise an error
    if len(skySpectra) == 0:
        raise RuntimeError("No sky spectra to use for sky subtraction")

    # Fit sky model using the given configuration
    fitSkyModel = FitBlockedOversampledSplineTask(config=fitSkyModelConfig)
    sky1d = fitSkyModel.run(skySpectra, skyConfig)

    # Apply sky subtraction to the full spectra
    subtractSky1d(spectra, pfsConfig, sky1d)

    # Return only the sky-subtracted spectrum of the excluded fiber
    return spectra[spectra.fiberId == excludeFiberId]


def getSkyFiberIds(butler, dataId):
    """Retrieve the fiber IDs of sky fibers for a given dataset.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Data butler used to retrieve `pfsConfig` and `pfsArm` data.
    dataId : `dict`
        Identifier for the data to retrieve (e.g., visit, spectrograph, arm).

    Returns
    -------
    fiberIds : `numpy.ndarray`
        Array of fiber IDs that are classified as sky fibers.

    Notes
    -----
    - Uses `SelectFibersTask` to select sky fibers.
    - Only fibers present in `pfsArm` are considered.
    """
    pfsConfig = butler.get('pfsConfig', dataId)
    spectra = butler.get('pfsArm', dataId)

    # Select sky fibers
    selectSky = SelectFibersTask()
    selectSky.config.targetType = ("SKY",)
    skyConfig = selectSky.run(pfsConfig.select(fiberId=spectra.fiberId))

    return skyConfig.fiberId


def runLeaveOneOutSkyResiduals(butler, dataId, fitSkyModelConfig):
    """Generate residual spectra using a leave-one-out sky subtraction approach.

    This function performs sky subtraction multiple times,
    leaving out one sky fiber at a time and computing the residuals.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Data butler used to retrieve `pfsArm` and `pfsConfig` data.
    dataId : `dict`
        Identifier for the data to retrieve (e.g., visit, spectrograph, arm).
    fitSkyModelConfig : `FitBlockedOversampledSplineConfig`
        Configuration for sky model fitting.

    Returns
    -------
    pfsArm : `pfs.datamodel.PfsArm`
        A merged `PfsArm` object containing the sky-subtracted residuals.

    Notes
    -----
    - Uses `getSkyFiberIds` to determine sky fibers.
    - Calls `subtractSkyWithExclusion` for each sky fiber to compute residuals.
    - Uses `FitBlockedOversampledSplineConfig` to configure sky fitting.
    """
    # Get all sky fiber IDs
    skyFiberIds = getSkyFiberIds(butler, dataId)

    # List to store the sky-subtracted spectra
    spectras = []

    # Perform sky subtraction for each excluded fiber
    for excludeFiberId in skyFiberIds:
        spectras.append(subtractSkyWithExclusion(butler, dataId, excludeFiberId, fitSkyModelConfig=fitSkyModelConfig))

    # Merge the individual residual spectra into a single `PfsArm` object
    return PfsArm.fromMerge(spectras)


def extractFiber(spectra, fiberId, finite=True):
    """
    Extract relevant spectral data for a specific fiber.

    This function retrieves wavelength, flux, sky background, variance, and 
    computes standard deviation and chi values for the given fiber.

    Parameters
    ----------
    spectra : `pfs.datamodel.PfsArm`
        PFS spectra object containing spectral data.
    fiberId : `int`
        The fiber ID for which the data needs to be extracted.
    finite : `bool`, optional
        If True, only finite values (non-NaN, positive sky, and positive variance) are returned.

    Returns
    -------
    wave : `numpy.ndarray`
        Wavelength array for the selected fiber.
    flux : `numpy.ndarray`
        Flux values for the selected fiber.
    std : `numpy.ndarray`
        Standard deviation (sqrt of variance) for the selected fiber.
    sky : `numpy.ndarray`
        Sky background values for the selected fiber.
    chi : `numpy.ndarray`
        Chi values computed as flux / sqrt(variance).
    C : `numpy.ndarray`
        Boolean mask indicating valid (finite) data points.

    Notes
    -----
    - The function applies filtering to ensure valid data: sky > 0 and variance > 0.
    - If `finite` is True, only valid data points are returned.
    """
    # Identify the index of the fiber in the spectra
    j = spectra.fiberId == fiberId

    # Retrieve corresponding arrays
    wave = spectra.wavelength[j][0]
    flux = spectra.flux[j][0]
    sky = spectra.sky[j][0]
    var = spectra.variance[j][0]

    # Define a validity mask: data must be finite, with positive sky & variance
    C = np.isfinite(flux) & (sky > 0) & (var > 0)
    # adding masked values
    C = np.logical_and(C, spectra.mask[j][0] == 0)

    # Compute standard deviation (sqrt of variance), setting invalid values to NaN
    std = np.ones_like(var)
    std[C] = np.sqrt(var[C])
    std[~C] = np.nan

    # Compute chi (flux divided by standard deviation), handling invalid cases
    chi = np.ones_like(var)
    chi[C] = flux[C] / np.sqrt(var)[C]
    chi[~C] = np.nan

    # Return only finite values if requested
    if finite:
        wave, flux, std, sky, chi = wave[C], flux[C], std[C], sky[C], chi[C]

    return wave, flux, std, sky, chi, C


def buildReference(spectra, func=np.mean, model='residuals'):
    """
    Build a reference spectrum by aggregating spectral data from multiple fibers.

    The reference spectrum is constructed by applying a specified aggregation function 
    (e.g., mean, median) to the selected model across all fibers.

    Parameters
    ----------
    spectra : `pfs.datamodel.PfsArm`
        PFS spectra object containing spectral data for multiple fibers.
    func : `callable` or `str`, optional
        Function used to aggregate the spectra (default: `numpy.mean`).
        - If 'quadrature', computes the quadrature sum (sqrt of sum of squares).
    model : `str`, optional
        Specifies which data to use when building the reference spectrum.
        Options:
        - 'none'        : Total flux (sky + object flux).
        - 'sky'         : Sky model.
        - 'chi'         : Chi (flux / standard deviation).
        - 'chi_poisson' : Poissonian chi (flux / sqrt(sky + flux)).
        - 'residuals'   : Residual flux (default).
        - 'variance'    : Variance.
        - 'sky_chi'     : (sky + flux) / standard deviation.

    Returns
    -------
    wave_ref : `numpy.ndarray`
        Reference wavelength array.
    sky_ref : `numpy.ndarray`
        Aggregated reference spectrum based on the chosen model.

    Notes
    -----
    - The function extracts and aligns spectra from all fibers to a common wavelength grid.
    - Uses interpolation to map each fiber's spectrum onto the reference wavelength array.
    - Applies the selected aggregation function (`func`) to compute the final reference spectrum.
    """
    # Containers for spectral data
    x, y = [], []

    # Process each fiber
    for fiberId in spectra.fiberId:
        wave, flux, std, sky, chi, C = extractFiber(spectra, fiberId=fiberId, finite=True)

        # Select model to build reference spectrum
        if model == 'none':
            y.append(sky + flux)
        elif model == 'sky':
            y.append(sky)
        elif model == 'chi':
            y.append(chi)
        elif model == 'chi_poisson':
            y.append(flux / np.sqrt(sky + flux))
        elif model == 'residuals':
            y.append(flux)
        elif model == 'variance':
            y.append(std ** 2)
        elif model == 'sky_chi':
            y.append((sky + flux) / std)
        else:
            raise ValueError(
                "Unsupported model. Choose from [residuals, chi, sky, variance, none, chi_poisson, sky_chi]"
            )

        x.append(wave)

    # Choose the longest wavelength grid as the reference
    wave_ref = x[np.argmax([len(xi) for xi in x])]

    # Interpolate all spectra to the reference wavelength grid
    sky_ref = [np.interp(wave_ref, wave, sky) for wave, sky in zip(x, y)]

    # Apply the aggregation function to compute final reference spectrum
    if func:
        if func == 'quadrature':
            sky_ref = np.sqrt(np.sum(np.array(sky_ref) ** 2, axis=0))
        else:
            sky_ref = func(np.array(sky_ref), axis=0)

    return wave_ref, sky_ref


def splitSpectraIntoReferenceAndTest(spectra, referenceFraction=0.1):
    """
    Randomly split spectra into reference (10%) and test (90%) subsets.

    This function randomly selects a subset of spectra corresponding to the given `referenceFraction`
    (default: 10%) and assigns the remaining spectra to the test subset.

    Parameters
    ----------
    spectra : `pfs.datamodel.PfsArm`
        The PFS spectra object containing spectral data for multiple fibers.
    referenceFraction : `float`, optional
        The fraction of spectra to include in the reference subset (default: 0.1).
        Must be between 0 and 1.

    Returns
    -------
    referenceSpectra : `pfs.datamodel.PfsArm`
        A subset containing `referenceFraction` of the spectra, used as the reference.
    testSpectra : `pfs.datamodel.PfsArm`
        The remaining `1 - referenceFraction` of the spectra, used for testing.

    Notes
    -----
    - The function shuffles fiber IDs before splitting to ensure randomness.
    - Uses `np.isin` to efficiently filter the spectra based on fiber ID.
    - The sum of spectra in both subsets equals the original spectra.
    """
    if not (0 < referenceFraction < 1):
        raise ValueError("referenceFraction must be between 0 and 1.")

    # Randomly shuffle fiber IDs
    shuffled_fiber_ids = np.random.choice(spectra.fiberId, size=len(spectra), replace=False)

    # Split at the reference fraction point
    split_idx = int(len(shuffled_fiber_ids) * referenceFraction)
    reference_fiber_ids = shuffled_fiber_ids[:split_idx]
    test_fiber_ids = shuffled_fiber_ids[split_idx:]

    # Filter spectra based on selected fiber IDs
    referenceSpectra = spectra[np.isin(spectra.fiberId, reference_fiber_ids)]
    testSpectra = spectra[np.isin(spectra.fiberId, test_fiber_ids)]

    return referenceSpectra, testSpectra


def getStdev(x, axis=0, useIQR=True):
    """
    Compute the standard deviation of an array using either the interquartile range (IQR)
    or the standard deviation method.

    Parameters
    ----------
    x : `numpy.ndarray`
        Input array for which the standard deviation is computed.
    axis : `int`, optional
        Axis along which the standard deviation is computed (default: 0).
    useIQR : `bool`, optional
        If `True`, computes a robust estimate of standard deviation using the IQR method.
        If `False`, computes the standard deviation using `np.std` (default: True).

    Returns
    -------
    stdev : `float` or `numpy.ndarray`
        Estimated standard deviation along the specified axis.

    Notes
    -----
    - The IQR-based estimator uses: `alpha = 0.741 * (Q3 - Q1)`, where:
      - Q1 is the first quartile (25th percentile)
      - Q3 is the third quartile (75th percentile)
      - 0.741 is a conversion factor to approximate standard deviation for a normal distribution.
    - If `useIQR=False`, it falls back to the standard deviation computed via `np.std`.

    """
    if useIQR:
        q1, q3 = np.nanpercentile(x, [25, 75], axis=axis)
        alpha = 0.741 * (q3 - q1)
        return alpha
    else:
        return np.nanstd(x, axis=axis)  # Using nanstd to ignore NaNs


def convertToDict(hold, finite=True):
    """
    Convert spectral data into a structured dictionary format.

    This function processes PFS spectral data and organizes it into a nested dictionary,
    where each entry corresponds to a spectrograph arm and fiber ID, storing relevant spectral 
    properties such as wavelength, flux, standard deviation, sky background, and chi values.

    Parameters
    ----------
    hold : `dict`
        Dictionary containing spectral data indexed by `(spectrograph, arm)`.
        Expected to contain a `pfsConfig` key for positional metadata.
    finite : `bool`, optional
        If `True`, filters out non-finite values (NaN, invalid sky, invalid variance) (default: `True`).

    Returns
    -------
    ret : `dict`
        Nested dictionary with structure:
        ```
        {
            (spectrograph, arm): {
                fiberId: {
                    'wave'  : numpy.ndarray,  # Wavelength values
                    'flux'  : numpy.ndarray,  # Flux values
                    'std'   : numpy.ndarray,  # Standard deviation of flux
                    'sky'   : numpy.ndarray,  # Sky background values
                    'chi'   : numpy.ndarray,  # Chi values (flux/std)
                    'xy'    : tuple(float, float),  # PFI nominal position (x, y)
                    'ra_dec': tuple(float, float)   # Sky coordinates (RA, Dec)
                },
                ...
            },
            ...
        }
        ```

    Notes
    -----
    - Uses `extractFiber` to extract spectral information for each fiber.
    - Each spectrograph arm has its own dictionary containing fiber-specific data.
    - If `pfsConfig` is not found in `hold`, fiber positions (`xy`, `ra_dec`) will be omitted.

    """
    hold = hold.copy()  # Prevent modification of the original input
    pfsConfig = hold.pop('pfsConfig', None)  # Extract pfsConfig metadata if available

    ret = {}  # Dictionary to store processed data

    # Iterate over spectrograph-arm combinations
    for (spectrograph, arm), spectra in hold.items():
        ret[(spectrograph, arm)] = {}

        # Process each fiber
        for iFib, fiberId in enumerate(spectra.fiberId):
            # Extract spectral data
            wave, flux, std, sky, chi, _ = extractFiber(spectra, fiberId=fiberId, finite=finite)

            # Initialize fiber entry
            ret[(spectrograph, arm)][fiberId] = {
                'wave': wave,
                'flux': flux,
                'std': std,
                'sky': sky,
                'chi': chi
            }

            # Add positional metadata if `pfsConfig` is available
            if pfsConfig:
                ret[(spectrograph, arm)][fiberId].update({
                    'xy': tuple(pfsConfig.pfiNominal[iFib]),  # PFI nominal position (x, y)
                    'ra_dec': (pfsConfig.ra[iFib], pfsConfig.dec[iFib])  # Sky coordinates (RA, Dec)
                })

    return ret


def rolling(x, y, sep):
    """
    Compute a rolling statistic over a dataset, binning data points into segments of size `sep`.

    Parameters
    ----------
    x : `numpy.ndarray`
        The independent variable (e.g., wavelength, time).
    y : `numpy.ndarray`
        The dependent variable (e.g., flux, intensity).
    sep : `float`
        The bin width for segmenting `x`.

    Returns
    -------
    xw : `numpy.ndarray`
        The center points of the bins.
    yw : `numpy.ndarray`
        The median of `y` values in each bin.
    ew : `numpy.ndarray`
        The computed standard deviation (or IQR-based metric) for `y` in each bin.

    Notes
    -----
    - The function slides over `x` in steps of `sep`, computing statistics for each window.
    - The rolling statistic is defined as:
        - `yw`: Median of `y` in the bin.
        - `ew`: Standard deviation (or IQR-based deviation if `get_stdev_func` is IQR-based).
    """
    x0 = x.min()
    xw, yw, ew = [], [], []

    while x0 + sep / 2 < x.max():
        # Define the bin mask
        C = (x > x0) & (x <= x0 + sep)

        if np.any(C):  # Ensure there are valid points in the bin
            xw.append(x0 + sep / 2)  # Bin center
            yw.append(np.median(y[C]))  # Median value of y in the bin
            ew.append(getStdev(y[C]))  # Custom standard deviation function

        x0 += sep  # Move to the next bin

    return np.array(xw), np.array(yw), np.array(ew)
