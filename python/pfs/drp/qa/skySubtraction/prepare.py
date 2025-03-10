import numpy as np
from pfs.drp.qa.skySubtraction.skySubtractionQa import runLeaveOneOutSkyResiduals, convertToDict
from lsst.daf.butler import Butler


def prepareDataset(collection, dataId, arms, fitSkyModelConfig):
    """
    Prepare a dataset for sky subtraction QA by running leave-one-out sky residuals.

    This function retrieves spectral data for the given `arms`, performs sky subtraction
    while leaving out one sky fiber at a time, and stores the results in a dictionary.

    Parameters
    ----------
    collection : `str`
        Butler collection name.
    dataId : `dict`
        Dictionary containing data identifiers (e.g., `visit`, `spectrograph`).
    arms : `list` of `str`
        List of spectral arms to process (e.g., `['b', 'r', 'n']`).
    fitSkyModelConfig : `FitBlockedOversampledSplineConfig`
        Configuration object for sky model fitting.

    Returns
    -------
    hold : `dict`
        Dictionary containing processed spectra per spectral arm.
    holdAsDict : `dict`
        Dictionary version of `hold`, with per-fiber data structures.
    plotId : `dict`
        Updated `dataId` with additional metadata for plotting.

    Notes
    -----
    - Uses `runLeaveOneOutSkyResiduals` to compute sky residuals per arm.
    - Retrieves the `pfsConfig` and filters it to match the extracted fibers.
    - Converts the dataset to a dictionary format for easier access.
    """
    butler = Butler('/work/datastore', collections=[collection])

    # Ensure spectrograph key exists in dataId
    if 'spectrograph' not in dataId:
        raise ValueError("Missing 'spectrograph' key in dataId")

    spectrograph = dataId['spectrograph']
    hold = {}

    # Process each arm separately
    for arm in arms:
        armDataId = dataId.copy()  # Avoid modifying the original dataId
        armDataId.update(arm=arm)

        hold[(spectrograph, arm)] = runLeaveOneOutSkyResiduals(
            butler, armDataId, fitSkyModelConfig=fitSkyModelConfig
        )

    # Retrieve and filter pfsConfig to match extracted fibers
    pfsConfig = butler.get('pfsConfig', dataId)
    pfsConfig = pfsConfig[np.isin(pfsConfig.fiberId, hold[(spectrograph, arm)].fiberId)]
    hold['pfsConfig'] = pfsConfig

    # Convert data structure for easier access
    holdAsDict = convertToDict(hold)

    # Prepare plotting metadata
    plotId = dataId.copy()
    plotId['block'] = fitSkyModelConfig.blockSize

    return hold, holdAsDict, plotId
