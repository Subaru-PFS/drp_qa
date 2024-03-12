import warnings
from contextlib import suppress

from pathlib import Path

import lsst.daf.persistence as dafPersist

import numpy as np
import pandas as pd

from pfs.datamodel import TargetType
from pfs.drp.stella import ArcLineSet, DetectorMap, ReferenceLineStatus
from pfs.drp.stella.fitDistortedDetectorMap import calculateFitStatistics

warnings.filterwarnings('ignore', message='Input data contains invalid values')
warnings.filterwarnings('ignore', message='Gen2 Butler')
warnings.filterwarnings('ignore', message='addPfsCursor')
warnings.filterwarnings('ignore', message='All-NaN slice')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='This figure')


def getObjects(dataId: Path,
               rerun: Path,
               calibDir: Path = '/work/drp/CALIB',
               butler: dafPersist.Butler = None) -> (ArcLineSet, DetectorMap):
    """Get the objects from the butler.

    Parameters
    ----------
    dataId : `Path`
        The dataId for the butler.
    rerun : `Path`
        The rerun directory.
    calibDir : `Path`, optional
        The calib directory. Default is '/work/drp/CALIB'.
    butler : `dafPersist.Butler`, optional
        The butler to use. Default is None.

    Returns
    -------
    arcLines : `ArcLineSet`
    detectorMap : `DetectorMap`
    """
    if butler is None:
        butler = dafPersist.Butler(rerun.as_posix(), calibRoot=calibDir.as_posix())

    arcLines = butler.get('arcLines', dataId)
    detectorMap = butler.get('detectorMap_used', dataId)

    return arcLines, detectorMap


def loadData(arcLines: ArcLineSet, detectorMap: DetectorMap, dropNaColumns: bool = True) -> pd.DataFrame:
    """Looks up the data in butler and returns a dataframe with the arcline data.

    The arcline data includes basic statistics, such as the median and sigma of the residuals.

    This method is called on init.

    Parameters
    ----------
    arcLines : `ArcLineSet`
        The arc lines.
    detectorMap : `DetectorMap`
        The detector map.
    dropNaColumns : `bool`, optional
        Drop columns where all values are NaN. Default is True.

    Returns
    -------
    arcData : `pandas.DataFrame`
    """

    # Get dataframe for arc lines and add detectorMap information, then calculate residuals.
    arcData = getArclineData(arcLines, dropNaColumns=dropNaColumns)
    arcData = addTraceLambdaToArclines(arcData, detectorMap)
    arcData = addResidualsToArclines(arcData)
    arcData.reset_index(drop=True, inplace=True)

    return arcData


def getArclineData(arcLines: ArcLineSet,
                   dropNaColumns: bool = False,
                   removeFlagged: bool = True) -> pd.DataFrame:
    """Gets a copy of the arcline data, with some columns added.

    Parameters
    ----------
    arcLines : `ArcLineSet`
        The arc lines.
    dropNaColumns : `bool`, optional
        Drop columns where all values are NaN. Default is True.
    removeFlagged : `bool`, optional
        Remove rows with ``flag=True``? Default is False.

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """
    # Get the data from the ArcLineSet.
    arc_data = arcLines.data.copy()

    if removeFlagged:
        arc_data = arc_data.query('flag == False').copy()

    # Get USED and RESERVED status.
    is_reserved = (arc_data.status & ReferenceLineStatus.DETECTORMAP_RESERVED) != 0
    is_used = (arc_data.status & ReferenceLineStatus.DETECTORMAP_USED) != 0

    # Make one-hot columns for status_names.
    arc_data.loc[:, 'isUsed'] = is_used
    arc_data.loc[:, 'isReserved'] = is_reserved

    # Filter to only the RESERVED and USED data.
    arc_data = arc_data[is_used | is_reserved]

    # Drop empty rows.
    if dropNaColumns:
        arc_data = arc_data.dropna(axis=1, how='all')

    # Drop rows without enough info in position.
    arc_data = arc_data.dropna(subset=['x', 'y'])

    # Change some of the dtypes explicitly.
    arc_data.y = arc_data.y.astype(np.float64)

    # Replace inf with nans.
    arc_data = arc_data.replace([np.inf, -np.inf], np.nan)

    # Get status names. (the .name attribute doesn't work properly so need the str instance)
    arc_data['status_name'] = arc_data.status.map(lambda x: str(ReferenceLineStatus(x)).split('.')[-1])
    arc_data['status_name'] = arc_data['status_name'].astype('category')
    arc_data.status_name = arc_data.status_name.cat.remove_unused_categories()

    # Make a one-hot for the Trace.
    arc_data['isTrace'] = False
    with suppress():
        arc_data.loc[arc_data.query('description == "Trace"').index, 'isTrace'] = True

    return arc_data


def addTraceLambdaToArclines(arc_data: pd.DataFrame, detectorMap: DetectorMap) -> pd.DataFrame:
    """Adds detector map trace position and wavelength to arcline data.

    This will add the following columns to the arcline data:

    - ``lam``: Wavelength according to the detectormap for fiberId.
    - ``lamErr``: Error in wavelength.
    - ``dispersion``: Dispersion at the center of the detector.
    - ``tracePos``: Trace position according to the detectormap.

    Parameters
    ----------
    arc_data : `pandas.DataFrame`
    detectorMap : `DetectorMap`

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """
    # Get the wavelength according to the detectormap for fiberId.
    fiberList = arc_data.fiberId.to_numpy()
    yList = arc_data.y.to_numpy()

    arc_data['lam'] = detectorMap.findWavelength(fiberList, yList)
    arc_data['lamErr'] = arc_data.yErr * arc_data.lam / arc_data.y

    # Convert nm to pixels.
    dispersion = detectorMap.getDispersionAtCenter()
    arc_data['dispersion'] = dispersion

    # Get the trace positions according to the detectormap.
    arc_data['tracePos'] = detectorMap.getXCenter(fiberList, yList)

    return arc_data


def addResidualsToArclines(arc_data: pd.DataFrame) -> pd.DataFrame:
    """Adds residuals to arcline data.

    This will calculate residuals for the X-center position and wavelength.

    Adds the following columns to the arcline data:

    - ``dx``: X-center position minus trace position (from DetectorMap `getXCenter`).
    - ``dy``: Y-center wavelegnth minus trace wavelength (from DetectorMap `findWavelength`).
    - ``dy_nm``: Wavelength minus trace wavelength in nm.
    - ``centroidErr``: Hypotenuse of ``xErr`` and ``yErr``.
    - ``detectorMapErr``: Hypotenuse of ``dx`` and ``dy``.


    Parameters
    ----------
    arc_data : `pandas.DataFrame`

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """
    # Get `observed - expected` for position and wavelength.
    arc_data['dx'] = arc_data.tracePos - arc_data.x
    arc_data['dy_nm'] = arc_data.lam - arc_data.wavelength

    # Set the dy columns to NA (instead of 0) for Trace.
    arc_data.dy_nm = arc_data.apply(lambda row: row.dy_nm if row.isTrace is False else np.NaN, axis=1)

    # Do the dispersion correction to get pixels.
    arc_data['dy'] = arc_data.dy_nm / arc_data.dispersion

    arc_data['centroidErr'] = np.hypot(arc_data.xErr, arc_data.yErr)
    arc_data['detectorMapErr'] = np.hypot(arc_data.dx, arc_data.dy)

    return arc_data


def getTargetType(arc_data, pfsConfig) -> pd.DataFrame:
    """Add targetType to the arcline data.

    Parameters
    ----------
    arc_data : `pandas.DataFrame`
    pfsConfig : `pfs.datamodel.PfsConfig`

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """
    # Add TargetType for each fiber.
    arc_data = arc_data.merge(pd.DataFrame({
        'fiberId': pfsConfig.fiberId,
        'targetType': [str(TargetType(x)) for x in pfsConfig.targetType]
    }), left_on='fiberId', right_on='fiberId')
    arc_data['targetType'] = arc_data.targetType.astype('category')

    return arc_data


def getFitStats(
        arcLines: ArcLineSet,
        detectorMap: DetectorMap,
        selection: 'np.ndarray[bool]',
        numParams: int = 0):
    """Gets output from calculateFitStatistics

    Parameters
    ----------
    arcLines : `ArcLineSet`
        The arc lines.
    detectorMap : `DetectorMap`
        The detector map.
    selection : `np.ndarray[bool]`
        The selection of fibers to use.
    numParams : `int`, optional
        The number of parameters to use. Default is 0.

    Returns
    -------
    stats : `dict`
        The fit statistics.
    """
    isTrace = np.array(arcLines.description == "Trace")
    isLine = ~isTrace

    fitPosition = np.full((len(arcLines), 2), np.nan, dtype=float)

    if isLine.any():
        fitPosition[isLine] = detectorMap.findPoint(
            arcLines.fiberId[isLine],
            arcLines.wavelength[isLine])
    if isTrace.any():
        fitPosition[isTrace, 0] = detectorMap.getXCenter(
            arcLines.fiberId[isTrace], arcLines.y[isTrace])
        fitPosition[isTrace, 1] = np.nan

    return calculateFitStatistics(fitPosition, arcLines, selection, numParams)
