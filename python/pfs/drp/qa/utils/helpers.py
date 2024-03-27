import warnings
from contextlib import suppress

from functools import partial
from pathlib import Path
from dataclasses import dataclass

import lsst.daf.persistence as dafPersist

import numpy as np
import pandas as pd
from scipy.optimize import bisect
from scipy.stats import iqr
from astropy.stats import sigma_clip

from pfs.datamodel import TargetType
from pfs.drp.stella import ArcLineSet, DetectorMap, ReferenceLineStatus


warnings.filterwarnings('ignore', message='Input data contains invalid values')
warnings.filterwarnings('ignore', message='Gen2 Butler')
warnings.filterwarnings('ignore', message='addPfsCursor')
warnings.filterwarnings('ignore', message='All-NaN slice')
warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='This figure')


@dataclass
class FitStat:
    median: float
    robustRms: float
    weightedRms: float
    softenFit: float
    dof: float
    num_fibers: int
    num_lines: int

    def __str__(self):
        return f'''median      = {self.median:>10.05f}
robustRms   = {self.robustRms:>10.05f}
weightedRms = {self.weightedRms:>10.05f}
soften      = {self.softenFit:>10.05f}
fibers      = {self.num_fibers:>10d}
lines       = {self.num_lines:>10d}
'''


@dataclass
class FitStats:
    dof: int
    chi2: float
    spatial: FitStat
    wavelength: FitStat

    def to_dict(self):
        """Output as dict."""
        return dict(
            dof=self.dof,
            chi2=self.chi2,
            spatial=self.spatial.__dict__,
            wavelength=self.wavelength.__dict__
        )


def iqr_sigma(x) -> float:
    """Calculate the sigma of the interquartile range as a robust estimate of the std.

    Note: This will ignore NaNs.

    Parameters
    ----------
    x : `numpy.ndarray`
        The data.


    Returns
    -------
    sigma : `float`
        The sigma of the interquartile range.
    """
    return iqr(x, nan_policy='omit') / 1.349


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


def loadData(
        arcLines: ArcLineSet,
        detectorMap: DetectorMap,
        dropNaColumns: bool = True,
        **kwargs) -> pd.DataFrame:
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
    arcData = getArclineData(arcLines, detectorMap, dropNaColumns=dropNaColumns, **kwargs)

    # Mark the sigma-clipped outliers for each relevant group.
    def maskOutliers(grp):
        grp['xResidOutlier'] = sigma_clip(grp.xResid).mask
        grp['yResidOutlier'] = sigma_clip(grp.yResid).mask
        return grp

    arcData = arcData.groupby(['status_type', 'isLine']).apply(maskOutliers)
    arcData.reset_index(drop=True, inplace=True)

    return arcData


def getArclineData(
        arcLines: ArcLineSet,
        detectorMap: DetectorMap,
        dropNaColumns: bool = False,
        removeFlagged: bool = True,
        onlyReservedAndUsed: bool = True) -> pd.DataFrame:
    """Gets a copy of the arcline data, with some columns added.

    Parameters
    ----------
    arcLines : `ArcLineSet`
        The arc lines.
    dropNaColumns : `bool`, optional
        Drop columns where all values are NaN. Default is True.
    removeFlagged : `bool`, optional
        Remove rows with ``flag=True``? Default is True.

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """

    isTrace = arcLines.description == "Trace"
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

    arcLines.data['isTrace'] = isTrace
    arcLines.data['isLine'] = isLine
    arcLines.data['xModel'] = fitPosition[:, 0]
    arcLines.data['yModel'] = fitPosition[:, 1]

    arcLines.data['xResid'] = arcLines.data.x - arcLines.data.xModel
    arcLines.data['yResid'] = arcLines.data.y - arcLines.data.yModel

    # Copy the dataframe from the arcline set.
    arc_data = arcLines.data.copy()

    # Convert nm to pixels.
    arc_data['dispersion'] = detectorMap.getDispersion(arcLines.fiberId, arcLines.wavelength)

    if removeFlagged:
        arc_data = arc_data.query('flag == False').copy()

    # Get USED and RESERVED status.
    is_reserved = (arc_data.status & ReferenceLineStatus.DETECTORMAP_RESERVED) != 0
    is_used = (arc_data.status & ReferenceLineStatus.DETECTORMAP_USED) != 0

    # Make one-hot columns for status_names.
    arc_data.loc[:, 'isUsed'] = is_used
    arc_data.loc[:, 'isReserved'] = is_reserved
    arc_data.loc[arc_data.isReserved, 'status_type'] = 'RESERVED'
    arc_data.loc[arc_data.isUsed, 'status_type'] = 'USED'

    # Filter to only the RESERVED and USED data.
    if onlyReservedAndUsed is True:
        arc_data = arc_data[is_used | is_reserved]

    # Drop empty rows.
    if dropNaColumns:
        arc_data = arc_data.dropna(axis=0, how='all')

        # Drop rows without enough info in position.
        arc_data = arc_data.dropna(subset=['x', 'y'])

    # Change some of the dtypes explicitly.
    try:
        arc_data.y = arc_data.y.astype(np.float64)
    except AttributeError:
        pass

    # Replace inf with nans.
    arc_data = arc_data.replace([np.inf, -np.inf], np.nan)

    # Get full status names. (the .name attribute doesn't work properly so need the str instance)
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
        The arcLines data.

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
        arc_data: pd.DataFrame,
        xSoften: float = 0.,
        ySoften: float = 0.,
        numParams: int = 0,
        maxSoften: float = 1.,
        sigmaClipOnly: bool = True) -> FitStats:
    """Get the fit stats."""
    traces = arc_data.query('isTrace == True')
    lines = arc_data.query('isLine == True')

    if sigmaClipOnly is True:
        traces = traces.query('xResidOutlier == False')
        lines = lines.query('yResidOutlier == False')

    xNum = len(traces)
    try:
        yNum = lines.isLine.value_counts()[True]
    except KeyError:
        yNum = np.nan

    xWeightedRms = getWeightedRMS(traces.xResid, traces.xErr, soften=xSoften)
    yWeightedRms = getWeightedRMS(lines.yResid, lines.yErr, soften=ySoften)

    xRobustRms = iqr_sigma(traces.xResid)
    yRobustRms = iqr_sigma(lines.yResid)

    chi2X = getChi2(traces.xResid, traces.xErr, xSoften)
    chi2Y = getChi2(lines.yResid, lines.yErr, ySoften)
    chi2 = chi2X + chi2Y

    xDof = xNum - numParams / 2
    yDof = yNum - numParams / 2
    dof = xDof + yDof

    def getSoften(resid, err, dof, soften=0):
        return getChi2(resid, err, soften) / dof - 1

    f_x = partial(getSoften, traces.xResid, traces.xErr, xDof)
    f_y = partial(getSoften, lines.yResid, lines.yErr, yDof)

    if f_x(0) < 0:
        xSoftFit = 0.
    elif f_x(maxSoften) > 0:
        xSoftFit = np.nan
    else:
        xSoftFit = bisect(f_x, 0, 1)

    if f_y(0) < 0:
        ySoftFit = 0.
    elif f_y(maxSoften) > 0:
        ySoftFit = np.nan
    else:
        ySoftFit = bisect(f_y, 0, 1)

    xFibers = len(traces.fiberId.unique())
    yFibers = len(lines.fiberId.unique())

    xFitStat = FitStat(traces.xResid.median(), xRobustRms, xWeightedRms, xSoftFit, xDof, xFibers, xNum)
    yFitStat = FitStat(lines.yResid.median(), yRobustRms, yWeightedRms, ySoftFit, yDof, yFibers, yNum)

    return FitStats(dof, chi2, xFitStat, yFitStat)


def getWeightedRMS(resid, err, soften=0):
    """Small helper function to get the weighted RMS with optional softening."""
    weight = 1 / (err ** 2 + soften ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.sum(weight * resid ** 2) / np.sum(weight))


def getChi2(resid, err, soften=0):
    """Small helper function to get the chi2 with optional softening."""
    return (resid**2 / (err ** 2 + soften ** 2)).sum()
