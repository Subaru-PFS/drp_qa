from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import Iterable

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from lsst.daf.persistence import NoResults
from pfs.drp.stella import ArcLineSet, DetectorMap, ReferenceLineStatus
from pfs.drp.stella.utils.math import robustRms
from pfs.utils.fiberids import FiberIds
from scipy.optimize import bisect

from pfs.drp.qa.utils.math import getChi2, getWeightedRMS


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
        return f"""median  = {self.median:> 7.05f}
rms     = {self.weightedRms:> 7.05f}
soften  = {self.softenFit:> 7.05f}
fibers  = {self.num_fibers:>8d}
lines   = {self.num_lines:>8d}
"""


@dataclass
class FitStats:
    dof: int
    chi2X: float
    chi2Y: float
    spatial: FitStat
    wavelength: FitStat

    def to_dict(self):
        """Output as dict."""
        return dict(
            dof=self.dof,
            chi2X=self.chi2X,
            chi2Y=self.chi2Y,
            spatial=self.spatial.__dict__,
            wavelength=self.wavelength.__dict__,
        )


def loadData(
    arcLines: ArcLineSet,
    detectorMap: DetectorMap,
    dropNaColumns: bool = True,
    addFiberInfo: bool = True,
    **kwargs,
) -> pd.DataFrame:
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
    addFiberInfo : `bool`, optional
        Add fiber information to the dataframe. Default is True.

    Returns
    -------
    arcData : `pandas.DataFrame`
    """

    # Get dataframe for arc lines and add detectorMap information, then calculate residuals.
    arcData = getArclineData(arcLines, detectorMap, dropNaColumns=dropNaColumns, **kwargs)

    # Mark the sigma-clipped outliers for each relevant group.
    def maskOutliers(grp):
        grp["xResidOutlier"] = sigma_clip(grp.xResid).mask
        grp["yResidOutlier"] = sigma_clip(grp.yResid).mask
        return grp

    arcData = arcData.groupby(["status_type", "isLine"]).apply(maskOutliers)
    arcData.reset_index(drop=True, inplace=True)

    if addFiberInfo is True:
        mtp_df = pd.DataFrame(
            FiberIds().fiberIdToMTP(detectorMap.fiberId), columns=["mtpId", "mtpHoles", "cobraId"]
        )
        mtp_df.index = detectorMap.fiberId
        mtp_df.index.name = "fiberId"
        arcData = arcData.merge(mtp_df.reset_index(), on="fiberId")

    return arcData


def getArclineData(
    arcLines: ArcLineSet,
    detectorMap: DetectorMap,
    dropNaColumns: bool = False,
    removeFlagged: bool = True,
    onlyReservedAndUsed: bool = True,
) -> pd.DataFrame:
    """Gets a copy of the arcline data, with some columns added.

    Parameters
    ----------
    arcLines : `ArcLineSet`
        The arc lines.
    detectorMap : `DetectorMap`
        The detector map.
    dropNaColumns : `bool`, optional
        Drop columns where all values are NaN. Default is True.
    removeFlagged : `bool`, optional
        Remove rows with ``flag=True``? Default is True.
    onlyReservedAndUsed : `bool`, optional
        Only include rows with status RESERVED or USED? Default is True.

    Returns
    -------
    arc_data : `pandas.DataFrame`
    """
    isTrace = arcLines.description == "Trace"
    isLine = ~isTrace

    fitPosition = np.full((len(arcLines), 2), np.nan, dtype=float)

    if isLine.any():
        fitPosition[isLine] = detectorMap.findPoint(arcLines.fiberId[isLine], arcLines.wavelength[isLine])
    if isTrace.any():
        fitPosition[isTrace, 0] = detectorMap.getXCenter(arcLines.fiberId[isTrace], arcLines.y[isTrace])
        fitPosition[isTrace, 1] = np.nan

    arcLines.data["isTrace"] = isTrace
    arcLines.data["isLine"] = isLine
    arcLines.data["xModel"] = fitPosition[:, 0]
    arcLines.data["yModel"] = fitPosition[:, 1]

    arcLines.data["xResid"] = arcLines.data.x - arcLines.data.xModel
    arcLines.data["yResid"] = arcLines.data.y - arcLines.data.yModel

    # Copy the dataframe from the arcline set.
    arc_data = arcLines.data.copy()

    # Convert nm to pixels.
    arc_data["dispersion"] = detectorMap.getDispersion(arcLines.fiberId, arcLines.wavelength)

    if removeFlagged:
        arc_data = arc_data.query("flag == False").copy()

    # Get USED and RESERVED status.
    is_reserved = (arc_data.status & ReferenceLineStatus.DETECTORMAP_RESERVED) != 0
    is_used = (arc_data.status & ReferenceLineStatus.DETECTORMAP_USED) != 0

    # Make one-hot columns for status_names.
    arc_data.loc[:, "isUsed"] = is_used
    arc_data.loc[:, "isReserved"] = is_reserved
    arc_data.loc[arc_data.isReserved, "status_type"] = "RESERVED"
    arc_data.loc[arc_data.isUsed, "status_type"] = "USED"

    # Filter to only the RESERVED and USED data.
    if onlyReservedAndUsed is True:
        arc_data = arc_data[is_used | is_reserved]

    # Drop empty rows.
    if dropNaColumns:
        arc_data = arc_data.dropna(axis=0, how="all")

        # Drop rows without enough info in position.
        arc_data = arc_data.dropna(subset=["x", "y"])

    # Change some of the dtypes explicitly.
    try:
        arc_data.y = arc_data.y.astype(np.float64)
    except AttributeError:
        pass

    # Replace inf with nans.
    arc_data = arc_data.replace([np.inf, -np.inf], np.nan)

    # Get full status names. (the .name attribute doesn't work properly so need the str instance)
    arc_data["status_name"] = arc_data.status.map(lambda x: str(ReferenceLineStatus(x)).split(".")[-1])
    arc_data["status_name"] = arc_data["status_name"].astype("category")
    arc_data.status_name = arc_data.status_name.cat.remove_unused_categories()

    return arc_data


def getFitStats(
    arc_data: pd.DataFrame,
    xSoften: float = 0.0,
    ySoften: float = 0.0,
    numParams: int = 0,
    maxSoften: float = 1.0,
    sigmaClipOnly: bool = True,
) -> FitStats:
    """Get the fit stats.

    Parameters
    ----------
    arc_data : `pandas.DataFrame`
        The arc data.
    xSoften : `float`, optional
        The softening parameter for the x residuals. Default is 0.0.
    ySoften : `float`, optional
        The softening parameter for the y residuals. Default is 0.0.
    numParams : `int`, optional
        The number of parameters in the model. Default is 0.
    maxSoften : `float`, optional
        The maximum value for the softening parameter. Default is 1.0.
    sigmaClipOnly : `bool`, optional
        Only include non-outliers in the fit stats. Default is True.

    Returns
    -------
    fitStats : `FitStats`
    """
    traces = arc_data.query("isTrace == True")
    lines = arc_data.query("isLine == True").dropna(subset=["yResid"])

    if sigmaClipOnly is True:
        arc_data = arc_data.query("xResidOutlier == False")
        traces = traces.query("xResidOutlier == False")
        lines = lines.query("yResidOutlier == False")

    xNum = len(arc_data)
    try:
        yNum = lines.isLine.value_counts()[True]
    except KeyError:
        yNum = 0

    xWeightedRms = getWeightedRMS(arc_data.xResid, arc_data.xErr, soften=xSoften)
    yWeightedRms = getWeightedRMS(lines.yResid, lines.yErr, soften=ySoften)

    def doRobust(x):
        try:
            return robustRms(x.dropna())
        except (IndexError, ValueError):
            return np.nan

    xRobustRms = doRobust(arc_data.xResid)
    yRobustRms = doRobust(lines.yResid)

    chi2X = getChi2(arc_data.xResid, arc_data.xErr, xSoften)
    chi2Y = getChi2(lines.yResid, lines.yErr, ySoften)

    xDof = xNum - numParams / 2
    yDof = yNum - numParams / 2
    dof = xDof + yDof

    def getSoften(resid, err, dof, soften=0):
        if len(resid) == 0:
            return 0
        with np.errstate(invalid="ignore"):
            return (getChi2(resid, err, soften) / dof) - 1

    f_x = partial(getSoften, arc_data.xResid, arc_data.xErr, xDof)
    f_y = partial(getSoften, lines.yResid, lines.yErr, yDof)

    if f_x(0) < 0:
        xSoftFit = 0.0
    elif f_x(maxSoften) > 0:
        xSoftFit = np.nan
    else:
        xSoftFit = bisect(f_x, 0, maxSoften)

    if f_y(0) < 0:
        ySoftFit = 0.0
    elif f_y(maxSoften) > 0:
        ySoftFit = np.nan
    else:
        ySoftFit = bisect(f_y, 0, maxSoften)

    xFibers = len(traces.fiberId.unique())
    yFibers = len(lines.fiberId.unique())

    xFitStat = FitStat(arc_data.xResid.median(), xRobustRms, xWeightedRms, xSoftFit, xDof, xFibers, xNum)
    yFitStat = FitStat(lines.yResid.median(), yRobustRms, yWeightedRms, ySoftFit, yDof, yFibers, yNum)

    return FitStats(dof, chi2X, chi2Y, xFitStat, yFitStat)


def getStats(
    arcLinesSet: Iterable[ArcLineSet], detectorMaps: Iterable[DetectorMap], dataIds: Iterable[dict]
) -> tuple:
    """Get the stats for the arc lines.

    Parameters
    ----------
    arcLinesSet : `Iterable[ArcLineSet]`
        The arc lines.
    detectorMaps : `Iterable[DetectorMap]`
        The detector maps.
    dataIds : `Iterable[dict]`
        The data IDs.

    Returns
    -------
    all_arc_data : `pandas.DataFrame`
    all_visit_stats : `pandas.DataFrame`
    all_detector_stats : `pandas.DataFrame`
    """
    all_data = list()
    visit_stats = list()
    detector_stats = list()

    all_arc_data = None
    all_visit_stats = None
    all_detector_stats = None

    for arcLines, detectorMap, dataId in zip(arcLinesSet, detectorMaps, dataIds):
        try:
            arc_data = loadData(arcLines, detectorMap)

            if len(arc_data) == 0:
                print(f"No data for {dataId}")
                continue

            visit = dataId["visit"]
            arm = dataId["arm"]
            spectrograph = dataId["spectrograph"]
            ccd = f"{arm}{spectrograph}"
            arc_data["visit"] = visit
            arc_data["arm"] = arm
            arc_data["spectrograph"] = spectrograph

            all_data.append(arc_data)

            descriptions = sorted(list(arc_data.description.unique()))
            with suppress(ValueError):
                if len(descriptions) > 1:
                    descriptions.remove("Trace")

            dmap_bbox = detectorMap.getBBox()
            fiberIdMin = detectorMap.fiberId.min()
            fiberIdMax = detectorMap.fiberId.max()
            wavelengthMin = int(arcLines.wavelength.min())
            wavelengthMax = int(arcLines.wavelength.max())

            for idx, rows in arc_data.groupby("status_type"):
                visit_stat = pd.json_normalize(getFitStats(rows).to_dict())
                visit_stat["visit"] = visit
                visit_stat["arm"] = arm
                visit_stat["spectrograph"] = spectrograph
                visit_stat["ccd"] = ccd
                visit_stat["status_type"] = idx
                visit_stat["description"] = ",".join(descriptions)
                visit_stat["detector_width"] = dmap_bbox.width
                visit_stat["detector_height"] = dmap_bbox.height
                visit_stat["fiberId_min"] = fiberIdMin
                visit_stat["fiberId_max"] = fiberIdMax
                visit_stat["wavelength_min"] = wavelengthMin
                visit_stat["wavelength_max"] = wavelengthMax
                visit_stats.append(visit_stat)
        except NoResults:
            print(f"No results found for {dataId}")
        except Exception as e:
            print(e)

    if len(visit_stats):
        all_visit_stats = pd.concat(visit_stats)

    if len(all_data):
        all_arc_data = pd.concat(all_data)

        # Get the stats for the whole detector by status type.
        for status_type, rows in all_arc_data.groupby(["status_type"]):
            try:
                detectorStats = pd.json_normalize(getFitStats(rows).to_dict())
                arm = rows["arm"].iloc[0]
                spectrograph = rows["spectrograph"].iloc[0]
                ccd = f"{arm}{spectrograph}"
                detectorStats["ccd"] = ccd
                detectorStats["status_type"] = status_type
                detectorStats["description"] = "all"
                detector_stats.append(detectorStats)
            except Exception as e:
                print(status_type, e)

        # Get stats for each description type.
        for (status_type, desc), rows in all_arc_data.groupby(["status_type", "description"]):
            try:
                detectorStats = pd.json_normalize(getFitStats(rows).to_dict())
                arm = rows["arm"].iloc[0]
                spectrograph = rows["spectrograph"].iloc[0]
                ccd = f"{arm}{spectrograph}"
                detectorStats["ccd"] = ccd
                detectorStats["status_type"] = status_type
                detectorStats["description"] = desc
                detector_stats.append(detectorStats)
            except Exception as e:
                print(status_type, desc, e)

        all_detector_stats = pd.concat(detector_stats)

    return all_arc_data, all_visit_stats, all_detector_stats
