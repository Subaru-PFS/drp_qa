import contextlib
import warnings
from functools import partial
from logging import Logger

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from lsst.afw.image import VisitInfo
from lsst.pex.config import Config, Field
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)
from lsst.pipe.base.connectionTypes import (
    Input as InputConnection,
)
from lsst.pipe.base.connectionTypes import (
    Output as OutputConnection,
)
from scipy.optimize import bisect

from pfs.drp.qa.metrics.fitStats import FitStat, FitStats
from pfs.drp.qa.plotting.dmResiduals import plot_detectormap_residuals, plot_residual
from pfs.drp.qa.utils.math import getChi2, getWeightedRMS
from pfs.drp.stella import ArcLineSet, DetectorMap, ReferenceLineStatus
from pfs.drp.stella.applyExclusionZone import getExclusionZone
from pfs.drp.stella.fitDetectorMap import getDescriptionCounts
from pfs.drp.stella.utils.math import robustRms
from pfs.utils.fiberids import FiberIds

# Re-exported for backwards compatibility: `FitStat`/`FitStats` moved to
# `pfs.drp.qa.metrics.fitStats` and the plotting functions to
# `pfs.drp.qa.plotting.dmResiduals`, where they can be tested without the stack.
# Prefer importing from those modules in new code.
__all__ = [
    "DetectorMapResidualsTask",
    "FitStat",
    "FitStats",
    "getGoodLines",
    "get_data_and_stats",
    "get_fit_stats",
    "plot_detectormap_residuals",
    "plot_residual",
    "scrub_data",
]


class DetectorMapResidualsConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Connections for DetectorMapQaTask."""

    visitInfo = InputConnection(
        name="raw.visitInfo",
        doc="Visit info from the raw exposure",
        storageClass="VisitInfo",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    detectorMap = InputConnection(
        name="detectorMap",
        doc="Adjusted detector mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    arcLines = InputConnection(
        name="lines",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    reduceExposure_config = InputConnection(
        name="reduceExposure_config",
        doc="Configuration for reduceExposure",
        storageClass="Config",
        dimensions=(),
    )

    dmQaResidualData = OutputConnection(
        name="dmQaResidualData",
        doc="The dataframe of the detectormap residuals.",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    dmQaResidualStats = OutputConnection(
        name="dmQaResidualStats",
        doc="Statistics of the DM residual analysis.",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    dmQaResidualPlot = OutputConnection(
        name="dmQaResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for a given visit.",
        storageClass="Plot",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class DetectorMapResidualsConfig(PipelineTaskConfig, pipelineConnections=DetectorMapResidualsConnections):
    """Configuration for DetectorMapQaTask."""

    generatePlot = Field(dtype=bool, default=False, doc="Generate 2D residual plot for visit, default False.")
    useSigmaRange = Field(dtype=bool, default=False, doc="Use ±2.5 sigma as range")
    spatialRange = Field(
        dtype=float, default=0.1, doc="Spatial range for the residual plot, implies useSigmaRange is False."
    )
    wavelengthRange = Field(
        dtype=float,
        default=0.1,
        doc="Wavelegnth range for the residual plot, implies useSigmaRange is False.",
    )
    binWavelength = Field(dtype=float, default=0.1, doc="Wavelength bin for residual plot.")


class DetectorMapResidualsTask(PipelineTask):
    """Task for QA of detectorMap."""

    ConfigClass = DetectorMapResidualsConfig
    _DefaultName = "dmResiduals"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        # Get the dataIds for help with plotting.
        data_id = dict(**inputRefs.arcLines.dataId.mapping)
        data_id["run"] = inputRefs.arcLines.run

        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = data_id

        try:
            # Perform the actual processing.
            outputs = self.run(**inputs)
        except ValueError as e:
            self.log.error(e)
        else:
            # Store the results if valid.
            butlerQC.put(outputs, outputRefs)

    def run(
        self,
        arcLines: ArcLineSet,
        detectorMap: DetectorMap,
        visitInfo: VisitInfo,
        dataId: dict,
        dropNaColumns: bool = True,
        removeOutliers: bool = True,
        addFiberInfo: bool = True,
        reduceExposure_config: Config = None,
        **kwargs,
    ) -> Struct:
        """Clean and mask the data. Adds fiberInfo if requested.

        The arcline data includes basic statistics, such as the median and sigma of the residuals.

        This method is called on init.

        Parameters
        ----------
        arcLines : `ArcLineSet`
            The arc lines.
        detectorMap : `DetectorMap`
            The detector map.
        visitInfo : `VisitInfo`
            The visit info containing the observationReason, which determines
            some plotting parameters.
        dataId : `dict`
            The dataId for the visit.
        dropNaColumns : `bool`, optional
            Drop columns where all values are NaN. Default is True.
        removeOutliers : `bool`, optional
            Remove rows with ``flag=False``? Default is True.
        addFiberInfo : `bool`, optional
            Add fiber information to the dataframe. Default is True.
        reduceExposure_config : `Config`, optional
            Configuration for reduceExposure.

        Returns
        -------
        arc_data : `pandas.DataFrame`
        """
        # Get dataframe for arc lines and add detectorMap information, then calculate residuals.
        self.log.info("Getting and scrubbing the data")
        adjustDM_config = {} if reduceExposure_config is None else reduceExposure_config.adjustDetectorMap

        arc_data, stats = get_data_and_stats(
            dataId,
            arcLines,
            detectorMap,
            visitInfo,
            adjustDM_config=adjustDM_config,
            log=self.log,
        )

        self.log.info("Making residual plots")
        residFig = plot_detectormap_residuals(
            arc_data,
            stats,
            detectorMap,
            spatialRange=self.config.spatialRange,
            wavelengthRange=self.config.wavelengthRange,
        )

        # Update the title with the detector name.
        suptitle = "DetectorMap Residuals\n{visit} {arm}{spectrograph}\n{run}".format(**dataId)
        residFig.suptitle(suptitle, weight="bold")

        return Struct(
            dmQaResidualData=arc_data,
            dmQaResidualStats=stats,
            dmQaResidualPlot=residFig,
        )


def get_data_and_stats(
    dataId: dict,
    arcLines: ArcLineSet,
    detectorMap: DetectorMap,
    visitInfo: VisitInfo,
    adjustDM_config=None,
    log=None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_science = visitInfo.observationReason == "science"

    good_lines_idx = getGoodLines(
        arcLines,
        dispersion=detectorMap.getDispersionAtCenter(),
        isScience=is_science,
        adjustDMConfig=adjustDM_config,
        log=log,
        **kwargs,
    )
    arcLines = arcLines[good_lines_idx].copy()

    arc_data = scrub_data(arcLines, detectorMap, dropNaColumns=True, log=log)
    if len(arc_data) == 0:
        raise ValueError("After scrubbing the data, the data is empty, cannot proceed.")

    # Mark the sigma-clipped outliers for each relevant group.
    def maskOutliers(grp):
        grp["xResidOutlier"] = sigma_clip(grp.xResid).mask
        grp["yResidOutlier"] = sigma_clip(grp.yResid).mask
        return grp

    # Ignore the warnings about NaNs and inf.
    log.info("Masking outliers")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arc_data = arc_data.groupby(["status_type", "description"]).apply(maskOutliers)
        arc_data.reset_index(drop=True, inplace=True)

    log.info("Adding fiber information")
    mtp_df = pd.DataFrame(
        FiberIds().fiberIdToMTP(detectorMap.fiberId), columns=["mtpId", "mtpHoles", "cobraId"]
    )
    mtp_df.index = detectorMap.fiberId
    mtp_df.index.name = "fiberId"
    arc_data = arc_data.merge(mtp_df.reset_index(), on="fiberId")

    log.info("Removing outliers")
    arc_data = arc_data.query(
        "(isLine == True and yResidOutlier == False) or (isTrace == True and xResidOutlier == False)"
    ).copy()

    arc_data["arm"] = dataId["arm"]
    arc_data["spectrograph"] = dataId["spectrograph"]
    arc_data["visit"] = dataId["visit"]

    log.info("Getting residual stats")
    stats = []
    for (status_type, description), rows in arc_data.groupby(["status_type", "description"]):
        visit_stats = pd.json_normalize(get_fit_stats(rows).to_dict())
        visit_stats["status_type"] = status_type
        visit_stats["description"] = description
        visit_stats["arm"] = dataId["arm"]
        visit_stats["spectrograph"] = dataId["spectrograph"]
        visit_stats["visit"] = dataId["visit"]
        visit_stats["ccd"] = "{arm}{spectrograph}".format(**dataId)
        visit_stats["observationReason"] = visitInfo.observationReason
        stats.append(visit_stats)

    stats = pd.concat(stats)

    return arc_data, stats


def getGoodLines(
    lines: ArcLineSet,
    dispersion: float | None,
    adjustDMConfig: Config,
    isScience: bool = False,
    lineFlags: int | None = None,
    minSignalToNoise: float | None = 0,
    maxCentroidError: float | None = 0,
    exclusionRadius: float | None = 0,
    log: Logger | None = None,
) -> np.ndarray:
    """Get the good lines.

    Parameters
    ----------
    lines : `ArcLineSet`
        The arc lines.
    dispersion : `float`, optional
        The dispersion. Default is None.
    adjustDMConfig : `Config`
        Configuration used for the detector map adjustment.
    isScience : `bool`, optional
        Is this a science visit? Default is False.
    lineFlags : `int`, optional
        The line flags. Default is None, which uses the lineFlags from the adjustDMConfig.
    minSignalToNoise : `float`, optional
        The minimum signal to noise ratio. Default is 0, which turns of the check.
    maxCentroidError : `float`, optional
        The maximum centroid error. Default is 0, which turns of the check.
    exclusionRadius : `float`, optional
        The exclusion radius. Default is 0, which turns of the check.
    log : `Logger`, optional
        The logger for the class object. Default is None.

    Returns
    -------
    good : `np.ndarray`
        The index of the good lines.
    """
    log.debug(f"Scrubbing data using config={adjustDMConfig.toDict()}")
    traceIndex = lines.description == "Trace"
    lineIndex = ~traceIndex
    numTraceLines = len(set(lines[traceIndex].fiberId))
    numArcLines = len(set(lines[lineIndex].fiberId))

    isTrace = lineIndex.sum() == 0
    isArc = not isTrace

    log.debug(f"{traceIndex.sum()} line centroids for {numTraceLines} traces")
    log.debug(f"{lineIndex.sum()} line centroids for {numArcLines} traces")
    log.debug(f"{lineIndex.sum() + traceIndex.sum()} lines in list")

    def getCounts():
        """Provide a list of counts of different species."""
        return getDescriptionCounts(lines.description, good)

    good = lines.flag == 0
    log.debug(f"{good.sum()} good lines after initial flags ({getCounts()})")

    if not isScience and isArc:
        log.info("Found lamp species, ignoring traces.")
        good &= lineIndex
        log.debug(f"{good.sum()} good lines after ignoring traces ({getCounts()})")

    if lineFlags is None:
        lineFlags = adjustDMConfig.lineFlags
    if lineFlags is not None:
        good &= (lines.status & ReferenceLineStatus.fromNames(*lineFlags)) == 0
        log.debug(f"{good.sum()} good lines after line flags ({getCounts()})")

    good &= np.isfinite(lines.x) & np.isfinite(lines.y)
    good &= np.isfinite(lines.xErr) & np.isfinite(lines.yErr)

    if hasattr(lines, "slope"):
        good &= np.isfinite(lines.slope) | ~traceIndex
    log.debug(f"{good.sum()} good lines after finite positions ({getCounts()})")

    if minSignalToNoise is None:
        minSignalToNoise = adjustDMConfig.minSignalToNoise
    if minSignalToNoise > 0:
        good &= np.isfinite(lines.flux) & np.isfinite(lines.fluxErr)
        log.debug(f"{good.sum()} good lines after finite intensities ({getCounts()})")

        with np.errstate(invalid="ignore", divide="ignore"):
            sn = lines.flux / lines.fluxErr
            mean_sn = np.nanmean(sn[good])
            std_sn = np.nanstd(sn[good])
            # Use the minimum of the mean - std and the config value.
            sn_cut = min(mean_sn - std_sn, minSignalToNoise)
            log.debug(f"Filtering SN < {sn_cut=:.02f}")
            good &= sn >= sn_cut

        log.debug(f"{good.sum()} good lines after SN filtering ({getCounts()})")

    if maxCentroidError is None:
        maxCentroidError = adjustDMConfig.maxCentroidError
    if maxCentroidError > 0:
        good &= (lines.xErr > 0) & (lines.xErr < maxCentroidError)
        good &= ((lines.yErr > 0) & (lines.yErr < maxCentroidError)) | traceIndex
        log.debug(f"{good.sum()} good lines after {maxCentroidError=} centroid errors ({getCounts()})")

    if exclusionRadius is None:
        exclusionRadius = adjustDMConfig.exclusionRadius
    if dispersion is not None and exclusionRadius > 0 and not np.all(traceIndex):
        wavelength = np.unique(lines.wavelength[~traceIndex])
        status = [np.bitwise_or.reduce(lines.status[lines.wavelength == wl]) for wl in wavelength]
        exclusionRadius = dispersion * exclusionRadius
        exclude = getExclusionZone(wavelength, exclusionRadius, np.array(status))
        good &= np.isin(lines.wavelength, wavelength[exclude], invert=True) | traceIndex
        log.debug(f"{good.sum()} good lines after {exclusionRadius=:.03f} exclusion zone ({getCounts()})")

    return good


def scrub_data(
    arcLines: ArcLineSet,
    detectorMap: DetectorMap,
    dropNaColumns: bool = False,
    removeFlagged: bool = True,
    onlyReservedAndUsed: bool = True,
    log: Logger | None = None,
) -> pd.DataFrame:
    """Get a copy of the arcline data, with some columns added.

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
    log : `Logger`, optional
        The logger for the class object. Default is None.

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

    if removeFlagged:
        arc_data = arc_data.query("flag == False").copy()

    # Convert nm to pixels.
    arc_data["dispersion"] = detectorMap.getDispersion(
        arc_data.fiberId.to_numpy(), arc_data.wavelength.to_numpy()
    )

    # Get USED and RESERVED status.
    is_reserved = (arc_data.status & ReferenceLineStatus.DETECTORMAP_RESERVED.value) != 0
    is_used = (arc_data.status & ReferenceLineStatus.DETECTORMAP_USED.value) != 0

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
    with contextlib.suppress(AttributeError):
        arc_data.y = arc_data.y.astype(np.float64)

    # Replace inf with nans.
    arc_data = arc_data.replace([np.inf, -np.inf], np.nan)

    # Get full status names.
    arc_data["status_name"] = arc_data.status.map(lambda x: ReferenceLineStatus(x).name)
    arc_data["status_name"] = arc_data["status_name"].astype("category")
    arc_data.status_name = arc_data.status_name.cat.remove_unused_categories()

    return arc_data


def get_fit_stats(
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
    if sigmaClipOnly is True:
        arc_data = arc_data.query("xResidOutlier == False")

    traces = arc_data.query("isTrace == True").copy()
    lines = arc_data.query("isLine == True").dropna(subset=["yResid"]).copy()

    xNum = len(arc_data)
    numTraces = traces.fiberId.nunique()
    try:
        yNum = lines.isLine.value_counts()[True]
        numLines = lines.wavelength.nunique()
    except KeyError:
        yNum = 0
        numLines = 0

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
        # dof <= 0 divides by zero and yields NaN, which bisect cannot start
        # from. It happens when the parameter count eats the whole sample --
        # a fiber left with one surviving line, say. The same guard exists in
        # drp_stella's calculateSoftening, for the same reason.
        if len(resid) == 0 or dof <= 0:
            return 0
        with np.errstate(invalid="ignore"):
            return (getChi2(resid, err, soften) / dof) - 1

    f_x = partial(getSoften, arc_data.xResid, arc_data.xErr, xDof)
    f_y = partial(getSoften, lines.yResid, lines.yErr, yDof)

    def solveSoften(f):
        """Solve for the softening that brings chi2/dof to 1, or give up cleanly.

        Parameters
        ----------
        f : `callable`
            Softening residual function; ``f(s) == 0`` at the solution.

        Returns
        -------
        `float`
            The softening, 0.0 when none is needed, or NaN when the fit cannot
            be softened within ``maxSoften`` or the endpoints are not finite.
            Returning NaN beats raising: one unsolvable detector must not take
            down the quantum.
        """
        low, high = f(0), f(maxSoften)
        if not np.isfinite(low) or not np.isfinite(high):
            return np.nan
        if low < 0:
            return 0.0
        if high > 0:
            return np.nan
        return bisect(f, 0, maxSoften)

    xSoftFit = solveSoften(f_x)
    ySoftFit = solveSoften(f_y)

    xFibers = len(traces.fiberId.unique())
    yFibers = len(lines.fiberId.unique())

    xFitStat = FitStat(arc_data.xResid.median(), xRobustRms, xWeightedRms, xSoftFit, xDof, xFibers, numTraces)
    yFitStat = FitStat(lines.yResid.median(), yRobustRms, yWeightedRms, ySoftFit, yDof, yFibers, numLines)

    return FitStats(dof, chi2X, chi2Y, xFitStat, yFitStat)
