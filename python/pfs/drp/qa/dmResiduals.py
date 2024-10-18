import warnings
from contextlib import suppress
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from lsst.pex.config import Field
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
    Output as OutputConnection,
)
from pfs.drp.stella import ArcLineSet, DetectorMap, ReferenceLineStatus
from pfs.drp.stella.utils.math import robustRms
from pfs.utils.fiberids import FiberIds
from scipy.optimize import bisect

from pfs.drp.qa.utils.math import getChi2, getWeightedRMS
from pfs.drp.qa.utils.plotting import plot_detectormap_residuals


class DetectorMapQaConnections(
    PipelineTaskConnections,
    dimensions=(
        "instrument",
        "exposure",
        "arm",
        "spectrograph",
    ),
):
    """Connections for DetectorMapQaTask"""

    detectorMap = InputConnection(
        name="detectorMap",
        doc="Adjusted detector mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
    )

    arcLines = InputConnection(
        name="lines",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
    )
    dmQaResidualStats = OutputConnection(
        name="dmQaResidualStats",
        doc="Statistics of the DM residual analysis.",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
    )
    dmQaResidualPlot = OutputConnection(
        name="dmQaResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for a given visit.",
        storageClass="Plot",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
    )


class DetectorMapQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapQaConnections):
    """Configuration for DetectorMapQaTask"""

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


class DetectorMapQaTask(PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapQaConfig
    _DefaultName = "dmResiduals"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        # Get the dataIds for help with plotting.
        data_id = {k: v for k, v in inputRefs.arcLines.dataId.full.items()}
        data_id["run"] = inputRefs.arcLines.run

        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = data_id

        # Perform the actual processing.
        outputs = self.run(**inputs)

        # Store the results.
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        arcLines: ArcLineSet,
        detectorMap: DetectorMap,
        dropNaColumns: bool = True,
        removeOutliers: bool = True,
        addFiberInfo: bool = True,
        dataId: dict = None,
        **kwargs,
    ) -> Struct:
        """Cleans and masks the data. Adds fiberInfo if requested.

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
        removeOutliers : `bool`, optional
            Remove rows with ``flag=False``? Default is True.
        addFiberInfo : `bool`, optional
            Add fiber information to the dataframe. Default is True.
        dataId : dict, optional
            Dictionary of the dataId.

        Returns
        -------
        arc_data : `pandas.DataFrame`
        """

        # Get dataframe for arc lines and add detectorMap information, then calculate residuals.
        self.log.info("Getting and scrubbing the data")
        arc_data = scrub_data(arcLines, detectorMap, dropNaColumns=dropNaColumns, **kwargs)

        # Mark the sigma-clipped outliers for each relevant group.
        def maskOutliers(grp):
            grp["xResidOutlier"] = sigma_clip(grp.xResid).mask
            grp["yResidOutlier"] = sigma_clip(grp.yResid).mask
            return grp

        # Ignore the warnings about NaNs and inf.
        self.log.info("Masking outliers")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arc_data = arc_data.groupby(["status_type", "isLine"]).apply(maskOutliers)
            arc_data.reset_index(drop=True, inplace=True)

        if addFiberInfo is True:
            self.log.info("Adding fiber information")
            mtp_df = pd.DataFrame(
                FiberIds().fiberIdToMTP(detectorMap.fiberId), columns=["mtpId", "mtpHoles", "cobraId"]
            )
            mtp_df.index = detectorMap.fiberId
            mtp_df.index.name = "fiberId"
            arc_data = arc_data.merge(mtp_df.reset_index(), on="fiberId")

        if removeOutliers is True:
            self.log.info("Removing outliers")
            arc_data = arc_data.query(
                "(isLine == True and yResidOutlier == False and xResidOutlier == False)"
                " or "
                "(isTrace == True and xResidOutlier == False)"
            ).copy()

        descriptions = sorted(list(arc_data.description.unique()))
        with suppress(ValueError):
            if len(descriptions) > 1:
                descriptions.remove("Trace")

        dmap_bbox = detectorMap.getBBox()
        fiberIdMin = detectorMap.fiberId.min()
        fiberIdMax = detectorMap.fiberId.max()
        wavelengthMin = int(arcLines.wavelength.min())
        wavelengthMax = int(arcLines.wavelength.max())

        arc_data["arm"] = dataId["arm"]
        arc_data["spectrograph"] = dataId["spectrograph"]
        arc_data["exposure"] = dataId["exposure"]

        self.log.info("Getting residual stats")
        stats = list()
        for idx, rows in arc_data.groupby("status_type"):
            exposure_stats = pd.json_normalize(get_fit_stats(rows).to_dict())
            exposure_stats["status_type"] = idx
            exposure_stats["arm"] = dataId["arm"]
            exposure_stats["spectrograph"] = dataId["spectrograph"]
            exposure_stats["exposure"] = dataId["exposure"]
            exposure_stats["ccd"] = "{arm}{spectrograph}".format(**dataId)
            exposure_stats["description"] = ",".join(descriptions)
            exposure_stats["detector_width"] = dmap_bbox.width
            exposure_stats["detector_height"] = dmap_bbox.height
            exposure_stats["fiberId_min"] = fiberIdMin
            exposure_stats["fiberId_max"] = fiberIdMax
            exposure_stats["wavelength_min"] = wavelengthMin
            exposure_stats["wavelength_max"] = wavelengthMax
            stats.append(exposure_stats)

        stats = pd.concat(stats)

        self.log.info("Making residual plots")
        residFig = plot_detectormap_residuals(
            arc_data,
            stats,
            dataId["arm"],
            dataId["spectrograph"],
            useSigmaRange=self.config.useSigmaRange,
            spatialRange=self.config.spatialRange,
            wavelengthRange=self.config.wavelengthRange,
            binWavelength=self.config.binWavelength,
        )

        # Update the title with the detector name.
        suptitle = "DetectorMap Residuals {arm}{spectrograph}\n{run}".format(**dataId)
        residFig.suptitle(suptitle, weight="bold")

        return Struct(
            dmQaResidualStats=stats,
            dmQaResidualPlot=residFig,
        )


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


def scrub_data(
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
    arc_data.rename(columns={"visit": "exposure"}, inplace=True)

    if removeFlagged:
        arc_data = arc_data.query("flag == False").copy()

    # Convert nm to pixels.
    arc_data["dispersion"] = detectorMap.getDispersion(
        arc_data.fiberId.to_numpy(), arc_data.wavelength.to_numpy()
    )

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
