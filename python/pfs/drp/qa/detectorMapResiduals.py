import warnings
from contextlib import suppress

import pandas as pd
from astropy.stats import sigma_clip
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
from pfs.drp.stella import ArcLineSet, DetectorMap
from pfs.utils.fiberids import FiberIds

from pfs.drp.qa.tasks.detectorMapResiduals import (
    get_fit_stats,
    scrub_data,
)


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


class DetectorMapQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapQaConnections):
    """Configuration for DetectorMapQaTask"""


class DetectorMapQaTask(PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapQaConfig
    _DefaultName = "detectorMapQa"

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
            )

        descriptions = sorted(list(arc_data.description.unique()))
        with suppress(ValueError):
            if len(descriptions) > 1:
                descriptions.remove("Trace")

        dmap_bbox = detectorMap.getBBox()
        fiberIdMin = detectorMap.fiberId.min()
        fiberIdMax = detectorMap.fiberId.max()
        wavelengthMin = int(arcLines.wavelength.min())
        wavelengthMax = int(arcLines.wavelength.max())

        stats = list()
        for idx, rows in arc_data.groupby("status_type"):
            exposure_stats = pd.json_normalize(get_fit_stats(rows).to_dict())
            exposure_stats["exposure"] = dataId["exposure"]
            exposure_stats["arm"] = dataId["arm"]
            exposure_stats["spectrograph"] = dataId["spectrograph"]
            exposure_stats["ccd"] = "{arm}{spectrograph}".format(**dataId)
            exposure_stats["status_type"] = idx
            exposure_stats["description"] = ",".join(descriptions)
            exposure_stats["detector_width"] = dmap_bbox.width
            exposure_stats["detector_height"] = dmap_bbox.height
            exposure_stats["fiberId_min"] = fiberIdMin
            exposure_stats["fiberId_max"] = fiberIdMax
            exposure_stats["wavelength_min"] = wavelengthMin
            exposure_stats["wavelength_max"] = wavelengthMax
            stats.append(exposure_stats)

        return Struct(dmQaResidualStats=pd.concat(stats))
