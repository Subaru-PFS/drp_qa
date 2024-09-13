from collections import defaultdict
from typing import Dict, Iterable

import lsstDebug
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import (
    CmdLineTask,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    TaskRunner,
)
from lsst.pipe.base.connectionTypes import Input as InputConnection, Output as OutputConnection
from pfs.drp.stella import ArcLineSet, DetectorMap

from pfs.drp.qa.tasks.detectorMapResiduals import PlotResidualTask


class DetectorMapQaConnections(
    PipelineTaskConnections,
    dimensions=("exposure", "detector", "physical_filter"),
):
    """Connections for DetectorMapQaTask"""

    detectorMap = InputConnection(
        name="detectorMap_used",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("exposure", "detector"),
    )
    arcLines = InputConnection(
        name="arcLines",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("exposure", "detector"),
        multiple=True,
    )
    dmQaResidualPlot = OutputConnection(
        name="dmQaResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for a given visit.",
        storageClass="MultipagePdfFigure",
        dimensions=("exposure", "detector"),
    )
    dmQaCombinedResidualPlot = OutputConnection(
        name="dmQaCombinedResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for the entire detector.",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument",),
    )
    dmQaResidualStats = OutputConnection(
        name="dmQaResidualStats",
        doc="Statistics of the residual analysis for the visit.",
        storageClass="pandas.core.frame.DataFrame",
        dimensions=("exposure", "detector"),
    )
    dmQaDetectorStats = OutputConnection(
        name="dmQaDetectorStats",
        doc="Statistics of the residual analysis for the entire detector.",
        storageClass="pandas.core.frame.DataFrame",
        dimensions=("detector",),
    )


class DetectorMapQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapQaConnections):
    """Configuration for DetectorMapQaTask"""

    checkOverlap = Field(dtype=bool, default=False, doc="If the overlapRegionLines should be checked.")
    plotResidual = ConfigurableField(target=PlotResidualTask, doc="Plot the detector map residual.")


class DetectorMapQaRunner(TaskRunner):
    """Runner for DetectorMapQaTask"""

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Produce list of targets for DetectorMapQaTask.

        The visits and detector are processed as part of a group, with group
        membership consisting of:

            * If combineVisits, group all visits by detector.
            * If checkOverlap, group all detectors by visit.
            * Otherwise, group per visit per detector (i.e. don't group).
        """
        combineVisits = parsedCmd.config.plotResidual.combineVisits
        checkOverlap = parsedCmd.config.checkOverlap

        groups = defaultdict(list)
        for ref in parsedCmd.id.refList:
            visit = ref.dataId["visit"]
            spectrograph = ref.dataId["spectrograph"]
            arm = ref.dataId["arm"]
            ccd = f"{arm}{spectrograph}"

            if combineVisits is True:
                groups[ccd].append(ref)
            elif checkOverlap is True:
                groups[visit].append(ref)
            else:
                groups[f"{visit}-{ccd}"].append(ref)

        processGroups = [((key, group), kwargs) for key, group in groups.items()]

        return processGroups


class DetectorMapQaTask(CmdLineTask, PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapQaConfig
    _DefaultName = "detectorMapQa"
    RunnerClass = DetectorMapQaRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("plotResidual")
        self.debugInfo = lsstDebug.Info(__name__)

    def runDataRef(self, expSpecRefList) -> Struct:
        """Calls ``self.run()``

        Parameters
        ----------
        expSpecRefList : iterable of iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for each sensor, grouped either by visit or by detector.

        Returns
        -------
        Struct
            Output data products. See `DetectorMapQaConnections`.
        """
        groupName = expSpecRefList[0]
        groupDataRefs = expSpecRefList[1]

        self.log.info(f"Starting processing for {groupName=} with {len(groupDataRefs)} dataIds")

        arcLinesSet = list()
        detectorMaps = list()
        dataIds = list()

        for dataRef in groupDataRefs:
            try:
                detectorMap = dataRef.get("detectorMap_used")
                arcLines = dataRef.get("arcLines")

                arcLinesSet.append(arcLines)
                detectorMaps.append(detectorMap)
                dataIds.append(dataRef.dataId)
            except Exception as e:
                self.log.error(e)

        # Run the task and get the outputs.
        outputs = self.run(groupName, arcLinesSet, detectorMaps, dataIds)

        # Save the outputs in butler.
        if outputs is not None:
            for datasetType, data in outputs.getDict().items():
                # Add the rerun and calib dirs to the suptitle.
                if datasetType == "dmQaResidualPlot" or datasetType == "dmQaCombinedResidualPlot":
                    try:
                        # TODO fix this one day with Gen3.
                        repo_args = groupDataRefs[0].butlerSubset.butler._repos.inputs()[0].repoArgs
                        rerun_name = repo_args.root
                        calib_dir = repo_args.mapperArgs["calibRoot"]
                        suptitle = data._suptitle.get_text()
                        data.suptitle(f"{suptitle}\n{rerun_name}\n{calib_dir}")
                    except Exception as e:
                        self.log.error(e)

                # Store the combined info in the first dataRef and the individual info in the others.
                self.log.info(f"Saving {datasetType}")
                if self.config.plotResidual.combineVisits is True:
                    groupDataRefs[0].put(data, datasetType=datasetType)
                else:
                    for dataRef in groupDataRefs:
                        dataRef.put(data, datasetType=datasetType)

        return outputs

    def run(
        self,
        groupName: str,
        arclineSet: Iterable[ArcLineSet],
        detectorMaps: Iterable[DetectorMap],
        dataIds: Iterable[Dict],
    ) -> Struct:
        """Generate detectorMapQa plots.

        Parameters
        ----------
        groupName : `str`
            Group name, either the visit or the detector.
        arclineSet : iterable of `ArcLineSet`
            Emission line measurements by adjustDetectorMap.
        detectorMaps : iterable of `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        dataIds : iterable of `dict`
            List of dataIds.

        Returns
        -------
        Struct
            Output data products. See `DetectorMapQaConnections`.
        """
        return self.plotResidual.run(groupName, arclineSet, detectorMaps, dataIds)

    def _getMetadataName(self):
        return None
