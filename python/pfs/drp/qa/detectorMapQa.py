import warnings
from collections import defaultdict
from typing import Iterable

import lsstDebug
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr

from lsst.pex.config import Field, ConfigurableField, Config
from lsst.pipe.base import (
    ArgumentParser,
    CmdLineTask,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    Task,
    TaskRunner,
)

from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from pfs.drp.stella import ArcLineSet, DetectorMap, PfsArm
from pfs.drp.qa.utils import helpers
from pfs.drp.qa.utils import plotting

from .storageClasses import MultipagePdfFigure

warnings.filterwarnings('ignore', message='Gen2 Butler')
warnings.filterwarnings('ignore', message='addPfsCursor')


class PlotResidualConfig(Config):
    """Configuration for PlotResidualTask"""

    useSigmaRange = Field(dtype=bool, default=True, doc='Use ±2.5 sigma as range')
    xrange = Field(dtype=float, default=0.1, doc="Range of the residual (X center) in a plot in pix.")
    wrange = Field(dtype=float, default=0.03, doc="Range of the residual (wavelength) in a plot in nm.")


class PlotResidualTask(Task):
    """Task for QA of detectorMap."""

    ConfigClass = PlotResidualConfig
    _DefaultName = "plotResidual"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, detectorMap: DetectorMap, arcLines: ArcLineSet, pfsArm: PfsArm) -> Struct:
        """QA of adjustDetectorMap by plotting the fitting residual.

        Parameters
        ----------
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y
        arcLines : `ArcLineSet`
            Emission line measurements by adjustDetectorMap.
        pfsArm : `PfsArm`
            Extracted spectra from arm.

        Returns
        -------
        dmQaResidualImage : `MultipagePdfFigure`
            Detail plots.
            1D and 2D plots of the residual between the detectormap and the arclines.

        Outputs
        -------
        plot : `dm-residuals-{visit}-{arm}{spectrograph}.pdf`
        """
        visit = pfsArm.identity.visit
        arm = pfsArm.identity.arm
        spectrograph = pfsArm.identity.spectrograph
        dataIdStr = f'v{visit}-{arm}{spectrograph}'

        title_str = f'{visit:06}-{arm}{spectrograph}'
        self.log.info(f'Getting data for {title_str}')

        try:
            arc_data = helpers.loadData(arcLines, detectorMap)
            arc_data.reset_index(drop=True, inplace=True)
        except Exception:
            self.log.error(f'Not enough data for {dataIdStr}')
            return Struct()

        num_fibers = len(arc_data.fiberId.unique())
        num_lines = len(arc_data)

        if num_fibers == 0 or num_lines == 0:
            self.log.info(f'No measured lines found for {title_str}')
            return

        self.log.info(f"Number of fibers: {num_fibers}")
        self.log.info(f"Number of Measured lines: {num_lines}")

        dmQaResidualImagePdf = MultipagePdfFigure()
        useSigmaRange = self.config.useSigmaRange
        for column in ['dx', 'dy_nm']:
            which_range = 'xrange' if column == 'dx' else 'wrange'
            slimit = getattr(self.config, which_range) if useSigmaRange is False else None
            slimit = (-slimit, slimit) if slimit is not None else (None, None)
            self.log.debug(f'Generating {column} plot for {dataIdStr}')
            try:
                fig = plotting.plotResidual(arc_data, column=column, vmin=slimit[0], vmax=slimit[1])
                fig.suptitle(f'DetectorMap Residuals\n{dataIdStr}\n{column}', weight='bold')
                dmQaResidualImagePdf.append(fig)
            except ValueError:
                self.log.info(f'No residual wavelength information for {dataIdStr}, skipping')

        return Struct(
            dmQaResidualImage=dmQaResidualImagePdf
        )

    @classmethod
    def _makeArgumentParser(cls) -> ArgumentParser:
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(
            name="--id", datasetType="arcLines", level="Visit", help="data IDs, e.g. --id exp=12345"
        )
        return parser


class OverlapRegionLinesConfig(Config):
    """Configuration for OverlapRegionLinesTask"""

    pass


class OverlapRegionLinesTask(Task):
    """Task for QA of detectorMap"""

    ConfigClass = OverlapRegionLinesConfig
    _DefaultName = "overlapRegionLines"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(
        self, detectorMap: Iterable[DetectorMap], arcLines: Iterable[ArcLineSet],
        pfsArm: Iterable[PfsArm]
    ) -> Struct:
        """QA of adjustDetectorMap by plotting the wavelength difference of sky lines detected in multiple
        arms.

        Parameters
        ----------
        detectorMap : iterable of `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        arcLines : iterable of `ArcLineSet`
            Emission line measurements by adjustDetectorMap.
        pfsArm : iterable of `PfsArm`
            Extracted spectra from arm.

        Returns
        -------
        None

        Outputs
        -------
        plot : `overlapLines-{:06}-{}{}{}.png`
        """

        visit = pfsArm[0].identity.visit
        arm = [pp.identity.arm for pp in pfsArm]
        spectrograph = pfsArm[0].identity.spectrograph
        fmin, fmax = [np.amin(aa.fiberId) for aa in arcLines], [np.amax(aa.fiberId) for aa in arcLines]

        measured = [
            np.logical_not(np.isnan(aa.flux))
            & np.logical_not(np.isnan(aa.x))
            & np.logical_not(np.isnan(aa.y))
            & np.logical_not(np.isnan(aa.xErr))
            & np.logical_not(np.isnan(aa.yErr))
            & np.logical_not(np.isnan(aa.fluxErr))
            for aa in arcLines
        ]

        flist = []
        for aa in range(len(arcLines)):
            flist.append([])
            for f in range(fmin[aa], fmax[aa] + 1):
                notNan_f = (arcLines[aa].fiberId == f) & measured[aa]
                if np.sum(notNan_f) > 0:
                    flist[aa].append(f)
            self.log.info(f"Fiber number ({arm[aa]}{spectrograph}): {len(flist[aa])}")
            self.log.info(f"Measured line ({arm[aa]}{spectrograph}): {np.sum(measured[aa])}")

        plt.figure()
        fcommon = set(flist[0]) | set(flist[1])
        fibers = {}
        difference = {}
        wcommon = []
        goodLines = [
            "630.20",
            "636.55",
            "937.69",
            "937.85",
            "942.23",
            "947.94",
            "952.20",
            "957.00",
            "970.19",
            "972.25",
            "979.21",
            "979.38",
            "980.24",
        ]
        for f in fcommon:
            b0 = (arcLines[0].fiberId == f) & measured[0]
            b1 = (arcLines[1].fiberId == f) & measured[1]
            wav0 = set(arcLines[0][b0].wavelength)
            wav1 = set(arcLines[1][b1].wavelength)
            wav = list(wav0 & wav1)
            if len(wav) > 0:
                wav.sort()
                for w in wav:
                    if "{:.2f}".format(w) in goodLines:
                        if w not in wcommon:
                            wcommon.append(w)
                            fibers[w] = []
                            difference[w] = []
                        y = [aa[(aa.fiberId == f) & (aa.wavelength == w)].y[0] for aa in arcLines]
                        fibers[w].append(f)
                        diff_wl0 = detectorMap[0].findWavelength(fiberId=f, row=y[0])
                        diff_wl1 = detectorMap[1].findWavelength(fiberId=f, row=y[1])
                        difference[w].append(diff_wl0 - diff_wl1)
        plt.figure()
        wcommon.sort()
        for w in wcommon:
            self.log.info(f"{w} nm ({len(fibers[w])} fibers, "
                          f"median={np.median(difference[w]):.1e} nm, "
                          f"1sigma={iqr(difference[w]) / 1.349:.3f} nm)"
                          )
            plt.scatter(
                fibers[w],
                difference[w],
                s=3,
                label=f"{w} nm ({len(fibers[w])} fibers, "
                      f"median={np.median(difference[w]):.1e} nm, "
                      f"1sigma={iqr(difference[w]) / 1.349:.3f} nm)"
            )
        plt.legend(fontsize=7)
        plt.xlabel("fiberId")
        plt.ylabel(f"Wavelength difference ({arm[0]}-{arm[1]}) [nm]")
        plt.savefig(f"overlapLines-{visit:06}-{arm[0]}{arm[1]}{spectrograph}.png")
        plt.close()

        return Struct()

    @classmethod
    def _makeArgumentParser(cls) -> ArgumentParser:
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(
            name="--id", datasetType="arcLines", level="Visit", help="data IDs, e.g. --id exp=12345"
        )
        return parser


# (Gen3) If this task is ("instrument", "exposure", "detector") declare to be executed for each combination.
class DetectorMapQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "exposure"),
):
    """Connections for DetectorMapQaTask"""
    detectorMap = InputConnection(
        name="detectorMap_used",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "exposure", "detector"),
    )
    arcLines = InputConnection(
        name="arcLines",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra from arm",
        storageClass="PfsArm",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    dmQaResidualImage = OutputConnection(
        name="dmQaResidualImage",
        doc="Detail plots. The 1D and 2D residual plots of the detectormap with the arclines.",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument", "exposure", "detector"),
    )


class DetectorMapQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapQaConnections):
    """Configuration for DetectorMapQaTask"""

    plotResidual = ConfigurableField(target=PlotResidualTask, doc="Plot the detector map residual.")
    overlapRegionLines = ConfigurableField(
        target=OverlapRegionLinesTask,
        doc="Plot the wavelength difference of the sky lines commonly detected in multiple arms."
    )


class DetectorMapQaRunner(TaskRunner):
    """Runner for DetectorMapQaTask"""

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Produce list of targets for DetectorMapQaTask

        We want to operate on all data within a single exposure at once.
        """
        exposures = defaultdict(lambda: defaultdict(list))
        for ref in parsedCmd.id.refList:
            visit = ref.dataId["visit"]
            spectrograph = ref.dataId["spectrograph"]
            exposures[visit][spectrograph].append(ref)
        return [(list(specs.values()), kwargs) for specs in exposures.values()]


class DetectorMapQaTask(CmdLineTask, PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapQaConfig
    _DefaultName = "detectorMapQa"
    RunnerClass = DetectorMapQaRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("plotResidual")
        self.debugInfo = lsstDebug.Info(__name__)

    def runQuantum(
        self,
        butler: ButlerQuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with butler I/O

        Parameters
        ----------
        butler : `ButlerQuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        inputs = butler.get(inputRefs)
        outputs = self.run(**inputs)
        butler.put(outputs, outputRefs)

    def runDataRef(self, expSpecRefList) -> Struct:
        """Calls ``self.run()``

        Parameters
        ----------
        expSpecRefList : iterable of iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for each sensor, grouped by spectrograph.

        Returns
        -------
        None
        """
        for specRefList in expSpecRefList:
            for dataRef in specRefList:
                detectorMap = dataRef.get('detectorMap_used')
                arcLines = dataRef.get('arcLines')
                pfsArm = dataRef.get('pfsArm')

                # Run the task and get the outputs.
                outputs = self.run(detectorMap, arcLines, pfsArm)
                for datasetType, data in outputs.getDict().items():
                    if data is not None:
                        dataRef.put(data, datasetType=datasetType)

    def run(self, detectorMap: DetectorMap, arcLines: ArcLineSet, pfsArm: PfsArm) -> Struct:
        """Generate detectorMapQa plots.

        Parameters
        ----------
        detectorMap: `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        arcLines : `ArcLineSet`
            Emission line measurements by adjustDetectorMap.
        pfsArm : `PfsArm`
            Extracted spectra from arm.

        Returns
        -------
        None
        """
        return self.plotResidual.run(detectorMap, arcLines, pfsArm)

    def _getMetadataName(self):
        return None
