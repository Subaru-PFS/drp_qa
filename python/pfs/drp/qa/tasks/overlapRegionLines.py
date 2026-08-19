from typing import Iterable

import lsstDebug
import numpy as np
from lsst.pex.config import Config
from lsst.pipe.base import Struct, Task
from matplotlib import pyplot as plt
from pfs.drp.stella import ArcLineSet, DetectorMap, PfsArm
from scipy.stats import iqr


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
        self, detectorMap: Iterable[DetectorMap], arcLines: Iterable[ArcLineSet], pfsArm: Iterable[PfsArm]
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
            self.log.info(
                f"{w} nm ({len(fibers[w])} fibers, "
                f"median={np.median(difference[w]):.1e} nm, "
                f"1sigma={iqr(difference[w]) / 1.349:.3f} nm)"
            )
            plt.scatter(
                fibers[w],
                difference[w],
                s=3,
                label=f"{w} nm ({len(fibers[w])} fibers, "
                f"median={np.median(difference[w]):.1e} nm, "
                f"1sigma={iqr(difference[w]) / 1.349:.3f} nm)",
            )
        plt.legend(fontsize=7)
        plt.xlabel("fiberId")
        plt.ylabel(f"Wavelength difference ({arm[0]}-{arm[1]}) [nm]")
        plt.savefig(f"overlapLines-{visit:06}-{arm[0]}{arm[1]}{spectrograph}.png")
        plt.close()

        return Struct()
