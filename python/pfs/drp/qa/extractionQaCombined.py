from typing import Iterable

import numpy as np
from lsst.pex.config import Field, ListField
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
    PrerequisiteInput as PrerequisiteConnection,
)
from matplotlib import pyplot as plt
from pfs.datamodel import PfsConfig

from pfs.drp.qa.storageClasses import MultipagePdfFigure, QaDict


class ExtractionQaCombinedConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm",),
):
    """Connections for ExtractionQaCombinedTask"""

    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
    )
    extQaImage_pickle = InputConnection(
        name="extQaImage_pickle",
        doc="Statistics of the residual analysis.",
        storageClass="QaDict",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    extQaStatsCombined = OutputConnection(
        name="extQaStatsCombined",
        doc="Summary plots. Results of the residual analysis of extraction are plotted.",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument", "visit", "arm"),
    )


class ExtractionQaCombinedConfig(PipelineTaskConfig, pipelineConnections=ExtractionQaCombinedConnections):
    """Configuration for ExtractionQaCombinedTask"""

    plotMinChiMed = Field(dtype=float, default=-1.5, doc="Minimum median Chi to plot")
    plotMaxChiMed = Field(dtype=float, default=+1.5, doc="Maximum median Chi to plot")
    plotMinChiStd = Field(dtype=float, default=+0.0, doc="Minimum standard deviation of Chi to plot")
    plotMaxChiStd = Field(dtype=float, default=+3.5, doc="Maximum standard deviation of Chi to plot")
    plotMinChiAtPeak = Field(dtype=float, default=-1.5, doc="Minimum Chi at peak to plot")
    plotMaxChiAtPeak = Field(dtype=float, default=+1.5, doc="Maximum Chi at peak to plot")
    plotMinResFrac = Field(dtype=float, default=-5.0, doc="Minimum residual fraction")
    plotMaxResFrac = Field(dtype=float, default=5.0, doc="Maximum residual fraction")
    targetType = ListField(dtype=str, default=["^ENGINEERING"],
                           doc="Target type for which to calculate statistics")
    figureDpi = Field(dtype=int, default=72, doc="Resolution of plot for residual")
    footnoteSize = Field(dtype=int, default=9, doc="Fontsize of the footnote")


class ExtractionQaCombinedTask(PipelineTask):
    """Task for QA of extraction"""

    ConfigClass = ExtractionQaCombinedConfig
    _DefaultName = "extractionQaCombined"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):

        inputs = butlerQC.get(inputRefs)

        # Perform the actual processing.
        outputs = self.run(**inputs)

        # Store the results.
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        pfsConfig: PfsConfig,
        extQaImage_pickle: Iterable[QaDict],
    ) -> Struct:
        """QA of extraction by analyzing the residual image.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers.
        extQaImage_pickle : Iterable[QaDict]
            An iterable of DataFrames containing extraction QA residual statistics.
            These are combined into a single DataFrame for processing.
        Returns
        -------
        extQaStatsCombined : `MultipagePdfFigure`
            Summary plots.
            Results of the residual analysis of extraction are plotted.
        """

        qaStatsCombined = QaDict(
            {
                "dataId": [],
                "fiberIds": [],
                "xa": [],
                "pfsArmAve": [],
                "targetMask": [],
                "chiSquare": [],
                "chiAve": [],
                "chiMed": [],
                "chiStd": [],
                "chiAtPeak": [],
                "resFrac": [],
            }
        )

        for stats in extQaImage_pickle:
            dataId = stats["dataId"]
            self.log.info(
                "Extraction QA combined plots on (visit=%(visit)d arm=%(arm)s) RUN=%(run)s" % dataId
            )
            for k, v in stats.items():
                if k == "dataId":
                    qaStatsCombined[k].append(v)
                else:
                    qaStatsCombined[k] += list(v)

        qaCombinedStatsPdf = self.drawStatsCombined(pfsConfig, qaStatsCombined)

        return Struct(
            extQaStatsCombined=qaCombinedStatsPdf,
        )

    def drawStatsCombined(
        self,
        pfsConfig: PfsConfig,
        qaStats: QaDict,
    ) -> MultipagePdfFigure:
        """Draw figures on new pages of ``qaStatsPdf``.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers.
        qaStats : `QaDict`
            Mapping of combined extraction QA statistics. Expected entries include
            per-fiber arrays for ``fiberIds``, ``chiMed``, ``chiStd``,
            ``chiAtPeak``, ``resFrac``, and ``pfsArmAve``, plus ``dataId`` values
            used to label the plots.

        Returns
        -------
        qaCombinedStatsPdf : `MultipagePdfFigure`
            Multi-page PDF figure containing the combined extraction QA plots
            generated from ``qaStats``.
        """
        qaCombinedStatsPdf = MultipagePdfFigure()

        fiberIds_all = np.array(qaStats["fiberIds"])
        chiMed_all = np.array(qaStats["chiMed"])
        chiStd_all = np.array(qaStats["chiStd"])
        chiAtPeak_all = np.array(qaStats["chiAtPeak"])
        resFrac_all = np.array(qaStats["resFrac"])
        pfsArmAve_all = np.array(qaStats["pfsArmAve"])

        visit = qaStats["dataId"][0]["visit"]
        arm = qaStats["dataId"][0]["arm"]
        pfiCenters = np.array([pfsConfig.select(fiberId=fid).pfiCenter for fid in fiberIds_all])

        vmin, vmax = np.nanpercentile(pfsArmAve_all, [1, 99])

        # chiMed

        fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[2, 1.5]))
        fig.suptitle(f"visit={visit}, arm={arm}, spectrograph=1,2,3,4", y=1.1)
        sc = None
        # chiMed vs. fiberId (left panel)
        ax[0].set_title("chiMed vs. fiberIds")

        ymin = self.config.plotMinChiMed
        ymax = self.config.plotMaxChiMed

        mask_under = chiMed_all < ymin
        mask_over = chiMed_all > ymax
        mask_inside = ~(mask_under | mask_over)

        sc = ax[0].scatter(
            fiberIds_all[mask_inside],
            chiMed_all[mask_inside],
            s=5,
            c=pfsArmAve_all[mask_inside],
            vmin=vmin, vmax=vmax,
            rasterized=True,
        )
        if np.any(mask_over):
            ax[0].scatter(
                fiberIds_all[mask_over],
                np.full(np.sum(mask_over), ymax),
                marker="^",
                s=50,
                c=pfsArmAve_all[mask_over],
                vmin=vmin, vmax=vmax,
                edgecolors="k",
                rasterized=True,
                clip_on=False
            )
        if np.any(mask_under):
            ax[0].scatter(
                fiberIds_all[mask_under],
                np.full(np.sum(mask_under), ymin),
                marker="v",
                s=50,
                c=pfsArmAve_all[mask_under],
                vmin=vmin, vmax=vmax,
                edgecolors="k",
                rasterized=True,
                clip_on=False
            )
        if sc is not None:
            fig.colorbar(sc, ax=ax[0], location="right", fraction=0.04, alpha=1.0, label="pfsArmAve")
        ax[0].grid(color="gray", linestyle=":", linewidth=0.5)
        ax[0].set_xlabel("fiberId")
        ax[0].set_ylabel("chiMed")
        ax[0].set_ylim(ymin, ymax)

        # chiMed on PFI FoV (right panel)

        ax[1].set_title("chiMed")
        sc = ax[1].scatter(
            pfiCenters.T[0],
            pfiCenters.T[1],
            marker="h",
            s=20,
            c=chiMed_all,
            alpha=0.7,
            rasterized=True,
            vmin=ymin, vmax=ymax,
        )
        ax[1].set_xlabel("X(PFI) [mm]")
        ax[1].set_ylabel("Y(PFI) [mm]")
        if sc is not None:
            fig.colorbar(sc, ax=ax[1], location="right", fraction=0.04, alpha=1.0, label="chiMed")

        caption_text = (
            "Note: chiMed is the median of the Chi values "
            "(residuals of measured image and modeled image divided by sqrt(variance)) "
            "calculated across all pixels within a region of each fiber."
        )
        fig.text(0.02, 0.02, caption_text, ha="left", fontsize=self.config.footnoteSize)

        plt.subplots_adjust(wspace=0.3, bottom=0.15)

        qaCombinedStatsPdf.append(fig, bbox_inches="tight")
        plt.close(fig)

        # chiStd

        fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[2, 1.5]))
        fig.suptitle(f"visit={visit}, arm={arm}, spectrograph=1,2,3,4", y=1.1)

        # chiStd vs. fiberId (left panel)
        ax[0].set_title("chiStd vs. fiberIds")

        ymin = self.config.plotMinChiStd
        ymax = self.config.plotMaxChiStd

        mask_under = chiStd_all < ymin
        mask_over = chiStd_all > ymax
        mask_inside = ~(mask_under | mask_over)

        sc = ax[0].scatter(
            fiberIds_all[mask_inside],
            chiStd_all[mask_inside],
            s=5,
            c=pfsArmAve_all[mask_inside],
            vmin=vmin, vmax=vmax,
            rasterized=True
        )
        if np.any(mask_over):
            ax[0].scatter(
                fiberIds_all[mask_over],
                np.full(np.sum(mask_over), ymax),
                marker="^",
                s=50,
                c=pfsArmAve_all[mask_over],
                vmin=vmin, vmax=vmax,
                edgecolors="k",
                rasterized=True,
                clip_on=False
            )
        if np.any(mask_under):
            ax[0].scatter(
                fiberIds_all[mask_under],
                np.full(np.sum(mask_under), ymin),
                marker="v",
                s=50,
                c=pfsArmAve_all[mask_under],
                vmin=vmin, vmax=vmax,
                edgecolors="k",
                rasterized=True,
                clip_on=False
            )
        if sc is not None:
            fig.colorbar(sc, ax=ax[0], location="right", fraction=0.04, alpha=1.0, label="pfsArmAve")
        ax[0].grid(color="gray", linestyle=":", linewidth=0.5)
        ax[0].set_xlabel("fiberId")
        ax[0].set_ylabel("chiStd")
        ax[0].set_ylim(ymin, ymax)

        # chiStd on PFI FoV (right panel)

        ax[1].set_title("chiStd")
        sc = ax[1].scatter(
            pfiCenters.T[0],
            pfiCenters.T[1],
            marker="h",
            s=20,
            c=chiStd_all,
            alpha=0.7,
            rasterized=True,
            vmin=ymin, vmax=ymax,
        )
        ax[1].set_xlabel("X(PFI) [mm]")
        ax[1].set_ylabel("Y(PFI) [mm]")
        if sc is not None:
            fig.colorbar(sc, ax=ax[1], location="right", fraction=0.04, alpha=1.0, label="chiStd")

        caption_text = (
            "Note: chiStd is the standard deviation of the Chi values "
            "(residuals of measured image and modeled image divided by sqrt(variance)) "
            "calculated across all pixels within a region of each fiber."
        )
        fig.text(0.02, 0.02, caption_text, ha="left", fontsize=self.config.footnoteSize)

        plt.subplots_adjust(wspace=0.3, bottom=0.15)

        qaCombinedStatsPdf.append(fig, bbox_inches="tight")
        plt.close(fig)

        # chiAtPeak

        fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[2, 1.5]))
        fig.suptitle(f"visit={visit}, arm={arm}, spectrograph=1,2,3,4", y=1.1)

        # chiAtPeak vs. fiberId (left panel)
        ax[0].set_title("chiAtPeak vs. fiberIds")

        ymin = self.config.plotMinChiAtPeak
        ymax = self.config.plotMaxChiAtPeak

        mask_under = chiAtPeak_all < ymin
        mask_over = chiAtPeak_all > ymax
        mask_inside = ~(mask_under | mask_over)

        sc = ax[0].scatter(
            fiberIds_all[mask_inside],
            chiAtPeak_all[mask_inside],
            s=5,
            c=pfsArmAve_all[mask_inside],
            vmin=vmin, vmax=vmax,
            rasterized=True,
        )
        if np.any(mask_over):
            ax[0].scatter(
                fiberIds_all[mask_over],
                np.full(np.sum(mask_over), ymax),
                marker="^",
                s=50,
                c=pfsArmAve_all[mask_over],
                vmin=vmin, vmax=vmax,
                edgecolors="k",
                rasterized=True,
                clip_on=False
            )
        if np.any(mask_under):
            ax[0].scatter(
                fiberIds_all[mask_under],
                np.full(np.sum(mask_under), ymin),
                marker="v",
                s=50,
                c=pfsArmAve_all[mask_under],
                vmin=vmin, vmax=vmax,
                edgecolors="k",
                rasterized=True,
                clip_on=False
            )
        if sc is not None:
            fig.colorbar(sc, ax=ax[0], location="right", fraction=0.04, alpha=1.0, label="pfsArmAve")
        ax[0].grid(color="gray", linestyle=":", linewidth=0.5)
        ax[0].set_xlabel("fiberId")
        ax[0].set_ylabel("chiAtPeak")
        ax[0].set_ylim(ymin, ymax)

        # chiAtPeak on PFI FoV (right panel)

        ax[1].set_title("chiAtPeak")
        sc = ax[1].scatter(
            pfiCenters.T[0],
            pfiCenters.T[1],
            marker="h",
            s=20,
            c=chiAtPeak_all,
            alpha=0.7,
            rasterized=True,
            vmin=ymin, vmax=ymax,
        )
        ax[1].set_xlabel("X(PFI) [mm]")
        ax[1].set_ylabel("Y(PFI) [mm]")
        if sc is not None:
            fig.colorbar(sc, ax=ax[1], location="right", fraction=0.04, alpha=1.0, label="chiAtPeak")

        caption_text = (
            "Note: chiAtPeak is the median of the Chi values "
            "(residuals of measured image and modeled image divided by sqrt(variance)) "
            "calculated across all pixels at the peaks of trace of each fiber."
        )
        fig.text(0.02, 0.02, caption_text, ha="left", fontsize=self.config.footnoteSize)

        plt.subplots_adjust(wspace=0.3, bottom=0.15)

        qaCombinedStatsPdf.append(fig, bbox_inches="tight")
        plt.close(fig)

        # resFrac

        fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[2, 1.5]))
        fig.suptitle(f"visit={visit}, arm={arm}, spectrograph=1,2,3,4", y=1.1)

        # resFrac vs. fiberId (left panel)
        ax[0].set_title("resFrac vs. fiberIds")

        ymin = self.config.plotMinResFrac
        ymax = self.config.plotMaxResFrac

        mask_under = resFrac_all < ymin
        mask_over = resFrac_all > ymax
        mask_inside = ~(mask_under | mask_over)

        sc = ax[0].scatter(
            fiberIds_all[mask_inside],
            resFrac_all[mask_inside],
            s=5,
            c=pfsArmAve_all[mask_inside],
            vmin=vmin, vmax=vmax,
            rasterized=True,
        )
        if np.any(mask_over):
            ax[0].scatter(
                fiberIds_all[mask_over],
                np.full(np.sum(mask_over), ymax),
                marker="^",
                s=50,
                c=pfsArmAve_all[mask_over],
                vmin=vmin, vmax=vmax,
                edgecolors="k",
                rasterized=True,
                clip_on=False
            )
        if np.any(mask_under):
            ax[0].scatter(
                fiberIds_all[mask_under],
                np.full(np.sum(mask_under), ymin),
                marker="v",
                s=50,
                c=pfsArmAve_all[mask_under],
                vmin=vmin, vmax=vmax,
                edgecolors="k",
                rasterized=True,
                clip_on=False
            )
        if sc is not None:
            fig.colorbar(sc, ax=ax[0], location="right", fraction=0.04, alpha=1.0, label="pfsArmAve")
        ax[0].grid(color="gray", linestyle=":", linewidth=0.5)
        ax[0].set_xlabel("fiberId")
        ax[0].set_ylabel("resFrac")
        ax[0].set_ylim(ymin, ymax)

        # resFrac on PFI FoV (right panel)

        ax[1].set_title("resFrac")
        sc = ax[1].scatter(
            pfiCenters.T[0],
            pfiCenters.T[1],
            marker="h",
            s=20,
            c=resFrac_all,
            alpha=0.7,
            rasterized=True,
            vmin=ymin, vmax=ymax,
        )
        ax[1].set_xlabel("X(PFI) [mm]")
        ax[1].set_ylabel("Y(PFI) [mm]")
        if sc is not None:
            fig.colorbar(sc, ax=ax[1], location="right", fraction=0.04, alpha=1.0, label="resFrac")

        caption_text = (
            "Note: resFrac is the median, per fiber, of sum(residuals) / sum(original values) "
        )
        fig.text(0.02, 0.02, caption_text, ha="left", fontsize=self.config.footnoteSize)

        plt.subplots_adjust(wspace=0.3, bottom=0.15)

        qaCombinedStatsPdf.append(fig, bbox_inches="tight")
        plt.close(fig)

        return qaCombinedStatsPdf
