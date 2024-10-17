from typing import Iterable

import pandas as pd
import seaborn as sb
from lsst.pex.config import Field
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base.connectionTypes import (
    Input as InputConnection,
    Output as OutputConnection,
)
from matplotlib import pyplot as plt
from pandas import DataFrame

from pfs.drp.qa.utils.plotting import description_palette, detector_palette, plot_exposures
from pfs.drp.qa.utils.storageClasses import MultipagePdfFigure


class DetectorMapCombinedQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "arm", "spectrograph"),
):
    """Connections for DetectorMapCombinedQaTask"""

    dmQaResidualStats = InputConnection(
        name="dmQaResidualStats",
        doc="DM QA residual statistics",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "exposure",
            "arm",
            "spectrograph",
        ),
        multiple=True,
    )

    dmQaCombinedResidualPlot = OutputConnection(
        name="dmQaCombinedResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for the entire detector.",
        storageClass="Plot",
        dimensions=(
            "instrument",
            "arm",
            "spectrograph",
        ),
    )

    dmQaDetectorStats = OutputConnection(
        name="dmQaDetectorStats",
        doc="Statistics of the residual analysis for the entire detector.",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "arm",
            "spectrograph",
        ),
    )


class DetectorMapCombinedQaConfig(PipelineTaskConfig, pipelineConnections=DetectorMapCombinedQaConnections):
    """Configuration for DetectorMapCombinedQaTask"""

    useSigmaRange = Field(dtype=bool, default=False, doc="Use ±2.5 sigma as range")


class DetectorMapCombinedQaTask(PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapCombinedQaConfig
    _DefaultName = "dmCombinedResiduals"

    def run(self, dmQaResidualStats: Iterable[DataFrame]) -> Struct:
        """Create detector level stats and plots.

        Parameters
        ----------
        dmQaResidualStats : Iterable[DataFrame]
            A an iterable of DataFrames containing DM QA residual statistics. These
            are combined into a single DataFrame for processing.

        Returns
        -------
        dmQaCombinedResidualPlot : `MultipagePdfFigure`
            1D and 2D plots of the residual between the detectormap and the arclines for the entire detector.
        dmQaDetectorStats : `pd.DataFrame`
            Statistics of the residual analysis.
        """
        stats = pd.concat(dmQaResidualStats)

        pdf = MultipagePdfFigure()

        pdf.append(plot_detector_summary(stats))
        pdf.append(plot_detector_summary_per_desc(stats))

        for ccd in stats.ccd.unique():
            fig = plot_detector_exposures(stats, ccd)
            pdf.append(fig)

        return Struct(dmQaCombinedResidualPlot=pdf, dmQaDetectorStats=stats)


def plot_detector_exposures(data, ccd):
    plot_data = data.query("ccd == @ccd")

    summary_stats = plot_data.filter(regex="median|weighted").median().to_dict()

    fig = plot_exposures(plot_data, palette=description_palette)

    fig.axes[0].axvline(summary_stats["spatial.median"], c="k", ls="--")
    fig.axes[0].axvline(
        summary_stats["spatial.median"] + summary_stats["spatial.weightedRms"], c="g", ls="--"
    )
    fig.axes[0].axvline(
        summary_stats["spatial.median"] - summary_stats["spatial.weightedRms"], c="g", ls="--"
    )
    fig.axes[0].set_title(
        f'Spatial: median={summary_stats["spatial.median"]:5.04f} rms={summary_stats["spatial.weightedRms"]:5.04f}'
    )

    fig.axes[1].axvline(summary_stats["wavelength.median"], c="k", ls="--")
    fig.axes[1].axvline(
        summary_stats["wavelength.median"] + summary_stats["wavelength.weightedRms"], c="g", ls="--"
    )
    fig.axes[1].axvline(
        summary_stats["wavelength.median"] - summary_stats["wavelength.weightedRms"], c="g", ls="--"
    )
    fig.axes[1].set_title(
        f'Wavelength: median={summary_stats["wavelength.median"]:5.04f} rms={summary_stats["wavelength.weightedRms"]:5.04f}'
    )

    fig.set_size_inches(8, 8)
    fig.suptitle(f"{fig.get_suptitle()}\n{ccd}")

    return fig


def plot_detector_summary(data):
    plot_data = data.filter(regex="ccd|median|weighted").groupby("ccd").median()

    fig, (ax0, ax1) = plt.subplots(ncols=2, layout="constrained", sharey=True)
    fig.set_size_inches(12, 3)

    for ccd, row in plot_data.iterrows():
        ax0.errorbar(
            x=ccd,
            y=row["spatial.median"],
            yerr=row["spatial.weightedRms"],
            marker="o",
            color=detector_palette[ccd[0]],
        )
        ax1.errorbar(
            x=ccd,
            y=row["wavelength.median"],
            yerr=row["wavelength.weightedRms"],
            marker="o",
            color=detector_palette[ccd[0]],
        )

    ax0.axhline(0, c="k", ls="--", alpha=0.3)
    ax0.set_title("Spatial median and weightedRms")

    ax1.axhline(0, c="k", ls="--", alpha=0.3)
    ax1.set_title("Wavelength median and weightedRms")

    fig.suptitle(f"DetectorMap residuals summary")

    return fig


def plot_detector_summary_per_desc(data):
    plot_data = (
        data.set_index(["ccd", "description"])
        .filter(regex="median|weighted")
        .melt(ignore_index=False)
        .reset_index()
    )

    fg = sb.catplot(
        data=plot_data,
        x="ccd",
        y="value",
        col="variable",
        col_wrap=2,
        hue="description",
        kind="box",
        sharey=False,
        height=3,
        aspect=2,
        flierprops={"marker": ".", "ms": 2},
    )
    fg.fig.suptitle(f"DetectorMap Residuals by description", y=1)

    return fg.fig
