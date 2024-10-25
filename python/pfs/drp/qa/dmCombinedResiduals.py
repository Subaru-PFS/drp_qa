from typing import Iterable, Optional

import pandas as pd
import seaborn as sb
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
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame

from pfs.drp.qa.storageClasses import MultipagePdfFigure
from pfs.drp.qa.utils.plotting import description_palette, detector_palette


class DetectorMapCombinedResidualsConnections(
    PipelineTaskConnections,
    dimensions=("instrument",),
):
    """Connections for DetectorMapCombinedQaTask"""

    dmQaResidualStats = InputConnection(
        name="dmQaResidualStats",
        doc="DM QA residual statistics",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "visit",
            "arm",
            "spectrograph",
        ),
        multiple=True,
    )

    dmQaCombinedResidualPlot = OutputConnection(
        name="dmQaCombinedResidualPlot",
        doc="The 1D and 2D residual plots of the detectormap with the arclines for all detectors.",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument",),
    )

    dmQaDetectorStats = OutputConnection(
        name="dmQaDetectorStats",
        doc="Statistics of the residual analysis for all detectors.",
        storageClass="DataFrame",
        dimensions=("instrument",),
    )


class DetectorMapCombinedResidualsConfig(
    PipelineTaskConfig, pipelineConnections=DetectorMapCombinedResidualsConnections
):
    """Configuration for DetectorMapCombinedQaTask"""

    useSigmaRange = Field(dtype=bool, default=False, doc="Use ±2.5 sigma as range")


class DetectorMapCombinedResidualsTask(PipelineTask):
    """Task for QA of detectorMap"""

    ConfigClass = DetectorMapCombinedResidualsConfig
    _DefaultName = "dmCombinedResiduals"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        run_name = inputRefs.dmQaResidualStats[0].run

        inputs = butlerQC.get(inputRefs)
        inputs["run_name"] = run_name

        # Perform the actual processing.
        outputs = self.run(**inputs)

        # Store the results.
        butlerQC.put(outputs, outputRefs)

    def run(self, dmQaResidualStats: Iterable[DataFrame], run_name: str) -> Struct:
        """Create detector level stats and plots.

        Parameters
        ----------
        dmQaResidualStats : Iterable[DataFrame]
            A an iterable of DataFrames containing DM QA residual statistics. These
            are combined into a single DataFrame for processing.
        run_name : str
            The name of the collection that was used for the stats.

        Returns
        -------
        dmQaCombinedResidualPlot : `MultipagePdfFigure`
            1D and 2D plots of the residual between the detectormap and the arclines for the entire detector.
        dmQaDetectorStats : `pd.DataFrame`
            Statistics of the residual analysis.
        """
        stats = pd.concat(dmQaResidualStats).query('status_type == "RESERVED"')
        stats.sort_values(by=["visit", "arm", "spectrograph", "description"], inplace=True)

        stats.ccd = stats.ccd.astype("category")
        stats.ccd = stats.ccd.cat.as_ordered()

        self.log.info(stats.ccd.value_counts())

        pdf = MultipagePdfFigure()

        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            f"DetectorMap Residuals Summary\n{run_name}",
            ha="center",
            va="center",
            fontsize="xx-large",
        )
        pdf.append(fig)

        pdf.append(plot_detector_summary(stats))
        pdf.append(plot_detector_summary_per_desc(stats))

        for ccd in stats.ccd.unique():
            fig = plot_detector_visits(stats, ccd)
            pdf.append(fig)

        return Struct(dmQaCombinedResidualPlot=pdf, dmQaDetectorStats=stats)


def plot_detector_visits(data: DataFrame, ccd: str) -> Figure:
    plot_data = data.query("ccd == @ccd")

    summary_stats = plot_data.filter(regex="median|weighted").median().to_dict()

    fig = plot_visits(plot_data, palette=description_palette)

    for ax, dim in zip(fig.axes, ["spatial", "wavelength"]):
        upper_range = summary_stats[f"{dim}.median"] + summary_stats[f"{dim}.weightedRms"]
        lower_range = summary_stats[f"{dim}.median"] - summary_stats[f"{dim}.weightedRms"]

        ax.axvline(summary_stats[f"{dim}.median"], c="k", ls="--")
        ax.axvline(upper_range, c="g", ls="--")
        ax.axvline(lower_range, c="g", ls="--")
        ax.set_title(
            f"{dim.upper()}: "
            f'median={summary_stats[f"{dim}.median"]:5.04f} '
            f'rms={summary_stats[f"{dim}.weightedRms"]:5.04f}'
        )

    fig.set_size_inches(8, 8)
    fig.suptitle(f"{fig.get_suptitle()}\n{ccd}")

    return fig


def plot_detector_summary(data: DataFrame) -> Figure:
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


def plot_detector_summary_per_desc(data: DataFrame) -> Figure:
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
        palette=description_palette,
        kind="box",
        sharey=False,
        height=3,
        aspect=2,
        flierprops={"marker": ".", "ms": 2},
    )
    fg.fig.suptitle(f"DetectorMap Residuals by description", y=1)
    for ax in fg.axes:
        ax.axhline(0, c="k", ls="--", alpha=0.3)

    return fg.fig


def plot_visits(
    plotData: pd.DataFrame,
    palette: Optional[dict] = None,
    spatialRange: float = 0.1,
    wavelengthRange: float = 0.1,
    fig: Optional[Figure] = None,
) -> Figure:
    """Plot the visit statistics.

    Parameters
    ----------
    plotData : `pandas.DataFrame`
        The data.
    palette : `dict`, optional
        The palette to use for the arcline descriptions. Keys are the descriptions
        and values are the colors. Default is ``None``.
    spatialRange : `float`, optional
        The range for the spatial data. Default is 0.1.
    wavelengthRange : `float`, optional
        The range for the wavelength data. Default is 0.1.
    fig : `Figure`, optional
        The figure. Default is ``None``.

    Returns
    -------
    fig : `Figure`
        The visit statistics plot.

    """
    plotData = plotData.copy()
    fig = fig or Figure(layout="constrained")
    ax0 = fig.add_subplot(121)
    ax1 = fig.add_subplot(122, sharex=ax0, sharey=ax0)

    plotData["visit_idx"] = plotData.visit.rank(method="first")

    for ax, metric in zip([ax0, ax1], ["spatial", "wavelength"]):
        for desc, grp in plotData.groupby("description"):
            grp.plot.scatter(
                y="visit_idx",
                x=f"{metric}.median",
                xerr=f"{metric}.weightedRms",
                marker="o",
                color=palette.get(desc, "red") if palette is not None else None,
                label=desc,
                ax=ax,
            )

        ax.grid(alpha=0.2)
        ax.axvline(0, c="k", ls="--", alpha=0.5)
        ax.set_title(f"{metric}")
        ax.set_xlabel("pix")
        if spatialRange is not None and metric == "spatial":
            ax.set_xlim(-spatialRange, spatialRange)
        if wavelengthRange is not None and metric == "wavelength":
            ax.set_xlim(-wavelengthRange, wavelengthRange)

    visit_label = [f"{row.visit}" for idx, row in plotData.iterrows()]
    ax0.set_yticks(plotData.visit_idx, visit_label, fontsize="xx-small")
    ax0.set_ylabel("Visit")
    ax0.invert_yaxis()

    fig.suptitle("RESERVED median and 1-sigma weighted errors", fontsize="small")

    return fig
