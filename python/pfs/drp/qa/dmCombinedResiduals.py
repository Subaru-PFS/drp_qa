import itertools
from itertools import product
from typing import Dict, Iterable, Optional

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
from pfs.drp.stella import DetectorMap

from pfs.drp.qa.dmResiduals import plot_detectormap_residuals
from pfs.drp.qa.storageClasses import MultipagePdfFigure
from pfs.drp.qa.utils.plotting import description_palette, detector_palette


class DetectorMapCombinedResidualsConnections(
    PipelineTaskConnections,
    dimensions=("instrument",),
):
    """Connections for DetectorMapCombinedQaTask"""

    detectorMaps = InputConnection(
        name="detectorMap",
        doc="Adjusted detector mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=(
            "instrument",
            "visit",
            "arm",
            "spectrograph",
        ),
        multiple=True,
    )

    dmQaResidualData = InputConnection(
        name="dmQaResidualData",
        doc="DM QA residual data for plotting",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "visit",
            "arm",
            "spectrograph",
        ),
        multiple=True,
    )

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

    def run(
        self,
        detectorMaps: Iterable[DetectorMap],
        dmQaResidualData: Iterable[DataFrame],
        dmQaResidualStats: Iterable[DataFrame],
        run_name: str,
    ) -> Struct:
        """Create detector level residual_stats and plots.

        Parameters
        ----------
        detectorMaps : Iterable[DetectorMap]
            An iterable of detector maps. Used for plotting metadata.
        dmQaResidualData : Iterable[DataFrame]
            An iterable of DataFrames containing DM QA residual data. These
            are combined into a single DataFrame for processing.
        dmQaResidualStats : Iterable[DataFrame]
            An iterable of DataFrames containing DM QA residual statistics. These
            are combined into a single DataFrame for processing.
        run_name : str
            The name of the collection that was used for the residual_stats.

        Returns
        -------
        dmQaCombinedResidualPlot : `MultipagePdfFigure`
            1D and 2D plots of the residual between the detectormap and the arclines for the entire detector.
        dmQaDetectorStats : `pd.DataFrame`
            Statistics of the residual analysis.
        """
        # Put the DetectorMaps in a dict by CCD.
        self.log.debug(f"Visits: {set([dm.getVisitInfo().id for dm in detectorMaps])}")

        # Small helper to use while https://pfspipe.ipmu.jp/jira/browse/PIPE2D-1423
        def get_ccd(dm: DetectorMap) -> str:
            return "".join([x.split("=")[1] for x in dm.metadata["CALIB_ID"].split(" ")[:2]])

        detectorMaps = {get_ccd(detectorMap): detectorMap for detectorMap in detectorMaps}
        self.log.debug(f"DetectorMap CCDs: {detectorMaps.keys()}")

        residual_data = pd.concat(dmQaResidualData)
        residual_stats = pd.concat(dmQaResidualStats)
        residual_stats.sort_values(by=["visit", "arm", "spectrograph", "description"], inplace=True)

        # Put the CCD column in a wavelength sorted order.
        residual_stats.ccd = residual_stats.ccd.astype("category")
        spec_order = [1, 2, 3, 4]
        arm_order = ["b", "r", "m", "n"]
        detector_order = [f"{arm}{spec}" for arm, spec in itertools.product(arm_order, spec_order)]
        detector_order = [d for d in detector_order if d in residual_stats.ccd.cat.categories]
        residual_stats.ccd = residual_stats.ccd.cat.reorder_categories(detector_order, ordered=True)

        self.log.info("Making combined report")
        pdf = make_report(residual_stats, residual_data, detectorMaps, run_name=run_name, log=self.log)

        return Struct(dmQaCombinedResidualPlot=pdf, dmQaDetectorStats=residual_stats)


def make_report(
    residual_stats: DataFrame,
    arc_data: DataFrame,
    detectorMaps: Dict[str, DetectorMap],
    run_name: str,
    log: object,
) -> MultipagePdfFigure:
    pdf = MultipagePdfFigure()

    reserved_stats = residual_stats.query("status_type == 'RESERVED'").copy()

    # Add the title as a figure.
    pdf.append(plot_title(run_name))

    # Add the table data as a figure.
    pdf.append(plot_dataframe(reserved_stats))

    # Detector summaries.
    log.info("Making detector summary plots")
    pdf.append(plot_detector_summary(reserved_stats))
    pdf.append(plot_detector_summary_per_desc(reserved_stats))

    plot_cols = [
        "fiberId",
        "wavelength",
        "x",
        "xErr",
        "y",
        "yErr",
        "isTrace",
        "isLine",
        "xResid",
        "yResid",
        "xResidOutlier",
        "yResidOutlier",
        "isUsed",
        "isReserved",
        "status",
        "visit",
    ]

    # Per visit descriptions.
    for ccd, visit_stats in residual_stats.groupby("ccd", observed=False):
        log.info(f"Making plots for {ccd}")
        try:
            # Add the 2D residual plot.
            arm = ccd[0]
            spec = int(ccd[1])
            data = arc_data.query(f"arm == '{arm}' and spectrograph == {spec}")

            # If we are doing a combined report we want to get the mean across visits.
            grouped = data[plot_cols].groupby(["status", "isLine", "fiberId", "y"])
            data = grouped.mean().reset_index()

            residFig = plot_detectormap_residuals(data, visit_stats, detectorMaps[str(ccd)])
            residFig.suptitle(f"DetectorMap Residuals - Median of all visits - {ccd}", weight="bold")
            pdf.append(residFig, dpi=150)

            # Add the description per visit breakdown.
            fig = plot_detector_visits(visit_stats)
            fig.suptitle(f"{fig.get_suptitle()}\n{ccd}")
            pdf.append(fig)
        except KeyError:
            log.warning(f"DetectorMap not found for {ccd}. Skipping.")
        except Exception as e:
            log.warning(f"Error plotting for {ccd}: {e}")
            continue

    return pdf


def plot_detector_visits(plot_data: DataFrame) -> Figure:
    summary_stats = plot_data.filter(regex="median|weighted").mean().to_dict()

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

    return fig


def plot_detector_summary(stats: DataFrame) -> Figure:
    plot_data_spatial = (
        stats.query("description == 'Trace'")
        .filter(regex="ccd|median|weighted|soften")
        .groupby("ccd", observed=False)
        .mean()
    )
    plot_data_wavelength = (
        stats.query("description != 'Trace'")
        .filter(regex="ccd|median|weighted|soften")
        .groupby("ccd", observed=False)
        .mean()
    )

    fig, (ax0, ax1) = plt.subplots(ncols=2, sharey=True, layout="constrained")
    fig.set_size_inches(12, 4)

    for ccd, row in plot_data_spatial.iterrows():
        ax0.errorbar(
            x=ccd,
            y=row["spatial.median"],
            yerr=row["spatial.weightedRms"],
            markersize=max(row["spatial.softenFit"].mean() * 100, 1),
            marker="o",
            mec="k",
            linewidth=2,
            capsize=2,
            color=detector_palette[ccd[0]],
        )

    for ccd, row in plot_data_wavelength.iterrows():
        ax1.errorbar(
            x=ccd,
            y=row["wavelength.median"],
            yerr=row["wavelength.weightedRms"],
            markersize=max(row["wavelength.softenFit"].mean() * 100, 1),
            marker="o",
            markeredgecolor="k",
            linewidth=2,
            capsize=2,
            color=detector_palette[ccd[0]],
        )

    ax0.axhline(0, c="k", ls="--", alpha=0.3)
    ax0.set_title("Spatial median and weightedRms error (quartz only)")
    ax0.set_ylabel("Median (pixel)")

    ax1.axhline(0, c="k", ls="--", alpha=0.3)
    ax1.set_title("Wavelength median and weightedRms")

    ax0.grid(True, color="k", linestyle="--", alpha=0.15)
    ax1.grid(True, color="k", linestyle="--", alpha=0.15)

    return fig


def plot_detector_summary_per_desc(stats: DataFrame) -> Figure:
    plot_data = (
        stats.set_index(["ccd", "description"])
        .filter(regex="median|weighted|soften")
        .melt(ignore_index=False)
        .reset_index()
    )
    plot_data.loc[
        plot_data.query('description != "Trace" and variable.str.startswith("spatial")').index, "value"
    ] = pd.NA

    col_order = [
        f"{b}.{a}" for a, b in product(["median", "weightedRms", "softenFit"], ["spatial", "wavelength"])
    ]

    fg = sb.catplot(
        data=plot_data.dropna(),
        x="ccd",
        y="value",
        col="variable",
        col_wrap=2,
        col_order=col_order,
        hue="description",
        palette=description_palette,
        kind="box",
        sharey=False,
        height=3,
        aspect=2.5,
        flierprops={"marker": ".", "ms": 2},
    )
    fg.fig.suptitle("DetectorMap Residuals by description", y=1)
    for i, ax in enumerate(fg.figure.axes):
        ax.set_ylabel("Median residual (pixel)")
        if i == 0:
            # Should be spatial median.
            ax.set_ylim(-0.001, 0.001)
        elif i == 1:
            # Should be wavelength median.
            ax.set_ylim(-0.05, 0.05)
        else:
            # Last plots are the rms and soften.
            ax.set_ylim(0.0, 0.5)

        ax.axhline(0, c="k", ls="--", alpha=0.3)

        # Get the x-axis tick labels and their positions
        xticks = ax.get_xticks()

        # Shade alternate backgrounds
        for j in range(0, len(xticks), 2):
            ax.fill_between(
                [xticks[j] - 0.5, xticks[j] + 0.5],
                0,
                1,
                transform=ax.get_xaxis_transform(),
                color="lightgray",
                alpha=0.5,
            )

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


def plot_dataframe(stats: DataFrame) -> Figure:
    """Plot the residual data frame.

    Parameters
    ----------
    stats : `pandas.DataFrame`
        The data frame to plot.

    Returns
    -------
    fig : `Figure`
        The figure.
    """
    plot_data_spatial = (
        stats.query("description == 'Trace'")
        .filter(regex="ccd|spatial.(median|weighted|soften)")
        .groupby("ccd", observed=False)
        .mean()
    )
    plot_data_spatial.columns = [c.replace("spatial.", "") for c in plot_data_spatial.columns]
    plot_data_wavelength = (
        stats.query("description != 'Trace'")
        .filter(regex="ccd|wavelength.(median|weighted|soften)")
        .groupby("ccd", observed=False)
        .mean()
    )
    plot_data_wavelength.columns = [c.replace("wavelength.", "") for c in plot_data_wavelength.columns]

    formatter = "{:5.04f}".format

    fig = Figure(layout="constrained")
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)
    ax0.set_axis_off()
    ax1.set_axis_off()
    t0 = pd.plotting.table(ax0, plot_data_spatial.map(formatter), loc="center")
    t0.set_fontsize(11)
    t1 = pd.plotting.table(ax1, plot_data_wavelength.map(formatter), loc="center")
    t1.set_fontsize(11)

    ax0.set_title("Spatial (quartz only)", y=1.12)
    ax1.set_title("Wavelength", y=1.12)

    fig.suptitle("Residuals summary", y=1.15)

    return fig


def plot_title(run_name: str) -> Figure:
    """Plot a title page for the combined report."""
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.text(0.5, 0.5, "DetectorMap Residuals Summary", ha="center", va="center", fontsize="large")
    ax.text(0.5, 0.35, f"{run_name}", ha="center", va="center")
    return fig
