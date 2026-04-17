"""Image quality QA pipeline task.

Produces per-detector FWHM and image-shape QA plots from arc-line
second-moment measurements.
"""

import numpy as np
import pandas as pd
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
from pfs.drp.stella import ArcLineSet, DetectorMap
from pfs.drp.stella.utils.quality import computeImageQuality, opaqueColorbar
from pfs.drp.stella.utils.stability import addTraceLambdaToArclines

from pfs.drp.qa.storageClasses import MultipagePdfFigure

__all__ = ["ImageQualityQaTask", "plotImageQuality"]


class ImageQualityQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Connections for ImageQualityQaTask."""

    arcLines = InputConnection(
        name="lines",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    detectorMap = InputConnection(
        name="detectorMap",
        doc="Calibrated detector mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    iqQaData = OutputConnection(
        name="iqQaData",
        doc="Per-line image quality measurements, including FWHM and position angle.",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    iqQaPlot = OutputConnection(
        name="iqQaPlot",
        doc="Image quality QA plots (one page per enabled plot type).",
        storageClass="MultipagePdfFigure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class ImageQualityQaConfig(PipelineTaskConfig, pipelineConnections=ImageQualityQaConnections):
    """Configuration for ImageQualityQaTask."""

    showWhisker = Field(
        dtype=bool,
        default=True,
        doc="Show a whisker/quiver plot of FWHM and position angle.",
    )
    showFWHM = Field(
        dtype=bool,
        default=False,
        doc="Show a 2D spatial hexbin map of FWHM.",
    )
    showFWHMAgainstLambda = Field(
        dtype=bool,
        default=False,
        doc="Plot FWHM vs log(flux) (or S/N if useSN is True).",
    )
    showFWHMHistogram = Field(
        dtype=bool,
        default=False,
        doc="Show a histogram of FWHM values.",
    )
    showFluxHistogram = Field(
        dtype=bool,
        default=False,
        doc="Show a histogram of line fluxes.",
    )
    useSN = Field(
        dtype=bool,
        default=False,
        doc="Use signal/noise rather than log10(flux) in the showFWHMAgainstLambda plot.",
    )
    minFluxPercentile = Field(
        dtype=float,
        default=10.0,
        doc="Minimum flux percentile for line selection in spatial plots.",
    )
    vmin = Field(dtype=float, default=2.5, doc="Minimum FWHM for color scale (pixels).")
    vmax = Field(dtype=float, default=3.5, doc="Maximum FWHM for color scale (pixels).")
    logScale = Field(dtype=bool, default=True, doc="Log y-axis for histogram plots.")
    gridsize = Field(
        dtype=int,
        default=100,
        doc="Grid size for hexbin FWHM map. Use <=0 for scatter plot instead.",
    )
    stride = Field(
        dtype=int,
        default=1,
        doc="Fiber stride for downsampling arc lines in spatial plots.",
    )


class ImageQualityQaTask(PipelineTask):
    """QA task measuring image quality from arc-line second moments.

    Reads per-detector arc line measurements (``lines`` dataset), enriches
    them with wavelength and trace-position information from the
    ``detectorMap``, computes Gaussian-equivalent FWHM metrics, and produces
    configurable QA plots as a ``MultipagePdfFigure`` (one page per enabled
    plot type).
    """

    ConfigClass = ImageQualityQaConfig
    _DefaultName = "imageQualityQa"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        dataId = dict(**inputRefs.arcLines.dataId.mapping)
        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = dataId
        try:
            outputs = self.run(**inputs)
        except Exception as e:
            self.log.error("ImageQualityQaTask failed for %s: %s", dataId, e)
            raise
        else:
            butlerQC.put(outputs, outputRefs)

    def run(
        self,
        arcLines: ArcLineSet,
        detectorMap: DetectorMap,
        dataId: dict,
    ) -> Struct:
        """Compute image quality metrics and generate QA plots.

        Parameters
        ----------
        arcLines : `ArcLineSet`
            Arc line measurements.
        detectorMap : `DetectorMap`
            Calibrated detector mapping from fiberId,wavelength to x,y.
        dataId : `dict`
            The dataId for this quantum, used to label outputs and annotate
            the output DataFrame with ``visit``, ``arm``, ``spectrograph``.

        Returns
        -------
        iqQaData : `pandas.DataFrame`
            Per-line image quality data including ``fwhm``, ``theta``,
            ``traceOnly``, plus ``visit``, ``arm``, ``spectrograph`` for
            downstream aggregation by `ImageQualityCombinedQaTask`.
        iqQaPlot : `MultipagePdfFigure`
            QA plots, one page per enabled plot type.
        """
        self.log.info("Computing image quality metrics for %s", dataId)
        als = addTraceLambdaToArclines(arcLines, detectorMap)
        data = computeImageQuality(als)

        for key in ("visit", "arm", "spectrograph"):
            if key in dataId:
                data[key] = dataId[key]

        title = "{visit} {arm}{spectrograph}".format(**dataId)
        self.log.info("Generating image quality plots for %s", dataId)
        pdf = self._makePlots(data, title=title)

        return Struct(iqQaData=data, iqQaPlot=pdf)

    def _makePlots(self, data: pd.DataFrame, title: str = "") -> MultipagePdfFigure:
        """Generate all enabled QA plots.

        Parameters
        ----------
        data : `pandas.DataFrame`
            Output of `computeImageQuality`, annotated with dataId fields.
        title : `str`, optional
            Figure suptitle.

        Returns
        -------
        `MultipagePdfFigure`
            Enabled plot types, one page each.
        """
        cfg = self.config
        pdf = MultipagePdfFigure()

        plotKwargs = dict(
            minFluxPercentile=cfg.minFluxPercentile,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            logScale=cfg.logScale,
            gridsize=cfg.gridsize,
            stride=cfg.stride,
            useSN=cfg.useSN,
        )

        for plotMode in (
            "showWhisker",
            "showFWHM",
            "showFWHMAgainstLambda",
            "showFWHMHistogram",
            "showFluxHistogram",
        ):
            if not getattr(cfg, plotMode):
                continue

            fig, ax = plt.subplots(layout="constrained")
            C, colorbarLabel = plotImageQuality(ax, data, **{plotMode: True}, **plotKwargs)
            if C is not None:
                with opaqueColorbar(C):
                    fig.colorbar(C, label=colorbarLabel)
            if title:
                fig.suptitle(title)
            pdf.append(fig)
            plt.close(fig)

        return pdf


def plotImageQuality(
    ax,
    data: pd.DataFrame,
    *,
    showWhisker: bool = False,
    showFWHM: bool = False,
    showFWHMAgainstLambda: bool = False,
    showFWHMHistogram: bool = False,
    showFluxHistogram: bool = False,
    minFluxPercentile: float = 10,
    vmin: float = 2.5,
    vmax: float = 3.5,
    logScale: bool = True,
    gridsize: int = 100,
    stride: int = 1,
    useSN: bool = False,
):
    """Draw a single image-quality panel on *ax*.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes to draw on.
    data : `pandas.DataFrame`
        Output of `computeImageQuality`.  Must contain ``fwhm``, ``theta``,
        ``traceOnly``, ``x``, ``y``, ``flux``, ``fluxErr``, ``flag``,
        ``fiberId``, and ``lam`` columns.
    showWhisker : `bool`
        Draw FWHM as a whisker (quiver) plot coloured by FWHM magnitude.
    showFWHM : `bool`
        Draw a 2D spatial hexbin / scatter map of FWHM.
    showFWHMAgainstLambda : `bool`
        Scatter FWHM vs log(flux) or S/N, coloured by wavelength.
    showFWHMHistogram : `bool`
        Histogram of FWHM values.
    showFluxHistogram : `bool`
        Histogram of line fluxes.
    minFluxPercentile : `float`
        Minimum flux percentile for line selection in spatial plots.
    vmin, vmax : `float`
        FWHM color-scale range (pixels).
    logScale : `bool`
        Log y-axis for histograms.
    gridsize : `int`
        hexbin grid size; use ``<=0`` for scatter plot instead.
    stride : `int`
        Fiber-ID stride for downsampling in spatial plots.
    useSN : `bool`
        Use S/N instead of log10(flux) on the x-axis of FWHM-vs-λ.

    Returns
    -------
    C : `matplotlib.cm.ScalarMappable` or ``None``
        Colorable artist suitable for passing to ``fig.colorbar()``,
        or ``None`` when no colorbar is applicable.
    colorbarLabel : `str` or ``None``
        Colorbar label string, or ``None``.
    """
    plt.sca(ax)

    traceOnly = bool(data["traceOnly"].iloc[0]) if len(data) > 0 else False
    fwhm = data["fwhm"]
    theta = data["theta"]
    C = None
    colorbarLabel = None

    if showWhisker or showFWHM or showFWHMAgainstLambda:
        q10_arr = np.nanpercentile(data["flux"], [minFluxPercentile])
        q10 = np.nan if np.isnan(q10_arr).all() else float(q10_arr[0])

        ll = np.isfinite(data["xx"] if traceOnly else data["xx"] + data["xy"] + data["yy"])
        ll &= data["flag"] == False  # noqa: E712
        ll &= fwhm < 8
        ll &= data["flux"] > q10
        if stride > 1:
            ll &= (data["fiberId"] % stride) == 0

        norm = plt.Normalize(vmin, vmax)
        colorbarLabel = "FWHM (pixels)"

        if showWhisker:
            imageSize = 4096
            arrowSize = 4
            cmap = plt.colormaps["viridis"]
            C = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            Q = plt.quiver(
                data["x"][ll], data["y"][ll],
                (fwhm * np.cos(theta))[ll], (fwhm * np.sin(theta))[ll],
                fwhm[ll], cmap=cmap, norm=norm,
                headwidth=0, pivot="middle",
                angles="xy", scale_units="xy", scale=arrowSize * 30 / imageSize,
            )
            plt.quiverkey(Q, 0.1, 1.025, arrowSize, label=f"{arrowSize:.2g} pixels")
        elif showFWHM:
            if gridsize <= 0:
                C = plt.scatter(data["x"][ll], data["y"][ll], c=fwhm[ll], s=5, norm=norm)
            else:
                C = plt.hexbin(data["x"][ll], data["y"][ll], fwhm[ll], norm=norm, gridsize=gridsize)
        elif showFWHMAgainstLambda:
            xarr = data["flux"] / data["fluxErr"] if useSN else np.log10(data["flux"])
            C = plt.scatter(xarr[ll], fwhm[ll], c=data["lam"][ll], marker=".", alpha=0.75)
            colorbarLabel = "Wavelength (nm)"
            plt.xlabel("Signal/Noise" if useSN else "lg(flux)")
            plt.ylabel("FWHM (pixels)")

        if not showFWHMAgainstLambda:
            plt.ylim(-1, 4096)
            plt.xlim(-1, 4096)
            plt.xlabel("x (pixels)")
            plt.ylabel("y (pixels)")
            ax.set_aspect(1)
    else:
        if showFWHMHistogram:
            plt.hist(fwhm, bins=np.linspace(0, 10, 100))
            plt.xlabel("FWHM (pix)")
        elif showFluxHistogram:
            q99 = np.nanpercentile(data["flux"], [99])[0]
            plt.hist(data["flux"], bins=np.linspace(0, q99, 100))
            plt.xlabel("flux")
        if logScale:
            plt.yscale("log")

    return C, colorbarLabel
