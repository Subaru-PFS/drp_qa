"""Combined image quality QA pipeline task.

Aggregates per-detector image quality data across all visit/arm/spectrograph
quanta and produces a single multi-panel figure laid out as an
arm × spectrograph grid (one page per enabled plot type).
"""

from typing import Iterable

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

from pfs.drp.qa.imageQualityQa import plotImageQuality
from pfs.drp.qa.storageClasses import MultipagePdfFigure
from pfs.drp.stella.utils.quality import opaqueColorbar

__all__ = ["ImageQualityCombinedQaTask"]


class ImageQualityCombinedQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument",),
):
    """Connections for ImageQualityCombinedQaTask."""

    iqQaData = InputConnection(
        name="iqQaData",
        doc="Per-line image quality measurements from ImageQualityQaTask.",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    iqQaCombinedPlot = OutputConnection(
        name="iqQaCombinedPlot",
        doc=(
            "Combined image quality QA figure arranged as an arm × spectrograph"
            " grid (one page per enabled plot type)."
        ),
        storageClass="MultipagePdfFigure",
        dimensions=("instrument",),
    )


class ImageQualityCombinedQaConfig(
    PipelineTaskConfig, pipelineConnections=ImageQualityCombinedQaConnections
):
    """Configuration for ImageQualityCombinedQaTask."""

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
    maxFwhm = Field(dtype=float, default=8.0, doc="Upper FWHM cutoff for line selection and histogram binning (pixels).")
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


class ImageQualityCombinedQaTask(PipelineTask):
    """QA task that aggregates per-detector image quality into a combined figure.

    Collects ``iqQaData`` DataFrames produced by `ImageQualityQaTask` for all
    visit/arm/spectrograph quanta and produces a single multi-panel figure
    arranged as arms (rows) × spectrographs (columns), with one page per
    enabled plot type.
    """

    ConfigClass = ImageQualityCombinedQaConfig
    _DefaultName = "imageQualityCombinedQa"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, iqQaData: Iterable[pd.DataFrame]) -> Struct:
        """Build a combined multi-panel image quality figure.

        Parameters
        ----------
        iqQaData : iterable of `pandas.DataFrame`
            Per-detector quality DataFrames from `ImageQualityQaTask`, each
            containing ``visit``, ``arm``, and ``spectrograph`` columns.

        Returns
        -------
        iqQaCombinedPlot : `MultipagePdfFigure`
            Multi-panel figure with one page per enabled plot type.
        """
        combined = pd.concat(list(iqQaData), ignore_index=True)
        pdf = self._makePlots(combined)
        return Struct(iqQaCombinedPlot=pdf)

    def _makePlots(self, data: pd.DataFrame) -> MultipagePdfFigure:
        """Produce the arm × spectrograph multi-panel figure.

        Parameters
        ----------
        data : `pandas.DataFrame`
            Concatenated per-detector DataFrames.

        Returns
        -------
        `MultipagePdfFigure`
            One page per enabled plot type.
        """
        cfg = self.config
        pdf = MultipagePdfFigure()

        arm_order = [a for a in "brmn" if a in data["arm"].unique()]
        spec_order = sorted(data["spectrograph"].unique())
        visits = sorted(data["visit"].unique())

        plotKwargs = dict(
            minFluxPercentile=cfg.minFluxPercentile,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            maxFwhm=cfg.maxFwhm,
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

            ny, nx = len(arm_order), len(spec_order)
            fig, axs = plt.subplots(
                ny, nx, sharex=True, sharey=True, squeeze=False, layout="constrained"
            )

            C = None
            colorbarLabel = None
            for ax, (arm, spec) in zip(
                axs.flatten(), [(a, s) for a in arm_order for s in spec_order]
            ):
                subset = data[(data["arm"] == arm) & (data["spectrograph"] == spec)]
                if subset.empty:
                    ax.set_axis_off()
                    continue
                C_panel, lbl = plotImageQuality(ax, subset, **{plotMode: True}, **plotKwargs)
                if C_panel is not None:
                    C = C_panel
                    colorbarLabel = lbl
                ax.text(0.9, 1.02, f"{arm}{spec}", transform=ax.transAxes, ha="right")
                ax.label_outer()

            if C is not None and colorbarLabel is not None:
                shrink = {1: 0.99, 2: 0.99}.get(ny, 0.93) if nx <= 4 else 0.85
                with opaqueColorbar(C):
                    fig.colorbar(C, shrink=shrink, label=colorbarLabel, ax=axs)

            title = f"visit{'s' if len(visits) > 1 else ''} {' '.join(str(v) for v in visits)}"
            fig.suptitle(title, y={1: 1.0, 2: 0.8, 3: 1.0}.get(ny, 1))
            pdf.append(fig)
            plt.close(fig)

        return pdf
