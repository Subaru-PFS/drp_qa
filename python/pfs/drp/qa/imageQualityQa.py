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
    PrerequisiteInput as PrerequisiteConnection,
)
from matplotlib import pyplot as plt
from pfs.drp.stella import ArcLineSet, DetectorMap, FiberProfileSet
from pfs.drp.stella.utils.quality import computeImageQuality, opaqueColorbar, plotImageQuality
from pfs.drp.stella.utils.stability import addTraceLambdaToArclines

from pfs.drp.qa.storageClasses import MultipagePdfFigure

_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # sigma → Gaussian-equivalent FWHM

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

    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc=(
            "Fiber profile shapes; used to derive trace-width FWHM when arc-line"
            " shape measurements are unavailable."
        ),
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
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
    maxFwhm = Field(
        dtype=float,
        default=8.0,
        doc="Upper FWHM cutoff for line selection and histogram binning (pixels).",
    )
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
        fiberProfiles: FiberProfileSet | None,
        dataId: dict,
    ) -> Struct:
        """Compute image quality metrics and generate QA plots.

        For arc visits, FWHM is derived from 2D Gaussian second moments of the
        arc lines.  For non-arc visits (e.g. quartz or flat), arc-line shape
        measurements are unavailable; the task then falls back to computing
        Gaussian-equivalent FWHM from the fiber profile trace widths stored in
        ``fiberProfiles``.

        Parameters
        ----------
        arcLines : `ArcLineSet`
            Arc line measurements.
        detectorMap : `DetectorMap`
            Calibrated detector mapping from fiberId,wavelength to x,y.
        fiberProfiles : `FiberProfileSet` or `None`
            Fiber profile shapes.  If ``None`` and arc-line shape measurements
            are unavailable the task will still succeed but produce all-NaN
            FWHM (same behaviour as before this fallback was added).
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

        if not data["fwhm"].notna().any():
            if fiberProfiles is not None:
                self.log.info(
                    "No finite arc-line FWHM for %s; falling back to fiber profile trace widths.", dataId
                )
                data = self._buildProfileData(fiberProfiles, detectorMap)
            else:
                self.log.warning(
                    "No finite arc-line FWHM for %s and no fiberProfiles available; FWHM will be all NaN.",
                    dataId,
                )

        for key in ("visit", "arm", "spectrograph"):
            if key in dataId:
                data[key] = dataId[key]

        title = "{visit} {arm}{spectrograph}".format(**dataId)
        self.log.info("Generating image quality plots for %s", dataId)
        pdf = self._makePlots(data, title=title)

        return Struct(iqQaData=data, iqQaPlot=pdf)

    def _buildProfileData(self, fiberProfiles: FiberProfileSet, detectorMap: DetectorMap) -> pd.DataFrame:
        """Build an image-quality DataFrame from fiber profile trace widths.

        Computes Gaussian-equivalent FWHM (``2√(2 ln 2) × σ``) per swath from
        `FiberProfile.calculateStatistics` and maps each swath centre to
        detector (x, y) coordinates and wavelength via ``detectorMap``.

        Parameters
        ----------
        fiberProfiles : `FiberProfileSet`
            Fiber profile shapes from the Butler calibration.
        detectorMap : `DetectorMap`
            Calibrated detector mapping.

        Returns
        -------
        `pandas.DataFrame`
            Columns: ``fiberId``, ``x``, ``y``, ``lam``, ``fwhm``, ``theta``,
            ``flux``, ``fluxErr``, ``flag``, ``traceOnly``.
        """
        rows_list, fibers_list, fwhm_list = [], [], []
        x_list, lam_list, flux_list = [], [], []

        for fiberId in fiberProfiles.fiberId:
            profile = fiberProfiles[fiberId]
            stats = profile.calculateStatistics()
            swath_rows = profile.rows
            n = len(swath_rows)
            if n == 0:
                continue

            fwhm = _FWHM_FACTOR * np.asarray(stats.width)

            fiberIds_arr = np.full(n, fiberId, dtype=np.int32)
            swath_rows_f64 = np.asarray(swath_rows, dtype=np.float64)
            x = detectorMap.getXCenter(fiberIds_arr, swath_rows_f64)
            lam = detectorMap.findWavelength(fiberIds_arr, swath_rows_f64)

            if profile.norm is not None and len(profile.norm) > 0:
                idx = np.clip(np.round(swath_rows_f64).astype(int), 0, len(profile.norm) - 1)
                flux = np.asarray(profile.norm[idx], dtype=float)
            else:
                flux = np.ma.sum(profile.profiles, axis=1).filled(np.nan)

            rows_list.append(swath_rows_f64)
            fibers_list.append(fiberIds_arr)
            fwhm_list.append(fwhm)
            x_list.append(x)
            lam_list.append(lam)
            flux_list.append(flux)

        if not rows_list:
            return pd.DataFrame(
                columns=["fiberId", "x", "y", "lam", "fwhm", "theta", "flux", "fluxErr", "flag", "traceOnly"]
            )

        n_total = sum(len(r) for r in rows_list)
        return pd.DataFrame(
            {
                "fiberId": np.concatenate(fibers_list),
                "y": np.concatenate(rows_list),
                "x": np.concatenate(x_list),
                "lam": np.concatenate(lam_list),
                "fwhm": np.concatenate(fwhm_list),
                "theta": np.zeros(n_total),
                "flux": np.concatenate(flux_list),
                "fluxErr": np.ones(n_total),
                "flag": np.zeros(n_total, dtype=bool),
                "traceOnly": True,
            }
        )

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

            fig, ax = plt.subplots(layout="constrained")
            C, colorbarLabel = plotImageQuality(ax, data, **{plotMode: True}, **plotKwargs)
            if C is not None:
                with opaqueColorbar(C):
                    fig.colorbar(C, ax=ax, label=colorbarLabel)
            if title:
                fig.suptitle(title)
            pdf.append(fig)
            plt.close(fig)

        return pdf
