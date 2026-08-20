"""Detector map drift monitor QA task.

Compares daily trace and neon arc centroids against the predictions of the
static detectormap (fitted once per observation run) to measure centroid drift
and flag when recalibration is needed.
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
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection

from pfs.drp.stella import ArcLineSet, DetectorMap

__all__ = ["DmDriftMonitorTask"]

_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # sigma -> Gaussian-equivalent FWHM


class DmDriftMonitorConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Connections for DmDriftMonitorTask."""

    arcLines = InputConnection(
        name="lines",
        doc="Daily arc/trace emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    detectorMap = InputConnection(
        name="detectorMap",
        doc="Static detectormap fitted once per observation run",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    dmDriftMetrics = OutputConnection(
        name="dmDriftMetrics",
        doc="Per-detector drift metrics comparing daily centroids to detectormap predictions.",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class DmDriftMonitorConfig(PipelineTaskConfig, pipelineConnections=DmDriftMonitorConnections):
    """Configuration for DmDriftMonitorTask."""

    driftWarnThreshold = Field(
        dtype=float,
        default=0.05,
        doc="Drift magnitude (px) above which qaStatus is WARN.",
    )
    driftFailThreshold = Field(
        dtype=float,
        default=0.15,
        doc="Drift magnitude (px) above which qaStatus is FAIL.",
    )
    profileWarnThreshold = Field(
        dtype=float,
        default=0.05,
        doc="Profile width change deltaWx (px) above which qaStatus is WARN.",
    )
    profileFailThreshold = Field(
        dtype=float,
        default=0.10,
        doc="Profile width change deltaWx (px) above which qaStatus is FAIL.",
    )
    minLines = Field(
        dtype=int,
        default=20,
        doc=(
            "Minimum number of non-flagged lines required to compute drift metrics."
            " When fewer lines are available, NaN metrics are written with qaStatus=UNKNOWN."
        ),
    )


class DmDriftMonitorTask(PipelineTask):
    """QA task measuring daily centroid drift against the static detectormap.

    Reads daily arc/trace line measurements and the static detectormap (one per
    observation run), computes residuals between measured centroids and
    detectormap predictions, and writes dmDriftMetrics DataFrame.
    """

    ConfigClass = DmDriftMonitorConfig
    _DefaultName = "dmDriftMonitor"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        dataId = dict(**inputRefs.arcLines.dataId.mapping)
        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = dataId
        try:
            outputs = self.run(**inputs)
        except ValueError as e:
            self.log.error("DmDriftMonitorTask failed for %s: %s", dataId, e)
        else:
            butlerQC.put(outputs, outputRefs)

    def run(self, arcLines: ArcLineSet, detectorMap: DetectorMap, dataId: dict) -> Struct:
        """Compute drift metrics between daily arc centroids and detectormap predictions.

        Parameters
        ----------
        arcLines : ArcLineSet
            Daily arc/trace line measurements.
        detectorMap : DetectorMap
            Static detectormap fitted once per observation run.
        dataId : dict
            Data identifier (visit, arm, spectrograph, instrument).

        Returns
        -------
        Struct with dmDriftMetrics DataFrame.
        """
        self.log.info("Computing drift metrics for %s", dataId)

        isTrace = arcLines.description == "Trace"
        isLine = ~isTrace
        goodMask = arcLines.flag == 0

        # --- deltaX: mean spatial shift from trace lines ---
        traceGood = goodMask & isTrace & np.isfinite(arcLines.x) & np.isfinite(arcLines.y)
        deltaX = np.nan
        if traceGood.sum() >= self.config.minLines:
            xPredicted = detectorMap.getXCenter(
                arcLines.fiberId[traceGood].astype(np.int32),
                arcLines.y[traceGood].astype(np.float64),
            )
            deltaX = float(np.mean(arcLines.x[traceGood] - xPredicted))
        else:
            self.log.warning(
                "Too few trace lines (%d < minLines=%d) for deltaX in %s",
                int(traceGood.sum()), self.config.minLines, dataId,
            )

        # --- deltaY: mean spectral shift from arc lines ---
        arcGood = (
            goodMask & isLine & np.isfinite(arcLines.x) & np.isfinite(arcLines.y)
            & np.isfinite(arcLines.wavelength)
        )
        deltaY = np.nan
        if arcGood.sum() >= self.config.minLines:
            predicted = detectorMap.findPoint(
                arcLines.fiberId[arcGood].astype(np.int32),
                arcLines.wavelength[arcGood].astype(np.float64),
            )
            yPredicted = predicted[:, 1]
            deltaY = float(np.mean(arcLines.y[arcGood] - yPredicted))
        else:
            self.log.warning(
                "Too few arc lines (%d < minLines=%d) for deltaY in %s",
                int(arcGood.sum()), self.config.minLines, dataId,
            )

        driftMag = (
            float(np.sqrt(deltaX**2 + deltaY**2))
            if (np.isfinite(deltaX) and np.isfinite(deltaY))
            else np.nan
        )

        # --- deltaWx: profile width change from xx second moment ---
        deltaWx = np.nan
        if hasattr(arcLines, "xx"):
            xxGood = goodMask & np.isfinite(arcLines.xx) & (arcLines.xx > 0)
            if xxGood.sum() >= self.config.minLines:
                wxMeasured = _FWHM_FACTOR * np.sqrt(arcLines.xx[xxGood])
                # detectormap does not store profile widths directly; use median measured width
                # as the reference (drift is relative to the run-level median)
                wxRef = float(np.median(wxMeasured))
                deltaWx = float(np.mean(wxMeasured - wxRef))

        # --- qaStatus and recommendedAction ---
        if np.isnan(driftMag) and np.isnan(deltaWx):
            qaStatus = "UNKNOWN"
            recommendedAction = "INSUFFICIENT_DATA"
        else:
            _level = {"PASS": 0, "WARN": 1, "FAIL": 2}
            status = "PASS"
            if np.isfinite(driftMag):
                if driftMag >= self.config.driftFailThreshold:
                    status = "FAIL"
                elif driftMag >= self.config.driftWarnThreshold:
                    status = max(status, "WARN", key=lambda s: _level[s])
            if np.isfinite(deltaWx):
                absDeltaWx = abs(deltaWx)
                if absDeltaWx >= self.config.profileFailThreshold:
                    status = "FAIL"
                elif absDeltaWx >= self.config.profileWarnThreshold:
                    status = max(status, "WARN", key=lambda s: _level[s])
            qaStatus = status
            if status == "FAIL":
                recommendedAction = "RECALIBRATE"
            elif status == "WARN":
                recommendedAction = "APPLY_SHIFT"
            else:
                recommendedAction = "NOMINAL"

        self.log.info(
            "Drift monitor %s: deltaX=%.4f deltaY=%.4f driftMag=%.4f deltaWx=%.4f -> %s (%s)",
            dataId, deltaX, deltaY, driftMag if np.isfinite(driftMag) else float("nan"),
            deltaWx if np.isfinite(deltaWx) else float("nan"),
            qaStatus, recommendedAction,
        )

        metrics = pd.DataFrame({
            "visit": [dataId.get("visit")],
            "arm": [dataId.get("arm")],
            "spectrograph": [dataId.get("spectrograph")],
            "deltaX": [deltaX],
            "deltaY": [deltaY],
            "driftMag": [driftMag],
            "deltaWx": [deltaWx],
            "qaStatus": [qaStatus],
            "recommendedAction": [recommendedAction],
        })

        return Struct(dmDriftMetrics=metrics)
