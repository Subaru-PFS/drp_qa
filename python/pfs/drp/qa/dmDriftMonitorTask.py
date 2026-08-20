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
        """Compute per-species drift metrics between daily arc centroids and detectormap predictions.

        One row is written per arc species (``description`` value) present in
        ``arcLines``.  Trace lines are treated as a species named ``"Trace"``
        and contribute ``deltaX`` and ``deltaWx``; emission-line species (e.g.
        ``"NeI"``, ``"OI"``, ``"OH"``) contribute ``deltaY``.  This ensures
        that, for example, neon and sky OH lines are reported separately for
        science frames, and HgI / CdI are reported separately for HgCd arcs.

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
        Struct with dmDriftMetrics DataFrame (one row per species).
        """
        self.log.info("Computing per-species drift metrics for %s", dataId)

        _level = {"PASS": 0, "WARN": 1, "FAIL": 2}
        goodMask = arcLines.flag == 0
        hasXX = hasattr(arcLines, "xx")
        descriptions = arcLines.description if hasattr(arcLines, "description") else np.array([])
        allSpecies = np.unique(descriptions) if len(descriptions) > 0 else np.array([])

        rows = []
        for species in allSpecies:
            specMask = goodMask & (descriptions == species)
            isTraceSp = species == "Trace"

            # --- deltaX and deltaWx: from trace lines ---
            deltaX = np.nan
            deltaWx = np.nan
            if isTraceSp:
                traceGood = specMask & np.isfinite(arcLines.x) & np.isfinite(arcLines.y)
                if traceGood.sum() >= self.config.minLines:
                    xPredicted = detectorMap.getXCenter(
                        arcLines.fiberId[traceGood].astype(np.int32),
                        arcLines.y[traceGood].astype(np.float64),
                    )
                    deltaX = float(np.mean(arcLines.x[traceGood] - xPredicted))
                else:
                    self.log.warning(
                        "Too few Trace lines (%d < minLines=%d) for deltaX in %s",
                        int(traceGood.sum()), self.config.minLines, dataId,
                    )
                if hasXX:
                    xxGood = specMask & np.isfinite(arcLines.xx) & (arcLines.xx > 0)
                    if xxGood.sum() >= self.config.minLines:
                        wxMeasured = _FWHM_FACTOR * np.sqrt(arcLines.xx[xxGood])
                        wxRef = float(np.median(wxMeasured))
                        deltaWx = float(np.mean(wxMeasured - wxRef))

            # --- deltaY: from emission-line species ---
            deltaY = np.nan
            if not isTraceSp:
                arcGood = (
                    specMask
                    & np.isfinite(arcLines.x)
                    & np.isfinite(arcLines.y)
                    & np.isfinite(arcLines.wavelength)
                )
                if arcGood.sum() >= self.config.minLines:
                    predicted = detectorMap.findPoint(
                        arcLines.fiberId[arcGood].astype(np.int32),
                        arcLines.wavelength[arcGood].astype(np.float64),
                    )
                    yPredicted = predicted[:, 1]
                    deltaY = float(np.mean(arcLines.y[arcGood] - yPredicted))
                else:
                    self.log.warning(
                        "Too few %s lines (%d < minLines=%d) for deltaY in %s",
                        species, int(arcGood.sum()), self.config.minLines, dataId,
                    )

            driftMag = (
                float(np.sqrt(deltaX**2 + deltaY**2))
                if (np.isfinite(deltaX) and np.isfinite(deltaY))
                else np.nan
            )

            # --- qaStatus and recommendedAction ---
            if np.isnan(driftMag) and np.isnan(deltaWx):
                qaStatus = "UNKNOWN"
                recommendedAction = "INSUFFICIENT_DATA"
            else:
                status = "PASS"
                if np.isfinite(driftMag):
                    if driftMag >= self.config.driftFailThreshold:
                        status = "FAIL"
                    elif driftMag >= self.config.driftWarnThreshold:
                        status = max(status, "WARN", key=lambda s: _level[s])
                if np.isfinite(deltaWx):
                    if abs(deltaWx) >= self.config.profileFailThreshold:
                        status = "FAIL"
                    elif abs(deltaWx) >= self.config.profileWarnThreshold:
                        status = max(status, "WARN", key=lambda s: _level[s])
                qaStatus = status
                recommendedAction = (
                    "RECALIBRATE" if status == "FAIL" else "APPLY_SHIFT" if status == "WARN" else "NOMINAL"
                )

            self.log.info(
                "Drift monitor %s species=%s: deltaX=%.4f deltaY=%.4f driftMag=%.4f deltaWx=%.4f -> %s (%s)",
                dataId, species,
                deltaX if np.isfinite(deltaX) else float("nan"),
                deltaY if np.isfinite(deltaY) else float("nan"),
                driftMag if np.isfinite(driftMag) else float("nan"),
                deltaWx if np.isfinite(deltaWx) else float("nan"),
                qaStatus, recommendedAction,
            )

            rows.append({
                "visit": dataId.get("visit"),
                "arm": dataId.get("arm"),
                "spectrograph": dataId.get("spectrograph"),
                "description": species,
                "deltaX": deltaX,
                "deltaY": deltaY,
                "driftMag": driftMag,
                "deltaWx": deltaWx,
                "qaStatus": qaStatus,
                "recommendedAction": recommendedAction,
            })

        if not rows:
            raise ValueError(f"No arc line species found in arcLines for {dataId}; cannot compute drift metrics.")

        return Struct(dmDriftMetrics=pd.DataFrame(rows))
