"""Detector-map drift monitoring against ``detectorMap_calib``.

Per detector, has the instrument moved away from the calibration the visit is
reduced against? The measurement and the verdict are stack-free, in
`pfs.drp.qa.metrics.drift`; this module binds them to the Butler.
"""

from contextlib import suppress
from typing import Any

import numpy as np
import pandas as pd
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
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from pfs.datamodel import PfsConfig
from pfs.drp.qa.imageQualityQa import litFibers
from pfs.drp.qa.metrics.drift import (
    BOTH,
    cameraSet,
    driftOffsets,
    driftThresholds,
    driftVerdict,
    longDrift,
    speciesDrift,
)
from pfs.drp.qa.metrics.gate import loadThresholds, thresholdsPath
from pfs.drp.stella import ArcLineSet, DetectorMap, ReferenceLineStatus

__all__ = ["DmDriftMonitorConfig", "DmDriftMonitorTask"]


class DmDriftMonitorConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Connections for DmDriftMonitorTask."""

    arcLines = InputConnection(
        name="lines",
        doc="Line and trace centroids measured by fitDetectorMap or reduceExposure.",
        storageClass="ArcLineSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    detectorMapCalib = PrerequisiteConnection(
        name="detectorMap_calib",
        doc=(
            "The calibration detectorMap the visit is reduced against: the reference. Never the"
            " visit's own detectorMap, which was fitted to these lines."
        ),
        storageClass="DetectorMap",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )

    # A prerequisite with minimum=0, as in imageQualityQa: a dataset type must be
    # a prerequisite to every task of a graph or to none.
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Fiber configuration: the lit fibers (linesPerFiber) and the arms read (cameraSet).",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
        minimum=0,
    )

    dmDriftMetrics = OutputConnection(
        name="dmDriftMetrics",
        doc=(
            "Drift against detectorMap_calib in long format, one row per line species and metric:"
            " ``visit``, ``arm``, ``spectrograph``, ``cameraSet``, ``description``, ``statusType``"
            " (``BOTH``, judged, over the lines the visit's detectorMap fit marked USED or RESERVED;"
            " ``USED`` and ``RESERVED`` alone, diagnostics), ``metric``"
            " (``nLines``, ``linesPerFiber``; ``medianDeltaX``, ``robustRmsDeltaX`` for trace"
            " centroids; ``medianDeltaY``, ``robustRmsDeltaY`` for lamp lines), ``value``,"
            " ``status`` (the metric's verdict, empty when not judged)."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    dmDriftSummary = OutputConnection(
        name="dmDriftSummary",
        doc=(
            "One row per detector: ``visit``, ``arm``, ``spectrograph``, ``cameraSet``,"
            " ``nSpecies``, ``nSpeciesMeasured``, ``qaStatus`` (``UNKNOWN`` when no species"
            " has ``minLines`` good lines), ``qaDecidedBy``, ``qaReason``, ``recommendedAction``"
            " (``NOMINAL``, ``APPLY_SHIFT``, ``RECALIBRATE``, ``INSUFFICIENT_DATA``), and the"
            " calibration: ``calibId`` and ``calibRun`` (look up its validity range with"
            " `pfs.drp.qa.metrics.readers.readCalibValidity`), ``calibCreated``,"
            " ``calibFirstVisit`` and ``calibLastVisit`` (the visits it was made from)."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class DmDriftMonitorConfig(PipelineTaskConfig, pipelineConnections=DmDriftMonitorConnections):
    """Configuration for DmDriftMonitorTask."""

    minLines = Field(
        dtype=int,
        default=10,
        doc=(
            "Fewest good lines a species needs for its drift to be measured. An absolute count, not"
            " per lit fiber: it bounds the error of a median, which a design lighting one fiber"
            " group does not change."
        ),
    )
    lineFlags = ListField(
        dtype=str,
        default=["BAD"],
        doc="ReferenceLineStatus names that exclude a line, as fitDetectorMap's lineFlags.",
    )
    thresholdsFile = Field(
        dtype=str,
        default="",
        doc=(
            "Thresholds file (`pfs.drp.qa.metrics.calibration.writeThresholds`), judged before the"
            " threshold fields: an absolute path, or one relative to ``pfs/drp/qa/metrics/data``;"
            " empty for the fields alone. Entries may name ``arm``, ``spectrograph``,"
            " ``description`` and ``cameraSet``."
        ),
    )
    xWarnThreshold = Field(
        dtype=float,
        default=1.0,
        doc=(
            "|medianDeltaX| (pixels) of a trace species at or above which it is WARN. Provisional:"
            " imageQualityQa's dxCenterWarnThreshold, origin unrecorded. NaN to disable."
        ),
    )
    xFailThreshold = Field(
        dtype=float,
        default=2.0,
        doc=(
            "|medianDeltaX| (pixels) of a trace species at or above which it is FAIL. Provisional:"
            " imageQualityQa's dxCenterFailThreshold, origin unrecorded. NaN to disable."
        ),
    )
    yWarnThreshold = Field(
        dtype=float,
        default=np.nan,
        doc="|medianDeltaY| (pixels) of a lamp species at or above which it is WARN. NaN: not judged.",
    )
    yFailThreshold = Field(
        dtype=float,
        default=np.nan,
        doc="|medianDeltaY| (pixels) of a lamp species at or above which it is FAIL. NaN: not judged.",
    )


class DmDriftMonitorTask(PipelineTask):
    """Measure each line species' offset from ``detectorMap_calib``, and judge it."""

    ConfigClass = DmDriftMonitorConfig
    _DefaultName = "dmDriftMonitor"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Highest priority first; see pfs.drp.qa.metrics.gate.
        files = [thresholdsPath(self.config.thresholdsFile)] if self.config.thresholdsFile else []
        fields = driftThresholds(
            self.config.xWarnThreshold,
            self.config.xFailThreshold,
            self.config.yWarnThreshold,
            self.config.yFailThreshold,
        )
        self.thresholds = [*loadThresholds(files), fields]

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
        except ValueError as e:
            self.log.error("DmDriftMonitorTask failed for %s: %s", dataId, e)
        else:
            # The validity range is in the registry, which a quantum cannot read: record which
            # calibration was used, so a reader can look it up.
            calibRef = inputRefs.detectorMapCalib
            outputs.dmDriftSummary["calibId"] = str(calibRef.id)
            outputs.dmDriftSummary["calibRun"] = calibRef.run
            butlerQC.put(outputs.dmDriftMetrics, outputRefs.dmDriftMetrics)
            butlerQC.put(outputs.dmDriftSummary, outputRefs.dmDriftSummary)

    def run(
        self,
        arcLines: ArcLineSet,
        detectorMapCalib: DetectorMap,
        dataId: dict,
        pfsConfig: PfsConfig | None = None,
    ) -> Struct:
        """Measure and judge the drift of one detector.

        Parameters
        ----------
        arcLines : `ArcLineSet`
            Line and trace centroids.
        detectorMapCalib : `DetectorMap`
            The calibration detectorMap: the reference.
        dataId : `dict`
            ``visit``, ``arm``, ``spectrograph``.
        pfsConfig : `PfsConfig`, optional
            Fiber configuration of the visit.

        Returns
        -------
        dmDriftMetrics : `pandas.DataFrame`
            `pfs.drp.qa.metrics.drift.longDrift`.
        dmDriftSummary : `pandas.DataFrame`
            One row; see the connection.
        """
        lines = arcLines.data if hasattr(arcLines, "data") else arcLines
        badStatus = int(ReferenceLineStatus.fromNames(*self.config.lineFlags)) if self.config.lineFlags else 0
        offsets = driftOffsets(lines, detectorMapCalib, badStatus=badStatus)
        lit, _ = litFibers(pfsConfig, np.asarray(detectorMapCalib.fiberId))
        species = speciesDrift(offsets, self.config.minLines, lit)
        camera = cameraSet(pfsConfig, dataId.get("arm"))
        numUnfitted = int((offsets["statusType"] == "").sum())
        judged, verdict = driftVerdict(
            species, dataId, self.thresholds, self.config.minLines, camera, numUnfitted=numUnfitted
        )

        summary = {key: dataId.get(key) for key in ("visit", "arm", "spectrograph")}
        summary |= {"cameraSet": camera} | verdict | _calibProvenance(detectorMapCalib)
        self.log.info(
            "DM drift %-7s %s%s %s: %s [%s]",
            verdict["qaStatus"],
            dataId.get("arm"),
            dataId.get("spectrograph"),
            verdict["recommendedAction"],
            ", ".join(
                f"{row.description} dx={row.medianDeltaX:+.3f}"
                if row.description == "Trace"
                else f"{row.description} dy={row.medianDeltaY:+.3f}"
                for row in species[species["statusType"] == BOTH].itertuples()
            )
            or "no lines",
            verdict["qaReason"] or "nominal",
        )
        return Struct(
            dmDriftMetrics=longDrift(species, judged, dataId, camera),
            dmDriftSummary=pd.DataFrame([summary]),
        )


def _calibProvenance(detectorMapCalib: Any) -> dict[str, Any]:
    """Return when the calibration was made, and from which visits, from its header."""
    metadata = getattr(detectorMapCalib, "metadata", None)
    created, visits = None, []
    if metadata is not None:
        names = metadata.names() if hasattr(metadata, "names") else list(metadata)
        date = metadata.get("CALIB_CREATION_DATE") if "CALIB_CREATION_DATE" in names else None
        created = str(date) if date is not None else None
        for name in names:
            if str(name).startswith("CALIB_INPUT_"):
                with suppress(TypeError, ValueError):  # skip one that is not a visit id
                    visits.append(int(metadata.get(name)))
    return {
        "calibCreated": created,
        "calibFirstVisit": min(visits) if visits else None,
        "calibLastVisit": max(visits) if visits else None,
    }
