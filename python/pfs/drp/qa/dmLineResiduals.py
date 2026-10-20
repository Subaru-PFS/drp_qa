"""Line residuals against ``detectorMap_calib`` and the visit's ``detectorMap``.

Per detector: has the instrument moved away from the calibration the visit is
reduced against, and how well does the visit's adjusted detectorMap predict
the lines it was not fitted to? The measurement and the verdict are
stack-free, in `pfs.drp.qa.metrics.lineResiduals`; this module binds them to
the Butler. It replaces ``dmResiduals`` and ``dmCombinedResiduals``
(PIPE2D-1956).
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
from pfs.drp.qa.metrics.gate import loadThresholds, thresholdsPath
from pfs.drp.qa.metrics.lineResiduals import (
    BOTH,
    CALIB,
    RESERVED,
    TRACE,
    UNFITTED,
    VISIT,
    cameraSet,
    lineResiduals,
    lineTable,
    longResiduals,
    residualThresholds,
    residualVerdict,
    speciesResiduals,
)
from pfs.drp.stella import ArcLineSet, DetectorMap, ReferenceLineStatus

__all__ = ["DmLineResidualsConfig", "DmLineResidualsTask"]


class DmLineResidualsConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Connections for DmLineResidualsTask."""

    arcLines = InputConnection(
        name="lines",
        doc="Line and trace centroids measured by fitDetectorMap or reduceExposure.",
        storageClass="ArcLineSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    detectorMap = InputConnection(
        name="detectorMap",
        doc="The visit's detectorMap, adjusted to these lines: the ``visit`` reference.",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    detectorMapCalib = PrerequisiteConnection(
        name="detectorMap_calib",
        doc="The calibration detectorMap the visit is reduced against: the ``calib`` reference.",
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

    dmLineResidualMetrics = OutputConnection(
        name="dmLineResidualMetrics",
        doc=(
            "Line residuals in long format, one row per reference, line species, line set and metric:"
            " ``visit``, ``arm``, ``spectrograph``, ``cameraSet``, ``reference`` (``calib``:"
            " detectorMap_calib; ``visit``: the visit's adjusted detectorMap), ``description``,"
            " ``statusType`` (``BOTH``, ``USED``, ``RESERVED`` lines of the visit's detectorMap fit, or"
            " ``UNFITTED``, every good line, when the fit marked none), ``metric`` (``nLines``,"
            " ``linesPerFiber``; ``medianDeltaX``, ``robustRmsDeltaX`` for every species;"
            " ``medianDeltaY``, ``robustRmsDeltaY`` for lamp lines), ``value``, ``status`` (the"
            " metric's verdict; empty when not judged: only calib BOTH or UNFITTED rows and visit"
            " RESERVED robust RMS are)."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    dmLineResidualSummary = OutputConnection(
        name="dmLineResidualSummary",
        doc=(
            "One row per detector: ``visit``, ``arm``, ``spectrograph``, ``cameraSet``, ``fitted``"
            " (false when the visit's detectorMap fit marked no line, so it is reduced against the"
            " calibration unadjusted),"
            " ``nSpecies``, ``nSpeciesMeasured``, ``qaStatus`` (``UNKNOWN`` when no species"
            " has ``minLines`` good lines), ``qaDecidedBy``, ``qaReason``, ``recommendedAction``"
            " (``NOMINAL``, ``APPLY_SHIFT``, ``RECALIBRATE``, ``INSUFFICIENT_DATA``, ``UNJUDGED``), and"
            " the calibration: ``calibId`` and ``calibRun`` (look up its validity range with"
            " `pfs.drp.qa.metrics.readers.readCalibValidity`), ``calibCreated``,"
            " ``calibFirstVisit`` and ``calibLastVisit`` (the visits it was made from)."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    dmLineResidualData = OutputConnection(
        name="dmLineResidualData",
        doc=(
            "One row per line measured (USED and RESERVED trace rows sampled: maxTraceRowsPerFiber), for plots drawn"
            " later (pfs.drp.qa.plotting): ``visit``, ``arm``,"
            " ``spectrograph``, ``fiberId``, ``wavelength``, ``x``, ``y``, ``xErr``, ``yErr``,"
            " ``description``, ``statusType``, and the residual against each reference, measured minus"
            " reference: ``dxCalib``, ``dyCalib``, ``dxVisit``, ``dyVisit`` (``dy`` NaN for traces)."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class DmLineResidualsConfig(PipelineTaskConfig, pipelineConnections=DmLineResidualsConnections):
    """Configuration for DmLineResidualsTask."""

    minLines = Field(
        dtype=int,
        default=10,
        doc=(
            "Fewest good lines a species needs for its statistics to be measured. An absolute count, not"
            " per lit fiber: it bounds the error of a median, which a design lighting one fiber"
            " group does not change."
        ),
    )
    lineFlags = ListField(
        dtype=str,
        default=["BAD"],
        doc="ReferenceLineStatus names that exclude a line, as fitDetectorMap's lineFlags.",
    )
    maxTraceRowsPerFiber = Field(
        dtype=int,
        default=100,
        doc=(
            "Most USED, and most RESERVED, trace rows per fiber stored in dmLineResidualData, evenly spaced"
            " in y; 0 for all. Lamp and sky lines and UNFITTED rows are always stored, and the statistics use"
            " every line: a fit can keep millions of trace rows (twilight b3/b4 kept about 2 million), and"
            " RESERVED trace rows were ~80% of a quartz or twilight visit's table before they were sampled."
        ),
    )
    thresholdsFile = Field(
        dtype=str,
        default="",
        doc=(
            "Thresholds file (`pfs.drp.qa.metrics.calibration.writeThresholds`), judged before the"
            " threshold fields: an absolute path, or one relative to ``pfs/drp/qa/metrics/data``;"
            " empty for the fields alone. Entries may name ``reference``, ``statusType``, ``arm``,"
            " ``spectrograph``, ``description`` and ``cameraSet``; an entry that leaves ``reference``"
            " null applies to both references."
        ),
    )
    xWarnThreshold = Field(
        dtype=float,
        default=1.0,
        doc=(
            "|medianDeltaX| (pixels) of trace centroids against detectorMap_calib at or above which it is"
            " WARN (APPLY_SHIFT): the calibration no longer matches, though the visit's adjustment absorbs a"
            " shift. A third of the 3 px capture radius (FAIL); above Run25's known-good offsets, which"
            " reach 0.57 px (b read with m) and 0.33 px otherwise (PIPE2D-1921, u/wtg/qa-thresholds/006)."
            " Lamp lines' x offset is not judged by it. NaN to disable."
        ),
    )
    xFailThreshold = Field(
        dtype=float,
        default=2.0,
        doc=(
            "|medianDeltaX| (pixels) at or above which it is FAIL (RECALIBRATE). A physical limit: drp_stella"
            " finds lines within 3 px of the calibration's prediction (centroidLines.peakSearch,"
            " centroidTraces.searchRadius); 2 px leaves room for the scatter (known-good robust RMS up to"
            " 0.22 px) and the line width. NaN to disable."
        ),
    )
    yWarnThreshold = Field(
        dtype=float,
        default=1.0,
        doc=(
            "|medianDeltaY| (pixels) of a lamp species against detectorMap_calib at or above which it is"
            " WARN (APPLY_SHIFT); as xWarnThreshold. Run25 after its calibration block sits up to 1.5 px"
            " off in y on SM3, so those visits WARN; the validation set records them as unconfirmed"
            " (PIPE2D-1921). NaN to disable."
        ),
    )
    yFailThreshold = Field(
        dtype=float,
        default=2.0,
        doc="|medianDeltaY| (pixels) at or above which it is FAIL (RECALIBRATE); as xFailThreshold.",
    )
    xRmsWarnThreshold = Field(
        dtype=float,
        default=float("nan"),
        doc=(
            "robustRmsDeltaX (pixels) of a species' RESERVED lines against the visit's detectorMap at or"
            " above which it is WARN (RECALIBRATE): scatter the adjustment leaves on lines it was not"
            " fitted to. NaN, not judged, until derived from the validation set (PIPE2D-1956)."
        ),
    )
    xRmsFailThreshold = Field(
        dtype=float,
        default=float("nan"),
        doc="robustRmsDeltaX (pixels) at or above which it is FAIL; as xRmsWarnThreshold.",
    )
    yRmsWarnThreshold = Field(
        dtype=float,
        default=float("nan"),
        doc="robustRmsDeltaY (pixels) of lamp lines at or above which it is WARN; as xRmsWarnThreshold.",
    )
    yRmsFailThreshold = Field(
        dtype=float,
        default=float("nan"),
        doc="robustRmsDeltaY (pixels) at or above which it is FAIL; as xRmsWarnThreshold.",
    )


class DmLineResidualsTask(PipelineTask):
    """Measure each line species' residuals from the calibration and the visit's detectorMap, and judge them."""

    ConfigClass = DmLineResidualsConfig
    _DefaultName = "dmLineResiduals"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Highest priority first; see pfs.drp.qa.metrics.gate.
        files = [thresholdsPath(self.config.thresholdsFile)] if self.config.thresholdsFile else []
        fields = residualThresholds(
            self.config.xWarnThreshold,
            self.config.xFailThreshold,
            self.config.yWarnThreshold,
            self.config.yFailThreshold,
            self.config.xRmsWarnThreshold,
            self.config.xRmsFailThreshold,
            self.config.yRmsWarnThreshold,
            self.config.yRmsFailThreshold,
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
            self.log.error("DmLineResidualsTask failed for %s: %s", dataId, e)
        else:
            # The validity range is in the registry, which a quantum cannot read: record which
            # calibration was used, so a reader can look it up.
            calibRef = inputRefs.detectorMapCalib
            outputs.dmLineResidualSummary["calibId"] = str(calibRef.id)
            outputs.dmLineResidualSummary["calibRun"] = calibRef.run
            butlerQC.put(outputs, outputRefs)

    def run(
        self,
        arcLines: ArcLineSet,
        detectorMap: DetectorMap,
        detectorMapCalib: DetectorMap,
        dataId: dict,
        pfsConfig: PfsConfig | None = None,
    ) -> Struct:
        """Measure and judge the line residuals of one detector.

        Parameters
        ----------
        arcLines : `ArcLineSet`
            Line and trace centroids.
        detectorMap : `DetectorMap`
            The visit's adjusted detectorMap: the ``visit`` reference.
        detectorMapCalib : `DetectorMap`
            The calibration detectorMap: the ``calib`` reference.
        dataId : `dict`
            ``visit``, ``arm``, ``spectrograph``.
        pfsConfig : `PfsConfig`, optional
            Fiber configuration of the visit.

        Returns
        -------
        dmLineResidualMetrics : `pandas.DataFrame`
            `pfs.drp.qa.metrics.lineResiduals.longResiduals`.
        dmLineResidualSummary : `pandas.DataFrame`
            One row; see the connection.
        dmLineResidualData : `pandas.DataFrame`
            `pfs.drp.qa.metrics.lineResiduals.lineTable`.
        """
        lines = arcLines.data if hasattr(arcLines, "data") else arcLines
        badStatus = int(ReferenceLineStatus.fromNames(*self.config.lineFlags)) if self.config.lineFlags else 0
        residuals = lineResiduals(lines, {CALIB: detectorMapCalib, VISIT: detectorMap}, badStatus=badStatus)
        lit, _ = litFibers(pfsConfig, np.asarray(detectorMapCalib.fiberId))
        species = speciesResiduals(residuals, self.config.minLines, lit)
        camera = cameraSet(pfsConfig, dataId.get("arm"))
        judged, verdict = residualVerdict(species, dataId, self.thresholds, self.config.minLines, camera)

        summary = {key: dataId.get(key) for key in ("visit", "arm", "spectrograph")}
        summary |= {"cameraSet": camera} | verdict | _calibProvenance(detectorMapCalib)
        self.log.info(
            "DM line residuals %-7s %s%s %s: %s [%s]",
            verdict["qaStatus"],
            dataId.get("arm"),
            dataId.get("spectrograph"),
            verdict["recommendedAction"],
            _describe(species) or "no lines",
            verdict["qaReason"] or "nominal",
        )
        return Struct(
            dmLineResidualMetrics=longResiduals(species, judged, dataId, camera),
            dmLineResidualSummary=pd.DataFrame([summary]),
            dmLineResidualData=lineTable(residuals, dataId, self.config.maxTraceRowsPerFiber),
        )


def _describe(species: pd.DataFrame) -> str:
    """Return the judged statistics of each species for the log line."""
    calib = species[(species["reference"] == CALIB) & species["statusType"].isin((BOTH, UNFITTED))]
    visit = species[(species["reference"] == VISIT) & (species["statusType"] == RESERVED)]
    rms = {row.description: row for row in visit.itertuples()}
    parts = []
    for row in calib.itertuples():
        component = "x" if row.description == TRACE else "y"
        offset = getattr(row, f"medianDelta{component.upper()}")
        text = f"{row.description} d{component}={offset:+.3f}"
        reserved = rms.get(row.description)
        if reserved is not None:
            text += f" rms{component}={getattr(reserved, f'robustRmsDelta{component.upper()}'):.3f}"
        parts.append(text)
    return ", ".join(parts)


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
