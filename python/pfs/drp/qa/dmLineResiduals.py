"""Line residuals against ``detectorMap_calib`` and the visit's ``detectorMap``.

Per detector: has the instrument moved away from the calibration the visit is
reduced against, and how well does the visit's adjusted detectorMap predict
the lines it was not fitted to? The task stores numbers only: the measurement
is stack-free, in `pfs.drp.qa.metrics.lineResiduals`, and this module binds it
to the Butler. The verdict is given afterwards, by `pfs.drp.qa.metrics.gate`.
It replaces ``dmResiduals`` and ``dmCombinedResiduals``.
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
from pfs.drp.qa.metrics.lineResiduals import (
    BOTH,
    CALIB,
    RESERVED,
    TRACE,
    UNFITTED,
    VISIT,
    cameraSet,
    frameType,
    lineResiduals,
    lineTable,
    longResiduals,
    residualSummary,
    sequenceType,
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

    rawMetadata = InputConnection(
        name="raw.metadata",
        doc=(
            "The raw exposure's header: the frame's sequence type (seqType, W_SEQTYP as it is read) and the"
            " observation type it gives with W_SEQNAM (obsType)."
            " The pfsConfig header stands in when the raw's lacks them; Run25 and Run26 pfsConfigs carry"
            " neither, so the raw's is read first."
        ),
        storageClass="PropertyList",
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
        doc=(
            "Fiber configuration: the lit fibers (linesPerFiber), the arms read (cameraSet) and, when the raw"
            " header lacks it, the frame's observation type (obsType, from its header's W_SEQTYP)."
        ),
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
        minimum=0,
    )

    dmLineResidualMetrics = OutputConnection(
        name="dmLineResidualMetrics",
        doc=(
            "Line residuals in long format, one row per reference, line species, line set and metric:"
            " ``visit``, ``arm``, ``spectrograph``, ``cameraSet``, ``obsType`` (the frame's observation type"
            " as imageQualityQa classifies it, from the raw or the pfsConfig header: ``trace`` for a quartz,"
            " ``arc``, ``twilight``, ...; ``unknown`` when neither header says), ``seqType`` (the IIC"
            " sequence type, W_SEQTYP as read; null when neither header has it), ``reference`` (``calib``:"
            " detectorMap_calib; ``visit``: the visit's adjusted detectorMap), ``description``,"
            " ``statusType`` (``BOTH``, ``USED``, ``RESERVED`` lines of the visit's detectorMap fit, or"
            " ``UNFITTED``, every good line, when the fit marked none), ``metric`` (``nLines``,"
            " ``linesPerFiber``; ``medianDeltaX``, ``robustRmsDeltaX`` for every species;"
            " ``medianDeltaY``, ``robustRmsDeltaY`` in pixels and ``medianDeltaYKms`` in km/s for lamp and"
            " sky lines), ``value``. No verdict: `pfs.drp.qa.metrics.gate` judges the rows afterwards."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    dmLineResidualSummary = OutputConnection(
        name="dmLineResidualSummary",
        doc=(
            "One row per detector: ``visit``, ``arm``, ``spectrograph``, ``cameraSet``, ``obsType``,"
            " ``seqType``, ``fitted``"
            " (false when the visit's detectorMap fit marked no line, so it is reduced against the"
            " calibration unadjusted),"
            " ``nSpecies`` (species with a line against the calibration), ``nSpeciesMeasured`` (those with"
            " ``minLines`` good lines), and"
            " the calibration: ``calibCreated``, ``calibFirstVisit`` and ``calibLastVisit`` (the visits it"
            " was made from), ``calibId`` and ``calibRun`` (look up its validity range with"
            " `pfs.drp.qa.metrics.readers.readCalibValidity`). No verdict."
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
            " reference: ``dxCalib``, ``dyCalib``, ``dxVisit``, ``dyVisit`` in pixels, then ``dyKmsCalib``,"
            " ``dyKmsVisit``, ``dy`` in km/s by that reference's dispersion at the line (``dy`` NaN for"
            " traces)."
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


class DmLineResidualsTask(PipelineTask):
    """Measure each line species' residuals from the calibration and the visit's detectorMap."""

    ConfigClass = DmLineResidualsConfig
    _DefaultName = "dmLineResiduals"

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
        rawMetadata: Any = None,
    ) -> Struct:
        """Measure the line residuals of one detector.

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
        rawMetadata : `lsst.daf.base.PropertyList`, optional
            The raw exposure's header, for the observation type.

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
        headers = (rawMetadata, getattr(pfsConfig, "header", None))
        obsType, seqType = frameType(*headers), sequenceType(*headers)
        measured = residualSummary(species, self.config.minLines)

        summary = {key: dataId.get(key) for key in ("visit", "arm", "spectrograph")}
        summary |= {"cameraSet": camera, "obsType": obsType, "seqType": seqType}
        summary |= measured | _calibProvenance(detectorMapCalib)
        self.log.info(
            "DM line residuals %s%s (%s, %s): %s",
            dataId.get("arm"),
            dataId.get("spectrograph"),
            obsType,
            "fitted" if measured["fitted"] else "unfitted",
            _describe(species) or "no lines",
        )
        return Struct(
            dmLineResidualMetrics=longResiduals(species, dataId, camera, obsType, seqType),
            dmLineResidualSummary=pd.DataFrame([summary]),
            dmLineResidualData=lineTable(residuals, dataId, self.config.maxTraceRowsPerFiber),
        )


def _describe(species: pd.DataFrame) -> str:
    """Return, for the log line, each species' offset from the calibration and scatter on RESERVED lines."""
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
