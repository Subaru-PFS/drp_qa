"""Record each detector's reduction statistics, read from task metadata and the ISR header.

Per detector: the ISR counts and dark scale, the per-amplifier gain and read noise, and
the detectorMap fit, scattered-light and PFI-correction scalars the reduction recorded.
The task binds the Butler to the stack-free measurement in
`pfs.drp.qa.metrics.reductionStats`; it stores numbers only and judges nothing. Detector
health is stored, not gated: the limits come after a run of values.

The task is anchored on the raw header, so a quantum runs on every exposed detector — even
one whose reduction never ran — rather than only where ISR produced metadata. The
reduction's own metadata inputs (``isr_metadata``, ``reduceExposure_metadata``) and the ISR
header are optional.

This replaces the log scraping ``imageQualityQa`` used to do for the detectorMap fit and
ISR counts: the same numbers are read here from the task metadata, not from log text.
"""

from typing import Any

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

from pfs.drp.qa.metrics.reductionStats import asMapping, reductionStatsFrame

__all__ = ["ReductionStatsConfig", "ReductionStatsTask"]


class ReductionStatsConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Butler inputs and output for `ReductionStatsTask`."""

    rawMetadata = InputConnection(
        name="raw.metadata",
        doc=(
            "The raw exposure header, the anchor: present for every exposed detector, so a quantum runs"
            " on each — even one whose reduction never ran — rather than only where ISR produced"
            " metadata. Its presence defines the quantum; deferLoad keeps its content unread."
        ),
        storageClass="PropertyList",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        deferLoad=True,
    )

    isrMetadata = InputConnection(
        name="isr_metadata",
        doc=(
            "ISR task metadata: the n-arm whole-detector counts (nSat, nLow, nDefects, ..., nCR) and,"
            " on a CCD with a dark, the dark scale. Each count is recorded only when its step ran, and"
            " the whole input is absent when ISR did not run."
        ),
        storageClass="TaskMetadata",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        minimum=0,
    )

    reduceExposureMetadata = InputConnection(
        name="reduceExposure_metadata",
        doc=(
            "reduceExposure task metadata: the detectorMap fit scalars (adjustDetectorMap, read only when"
            " the adjustment applied), and the scattered-light and PFI-correction scalars. Absent when"
            " only ISR and fitDetectorMap ran."
        ),
        storageClass="TaskMetadata",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        minimum=0,
    )

    header = InputConnection(
        name="postISRCCD.metadata",
        doc=(
            "The ISR output header: per-amplifier gain and read noise (``LSST GAIN <amp>``,"
            " ``LSST READNOISE <amp>``) on the CCDs."
        ),
        storageClass="PropertyList",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        minimum=0,
    )

    reductionStats = OutputConnection(
        name="reductionStats",
        doc=(
            "Detector-health statistics in long format, one row per amplifier and metric:"
            " ``visit``, ``arm``, ``spectrograph``, ``amplifier`` (empty for a whole-detector"
            " statistic, as on the n arm), ``metric`` and ``value``. Stored, not judged; loaded into"
            " qadb's ``detector_health_qa``."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class ReductionStatsConfig(PipelineTaskConfig, pipelineConnections=ReductionStatsConnections):
    """Configuration for `ReductionStatsTask` (no tunables: it reads and records)."""


class ReductionStatsTask(PipelineTask):
    """Read one detector's reduction statistics into a long-format table."""

    ConfigClass = ReductionStatsConfig
    _DefaultName = "reductionStats"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        dataId = dict(**inputRefs.rawMetadata.dataId.mapping)
        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = dataId
        try:
            outputs = self.run(**inputs)
        except ValueError as e:
            self.log.error("ReductionStatsTask failed for %s: %s", dataId, e)
        else:
            butlerQC.put(outputs, outputRefs)

    def run(
        self,
        dataId: dict,
        rawMetadata: Any = None,
        isrMetadata: Any = None,
        reduceExposureMetadata: Any = None,
        header: Any = None,
    ) -> Struct:
        """Return one detector's reduction statistics as a long-format table.

        Parameters
        ----------
        dataId : `dict`
            The image: ``visit``, ``arm``, ``spectrograph``.
        rawMetadata : `lsst.pipe.base.DeferredDatasetHandle`, optional
            A deferred handle to the raw header — the Butler anchor that gives one quantum per
            exposed detector. Its content is not read here.
        isrMetadata, reduceExposureMetadata : `lsst.pipe.base.TaskMetadata`, optional
            The ISR and reduceExposure task metadata.
        header : `lsst.daf.base.PropertyList`, optional
            The ISR output header.

        Returns
        -------
        `lsst.pipe.base.Struct`
            ``reductionStats``: `~pfs.drp.qa.metrics.reductionStats.REDUCTION_COLUMNS`,
            one row per amplifier and metric recorded, always including the coverage rows,
            so it is never empty for an exposed detector.
        """
        frame = reductionStatsFrame(
            dataId,
            isrMetadata=asMapping(isrMetadata),
            reduceExposureMetadata=asMapping(reduceExposureMetadata),
            header=asMapping(header),
        )
        return Struct(reductionStats=frame)
