"""Measure a ``fiberProfiles`` calibration.

Extraction weights every exposure by the calibration's profiles, so a profile
that carries another fiber's light, or that is a copy of its neighbours and
not a measurement, biases every spectrum of that fiber for the run. This task
stores, per detector, one row per fiber of the calibration and the detector's
metrics; the measurement is the stack-free `pfs.drp.qa.metrics.fiberProfiles`.
It stores numbers only and judges nothing.

It has a pipeline of its own, ``pipelines/fiberProfilesQa.yaml``, which
``drpQa.py calib profiles`` runs on a candidate calibration before it is
certified: the calibration measured is the one the collections resolve for the
visit, so with the candidate's collection first in the inputs it is the
candidate. ``calibVisit0`` and ``calibRun`` in both tables say which calibration
a row is of and where it was read from.

It is not in ``drpQA.yaml``: nothing reads its tables after a visit yet. It is
written to join that pipeline: a visit that is not a quartz (``obsType``
``trace``) is skipped from its raw header before the calibration is read, so
there it would cost the header and the pfsConfig on every other visit, and
measure the calibration in force on each quartz. ``anyFrameType`` measures on a
visit of any type, as ``drpQa.py calib profiles`` does.
"""

from typing import Any

import numpy as np
from lsst.daf.butler import DeferredDatasetHandle
from lsst.pex.config import Field
from lsst.pipe.base import (
    InputQuantizedConnection,
    NoWorkFound,
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

from pfs.datamodel import TargetType
from pfs.drp.qa.metrics.fiberProfiles import FIBER_TABLE_COLUMNS, METRICS, profileQa
from pfs.drp.qa.metrics.lineResiduals import frameType

__all__ = ["FiberProfilesQaConfig", "FiberProfilesQaConnections", "FiberProfilesQaTask", "expectedFibers"]


class FiberProfilesQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Connections for FiberProfilesQaTask."""

    rawMetadata = InputConnection(
        name="raw.metadata",
        doc="The raw exposure's header: whether the visit is a quartz (W_SEQTYP, W_SEQNAM).",
        storageClass="PropertyList",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    # A prerequisite with minimum=0, as in the other drpQA tasks: a dataset type must be a prerequisite to every
    # task of a graph or to none, and a visit without one must not stop the graph.
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Fiber configuration: the fibers the detector should have a profile for. Without one, nMissing is NaN.",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
        minimum=0,
    )
    # minimum=0: the task is anchored on the raw header, so a detector with no calibration valid for it still
    # has a quantum, and a required prerequisite that is missing must not stop the graph.
    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc=(
            "The calibration that is measured: the fiber profiles the collections resolve for the visit; read"
            " only on a quartz. A quartz without one is skipped."
        ),
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
        deferLoad=True,
        minimum=0,
    )

    fiberProfilesQaFibers = OutputConnection(
        name="fiberProfilesQaFibers",
        doc=(
            "One row per fiber of the calibration: ``visit``, ``arm``, ``spectrograph``, ``calibVisit0``,"
            " ``calibObsDate`` and ``calibRun`` (the calibration measured, and the RUN it was read from), then "
            + ", ".join(f"``{name}``" for name in FIBER_TABLE_COLUMNS)
            + " (`pfs.drp.qa.metrics.fiberProfiles.fiberTable`)."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    fiberProfilesQaMetrics = OutputConnection(
        name="fiberProfilesQaMetrics",
        doc=(
            "The detector's metrics in long format: ``visit``, ``arm``, ``spectrograph``, ``calibVisit0``,"
            " ``calibRun``, ``metric`` and ``value``, with one row for each of "
            + ", ".join(f"``{name}``" for name in METRICS)
            + ". Stored, not judged."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class FiberProfilesQaConfig(PipelineTaskConfig, pipelineConnections=FiberProfilesQaConnections):
    """Configuration for FiberProfilesQaTask."""

    anyFrameType = Field(
        dtype=bool,
        default=False,
        doc=(
            "Measure on every visit, whatever its header says: to measure the calibration a chosen visit"
            " resolves (drpQa.py calib profiles sets it), or for raws without W_SEQTYP. With a data query that"
            " selects the visits; never in a nightly run."
        ),
    )


def expectedFibers(pfsConfig: Any, spectrograph: int) -> np.ndarray:
    """Return the fibers a detector's calibration should hold a profile for.

    The profile build makes one for every fiber of the spectrograph but the
    engineering fibers.

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.PfsConfig`
        Fiber configuration of the visit.
    spectrograph : `int`
        Spectrograph number.

    Returns
    -------
    `numpy.ndarray` of `int`
        Fiber identifiers.
    """
    config = pfsConfig.select(spectrograph=spectrograph)
    keep = np.asarray(config.targetType, dtype=np.int64) != int(TargetType.ENGINEERING)
    return np.asarray(config.fiberId, dtype=np.int64)[keep]


class FiberProfilesQaTask(PipelineTask):
    """Store the statistics of a detector's ``fiberProfiles`` calibration."""

    ConfigClass = FiberProfilesQaConfig
    _DefaultName = "fiberProfilesQa"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        dataId = dict(**inputRefs.rawMetadata.dataId.mapping)
        inputs = butlerQC.get(inputRefs)
        pfsConfig = inputs.get("pfsConfig")
        obsType = frameType(inputs["rawMetadata"], getattr(pfsConfig, "header", None))
        if obsType != "trace" and not self.config.anyFrameType:
            raise NoWorkFound(f"{dataId}: not a quartz ({obsType})")
        fiberProfiles = inputs.get("fiberProfiles")
        if fiberProfiles is None:
            raise NoWorkFound(f"{dataId}: a quartz with no fiberProfiles calibration to measure")
        calibRun = getattr(inputRefs.fiberProfiles, "run", "") or ""
        try:
            outputs = self.run(
                fiberProfiles=fiberProfiles, dataId=dataId, pfsConfig=pfsConfig, calibRun=calibRun
            )
        except ValueError as e:
            self.log.error("FiberProfilesQaTask failed for %s: %s", dataId, e)
        else:
            butlerQC.put(outputs, outputRefs)

    def run(
        self, fiberProfiles: Any, dataId: dict[str, Any], pfsConfig: Any = None, calibRun: str = ""
    ) -> Struct:
        """Measure one detector's calibration.

        Parameters
        ----------
        fiberProfiles : `pfs.drp.stella.FiberProfileSet` or `lsst.daf.butler.DeferredDatasetHandle`
            The calibration of the detector, or the handle to read it from.
        dataId : `dict`
            ``visit``, ``arm``, ``spectrograph``.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Fiber configuration of the visit, for the expected fibers.
        calibRun : `str`, optional
            The RUN collection the calibration was read from.

        Returns
        -------
        fiberProfilesQaFibers, fiberProfilesQaMetrics : `pandas.DataFrame`
            See the connections.

        Raises
        ------
        ValueError
            The calibration holds no profile.
        """
        if isinstance(fiberProfiles, DeferredDatasetHandle):
            fiberProfiles = fiberProfiles.get()
        if len(fiberProfiles) == 0:
            raise ValueError("the fiberProfiles calibration holds no profile")
        expected = expectedFibers(pfsConfig, int(dataId["spectrograph"])) if pfsConfig is not None else None
        identity = getattr(fiberProfiles, "identity", None)
        fibers, metrics = profileQa(
            fiberProfiles,
            dataId,
            expected=expected,
            calibVisit0=int(getattr(identity, "visit0", -1)),
            calibObsDate=str(getattr(identity, "obsDate", "")),
            calibRun=calibRun,
        )
        values = dict(zip(metrics["metric"], metrics["value"], strict=True))
        self.log.info(
            "fiberProfiles %s (visit0 %d): %d profiles, %d replaced, %d missing; median FWHM %.3f px,"
            " largest wing excess %.4f measured and %.4f replaced",
            dataId,
            int(getattr(identity, "visit0", -1)),
            int(values["nProfiles"]),
            int(values["nReplaced"]),
            -1 if np.isnan(values["nMissing"]) else int(values["nMissing"]),
            values["medProfileFwhm"],
            values["maxWingExcess"],
            values["maxWingExcessReplaced"],
        )
        return Struct(fiberProfilesQaFibers=fibers, fiberProfilesQaMetrics=metrics)
