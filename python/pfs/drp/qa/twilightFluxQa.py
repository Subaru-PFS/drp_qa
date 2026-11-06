"""Each fiber's flux on a twilight, for judging the twilight sequence afterwards.

A twilight is judged against the other exposures of its sequence
(`pfs.drp.qa.metrics.twilight`), which a quantum of one visit cannot see. So
this task stores what the judgement needs, per detector: each lit fiber's
median flux / norm over the arm's window, its S/N and its focal-plane
position, with the sequence id. ``bin.src/twilightQa.py`` judges the stored
tables of a collection.

It runs in ``drpQA.yaml``, on every visit. A visit that is not a twilight is
skipped from its raw header (`pfs.drp.qa.metrics.lineResiduals.frameType`)
before the pfsArm is read, so it costs the header and the pfsConfig.

Raws from before the IIC wrote ``W_SEQTYP``, ``W_SEQNAM`` and ``W_SEQID``
(Run21, 2025-03/04, has none) are not recognised: set ``assumeTwilight`` with
a data query that selects twilights, and give ``bin.src/twilightQa.py`` the
sequences (``--sequences``).
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

from pfs.datamodel import PfsConfig
from pfs.drp.qa.imageQualityQa import litFibers
from pfs.drp.qa.metrics.lineResiduals import frameType
from pfs.drp.qa.metrics.twilight import FIBER_COLUMNS, WINDOWS, addPositions, fiberFlux
from pfs.drp.stella.datamodel import PfsArm

__all__ = ["TwilightFluxQaConfig", "TwilightFluxQaConnections", "TwilightFluxQaTask"]


class TwilightFluxQaConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    """Connections for TwilightFluxQaTask."""

    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra of the detector; read only for a twilight.",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        deferLoad=True,
    )
    rawMetadata = InputConnection(
        name="raw.metadata",
        doc=(
            "The raw exposure's header: whether the visit is a twilight (W_SEQTYP, W_SEQNAM), and its"
            " sequence (W_SEQID). The pfsConfig header stands in when the raw's lacks the type."
        ),
        storageClass="PropertyList",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    # A prerequisite with minimum=0, as in the other drpQA tasks: a dataset type must be a prerequisite to every
    # task of a graph or to none, and a visit without one must not stop the graph.
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Fiber configuration: the lit fibers and their focal-plane positions. A twilight without one is skipped.",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
        minimum=0,
    )

    twilightFiberFlux = OutputConnection(
        name="twilightFiberFlux",
        doc=(
            "One row per lit fiber of a twilight: ``visit``, ``arm``, ``spectrograph``, ``sequence`` (W_SEQID;"
            " -1 when no header has it), ``seqName`` (W_SEQNAM), "
            + ", ".join(f"``{name}`` ({doc.rstrip('.')})" for name, doc in FIBER_COLUMNS.items())
            + "."
        ),
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class TwilightFluxQaConfig(PipelineTaskConfig, pipelineConnections=TwilightFluxQaConnections):
    """Configuration for TwilightFluxQaTask; the windows are `pfs.drp.qa.metrics.twilight.WINDOWS`."""

    assumeTwilight = Field(
        dtype=bool,
        default=False,
        doc=(
            "Treat every visit as a twilight, whatever its header says: for raws without W_SEQTYP and W_SEQNAM"
            " (before 2025-11), with a data query that selects twilights. Never in drpQA's nightly run."
        ),
    )


class TwilightFluxQaTask(PipelineTask):
    """Store each lit fiber's median flux on a twilight."""

    ConfigClass = TwilightFluxQaConfig
    _DefaultName = "twilightFluxQa"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        dataId = dict(**inputRefs.pfsArm.dataId.mapping)
        inputs = butlerQC.get(inputRefs)
        pfsConfig = inputs.get("pfsConfig")
        obsType = frameType(inputs["rawMetadata"], getattr(pfsConfig, "header", None))
        if obsType != "twilight" and not self.config.assumeTwilight:
            raise NoWorkFound(f"{dataId}: not a twilight ({obsType})")
        if pfsConfig is None:
            raise NoWorkFound(f"{dataId}: a twilight without a pfsConfig, so no fiber positions")
        try:
            outputs = self.run(**inputs, dataId=dataId)
        except ValueError as e:
            self.log.error("TwilightFluxQaTask failed for %s: %s", dataId, e)
        else:
            butlerQC.put(outputs, outputRefs)

    def run(
        self,
        pfsArm: PfsArm | DeferredDatasetHandle,
        pfsConfig: PfsConfig,
        dataId: dict[str, Any],
        rawMetadata: Any = None,
    ) -> Struct:
        """Reduce a twilight detector's spectra to one row per lit fiber.

        Parameters
        ----------
        pfsArm : `PfsArm` or `lsst.daf.butler.DeferredDatasetHandle`
            Extracted spectra, or the handle to read them from.
        pfsConfig : `PfsConfig`
            Fiber configuration of the visit.
        dataId : `dict`
            ``visit``, ``arm``, ``spectrograph``.
        rawMetadata : `lsst.daf.base.PropertyList`, optional
            The raw exposure's header, for the sequence; the pfsArm's stands in.

        Returns
        -------
        twilightFiberFlux : `pandas.DataFrame`
            See the connection.
        """
        if isinstance(pfsArm, DeferredDatasetHandle):
            pfsArm = pfsArm.get()
        header = rawMetadata if rawMetadata is not None else pfsArm.metadata
        arm = dataId["arm"]
        lit, _ = litFibers(pfsConfig, pfsArm.fiberId)
        keep = np.isin(pfsArm.fiberId, sorted(lit))
        satBit = pfsArm.flags["SAT"] if "SAT" in pfsArm.flags.flags else 0
        table = fiberFlux(
            pfsArm.fiberId[keep],
            pfsArm.wavelength[keep],
            pfsArm.flux[keep],
            pfsArm.norm[keep],
            pfsArm.variance[keep],
            pfsArm.mask[keep],
            WINDOWS[arm],
            satBit,
        )
        table = addPositions(table, pfsConfig.fiberId, pfsConfig.pfiCenter)
        sequence = header.get("W_SEQID")
        if sequence is None:
            sequence = pfsArm.metadata.get("W_SEQID")
        table.insert(0, "visit", int(dataId["visit"]))
        table.insert(1, "arm", arm)
        table.insert(2, "spectrograph", int(dataId["spectrograph"]))
        table.insert(3, "sequence", int(sequence) if sequence is not None else -1)
        table.insert(4, "seqName", str(header.get("W_SEQNAM") or pfsArm.metadata.get("W_SEQNAM") or ""))
        self.log.info(
            "Twilight %s, sequence %s: %d lit fibers, median S/N %.1f",
            dataId,
            sequence,
            len(table),
            float(np.nanmedian(table["snr"])) if len(table) else float("nan"),
        )
        return Struct(twilightFiberFlux=table)
