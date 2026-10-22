"""The Tier 1 table's schema, shared by `serving` (which writes it) and `store` (which reads it).

Imports only pandas' companions, so the serving side needs no scipy or stack.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq

__all__ = [
    "IMAGE_STATUS_ORDER",
    "NOTE",
    "PROVENANCE_KEY",
    "STATUS_ORDER",
    "TIER1_COLUMNS",
    "VERDICT",
    "readProvenance",
]

#: Verdicts from best to worst, as `pfs.drp.qa.metrics.gate.STATUS_ORDER` (checked by a test).
STATUS_ORDER = ("PASS", "WARN", "FAIL")

#: The ``metric`` of a row holding a source's verdict on a whole image.
VERDICT = "<verdict>"

#: The ``metric`` of a row holding an opdb note on a visit, its sequence, or one camera (``reason``).
NOTE = "<note>"

#: Parquet metadata key holding the provenance JSON.
PROVENANCE_KEY = b"pfs.drp.qa.dashboard.provenance"

#: Image statuses from best to worst; ``UNKNOWN`` (nothing judged) ranks below ``PASS``.
IMAGE_STATUS_ORDER = ("UNKNOWN", *STATUS_ORDER)

#: Columns of the Tier 1 table.
TIER1_COLUMNS = {
    "visit": "int64",
    "arm": "string",
    "spectrograph": "int64",
    "obsType": "string",
    "seqName": "string",
    "sequenceKey": "int64",
    "sequenceSource": "string",
    "timeExpStart": "datetime64[ns]",
    "source": "string",
    "metric": "string",
    "description": "string",
    "reference": "string",
    "statusType": "string",
    "value": "float64",
    "status": "string",
    "warn": "float64",
    "fail": "float64",
    "population": "string",
    "reason": "string",
    "run": "string",
}


def readProvenance(path: Path | str) -> dict:
    """Return the provenance `writeTier1` stored in ``path``; empty if none."""
    metadata = pq.read_schema(path).metadata or {}
    return json.loads(metadata[PROVENANCE_KEY]) if PROVENANCE_KEY in metadata else {}
