"""The metrics table's schema, shared by `serving` (which builds it) and `store` (which queries it).

Imports nothing, so the store needs no scipy or stack.
"""

__all__ = [
    "IMAGE_STATUS_ORDER",
    "METRICS_COLUMNS",
    "NOTE_COLUMNS",
    "STATUS_ORDER",
    "VERDICT",
]

#: Verdicts from best to worst, as `pfs.drp.qa.metrics.gate.STATUS_ORDER` (checked by a test).
STATUS_ORDER = ("PASS", "WARN", "FAIL")

#: The ``metric`` of a row holding a source's verdict on a whole image.
VERDICT = "<verdict>"

#: Columns of the notes table (`serving.noteRows`).
NOTE_COLUMNS = ["visit", "arm", "spectrograph", "source", "note"]

#: Image statuses from best to worst; ``UNKNOWN`` (nothing judged) ranks below ``PASS``.
IMAGE_STATUS_ORDER = ("UNKNOWN", *STATUS_ORDER)

#: Columns of the metrics table.
METRICS_COLUMNS = {
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
