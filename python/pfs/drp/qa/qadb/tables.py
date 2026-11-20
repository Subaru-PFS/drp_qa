"""The QA numbers, verdicts and normal values as qadb rows.

Each numbers table of qadb mirrors one table of numbers, column for column with
the names in snake_case (`snakeCase`), keyed by the processing (a collection and
Butler RUN: ``data_processing_pipe2d``), the visit and the camera (``b1``): the
tasks' Butler tables, and the twilight metrics the loader computes once a
sequence is complete. ``qa_gate`` holds the one verdict per image
(`pfs.drp.qa.metrics.gate`), and ``qa_normal_value`` each camera's normal
values (`pfs.drp.qa.metrics.normalValues`). No numbers table holds a verdict.

Nothing here touches a database: every function takes and returns
`pandas.DataFrame`, with Python scalars and `None` for missing values, ready
for `pfs.drp.qa.qadb.writer`.

The tables are ``spt_qa_database``'s (its ``qadb.models``); a column added to a
Butler table reaches qadb only once both that schema and the spec here name it.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "IMAGE_KEYS",
    "IMAGE_QUALITY",
    "IMAGE_QUALITY_SPECIES",
    "KEYS",
    "LINE_RESIDUAL",
    "LINE_RESIDUAL_METRIC",
    "NORMAL_VALUE",
    "NORMAL_VALUE_KEYS",
    "QA_GATE",
    "TWILIGHT_QA",
    "TableSpec",
    "cameraName",
    "gateRows",
    "imageRows",
    "normalValueFrame",
    "normalValueRows",
    "snakeCase",
]

#: Key columns of every per-image table.
KEYS = ("processing_id", "pfs_visit_id", "camera_name")

#: Columns identifying an image in a Butler table.
IMAGE_KEYS = ("visit", "arm", "spectrograph")

_ACRONYM = re.compile(r"([A-Z]+)([A-Z][a-z])")
_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")


def snakeCase(name: str) -> str:
    """Return a Butler column name as a qadb column name.

    ``medFwhm`` is ``med_fwhm``, ``pctLowSN`` is ``pct_low_sn`` and
    ``fitReservedXRms`` is ``fit_reserved_x_rms``; a name already in snake_case
    is unchanged.
    """
    return _BOUNDARY.sub(r"\1_\2", _ACRONYM.sub(r"\1_\2", name)).lower()


def cameraName(arm: Any, spectrograph: Any) -> str:
    """Return the camera name of an arm and spectrograph, e.g. ``b1``."""
    return f"{arm}{int(spectrograph)}"


@dataclass(frozen=True)
class TableSpec:
    """A qadb table mirroring a Butler table.

    Parameters
    ----------
    name : `str`
        The qadb table.
    source : `str`
        The Butler dataset type.
    columns : mapping [`str`, `str`]
        The Butler columns copied, in order, with the type each is stored as:
        ``int``, ``float``, ``bool``, ``str`` or ``datetime`` (anything
        `pandas.to_datetime` reads; null when it doesn't). The qadb column is
        `snakeCase` of the Butler column.
    extraKeys : `tuple` [`str`, ...], optional
        Butler columns that, with `KEYS`, identify a row of a long table;
        stored as strings.
    renames : `tuple` [`tuple` [`str`, `str`], ...], optional
        Butler columns whose qadb column is not their `snakeCase`, as
        ``(butlerColumn, qadbColumn)`` pairs.
    """

    name: str
    source: str
    columns: Mapping[str, str]
    extraKeys: tuple[str, ...] = ()
    renames: tuple[tuple[str, str], ...] = ()

    def qadbColumn(self, column: str) -> str:
        """Return the qadb column of a Butler column."""
        return dict(self.renames).get(column) or snakeCase(column)

    @property
    def keys(self) -> tuple[str, ...]:
        """The qadb key columns."""
        return (*KEYS, *(self.qadbColumn(key) for key in self.extraKeys))

    @property
    def qadbColumns(self) -> tuple[str, ...]:
        """Every qadb column, keys first."""
        return (*self.keys, *(self.qadbColumn(column) for column in self.columns))


IMAGE_QUALITY = TableSpec(
    "image_quality_qa",
    "iqQaMetrics",
    {
        "spectrograph": "int",
        "arm": "str",
        "obsType": "str",
        "seqName": "str",
        "seqType": "str",
        "illumination": "str",
        "lamps": "str",
        "traceOnly": "bool",
        "medFwhm": "float",
        "medDxCenter": "float",
        "dxCenterRms": "float",
        "pctFlagged": "float",
        "pctLowSN": "float",
        "pctMeasFail": "float",
        "nLines": "int",
        "linesPerFiber": "float",
        "nLitFibers": "int",
        "nFibers": "int",
        "nTraceRows": "int",
        "nSaturatedLines": "int",
        "pctSaturatedPixels": "float",
        "calexpRun": "str",
        "isrBadPixels": "int",
        "isrTime": "float",
        "cosmicRayCount": "int",
        "cosmicRayPixels": "int",
        "cosmicRayTime": "float",
        "reduceExposureTime": "float",
        "fitChi2": "float",
        "fitDof": "int",
        "fitXRms": "float",
        "fitYRms": "float",
        "fitXSoften": "float",
        "fitYSoften": "float",
        "fitNLines": "int",
        "fitTotalLines": "int",
        "fitReservedChi2": "float",
        "fitReservedXRms": "float",
        "fitReservedYRms": "float",
        "fitReservedXSoften": "float",
        "fitReservedYSoften": "float",
        "fitReservedNLines": "int",
    },
)

IMAGE_QUALITY_SPECIES = TableSpec(
    "image_quality_qa_species", "iqQaSpeciesMetrics", {"value": "float"}, extraKeys=("description", "metric")
)

LINE_RESIDUAL = TableSpec(
    "dm_line_residual_qa",
    "dmLineResidualSummary",
    {
        "spectrograph": "int",
        "arm": "str",
        "cameraSet": "str",
        "obsType": "str",
        "seqType": "str",
        "fitted": "bool",
        "nSpecies": "int",
        "nSpeciesMeasured": "int",
        "calibId": "str",
        "calibRun": "str",
        "calibCreated": "datetime",
        "calibFirstVisit": "int",
        "calibLastVisit": "int",
    },
)

LINE_RESIDUAL_METRIC = TableSpec(
    "dm_line_residual_qa_metric",
    "dmLineResidualMetrics",
    {"value": "float"},
    extraKeys=("reference", "description", "statusType", "metric"),
)

TWILIGHT_QA = TableSpec(
    "twilight_qa",
    "twilightMetrics",
    {
        "spectrograph": "int",
        "arm": "str",
        "sequence": "int",
        "seqName": "str",
        "nExposures": "int",
        "snr": "float",
        "brightness": "float",
        "tooFaint": "bool",
        "nReference": "int",
        "noise": "float",
        "satFraction": "float",
        "nMeasured": "int",
        "twilightLevelOffset": "float",
        "twilightResidRms": "float",
        "twilightPatch": "float",
        "twilightArmMean": "float",
        "twilightArmHalfDiff": "float",
    },
    renames=(("sequence", "iic_sequence_id"),),
)

#: ``qa_gate``: the gate's verdict on an image, the check that decided it, and why.
QA_GATE = (*KEYS, "frame_type", "qa_status", "qa_decided_by", "qa_reason", "mode", "decided_at")

#: ``qa_normal_value``: by qadb column, the column of `pfs.drp.qa.metrics.normalValues.NORMAL_COLUMNS`.
NORMAL_VALUE = {
    "collection_name": "collection",
    "metric": "metric",
    "camera": "camera",
    "frame_type": "frameType",
    "selector": "selector",
    "valid_from": "validFrom",
    "value": "value",
    "n_frames": "count",
    "source_visits": "sourceVisits",
    "accepted_by": "acceptedBy",
    "accepted_at": "acceptedAt",
    "reason": "reason",
}

#: What identifies a ``qa_normal_value`` row: storing the same key again replaces it.
NORMAL_VALUE_KEYS = ("collection_name", "metric", "camera", "frame_type", "selector", "valid_from")

_NORMAL_TYPES = {"valid_from": "int", "value": "float", "n_frames": "int", "accepted_at": "datetime"}


def _scalar(value: Any, dtype: str) -> Any:
    """Return ``value`` as a Python ``dtype`` scalar, or `None` when missing or not finite."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if dtype == "int":
        return int(value)
    if dtype == "float":
        return float(value)
    if dtype == "bool":
        return bool(value)
    if dtype == "datetime":
        when = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(when) else when.to_pydatetime()
    if isinstance(value, list | tuple):
        return ",".join(str(item) for item in value)
    return str(value)


def _column(frame: pd.DataFrame, name: str, dtype: str) -> list[Any]:
    """Return a Butler column as Python scalars, all `None` when the table lacks it."""
    if name not in frame.columns:
        return [None] * len(frame)
    return [_scalar(value, dtype) for value in frame[name].to_numpy(dtype=object)]


def _processingIds(frame: pd.DataFrame, processingId: int | Mapping[str, int]) -> list[int]:
    """Return each row's processing: one id, or the id of the row's ``run``."""
    if isinstance(processingId, Mapping):
        if "run" not in frame.columns:
            raise ValueError("A mapping of processing ids by run needs a 'run' column")
        return [int(processingId[run]) for run in frame["run"]]
    return [int(processingId)] * len(frame)


def imageRows(frame: pd.DataFrame, spec: TableSpec, processingId: int | Mapping[str, int]) -> pd.DataFrame:
    """Return a Butler table's rows as rows of its qadb table.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        Rows of ``spec.source``, with `IMAGE_KEYS` (and ``run`` when
        ``processingId`` is a mapping).
    spec : `TableSpec`
        The table.
    processingId : `int` or mapping [`str`, `int`]
        The processing of every row, or of each row's Butler ``run``.

    Returns
    -------
    `pandas.DataFrame`
        ``spec.qadbColumns``, in order, as Python scalars with `None` where the
        Butler table has no value or lacks the column. The row order is kept.
    """
    rows = {
        "processing_id": _processingIds(frame, processingId),
        "pfs_visit_id": _column(frame, "visit", "int"),
        "camera_name": [
            cameraName(arm, spectrograph)
            for arm, spectrograph in zip(frame["arm"], frame["spectrograph"], strict=True)
        ],
    }
    for key in spec.extraKeys:
        rows[spec.qadbColumn(key)] = _column(frame, key, "str")
    for column, dtype in spec.columns.items():
        rows[spec.qadbColumn(column)] = _column(frame, column, dtype)
    return pd.DataFrame(rows, columns=list(spec.qadbColumns), dtype=object)


def gateRows(
    verdicts: pd.DataFrame, processingId: int | Mapping[str, int], mode: str, decidedAt: Any
) -> pd.DataFrame:
    """Return the gate's verdicts as ``qa_gate`` rows.

    Parameters
    ----------
    verdicts : `pandas.DataFrame`
        `pfs.drp.qa.metrics.gate.gate`: `IMAGE_KEYS`, ``frameType``,
        ``qaStatus``, ``qaDecidedBy`` and ``qaReason`` (and ``run`` when
        ``processingId`` is a mapping).
    processingId : `int` or mapping [`str`, `int`]
        As for `imageRows`.
    mode : `str`
        ``realtime``, ``comparison`` or ``backfill``.
    decidedAt : `datetime.datetime`
        When the gate decided.

    Returns
    -------
    `pandas.DataFrame`
        `QA_GATE`, one row per image, in the order given.
    """
    rows = {
        "processing_id": _processingIds(verdicts, processingId),
        "pfs_visit_id": _column(verdicts, "visit", "int"),
        "camera_name": [
            cameraName(arm, spectrograph)
            for arm, spectrograph in zip(verdicts["arm"], verdicts["spectrograph"], strict=True)
        ],
        "frame_type": _column(verdicts, "frameType", "str"),
        "qa_status": _column(verdicts, "qaStatus", "str"),
        "qa_decided_by": _column(verdicts, "qaDecidedBy", "str"),
        "qa_reason": _column(verdicts, "qaReason", "str"),
        "mode": [mode] * len(verdicts),
        "decided_at": [decidedAt] * len(verdicts),
    }
    return pd.DataFrame(rows, columns=list(QA_GATE), dtype=object)


def normalValueRows(normalValues: pd.DataFrame) -> pd.DataFrame:
    """Return normal values (`pfs.drp.qa.metrics.normalValues.NORMAL_COLUMNS`) as ``qa_normal_value`` rows."""
    rows = {
        column: _column(normalValues, source, _NORMAL_TYPES.get(column, "str"))
        for column, source in NORMAL_VALUE.items()
    }
    # A value kept per arm, or with no lamp or species, has an empty selector, not a null: it is part of the key.
    for column in ("frame_type", "selector"):
        rows[column] = [value or "" for value in rows[column]]
    return pd.DataFrame(rows, columns=list(NORMAL_VALUE), dtype=object)


def normalValueFrame(rows: pd.DataFrame) -> pd.DataFrame:
    """Return ``qa_normal_value`` rows as `pfs.drp.qa.metrics.normalValues.NORMAL_COLUMNS`."""
    frame = rows.rename(columns=NORMAL_VALUE)
    return frame.reindex(columns=list(NORMAL_VALUE.values())).reset_index(drop=True)
