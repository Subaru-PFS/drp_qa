"""The QA tasks' Butler tables as qadb rows.

Each qadb table mirrors one Butler dataset type, column for column with the
names in snake_case (`snakeCase`), keyed by the processing (a collection and
Butler RUN: ``data_processing_pipe2d``), the visit and the camera (``b1``).
Nothing here touches a database: every function takes and returns
`pandas.DataFrame`, with Python scalars and `None` for missing values, ready
for `pfs.drp.qa.qadb.writer`.

The tables are those of ``spt_qa_database`` 0.2.0 (its ``qadb.models``); a
column added to a Butler table reaches qadb only once both that schema and the
spec here name it.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pfs.drp.qa.metrics.gate import THRESHOLD_FIELDS

__all__ = [
    "GATE_ORDER",
    "IMAGE_KEYS",
    "IMAGE_QUALITY",
    "IMAGE_QUALITY_SPECIES",
    "KEYS",
    "LINE_RESIDUAL",
    "LINE_RESIDUAL_METRIC",
    "QA_GATE",
    "QA_THRESHOLDS",
    "QA_THRESHOLD_ENTRY",
    "TASK_TABLES",
    "TableSpec",
    "cameraName",
    "gateRows",
    "imageRows",
    "populationKey",
    "snakeCase",
    "thresholdRows",
    "worstStatus",
]

#: Key columns of every per-image table.
KEYS = ("processing_id", "pfs_visit_id", "camera_name")

#: Columns identifying an image in a Butler table.
IMAGE_KEYS = ("visit", "arm", "spectrograph")

#: Verdicts from best to worst, for the gate: a task with nothing judged
#: (``UNKNOWN``) never outranks one that judged.
GATE_ORDER = ("UNKNOWN", "PASS", "WARN", "FAIL")

#: The tasks, with the Butler tables each writes: the per-image table first,
#: then the long table whose rows take the image's processing.
TASK_TABLES = {
    "imageQualityQa": ("iqQaMetrics", "iqQaSpeciesMetrics"),
    "dmLineResiduals": ("dmLineResidualSummary", "dmLineResidualMetrics"),
}

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
    """

    name: str
    source: str
    columns: Mapping[str, str]
    extraKeys: tuple[str, ...] = ()

    @property
    def keys(self) -> tuple[str, ...]:
        """The qadb key columns."""
        return (*KEYS, *(snakeCase(key) for key in self.extraKeys))

    @property
    def qadbColumns(self) -> tuple[str, ...]:
        """Every qadb column, keys first."""
        return (*self.keys, *(snakeCase(column) for column in self.columns))


IMAGE_QUALITY = TableSpec(
    "image_quality_qa",
    "iqQaMetrics",
    {
        "spectrograph": "int",
        "arm": "str",
        "obsType": "str",
        "seqName": "str",
        "illumination": "str",
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
        "qaStatus": "str",
        "qaDecidedBy": "str",
        "qaReason": "str",
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
        "fitted": "bool",
        "nSpecies": "int",
        "nSpeciesMeasured": "int",
        "qaStatus": "str",
        "qaDecidedBy": "str",
        "qaReason": "str",
        "recommendedAction": "str",
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
    {"value": "float", "status": "str"},
    extraKeys=("reference", "description", "statusType", "metric"),
)

#: ``qa_thresholds``: one row per thresholds file, identified by its content.
QA_THRESHOLDS = (
    "thresholds_id",
    "sha256",
    "name",
    "derived_on",
    "collection",
    "reference_runs",
    "drp_qa_version",
    "datamodel_version",
    "obs_pfs_version",
    "drp_stella_version",
    "n_entries",
    "registered_at",
)

#: Population columns of a thresholds table that ``qa_threshold_entry`` stores
#: in their own columns; any other goes into ``population`` alone.
_POPULATION_COLUMNS = (
    "arm",
    "spectrograph",
    "obsType",
    "description",
    "cameraSet",
    "reference",
    "statusType",
    "illumination",
)

#: ``qa_threshold_entry``: one row per metric and population of a file.
QA_THRESHOLD_ENTRY = (
    "thresholds_id",
    "metric",
    "population",
    *(snakeCase(column) for column in _POPULATION_COLUMNS),
    "higher_is_worse",
    "absolute",
    "method",
    "warn",
    "fail",
    "n_good",
    "n_good_visits",
    "visit_range",
    "reliable",
    "fail_bounded",
    "degenerate",
    "provenance",
)

_ENTRY_TYPES = {
    "spectrograph": "int",
    "higherIsWorse": "bool",
    "absolute": "bool",
    "warn": "float",
    "fail": "float",
    "nGood": "int",
    "nGoodVisits": "int",
    "reliable": "bool",
    "failBounded": "bool",
    "degenerate": "bool",
}

#: ``qa_gate``: the worst verdict of the tasks on an image.
QA_GATE = (
    *KEYS,
    "qa_status",
    "decided_by_task",
    "qa_decided_by",
    "qa_reason",
    "provisional",
    "mode",
    "iq_thresholds_id",
    "dm_thresholds_id",
    "decided_at",
)


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
        rows[snakeCase(key)] = _column(frame, key, "str")
    for column, dtype in spec.columns.items():
        rows[snakeCase(column)] = _column(frame, column, dtype)
    return pd.DataFrame(rows, columns=list(spec.qadbColumns), dtype=object)


def populationKey(population: Mapping[str, Any]) -> str:
    """Return a population as ``key=value`` pairs, sorted by key, nulls left out."""
    return " ".join(f"{key}={value}" for key, value in sorted(population.items()) if not pd.isna(value))


def thresholdRows(
    metadata: Mapping[str, Any],
    thresholds: pd.DataFrame,
    thresholdsId: int,
    name: str,
    sha256: str,
    registeredAt: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a thresholds file as a ``qa_thresholds`` row and its ``qa_threshold_entry`` rows.

    Parameters
    ----------
    metadata, thresholds
        From `pfs.drp.qa.metrics.calibration.readThresholds`.
    thresholdsId : `int`
        The file's ``thresholds_id``.
    name : `str`
        The file name.
    sha256 : `str`
        The hex digest of the file.
    registeredAt : `datetime.datetime`
        When it was registered.

    Returns
    -------
    header : `pandas.DataFrame`
        One row, `QA_THRESHOLDS`.
    entries : `pandas.DataFrame`
        One row per metric and population, `QA_THRESHOLD_ENTRY`.
    """
    pipeline = metadata.get("pipelineVersions") or {}
    runs = metadata.get("referenceRuns") or []
    header = pd.DataFrame(
        [
            {
                "thresholds_id": int(thresholdsId),
                "sha256": sha256,
                "name": name,
                "derived_on": _scalar(metadata.get("derivedOn"), "datetime"),
                "collection": _scalar(metadata.get("collection"), "str"),
                "reference_runs": ",".join(str(run) for run in runs) if runs else None,
                "drp_qa_version": _scalar(metadata.get("drpQaVersion"), "str"),
                "datamodel_version": pipeline.get("datamodel"),
                "obs_pfs_version": pipeline.get("obs_pfs"),
                "drp_stella_version": pipeline.get("drp_stella"),
                "n_entries": len(thresholds),
                "registered_at": registeredAt,
            }
        ],
        columns=list(QA_THRESHOLDS),
        dtype=object,
    )
    populationColumns = [c for c in thresholds.columns if c != "metric" and c not in THRESHOLD_FIELDS]
    rows = []
    for record in thresholds.to_dict("records"):
        population = {column: record[column] for column in populationColumns}
        row = {
            "thresholds_id": int(thresholdsId),
            "metric": record["metric"],
            "population": populationKey(population),
        }
        for column in _POPULATION_COLUMNS:
            row[snakeCase(column)] = _scalar(population.get(column), _ENTRY_TYPES.get(column, "str"))
        for field in THRESHOLD_FIELDS:
            row[snakeCase(field)] = _scalar(record.get(field), _ENTRY_TYPES.get(field, "str"))
        rows.append(row)
    entries = pd.DataFrame(rows, columns=list(QA_THRESHOLD_ENTRY), dtype=object)
    return header, entries


def worstStatus(statuses: Iterable[Any]) -> str:
    """Return the worst of some verdicts by `GATE_ORDER`; ``UNKNOWN`` when there is none."""
    known = [status for status in statuses if isinstance(status, str) and status in GATE_ORDER]
    return max(known, key=GATE_ORDER.index) if known else "UNKNOWN"


def gateRows(
    verdicts: Mapping[str, pd.DataFrame],
    processingId: int | Mapping[str, int],
    mode: str,
    decidedAt: Any,
    thresholdsIds: Mapping[str, int | None] | None = None,
) -> pd.DataFrame:
    """Return the gate's verdict on each image: the worst of its tasks'.

    Parameters
    ----------
    verdicts : mapping [`str`, `pandas.DataFrame`]
        The per-image table of each task in `TASK_TABLES`, by task:
        `IMAGE_KEYS`, ``qaStatus``, ``qaDecidedBy``, ``qaReason``, and
        ``qaProvisional`` where known (``run`` when ``processingId`` is a
        mapping), and ``thresholdsId``, the ``thresholds_id`` the image was
        judged with, where known. A task may be absent, or have no row for an
        image.
    processingId : `int` or mapping [`str`, `int`]
        As for `imageRows`.
    mode : `str`
        ``realtime``, ``comparison`` or ``backfill``.
    decidedAt : `datetime.datetime`
        When the gate decided.
    thresholdsIds : mapping [`str`, `int`], optional
        The ``thresholds_id`` each task judged with, by task, for a task
        whose table has no ``thresholdsId``.

    Returns
    -------
    `pandas.DataFrame`
        `QA_GATE`, one row per image in any task's table, in order of visit,
        arm and spectrograph. ``decided_by_task`` is the first task, in
        `TASK_TABLES` order, with the worst verdict; ``qa_reason`` joins every
        task's reason, each prefixed by its task.
    """
    thresholdsIds = thresholdsIds or {}
    tasks = [task for task in TASK_TABLES if task in verdicts and not verdicts[task].empty]
    keys = list(IMAGE_KEYS)
    merged = None
    for task in tasks:
        table = verdicts[task]
        columns = [
            c
            for c in ("qaStatus", "qaDecidedBy", "qaReason", "qaProvisional", "thresholdsId", "run")
            if c in table.columns
        ]
        part = table[keys + columns].rename(columns={c: f"{c}_{task}" for c in columns})
        part[f"present_{task}"] = True
        merged = part if merged is None else merged.merge(part, on=keys, how="outer")
    if merged is None:
        return pd.DataFrame(columns=list(QA_GATE), dtype=object)
    merged = merged.sort_values(keys, ignore_index=True)
    if isinstance(processingId, Mapping):
        runs = [merged[f"run_{task}"] for task in tasks if f"run_{task}" in merged.columns]
        if not runs:
            raise ValueError("A mapping of processing ids by run needs a 'run' column")
        run = runs[0]
        for other in runs[1:]:
            run = run.combine_first(other)
        merged["run"] = run

    def column(task: str, name: str) -> pd.Series:
        label = f"{name}_{task}"
        return (
            merged[label] if label in merged.columns else pd.Series([None] * len(merged), index=merged.index)
        )

    statuses = {task: column(task, "qaStatus") for task in tasks}
    present = {task: merged[f"present_{task}"].fillna(False).astype(bool) for task in tasks}
    worst, decidedByTask, decidedBy, reasons, provisional = [], [], [], [], []
    for index in merged.index:
        # A task's row with no verdict counts as UNKNOWN; a task with no row at all does not count.
        rowStatuses = {
            task: (statuses[task][index] if statuses[task][index] in GATE_ORDER else "UNKNOWN")
            for task in tasks
            if present[task][index]
        }
        status = worstStatus(rowStatuses.values())
        decider = next((task for task in tasks if rowStatuses.get(task) == status), None)
        worst.append(status)
        decidedByTask.append(decider)
        decidedBy.append(_scalar(column(decider, "qaDecidedBy")[index], "str") if decider else None)
        parts = []
        for task in tasks:
            reason = _scalar(column(task, "qaReason")[index], "str")
            if reason:
                parts.append(f"{task}: {reason}")
        reasons.append("; ".join(parts))
        flags = [_scalar(column(task, "qaProvisional")[index], "bool") for task in tasks]
        flags = [flag for flag in flags if flag is not None]
        provisional.append(any(flags) if flags else None)

    def thresholdsId(task: str) -> list[int | None]:
        """Return the thresholds a task judged each image with; a task's apply only to the images it judged."""
        if task not in present:
            return [None] * len(merged)
        ids = (
            [_scalar(value, "int") for value in column(task, "thresholdsId")]
            if f"thresholdsId_{task}" in merged.columns
            else [thresholdsIds.get(task)] * len(merged)
        )
        return [found if judged else None for found, judged in zip(ids, present[task], strict=True)]

    return pd.DataFrame(
        {
            "processing_id": _processingIds(merged, processingId),
            "pfs_visit_id": _column(merged, "visit", "int"),
            "camera_name": [
                cameraName(a, s) for a, s in zip(merged["arm"], merged["spectrograph"], strict=True)
            ],
            "qa_status": worst,
            "decided_by_task": decidedByTask,
            "qa_decided_by": decidedBy,
            "qa_reason": reasons,
            "provisional": provisional,
            "mode": [mode] * len(merged),
            "iq_thresholds_id": thresholdsId("imageQualityQa"),
            "dm_thresholds_id": thresholdsId("dmLineResiduals"),
            "decided_at": [decidedAt] * len(merged),
        },
        columns=list(QA_GATE),
        dtype=object,
    )
