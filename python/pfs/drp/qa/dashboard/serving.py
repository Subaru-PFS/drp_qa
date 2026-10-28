"""Build the dashboard's metrics table: one long-format table per collection.

The dashboard reads only this table (with DuckDB), never a Butler. It is built
from the QA tasks' per-detector tables as qadb holds them
(`qadbSource.readQadb`), so every number in it was stored by the pipeline or
judged by `pfs.drp.qa.metrics.gate`: the dashboard derives none.

One row per (image, source, metric, population), in `METRICS_COLUMNS`. An image's
verdict from each source is a row too, with ``metric`` `VERDICT`. The thresholds
file and drp_qa version go in the store's provenance (`thresholdsProvenance`),
so a reader can show "gated with thresholds v2".
"""

import ast
import importlib.util
import math
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from pfs.drp.qa.dashboard.schema import (
    IMAGE_STATUS_ORDER,
    METRICS_COLUMNS,
    NOTE,
    VERDICT,
)
from pfs.drp.qa.metrics.gate import (
    ThresholdsLike,
    configThresholds,
    judge,
    loadThresholds,
    thresholdsNames,
    thresholdsPath,
)
from pfs.drp.qa.metrics.lineResiduals import residualThresholds

__all__ = [
    "IMAGE_STATUS_ORDER",
    "METRICS_COLUMNS",
    "VERDICT",
    "buildMetricsTable",
    "configDefaults",
    "firstThresholdsFile",
    "inferSequences",
    "noteRows",
    "taskThresholds",
    "thresholdsProvenance",
]

#: iqQaMetrics columns kept as values when no threshold judges them.
_IQ_EXTRA = (
    "medDxCenter",
    "dxCenterRms",
    "pctFlagged",
    "nLines",
    "nLitFibers",
    "cosmicRayCount",
    "fitXRms",
    "fitYRms",
)

#: Visits further apart than this, or of another seqName, start a new inferred sequence.
_SEQUENCE_GAP = 3


def inferSequences(images: pd.DataFrame, visitTimes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return each visit's sequence and start time.

    Parameters
    ----------
    images : `pandas.DataFrame`
        Rows with ``visit`` and ``seqName``.
    visitTimes : `pandas.DataFrame`, optional
        opdb's visit listing (`pfs.drp.qa.comparison.queries.readVisitListing`
        columns ``pfs_visit_id``, ``time_exp_start``, ``iic_sequence_id``).

    Returns
    -------
    `pandas.DataFrame`
        One row per visit: ``visit``, ``sequenceKey``, ``sequenceSource``
        (``"opdb"``, or ``"inferred"`` when the opdb listing is absent or
        lacks the visit) and ``timeExpStart`` (NaT without the listing).
        An inferred sequence is a run of visits with one ``seqName``, none
        more than a few visit numbers after the last.
    """
    visits = images.groupby("visit", sort=True)["seqName"].first().reset_index()
    newSequence = (visits["seqName"] != visits["seqName"].shift()) | (visits["visit"].diff() > _SEQUENCE_GAP)
    # Negative, so an inferred key never collides with an iic_sequence_id.
    # As plain booleans: a pyarrow-backed seqName compares to bool[pyarrow], which has no cumsum, and the
    # first visit, against a missing one, to NA: it starts a sequence.
    visits["sequenceKey"] = -np.cumsum(newSequence.to_numpy(dtype=bool, na_value=True)).astype("int64")
    visits["sequenceSource"] = "inferred"
    visits["timeExpStart"] = pd.NaT
    if visitTimes is not None and not visitTimes.empty:
        listing = visitTimes.set_index("pfs_visit_id")
        known = visits["visit"].isin(listing.index)
        ids = listing["iic_sequence_id"].reindex(visits["visit"]).to_numpy()
        fromOpdb = known & pd.notna(ids)
        visits.loc[fromOpdb, "sequenceKey"] = ids[fromOpdb.to_numpy()].astype("int64")
        visits.loc[fromOpdb, "sequenceSource"] = "opdb"
        times = pd.to_datetime(listing["time_exp_start"]).reindex(visits["visit"]).to_numpy()
        visits["timeExpStart"] = times
    return visits.drop(columns="seqName")


def _iqRows(iq: pd.DataFrame, thresholds: ThresholdsLike | Sequence[ThresholdsLike]) -> pd.DataFrame:
    """Per-metric rows of ``iqQaMetrics``, judged by ``thresholds``, plus its stored verdicts."""
    judged = judge(iq, thresholds)
    judged = judged.assign(
        source="iqQaMetrics",
        description="",
        reference="",
        statusType="",
        run=iq["calexpRun"].to_numpy()[judged["row"]] if "calexpRun" in iq else "",
    )
    judgedNames = set(judged["metric"])
    extras = [name for name in _IQ_EXTRA if name in iq.columns and name not in judgedNames]
    plain = iq[["visit", "arm", "spectrograph", *extras]].melt(
        id_vars=["visit", "arm", "spectrograph"], var_name="metric", value_name="value"
    )
    plain = plain.assign(source="iqQaMetrics", status="", population="", reason="")
    verdicts = iq[["visit", "arm", "spectrograph", "qaStatus", "qaReason"]].rename(
        columns={"qaStatus": "status", "qaReason": "reason"}
    )
    verdicts = verdicts.assign(source="iqQaMetrics", metric=VERDICT, value=np.nan)
    if "calexpRun" in iq:
        plain["run"] = np.tile(iq["calexpRun"].to_numpy(), len(extras))
        verdicts["run"] = iq["calexpRun"].to_numpy()
    return pd.concat(
        [judged.drop(columns=["row", "layer", "provenance"]), plain, verdicts], ignore_index=True
    )


def noteRows(notes: pd.DataFrame, visitTimes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return opdb's notes as metrics-table rows: ``metric`` `NOTE`, the text in ``reason``.

    A note on a visit applies to all its cameras (``arm`` empty, ``spectrograph`` 0); an
    ``sps_annotation`` to its camera; a sequence's note to each of its visits in ``visitTimes``.

    Parameters
    ----------
    notes : `pandas.DataFrame`
        `pfs.drp.qa.comparison.queries.readNotes`.
    visitTimes : `pandas.DataFrame`, optional
        opdb's visit listing, mapping sequences to visits.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph``, ``source`` (``opdb:`` and the
        note's source), ``metric``, ``value``, ``status``, ``reason``.
    """
    onVisits = notes[notes["pfs_visit_id"].notna()].rename(columns={"pfs_visit_id": "visit"})
    onSequences = notes[notes["pfs_visit_id"].isna() & notes["iic_sequence_id"].notna()]
    if visitTimes is not None and not onSequences.empty:
        members = visitTimes[["pfs_visit_id", "iic_sequence_id"]].dropna().astype("int64")
        expanded = onSequences.drop(columns="pfs_visit_id").merge(
            members.astype({"iic_sequence_id": "Int64"}), on="iic_sequence_id"
        )
        onVisits = pd.concat(
            [onVisits, expanded.rename(columns={"pfs_visit_id": "visit"})], ignore_index=True
        )
    camera = onVisits["camera"].fillna("").astype(str)
    return pd.DataFrame(
        {
            "visit": onVisits["visit"].astype("int64").to_numpy(),
            "arm": camera.str[:1].to_numpy(),
            "spectrograph": pd.to_numeric(camera.str[1:], errors="coerce")
            .fillna(0)
            .astype("int64")
            .to_numpy(),
            "source": ("opdb:" + onVisits["source"].astype(str)).to_numpy(),
            "metric": NOTE,
            "value": np.nan,
            "status": "",
            "reason": onVisits["note"].astype(str).str.strip().to_numpy(),
        }
    )


def _lineLimits(metrics: pd.DataFrame, thresholds: Sequence[ThresholdsLike]) -> pd.DataFrame:
    """Return ``warn``, ``fail`` and ``population`` for each ``dmLineResidualMetrics`` row.

    The task stores each row's verdict but not the thresholds behind it; these are what
    `pfs.drp.qa.metrics.gate.judge` finds for the row in ``thresholds``, highest priority first.
    """
    limits = pd.DataFrame({"warn": np.nan, "fail": np.nan, "population": ""}, index=metrics.index)
    tables = loadThresholds(thresholds)
    for name, rows in metrics.groupby("metric", sort=False):
        named = [table[table["metric"] == name] for table in tables]
        if all(table.empty for table in named):
            continue
        wide = rows.drop(columns=["metric", "value", "status"]).assign(**{name: rows["value"].to_numpy()})
        judged = judge(wide.reset_index(drop=True), named)
        limits.loc[rows.index, ["warn", "fail", "population"]] = judged[
            ["warn", "fail", "population"]
        ].to_numpy()
    return limits


def _lineRows(
    summary: pd.DataFrame, metrics: pd.DataFrame, thresholds: Sequence[ThresholdsLike] | None = None
) -> pd.DataFrame:
    """Rows of ``dmLineResidualMetrics`` as stored, with their thresholds, and ``dmLineResidualSummary``'s verdicts."""
    rows = metrics.assign(source="dmLineResidualMetrics", population="", reason="")
    if thresholds:
        rows[["warn", "fail", "population"]] = _lineLimits(metrics, thresholds)
    verdicts = summary[["visit", "arm", "spectrograph", "qaStatus", "qaReason", "recommendedAction"]]
    verdicts = verdicts.rename(columns={"qaStatus": "status"})
    verdicts = verdicts.assign(
        source="dmLineResidualSummary",
        metric=VERDICT,
        value=np.nan,
        reason=[
            "; ".join(str(part) for part in parts if isinstance(part, str) and part)
            for parts in zip(verdicts["qaReason"], verdicts["recommendedAction"], strict=True)
        ],
    ).drop(columns=["qaReason", "recommendedAction"])
    return pd.concat([rows.drop(columns=["cameraSet"], errors="ignore"), verdicts], ignore_index=True)


def buildMetricsTable(
    iqQaMetrics: pd.DataFrame,
    thresholds: ThresholdsLike | Sequence[ThresholdsLike],
    dmLineResidualSummary: pd.DataFrame | None = None,
    dmLineResidualMetrics: pd.DataFrame | None = None,
    visitTimes: pd.DataFrame | None = None,
    lineThresholds: Sequence[ThresholdsLike] | None = None,
    notes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the metrics table for one collection.

    Parameters
    ----------
    iqQaMetrics : `pandas.DataFrame`
        The collection's ``iqQaMetrics`` rows, one per image.
    thresholds : path, `pandas.DataFrame`, or a sequence of them
        The thresholds the images are judged with, highest priority first, as
        `pfs.drp.qa.metrics.gate.judge` takes them: each metric's verdict and
        limits are recorded. The stored verdict of each image is kept as it
        is.
    dmLineResidualSummary, dmLineResidualMetrics : `pandas.DataFrame`, optional
        The same collection's ``dmLineResiduals`` outputs.
    visitTimes : `pandas.DataFrame`, optional
        opdb's visit listing, for times and sequences (`inferSequences`).
    lineThresholds : sequence, optional
        The thresholds ``dmLineResiduals`` judges with, highest priority
        first, to record beside each stored line-residual verdict.
    notes : `pandas.DataFrame`, optional
        opdb's notes (`pfs.drp.qa.comparison.queries.readNotes`), recorded
        as `noteRows`; a sequence's notes need ``visitTimes``.

    Returns
    -------
    `pandas.DataFrame`
        `METRICS_COLUMNS`, sorted by visit, arm, spectrograph and source.
        Only visits in ``iqQaMetrics``: rows on any other visit (a note on a
        visit that was not judged) are left out.
    """
    parts = [_iqRows(iqQaMetrics, thresholds)]
    if dmLineResidualSummary is not None and dmLineResidualMetrics is not None:
        parts.append(_lineRows(dmLineResidualSummary, dmLineResidualMetrics, lineThresholds))
    if notes is not None:
        parts.append(noteRows(notes, visitTimes))
    table = pd.concat(parts, ignore_index=True)
    # A visit without an iqQaMetrics row has no sequence to file its rows under.
    table = table[table["visit"].isin(iqQaMetrics["visit"])]
    images = iqQaMetrics[["visit", "arm", "spectrograph", "obsType", "seqName"]].drop_duplicates(
        ["visit", "arm", "spectrograph"]
    )
    table = table.drop(columns=["obsType", "seqName"], errors="ignore").merge(
        images, on=["visit", "arm", "spectrograph"], how="left"
    )
    table = table.merge(inferSequences(images, visitTimes), on="visit", how="left")
    for column, dtype in METRICS_COLUMNS.items():
        if column not in table:
            table[column] = np.nan if dtype in ("float64",) else ""
        if dtype == "string":
            table[column] = table[column].fillna("").astype(str)
    table = table[list(METRICS_COLUMNS)].astype(
        {k: v for k, v in METRICS_COLUMNS.items() if v not in ("string", "datetime64[ns]")}
    )
    return table.sort_values(["visit", "arm", "spectrograph", "source"], kind="stable", ignore_index=True)


def thresholdsProvenance(thresholds: Path | str) -> dict:
    """Return what a thresholds file says about itself, for a store's provenance."""
    header = yaml.safe_load(Path(thresholds).read_text())
    keep = ("version", "derivedOn", "collection", "referenceRuns", "drpQaVersion", "pipelineVersions")
    return {"file": Path(thresholds).name, "nEntries": len(header.get("thresholds", []))} | {
        k: header.get(k) for k in keep
    }


def firstThresholdsFile(layers: Sequence[ThresholdsLike]) -> Path:
    """Return the highest-priority thresholds file among ``layers`` (`taskThresholds`), for the provenance."""
    try:
        return next(Path(layer) for layer in layers if isinstance(layer, Path | str))
    except StopIteration:
        raise ValueError("No thresholds file among the layers: only config fields") from None


def configDefaults(module: str, configClass: str) -> SimpleNamespace:
    """Return the defaults of a task config's fields, read from its source without importing it.

    Stack-free: a config class derives from ``lsst.pex.config``, so it cannot be
    imported here, but its ``Field(default=...)`` literals are the task's
    defaults. With the stack, use the config class itself.

    Parameters
    ----------
    module : `str`
        The module, e.g. ``"pfs.drp.qa.imageQualityQa"``.
    configClass : `str`
        The class, e.g. ``"ImageQualityQaConfig"``.

    Returns
    -------
    `types.SimpleNamespace`
        One attribute per field whose default is a literal (``nan`` included).
    """
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise ImportError(f"No module {module}")
    tree = ast.parse(Path(spec.origin).read_text())
    node = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == configClass),
        None,
    )
    if node is None:
        raise LookupError(f"No class {configClass} in {spec.origin}")
    defaults = {}
    for statement in node.body:
        if not (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)):
            continue
        default = next((k.value for k in statement.value.keywords if k.arg == "default"), None)
        if default is None:
            continue
        try:
            value = ast.literal_eval(default)
        except ValueError:
            text = ast.unparse(default)
            if text not in ("np.nan", "math.nan", "float('nan')", 'float("nan")'):
                continue
            value = math.nan
        for target in statement.targets:
            if isinstance(target, ast.Name):
                defaults[target.id] = value
    return SimpleNamespace(**defaults)


def taskThresholds() -> tuple[dict[str, list], dict[str, str]]:
    """Return the thresholds ``imageQualityQa`` and ``dmLineResiduals`` judge with, at their defaults.

    The same layers the tasks build: the thresholds file their config names,
    then their config fields. Uses the config classes when the stack is set
    up, else their defaults read from source (`configDefaults`).

    Returns
    -------
    layers : `dict` [`str`, `list`]
        ``"iqQaMetrics"`` and ``"dmLineResidualMetrics"``: threshold tables or
        files, highest priority first, for `buildMetricsTable`.
    sources : `dict` [`str`, `str`]
        How each was found, for the provenance.
    """
    configs, sources = {}, {}
    for key, module, name in (
        ("iqQaMetrics", "pfs.drp.qa.imageQualityQa", "ImageQualityQaConfig"),
        ("dmLineResidualMetrics", "pfs.drp.qa.dmLineResiduals", "DmLineResidualsConfig"),
    ):
        try:
            configs[key] = getattr(importlib.import_module(module), name)()
            sources[key] = f"{name}()"
        except ImportError:
            configs[key] = configDefaults(module, name)
            sources[key] = f"{name} defaults, read from source"
    layers = {}
    for key, config in configs.items():
        # The task's own layers: thresholdsFile, then extraThresholdsFiles (the IIS file, PIPE2D-1954).
        files = [thresholdsPath(name) for name in thresholdsNames(config)]
        sources[key] += "; " + " + ".join([*(path.name for path in files), "config fields"])
        if key == "iqQaMetrics":
            layers[key] = [*files, configThresholds(config)]
        else:
            layers[key] = [
                *files,
                residualThresholds(
                    config.xWarnThreshold,
                    config.xFailThreshold,
                    config.yWarnThreshold,
                    config.yFailThreshold,
                    config.xRmsWarnThreshold,
                    config.xRmsFailThreshold,
                    config.yRmsWarnThreshold,
                    config.yRmsFailThreshold,
                ),
            ]
    return layers, sources
