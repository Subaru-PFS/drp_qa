"""Build the dashboard's Tier 1 table: one long-format Parquet file per collection.

The dashboard reads only this file (with DuckDB), never a Butler (rebuild plan
5.1). It is made by a post-run job from the per-detector tables
`pfs.drp.qa.metrics.readers.exportMetrics` writes, so every number in it was
stored by the pipeline or judged by `pfs.drp.qa.metrics.gate`: the dashboard
derives none (5.7).

One row per (image, source, metric, population), in `TIER1_COLUMNS`. An image's
verdict from each source is a row too, with ``metric`` `VERDICT`. The thresholds
file, collections and pipeline versions go in the Parquet key-value metadata
under `PROVENANCE_KEY` (5.5), so a reader can show "gated with thresholds v2".
"""

import ast
import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from pfs.drp.qa.dashboard.schema import (
    IMAGE_STATUS_ORDER,
    NOTE,
    PROVENANCE_KEY,
    STATUS_ORDER,
    TIER1_COLUMNS,
    VERDICT,
)
from pfs.drp.qa.metrics.gate import ThresholdsLike, configThresholds, judge, loadThresholds, thresholdsPath
from pfs.drp.qa.metrics.lineResiduals import residualThresholds

__all__ = [
    "IMAGE_STATUS_ORDER",
    "TIER1_COLUMNS",
    "VERDICT",
    "buildTier1",
    "configDefaults",
    "inferSequences",
    "noteRows",
    "rollupStatus",
    "taskThresholds",
    "thresholdsProvenance",
    "verdictDisagreements",
    "writeTier1",
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


def rollupStatus(statuses) -> str:
    """Return the worst of some statuses; ``UNKNOWN`` when none was judged.

    Empty strings, NaN and ``UNKNOWN`` count as not judged.
    """
    judged = [s for s in statuses if isinstance(s, str) and s in STATUS_ORDER]
    return max(judged, key=STATUS_ORDER.index) if judged else "UNKNOWN"


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
    visits["sequenceKey"] = -newSequence.cumsum().astype("int64")
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
    """Return opdb's notes as Tier 1 rows: ``metric`` `NOTE`, the text in ``reason``.

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


def buildTier1(
    iqQaMetrics: pd.DataFrame,
    thresholds: ThresholdsLike | Sequence[ThresholdsLike],
    dmLineResidualSummary: pd.DataFrame | None = None,
    dmLineResidualMetrics: pd.DataFrame | None = None,
    visitTimes: pd.DataFrame | None = None,
    lineThresholds: Sequence[ThresholdsLike] | None = None,
    notes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the Tier 1 table for one collection.

    Parameters
    ----------
    iqQaMetrics : `pandas.DataFrame`
        The collection's ``iqQaMetrics`` rows, one per image.
    thresholds : path, `pandas.DataFrame`, or a sequence of them
        The thresholds the images are judged with, highest priority first, as
        `pfs.drp.qa.metrics.gate.judge` takes them: each metric's verdict and
        limits are recorded. The stored verdict of each image is kept as it
        is; `verdictDisagreements` finds where the two differ.
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
        `TIER1_COLUMNS`, sorted by visit, arm, spectrograph and source.
    """
    parts = [_iqRows(iqQaMetrics, thresholds)]
    if dmLineResidualSummary is not None and dmLineResidualMetrics is not None:
        parts.append(_lineRows(dmLineResidualSummary, dmLineResidualMetrics, lineThresholds))
    if notes is not None:
        parts.append(noteRows(notes, visitTimes))
    table = pd.concat(parts, ignore_index=True)
    images = iqQaMetrics[["visit", "arm", "spectrograph", "obsType", "seqName"]].drop_duplicates(
        ["visit", "arm", "spectrograph"]
    )
    table = table.drop(columns=["obsType", "seqName"], errors="ignore").merge(
        images, on=["visit", "arm", "spectrograph"], how="left"
    )
    table = table.merge(inferSequences(images, visitTimes), on="visit", how="left")
    for column, dtype in TIER1_COLUMNS.items():
        if column not in table:
            table[column] = np.nan if dtype in ("float64",) else ""
        if dtype == "string":
            table[column] = table[column].fillna("").astype(str)
    table = table[list(TIER1_COLUMNS)].astype(
        {k: v for k, v in TIER1_COLUMNS.items() if v not in ("string", "datetime64[ns]")}
    )
    return table.sort_values(["visit", "arm", "spectrograph", "source"], kind="stable", ignore_index=True)


def writeTier1(table: pd.DataFrame, path: Path | str, provenance: Mapping) -> Path:
    """Write the Tier 1 table with its provenance in the Parquet metadata.

    Parameters
    ----------
    table : `pandas.DataFrame`
        From `buildTier1`.
    path : path
        The file to write.
    provenance : mapping
        JSON-serializable: at least ``collection`` and ``thresholds``; see
        `thresholdsProvenance`.

    Returns
    -------
    `pathlib.Path`
        ``path``.
    """
    path = Path(path)
    arrow = pa.Table.from_pandas(table, preserve_index=False)
    metadata = dict(arrow.schema.metadata or {})
    metadata[PROVENANCE_KEY] = json.dumps(provenance, default=str).encode()
    pq.write_table(arrow.replace_schema_metadata(metadata), path, compression="zstd")
    return path


def thresholdsProvenance(thresholds: Path | str) -> dict:
    """Return what a thresholds file says about itself, for `writeTier1`."""
    header = yaml.safe_load(Path(thresholds).read_text())
    keep = ("version", "derivedOn", "collection", "referenceRuns", "drpQaVersion", "pipelineVersions")
    return {"file": Path(thresholds).name, "nEntries": len(header.get("thresholds", []))} | {
        k: header.get(k) for k in keep
    }


def verdictDisagreements(table: pd.DataFrame, source: str = "iqQaMetrics") -> pd.DataFrame:
    """Return the images whose stored verdict differs from the worst per-metric verdict.

    A difference means the images were gated with other thresholds than the
    ones the table was built with: an older thresholds file, or a layer (the
    task config) missing from them.

    Parameters
    ----------
    table : `pandas.DataFrame`
        From `buildTier1`.
    source : `str`, optional
        The source to compare.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph``, ``stored``, ``rejudged`` and the
        stored ``reason``, one row per image that differs.
    """
    rows = table[table["source"] == source]
    keys = ["visit", "arm", "spectrograph"]
    stored = rows[rows["metric"] == VERDICT].set_index(keys)[["status", "reason"]]
    rejudged = rows[rows["metric"] != VERDICT].groupby(keys)["status"].agg(rollupStatus)
    both = stored.rename(columns={"status": "stored"}).join(rejudged.rename("rejudged"), how="left")
    both["rejudged"] = both["rejudged"].fillna("UNKNOWN")
    return both[both["stored"] != both["rejudged"]].reset_index()[[*keys, "stored", "rejudged", "reason"]]


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
        files, highest priority first, for `buildTier1`.
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
        files = [thresholdsPath(config.thresholdsFile)] if getattr(config, "thresholdsFile", "") else []
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
