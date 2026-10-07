"""Plan a comparison: what is judged, what still needs work, and the ``pipetask`` runs that do it.

Coverage is per detector image: the cameras opdb says took an exposure, joined with what the
Butler holds (`pfs.drp.qa.comparison.butlerQueries.detectorHoldings`). Visits the gate doesn't
judge are counted, not expanded.

The reductions are drp_stella's ``reduceExposure`` with the configuration ``drpActor`` uses for
the sequence type (`drpActorConfig`), and ``cosmicray`` combines the exposures of one sequence
only, as ``drpActor``, which reduces one sequence at a time. Reductions already in
``drpActor/reductions`` and verdicts already in the output collection are reused through
``--skip-existing-in``, so re-running a plan after more nights only adds quanta.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from pfs.drp.qa.metrics.validationVisits import visitExpression

__all__ = [
    "STATUS_ORDER",
    "Pass",
    "coverage",
    "drpActorConfig",
    "expectedDetectors",
    "outputCollection",
    "passes",
    "pipetaskCommand",
    "summarize",
]

#: Coverage of a judged visit's detector, from done to blocked.
STATUS_ORDER = ("judged", "to judge", "to reduce", "no raw", "raw not in opdb")

_CAMERA_RE = re.compile(r"^([brnm])([1-4])$")


def drpActorConfig(sequenceType: str) -> dict[str, bool]:
    """Return the configuration ``drpActor`` reduces a sequence type with.

    As ``ics_drpActor``'s ``Engine.newVisitGroup``.

    Parameters
    ----------
    sequenceType : `str`
        The IIC sequence type.

    Returns
    -------
    `dict` [`str`, `bool`]
        ``pipetask -c`` overrides, ``label:field`` to value.
    """
    return {
        "reduceExposure:requireAdjustDetectorMap": sequenceType == "scienceObject",
        "isr:h4.quickCDS": sequenceType not in ("scienceObject", "masterDarks"),
        "cosmicray:doNormalizeChiRms": sequenceType != "darks",
    }


def expectedDetectors(visits: pd.DataFrame) -> pd.DataFrame:
    """Expand each visit into the detectors opdb says took an exposure.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        ``pfs_visit_id`` and ``cameras`` (``b1,r1,n1``).

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm`` and ``spectrograph``.
    """
    rows = []
    for visit, cameras in zip(visits["pfs_visit_id"], visits["cameras"].fillna(""), strict=True):
        for camera in str(cameras).split(","):
            match = _CAMERA_RE.match(camera.strip())
            if match:
                rows.append((int(visit), match.group(1), int(match.group(2))))
    return pd.DataFrame(rows, columns=["visit", "arm", "spectrograph"])


def coverage(visits: pd.DataFrame, holdings: pd.DataFrame) -> pd.DataFrame:
    """Return the status of each detector image of the judged visits.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.classify.classifyVisits`.
    holdings : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.butlerQueries.detectorHoldings`.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph``, the visit's ``period``,
        ``sequence_type``, ``iic_sequence_id``, ``category`` and ``night``,
        and ``status``, one of `STATUS_ORDER`: ``raw not in opdb`` for a
        detector with raw data that opdb didn't list.
    """
    keys = ["visit", "arm", "spectrograph"]
    judged = visits[visits["judged"]]
    expected = expectedDetectors(judged).assign(expected=True)
    held = holdings[holdings["visit"].isin(judged["pfs_visit_id"])]
    detectors = expected.merge(held, on=keys, how="outer")
    for column in ("expected", "raw", "reduced", "judged"):
        detectors[column] = detectors[column].astype("boolean").fillna(False).astype(bool)

    status = pd.Series("to reduce", index=detectors.index, dtype=object)
    status[detectors["reduced"]] = "to judge"
    status[detectors["judged"]] = "judged"
    status[~detectors["raw"] & ~detectors["judged"]] = "no raw"
    status[~detectors["expected"]] = "raw not in opdb"
    detectors["status"] = status

    info = judged[["pfs_visit_id", "period", "sequence_type", "iic_sequence_id", "category", "night"]]
    detectors = detectors.merge(info.rename(columns={"pfs_visit_id": "visit"}), on="visit", how="left")
    columns = [*keys, "period", "sequence_type", "iic_sequence_id", "category", "night", "status"]
    return detectors[columns].sort_values(keys, ignore_index=True)


def summarize(visits: pd.DataFrame, detectors: pd.DataFrame) -> pd.DataFrame:
    """Count a period's visits and detector images by what happens to them.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.classify.classifyVisits`.
    detectors : `pandas.DataFrame`
        From `coverage`.

    Returns
    -------
    `pandas.DataFrame`
        One row per ``sequence_type`` and ``category``: the number of
        ``visits``, the ``reason`` they aren't judged (or ``judged``), and one
        column per `STATUS_ORDER` counting the detector images.
    """
    counts = (
        visits.assign(sequence_type=visits["sequence_type"].fillna("(none)"))
        .groupby(["sequence_type", "category", "reason"], dropna=False)
        .size()
        .rename("visits")
        .reset_index()
    )
    if detectors.empty:
        images = pd.DataFrame(columns=["sequence_type", "category", *STATUS_ORDER])
    else:
        images = (
            detectors.pivot_table(
                index=["sequence_type", "category"],
                columns="status",
                values="visit",
                aggfunc="size",
                fill_value=0,
            )
            .reindex(columns=list(STATUS_ORDER), fill_value=0)
            .reset_index()
        )
        images.columns.name = None
    summary = counts.merge(images, on=["sequence_type", "category"], how="left")
    for status in STATUS_ORDER:
        summary[status] = summary[status].fillna(0).astype(int)
    summary.loc[summary["reason"] != "judged", list(STATUS_ORDER)] = 0
    return summary.sort_values(["reason", "sequence_type", "category"], ignore_index=True)


@dataclass(frozen=True)
class Pass:
    """One ``pipetask run``: visits sharing ``drpActor``'s configuration.

    Parameters
    ----------
    name : `str`
        A label, e.g. ``scienceArc+scienceTrace``.
    visits : `tuple` [`int`, ...]
        The visits.
    config : `dict` [`str`, `bool`]
        ``pipetask -c`` overrides, from `drpActorConfig`.
    groups : `dict` [`int`, `int`]
        Visit to cosmic-ray group, the group being the first visit of its
        sequence: ``cosmicray``'s ``groups`` with ``grouping="manual"``.
    """

    name: str
    visits: tuple[int, ...]
    config: dict[str, bool] = field(hash=False)
    groups: dict[int, int] = field(hash=False)

    def cosmicrayConfig(self) -> str:
        """Return the ``cosmicray`` config file that groups by sequence."""
        return f'config.grouping = "manual"\nconfig.groups = {self.groups!r}\n'


def passes(visits: pd.DataFrame, detectors: pd.DataFrame) -> list[Pass]:
    """Group the visits with work left into one `Pass` per configuration.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.classify.classifyVisits`.
    detectors : `pandas.DataFrame`
        From `coverage`; a visit with any detector ``to reduce`` or ``to
        judge`` has work left.

    Returns
    -------
    `list` [`Pass`]
        Ordered by name.
    """
    pending = set(detectors.loc[detectors["status"].isin(["to reduce", "to judge"]), "visit"])
    todo = visits[visits["pfs_visit_id"].isin(pending)]
    if todo.empty:
        return []

    sequenceFirst = todo.groupby("iic_sequence_id")["pfs_visit_id"].transform("min")
    todo = todo.assign(group=sequenceFirst.astype(int), configKey=todo["sequence_type"].map(_configKey))
    result = []
    for _, group in todo.groupby("configKey"):
        types = sorted(group["sequence_type"].unique())
        result.append(
            Pass(
                name="+".join(types),
                visits=tuple(sorted(int(visit) for visit in group["pfs_visit_id"])),
                config=drpActorConfig(types[0]),
                groups={
                    int(visit): int(first)
                    for visit, first in sorted(zip(group["pfs_visit_id"], group["group"], strict=True))
                },
            )
        )
    return sorted(result, key=lambda item: item.name)


def outputCollection(prefix: str, period: str, version: str) -> str:
    """Return the output collection of a period's comparison.

    Parameters
    ----------
    prefix : `str`
        E.g. ``u/someone/comparison``.
    period : `str`
        E.g. ``run30`` or ``run30-pre``.
    version : `str`
        The drp_qa version, e.g. from ``git describe``.

    Returns
    -------
    `str`
        ``<prefix>/<period>/<version>``.

    Raises
    ------
    ValueError
        If the version marks a modified tree (``-dirty``), whose results no
        version could reproduce.
    """
    if version.endswith("-dirty"):
        raise ValueError(f"drp_qa version {version!r} has uncommitted changes: commit them first")
    return f"{prefix.rstrip('/')}/{period}/{version}"


def pipetaskCommand(
    item: Pass,
    *,
    butler: str,
    pipeline: str,
    inputs: Sequence[str],
    output: str,
    skipExistingIn: Sequence[str],
    cosmicrayConfigFile: str,
    jobs: int = 8,
    instrument: str = "PFS",
) -> list[str]:
    """Return the ``pipetask run`` command of a pass, as a list.

    Parameters
    ----------
    item : `Pass`
        The pass.
    butler : `str`
        The Butler repository.
    pipeline : `str`
        The pipeline, normally ``$DRP_QA_DIR/pipelines/qaThresholds.yaml``.
    inputs : sequence of `str`
        Input collections, e.g. ``["drpActor/reductions", "PFS/defaults"]``.
    output : `str`
        The output collection, from `outputCollection`.
    skipExistingIn : sequence of `str`
        Collections whose outputs are reused, e.g. ``drpActor/reductions``
        and, once it exists, ``output``.
    cosmicrayConfigFile : `str`
        Where `Pass.cosmicrayConfig` was written.
    jobs : `int`, optional
        Parallel processes.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.

    Returns
    -------
    `list` [`str`]
        The command, one argument per element.
    """
    command = ["pipetask", "--long-log", "--log-level", "PFS=INFO", "run", "-j", str(jobs)]
    command += ["-b", butler, "-p", pipeline, "-i", ",".join(inputs), "-o", output]
    if skipExistingIn:
        command += ["--skip-existing-in", ",".join(skipExistingIn)]
    for key, value in item.config.items():
        command += ["-c", f"{key}={value}"]
    command += ["-C", f"cosmicray:{cosmicrayConfigFile}"]
    command += ["-d", f"instrument = '{instrument}' AND {visitExpression(item.visits)}"]
    return command


def _configKey(sequenceType: str) -> tuple:
    """Return a hashable key of `drpActorConfig`."""
    return tuple(sorted(drpActorConfig(sequenceType).items()))
