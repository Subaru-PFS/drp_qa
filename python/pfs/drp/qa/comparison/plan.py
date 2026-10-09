"""Plan a comparison: what is judged, what still needs work, and the ``pipetask`` runs that do it.

Coverage is per detector image: the cameras opdb says took an exposure, joined with what the
Butler holds (`pfs.drp.qa.comparison.butlerQueries.detectorHoldings`). Visits the gate doesn't
judge are counted, not expanded.

The reductions are drp_stella's ``reduceExposure`` with the configuration ``drpActor`` uses for
the sequence type (`drpActorConfig`), and ``cosmicray`` combines the exposures of one sequence
only, as ``drpActor``, which reduces one sequence at a time. Reductions already in
``drpActor/reductions`` and verdicts already in the output collection are reused through
``--skip-existing-in``, so re-running a plan after more nights only adds quanta. Detector images
whose reduction failed (`failedQuanta`, read from the ``pipetask`` logs) are ``failed``, not
rescheduled: the same exposure fails the same way every time.
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
    "failedQuanta",
    "outputCollection",
    "passes",
    "pipetaskCommand",
    "summarize",
]

#: Coverage of a judged visit's detector, from done to blocked.
STATUS_ORDER = ("judged", "to judge", "to reduce", "failed", "no pfsConfig", "no raw", "raw not in opdb")

#: A failed quantum in a ``pipetask`` log.
_FAILED_RE = re.compile(r"Execution of task '(\w+)' on quantum \{([^}]*)\} failed\. Exception (.*)$")
#: A ``key: value`` of a data ID; strings are quoted.
_DATA_ID_RE = re.compile(r"(\w+): '?([^,']*)'?")

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


def failedQuanta(text: str) -> pd.DataFrame:
    """Return the detector images whose quanta failed in a ``pipetask run`` log.

    Parameters
    ----------
    text : `str`
        The log.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph``, ``task`` and ``error`` (the
        exception and its message), one row per failed quantum with a detector
        data ID, in the log's order.
    """
    columns = ["visit", "arm", "spectrograph", "task", "error"]
    rows = []
    for line in text.splitlines():
        match = _FAILED_RE.search(line)
        if match is None:
            continue
        dataId = dict(_DATA_ID_RE.findall(match.group(2)))
        if not {"visit", "arm", "spectrograph"} <= dataId.keys():
            continue
        rows.append(
            (
                int(dataId["visit"]),
                dataId["arm"],
                int(dataId["spectrograph"]),
                match.group(1),
                match.group(3).strip(),
            )
        )
    return pd.DataFrame(rows, columns=columns)


def coverage(
    visits: pd.DataFrame, holdings: pd.DataFrame, failed: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return the status of each detector image of the judged visits.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.classify.classifyVisits`.
    holdings : `pandas.DataFrame`
        From `pfs.drp.qa.comparison.butlerQueries.detectorHoldings`.
    failed : `pandas.DataFrame`, optional
        ``visit``, ``arm`` and ``spectrograph`` of images whose reduction or
        judgement failed (`failedQuanta`); those not judged since are
        ``failed``.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph``, the visit's ``period``,
        ``sequence_type``, ``cadence``, ``iic_sequence_id``, ``category`` and
        ``night``,
        and ``status``, one of `STATUS_ORDER`: ``raw not in opdb`` for a
        detector with raw data that opdb didn't list.
    """
    keys = ["visit", "arm", "spectrograph"]
    judged = visits[visits["judged"]]
    expected = expectedDetectors(judged).assign(expected=True)
    held = holdings[holdings["visit"].isin(judged["pfs_visit_id"])]
    # A spectrograph has one red camera, read as arm r (low resolution) or m (medium); match the
    # camera, not the arm, and keep the Butler's arm where it has one.
    match = ["visit", "spectrograph", "camera"]
    detectors = (
        expected.rename(columns={"arm": "opdbArm"})
        .assign(camera=lambda f: _camera(f["opdbArm"]))
        .merge(held.assign(camera=lambda f: _camera(f["arm"])), on=match, how="outer")
    )
    detectors["arm"] = detectors["arm"].fillna(detectors["opdbArm"])
    detectors = detectors.drop(columns=["opdbArm", "camera"])
    for column in ("expected", "raw", "reduced", "judged"):
        detectors[column] = detectors[column].astype("boolean").fillna(False).astype(bool)
    if "pfsConfig" not in detectors:
        detectors["pfsConfig"] = True
    detectors["pfsConfig"] = detectors["pfsConfig"].astype("boolean").fillna(False).astype(bool)

    status = pd.Series("to reduce", index=detectors.index, dtype=object)
    status[detectors["reduced"]] = "to judge"
    if failed is not None and not failed.empty:
        failedKeys = failed.assign(camera=lambda f: _camera(f["arm"]))[["visit", "spectrograph", "camera"]]
        cameras = detectors[["visit", "spectrograph"]].assign(camera=_camera(detectors["arm"]))
        isFailed = pd.MultiIndex.from_frame(cameras).isin(
            pd.MultiIndex.from_frame(failedKeys.astype(cameras.dtypes))
        )
        status[isFailed] = "failed"
    # Without a pfsConfig the visit can't be reduced, and one such quantum stops the whole graph,
    # so none of the visit's pending detectors is scheduled.
    status[~detectors["pfsConfig"]] = "no pfsConfig"
    status[detectors["judged"]] = "judged"
    status[~detectors["raw"] & ~detectors["judged"]] = "no raw"
    status[~detectors["expected"]] = "raw not in opdb"
    detectors["status"] = status

    info = judged[
        ["pfs_visit_id", "period", "sequence_type", "cadence", "iic_sequence_id", "category", "night"]
    ]
    detectors = detectors.merge(info.rename(columns={"pfs_visit_id": "visit"}), on="visit", how="left")
    columns = [*keys, "period", "sequence_type", "cadence", "iic_sequence_id", "category", "night", "status"]
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
        One row per ``sequence_type``, ``cadence`` and ``category``: the number of
        ``visits``, the ``reason`` (``gated``, ``unvalidated`` or why they
        aren't judged), and one column per `STATUS_ORDER` counting the detector
        images of judged visits.
    """
    counts = (
        visits.assign(sequence_type=visits["sequence_type"].fillna("(none)"))
        .groupby(["sequence_type", "cadence", "category", "reason"], dropna=False)
        .size()
        .rename("visits")
        .reset_index()
    )
    if detectors.empty:
        images = pd.DataFrame(columns=["sequence_type", "cadence", "category", *STATUS_ORDER])
    else:
        images = (
            detectors.pivot_table(
                index=["sequence_type", "cadence", "category"],
                columns="status",
                values="visit",
                aggfunc="size",
                fill_value=0,
            )
            .reindex(columns=list(STATUS_ORDER), fill_value=0)
            .reset_index()
        )
        images.columns.name = None
    summary = counts.merge(images, on=["sequence_type", "cadence", "category"], how="left")
    for status in STATUS_ORDER:
        summary[status] = summary[status].fillna(0).astype(int)
    summary.loc[~summary["reason"].isin(["gated", "unvalidated"]), list(STATUS_ORDER)] = 0
    return summary.sort_values(["reason", "sequence_type", "cadence", "category"], ignore_index=True)


@dataclass(frozen=True)
class Pass:
    """One ``pipetask run``: visits sharing ``drpActor``'s configuration and whether they're gated.

    Parameters
    ----------
    name : `str`
        ``calibration`` or ``sky`` (by the configuration), prefixed
        ``unvalidated-`` for the types that aren't gated.
    types : `tuple` [`str`, ...]
        The sequence types in it.
    visits : `tuple` [`int`, ...]
        The visits.
    config : `dict` [`str`, `bool`]
        ``pipetask -c`` overrides, from `drpActorConfig`.
    groups : `dict` [`int`, `int`]
        Visit to cosmic-ray group, the group being the first visit of its
        sequence: ``cosmicray``'s ``groups`` with ``grouping="manual"``.
    judgeOnly : `bool`, optional
        Whether every image is already reduced, so only ``imageQualityQa``
        runs; its name ends ``-judge``.
    """

    name: str
    types: tuple[str, ...]
    visits: tuple[int, ...]
    config: dict[str, bool] = field(hash=False)
    groups: dict[int, int] = field(hash=False)
    judgeOnly: bool = False

    def cosmicrayConfig(self) -> str:
        """Return the ``cosmicray`` config file that groups by sequence."""
        return f'config.grouping = "manual"\nconfig.groups = {self.groups!r}\n'


def passes(visits: pd.DataFrame, detectors: pd.DataFrame) -> list[Pass]:
    """Group the visits with work left into passes, by configuration and whether they're gated.

    The gated passes come first in the list, so the gate's visits are done before the
    unvalidated ones, which can be many.

    Each is split by sequence: a sequence with an image ``to reduce`` gets the whole pipeline,
    and one whose images are all reduced only ``imageQualityQa`` (a ``-judge`` pass, listed
    first). Without the split, pipetask reruns ``cosmicray`` on every image whose reduction lacks
    a ``cosmicray_log``, an optional input of ``imageQualityQa``, though nothing is reduced again.

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
        Gated first, then by name.
    """
    pending = set(detectors.loc[detectors["status"].isin(["to reduce", "to judge"]), "visit"])
    todo = visits[visits["pfs_visit_id"].isin(pending)]
    if todo.empty:
        return []

    sequenceFirst = todo.groupby("iic_sequence_id")["pfs_visit_id"].transform("min")
    toReduce = set(detectors.loc[detectors["status"] == "to reduce", "visit"])
    reduceSequence = todo["pfs_visit_id"].isin(toReduce).groupby(todo["iic_sequence_id"]).transform("any")
    todo = todo.assign(
        group=sequenceFirst.astype(int),
        configKey=todo["sequence_type"].map(_configKey),
        judgeOnly=~reduceSequence.astype(bool),
    )
    result = []
    for (_, validated, judgeOnly), group in todo.groupby(["configKey", "validated", "judgeOnly"]):
        types = sorted(group["sequence_type"].unique())
        config = drpActorConfig(types[0])
        name = "sky" if config["reduceExposure:requireAdjustDetectorMap"] else "calibration"
        name = name if validated else f"unvalidated-{name}"
        result.append(
            Pass(
                name=f"{name}-judge" if judgeOnly else name,
                types=tuple(types),
                visits=tuple(sorted(int(visit) for visit in group["pfs_visit_id"])),
                config=config,
                groups={
                    int(visit): int(first)
                    for visit, first in sorted(zip(group["pfs_visit_id"], group["group"], strict=True))
                },
                judgeOnly=bool(judgeOnly),
            )
        )
    return sorted(
        result,
        key=lambda item: (
            item.name.startswith("unvalidated"),
            item.name.removesuffix("-judge"),
            not item.judgeOnly,
        ),
    )


def outputCollection(prefix: str, period: str, version: str, reductions: str = "") -> str:
    """Return the output collection of a period's comparison.

    Parameters
    ----------
    prefix : `str`
        E.g. ``u/someone/comparison``.
    period : `str`
        E.g. ``run30`` or ``run30-pre``.
    version : `str`
        The drp_qa version, e.g. from ``git describe``.
    reductions : `str`, optional
        What made the reductions, for fresh ones: e.g. ``drp_stella-w.2026.40``.
        Empty for drpActor's.

    Returns
    -------
    `str`
        ``<prefix>/<period>/<version>``, then ``/<reductions>`` if given.

    Raises
    ------
    ValueError
        If the version marks a modified tree (``-dirty``), whose results no
        version could reproduce.
    """
    if version.endswith("-dirty") or reductions.endswith("-dirty"):
        raise ValueError(f"{version} {reductions}: uncommitted changes; commit them first")
    collection = f"{prefix.rstrip('/')}/{period}/{version}"
    return f"{collection}/{reductions}" if reductions else collection


def pipetaskCommand(
    item: Pass,
    *,
    butler: str,
    pipeline: str,
    inputs: Sequence[str],
    output: str,
    skipExistingIn: Sequence[str],
    cosmicrayConfigFile: str | None = None,
    jobs: int = 8,
    instrument: str = "PFS",
    rebase: bool = False,
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
    cosmicrayConfigFile : `str`, optional
        Where `Pass.cosmicrayConfig` was written; needed unless
        ``item.judgeOnly``.
    jobs : `int`, optional
        Parallel processes.
    instrument : `str`, optional
        Instrument name. Default is ``PFS``.
    rebase : `bool`, optional
        Pass ``--rebase``, for an output collection that already exists:
        ``drpActor/reductions`` is a chain that grows with every reduction, so
        without it pipetask refuses an output whose recorded inputs no longer
        match.

    Returns
    -------
    `list` [`str`]
        The command, one argument per element.
    """
    command = ["pipetask", "--long-log", "--log-level", "PFS=INFO", "run", "-j", str(jobs)]
    # A judge-only pass runs one task, so the reduction's overrides would name labels it lacks.
    subset = f"{pipeline}#imageQualityQa" if item.judgeOnly else pipeline
    command += ["-b", butler, "-p", subset, "-i", ",".join(inputs), "-o", output]
    if rebase:
        command += ["--rebase"]
    if skipExistingIn:
        command += ["--skip-existing-in", ",".join(skipExistingIn)]
    if not item.judgeOnly:
        if cosmicrayConfigFile is None:
            raise ValueError(f"Pass {item.name} reduces, so it needs cosmicrayConfigFile")
        for key, value in item.config.items():
            command += ["-c", f"{key}={value}"]
        command += ["-C", f"cosmicray:{cosmicrayConfigFile}"]
    command += ["-d", f"instrument = '{instrument}' AND {visitExpression(item.visits)}"]
    return command


def _camera(arm: pd.Series) -> pd.Series:
    """Return the camera of each arm: ``m`` and ``r`` are the same red camera."""
    return arm.replace({"m": "r"})


def _configKey(sequenceType: str) -> tuple:
    """Return a hashable key of `drpActorConfig`."""
    return tuple(sorted(drpActorConfig(sequenceType).items()))
