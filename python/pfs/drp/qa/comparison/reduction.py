"""How comparison mode configures the reduction: what changes the products it writes.

This module, unlike the rest of `pfs.drp.qa.comparison`, counts towards the drp_qa version that
names the output collection (`pfs.drp.qa.comparison.redo.drpQaVersion`): a change here changes what
``pipetask`` writes, so it must start a new collection rather than reuse the old one's quanta.
"""

import pandas as pd

__all__ = [
    "MEASURE_LABELS",
    "MEASURE_PIPELINE",
    "REDUCE_LABELS",
    "REDUCE_PIPELINE",
    "cosmicrayConfig",
    "cosmicrayGroups",
    "drpActorConfig",
]

#: drp_stella's reduction pipeline, and the labels of it a reduce pass runs: all but ``mergeArms``,
#: which combines arms into a product the QA tasks don't read.
REDUCE_PIPELINE = "$DRP_STELLA_DIR/pipelines/reduceExposure.yaml"
REDUCE_LABELS = ("isr", "cosmicray", "reduceExposure")

#: drp_qa's QA pipeline, and the labels a measure pass runs on the reductions: the image quality, the
#: line residuals against the calibration and the visit's detectorMap, and the reduction statistics.
#: Which tasks run changes what a pass writes, so this module, not the neutral ``plan``, holds them.
MEASURE_PIPELINE = "$DRP_QA_DIR/pipelines/drpQA.yaml"
MEASURE_LABELS = ("imageQualityQa", "dmLineResiduals", "reductionStats")


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


def cosmicrayGroups(visits: pd.DataFrame) -> dict[int, int]:
    """Return each visit's cosmic-ray group: the first visit of its sequence.

    ``cosmicray`` combines the exposures of one sequence only, as ``drpActor``, which reduces one
    sequence at a time.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        ``pfs_visit_id`` and ``iic_sequence_id``: every visit of the sequences, not only those
        with work left, so a group starts where its sequence does.

    Returns
    -------
    `dict` [`int`, `int`]
        Visit to group, in visit order.
    """
    first = visits.groupby("iic_sequence_id")["pfs_visit_id"].transform("min")
    return {
        int(visit): int(group) for visit, group in sorted(zip(visits["pfs_visit_id"], first, strict=True))
    }


def cosmicrayConfig(groups: dict[int, int]) -> str:
    """Return the ``cosmicray`` config file that groups by sequence.

    Parameters
    ----------
    groups : `dict` [`int`, `int`]
        From `cosmicrayGroups`.

    Returns
    -------
    `str`
        ``grouping="manual"`` with ``groups``.
    """
    return f'config.grouping = "manual"\nconfig.groups = {groups!r}\n'
