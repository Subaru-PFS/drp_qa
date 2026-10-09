"""How comparison mode configures the reduction: what changes the products it writes.

This module, unlike the rest of `pfs.drp.qa.comparison`, counts towards the drp_qa version that
names the output collection (`pfs.drp.qa.comparison.cli.drpQaVersion`): a change here changes what
``pipetask`` writes, so it must start a new collection rather than reuse the old one's quanta.
"""

import pandas as pd

__all__ = ["cosmicrayConfig", "cosmicrayGroups", "drpActorConfig"]


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
