"""Per-species line flux of one detector image.

The inputs to cross-visit lamp stability: for each line species, the median
and robust RMS of the measured line fluxes, how many lines they are over, and
how many of the species' lines are saturated.

Within one image these say little about the lamp: a species' lines span
orders of magnitude, so the RMS is a property of the line list. The reference
(rule R1 of ``docs/qa-principles.md``) is the same species on the same
detector at another epoch, which is a comparison across visits and not made
here. Nothing here is judged.

The flux is ``lines``' own (``flux``, in the calexp's units and not per
second): compare visits of the same exposure time, or divide by it. The
median is over the lines measured, so a lamp fading far enough to lose its
faint lines also changes the set; ``nFluxLines`` says when.

Stack-free: `speciesLineFlux` takes the ``lines`` table as a DataFrame.
"""

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["BAD_STATUS", "FLUX_METRICS", "TRACE", "speciesLineFlux"]

#: The ``description`` of trace centroids in ``lines``.
TRACE = "Trace"

#: drp_stella's ``ReferenceLineStatus.BAD`` (NOT_VISIBLE, BLEND, SUSPECT,
#: REJECTED, BROAD): the status bits ``fitDetectorMap`` drops by default.
BAD_STATUS = 0x1F

#: The metrics of each species, in order. ``nSaturatedLines`` only when told
#: which lines are saturated.
FLUX_METRICS = ("medLineFlux", "robustRmsLineFlux", "nFluxLines", "nSaturatedLines")


def _robustRms(values: np.ndarray) -> float:
    """Return 0.741 times the interquartile range, as ``pfs.drp.stella.utils.math.robustRms`` (R5)."""
    if values.size == 0:
        return np.nan
    lower, upper = np.percentile(values, (25.0, 75.0))
    return float(0.741 * (upper - lower))


def speciesLineFlux(
    lines: pd.DataFrame, select: Any = None, saturated: Any = None
) -> dict[str, dict[str, float]]:
    """Return each line species' flux statistics.

    Parameters
    ----------
    lines : `pandas.DataFrame`
        The ``lines`` table: ``description``, ``flux``, ``flag``, and
        ``status`` when it has one. Trace centroids (``description``
        ``Trace``) are left out.
    select : array of `bool`, optional
        The rows to use, e.g. those on lit fibers; all when `None`.
    saturated : array of `bool`, optional
        Per row, whether the line is saturated
        (`pfs.drp.qa.metrics.saturation.saturatedAt`); `None` when unknown.

    Returns
    -------
    `dict` [`str`, `dict` [`str`, `float`]]
        ``{species: {metric: value}}`` for every species with a selected row:
        ``medLineFlux`` and ``robustRmsLineFlux`` over its lines that are
        unflagged, carry no `BAD_STATUS` bit and have a finite flux (NaN when
        none), ``nFluxLines`` how many those are, and, given ``saturated``,
        ``nSaturatedLines`` over all its selected lines, flagged or not.
    """
    if "description" not in lines.columns or len(lines) == 0:
        return {}
    description = lines["description"].to_numpy(dtype=object)
    use = description != TRACE
    if select is not None:
        use &= np.asarray(select, dtype=bool)
    flux = lines["flux"].to_numpy(dtype=float)
    good = use & ~lines["flag"].to_numpy(dtype=bool) & np.isfinite(flux)
    if "status" in lines.columns:
        good &= (lines["status"].to_numpy(dtype=np.int64) & BAD_STATUS) == 0
    if saturated is not None:
        saturated = np.asarray(saturated, dtype=bool)

    result: dict[str, dict[str, float]] = {}
    for species in sorted({str(name) for name in description[use]}):
        ofSpecies = description == species
        values = flux[good & ofSpecies]
        stats = {
            "medLineFlux": float(np.median(values)) if values.size else np.nan,
            "robustRmsLineFlux": _robustRms(values),
            "nFluxLines": float(values.size),
        }
        if saturated is not None:
            stats["nSaturatedLines"] = float((saturated & use & ofSpecies).sum())
        result[species] = stats
    return result
