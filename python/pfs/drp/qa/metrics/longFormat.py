"""Per-species metrics in long format.

One row per detector, line species and metric, so the schema is the same for
every quantum whatever species it has: frames concatenate without padding, and
a species that was not measured has no row rather than a NaN column.

===== === ============ =========== ======= ======
visit arm spectrograph description metric  value
===== === ============ =========== ======= ======
12345 r   1            NeI         fitXRms 0.021
12345 r   1            NeI         fitYRms 0.034
===== === ============ =========== ======= ======

``description`` is the line species of the line list (``HgI``, ``NeI``), or
``Trace`` for trace centroids, not the lamp (``species`` in
`~pfs.drp.qa.metrics.calibration.addSpecies`).

The values are drp_stella's own, read from the ``Stats for <species>`` lines
``fitDetectorMap`` logs during ``reduceExposure``, over the lines the fit used:
the error-weighted RMS (``fitXRms``, ``fitYRms``) and the softening that brings
chi^2/dof to 1 (``fitXSoften``, ``fitYSoften``). ``dmLineResiduals`` gives the
robust RMS beside them, and does not recompute these.
"""

import re
from collections.abc import Mapping, Sequence

import pandas as pd

__all__ = ["SPECIES_COLUMNS", "STAT_METRICS", "parseSpeciesStats", "speciesFrame", "speciesStats"]

#: The columns of a per-species metrics table, in order, with their dtypes.
SPECIES_COLUMNS = {
    "visit": "int64",
    "arm": "string",
    "spectrograph": "int64",
    "description": "string",
    "metric": "string",
    "value": "float64",
}

#: The metric names of the values of a ``speciesStats`` entry, in order.
STAT_METRICS = ("fitXRms", "fitYRms", "fitXSoften", "fitYSoften")

#: ``Stats for <species>:`` lines of ``fitDetectorMap``; newer drp_stella puts
#: the detector after the colon. Not the ``Stats for fiberId=...`` lines.
_SPECIES_STATS = re.compile(
    r"Stats for (\w+): (?:arm=\S+ spectrograph=\d+ )?chi2=\S+ dof=\d+ xRMS=(\S+) yRMS=(\S+)"
    r"(?: xSoften=(\S+) ySoften=(\S+))?"
)


def parseSpeciesStats(line: str) -> tuple[str, tuple[float, ...]] | None:
    """Return the species and `STAT_METRICS` values of one ``reduceExposure`` log line.

    Parameters
    ----------
    line : `str`
        A log line.

    Returns
    -------
    `tuple` [`str`, `tuple` [`float`, ...]] or `None`
        ``(species, (xRms, yRms, xSoften, ySoften))``, the softening NaN in
        a log that lacks it; `None` when the line is not a species' stats or
        a value is not a number.
    """
    match = _SPECIES_STATS.search(line)
    if match is None:
        return None
    try:
        values = tuple(float("nan") if value is None else float(value) for value in match.groups()[1:])
    except ValueError:
        return None
    return match.group(1), values


def speciesFrame(dataId: Mapping[str, object], stats: Mapping[str, Sequence[float]]) -> pd.DataFrame:
    """Return one quantum's per-species fit statistics in long format.

    Parameters
    ----------
    dataId : `Mapping`
        The quantum's data ID, with ``visit``, ``arm`` and ``spectrograph``.
    stats : `Mapping` [`str`, sequence of `float`]
        ``{species: (xRms, yRms, xSoften, ySoften)}`` (`parseSpeciesStats`);
        a shorter tuple gives rows for the leading `STAT_METRICS` only.

    Returns
    -------
    `pandas.DataFrame`
        `SPECIES_COLUMNS`, one row per species and metric, sorted by species;
        empty, with the same columns and dtypes, when ``stats`` is.
    """
    rows = [
        {
            "visit": dataId["visit"],
            "arm": dataId["arm"],
            "spectrograph": dataId["spectrograph"],
            "description": species,
            "metric": metric,
            "value": value,
        }
        for species in sorted(stats)
        for metric, value in zip(STAT_METRICS, stats[species], strict=False)
    ]
    return pd.DataFrame(rows, columns=list(SPECIES_COLUMNS)).astype(SPECIES_COLUMNS)


def speciesStats(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Return the ``{species: (xRms, yRms)}`` mapping of a `speciesFrame`'s lamp species.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        Long-format rows of one quantum.

    Returns
    -------
    `dict` [`str`, `tuple` [`float`, `float`]]
        Missing values are NaN. ``Trace`` is left out: ``imageQualityLogQa``
        reports it from ``iqQaMetrics`` (``fitTraceXRms``).
    """
    frame = frame[frame["description"] != "Trace"]
    wide = frame.pivot(index="description", columns="metric", values="value")
    wide = wide.reindex(columns=list(STAT_METRICS[:2]))
    return {str(species): (float(row.iloc[0]), float(row.iloc[1])) for species, row in wide.iterrows()}
