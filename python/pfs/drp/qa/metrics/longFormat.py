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

``description`` is the line species of the line list (``HgI``, ``NeI``), not the
lamp (``species`` in `~pfs.drp.qa.metrics.calibration.addSpecies`).
"""

from collections.abc import Mapping

import pandas as pd

__all__ = ["SPECIES_COLUMNS", "speciesFrame", "speciesStats"]

#: The columns of a per-species metrics table, in order, with their dtypes.
SPECIES_COLUMNS = {
    "visit": "int64",
    "arm": "string",
    "spectrograph": "int64",
    "description": "string",
    "metric": "string",
    "value": "float64",
}

#: The metric names of the two values of a ``speciesStats`` entry.
_STAT_METRICS = ("fitXRms", "fitYRms")


def speciesFrame(dataId: Mapping[str, object], stats: Mapping[str, tuple[float, float]]) -> pd.DataFrame:
    """Return one quantum's per-species fit statistics in long format.

    Parameters
    ----------
    dataId : `Mapping`
        The quantum's data ID, with ``visit``, ``arm`` and ``spectrograph``.
    stats : `Mapping` [`str`, `tuple` [`float`, `float`]]
        ``{species: (xRms, yRms)}``, from the ``Stats for <species>`` lines
        of the ``reduceExposure`` log.

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
        for metric, value in zip(_STAT_METRICS, stats[species], strict=True)
    ]
    return pd.DataFrame(rows, columns=list(SPECIES_COLUMNS)).astype(SPECIES_COLUMNS)


def speciesStats(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Return the ``{species: (xRms, yRms)}`` mapping of a `speciesFrame`.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        Long-format rows of one quantum.

    Returns
    -------
    `dict` [`str`, `tuple` [`float`, `float`]]
        Missing values are NaN.
    """
    wide = frame.pivot(index="description", columns="metric", values="value")
    wide = wide.reindex(columns=list(_STAT_METRICS))
    return {str(species): (float(row.iloc[0]), float(row.iloc[1])) for species, row in wide.iterrows()}
