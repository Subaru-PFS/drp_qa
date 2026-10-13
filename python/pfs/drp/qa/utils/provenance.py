"""Which calibrations went into a Butler dataset, read from its provenance.

daf_butler records, in each dataset it writes, the datasets its quantum read:
flat metadata keys such as ``LSST BUTLER INPUT 3 DATASETTYPE`` (FITS) or
``LSST.BUTLER.INPUT.3.RUN``, with ``ID`` and ``RUN`` beside them. A calexp's
inputs are those of ``reduceExposure``, so its provenance names the exact
``fiberProfiles`` and ``detectorMap_calib`` the reduction used. Looking a
calibration up again by the exposure's date can give another one: a later
certification, or a reduction run with another run's calibrations.

`provenanceInputs` parses the keys and needs no stack. `calibrationsUsed`
reads a dataset's metadata from a Butler it is given and returns its
calibration inputs, falling back to a lookup, marked as such, for any it needs
that no provenance names.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

__all__ = ["CALIBRATION_COLUMNS", "calibrationsUsed", "provenanceInputs"]

_INPUT_KEY = re.compile(
    r"^lsst[ .]butler[ .]input[ .](?P<index>\d+)[ .](?P<key>id|run|datasettype)$", re.IGNORECASE
)

#: Columns of the table `calibrationsUsed` returns.
CALIBRATION_COLUMNS = ("datasetType", "run", "id", "source", "via")


def provenanceInputs(metadata: Mapping[str, Any]) -> pd.DataFrame:
    """Return the input datasets recorded in a dataset's provenance metadata.

    Parameters
    ----------
    metadata : mapping
        The dataset's metadata as a flat mapping, e.g. ``PropertyList.toDict()``
        of an exposure's header.

    Returns
    -------
    `pandas.DataFrame`
        One row per recorded input, in input order, with columns ``datasetType``,
        ``run`` and ``id`` (a string UUID). Empty when there is no provenance,
        or when daf_butler dropped the inputs because there were too many.
    """
    inputs: dict[int, dict[str, str]] = {}
    for key, value in metadata.items():
        match = _INPUT_KEY.match(str(key))
        if match:
            field = {"id": "id", "run": "run", "datasettype": "datasetType"}[match["key"].lower()]
            inputs.setdefault(int(match["index"]), {})[field] = str(value)
    table = pd.DataFrame([inputs[ii] for ii in sorted(inputs)], columns=["datasetType", "run", "id"])
    return table


def calibrationsUsed(
    butler: Any,
    dataId: Mapping[str, Any],
    datasetTypes: Iterable[str] = ("calexp",),
    fallback: Iterable[str] = ("fiberProfiles", "detectorMap_calib"),
) -> pd.DataFrame:
    """Return the calibrations the datasets ``datasetTypes`` were made with.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler holding the datasets and their inputs.
    dataId : mapping
        Data ID of the datasets, e.g. one detector of one visit.
    datasetTypes : iterable of `str`, optional
        Datasets whose provenance to read; only their metadata is read. One
        that is missing is skipped.
    fallback : iterable of `str`, optional
        Calibrations to look up, as of the exposure's date, when no provenance
        names them.

    Returns
    -------
    `pandas.DataFrame`
        One row per calibration, with columns `CALIBRATION_COLUMNS`:
        ``source`` is ``"provenance"`` (recorded by the reduction) or
        ``"lookup"`` (found again now, which may differ from what the
        reduction used), and ``via`` the dataset whose provenance named it.
    """
    rows: list[dict[str, str]] = []
    for datasetType in datasetTypes:
        try:
            metadata = butler.get(f"{datasetType}.metadata", dataId)
        except LookupError:
            continue
        inputs = provenanceInputs(metadata.toDict() if hasattr(metadata, "toDict") else metadata)
        for record in inputs.to_dict("records"):
            if butler.get_dataset_type(record["datasetType"]).isCalibration():
                rows.append({**record, "source": "provenance", "via": datasetType})
    named = {row["datasetType"] for row in rows}
    for datasetType in fallback:
        ref = None if datasetType in named else butler.find_dataset(datasetType, dataId)
        if ref is not None:
            rows.append(
                {"datasetType": datasetType, "run": ref.run, "id": str(ref.id), "source": "lookup", "via": ""}
            )
    table = pd.DataFrame(rows, columns=list(CALIBRATION_COLUMNS))
    return table.drop_duplicates(["datasetType", "id"]).reset_index(drop=True)
