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

The pipeline that made a reduction is recorded too, but only in its spectra:
drp_stella's ``getPfsVersions`` writes the versions of `PIPELINE_PRODUCTS` to
each ``pfsArm``'s primary header, and the calexp has none. `pipelineVersions`
parses them from a header, and `commonPipeline` reduces a table of them to the
one pipeline a set of reductions was made with.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

__all__ = [
    "CALIBRATION_COLUMNS",
    "PIPELINE_COLUMNS",
    "PIPELINE_PRODUCTS",
    "calibrationsUsed",
    "commonPipeline",
    "pipelineVersions",
    "provenanceInputs",
]

_INPUT_KEY = re.compile(
    r"^lsst[ .]butler[ .]input[ .](?P<index>\d+)[ .](?P<key>id|run|datasettype)$", re.IGNORECASE
)

#: Columns of the table `calibrationsUsed` returns.
CALIBRATION_COLUMNS = ("datasetType", "run", "id", "source", "via")

#: The products whose versions make up the reducing pipeline, as ``getPfsVersions`` names them.
PIPELINE_PRODUCTS = ("datamodel", "obs_pfs", "drp_stella")

#: The column of a metrics table holding each product's version.
PIPELINE_COLUMNS = {
    "datamodel": "versionDatamodel",
    "obs_pfs": "versionObsPfs",
    "drp_stella": "versionDrpStella",
}

_VERSION_KEY = re.compile(
    rf"^(?:hierarch[ .])?(?:version_)?(?P<product>{'|'.join(PIPELINE_PRODUCTS)})$", re.IGNORECASE
)


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


def pipelineVersions(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Return the versions of the pipeline a reduction recorded in its header.

    Parameters
    ----------
    metadata : mapping
        A header as a flat mapping, e.g. a ``pfsArm``'s primary header. The
        keys are ``getPfsVersions``'s: ``VERSION_DRP_STELLA`` (``HIERARCH``
        in FITS, which may or may not be kept), or ``DRP_STELLA`` with
        ``prefix=""``; any case.

    Returns
    -------
    `dict` [`str`, `str`]
        Version per product, in `PIPELINE_PRODUCTS` order. A product the
        header doesn't name is missing.
    """
    found = {}
    for key, value in metadata.items():
        match = _VERSION_KEY.match(str(key).strip())
        if match and value is not None and str(value).strip():
            found[match["product"].lower()] = str(value).strip()
    return {product: found[product] for product in PIPELINE_PRODUCTS if product in found}


def commonPipeline(versions: pd.DataFrame) -> dict[str, str]:
    """Return the one pipeline a set of reductions was made with.

    Thresholds hold for the pipeline that reduced the visits they were derived
    from, so the thresholds file records it; that needs one pipeline.

    Parameters
    ----------
    versions : `pandas.DataFrame`
        One row per reduction, with `PIPELINE_COLUMNS`, e.g. from
        `pfs.drp.qa.metrics.readers.readPipelineVersions`.

    Returns
    -------
    `dict` [`str`, `str`]
        Version per product, in `PIPELINE_PRODUCTS` order.

    Raises
    ------
    ValueError
        If there are no reductions, or a product's version is unknown for some
        or differs between them.
    """
    if versions.empty:
        raise ValueError("No reductions to read the pipeline of")
    common = {}
    for product, column in PIPELINE_COLUMNS.items():
        values = versions[column] if column in versions else pd.Series(None, index=versions.index)
        if values.isna().any():
            raise ValueError(
                f"{int(values.isna().sum())} of {len(values)} reductions record no {product} version"
            )
        distinct = sorted(set(values))
        if len(distinct) > 1:
            counts = values.value_counts()
            raise ValueError(
                f"The reductions were made with {product} {', '.join(f'{v} ({n})' for v, n in counts.items())}"
            )
        common[product] = distinct[0]
    return common
