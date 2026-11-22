"""Detector-health statistics in long format, read from the reduction's metadata.

One row per detector, amplifier and metric, so the schema is the same for every
quantum whatever it measured: a statistic a reduction did not record has no row
rather than a NaN column, and frames concatenate without padding.

===== === ============ ========= ========== ======
visit arm spectrograph amplifier metric     value
===== === ============ ========= ========== ======
12345 r   1            c1        readNoise  3.912
12345 r   1            c1        gain       1.238
12345 r   1                      fitXRms    0.021
12345 n   3                      nCR        1356.0
===== === ============ ========= ========== ======

``amplifier`` is a CCD amplifier (``c1``) for a per-amplifier statistic, and empty
for a whole-detector one (every n-arm statistic, and the detectorMap fit). ``value``
is always a float: a count is stored as a float, and a string status (whether the
detectorMap adjustment applied) is not stored here.

The statistics are measured by the reduction and recorded in its task metadata
(``isr_metadata``, ``reduceExposure_metadata``) and the ISR header; this module reads
them and judges nothing. The limits come after a run of values, so detector health is
stored, not gated: no check in `~pfs.drp.qa.metrics.registry` reads these rows.

Every frame also carries the `COVERAGE_METRICS` (``isrRan``, ``reduceExposureRan`` and the
derived ``reductionStage``), whole-detector rows written whatever ran. The task is anchored
on the raw header, so a detector whose reduction never started still gets these: that is how
every exposed detector ends up with a row.

This replaces the log parsing the image-quality task used to do: the same detectorMap
fit statistics that were scraped from the ``reduceExposure`` log are read here from
``reduceExposure_metadata`` instead.

Metadata layout this reads (as recorded by drp_stella and obs_pfs):

- ``reduceExposure_metadata``
    - ``reduceExposure.detectorMapAdjustment``: ``applied``, ``failed`` or ``skipped``
      (the top task's own scalar, under the ``reduceExposure`` key; `_adjustmentApplied`).
      The ``fit*`` and ``reserved*`` statistics are read only when it is ``applied``,
      as they can linger from a fit that later failed.
    - section ``reduceExposure:adjustDetectorMap``: the ``fit*``, ``reserved*`` and
      ``final*`` scalars (`FIT_METRICS`) read by `reductionStatsFrame`, and the per-species
      arrays (``speciesName``, ``speciesXRms`` …) read by `speciesFitStats` into the
      per-species table (``iqQaSpeciesMetrics``), which has a species column and is kept
      separate from this whole-detector table. The per-QA-fiber arrays are not read.
    - section ``reduceExposure:scatteredLight``: `SCATTERED_METRICS`.
    - section ``reduceExposure:pfiCorrection``: `PFI_METRICS`.
- ``isr_metadata``: the n-arm whole-detector counts (`ISR_COUNT_METRICS`) and, on a
  CCD with a dark, `ISR_DARK_METRICS`. Each is recorded only when the step that sets
  it ran, so a row is written only for a statistic that is present.
- the ISR header: per-amplifier ``LSST GAIN <amp>`` and ``LSST READNOISE <amp>``,
  stored as the ``gain`` and ``readNoise`` metrics of that amplifier.
"""

from collections.abc import Mapping
from typing import Any

import pandas as pd

__all__ = [
    "COVERAGE_METRICS",
    "FIT_METRICS",
    "ISR_COUNT_METRICS",
    "ISR_DARK_METRICS",
    "PFI_METRICS",
    "REDUCTION_COLUMNS",
    "SCATTERED_METRICS",
    "SPECIES_STAT_SUFFIXES",
    "asMapping",
    "reductionStatsFrame",
    "speciesFitStats",
]

#: The columns of a detector-health table, in order, with their dtypes.
REDUCTION_COLUMNS = {
    "visit": "int64",
    "arm": "string",
    "spectrograph": "int64",
    "amplifier": "string",
    "metric": "string",
    "value": "float64",
}

#: n-arm whole-detector ISR counts, recorded per step (absent when the step did not run).
ISR_COUNT_METRICS = (
    "nSat",
    "nLow",
    "nDefects",
    "nLinDefects",
    "nUnstable",
    "nRateUnstable",
    "nUnclassified",
    "nAsicGlitch",
    "nGlitchMasked",
    "nCR",
    "nCrFlags",
    "nGlitchPairs",
    "crIterations",
    "nRateUntestable",
)

#: ISR dark statistics, on a CCD that has a dark (e.g. r3).
ISR_DARK_METRICS = ("darkScale", "darkScalePerSecond")

#: The ``adjustDetectorMap`` fit scalars. ``fit*`` (the final distortion fit) and
#: ``reserved*`` (the reserved lines) are read only when the adjustment applied;
#: ``final*`` (the final detectorMap) is read whenever present.
_FIT_SUFFIXES = (
    "Chi2",
    "Dof",
    "XRms",
    "YRms",
    "XRobustRms",
    "YRobustRms",
    "XSoften",
    "YSoften",
    "NumLines",
)
_FIT_GATED = (
    *(f"{group}{suffix}" for group in ("fit", "reserved") for suffix in _FIT_SUFFIXES),
    "fitNumAvailable",
)
_FIT_UNGATED = tuple(f"final{suffix}" for suffix in _FIT_SUFFIXES)
FIT_METRICS = _FIT_GATED + _FIT_UNGATED

#: ``scatteredLight`` scalars, namespaced because the keys (``top``, ``bottom``) are generic.
SCATTERED_METRICS = ("top", "bottom", "modelMean", "modelMax")

#: ``pfiCorrection`` scalars (``skyNorms*`` are NaN without sky norms).
PFI_METRICS = ("correctionMean", "correctionStd", "skyNormsMean", "skyNormsStd", "numBad")

#: Reduction-coverage metrics, written for every exposed detector whatever ran: ``isrRan`` and
#: ``reduceExposureRan`` (1.0/0.0, whether that step produced metadata) and the derived
#: ``reductionStage`` (0 raw only, 1 ISR, 2 reduceExposure). Whole-detector (empty amplifier).
COVERAGE_METRICS = ("isrRan", "reduceExposureRan", "reductionStage")

#: The ``adjustDetectorMap`` per-species array suffixes read into one ``speciesStats`` entry,
#: in the order of `~pfs.drp.qa.metrics.longFormat.STAT_METRICS`: (xRms, yRms, xSoften, ySoften).
#: The arrays (``speciesXRms`` …) run parallel to ``speciesName``.
SPECIES_STAT_SUFFIXES = ("XRms", "YRms", "XSoften", "YSoften")


def asMapping(obj: Any) -> Mapping[str, Any] | None:
    """Return task metadata or a header as a plain mapping.

    ``TaskMetadata`` (``to_dict``) and ``PropertyList``/``PropertySet`` (``toDict``) are
    converted to nested dictionaries; a mapping is returned as is; anything else is `None`.
    """
    if obj is None:
        return None
    for method in ("to_dict", "toDict"):
        convert = getattr(obj, method, None)
        if callable(convert):
            return convert()
    return obj if isinstance(obj, Mapping) else None


def _asList(value: Any) -> list:
    """Return a metadata array as a list; a scalar as a one-element list, `None` as empty.

    A string (a single species name) or a mapping counts as one element, not a sequence to
    iterate; any other iterable (list, tuple, numpy array) becomes a list of its elements.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _section(metadata: Mapping[str, Any] | None, label: str) -> Mapping[str, Any]:
    """Return the sub-mapping for a subtask label, or an empty mapping.

    Tolerant of how task metadata nests a subtask. ``reduceExposure_metadata`` is
    already scoped to the task, so a subtask is normally reached by its bare name
    (``adjustDetectorMap``); the full label (``reduceExposure:adjustDetectorMap``)
    and a flattened form, whose keys are the sub-keys prefixed with the label and a
    ``:`` or ``.``, are accepted too.
    """
    if not isinstance(metadata, Mapping):
        return {}
    tail = label.rsplit(":", 1)[-1]
    for key in (label, tail):
        section = metadata.get(key)
        if isinstance(section, Mapping):
            return section
    # A flattened mapping: keys like ``adjustDetectorMap.fitChi2`` or ``...:fitChi2``.
    flattened = {}
    for prefix in (f"{tail}.", f"{tail}:", f"{label}.", f"{label}:"):
        for key, value in metadata.items():
            if isinstance(key, str) and key.startswith(prefix):
                flattened[key[len(prefix) :]] = value
        if flattened:
            return flattened
    return {}


def _adjustmentApplied(metadata: Mapping[str, Any]) -> bool:
    """Whether the detectorMap adjustment applied, from wherever reduceExposure recorded it.

    ``reduceExposure`` records ``detectorMapAdjustment`` on its own task metadata. The serialized
    ``reduceExposure_metadata`` nests the top task's scalars under the ``reduceExposure`` key
    (beside the subtask sections ``reduceExposure:adjustDetectorMap`` …), so it is read from there,
    falling back to a flattened or top-level mapping.
    """
    for source in (_section(metadata, "reduceExposure"), metadata):
        value = source.get("detectorMapAdjustment")
        if value is not None:
            return str(value) == "applied"
    return False


def _toFloat(value: Any) -> float | None:
    """Return a scalar as a float, or `None` when it is missing or not a number."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _metrics(source: Mapping[str, Any], names: tuple[str, ...]) -> list[tuple[str, float]]:
    """Return ``(metric, value)`` for each name present in ``source`` with a numeric value."""
    rows: list[tuple[str, float]] = []
    for name in names:
        if name not in source:
            continue
        value = _toFloat(source[name])
        if value is not None:
            rows.append((name, value))
    return rows


def _ampMetrics(header: Mapping[str, Any] | None) -> list[tuple[str, str, float]]:
    """Return ``(amplifier, metric, value)`` for the per-amplifier ISR header keys.

    Reads ``LSST GAIN <amp>`` and ``LSST READNOISE <amp>`` as the ``gain`` and
    ``readNoise`` metrics of each amplifier.
    """
    if not isinstance(header, Mapping):
        return []
    prefixes = {"LSST GAIN ": "gain", "LSST READNOISE ": "readNoise"}
    rows: list[tuple[str, str, float]] = []
    for key, raw in header.items():
        for prefix, metric in prefixes.items():
            if isinstance(key, str) and key.startswith(prefix):
                value = _toFloat(raw)
                if value is not None:
                    rows.append((key[len(prefix) :].strip(), metric, value))
    return rows


def reductionStatsFrame(
    dataId: Mapping[str, Any],
    *,
    isrMetadata: Mapping[str, Any] | None = None,
    reduceExposureMetadata: Mapping[str, Any] | None = None,
    header: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return the detector-health statistics of one image as long-format rows.

    Parameters
    ----------
    dataId : mapping
        The image, with ``visit``, ``arm`` and ``spectrograph``.
    isrMetadata : mapping, optional
        The ISR task metadata (``isr_metadata``): the n-arm counts and dark scale.
    reduceExposureMetadata : mapping, optional
        The ``reduceExposure`` task metadata: the detectorMap fit, scattered-light
        and PFI-correction scalars, under their subtask sections.
    header : mapping, optional
        The ISR output header: per-amplifier ``LSST GAIN``/``LSST READNOISE``.

    Returns
    -------
    `pandas.DataFrame`
        `REDUCTION_COLUMNS`, one row per amplifier and metric present. Whole-detector
        statistics have an empty ``amplifier``. An input that is `None` or lacks a
        statistic contributes no row for it, except that the `COVERAGE_METRICS` rows are
        always written, so the frame is never empty for an exposed detector.
    """
    visit = int(dataId["visit"])
    arm = str(dataId["arm"])
    spectrograph = int(dataId["spectrograph"])

    rows: list[tuple[str, str, float]] = []  # (amplifier, metric, value)

    # Reduction coverage, whole-detector: a row for every exposed detector whatever ran, so one
    # whose reduction never started still has rows (the task is anchored on raw). isrRan and
    # reduceExposureRan say whether each step produced metadata; reductionStage is how far it got
    # — the highest step reached, not a conjunction of the flags.
    isrRan = isrMetadata is not None
    reduceExposureRan = reduceExposureMetadata is not None
    stage = 2.0 if reduceExposureRan else 1.0 if isrRan else 0.0
    rows.append(("", "isrRan", float(isrRan)))
    rows.append(("", "reduceExposureRan", float(reduceExposureRan)))
    rows.append(("", "reductionStage", stage))

    # Whole-detector ISR counts (n arm) and dark scale (CCDs with a dark).
    if isrMetadata is not None:
        for name, value in _metrics(isrMetadata, ISR_COUNT_METRICS + ISR_DARK_METRICS):
            rows.append(("", name, value))

    # detectorMap fit, scattered light and PFI correction.
    if reduceExposureMetadata is not None:
        adjust = _section(reduceExposureMetadata, "reduceExposure:adjustDetectorMap")
        applied = _adjustmentApplied(reduceExposureMetadata)
        gated = FIT_METRICS if applied else _FIT_UNGATED
        for name, value in _metrics(adjust, gated):
            rows.append(("", name, value))

        scattered = _section(reduceExposureMetadata, "reduceExposure:scatteredLight")
        for name, value in _metrics(scattered, SCATTERED_METRICS):
            rows.append(("", f"scatteredLight.{name}", value))

        pfi = _section(reduceExposureMetadata, "reduceExposure:pfiCorrection")
        for name, value in _metrics(pfi, PFI_METRICS):
            rows.append(("", f"pfiCorrection.{name}", value))

    # Per-amplifier gain and read noise, from the ISR header. CCD arms only: the n arm is an
    # H4 with channels, not the CCD amplifiers these keys name.
    if arm != "n":
        for amplifier, metric, value in _ampMetrics(header):
            rows.append((amplifier, metric, value))

    frame = pd.DataFrame(
        {
            "visit": [visit] * len(rows),
            "arm": [arm] * len(rows),
            "spectrograph": [spectrograph] * len(rows),
            "amplifier": [amplifier for amplifier, _, _ in rows],
            "metric": [metric for _, metric, _ in rows],
            "value": [value for _, _, value in rows],
        }
    )
    return frame.astype(REDUCTION_COLUMNS)


def speciesFitStats(reduceExposureMetadata: Any) -> dict[str, tuple[float, ...]]:
    """Return the per-species detectorMap fit statistics from the reduceExposure metadata.

    drp_stella's ``fitDetectorMap`` records, in the ``reduceExposure:adjustDetectorMap``
    metadata section, the species it fitted (``speciesName``) and parallel arrays of the
    error-weighted RMS and the softening per species (``speciesXRms``, ``speciesYRms``,
    ``speciesXSoften``, ``speciesYSoften``), ``Trace`` among them. This reads them into the
    ``{species: (xRms, yRms, xSoften, ySoften)}`` mapping `~pfs.drp.qa.metrics.longFormat`'s
    ``speciesFrame`` takes, the source that replaces the ``Stats for <species>`` log scraping
    ``imageQualityQa`` used to do.

    The statistics are read only when the detectorMap adjustment applied: a fit that later
    failed can leave stale arrays in the metadata, as for the scalar `FIT_METRICS`.

    Parameters
    ----------
    reduceExposureMetadata : `lsst.pipe.base.TaskMetadata` or mapping or `None`
        The ``reduceExposure`` task metadata (converted with `asMapping`).

    Returns
    -------
    `dict` [`str`, `tuple` [`float`, ...]]
        ``{species: (xRms, yRms, xSoften, ySoften)}``; empty when the metadata is absent, the
        adjustment did not apply, or no species were recorded. A value a fit left undefined is
        NaN, so every species has the four `SPECIES_STAT_SUFFIXES` values.
    """
    metadata = asMapping(reduceExposureMetadata)
    if metadata is None:
        return {}
    if not _adjustmentApplied(metadata):
        return {}
    adjust = _section(metadata, "reduceExposure:adjustDetectorMap")
    names = _asList(adjust.get("speciesName"))
    if not names:
        return {}
    columns = [_asList(adjust.get(f"species{suffix}")) for suffix in SPECIES_STAT_SUFFIXES]
    stats: dict[str, tuple[float, ...]] = {}
    for index, species in enumerate(names):
        values = tuple(_toFloat(column[index]) if index < len(column) else None for column in columns)
        stats[str(species)] = tuple(float("nan") if value is None else value for value in values)
    return stats
