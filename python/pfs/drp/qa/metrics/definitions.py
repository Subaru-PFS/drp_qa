"""The metrics ``drp_qa`` gates on, declared once.

Each definition records what external reference the metric is measured against
(R1) and where its thresholds came from (R2). The numbers themselves stay in the
task's config fields so that operators can still override them on the command
line; `buildImageQualityRegistry` folds the config in force into the static
definitions, and the task gates through the result.

Nothing here imports the LSST stack or the Butler.
"""

from collections.abc import Mapping
from typing import Any

from pfs.drp.qa.metrics.registry import MetricDef, MetricRegistry, Thresholds

__all__ = [
    "IQ_FLAG_RATE_FALLBACK",
    "IQ_TRACE_FWHM_FALLBACK",
    "buildImageQualityRegistry",
    "imageQualityMetricDefs",
]

#: Provenance of the thresholds that predate the golden visit set.
#:
#: Say only what is known. These values arrived with the task itself, in commit
#: 8a69c05 (2026-08-11), and **no derivation was recorded anywhere** -- not in
#: the commit message, which explains the threshold structure at length but not
#: the numbers, not in the field docs beyond "Tuned for arm-b", and not in
#: README.md or AGENTS.md, which restate them without justification. Calling
#: them "hand-tuned against engineering-run data" would be a plausible guess
#: dressed as a fact. They are thresholds of unrecorded origin, and R2 applies
#: to them exactly as it does to a new one.
_INHERITED = (
    "ORIGIN UNRECORDED. Present since the task was added (8a69c05, 2026-08-11) with no "
    "derivation recorded in the commit, the field docs, or the project documentation. "
    "Not derived from the golden visit set. Re-derive with "
    "bin.src/calibrateQaThresholds.py; see doc/qa-rebuild-plan.md section 1.2."
)

_INHERITED_FLAG_RATE = (
    "ORIGIN UNRECORDED for the numeric values, as for the FWHM thresholds (8a69c05, "
    "2026-08-11). What IS documented is the shape rather than the magnitudes: the "
    "permissive blue-arm values exist because Ar, Xe and Kr have very few or very faint "
    "lines in the blue, so the global S/N cut flags almost all of them -- lamp physics, "
    "not optics. See AGENTS.md, arc lamp physics. The per-species split is therefore "
    "justified; the specific percentages are not. Not derived from the golden visit set."
)

#: Flag-rate thresholds for an arm with no entry in either config dict. Matches
#: the fallback the task applied before the registry existed.
IQ_FLAG_RATE_FALLBACK = Thresholds(warn=15.0, fail=20.0)

#: Trace/quartz FWHM thresholds for an arm with no entry in either config dict.
#:
#: These are the arc-line numbers, applied to a quantity they were never meant
#: for -- a fiber-profile width is not an arc-line second moment -- and those
#: numbers are themselves of unrecorded origin (see ``_INHERITED``). So this is
#: not "the arc threshold, pending a better one": it is an unjustified value
#: borrowed for an unrelated measurement. It exists so that the trace path has
#: *something* that can fail, which is strictly better than the previous state
#: of having no gate at all, and it should be replaced at the first opportunity.
IQ_TRACE_FWHM_FALLBACK = Thresholds(warn=3.2, fail=3.5)


def imageQualityMetricDefs() -> tuple[MetricDef, ...]:
    """Return the ``imageQualityQa`` metric definitions, with default thresholds.

    Returns
    -------
    `tuple` [`MetricDef`]
        The definitions. The thresholds carried here are the task's config
        defaults; `buildImageQualityRegistry` replaces them with the values
        actually in force.
    """
    return (
        MetricDef(
            name="medFwhm",
            units="pixels",
            reference=(
                "Gaussian-equivalent FWHM from arc-line second moments, or from a "
                "cross-dispersion profile fit to calexp pixels, or from the stored "
                "fiberProfiles widths. The reference is the optical design's expected "
                "spot size, not the frame's own distribution."
            ),
            higherIsWorse=True,
            thresholds=Thresholds(warn=3.2, fail=3.5),
            provenance=f"{_INHERITED} Tuned for arm b (400-650 nm).",
            label="medFWHM",
            unitSuffix="px",
            valueFormat=".2f",
            description="Median FWHM across the detector. A detector-wide median cannot "
            "separate uniform defocus from a tilted focal plane; see plan section 6.1.",
        ),
        MetricDef(
            name="pctFlagged",
            units="percent",
            reference=(
                "The fitDetectorMap flag bits on the arc lines, i.e. the line list and "
                "the fit's own S/N threshold."
            ),
            higherIsWorse=True,
            thresholds=IQ_FLAG_RATE_FALLBACK,
            provenance=_INHERITED_FLAG_RATE,
            unitSuffix="%",
            valueFormat=".1f",
            description="Percentage of arc lines flagged by fitDetectorMap. Gated per "
            "arm and per arm:species, never blended (R7).",
        ),
        MetricDef(
            name="medDxCenter",
            units="pixels",
            reference=(
                "detectorMap_calib, the static calibration product the exposure is "
                "reduced against. A large offset against an out-of-date calib is a "
                "calib problem, not an instrument problem."
            ),
            higherIsWorse=True,
            thresholds=Thresholds(warn=1.0, fail=2.0),
            provenance=_INHERITED,
            label="|dxCenter|",
            unitSuffix="px",
            valueFormat=".3f",
            useAbsolute=True,
            description="Median spatial offset between measured fiber positions and "
            "detectorMap_calib; a flexure diagnostic.",
        ),
        MetricDef(
            name="dxCenterRms",
            units="pixels",
            reference="detectorMap_calib, as for medDxCenter.",
            higherIsWorse=True,
            provenance="Not gated: no threshold has been derived from the golden set.",
            unitSuffix="px",
            valueFormat=".3f",
            description="Scatter of the spatial offset. Large scatter with a near-zero "
            "median is distortion rather than bulk shift.",
        ),
        MetricDef(
            name="fitSpeciesXRms",
            units="pixels",
            reference="The fitDetectorMap solution, per species, from the reduceExposure log.",
            higherIsWorse=True,
            provenance="Not gated: no threshold has been derived from the golden set.",
            unitSuffix="px",
            valueFormat=".4f",
            description="Per-species spatial RMS of the detectorMap fit.",
        ),
        MetricDef(
            name="fitSpeciesYRms",
            units="pixels",
            reference="The fitDetectorMap solution, per species, from the reduceExposure log.",
            higherIsWorse=True,
            provenance="Not gated: no threshold has been derived from the golden set.",
            unitSuffix="px",
            valueFormat=".4f",
            description="Per-species wavelength RMS of the detectorMap fit.",
        ),
    )


def buildImageQualityRegistry(config: Any) -> MetricRegistry:
    """Build the ``imageQualityQa`` registry from the config in force.

    The definitions supply the metadata -- units, direction, reference,
    provenance, how to phrase the reason -- and the config supplies the numbers,
    so that a command-line override such as
    ``-c "imageQualityQa:flagRateWarnThreshold={'b': 50.0}"`` still takes effect.

    Parameters
    ----------
    config : `Any`
        An ``ImageQualityQaConfig``, or any object exposing the same threshold
        attributes. Duck-typed on purpose: the registry is unit-tested without
        ``lsst.pex.config``.

    Returns
    -------
    `MetricRegistry`
        The registry the task gates through.
    """
    byName = {metric.name: metric for metric in imageQualityMetricDefs()}
    return MetricRegistry(
        [
            byName["medFwhm"].withThresholds(
                warn=_attr(config, "fwhmWarnThreshold"),
                fail=_attr(config, "fwhmFailThreshold"),
                # Trace/quartz quanta measure a fiber-profile width, not an
                # arc-line second moment, so they gate on their own keys:
                # ``trace:<arm>`` first, then a bare ``trace`` fallback.
                overrides=_traceOverrides(
                    _attr(config, "traceFwhmWarnThreshold") or {},
                    _attr(config, "traceFwhmFailThreshold") or {},
                ),
            ),
            byName["pctFlagged"].withThresholds(
                warn=IQ_FLAG_RATE_FALLBACK.warn,
                fail=IQ_FLAG_RATE_FALLBACK.fail,
                overrides=_pairOverrides(
                    _attr(config, "flagRateWarnThreshold") or {},
                    _attr(config, "flagRateFailThreshold") or {},
                ),
            ),
            byName["medDxCenter"].withThresholds(
                warn=_attr(config, "dxCenterWarnThreshold"),
                fail=_attr(config, "dxCenterFailThreshold"),
            ),
            byName["dxCenterRms"],
            byName["fitSpeciesXRms"],
            byName["fitSpeciesYRms"],
        ]
    )


def _traceOverrides(warn: Mapping[str, float], fail: Mapping[str, float]) -> dict[str, Thresholds]:
    """Build the trace/quartz FWHM override keys from the per-arm config dicts.

    Parameters
    ----------
    warn : `Mapping` [`str`, `float`]
        WARN thresholds keyed by arm.
    fail : `Mapping` [`str`, `float`]
        FAIL thresholds keyed by arm.

    Returns
    -------
    `dict` [`str`, `Thresholds`]
        A ``trace:<arm>`` entry per configured arm, plus a bare ``trace`` entry
        carrying `IQ_TRACE_FWHM_FALLBACK` for an arm with neither. The bare key
        is what a trace quantum on an unconfigured arm lands on, so it must
        exist or the metric would fall through to the arc-line thresholds
        silently -- the very conflation the separate keys exist to prevent.
    """
    overrides = {"trace": IQ_TRACE_FWHM_FALLBACK}
    for arm in sorted(set(warn) | set(fail)):
        overrides[f"trace:{arm}"] = Thresholds(
            warn=warn.get(arm, IQ_TRACE_FWHM_FALLBACK.warn),
            fail=fail.get(arm, IQ_TRACE_FWHM_FALLBACK.fail),
        )
    return overrides


def _pairOverrides(warn: Mapping[str, float], fail: Mapping[str, float]) -> dict[str, Thresholds]:
    """Zip two per-key threshold dicts into one mapping of `Thresholds`.

    Parameters
    ----------
    warn : `Mapping` [`str`, `float`]
        WARN thresholds, keyed by ``arm`` or ``arm:species``.
    fail : `Mapping` [`str`, `float`]
        FAIL thresholds, keyed the same way.

    Returns
    -------
    `dict` [`str`, `Thresholds`]
        One entry per key present in either dict.

    Notes
    -----
    A key present in only one dict must resolve the other side the way the
    task's original lookup did: ``arm:species`` then ``arm`` then the global
    fallback. Going straight to the fallback silently rewrites a configured
    verdict -- with ``warn={"b": 50, "b:Argon": 93}`` and ``fail={"b": 60}``,
    the FAIL level for ``b:Argon`` is 60, not 20, because the arm entry stands
    in for the species that has none.
    """

    def resolve(source: Mapping[str, float], key: str, fallback: float | None) -> float | None:
        if key in source:
            return source[key]
        arm = key.split(":", 1)[0]
        return source.get(arm, fallback)

    return {
        key: Thresholds(
            warn=resolve(warn, key, IQ_FLAG_RATE_FALLBACK.warn),
            fail=resolve(fail, key, IQ_FLAG_RATE_FALLBACK.fail),
        )
        for key in sorted(set(warn) | set(fail))
    }


def _attr(config: Any, name: str) -> Any:
    """Read a config attribute, tolerating its absence.

    Parameters
    ----------
    config : `Any`
        The config object.
    name : `str`
        Attribute name.

    Returns
    -------
    `Any`
        The value, or ``None`` when the attribute is absent.
    """
    return getattr(config, name, None)
