"""Threshold derivation from the golden visit set.

Implements steps 1-3 of the threshold derivation procedure in
``doc/qa-rebuild-plan.md`` section 1.2:

1. run the metric over the ``known_good`` visits with no gating;
2. take the distribution of the metric across all good detectors;
3. WARN at the 95th percentile, FAIL at the 99th, rounded to a readable value.

Step 4 (verify the ``known_bad`` visits exceed FAIL) is `verifyKnownBad`, and
step 5 (record the provenance) is the ``provenance`` string this module builds
for you to paste into the config field's ``doc``.

This module deliberately does not know what a metric *is*. It takes the values
and returns numbers, so it can be unit-tested with no Butler and no stack.
Butler access lives in ``bin.src/calibrateQaThresholds.py``.
"""

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "ThresholdSuggestion",
    "deriveThresholds",
    "formatProvenance",
    "roundToReadable",
    "verifyKnownBad",
]

#: Percentile of the known-good distribution used for the WARN threshold.
DEFAULT_WARN_PERCENTILE = 95.0

#: Percentile of the known-good distribution used for the FAIL threshold.
DEFAULT_FAIL_PERCENTILE = 99.0

#: Below this many good samples a suggestion is reported but marked unreliable.
MIN_SAMPLES = 20


@dataclass(frozen=True)
class ThresholdSuggestion:
    """A WARN/FAIL pair derived from a known-good distribution.

    Attributes
    ----------
    metric : `str`
        Name of the metric the suggestion is for.
    warn : `float`
        Suggested WARN threshold, rounded to a readable value.
    fail : `float`
        Suggested FAIL threshold, rounded to a readable value.
    warnRaw : `float`
        The unrounded percentile the WARN threshold came from.
    failRaw : `float`
        The unrounded percentile the FAIL threshold came from.
    median : `float`
        Median of the known-good distribution, for context.
    robustRms : `float`
        Robust scatter of the known-good distribution, for context.
    numSamples : `int`
        How many finite values the distribution contained.
    higherIsWorse : `bool`
        Direction of the metric. When False the percentiles are taken from the
        lower tail, so that ``warn``/``fail`` are still the values a bad
        detector exceeds *in the bad direction*.
    reliable : `bool`
        False when fewer than `MIN_SAMPLES` finite values were available. A
        threshold from a handful of detectors is a guess with a decimal point.
    degenerate : `bool`
        True when WARN and FAIL round to the same value. The pair is then
        unusable: `MetricDef.gate` tests FAIL first, so an equal WARN can never
        fire and the metric silently loses its warning level. It happens when
        the known-good distribution is tight enough that p95 and p99 fall inside
        one rounding step -- more digits would not fix it, only disguise a band
        narrower than the scatter.
    provenance : `str`
        Human-readable record of where the numbers came from, to be pasted into
        the config field's ``doc`` string (rule R2).
    """

    metric: str
    warn: float
    fail: float
    warnRaw: float
    failRaw: float
    median: float
    robustRms: float
    numSamples: int
    higherIsWorse: bool
    reliable: bool
    provenance: str
    degenerate: bool = False

    def __str__(self) -> str:
        """Return a one-line summary suitable for CLI output."""
        flag = "" if self.reliable else "  [UNRELIABLE: too few samples]"
        if self.degenerate:
            flag += "  [DEGENERATE: WARN == FAIL, so WARN can never fire]"
        return (
            f"{self.metric:<24s} warn={self.warn:<10.4g} fail={self.fail:<10.4g} "
            f"(raw {self.warnRaw:.4g}/{self.failRaw:.4g}, "
            f"median={self.median:.4g}, robustRms={self.robustRms:.4g}, n={self.numSamples}){flag}"
        )


def roundToReadable(value: float, significant: int = 2) -> float:
    """Round a threshold to a value a human would have chosen.

    A threshold carried to full floating-point precision implies a precision the
    derivation does not have: the 95th percentile of a few hundred detectors is
    not determined to six digits. Rounding also makes config diffs readable.

    Parameters
    ----------
    value : `float`
        The raw threshold.
    significant : `int`, optional
        Number of significant digits to keep. Default is 2.

    Returns
    -------
    `float`
        The rounded value. Zero and non-finite inputs are returned unchanged.
    """
    if value == 0 or not math.isfinite(value):
        return value
    magnitude = math.floor(math.log10(abs(value)))
    digits = significant - 1 - magnitude
    return round(value, digits)


def deriveThresholds(
    values: ArrayLike,
    metric: str,
    higherIsWorse: bool = True,
    warnPercentile: float = DEFAULT_WARN_PERCENTILE,
    failPercentile: float = DEFAULT_FAIL_PERCENTILE,
    physicalLimit: float | None = None,
    visitRange: str | None = None,
    derivedOn: date | None = None,
    significant: int = 2,
) -> ThresholdSuggestion:
    """Suggest WARN and FAIL thresholds from a known-good distribution.

    Parameters
    ----------
    values : array-like
        The metric's values over the ``known_good`` visits, one per detector or
        per (detector, species). Non-finite values are dropped.
    metric : `str`
        Name of the metric, used for reporting.
    higherIsWorse : `bool`, optional
        True when large values indicate a problem (FWHM, residual RMS). False
        when small values do (a line count, a good-fraction). Default is True.
    warnPercentile : `float`, optional
        Percentile for the WARN threshold. Default is 95.
    failPercentile : `float`, optional
        Percentile for the FAIL threshold. Default is 99.
    physicalLimit : `float`, optional
        A hard physical limit (saturation level, fiber pitch). When given, it
        overrides the FAIL percentile, since a measured percentile of good data
        cannot be a better bound than physics.
    visitRange : `str`, optional
        The visit range the values came from, recorded in ``provenance``.
    derivedOn : `datetime.date`, optional
        Derivation date, recorded in ``provenance``. Defaults to today.
    significant : `int`, optional
        Significant digits to round the thresholds to. Default is 2.

    Returns
    -------
    `ThresholdSuggestion`
        The suggested thresholds and the context needed to justify them.

    Raises
    ------
    ValueError
        If no finite values are supplied, or if the percentiles are out of
        range or in the wrong order.

    Notes
    -----
    When ``higherIsWorse`` is False the percentiles are reflected, so that the
    95th percentile becomes the 5th: the returned thresholds are always values
    that a *bad* detector is on the wrong side of.
    """
    for name, percentile in (("warnPercentile", warnPercentile), ("failPercentile", failPercentile)):
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"{name}={percentile} is not a percentile in [0, 100]")
    if failPercentile < warnPercentile:
        raise ValueError(
            f"failPercentile={failPercentile} is below warnPercentile={warnPercentile}; "
            "FAIL must be the more extreme of the two"
        )

    values = np.asarray(values, dtype=float).ravel()
    good = values[np.isfinite(values)]
    if good.size == 0:
        raise ValueError(f"No finite values for metric {metric!r}; nothing to derive a threshold from")

    # Reflect the percentiles for a metric whose bad direction is downwards, so
    # the caller always gets "the value a bad detector is beyond".
    warnQ = warnPercentile if higherIsWorse else 100.0 - warnPercentile
    failQ = failPercentile if higherIsWorse else 100.0 - failPercentile

    warnRaw = float(np.percentile(good, warnQ))
    failRaw = float(np.percentile(good, failQ))

    if physicalLimit is not None:
        failRaw = float(physicalLimit)

    warn = roundToReadable(warnRaw, significant)
    fail = roundToReadable(failRaw, significant)

    degenerate = warn == fail
    median = float(np.median(good))
    q25, q75 = (float(x) for x in np.percentile(good, [25.0, 75.0]))
    robustRms = 0.741 * (q75 - q25)

    return ThresholdSuggestion(
        metric=metric,
        warn=warn,
        fail=fail,
        warnRaw=warnRaw,
        failRaw=failRaw,
        median=median,
        robustRms=robustRms,
        numSamples=int(good.size),
        higherIsWorse=higherIsWorse,
        reliable=good.size >= MIN_SAMPLES,
        degenerate=degenerate,
        provenance=formatProvenance(
            visitRange=visitRange,
            derivedOn=derivedOn,
            numSamples=int(good.size),
            # The reflected quantiles, not the requested ones: for a metric that
            # fails low these are p5/p1, and recording p95/p99 would make the
            # provenance R2 requires describe a derivation that never happened.
            warnPercentile=warnQ,
            failPercentile=failQ,
            physicalLimit=physicalLimit,
            reliable=good.size >= MIN_SAMPLES,
            degenerate=degenerate,
        ),
    )


def formatProvenance(
    visitRange: str | None,
    derivedOn: date | None,
    numSamples: int,
    warnPercentile: float,
    failPercentile: float,
    physicalLimit: float | None = None,
    reliable: bool = True,
    degenerate: bool = False,
) -> str:
    """Build the provenance sentence for a config field's ``doc`` string.

    Rule R2 requires every threshold to record the visit range and the date it
    was derived from. This produces that sentence in a uniform form so it can be
    grepped for later.

    Parameters
    ----------
    visitRange : `str` or `None`
        The visit range the values came from.
    derivedOn : `datetime.date` or `None`
        Derivation date. Defaults to today.
    numSamples : `int`
        How many finite values the distribution contained.
    warnPercentile : `float`
        Percentile used for WARN.
    failPercentile : `float`
        Percentile used for FAIL.
    physicalLimit : `float`, optional
        The physical limit used for FAIL, if any.
    degenerate : `bool`, optional
        True when WARN and FAIL rounded to the same value, which silently costs
        the metric its warning level.
    reliable : `bool`, optional
        False when the sample was too small to stand behind. The sentence then
        carries that warning, which is the whole point: a threshold derived from
        16 detectors and one derived from 400 must not read identically once
        they are sitting in a config file. Someone reading the field a year from
        now sees only this string.

    Returns
    -------
    `str`
        A single sentence, e.g. ``"Derived 2026-09-17 from golden visits
        140200-140260 (n=384): WARN at p95, FAIL at p99."`` -- with an explicit
        NOT RELIABLE clause appended when ``reliable`` is False.
    """
    derivedOn = derivedOn or date.today()
    source = f"golden visits {visitRange}" if visitRange else "the golden visit set"
    failClause = (
        f"FAIL at the physical limit {physicalLimit:g}"
        if physicalLimit is not None
        else f"FAIL at p{failPercentile:g}"
    )
    sentence = (
        f"Derived {derivedOn.isoformat()} from {source} (n={numSamples}): "
        f"WARN at p{warnPercentile:g}, {failClause}."
    )
    if degenerate:
        sentence += (
            " DEGENERATE: WARN and FAIL round to the same value, so WARN can never fire --"
            " the gate tests FAIL first. Widen the pair by hand, or set FAIL from a physical"
            " limit, rather than carrying a metric with no warning level."
        )
    if not reliable:
        sentence += (
            f" NOT RELIABLE: n={numSamples} is below the {MIN_SAMPLES}-sample floor, so this"
            " is indicative only and must not be treated as derived. Replace it once more"
            " data exists; until then it is a considered guess, not a measurement."
        )
    return sentence


def verifyKnownBad(
    values: ArrayLike,
    suggestion: ThresholdSuggestion,
) -> tuple[bool, str]:
    """Check that known-bad values fall beyond the suggested FAIL threshold.

    This is step 4 of the procedure. A threshold that the known-bad data does
    not cross is not a threshold; it is a number in a config file.

    Parameters
    ----------
    values : array-like
        The metric's values over the ``known_bad`` visits. Non-finite values are
        dropped.
    suggestion : `ThresholdSuggestion`
        The thresholds under test.

    Returns
    -------
    ok : `bool`
        True when every finite known-bad value is beyond FAIL.
    message : `str`
        A human-readable summary, listing how many values crossed.

    Raises
    ------
    ValueError
        If no finite values are supplied. "No known-bad data" is not a pass;
        the caller must decide what to do about it.
    """
    values = np.asarray(values, dtype=float).ravel()
    bad = values[np.isfinite(values)]
    if bad.size == 0:
        raise ValueError(
            f"No finite known-bad values for metric {suggestion.metric!r}; cannot verify the FAIL threshold"
        )

    crossed = bad >= suggestion.fail if suggestion.higherIsWorse else bad <= suggestion.fail
    numCrossed = int(np.count_nonzero(crossed))
    ok = numCrossed == bad.size
    direction = ">=" if suggestion.higherIsWorse else "<="
    return ok, (
        f"{suggestion.metric}: {numCrossed}/{bad.size} known-bad values "
        f"{direction} FAIL={suggestion.fail:g}"
        + ("" if ok else "  <-- threshold does not separate the known-bad data")
    )
