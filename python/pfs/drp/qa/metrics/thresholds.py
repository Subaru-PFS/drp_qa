"""Threshold derivation from the validation visit set.

The procedure, documented in ``docs/qa-principles.md``:

1. run the metric over the ``known_good`` visits with no gating;
2. take its distribution over the good detectors, one population at a time;
3. WARN at the 95th percentile and FAIL at the 99th, rounded outwards to a
   step under a tenth of the good scatter, or FAIL at a physical limit;
4. check that each ``known_bad`` visit reaches the verdict it expects (`verifyKnownBad`);
5. record the provenance in the config field's ``doc`` (`formatProvenance`).

This module does not know what a metric *is*. It takes values and returns
numbers, so it is tested with no Butler and no stack. Butler access lives in
`pfs.drp.qa.metrics.readers`.
"""

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import binom

__all__ = [
    "MIN_SAMPLES",
    "ThresholdSuggestion",
    "deriveThresholds",
    "formatProvenance",
    "quantileInterval",
    "resolutionFor",
    "roundToReadable",
    "roundToStep",
    "verifyKnownBad",
]

#: Percentile of the known-good distribution used for the WARN threshold.
DEFAULT_WARN_PERCENTILE = 95.0

#: Percentile of the known-good distribution used for the FAIL threshold.
DEFAULT_FAIL_PERCENTILE = 99.0

#: Below this many good samples a suggestion is reported but marked unreliable.
MIN_SAMPLES = 20

#: Confidence of the interval reported for the FAIL percentile.
CONFIDENCE = 0.95


@dataclass(frozen=True)
class ThresholdSuggestion:
    """A WARN/FAIL pair derived from a known-good distribution.

    Attributes
    ----------
    metric : `str`
        Name of the metric (and group) the suggestion is for.
    warn, fail : `float`
        Suggested thresholds, rounded outwards (away from the good data, so
        rounding never makes a threshold flag more of it) to `resolutionFor`
        the good scatter.
    warnRaw, failRaw : `float`
        The unrounded percentiles, or the physical limit for ``failRaw``.
    failInterval : `tuple` [`float`, `float`]
        Distribution-free confidence interval (`CONFIDENCE`) on the FAIL
        percentile, from order statistics. An infinite end means the sample is
        too small to bound the percentile on that side.
    median, robustRms : `float`
        Centre and robust scatter of the known-good distribution.
    numSamples : `int`
        Number of finite values in the distribution.
    higherIsWorse : `bool`
        Direction of the metric. When False the percentiles are reflected, so
        ``warn`` and ``fail`` are still the values a bad detector is beyond.
    reliable : `bool`
        False when fewer than `MIN_SAMPLES` finite values were available.
    failBounded : `bool`
        False when the sample cannot bound the FAIL percentile on its bad side:
        The confidence interval on the FAIL percentile is then open on that
        side: no order statistic of the sample bounds it, so FAIL is an
        interpolated estimate near the sample's tail that more data may move
        a long way.
        Always True for a physical limit.
    degenerate : `bool`
        True when WARN is not strictly less severe than FAIL. The gates test
        FAIL first, so such a WARN can never fire. With scatter-based rounding
        it means ties in the tail, or a physical limit inside the good data.
    goodFlaggedWarn, goodFlaggedFail : `float`
        Fraction of the known-good values the rounded thresholds would flag at
        WARN-or-worse and at FAIL: the in-sample false-alarm rates.
    physicalLimit : `float` or `None`
        The physical limit used for FAIL, if any.
    provenance : `str`
        The sentence for the config field's ``doc`` (rule R2).
    """

    metric: str
    warn: float
    fail: float
    warnRaw: float
    failRaw: float
    failInterval: tuple[float, float]
    median: float
    robustRms: float
    numSamples: int
    higherIsWorse: bool
    reliable: bool
    failBounded: bool
    degenerate: bool
    goodFlaggedWarn: float
    goodFlaggedFail: float
    physicalLimit: float | None
    provenance: str

    def __str__(self) -> str:
        """Return a short summary for CLI output."""
        low, high = self.failInterval
        lines = [
            f"{self.metric}: warn={self.warn:.4g} fail={self.fail:.4g} "
            f"(raw {self.warnRaw:.4g}/{self.failRaw:.4g}; "
            f"{CONFIDENCE:.0%} CI on the FAIL percentile [{low:.4g}, {high:.4g}])",
            f"    good: n={self.numSamples} median={self.median:.4g} robustRms={self.robustRms:.4g}; "
            f"flagged {self.goodFlaggedWarn:.1%} at WARN+, {self.goodFlaggedFail:.1%} at FAIL",
        ]
        if not self.reliable:
            lines.append(f"    UNRELIABLE: n={self.numSamples} is below the {MIN_SAMPLES}-sample floor")
        if not self.failBounded:
            lines.append("    FAIL UNBOUNDED: too few samples to bound the FAIL percentile")
        if self.degenerate:
            lines.append("    DEGENERATE: WARN is not less severe than FAIL, so WARN can never fire")
        return "\n".join(lines)


def roundToReadable(value: float, significant: int = 2, direction: int = 0) -> float:
    """Round a threshold to a value a human would have chosen.

    The 95th percentile of a few hundred detectors is not known to six digits,
    and rounded values make config diffs readable.

    Parameters
    ----------
    value : `float`
        The raw threshold.
    significant : `int`, optional
        Number of significant digits to keep. Default is 2.
    direction : `int`, optional
        0 rounds to nearest, a positive value rounds up and a negative value
        rounds down.

    Returns
    -------
    `float`
        The rounded value. Zero and non-finite inputs are returned unchanged.
    """
    if value == 0 or not math.isfinite(value):
        return value
    digits = significant - 1 - math.floor(math.log10(abs(value)))
    if direction == 0:
        return round(value, digits)
    scale = 10.0**digits
    # Round the scaled value first, so that 3.5 * 10 = 35.000000000000004 is
    # not taken up to 36.
    scaled = round(value * scale, 9)
    return (math.ceil(scaled) if direction > 0 else math.floor(scaled)) / scale


def roundToStep(value: float, step: float, direction: int = 0) -> float:
    """Round a value to a multiple of ``step``.

    Parameters
    ----------
    value : `float`
        The value.
    step : `float`
        The resolution, a positive power of ten in practice.
    direction : `int`, optional
        0 rounds to nearest, a positive value rounds up and a negative value
        rounds down.

    Returns
    -------
    `float`
        The rounded value; non-finite inputs are returned unchanged.
    """
    if not math.isfinite(value):
        return value
    # Round the quotient first, so that 3.5 / 0.1 = 35.00000000000001 is not
    # taken up to 36.
    quotient = round(value / step, 9)
    rounded = (
        round(quotient) if direction == 0 else math.ceil(quotient) if direction > 0 else math.floor(quotient)
    )
    # Format through the step's decimals so that 28 * 0.1 prints as 2.8.
    decimals = max(0, -math.floor(math.log10(step)))
    return round(rounded * step, decimals)


def resolutionFor(robustRms: float) -> float | None:
    """Return the rounding step for thresholds on a distribution.

    The power of ten between a hundredth and a tenth of the robust scatter.
    For a normal distribution p95 and p99 are 0.68 sigma apart, so this keeps
    them at least six steps apart while rounding moves a threshold by under a
    tenth of the scatter. A step tied to the value's magnitude instead (two
    significant figures) rounds an FWHM of 2.7 px in 0.1 px steps when p95 and
    p99 are 0.04 px apart, and WARN collapses onto FAIL.

    Parameters
    ----------
    robustRms : `float`
        Robust scatter of the known-good distribution.

    Returns
    -------
    `float` or `None`
        The step, or `None` when the scatter is zero or not finite (a
        distribution of ties), where there is no scale to round to.
    """
    if not math.isfinite(robustRms) or robustRms <= 0:
        return None
    return 10.0 ** math.floor(math.log10(robustRms / 10.0))


def quantileInterval(
    values: ArrayLike, quantile: float, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """Return a distribution-free confidence interval on a quantile.

    The number of samples below the true ``quantile`` is binomial, so a pair of
    order statistics brackets it with known probability whatever the
    distribution. Percentiles of tails are where a parametric interval would
    be least trustworthy.

    Parameters
    ----------
    values : array-like
        The sample. Non-finite values are dropped.
    quantile : `float`
        The quantile, in [0, 1].
    confidence : `float`, optional
        Coverage of the interval. Default is `CONFIDENCE`.

    Returns
    -------
    low, high : `float`
        The interval. An end is infinite when the sample is too small to bound
        the quantile on that side at this confidence.
    """
    data = np.sort(np.asarray(values, dtype=float).ravel())
    data = data[np.isfinite(data)]
    n = data.size
    if n == 0:
        return (-math.inf, math.inf)
    alpha = 1.0 - confidence
    # 1-based ranks: the interval [x_(lower), x_(upper)] covers the quantile
    # with probability >= confidence.
    lower = int(binom.ppf(alpha / 2, n, quantile))
    upper = int(binom.ppf(1.0 - alpha / 2, n, quantile)) + 1
    low = float(data[lower - 1]) if lower >= 1 else -math.inf
    high = float(data[upper - 1]) if upper <= n else math.inf
    return (low, high)


def deriveThresholds(
    values: ArrayLike,
    metric: str,
    higherIsWorse: bool = True,
    warnPercentile: float = DEFAULT_WARN_PERCENTILE,
    failPercentile: float = DEFAULT_FAIL_PERCENTILE,
    physicalLimit: float | None = None,
    visitRange: str | None = None,
    derivedOn: date | None = None,
    resolution: float | None = None,
) -> ThresholdSuggestion:
    """Suggest WARN and FAIL thresholds from a known-good distribution.

    Parameters
    ----------
    values : array-like
        The metric's values over the ``known_good`` visits, one per detector.
        Non-finite values are dropped. Pass one population at a time: pooling
        arms, lamps or estimators blends distributions whose tails differ.
    metric : `str`
        Name of the metric, used for reporting.
    higherIsWorse : `bool`, optional
        True when large values indicate a problem (FWHM, residual RMS), False
        when small values do (a line count). Default is True.
    warnPercentile, failPercentile : `float`, optional
        Percentiles for WARN and FAIL. Defaults are 95 and 99.
    physicalLimit : `float`, optional
        A hard physical limit (saturation level, fiber pitch). When given it is
        FAIL, exactly and unrounded: no percentile of good data bounds better
        than physics, and rounding a limit moves it past what it limits.
    visitRange : `str`, optional
        The visit range the values came from, recorded in ``provenance``.
    derivedOn : `datetime.date`, optional
        Derivation date, recorded in ``provenance``. Defaults to today.
    resolution : `float`, optional
        Step to round the thresholds to. Defaults to `resolutionFor` the
        good scatter; with no scatter, two significant figures.

    Returns
    -------
    `ThresholdSuggestion`
        The thresholds and what is needed to judge them.

    Raises
    ------
    ValueError
        If there are no finite values, or the percentiles are out of range or
        in the wrong order.
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
    outwards = 1 if higherIsWorse else -1

    q25, q75 = (float(x) for x in np.percentile(good, [25.0, 75.0]))
    robustRms = 0.741 * (q75 - q25)
    step = resolution if resolution is not None else resolutionFor(robustRms)

    def outwardsRound(value: float) -> float:
        if step is None:
            return roundToReadable(value, 2, outwards)
        return roundToStep(value, step, outwards)

    warnRaw = float(np.percentile(good, warnQ))
    warn = outwardsRound(warnRaw)
    if physicalLimit is not None:
        failRaw = fail = float(physicalLimit)
        failBounded = True
    else:
        failRaw = float(np.percentile(good, failQ))
        fail = outwardsRound(failRaw)
    failInterval = quantileInterval(good, failQ / 100.0)
    if physicalLimit is None:
        failBounded = math.isfinite(failInterval[1] if higherIsWorse else failInterval[0])

    if higherIsWorse:
        degenerate = warn >= fail
        flaggedWarn, flaggedFail = good >= warn, good >= fail
    else:
        degenerate = warn <= fail
        flaggedWarn, flaggedFail = good <= warn, good <= fail

    reliable = good.size >= MIN_SAMPLES

    return ThresholdSuggestion(
        metric=metric,
        warn=warn,
        fail=fail,
        warnRaw=warnRaw,
        failRaw=failRaw,
        failInterval=failInterval,
        median=float(np.median(good)),
        robustRms=robustRms,
        numSamples=int(good.size),
        higherIsWorse=higherIsWorse,
        reliable=reliable,
        failBounded=failBounded,
        degenerate=degenerate,
        goodFlaggedWarn=float(np.mean(flaggedWarn)),
        goodFlaggedFail=float(np.mean(flaggedFail)),
        physicalLimit=physicalLimit,
        provenance=formatProvenance(
            visitRange=visitRange,
            derivedOn=derivedOn,
            numSamples=int(good.size),
            # The reflected quantiles: for a metric that fails low these are
            # p5/p1, and the provenance must describe what was done.
            warnPercentile=warnQ,
            failPercentile=failQ,
            physicalLimit=physicalLimit,
            reliable=reliable,
            failBounded=failBounded,
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
    failBounded: bool = True,
    degenerate: bool = False,
) -> str:
    """Build the provenance sentence for a config field's ``doc`` string.

    Rule R2 requires every threshold to record the visits and date it was
    derived from. Someone reading the field later sees only this sentence, so
    it also carries any reason not to trust the numbers.

    Parameters
    ----------
    visitRange : `str` or `None`
        The visit range the values came from.
    derivedOn : `datetime.date` or `None`
        Derivation date. Defaults to today.
    numSamples : `int`
        Number of finite values in the distribution.
    warnPercentile, failPercentile : `float`
        Percentiles used for WARN and FAIL.
    physicalLimit : `float`, optional
        The physical limit used for FAIL, if any.
    reliable : `bool`, optional
        False when the sample is below `MIN_SAMPLES`.
    failBounded : `bool`, optional
        False when the sample cannot bound the FAIL percentile.
    degenerate : `bool`, optional
        True when WARN is not less severe than FAIL.

    Returns
    -------
    `str`
        E.g. ``"Derived 2026-09-17 from validation visits 133025-135850
        (n=384): WARN at p95, FAIL at p99."``, followed by any warnings.
    """
    derivedOn = derivedOn or date.today()
    source = f"validation visits {visitRange}" if visitRange else "the validation visit set"
    failClause = (
        f"FAIL at the physical limit {physicalLimit:g}"
        if physicalLimit is not None
        else f"FAIL at p{failPercentile:g}"
    )
    sentence = (
        f"Derived {derivedOn.isoformat()} from {source} (n={numSamples}): "
        f"WARN at p{warnPercentile:g}, {failClause}."
    )
    if not reliable:
        sentence += f" NOT RELIABLE: n={numSamples} is below the {MIN_SAMPLES}-sample floor."
    if not failBounded:
        sentence += (
            f" FAIL UNBOUNDED: n={numSamples} cannot bound p{failPercentile:g} at"
            f" {CONFIDENCE:.0%} confidence: its interval is open, and FAIL may move a long way with more data."
        )
    if degenerate:
        sentence += " DEGENERATE: WARN is not less severe than FAIL, so WARN can never fire."
    return sentence


def verifyKnownBad(
    values: ArrayLike, suggestion: ThresholdSuggestion, expect: str = "FAIL"
) -> tuple[bool, str]:
    """Check that known-bad values reach the verdict their entries expect.

    Parameters
    ----------
    values : array-like
        The metric's values over the ``known_bad`` visits that name it and
        expect ``expect``. Non-finite values are dropped.
    suggestion : `ThresholdSuggestion`
        The thresholds under test.
    expect : `str`, optional
        ``FAIL`` (the default) or ``WARN``: the level each value must reach,
        with the comparison the gates make (``>=``, or ``<=`` for a metric that
        fails low). A WARN entry that reaches FAIL still passes the check.

    Returns
    -------
    ok : `bool`
        True when every finite value reaches the expected level.
    message : `str`
        A summary: how many values reached FAIL, and how many WARN.

    Raises
    ------
    ValueError
        If there are no finite values ("no known-bad data" is not a pass), or
        ``expect`` is not WARN or FAIL.
    """
    if expect not in ("WARN", "FAIL"):
        raise ValueError(f"expect={expect!r}; a known-bad entry expects WARN or FAIL")
    values = np.asarray(values, dtype=float).ravel()
    bad = values[np.isfinite(values)]
    if bad.size == 0:
        raise ValueError(
            f"No finite known-bad values for metric {suggestion.metric!r}; cannot verify the {expect} threshold"
        )

    if suggestion.higherIsWorse:
        crossedFail, crossedWarn, direction = bad >= suggestion.fail, bad >= suggestion.warn, ">="
    else:
        crossedFail, crossedWarn, direction = bad <= suggestion.fail, bad <= suggestion.warn, "<="
    numFail = int(np.count_nonzero(crossedFail))
    numWarn = int(np.count_nonzero(crossedWarn))
    ok = (numFail if expect == "FAIL" else numWarn) == bad.size
    return ok, (
        f"{suggestion.metric}: {bad.size} known-bad values expecting {expect}: "
        f"{numFail} {direction} FAIL={suggestion.fail:g}, {numWarn} {direction} WARN={suggestion.warn:g}"
        + ("" if ok else f"  <-- threshold does not give the expected {expect}")
    )
