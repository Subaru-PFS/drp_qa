# QA principles

Rules for adding or changing a QA metric in `drp_qa`, the procedure for deriving its
thresholds, and the checklist a metric's PR answers. The rules encode failures that have
happened here; treat them as binding.

PFS QA answers one question per visit: *is this data good enough to reduce, and if not,
what is wrong with the instrument?* A metric that cannot separate a good detector from a
bad one is worse than none, because it teaches operators to ignore `qaStatus`.

## Rules

### R1 — Every metric needs an external reference

A statistic compared against a threshold derived from the same sample is self-referential
and pinned near a constant:

- the fraction of values above their own 95th percentile is always about 5 %;
- `mean(x) - median(x)` of one sample is always about 0;
- the ratio of the faint decile to the bright decile of one flux distribution is a
  property of the lamp's line list, not of the instrument.

The reference must be external: a calibration product (`detectorMap_calib`,
`fiberProfiles`), a detector constant (saturation level, gain, read noise), a physical
constant, or the same measurement from another epoch.

### R2 — Thresholds are measured, never invented

No threshold enters the code before it has been derived from known-good data by the
procedure in [`qa-thresholds.ipynb`](qa-thresholds.ipynb). Its provenance, in the
thresholds file or the config field's `doc`, records the visits and the date it was derived
from. A threshold whose origin is unknown
says so ("origin unrecorded"), rather than guessing a history.

### R3 — Separate measurement from judgement

Tasks emit numbers; a separate gating step (`pfs.drp.qa.metrics.gate`) maps them to
`PASS`/`WARN`/`FAIL`. Thresholds can then be re-tuned and re-applied to stored metrics
without re-running the pipeline over the pixels, which is what makes R2 practical.

### R4 — Never duplicate an existing metric

Before adding a metric, look in `dmQaResidualStats` and `iqQaMetrics` for a column that
measures the same thing. `get_fit_stats` in `dmResiduals.py` already gives error-weighted
and robust RMS values; an unweighted reimplementation is a regression, not an addition.

### R5 — Robust statistics only

Use `np.median` and `pfs.drp.stella.utils.math.robustRms`. Arc line fluxes span orders of
magnitude and centroid residuals have outliers from mismatched lines; `np.mean` and
`np.std` are unusable on either without clipping.

### R6 — Separate measurement from presentation

The pipeline runs under `ics_qaActor`, where per-visit latency is visible. It computes and
stores data; rendering happens later, in a notebook or the dashboard, from what was
stored. So a dashboard can only draw what the pipeline stored: store the data a detail
plot needs, not only scalar summaries. The plotting functions themselves are first-class,
tested code in `pfs.drp.qa.plotting`.

### R7 — Per-species, never blended

Arc lamps have very different line densities and intensities in each arm (see
[`qa-domain-notes.md`](qa-domain-notes.md): Ar and Xe are nearly absent in the blue, HgCd
is dense). A metric aggregated across species measures the species mix. Compute and gate
per species. The same holds for any mixture of populations: arms, and estimators (arc
moments, calexp trace widths, sky frames).

## Thresholds and the validation visit set

- [`validation-visits.md`](validation-visits.md): the visits with known verdicts that every
  threshold and metric is checked against, and how to change the set.
- [`qa-thresholds.ipynb`](qa-thresholds.ipynb): the step-by-step procedure, as a notebook,
  from running `imageQualityQa` over those visits to adopting the thresholds.

In short: `WARN` at the 95th and `FAIL` at the 99th percentile of the reference run's
(Run25's) known-good values,
separately for each population (arm, observation type, lamp), rounded outwards to a step
well under the scatter, or `FAIL` at a physical limit. Every known-bad visit must reach
the verdict it expects, and other runs are compared against the result. Offsets from a
calibration (`medDxCenter`) are not derived this way: they are near zero in the run the
calibration came from, so their thresholds are a tolerance.

## Metric checklist

State in the PR that adds or changes a metric:

1. The external reference it is measured against (R1).
2. Where its thresholds came from: the provenance sentence (R2).
3. The existing columns it does not duplicate (R4).
4. Its result on the validation visit set: every `known_good` passes, every `known_bad`
   naming it reaches its expected verdict.
