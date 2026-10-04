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
[procedure](#deriving-thresholds) below. The config field's `doc` records the visits and
the date it was derived from. A threshold whose origin is unknown says so
("origin unrecorded"), rather than guessing a history.

### R3 — Separate measurement from judgement

Tasks emit numbers; a separate gating step maps them to `PASS`/`WARN`/`FAIL`. Thresholds
can then be re-tuned and re-applied to stored metrics without re-running the pipeline over
the pixels, which is what makes R2 practical.

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

## The validation visit set

[`python/pfs/drp/qa/metrics/data/validationVisits.yaml`](../python/pfs/drp/qa/metrics/data/validationVisits.yaml)
lists visits with known verdicts: `known_good` entries expected to pass every metric, and
`known_bad` entries expected to `WARN` or `FAIL`, each naming the metric that should catch
it. A metric that flags a `known_good` visit, or passes a `known_bad` one, does not merge.

- Only visits with a verdict belong. A run's calibration block or a drift series carries
  none; select those from the Butler when needed.
- A suspected but unestablished verdict is `unconfirmed: true`: reported, never decisive.
- An entry awaiting a visit number is `placeholder: true` and is not loaded.
- The file is public. Use engineering and calibration visits only, identified by visit
  number.

`pfs.drp.qa.metrics.validationVisits` loads it strictly: a malformed entry, an unknown key
or a `known_good` entry expecting anything but `PASS` is an error.

## Deriving thresholds

`pfs.drp.qa.metrics.calibration.calibrate` does steps 2–5 and returns a table, one row per
metric and population; `pfs.drp.qa.plotting.plotThresholds` draws it, and
[`qa-thresholds.ipynb`](qa-thresholds.ipynb) runs both on the stored `iqQaMetrics` of the
validation visits.

1. Run the metric over the validation visits with no gating.
2. Split the known-good values into populations (by default arm and `obsType`; flag rates
   also by species, line counts by lamp) and take each distribution.
3. `WARN` at the 95th percentile and `FAIL` at the 99th, rounded *outwards* (so rounding
   never flags more of the good data) to the power of ten between a hundredth and a tenth
   of the good scatter. Rounding to significant figures instead would round an FWHM of
   2.7 px in 0.1 px steps, wider than the gap between p95 and p99. Where a physical limit
   exists (saturation, fiber pitch), `FAIL` is that limit, exactly.
4. Check that each `known_bad` value reaches the verdict its entry expects.
5. Copy the provenance sentence into the config field's `doc`.

Judge each suggestion before using it:

- **n and visits.** Fewer than 20 values is unreliable. Detectors of one visit and
  back-to-back exposures are correlated, so the number of visits is the more honest size.
- **The interval on FAIL.** A distribution-free 95 % interval on the FAIL percentile. When
  the sample cannot bound it (about 370 values are needed for p99), FAIL sits at the
  sample's extreme and more data will move it.
- **In-sample flag rates.** What the rounded thresholds flag of the good data; near 5 % and
  1 % by construction, more if the distribution has ties.
- **Degenerate pairs.** If WARN is not below FAIL, WARN can never fire. It happens with ties
  in the tail (a metric that is mostly zero) or a physical limit inside the good data.
- **Populations without good data.** Known-bad rows there are listed but cannot be
  checked.

## Metric checklist

State in the PR that adds or changes a metric:

1. The external reference it is measured against (R1).
2. Where its thresholds came from: the provenance sentence (R2).
3. The existing columns it does not duplicate (R4).
4. Its result on the validation visit set: every `known_good` passes, every `known_bad`
   naming it reaches its expected verdict.
