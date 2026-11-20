# QA principles

Rules for adding or changing a QA metric or check in `drp_qa`, the procedure for setting a
limit, and the checklist a PR answers. The rules encode failures that have happened here;
treat them as binding.

PFS QA answers one question per visit: *is this data good enough to reduce, and if not,
what is wrong with the instrument?* A metric that cannot separate a good detector from a
bad one is worse than none, because it teaches operators to ignore the verdict.

## Rules

### R1 — Every metric needs an external reference

A statistic compared against a limit derived from the same sample is self-referential
and pinned near a constant:

- the fraction of values above their own 95th percentile is always about 5 %;
- `mean(x) - median(x)` of one sample is always about 0;
- the ratio of the faint decile to the bright decile of one flux distribution is a
  property of the lamp's line list, not of the instrument.

The reference must be external: a calibration product (`detectorMap_calib`,
`fiberProfiles`), a detector constant (saturation level, gain, read noise), a physical
constant, or the same measurement from another epoch. `MetricSpec.reference` in
`pfs.drp.qa.metrics.registry` states it for each metric.

### R2 — A limit is measured, never invented, and lives in one place

A limit is one entry of `CHECKS` in `pfs.drp.qa.metrics.registry`: the metrics it reads,
the frame types it applies to, a WARN and a FAIL. No task config, script or dashboard
holds another. Before a limit enters the code it is shown on stored numbers: what the
known-good frames measure, what the known faults measure, and that the limit separates
them.

A check is one of two kinds:

- **Fixed**: limits in the metric's units, for a number that means the same on every
  camera (an offset from the calibration, a sky-line residual in km/s).
- **Relative**: limits in per cent from the camera's normal value, for a number on which
  cameras differ by more than the tolerance (a width, a scatter, a line count). A relative
  check needs a normal value for every camera or arm, frame type and lamp or species it
  judges; without one it gives no verdict. Normal values are made from a run's calibration
  frames and accepted by a person (`drpQa.py values`). They are rows of qadb's
  `qa_normal_value`, each with who accepted it, when and why; they are not code.

Whatever is measured and stored but not judged is one entry of `EXCLUSIONS`, with its
reason. Nothing is left unjudged without being named.

### R3 — Separate measurement from judgement

Tasks emit numbers; `pfs.drp.qa.metrics.gate` alone maps them to `PASS`, `WARN`, `FAIL`
or `UNKNOWN`. A task holds no limit and writes no verdict column. A limit can then be
changed and applied to stored numbers (`drpQa.py judge`) without running the pipeline
over the pixels again, which is what makes R2 practical.

An image with no check judged is `UNKNOWN`, never `PASS`.

### R4 — Never duplicate an existing metric

Before adding a metric, look in `dmLineResidualMetrics` and `iqQaMetrics` for a column that
measures the same thing. `dmLineResiduals` already gives the median and robust RMS of every
species against both references, and `iqQaSpeciesMetrics` carries drp_stella's weighted RMS and
softening; a reimplementation is a regression, not an addition.

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
is dense). A metric aggregated across species measures the species mix. Compute and judge
per species. The same holds for any mixture of populations: arms, frame types, and
estimators (arc moments, calexp trace widths, sky frames). A normal value is kept per
camera or arm, frame type and lamp or species for this reason.

## Adding or changing a check

1. **Measure first.** The task stores the number on every frame it can be measured on, and
   it is loaded into qadb. A check is added only for a number that is already stored.
2. **Look at it on stored runs**, per camera, frame type and lamp or species (R7): the
   known-good frames, and the known faults.
3. **Choose fixed or relative** (R2). For a relative check, confirm that normal values can
   be made: `pfs.drp.qa.metrics.normalValues.candidates` gives one per camera or arm,
   frame type and lamp or species from the calibration frames, each resting on at least
   3 frames.
4. **Change one entry of `CHECKS`.** If something measured is left unjudged, add it to
   `EXCLUSIONS` with the reason. `tests/metrics/test_registry.py` pins the checks and
   their limits; change it with them.
5. **Add the fault to the known-fault test.** `tests/metrics/test_knownFaults.py` holds a
   table of faults beside the test, on stored numbers of engineering and calibration
   visits (never science visits). Every fault must reach its verdict and the frames known
   to be good must not fail. Add the fault the check is for, and keep the rest passing.
6. **Judge stored frames again** with `drpQa.py judge --dry-run`, and compare the counts of
   each verdict with those before. Explain each difference.

A change of normal values is not a change of code: it is `drpQa.py values` at the start
of a run, or on a named calibration set during one. A value shifted beyond its check's
FAIL is refused unless it is accepted with a reason.

## Checklist

State in the PR that adds or changes a metric or a check:

1. The external reference the metric is measured against (R1).
2. The entry of `CHECKS`: fixed or relative, its limits, and the stored numbers they were
   set from (R2). For a relative check, the normal values it needs, and that they can be
   made.
3. The existing columns the metric does not duplicate (R4).
4. What is left unjudged: the entries of `EXCLUSIONS`, each with its reason.
5. That `tests/metrics/test_knownFaults.py` passes, with the fault the check is for in its
   table, and the counts of each verdict on stored runs before and after.
