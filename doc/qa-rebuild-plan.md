# drp_qa — QA metrics rebuild plan

Target repository: `drp_qa` (Subaru PFS DRP QA).
Assumed starting point: a clean checkout of `origin/main`.
Primary goal: **trustworthy quality assurance on PFS spectrograph images.**

This plan is self-contained. It assumes no knowledge of prior work beyond what is on
`origin/main`, and nothing here is bound by any existing feature branch.

---

## Scope

**This document is the full roadmap. `PIPE2D-1391-01` implements Phases 0 and 1 only.**

Phases 0 and 1 are foundations: they add no new QA metric and change no QA verdict. They
exist so that every later phase can be settled by measurement rather than by argument —
the golden visit set decides whether a metric works, and the threshold procedure decides
what its limits are.

| Phases | Ticket | Delivers |
|---|---|---|
| 0, 1 | **`PIPE2D-1391-01`** (this ticket) | golden visit set, threshold procedure, metric registry, long-format schema, storage contract, plotting library, test harness |
| 2 | later ticket | detector-map drift monitoring |
| 3 | later ticket | image-quality extensions |
| 4 | later ticket | visit-level rollup and alerting |
| 5 | later ticket | dashboard and serving layer |
| 6 | one ticket each | additional QA tasks |

Keep this ticket narrow. Phases 2–6 are documented here for context and sequencing, not
to be implemented now; an umbrella ticket that carries all six never closes. In
particular, resist adding a new metric "while we are in here" — a metric added before the
golden set exists cannot be validated, which is the failure mode Phase 1 is built to
prevent.

**Depends on `PIPE2D-1392`** ("Add CI testing for `drp_qa`"), which is assumed merged
first. It carries the `dmResiduals` import fix described in
[Phase 0](#phase-0--unblock-and-verify) step 1, without which the task cannot import and
no CI run over the pipeline is meaningful.

---

## Contents

1. [Objective and design principles](#1-objective-and-design-principles)
2. [Baseline: what is on `main` and what to keep](#2-baseline-what-is-on-main-and-what-to-keep)
3. [Phase 0 — Unblock and verify](#phase-0--unblock-and-verify)
4. [Phase 1 — Foundations](#phase-1--foundations)
5. [Phase 2 — Detector-map drift monitoring](#phase-2--detector-map-drift-monitoring)
6. [Phase 3 — Image-quality extensions](#phase-3--image-quality-extensions)
7. [Phase 4 — Visit-level rollup and alerting](#phase-4--visit-level-rollup-and-alerting)
8. [Phase 5 — Dashboard and serving layer](#phase-5--dashboard-and-serving-layer)
9. [Phase 6 — Additional QA tasks worth adding](#phase-6--additional-qa-tasks-worth-adding)
10. [Process changes](#process-changes)
11. [Acceptance criteria](#acceptance-criteria)

---

## 1. Objective and design principles

PFS QA exists to answer one operational question per visit: *is this data good enough to
reduce, and if not, what is wrong with the instrument?* Every metric must move that
question forward. A metric that cannot separate a good detector from a bad one is worse
than no metric, because it trains operators to ignore `qaStatus`.

Seven rules. Treat them as binding; they encode failures that have already happened here.

### R1 — Every metric needs an external reference

A statistic computed from a sample and then compared against a threshold derived from
that *same* sample is self-referential and pinned near a constant. Examples of the
failure mode:

- counting values above the 95th percentile of their own distribution → always ~5%
- `mean(x) - median(x)` on one sample → always ~0
- ratio of the faint decile to the bright decile of one flux distribution → a property of
  the lamp line list, not of the instrument

The reference must be external: a calibration product (`detectorMap_calib`,
`fiberProfiles`), a detector constant (saturation level, gain, read noise), a physical
constant, or the same measurement from a different epoch.

### R2 — Thresholds are measured, never invented

No threshold enters the codebase before it has been computed from a known-good visit
range. The procedure is in [Phase 1](#phase-1--foundations). Record the visit range and
the date the threshold was derived in the config field's `doc` string.

### R3 — Separate measurement from judgement

Tasks emit numbers. A separate gating layer maps numbers to `PASS`/`WARN`/`FAIL`. This
lets thresholds be re-tuned and re-applied to an existing collection without re-running
the pipeline over the pixels — which is what makes R2 practical.

### R4 — Never duplicate an existing metric

Before adding a metric, grep `dmQaResidualStats` and `iqQaMetrics` for an existing
column that measures the same thing. `get_fit_stats` in `dmResiduals.py` already produces
error-weighted and robust RMS values; an unweighted reimplementation is a regression, not
an addition.

### R5 — Robust statistics only

Use `np.median` and `pfs.drp.stella.utils.math.robustRms`. Arc line fluxes span orders of
magnitude and centroid residuals have outliers from mismatched lines; `np.mean` and
`np.std` are not usable on either without clipping.

### R6 — Separate measurement from presentation

The pipeline runs under `ics_qaActor`, where per-visit latency is operationally visible.
It computes and stores **data**; it does not render figures. Rendering happens later, in
a dashboard or an on-demand report, from the stored data.

The corollary is the constraint that makes or breaks the whole design: **a dashboard can
only ever draw what the pipeline stored.** Storing scalar summaries alone makes every
detail plot impossible and forces a pipeline re-run to debug — the precise cost this
separation exists to avoid. See [1.4](#14-data-product-tiers-and-the-storage-contract).

This rule is about *rendering*, not about plotting code. The plotting functions remain
first-class, tested, and shared ([1.5](#15-extract-plotting-into-a-library)).

### R7 — Per-species, never blended

Arc lamp species have wildly different line densities and intensities in each arm (see
the lamp physics section of `AGENTS.md`: Ar and Xe are nearly absent in the blue, HgCd is
dense). Any metric aggregated across species is dominated by the species mix, not by the
instrument. Compute per `description`, and gate per `description`.

---

## 2. Baseline: what is on `main` and what to keep

`pipelines/drpQA.yaml` registers five task labels:

| Label | Class | Dimensions |
|---|---|---|
| `dmResiduals` | `dmResiduals.DetectorMapResidualsTask` | instrument, visit, arm, spectrograph |
| `dmCombinedResiduals` | `dmCombinedResiduals.DetectorMapCombinedResidualsTask` | instrument |
| `extractionQa` | `extractionQa.ExtractionQaTask` | instrument, visit, arm, spectrograph |
| `extractionQaCombined` | `extractionQaCombined.ExtractionQaCombinedTask` | instrument, visit, arm |
| `imageQualityQa` | `imageQualityQa.ImageQualityQaTask` | instrument, visit, arm, spectrograph |

`skySubtractionQa`, `fluxCalQa` and `fiberNormsQa` exist in the package but are not in the
pipeline.

### Keep and build on

**`dmResiduals.get_fit_stats`** is the strongest code in the repository and the template
for everything below. Per `(status_type, description)` — i.e. separately for `RESERVED`
and `USED` lines of each species — it produces:

- `dof`, `chi2X`, `chi2Y`
- a `spatial.` and a `wavelength.` block, each with `median`, `robustRms`,
  `weightedRms`, `softenFit`, `dof`, `num_fibers`, `num_lines`

The RMS values are error-weighted with a robust variant, and `softenFit` is solved by
bisection. This already satisfies R5 and R7. **Do not add parallel unweighted RMS
columns.** If a gate on spatial scatter is needed, gate on `spatial.weightedRms`.

**`imageQualityQa`** is a mature ~1500-line task with three measurement paths (arc-line
moments, calexp profile fit, fiber-profile fallback), visit classification from FITS
headers, and per-arm/per-species flag-rate thresholds already tuned to lamp physics. Keep
its structure; extend only per R1.

**`dmQaResidualData`** is the per-line residual table: `fiberId`, `wavelength`, `xResid`,
`yResid`, `dispersion`, `status_name`, `isUsed`/`isReserved`, plus `mtpId`/`cobraId` from
`FiberIds`. **This is the plot source for the dashboard.** It is already stored and must
stay stored. See [1.4](#14-data-product-tiers-and-the-storage-contract).

**The plotting code** — `plot_detectormap_residuals`, `plot_residual`, `make_report`,
`plot_visits`, and `iqQaPlots` — is worth keeping and is *not* what gets removed under
R6. Extract it into a task-free library ([1.5](#15-extract-plotting-into-a-library)).
Scalar gates tell you *that* something is wrong; the residual plots tell you *what*.

**The rendered plot datasets** — `dmQaResidualPlot`, `dmQaCombinedResidualPlot`,
`extQaImage` — are what R6 removes from the pipeline. Rendering dense matplotlib scatter
to vector PDF is slow and the artifacts are large, and the pipeline runs under
`ics_qaActor` where that latency is visible. Drop the `Plot` / `MultipagePdfFigure`
output connections; keep every DataFrame they were rendered from.

Note the ordering constraint: **do not remove a rendered output until its data source is
confirmed stored and the plotting function has been extracted.** Deleting the render
while leaving orphaned plotting code behind produces a module whose functions raise
`NameError` on import of their dependencies — an easy and unhelpful mistake.

### Found during `PIPE2D-1391-01` verification

Two defects in the trace/quartz measurement path, both pre-existing and both out of scope
for that ticket because fixing either changes QA verdicts. Observed on Run25 visit 133040
(quartz), arm b, spectrograph 4.

**1. The fiber-profile fallback is unreachable when a calexp exists but fails.**
`imageQualityQa.run` structures the trace/quartz path as:

```python
if calexp is not None:
    ...
    else:                      # calexp present, measurement too sparse
        log.warning("Quartz calexp too sparse ...; FWHM will be sparse.")
elif fiberProfiles is not None:
    ...                        # only reached when there is NO calexp
```

So the documented secondary fallback is only tried when `calexp` is absent, never when it
is present and the measurement fails. On 133040/b4 the calexp path returned 10 good
samples at 100 % flagged, and the task reported a sparse FWHM without consulting
`fiberProfiles` — which would very likely have produced a usable trace width.

**2. A quantum that measures nothing reports `PASS`.**
When `medFwhm`, `pctFlagged` and `medDxCenter` are all NaN, every gate correctly returns
"not judged", and `worstStatus` then falls through to its `PASS` default. The verdict says
the detector is fine; what happened is that nothing was measured. This is the case
[2.3](#23-status-and-recommended-action) reserves `UNKNOWN` for: *`UNKNOWN` must mean we
could not measure, never we measured but could not combine.*

It also blunts the golden set: a `known_good` entry is satisfied by a vacuous PASS, so the
check passes without establishing anything. `examples/verify_PIPE2D-1391-01.ipynb`
section 6 now reports these separately for that reason.

**Reprocessing.** Fix 1 produces measurements that do not currently exist, so collections
reduced before it must be re-run to benefit. Fix 2 is pure gating over values already
stored in `iqQaMetrics`, so in principle it needs no pixel reprocessing — but nothing
today re-gates a stored collection, which is the gap [1.4](#14-data-product-tiers-and-the-storage-contract)
and R3 anticipate. Either add that re-gate step or re-run; for a handful of visits,
re-running is cheaper than writing the tool.

### Known defects on `main`

- `dmResiduals.py` imports `getDescriptionCounts` from
  `pfs.drp.stella.fitDistortedDetectorMap`. That module no longer exists in `drp_stella`;
  it is now `pfs.drp.stella.fitDetectorMap`. **This is an import-time failure — the task
  cannot load.** Fix first (Phase 0).
- `tests/test_dmResiduals.py::testResiduals` is an empty `pass`.
- `tests/SConscript` is vestigial (imports `lsst.sconsUtils`; nothing else uses SCons).
- ~242 Ruff findings repo-wide, mostly `E501`, `D4xx`, `UP`, `F401`.

---

## Phase 0 — Unblock and verify

Small, independently mergeable. Do this before any feature work.

1. **Fix the broken import** — **delivered by `PIPE2D-1392`, assumed already merged.**
   `dmResiduals.py` imported `getDescriptionCounts` from
   `pfs.drp.stella.fitDistortedDetectorMap`, a module that no longer exists in
   `drp_stella`; it is now `pfs.drp.stella.fitDetectorMap`. The stale import raised at
   module import time, so `DetectorMapResidualsTask` could not be constructed at all.
   It landed on the CI ticket so that CI has a pipeline that imports. Confirm it is
   present before starting, and if it is not, stop and rebase onto it:
   ```bash
   grep -n "from pfs.drp.stella.fitDetectorMap import getDescriptionCounts" \
     python/pfs/drp/qa/dmResiduals.py
   ```
2. **Confirm the pipeline builds a QuantumGraph** against a real repo:
   ```bash
   pipetask build -b /path/to/butler -p pipelines/drpQA.yaml --show pipeline-graph
   ```
   Both parts matter. `-b` is needed because the PFS dimensions `arm` and `spectrograph`
   come from the repository's dimension config rather than the default universe, and
   without it the graph fails to resolve with `KeyError: 'spectrograph'` — which looks
   like a pipeline defect and is not one. And it has to be a `--show` form that consumes
   the butler (`pipeline-graph`, `task-graph`, or `--pipeline-dot`); `--show tasks` does
   not, so passing `-b` alongside it is rejected outright.
3. **Establish the golden visit set** (see Phase 1). Nothing else proceeds without it.

---

## Phase 1 — Foundations

This phase produces no new metrics. It produces the machinery that makes every later
metric verifiable. Do not skip it.

### 1.1 The golden visit set

Curate a small, fixed list of visits with known verdicts, checked into the repo as
`tests/data/goldenVisits.yaml`:

```yaml
known_good:
  - {visit: NNNNNN, arms: [b, r, n, m], spectrographs: [1, 2, 3], seqType: "Arc: HgCd",
     note: "nominal focus, post-calibration"}
  - {visit: NNNNNN, ..., seqType: "Quartz"}
known_bad:
  - {visit: 140005, spectrographs: [1], expect: FAIL, reason: "SM1 focus/alignment,
     FWHM 3.83-4.86px across all lamp types"}
  - {visit: NNNNNN, expect: WARN, reason: "stale detectorMap_calib"}
```

The SM1 visit range 140005–140138 is a documented genuine optics failure (see
`AGENTS.md`) and is the natural first `known_bad` entry. Add a stale-calib case and, if
one can be found in the archive, a saturated frame.

**Every threshold and every new metric is validated against this set.** A metric that
flags a `known_good` visit, or passes a `known_bad` one, does not merge.

### 1.2 Threshold derivation procedure

Document this in `AGENTS.md` and follow it for every threshold:

1. Run the metric over the `known_good` visits with no gating.
2. Take the distribution of the metric across all good detectors.
3. `WARN` at the 95th percentile, `FAIL` at the 99th, rounded to a readable value — or,
   where a physical limit exists (saturation, fiber pitch), use the physical limit.
4. Verify the `known_bad` visits exceed `FAIL`.
5. Record the visit range and derivation date in the config field `doc` string.

Ship a helper that does steps 1–3: `bin.src/calibrateQaThresholds.py`, reading a
collection and the golden set and printing suggested config values.

### 1.3 Metric registry and long-format schema

Two structural changes that pay for themselves immediately.

**Metric registry.** A single declarative module, `python/pfs/drp/qa/metrics/registry.py`:

```python
@dataclass(frozen=True)
class MetricDef:
    name: str
    units: str
    reference: str       # what external reference it is measured against (R1)
    higherIsWorse: bool
    warn: float | None
    fail: float | None
    provenance: str      # visit range + date the thresholds came from (R2)
```

Gating becomes one shared function over the registry instead of a hand-written
if/elif ladder per task. This is what makes R3 mechanical rather than aspirational.

**Long-format per-species output.** Today per-species values become wide columns
(`fitSpeciesXRms_HgI`, `fitSpeciesXRms_CdI`, …). The column set therefore varies per
quantum, so concatenating across quanta yields a ragged frame full of NaN, and any
downstream `groupby` has to know the species in advance. Emit long format instead:

| visit | arm | spectrograph | description | metric | value | status |
|---|---|---|---|---|---|---|
| 12345 | b | 1 | HgI | spatialWeightedRms | 0.021 | PASS |
| 12345 | b | 1 | ArI | spatialWeightedRms | 0.088 | FAIL |

Stable schema, trivial concatenation, one gating path, and `groupby("description")` for
plots. Keep a wide summary view as a derived convenience if operators prefer it.

### 1.4 Data product tiers and the storage contract

"The pipeline stores only numbers" needs to be precise, or the dashboard will be unable
to draw anything and you will re-run the pipeline to debug. Adopt three explicit tiers:

| Tier | Content | Granularity | Stored in Butler? | Consumers |
|---|---|---|---|---|
| **1** | Scalar metrics + `qaStatus` + provenance | per (visit, detector, species) | always | `ics_qaActor` gating, alerting, dashboard landing page |
| **2** | Tidy residual / per-fiber tables | per line, per fiber | always | every detail plot |
| **3** | Pixel data, image cutouts | per pixel | **never** | regenerated on demand from `calexp` |

Two rules follow:

- *The pipeline stores data, never a rendering of it. The dashboard renders, never
  recomputes science.* If the dashboard would have to derive a number itself, that number
  belongs in Tier 1 — otherwise it cannot be alerted on or archived.
- *Every Tier 1 scalar must be reproducible from Tier 2.* This is what lets thresholds be
  re-applied to an existing collection without touching pixels (R3).

**Size the Tier 2 products before committing.** Estimate from one real visit: roughly
fibers × lines-per-fiber × detectors. If it lands in the few-MB-per-visit range as
compressed Parquet, store it without concern; if it is much larger, first downcast
residual columns to `float32` and make `description` / `arm` / `status_name` categorical,
then re-measure. Record the measured number in this document.

**Measure the plot cost too, rather than assuming it.** Time one `dmResiduals` quantum
with and without `generatePlot` and record the artifact size. The cost is usually
concentrated in vector-mode scatter, where a PDF stores every marker as a path object;
`rasterized=True` on scatter layers commonly cuts both time and size by one to two orders
of magnitude. The dashboard is still the right architecture, but knowing the real number
tells you whether an on-demand static report is cheap — it probably is — and stops "plots
are slow" from hardening into folklore that constrains later decisions.

### 1.5 Extract plotting into a library

Move every plotting function into `python/pfs/drp/qa/plotting/`, with one hard
constraint: **these functions take DataFrames and return `matplotlib` `Figure` objects,
and import no Butler and no task class.**

That single change yields three consumers from one implementation:

| Consumer | Use |
|---|---|
| Dashboard | interactive rendering, server-side |
| `bin.src/qaReport.py` | static PDF on demand, for night logs and tickets |
| Notebooks | the three under `examples/` |

It also makes the plotting code unit-testable for the first time — a smoke test that each
function returns a `Figure` from a small synthetic frame catches the entire class of
"orphaned plotting code" breakage described in §2.

### 1.6 Test strategy

The rule that makes tests possible: **pure functions take arrays and DataFrames; Butler
glue stays in `runQuantum` and is thin.**

- Every metric is a module-level function with no Butler and no task `self`. It takes
  numpy arrays / DataFrames plus a small config object and returns numbers.
- Those functions are unit-tested with synthetic input and **no LSST stack**, so they run
  in CI. Assert the metric responds correctly to an *injected* defect: shift centroids by
  0.2 px and assert the drift metric reports 0.2 px.
- Stack-dependent tests go in separate files guarded with `pytest.importorskip("lsst.utils.tests")`
  at module level — not a `try/except ImportError` inside the test body, which never runs
  because the module-level import has already failed collection.
- **No empty `pass` tests.** A test that asserts nothing is worse than a missing test: it
  reports green and hides the defect it was named after.

---

## Phase 2 — Detector-map drift monitoring

A new per-detector task answering: *has the instrument moved away from the calibration we
are reducing against?*

### 2.1 Specification (write this before any code)

| Item | Decision |
|---|---|
| Task | `DmDriftMonitorTask`, label `dmDriftMonitor` |
| Dimensions | `(instrument, visit, arm, spectrograph)` |
| Reference | **`detectorMap_calib`** — the calibration product, `isCalibration=True` |
| Lines | `lines` (`ArcLineSet`) from `fitDetectorMap` |
| Output | `dmDriftMetrics`, long format, one row per `(description, metric)` |

**The reference must be `detectorMap_calib`, not `detectorMap`.** The per-visit
`detectorMap` is the map `fitDetectorMap` fitted *to these very lines*; comparing them
measures fit residuals, not drift, and is circular. `imageQualityQa` already reads
`detectorMap_calib` for exactly this purpose — follow it.

### 2.2 Metrics

For each species (`description`), over lines with `flag == 0`:

- `medianDeltaX`, `robustRmsDeltaX` — `x - detectorMap_calib.getXCenter(fiberId, y)`,
  from trace lines (`description == "Trace"`)
- `medianDeltaY`, `robustRmsDeltaY` — `y - detectorMap_calib.findPoint(fiberId, wavelength)[:, 1]`,
  from emission lines
- `driftMag` — computed from **whichever components are available**. Trace lines
  constrain x only; emission lines constrain y only. A drift magnitude that requires both
  to be finite is NaN for every row and reports nothing. Either compute
  `sqrt(dx² + dy²)` when both exist and fall back to `|dx|` or `|dy|` when only one does,
  or drop `driftMag` entirely and gate on the components directly. **Gate on the
  components; `driftMag` is a convenience column at most.**
- `nLines` per species, so a NaN is distinguishable from an untested species

Deliberately **not** included: any "profile width change" metric, unless and until a
stored reference width from the calibration epoch exists to compare against
(see [6.1](#61-psf-shape-map-across-the-detector)).

### 2.3 Status and recommended action

Gate per species via the registry. Map to an operational recommendation:

| Condition | `qaStatus` | `recommendedAction` |
|---|---|---|
| all components within WARN | `PASS` | `NOMINAL` |
| any component in WARN band | `WARN` | `APPLY_SHIFT` |
| any component above FAIL | `FAIL` | `RECALIBRATE` |
| `nLines < minLines` for every species | `UNKNOWN` | `INSUFFICIENT_DATA` |

`UNKNOWN` must mean *we could not measure*, never *we measured but could not combine*.

Also surface the calib's validity range in the output. A large drift against an
out-of-date calib is a calib problem, not an instrument problem, and the operator needs
to see which it is.

### 2.4 Tests

- synthetic lines with zero offset → `medianDeltaX ≈ 0`, `PASS`
- synthetic lines shifted by a known 0.2 px → `medianDeltaX ≈ 0.2`, `FAIL` at a 0.15 px
  threshold
- fewer than `minLines` → `UNKNOWN` / `INSUFFICIENT_DATA`
- trace-only input → x metrics populated, y metrics NaN, status still decided
- golden set: `known_good` → `PASS`; stale-calib `known_bad` → `FAIL`

---

## Phase 3 — Image-quality extensions

Extend `imageQualityQa` only where an external reference exists.

### 3.1 Real saturation

Replace any percentile-based saturation proxy with the genuine article:

- read the detector saturation level from the `calexp` detector object or ISR metadata
- count pixels with the `SAT` mask plane set within each fiber's extraction aperture
- emit `nSaturatedLines` (lines whose peak pixel is flagged `SAT`) and
  `pctSaturatedPixels`

A percentile of the flux distribution contains no information about full well: the top 5%
of any distribution is 5% of it, whether the frame is perfectly exposed or wholly
saturated.

### 3.2 Lamp stability — cross-visit, not within-visit

Line-flux scatter *within* one visit is dominated by the intrinsic brightness range of
the line list, which spans orders of magnitude. It is not a stability metric.

Stability is a **cross-visit** quantity: for a given `(arm, spectrograph, description)`,
track the median line flux across visits and report the fractional deviation from a
rolling baseline. This belongs in the time-series layer ([Phase 4](#phase-4--visit-level-rollup-and-alerting)),
not in the per-quantum task. Per-quantum, emit only the inputs: `medLineFlux` and
`robustRmsLineFlux` **per species**.

### 3.3 Connection hygiene

**Partly delivered by `PIPE2D-1391-01`**, because the pipeline did not build without it:
`pfsConfig` was a `PrerequisiteInput` to the two extraction tasks and a plain `Input` to
`imageQualityQa`, and a dataset type must be a prerequisite to every task in a graph or to
none. It is now a prerequisite with an explicit `minimum=0` everywhere, and
`tests/test_connections.py` checks this without needing a Butler.

`pfsConfig` should remain optional. If it is declared as a `PrerequisiteInput`, note that
prerequisites default to `minimum=1` and are resolved at QuantumGraph build time — set
`minimum=0` explicitly, or the task cannot run against collections that lack a
`pfsConfig`, and no runtime `try/except` can rescue a quantum that was never created.
Fetch each input exactly once.

---

## Phase 4 — Visit-level rollup and alerting

### 4.1 `bin.src/qaVisitSummary.py`

CLI plus importable API returning a per-visit rollup across `dmQaDetectorStats`,
`iqQaMetrics`, `dmQaResidualStats` and `dmDriftMetrics`. Exit code encodes the worst
status (`0`=PASS, `1`=WARN, `2`=FAIL, `3`=no data) so it can gate a nightly job.

Two implementation requirements:

- **Query, don't guess.** Use `butler.registry.queryDatasets(datasetType, collections=...,
  where="visit = ...")` once. Do not loop over the 4×4 arm/spectrograph grid issuing
  speculative `butler.get` calls inside `except Exception: pass` — that pattern turns a
  mistyped collection name, an unregistered dataset type and a genuinely absent detector
  into the same silent "missing" result.
- **Distinguish "not found" from "failed".** Report a bad collection or registry error as
  an error, not as absent data.

### 4.2 Time-series regression detection

`bin.src/plotIqQaTimeSeries.py` plots trends but nobody watches plots nightly. Add
threshold-crossing and trend detection over `iqQaMetrics` / `dmDriftMetrics`:

- rolling median per `(arm, spectrograph, description)`
- flag when the current visit deviates by more than N robust sigma from the trailing
  window
- emit a machine-readable report suitable for a nightly cron

This is where genuine lamp-stability and slow-focus-drift detection lives, because both
are only visible across epochs.

---

## Phase 5 — Dashboard and serving layer

The dashboard is the presentation half of R6. Much of it lives outside this repository,
but the contract it depends on does not, so it is specified here.

### 5.1 Serving layer — do not query Butler live for aggregates

Butler registry queries across a whole run are slow enough to make a dashboard feel
broken. Split the access pattern:

| View | Source | Path |
|---|---|---|
| Aggregates, trends, landing page | Tier 1, exported to a single Parquet file | DuckDB over Parquet — sub-second over millions of rows, no Butler in the request path |
| Per-detector detail | Tier 2, in Butler | fetched for one dataId only when a detail view opens |

The export is a small post-run job that concatenates Tier 1 across a collection. It is
also what makes the data queryable from notebooks and ad-hoc scripts without a Butler.

### 5.2 Render dense scatter server-side

A per-detector residual plot is order 10⁵ points; shipping that to a browser per detector
will not stay interactive. Rasterize server-side — Datashader, with HoloViews/Panel, is
built precisely for this and is the single highest-leverage technical choice in the
dashboard. Client-side plotting libraries are fine for Tier 1 trend views, where the
point counts are small.

### 5.3 Cache on Butler dataset IDs

Butler dataset UUIDs are immutable, so
`(dataset_id, plot_spec, plotting_code_version)` is a sound cache key: a cached plot can
never go stale for the wrong reason, and bumping the plotting library version invalidates
cleanly. Cache rendered raster tiles, not DataFrames.

### 5.4 Two surfaces, not one

| Surface | Question it answers | Suggested tool |
|---|---|---|
| Trend / ops | "was last night OK, is anything drifting?" | Grafana over the Tier 1 Parquet, or a small Panel page |
| Detector deep-dive | "what exactly is wrong with b1 on visit N?" | Panel + HoloViews + Datashader over Tier 2 |

Grafana on the trend surface gets alerting essentially for free, which matters more than
it sounds — see 5.6.

### 5.5 The metric registry is the pipeline/dashboard contract

The registry from [1.3](#13-metric-registry-and-long-format-schema) tells the dashboard
how to label axes, which direction is bad, where to draw threshold lines, and which
threshold version produced a stored verdict. Without a shared registry the dashboard
grows its own copy of the thresholds, and the two drift apart within a month.

Store the QA code version and threshold-set version alongside the numbers, so the
dashboard can display "gated with thresholds v3" and historical data can be re-gated
without re-running the pipeline.

### 5.6 Keep a headless path

A dashboard nobody opens at 03:00 is not a QA gate. Two things must work without a
browser:

- `ics_qaActor` gets its verdict from Tier 1 in Butler, synchronously, with no dashboard
  dependency
- the nightly job in [4.2](#42-time-series-regression-detection) evaluates thresholds and
  pushes alerts

This is why [4.1](#41-binsrcqavisitsummarypy) keeps an exit-code CLI: same contract,
scriptable, and it still works when the dashboard is down.

### 5.7 Risk — the dashboard must not become the source of truth

If a number exists only once the dashboard computes it, it cannot be alerted on, archived,
or reproduced. Any such number gets pushed back into Tier 1 (R6, [1.4](#14-data-product-tiers-and-the-storage-contract)).

---

## Phase 6 — Additional QA tasks worth adding

Ordered by value to image QA per unit of effort. Each follows R1–R7 and Phase 1's
validation gate.

### 6.1 PSF shape map across the detector

**Highest value.** `imageQualityQa` reports a *median* FWHM per detector, which cannot
distinguish uniform defocus from a tilted focal plane or astigmatism — different faults
with different fixes.

Fit FWHM and ellipticity (`ixx`, `iyy`, `ixy` second moments, already in `ArcLineSet`) as
a low-order 2-D polynomial in `(x, y)`. Emit the coefficients plus `fwhmTilt` and
`medianEllipticity`, and render a per-detector focal-plane map. A tilt term that grows
over a run is a mechanical problem; a uniform offset is a focus problem.

### 6.2 Real fiber cross-talk

Measure what the name means: for each fiber, the flux at the inter-fiber midpoint
relative to the adjacent trace peaks, sampled at several rows. Anything derived from the
ratio of faint to bright *lines* is a property of the lamp line list, not of the detector.
Pair it with `minFiberPitch` from `detectorMap_calib` — but note that fiber pitch is a
property of the calibration and near-constant across visits, so it belongs in a
per-calib report rather than in per-visit gating.

### 6.3 Detector and electronics stability

From ISR outputs and metadata, per amplifier: read noise, overscan level, gain, bias
structure. Track across visits. These catch electronics faults that no line-based metric
sees, and they are cheap — the numbers are already in the ISR logs that
`imageQualityQa` optionally consumes.

### 6.4 Scattered light and background level

Median background at inter-trace positions in the `calexp`, per detector region.
Catches light leaks, ghosts, and the scattered-light contamination already documented as
a false-positive source on IIS engineering frames.

### 6.5 Cosmic-ray rate and defect growth

CR density per visit from the `cosmicray_log`, plus new-defect detection by differencing
the bad-pixel mask against the calibration defect list. A rising CR rate or a growing
defect cluster is a detector health signal.

### 6.6 Wavelength solution stability against sky lines

For science frames, compare measured OH/OI sky line positions against their catalog
wavelengths. This is an *independent* check of the wavelength solution on science data,
where arc lines are unavailable — the only metric here that validates the solution on the
data actually being reduced.

### 6.7 Per-amplifier image quality

Split existing FWHM and residual metrics by amplifier. PFS detectors are multi-amp, and
an amp-localised degradation is invisible in a detector-wide median.

---

## Process changes

1. **A metric design checklist in `AGENTS.md`.** Before a metric merges, state in the PR:
   what external reference it is measured against (R1); where its thresholds came from
   (R2); which existing column it does *not* duplicate (R4); its golden-set result.
2. **Gate CI on the stack-free unit tests.** They exist to run without the LSST stack —
   make that a check, not a convention.
3. **Run the golden set before merge.** A metric that has never been run over real data
   is a hypothesis, not a metric. Most of the failure modes R1 guards against are
   visible within minutes of a single real visit.
4. **Lint the files you touch.** `ruff format` + `ruff check` scoped to changed files.
   Never bulk-reformat unrelated files in a feature PR; the repo is not Ruff-clean and
   the noise buries the review.
5. **Keep `CHANGELOG.md` current** under `## [Unreleased]` for new tasks, new or removed
   config fields, and new output columns.
6. **Prefer small, independently mergeable PRs.** Phase 0, each Phase 1 component, and
   each metric should land separately.
7. **Delete `tests/SConscript`** or state why it stays. It is the last SCons file and
   imports `lsst.sconsUtils` for nothing.

---

## Acceptance criteria

### `PIPE2D-1391-01` — Phases 0 and 1

This ticket is done when all of the following hold. Note that none of these require a new
metric to exist:

- [x] `PIPE2D-1392` is merged, `dmResiduals` imports, and
      `pipetask build -b <repo> -p pipelines/drpQA.yaml --show pipeline-graph` resolves all
      five tasks against a real repository. This needed a fix: `pfsConfig` was a
      prerequisite to the extraction tasks and a plain input to `imageQualityQa`, so the
      full pipeline had never built. Running the tasks one at a time with `#label` hides
      it, which is why it survived this long.
- [x] `tests/data/goldenVisits.yaml` holds the Run25 stable calibration sequence (three
      blocks, 2025-11-10 and 2025-12-01) and two clear twilight-sky sets as `known_good`;
      the SM1 focus range (140005–140138) and a cloudy twilight set as `known_bad`. Two
      `known_bad` placeholders remain — the stale-calib and saturated-frame reference
      cases for Phases 2 and 3. Run30 contributes fault cases only:
      obstructed frames, partial slit illumination, and an off-zenith flexure pair.
      Its calibration block and drift series are deliberately **not** here — they carry
      no verdict, and a job that needs them queries the Butler on sequence type and date
      rather than reading a transcription that goes stale each run.
- [x] `bin.src/calibrateQaThresholds.py` runs the procedure and prints suggested
      thresholds with their provenance sentence. Exercised end to end against a CSV;
      **not yet run against a Butler collection.**
- [x] The metric registry exists (`pfs.drp.qa.metrics.registry`) and the `imageQualityQa`
      thresholds are gated through it, with provenance recorded. The provenance says what
      is true: the numbers are the hand-tuned defaults, not golden-set derived.
- [x] Per-species output is long-format, in the new `iqQaSpeciesMetrics` dataset;
      `tests/test_longFormat.py` asserts that concatenating quanta with different species
      mixes yields a stable schema with no NaN padding.
- [ ] Tier 2 volume per visit and the per-quantum plotting cost are both measured and
      recorded in this document. **Not done — both need a real visit through the
      pipeline.** See the note below.
- [x] `pfs.drp.qa.plotting` exists and imports no Butler and no task class —
      `tests/test_plotting.py` checks that statically over the whole subpackage, and
      smoke-tests each function against a synthetic frame.
- [x] The stack-free tests run green without the LSST stack (133 passed locally). CI now
      installs the PyPI wheels they need; **confirm the first CI run.**
- [x] `tests/test_dmResiduals.py` tests something: `get_fit_stats` against injected
      defects of known size. No `pass` bodies remain. It skips without the stack and has
      been checked against a stub, **not against the real stack.**
- [x] Files touched are Ruff-clean; `ruff check .` and `ruff format --check .` pass over
      the whole tree.

Explicitly **not** in this ticket: any new QA metric, any change to an existing QA
verdict, removal of the rendered plot datasets, and the dashboard.

**Verdict parity is measured, not asserted.** Putting the metric values stored in
`qaActor/reductions` through both the gating ladder as it stood on `main` and the new
registry gives the same verdict for all 666 quanta. See
`examples/verify_PIPE2D-1391-01.ipynb` section 3a; it reads only the measured values,
which this branch does not touch, so it needs no pipeline run.

Note what that collection cannot do. `qaActor/reductions` is reduced incrementally, each
visit by whatever code was deployed at the time, so its stored `qaStatus` is a mixture of
versions and cannot serve as a baseline for what `main` produces today. Comparing against
it (notebook section 3b) establishes nothing either way. A baseline has to be a fresh
reduction of chosen visits on `main`, compared against the same visits on the branch —
notebook section 3c.

#### Remaining work on this ticket

1. **Fill in the remaining `known_bad` entries** of `tests/data/goldenVisits.yaml` — the
   stale-calib and saturated-frame reference cases for Phases 2 and 3. `known_good` is
   done (Run25, 133025–133055).

   Sample sizes now clear the 20-detector floor `calibrateQaThresholds.py` enforces, per
   arm (b=244, n=148, r=148, m=96) and per b-arm species (HgCd 24, each of Argon, Xenon,
   Neon and Krypton 40). `b:HgCd` reaches 24 only because the 2025-12-01 block repeated
   it; it is the tightest key and the one that matters most, being the one lamp whose
   blue flag rates are genuine rather than lamp physics.
2. ~~**Run `pipetask build`** against a real repo.~~ Done: the graph resolves and
   `iqQaSpeciesMetrics` appears on `imageQualityQa` with dimensions
   `{arm, spectrograph, visit}` and storage class `DataFrame`.
3. **Re-derive the `imageQualityQa` thresholds** with `bin.src/calibrateQaThresholds.py`
   once (1) is done, and replace the inherited provenance strings in
   `pfs.drp.qa.metrics.definitions`. This is the point of Phases 0 and 1; until it
   happens the thresholds are still the hand-tuned ones.
4. **Measure and record Tier 2 volume and plotting cost** (section 1.4): time one
   `dmResiduals` quantum with and without `generatePlot`, record the artifact size, and
   estimate the compressed-Parquet size of `dmQaResidualData` for one visit.

### Full rebuild — Phases 2 to 6

The rebuild as a whole is done when all of the following hold:

- [x] `pipetask build -b <repo> -p pipelines/drpQA.yaml` succeeds; every task imports.
      Verified on the Run30 stack (2026-02-10) during `PIPE2D-1391-01`.
- [ ] Every `known_good` visit in the golden set reports `PASS` on every metric.
- [ ] Every `known_bad` visit reports the expected `WARN`/`FAIL`, **for the expected
      reason** — the failing metric's name identifies the actual fault.
- [ ] Every threshold's `doc` string names the visit range it was derived from.
- [ ] Every metric function has a unit test that injects a defect of known size and
      asserts the metric recovers it. No test bodies are `pass`.
- [ ] The stack-free tests run green in CI without the LSST stack.
- [ ] No metric compares a sample against a threshold derived from that same sample.
- [ ] Files touched are Ruff-clean.

Presentation split (R6):

- [ ] No `Plot` or `MultipagePdfFigure` output connection remains in `drpQA.yaml`.
- [ ] Every plot that used to be rendered in-pipeline can be reproduced from stored data
      alone, with no pixel access — verified by regenerating one of each from a
      collection.
- [ ] `pfs.drp.qa.plotting` imports no Butler and no task class, and each of its functions
      has a smoke test returning a `Figure` from a synthetic frame.
- [ ] Tier 2 volume per visit has been measured and recorded in this document.
- [ ] The measured per-quantum cost of in-pipeline plotting has been recorded, so the
      removal is justified by a number.
- [ ] `ics_qaActor` obtains its verdict from Tier 1 with no dashboard dependency.
- [ ] No number is computed only inside the dashboard.

### Suggested sequencing

| Order | Work | Ticket | Blocks |
|---|---|---|---|
| 1 | Phase 0 | `-01` | everything |
| 2 | Phase 1.1 golden set | `-01` | all validation |
| 3 | Phase 1.4 storage contract + measurements | `-01` | Phase 1.5, Phase 5 |
| 4 | Phase 1.5 extract plotting library | `-01` | dropping any rendered output |
| 5 | Phase 1.3 registry + long format | `-01` | Phases 2–5 |
| 6 | Phase 1.6 test harness | `-01` | all tests |
| 7 | Drop rendered plot datasets | later | needs 3 and 4 complete |
| 8 | Phase 2 drift monitor | later | — |
| 9 | Phase 3 IQ extensions | later | — |
| 10 | Phase 4 rollup + alerting | later | needs 8 and 9 |
| 11 | Phase 5 dashboard | later | needs 3, 4, 5 |
| 12 | Phase 6 tasks, in listed order | one each | independent of each other |

Steps 3, 4 and 7 are ordered deliberately: the storage contract and the plotting library
must both be in place *before* any rendered output is removed, or the plots become
unreproducible and the plotting code is orphaned.

Phases 6.1–6.7 are independent and can be parallelised across people once Phase 1 exists.
