# QA domain notes — `imageQualityQa`, arc-lamp physics, failure patterns

Empirical findings accumulated through QA analysis on PFS engineering-run data. These are
**observations, not rules**: thresholds, line counts and failure signatures reflect the data they
were derived from and will drift as the instrument and the pipeline change. Re-check against
current data before acting on a number here.

Anything labelled **proposed** below is not implemented in any merged branch — check the code
before relying on it. Claims here were last verified against `drp_qa` `main` and `drp_stella`
`master` `c771d507` on 2026-10-02; the trace-width section on 2026-10-12.

Conventions and build/test rules live in [`AGENTS.md`](../AGENTS.md); this file holds only the
domain knowledge that used to sit in sections 7, 8 and 10 of it.

---

## Domain Knowledge: `imageQualityQa`

The sections below capture domain knowledge accumulated through QA analysis work on the
Subaru PFS engineering run data: task-specific gotchas, failure-mode taxonomy, and
cross-repo dependencies.

### What it does

Measures image quality (FWHM, flag rates) on a per-detector quantum
`(instrument, visit, arm, spectrograph)`. It can draw from three data sources, in
descending preference order:

1. **Arc-line shape measurements** — second-moment `ixx`/`iyy` from `ArcLineSet`
   (`lines` dataset); requires `arcLines` connection.
2. **Calexp trace widths** — each fiber fitted jointly with its neighbours on sampled rows
   of the `calexp` (`pfs.drp.qa.crossDispersion`); used when arc lines are absent or sparse
   (`nGoodLines < minGoodLines`).
3. **Fiber profile calibration** — reads stored profile widths from `fiberProfiles`;
   last resort when neither of the above is reliable.

### Visit classification (`_classifyVisit`)

The task classifies each visit by reading FITS headers from either `calexp` metadata or
`pfsConfig.header`:

| Header | Meaning |
|---|---|
| `W_SEQTYP` | Observation type: `scienceArc`, `scienceTrace`, `scienceObject`, `scienceObject_windowed`, `scienceDark` |
| `W_SEQNAM` | Human-readable name, e.g. `"Arc: HgCd"`, `"Arc: Ne"`, `"Quartz"` |
| `W_SEQCMN` | Command name (rarely needed) |

Returns `(obs_type, is_iis, seq_nam)`:

- `obs_type`: one of `"arc"`, `"trace"`, `"science"`, `"allsky"` (`W_SEQNAM` starting
  "sky"), `"twilight"` (starting "twilight"), `"dark"` (`scienceDark`), `"unknown"`
- `is_iis`: `True` when illuminated by the 16 IIS engineering fibers (lamp header names
  from `getLamps()` end with `"_eng"`, e.g. `"Ar_eng"`). The headers are read without
  `DATA-TYP`: `getLamps()` answers `{"Quartz"}` for any `FLAT` before reading them, which took
  Run30's nightly "ImageQuality (Halogen)" IIS quartz (`W_IISQTH`) for a full-slit quartz, so
  its calexp failed and it fell back to `fiberProfiles` (PIPE2D-1953). Judging IIS frames on
  their own fibers is PIPE2D-1954.
- `seq_nam`: raw `W_SEQNAM` string

**IIS vs regular**: IIS frames illuminate only 16 engineering fibers rather than all 600
science fibers. Arc-line shape measurements are unreliable for IIS frames because the
line catalog doesn't match the sparse illumination; the calexp path or fiber-profile
fallback is preferred in that case.

### Key config fields

| Config field | Purpose |
|---|---|
| `minGoodLines` (default 10) | Min good arc-line measurements to trust the arc path |
| `minPeakSN` (default 5.0) | Min significance (fitted flux / its error) of a calexp trace sample |
| `maxCalexpFlagRate` (default 0.5) | Max fraction of bad calexp samples before rejecting calexp path |
| `minFluxstdGoodFrac` (default 0.10) | Min fraction of good FLUXSTD samples for stellar calexp path |
| `profileHalfWidth` | Deprecated, ignored: the calexp width uses no aperture |
| `profileYStride` (default 100) | Row sampling interval (px) for calexp trace widths; cost is proportional to the rows |
| `fwhmWarnThreshold` / `fwhmFailThreshold` (3.2/3.5 px) | FWHM pass/warn/fail thresholds |
| `dxCenterWarnThreshold` / `dxCenterFailThreshold` (1.0/2.0 px) | `\|medDxCenter\|` flexure thresholds |
| `flagRateWarnThreshold` / `flagRateFailThreshold` | `pctFlagged` thresholds; DictField keyed by `arm` **or** `arm:species` |

Flag-rate thresholds are looked up as `arm:species` → `arm` → 15.0/20.0, where the
species is the part of `W_SEQNAM` after the colon (`"Arc: HgCd"` → `HgCd`). Blue-arm
defaults are permissive per-lamp because several species have almost no usable b-arm
lines (see [Arc Lamp Physics](#arc-lamp-physics-and-b-arm-pctflagged-failures)):

| Key | WARN | FAIL |
|---|---|---|
| `b` | 50.0 | 60.0 |
| `b:HgCd` | 15.0 | 25.0 |
| `b:Neon` | 50.0 | 60.0 |
| `b:Krypton` | 55.0 | 65.0 |
| `b:Xenon` | 85.0 | 92.0 |
| `b:Argon` | 93.0 | 97.0 |
| `r`, `n`, `m` | 15.0 | 20.0 |

`DictField` values can't be set with dot notation on the command line — assign the whole
dict as a Python literal:
`-c "imageQualityQa:flagRateWarnThreshold={'b': 50.0, 'b:Argon': 93.0}"`.

### Trace widths (PIPE2D-1918)

From [`trace-widths.ipynb`](trace-widths.ipynb), Run25 quartz 133040 (12 detectors), Run27
SM1-defocused quartz 140032 and 140139, and the test fixtures in `tests/imageQualityQa/data`:

- **Fibers are 6.17 px apart** on every arm. Any aperture or background window wider than about
  3 px on each side of a trace takes in its neighbours; the old `profileHalfWidth` 7 aperture
  flagged nearly every quartz sample.
- **The trace profile is not Gaussian.** The Run25 `fiberProfiles` read 3.00-3.03 px at half
  maximum, 2.75-2.80 px by a Gaussian fit, and 2.80-2.94 px by the second moment of the median
  profile. The joint fit, on rows built from those profiles, recovers the half-maximum width to
  0-4 %; on the real rows it reads 3.18-3.32 px, a further 3-6 % that is not explained.
- **drp_stella's per-fiber second moment** (`FiberProfile.calculateStatistics().width`), median per
  detector, ranged 2.32-3.66 px where the Gaussian fit ranged 2.72-2.84 px. Cause not identified.
- **The calibration profiles fit a fully lit quartz** to within ±2.4 % of the peak (median residual
  by offset from the trace center), although they were built from sparse MOD4 quartz
  (133096-133102). A defocused spectrograph against its focused calibration leaves -11 % at the
  center and +32 % on the flanks.
- **SM1 defocus** reads 4.13-4.46 px, 1.24-1.41 times the same detectors in focus. The calibration
  used for those reductions (Run26) does not change: only the exposure shows it.
- **Which calibration a reduction used** is in the calexp's provenance; Run27 was reduced with
  Run26 `fiberProfiles` and Run29's for 150116. `pfs.drp.qa.utils.provenance` reads it.
- **Obstructed quartz** (150116, M1 cover closed): joint-fit usable fraction drops to 3 % (b),
  52 % (r) and 25 % (n), so b and n fall back to the calibration; `linesPerFiber` catches it
  (9-56 % below the known-good median).
- **`lines` on a quartz** holds about 2.4 M trace centroids: 0.8-1.1 s and about 400 MB to read.

### Output metrics

`iqQaMetrics` DataFrame (one row per quantum). Core columns:

- `medFwhm`: median FWHM in pixels
- `medDxCenter` / `dxCenterRms`: median and scatter of the spatial offset from
  `detectorMap_calib` (measured minus calibration, as drp_stella's residuals; the opposite sign
  before PIPE2D-1921), a flexure diagnostic
- `pctFlagged`: percentage of arc lines flagged by `fitDetectorMap`
- `nLines`: rows of the `lines` dataset, whichever path measures the FWHM; not judged
- `linesPerFiber`: unflagged lamp lines (trace centroids on a quartz) per lit fiber, the judged
  count; `nLitFibers` / `nFibers`: the detector's `GOOD` fibers, and those plus the hidden ones
- **What `lines` holds** (PIPE2D-1935): a row for every line of the lamp's list on every `GOOD`
  fiber, flagged when not measured, and trace centroids (`description == "Trace"`), on an arc
  as on a quartz: 55-88 % of an arc's rows. A fiber a design hides (`fiberStatus` BLACKSPOT)
  has none. Trace centroids per fiber follow the illumination (b1: 691 on a Run25 neon set,
  609 fully lit and 473 on a shaka exposure in Run30), lamp lines per lit fiber do not.
- `traceOnly`: True when falling back to fiber-profile widths; `medFwhm` is then not judged
- `obsType` / `seqName`: visit classification and raw `W_SEQNAM`
- `qaStatus`: `"PASS"`, `"WARN"`, or `"FAIL"` — the worst of the FWHM, flag-rate, and
  `|medDxCenter|` checks

Additional columns are added dynamically: a per-status-bit flag breakdown
(`pctNotVisible`, `pctBlend`, `pctSuspect`, `pctRejected`, `pctBroad`, plus `pctLowSN` /
`pctMeasFail` on the arc-line path), and — when the optional `isr_log`, `cosmicray_log`,
and `reduceExposure_log` connections are present — ISR, cosmic-ray, and `fitDetectorMap`
statistics (`fitChi2`, `fitXRms`, `fitYRms`, `fitReserved*`, per-fiber arrays; the
per-species values go to `iqQaSpeciesMetrics`).

### Shaka exposures: one fiber group lit (PIPE2D-1935)

A *shaka* exposure is taken on a fiber-group design that lights about a quarter of the fibers:
in the opdb, `pfs_design.design_name` `cobraHome-MOD4_group<n>` (`cobraHome` lights them all).
Group *n* lights every fourth cobra counted along the slit: number the cobras in fiber order through all four spectrographs from 0, skipping engineering fibers and empty
holes, and group *n* is those whose number is *n* − 1 modulo 4. It is exact on Run30's daily neon
arcs 150413, 149881, 149398 and 150367 (groups 1-4: 585-589 cobras each, none off the phase);
it is not `cobraId` modulo 4. The design drives its group home; the other cobras stay parked on
their black dots, not commanded, recorded as `fiberStatus` BLACKSPOT.

Spectrograph 1, holes 1-100 (opdb `pfs_config_fiber`):

```
150413  1    E●x··●···●···●···●···●c··●···●···●···●···●·_E··●···●···●···●···●···●···●···●···●···ccc·●··_E_·●···●·
149881  2    E·x○··●···●··○●···●···c···●···●···●···●···●_E···●···●···●···●···●···●···●···●···●··ccc··●·_E_··●···●
149398  3    E·x●···●···●···●···●··c●···●···●···●···●···_E●···●···●···●···●···●···●···●···●···●·ccc···●_E_···●···
150367  4    E·x·●···●···●···●···●·c·●···●···●···●···●··_E·●···●···●···●···●···●···●···●···●···●ccc●···_E_●···●··
149883  all  E●x●●●●●●●●●●●●●●●●●●●c●●●●●●●●●●●●●●●●●●●●_E●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●ccc●●●●_E_●●●●●●●
```

`●` lit, driven home (`GOOD`); `·` hidden behind its dot (`BLACKSPOT`); `○` lit, cobra not moved
(`GOOD`, no nominal position); `c` broken cobra; `x` broken fiber; `E` engineering fiber;
`_` empty hole. Engineering fibers, empty holes and broken cobras on the phase widen the gap
between lit fibers to 5-8; the unmoved `GOOD` cobras (14-40 per visit, nearly all off the phase)
narrow it to 1-3. Those are lit, measuring as many lamp lines as the driven ones, which is why
`litFibers` counts every `GOOD` fiber and not only those with a nominal position.

### DM residual metrics

`dmResiduals` is deprecated (PIPE2D-1956) in favour of `dmLineResiduals`, below; both run
through the next observing run so the two can be compared. `dmResiduals` re-selects lines
with its own copy of drp_stella's rules, sigma-clips them, and drops trace centroids on arcs,
so its statistics are not those of the lines the fit used.

`dmResiduals` writes `dmQaResidualData`, `dmQaResidualStats` and `dmQaResidualPlot`.
`get_fit_stats` builds `dmQaResidualStats` **per `(status_type, description)`** — that
is, separately for `RESERVED` and `USED` lines of each species — via the `FitStats` /
`FitStat` dataclasses (`pfs.drp.qa.metrics.fitStats`). Each row carries `dof`, `chi2X`, `chi2Y`, and a `spatial.` and
`wavelength.` block of `median`, `robustRms`, `weightedRms`, `softenFit`, `dof`,
`num_fibers`, `num_lines`. The RMS values are **error-weighted** (`getWeightedRMS`) with
a robust variant (`robustRms`); `softenFit` is solved by bisection, and is NaN when it
can't be (more than `maxSoften` needed, or zero dof). Prefer these over
adding parallel unweighted metrics.

`dmCombinedResiduals` aggregates across detectors into `dmQaDetectorStats` and renders a
multi-page `dmQaCombinedResidualPlot` via `make_report`, whose pages are drawn by
`pfs.drp.qa.plotting.dmCombined.reportFigures`.

The task writes **data only**. Plotting lives in `plotting/iqQa.py` and is driven after the
fact by `bin.src/plotIqQaTimeSeries.py`; there is no `iqQaPlot` dataset.

### Line residuals: drift against `detectorMap_calib` (PIPE2D-1921)

`dmLineResiduals` (named `dmDriftMonitor` until PIPE2D-1956) measures `lines` against the
calibration before the visit's adjustment (measured minus calibration, as drp_stella's
residuals), and against the visit's adjusted detectorMap (below). What the calibration
offsets mean:

- **A median offset is mostly harmless.** `adjustDetectorMap` fits a first-order correction
  per visit and absorbs a shift. It matters when it nears the 3 px radius within which
  drp_stella finds lines (`centroidLines.peakSearch`, `centroidTraces.searchRadius`): lines
  are then lost or matched to a neighbour. Hence WARN 1 px, FAIL 2 px.
- **Scatter about the median is what a shift cannot fix**, but against the calibration it
  also holds the linear change the adjustment absorbs. Known-good robust RMS (Run25, the
  calibration's own visits excluded): x 0.18–0.22 px at p99; y 0.08–0.13 px on lamp
  lines, more on sky lines (NaI 0.32 px).
- **The calibration's own input visits read zero** against the calibration by construction; exclude them from any
  drift derivation. Against the visit's own map they are visits like any other, and the scatter derivation keeps
  them (PIPE2D-1956).

Measured in the validation run (`u/wtg/qa-thresholds/006`, 2026-10-15):

- **Run25 after its calibration block** (from 133532): lines sit off in y by the same
  pixels in every arm of a spectrograph, n with the opposite sign: SM1 0.45, SM2 0.65,
  SM3 1.40, SM4 0.70 px. SM1 is also +0.25 px in x. Constant from 133532 to 135850, so a
  step after the calibration block, cause unknown; the r arm drifts a further 0.3–0.4 px
  within twilight sequences.
- **Run27 against its calibration** (built from 140502–140526, after these visits): SM3
  +1.15 px in x on every arm; SM2 −3.15 px in y on b and r, at the search radius; SM3
  −1.75 to +3.6 px in y. Cause unknown.
- **b read with m** sits −0.47 px in x, read with r +0.08 px (PIPE2D-1927).
- **Off-zenith flexure is small**: the Helium detectorMap tests at alt 45–49 (150779,
  150782) sit within 0.2 px in y on b and m, and b −0.33 to −0.51 px in x, the camera-set
  offset of b read with m; at most about 0.1 px of flexure, far under the 1 px WARN.
- **b-arm Argon arcs are never adjusted**: too few lamp lines, so no line is marked
  `USED` or `RESERVED`, and they are reduced against the calibration as it is
  (`UNFITTED`). Their offsets go into the reduction uncorrected: Run25 b3 Argon after the
  calibration block carries the SM3 y offset (1.13 px, WARN); on the calibration's own
  visits they read +0.22 px (`u/wtg/qa-thresholds/008`).
- **A robust RMS near 2.2 px is noise**: lines spread uniformly over drp_stella's ±3 px
  search window give 0.741 × 3 = 2.22 px. Known-good lamp lines sit at 0.05–0.25 px.
  Never-adjusted b-Argon species reach it (a few real Argon lines in noise; their median is
  still stable), and so does Run27 SM3 against its calibration (1.5–4.8 px, medians from
  −3.5 to +3.6 px): there the change is more than a shift and the median means little.
  Run27 SM2 is the opposite case: −3.15 px at 0.1 px scatter, a clean shift.
- **Twilight b3/b4** (134886, 135278, 135279): the fit kept all ~2 million trace rows and
  used none of 1,300–2,300 sky lines; only the reserved tenth survive. The signature of
  `rejectOutliers`' fallback, whose y limit is a percentile over rows that are mostly
  traces with zero y residual; not confirmed.
- **drp_stella's INFO lines are not in the Butler logs** of these reductions:
  the tasks log as `pfs.<task>`, lowercase, which `--log-level PFS=INFO` misses and a bare
  `INFO` (which sets only `lsst`) does too. The `reduceExposure_log` and `dmDriftMonitor_log` (now `dmLineResiduals_log`)
  hold only the executor's lines, and the fit statistics `imageQualityQa` parses are absent.
  Use `--log-level pfs=INFO`, as drpActor sets it (PIPE2D-1938).

### Line residuals against the visit's detectorMap (PIPE2D-1956)

Against the visit's own `detectorMap`, which `reduceExposure` adjusted to these lines, the
`USED` lines are the fit's residuals and the `RESERVED` lines, held back at random, test the
adjusted map on lines it was not fitted to: what a science frame, extracted with that map,
relies on. Their robust RMS per species is the judged metric; drp_stella's own "Fit quality
from reserved lines" log line gives a robust RMS too, but over every species at once, x
mixing traces and lamp lines (R7). The weighted RMS and softening per species are drp_stella's,
from the `Stats for <species>` log lines (`iqQaSpeciesMetrics`), over the `USED` lines.

Measured in `u/wtg/qa-thresholds/009` (2026-10-15): the validation visits only, so Run25's
known-good visits and the fault examples of Runs 27 and 30; no healthy visit of a later run.
Against the calibration it reproduces 008 exactly. RESERVED robust RMS against the visit's map:

- **Run25 known-good lamp lines and traces**: medians 0.05–0.11 px in x and y per arm; the
  highest 0.25 px in x, 0.17 px in y on b and m. HgI is the exception in y: n 0.42–0.62 px,
  r up to 1.12 px (median 0.14). n-arm values predate the pipeline changes that made the n
  arm reliable (about Run29–30).
- **Sky lines on twilights** (NaI, OI, OH, O2) reach 1–7 px in y, b3/b4 the worst: the fit
  fallback above, not the map's scatter. Not judged.
- **Run27**, used only for its faults: SM1 sits 0.32 px in x (median, up to 0.72 px), the
  spectrograph defocused in part of the run; against the calibration built later, SM3 sits
  0.85 px in y (median, up to 5.4 px; 0.88 px in x) and SM2 reaches 3.3 px in y. The metric
  separates both faults from Run25.
Judged in `u/wtg/qa-thresholds/010` (2026-10-20, every validation visit reduced afresh) and
checked with `bin.src/judgeScatterThresholds.py` on comparison mode's exports of Runs 25–30:

- **Trace centroids must be split by frame type.** On a quartz (`obsType` `trace`) the
  RESERVED trace scatter is 0.02–0.05 px (n1 0.08–0.13 through Run28); on an arc's faint
  continuum it is 0.10–0.14 px, and on a twilight 0.06–0.17. Pooled, the thresholds sat between
  and failed 14–20 % of Runs 25–27's healthy arcs while every quartz passed. The task now stores
  `obsType` and the derivation groups by it.
- **Quartz trace scatter is repeatable per detector to about 1 %, differs between detectors by
  20 %, and moves between runs.** In Run25 b1–b4 sit at 0.038, 0.044, 0.042, 0.046 px on every
  trace set; r3 reads 0.048 (Run25), 0.039 (Run26), 0.056 (Run27), 0.055 (Run28), 0.015 (Run29);
  b, r and n all drop to 0.015–0.03 px from Run29. A percentile over the pooled detectors is
  set by the worst detector (n by n1's 0.089) and cuts through the top detector's cluster (r4 at
  the p95 of its own run), so with 22–24 values the p95 and p99 coincide and the thresholds flag
  Run26's late b1/b4 trace sets at +10 % and all of Run28's r3. Adopted instead (2026-10-20): a
  tolerance about each detector's Run25 median, WARN 20 % and FAIL 50 %, as `medFwhm` has. The
  regular trace sets of Runs 26–30 sit within +12 % of it (b1 and r1 reach +24 % and +28 % at the
  end of Run26: WARN), the Run27 SM1 defocus at 1.7–4× (FAIL, with r2 at 2.2× and one Run27 set at
  1.5–1.9× on the other cameras); Runs 28–30 flag nothing. b2 and n4 keep Run25 as their scatter
  reference (their scatter is unremarkable there), Run29 being their image-quality reference only.
  The m arm has two known-good values per detector, below the floor: not judged.
- **Bootstrap, IIS and health-check quartz are not full-slit quartz.** Their trace scatter is
  0.07–0.13 px, twice a trace set's, on every camera (Run25 132766–132767, 132958–132999,
  135903–135904; Run26 Bootstrap sets; Run29 147778–147780 `ImageQuality (Halogen)`); they are
  `scienceTrace` frames, so comparison mode gated them with the trace sets until they were made
  unvalidated by sequence name (`ENGINEERING_SEQUENCES`). Cobra scans (`theta_scan_*`, `home`)
  with the quartz on scatter like a trace set and stay gated.
- **Lamp-line scatter is run-dependent.** Thresholds from Run29's 2026-07-14 and 07-23 arcs
  (24–28 values per species, 6–7 visits), judged at WARN, flag 20–70 % of the lamp arcs of Runs
  25–28 (r-arm Krypton 71 % in Run25) and about none of Runs 29–30's: the reserved-line scatter
  of lamp lines against the visit's adjusted map differs between runs by more than it varies
  within one, so no fixed reference transfers. A per-detector tolerance about Run25's median
  (20 %/50 %, as the traces) does: adopted 2026-10-20 in x for Ar, Kr, Ne and Xe on every arm,
  with the calibration's own input arcs kept in the reference (zero against the calibration is
  the drift's concern; against the visit's own map a calibration input is a visit like any other),
  which gives 4–10 values per detector and species. On the standard arcs of Runs 25, 26 and 28–30
  it fails at most one image a run; b-arm Xenon, with 60–110 lines per detector and weakening
  (b1: 89 in Run25, 65 in Run29), warns on 15–25 % of its arcs in every run, the one noisy
  population. Not adopted: y, where the HgCd species' scatter is intrinsically variable (medians
  0.6–1.1 px on n and r) and a tolerance failed 5–20 % of Runs 25–28's healthy arcs, and the HgCd
  lines in x with it. `LineCheck` and `DetectorMap test` arcs, taken off the calibration recipe,
  fail in bulk on the r arm (40–60 % in Runs 28–29) and are unvalidated by name.
- **The two unnamed Run25 quartz 135903–135904** are an IIS trace sequence (`IIS trace z=0`,
  `noLampCtl`, in the sequence comment; the name is empty) taken on 2025-12-02, after Run25's last
  science night (11-30); Run25's period now ends there, so they are outside every period and not
  judged. Had they fallen inside, the engineering rule, which reads names, would have gated them.

---

## Arc Lamp Physics and b-arm `pctFlagged` Failures

### Root cause

High `pctFlagged` in the b arm for certain lamp types is **lamp physics, not optics**.
Several lamp species have very few or very faint lines in the blue (400–650 nm) region.
`fitDetectorMap` flags lines that fall below its global S/N threshold, causing
artificially high flag rates.

Line counts and intensities in the b arm (from `obs_pfs/pfs/lineLists/`):

| Lamp | b-arm lines | Max intensity | Notes |
|---|---|---|---|
| HgCd | many | ~79 926 | Excellent b-arm coverage; flag rates are genuine |
| Ne | ~505 | high | Dense/crowded; b-arm flag rates reflect crowding |
| Ar | few | ~400 | Faint in b; flag rates are lamp physics, not optics |
| Xe | 148 | ~600 | Very faint in b; flag rates are lamp physics |
| Kr | 222 | ~10 (median) | Most b-arm lines extremely faint |

**SM1 exception**: visits 140005–140138 showed FWHM of 3.83–4.86 px across *all* lamp
types — confirmed as a genuine hardware/optics issue (bad focus or mirror alignment),
not lamp physics.

### Proposed fix: `minSignalToNoisePerSpecies` in `fitDetectorMap`

> **Not implemented.** The field does not exist in `drp_stella`: as of `master` `c771d507`
> (2026-10-02) nothing in `python/` matches `perSpecies`, and only the global
> `minSignalToNoise` is available (`FitDetectorMapConfig`, default 10.0, in
> `python/pfs/drp/stella/fitDetectorMap.py`). A working prototype exists only on the
> **local, unpushed** `drp_stella` branch `adjust-dm-fixes` (commits `e3d1a5db` field,
> `2ffbe7ba` logging; last touched 2026-08-11, no remote, no ticket). The CLI examples
> below describe that interface; against `master` they fail with an unknown-field error.

The proposal is to add a `DictField(keytype=str, itemtype=float, default={})` to
`FitDetectorMapConfig`, allowing per-species S/N thresholds to be set independently of the
global `minSignalToNoise` (default 10). Suggested values for the b arm:

- Ar: 3–5
- Xe: 3–5
- Kr: 5–7
- Ne/HgCd: keep global (10)

**Species string names** come from the `description` column of the line lists in
`obs_pfs/pfs/lineLists/`. The correct keys are ionic species names, **not** the lamp
names from `W_SEQNAM`:

| Lamp (`W_SEQNAM`) | `lines.description` species string |
|---|---|
| `Arc: Argon` | `ArI` |
| `Arc: Xenon` | `XeI` |
| `Arc: Krypton` | `KrI` |
| `Arc: Neon` | `NeI` |
| `Arc: HgCd` | `HgI`, `CdI` |

Using `Ar`, `Xe`, `Kr` as keys will silently match nothing — the global threshold will be
applied to all species.

**CLI syntax** (note: must assign the whole dict as a Python literal because `DictField`
keys can't be set via dot-notation):

```
-c "fitDetectorMap:fitDetectorMap.minSignalToNoisePerSpecies={'ArI': 3.0, 'XeI': 3.0, 'KrI': 5.0}"
```

The outer label (`fitDetectorMap:`) is the pipeline task label from `detectorMap.yaml`;
the inner path (`fitDetectorMap.minSignalToNoisePerSpecies`) refers to the
`ConfigurableField` sub-task and the DictField within it.

**Full example** with other commonly used overrides:

```bash
-c fitDetectorMap:fitDetectorMap.doSlitOffsets=True \
-c fitDetectorMap:fitDetectorMap.order=4 \
-c fitDetectorMap:fitDetectorMap.soften=0.03 \
-c "fitDetectorMap:fitDetectorMap.minSignalToNoisePerSpecies={'ArI': 3.0, 'XeI': 3.0, 'KrI': 5.0}"
```

### `calculateSoftening` NaN crash (dof = 0)

When per-species S/N thresholds are relaxed, individual fibers may have very few
surviving arc lines (e.g. 1 Ar line in b arm). With `yNum=1` and `numParameters=2`,
`yDof = 0`. If the residual is 0.0, `softenChi2(0.0) = 0/0/0 − 1 = NaN`, which crashes
`scipy.optimize.bisect`.

**Proposed fix, not committed — the crash is live.** As of `drp_stella` `master` `c771d507`
(2026-10-02) `calculateSoftening` guards only `residuals.size == 0`; there is no `dof`
guard. Since NaN compares False both ways, `softenChi2(0.0) < 0` and
`softenChi2(maxSoften) > 0` both fall through and `scipy.optimize.bisect` still raises
`f(a) = NaN`.

The fix exists on the same local, unpushed `drp_stella` branch `adjust-dm-fixes`
(commit `5c35840d`): guard `residuals.size == 0 or dof <= 0` → return `0.0` early, and
collapse the pre-bisect check into `if not np.isfinite(val) or val < 0`.

---


## Common Failure Patterns

From engineering run data:

| Symptom | Likely cause | Remedy |
|---|---|---|
| b-arm `pctFlagged` above the `b:Argon`/`b:Xenon`/`b:Krypton` thresholds | Lamp has very few/faint b-arm lines → global S/N cut flags almost all | No remedy available yet — `minSignalToNoisePerSpecies` is only proposed (see above). Treat as known lamp physics rather than a detector fault |
| b-arm `pctFlagged` > 15 % for HgCd arcs (or > 50 % for Ne) | Genuine crowding or optical problem | Investigate `medFwhm`; if FWHM is also high → optics issue |
| High `pctMeasFail` with low `pctLowSN` | Centroid/photometry failures rather than faint lines — not lamp physics | Investigate the image; relaxing S/N thresholds will not help |
| All arms FWHM > 3.5 px for a single spectrograph module | Hardware/focus issue | Flag the entire SM as bad for that visit range |
| `\|medDxCenter\|` > 1 px across all arms of an SM | Flexure or a stale `detectorMap_calib` | Check the calib validity range before blaming the optics |
| `medDxCenter` ≈ 0 but `dxCenterRms` large | Distortion rather than bulk shift | Look at the `dxCenter` distribution in `iqQaData`, not just the summary |
| `traceOnly=True` for all arc visits | `arcLines` connection missing or `minGoodLines` not satisfied | Check `fitDetectorMap` ran and produced `lines` |
| No `medFwhm` (sparse result) for IIS arc or IIS quartz frames | Deliberate. The science arc catalog does not match the 16 engineering fibers, and scattered light from them gave a spurious ~7 px calexp FWHM, so both `is_iis` branches of `imageQualityQa` set `force_sparse = True` and never run the calexp measurement. `pctFlagged` is suppressed as well | Expected; no action. Were the calexp path re-enabled for IIS, the ~7 px figure would return for a second, independent reason — see the next row |
| All `fit*` metric columns are `NaN`/0 | The `*_log` connections were absent from the input collection | Expected when `reduceExposure` hasn't run; the IQ metrics themselves are unaffected |
| `ValueError: f(a) = NaN` in bisect during `fitDetectorMap` | Per-fiber dof=0 when very few arc lines remain after S/N cut | Proposed `dof <= 0` guard in `calculateSoftening` is **not** committed (see above), so relaxing per-species S/N is not yet safe |

---
