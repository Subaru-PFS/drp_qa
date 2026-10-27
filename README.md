DRP QA
======

## Introduction

This repository contains the pipeline and corresponding tasks for the quality assurance (QA) of the Prime Focus
Spectrograph (PFS) Data Release Production (DRP)
pipeline. The QA tasks are implementations of the `PipelineTask` class in the LSST Science Pipelines. The tasks are run
on the output of the DRP pipeline to assess the quality of the data products.

## The image-quality gate

`imageQualityQa` is a **gate**: for each image (visit × detector) it decides whether the image should be
discarded and not counted in its group, and it shows problems that span an instrument, spectrograph, arm or
detector. Its verdict is persisted, so that calibration building and the science pipeline can gate on it.

- **It judges quick reductions.** `drpActor` reduces each image during observing (`reduceExposure`, its
  `hilo` pipeline). That is not the reduction the science release uses. `qaActor` listens to `drpActor` and
  reads `drpActor/reductions`, so QA never reduces anything a second time; production QA must not re-reduce.
  Thresholds are therefore derived in `drpActor`'s reduction context. Comparison mode is the one exception: it
  reduces only what `drpActor/reductions` lacks, with `drpActor`'s per-sequence-type config, into its own
  collection.
- **Two modes, one judgement.** *Real time*: `qaActor` judges each sequence group as `drpActor` finishes it
  (INSTRM-3040). *Comparison*: a batch over every visit of a run, listed from the opdb (PIPE2D-1929; see
  [Comparison mode](#comparison-mode)). Both judge with `pfs.drp.qa.metrics.gate`.
- **Comparison works on a partial run.** Run on day 10 of a 20-day run, it gives valid verdicts, and a later
  re-run adds the new nights' images without changing earlier verdicts: a verdict depends only on its own
  image, each period and drp_qa version has its own collection, and what is already judged is skipped.
- **Verdicts are stored in the Butler**, in `iqQaMetrics` and `dmLineResidualSummary` (`qaStatus`, `qaReason`),
  and copied to qadb (`spt_qa_database`) afterwards by `pfs.drp.qa.qadb`, with the thresholds they were judged by
  and one gate verdict per image; see [Loading QA results into qadb](#loading-qa-results-into-qadb) (PIPE2D-1385).
- **Gate, not product assessment.** Judging the quality of science products is a separate job, not yet
  designed; tasks here should stay on one side of that line.
- **Not covered yet:** cosmic-ray processing (PIPE2D-1609).

The work is tracked in epic PIPE2D-1928.

## Installation

`drp_qa` is a `pyproject.toml`-based package with no compiled components, so there is nothing to build:

```bash
pip install -e .          # or: uv sync
```

Within the LSST stack, EUPS setup still works and is the usual route — `drp_stella`
must be set up first:

```bash
source /path/to/stack/loadLSST.bash
setup -r ../drp_stella
setup -r .
```

`ups/drp_qa.table` only manipulates `PATH`/`PYTHONPATH`; there is no SCons build step. The `bin/` directory that SCons
used to generate is therefore not created, and the scripts under `bin.src/` are invoked directly
(see [Command-line tools](#command-line-tools)).

### Development

```bash
ruff format .          # formatting
ruff check --fix .     # linting and import sorting
pytest tests/
```

Ruff configuration, pytest configuration, and package metadata all live in
`pyproject.toml`; there are no standalone per-tool config files.

## QA Pipeline

The QA pipelines is located at `pipelines/drpQA.yaml`, which runs all of the QA tasks.

> Note: individual tasks can be specified by using the `pipelines/drpQA.yaml#extractionQA`
> syntax.
>
See [documentation](https://pipelines.lsst.io/modules/lsst.pipe.base/creating-a-pipeline.html#command-line-options-for-running-pipelines)
> for details.

Also see the example notebook [`examples/QA Pipelines.ipynb`](examples/QA%20Pipelines.ipynb).

`drpQA.yaml` currently wires up `dmLineResiduals`, `extractionQa`, `extractionQaCombined`, and `imageQualityQa`. The
remaining tasks below are implemented and importable
but are not registered in the pipeline; run them by referencing their task class directly.

### Tasks

#### `dmLineResiduals`

Per detector, measures every line of `lines` against two references (`reference`): `detectorMap_calib` (`calib`), the
calibration the visit is reduced against, whose offset is drift; and the visit's own `detectorMap` (`visit`), adjusted
to these lines, whose `RESERVED` lines test how well it predicts lines it was not fitted to, as it must on a science
frame. The lines are those the visit's detectorMap fit marked `DETECTORMAP_USED` or `DETECTORMAP_RESERVED`, less
`BAD` ones, with no other selection or clipping; trace centroids are kept on arcs. When the fit marked none, every
good line is measured, as `UNFITTED`. Trace centroids measure x, lamp lines x and y, measured minus reference.
Each row carries the frame's observation type (`obsType`, from the raw header's `W_SEQTYP` as `imageQualityQa`
classifies it: `trace` for a quartz, `arc`, `twilight`, ...; the pfsConfig header stands in when the raw's lacks the
card): trace centroids are measured on every frame, but on an
arc's faint continuum they scatter two to three times more than on a quartz, so they are a population per frame type.

Per reference, line species and line set (`statusType`: `BOTH`, `USED`, `RESERVED`, or `UNFITTED`): the median and
robust RMS of the x and y residuals, `nLines` and `linesPerFiber`. Judged: against `calib`, over `BOTH` (or
`UNFITTED`) lines, the median x offset of traces and median y offset of lamp lines (WARN 1 px, `APPLY_SHIFT`; FAIL 2
px, `RECALIBRATE`); against `visit`, the robust RMS of `RESERVED` lines (`RECALIBRATE`), by the thresholds file
`dmLineResidualThresholds-run25.yaml` (PIPE2D-1956): trace centroids on quartz frames and the Ar, Kr, Ne and Xe
lines in x, per detector, at WARN 20 % and FAIL 50 % above the detector's known-good median in Run25, and the lamp
lines in y at WARN 50 % and FAIL 150 %. Stored, not judged: the mercury and cadmium lines, whose y scatter is
intrinsically variable; trace centroids on arcs and twilights; and the sky-line species on twilights. The weighted RMS and
the softening are drp_stella's, in `iqQaMetrics` and `iqQaSpeciesMetrics`, not recomputed. Logic in the stack-free
`pfs.drp.qa.metrics.lineResiduals`. No plot is drawn as the task runs: plot `dmLineResidualData` afterwards.

It runs in `pipelines/qaThresholds.yaml` beside `imageQualityQa`, so the thresholds notebook and comparison mode
measure it on every visit they judge: comparison mode reports its verdicts beside the gate's
([Comparison mode](#comparison-mode)), and the notebook derives the scatter thresholds from the validation visits
([QA thresholds](#qa-thresholds-and-the-validation-visit-set)), written to `dmLineResidualThresholds-<run>.yaml` for
`thresholdsFile`.

##### Options

- `minLines`: fewest lines a species needs to be measured. Default `10`.
- `lineFlags`: `ReferenceLineStatus` names that exclude a line. Default `["BAD"]`.
- `maxTraceRowsPerFiber`: most `USED`, and most `RESERVED`, trace rows per fiber stored in `dmLineResidualData`,
  evenly spaced in y; `0` for all. Default `100`. Lamp and sky lines and `UNFITTED` rows are always stored; the
  statistics use every line, so a trace statistic recomputed from the table is an estimate.
- `thresholdsFile`: a thresholds file judged before the fields below; entries may name `reference`, `statusType`,
  `arm`, `spectrograph`, `description`, `cameraSet` and `obsType`. Default `dmLineResidualThresholds-run25.yaml`,
  adopted by `makeLineResidualThresholds.py` beside it; empty for the fields alone.
- `xWarnThreshold`, `xFailThreshold`, `yWarnThreshold`, `yFailThreshold`: on the median offsets from the
  calibration, in pixels. Defaults `1.0`, `2.0`, `1.0`, `2.0`.
- `xRmsWarnThreshold`, `xRmsFailThreshold`, `yRmsWarnThreshold`, `yRmsFailThreshold`: on the robust RMS of
  `RESERVED` lines against the visit's detectorMap, in pixels. Default NaN: not judged.

##### Outputs

| DataSet Type            | Dimensions                             | Description                                                                                                       |
|-------------------------|----------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `dmLineResidualMetrics` | `instrument, visit, arm, spectrograph` | Long format: one row per reference, species, `statusType` and metric, with `obsType` and the metric's verdict.    |
| `dmLineResidualSummary` | `instrument, visit, arm, spectrograph` | One row: `qaStatus`, `qaDecidedBy`, `qaReason`, `recommendedAction`, `fitted`, `obsType`, and the calibration used. |
| `dmLineResidualData`    | `instrument, visit, arm, spectrograph` | One row per line measured (`USED` and `RESERVED` traces sampled), with `dxCalib`, `dyCalib`, `dxVisit`, `dyVisit`. |

##### Coming from `dmResiduals`

`dmResiduals` and `dmCombinedResiduals` were retired on 2026-10-22 (PIPE2D-1956). Where their outputs went:

| Before                                            | Now                                                                                                      |
|---------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `dmQaResidualStats`, a row per `status_type` and species | `dmLineResidualMetrics`, long format; `lineResiduals.wideResiduals` gives a column per metric. `reference` `visit` is what `dmResiduals` measured against; `calib` is new. |
| `spatial.median`, `spatial.robustRms`             | `medianDeltaX`, `robustRmsDeltaX`                                                                        |
| `wavelength.median`, `wavelength.robustRms`       | `medianDeltaY`, `robustRmsDeltaY`                                                                        |
| `weightedRms`, `softenFit`                        | not recomputed: drp_stella's own, in `iqQaSpeciesMetrics`                                                |
| `num_lines`, per fiber                            | `nLines` (lines of the species) and `linesPerFiber`                                                      |
| the verdict                                       | `dmLineResidualSummary`: `qaStatus`, `qaDecidedBy`, `qaReason`, `recommendedAction`, the calibration used |
| `dmQaResidualData`                                | `dmLineResidualData`, with a residual against each reference                                             |
| `dmQaResidualPlot`                                | nothing per quantum: drawn from `dmLineResidualData` after the fact (PIPE2D-1920)                        |
| `dmQaDetectorStats`, `dmQaCombinedResidualPlot`   | comparison mode's report, *Line residuals* section; `bin.src/exportMetrics.py` for the tables           |

#### `imageQualityQa`

Measures image quality (FWHM), spatial flexure, and arc-line flag rates on a per-detector quantum
`(instrument, visit, arm, spectrograph)`. The task classifies each visit by reading FITS headers (`W_SEQTYP`,
`W_SEQNAM`) and routes it to the appropriate measurement path. When `reduceExposure` logs are available it also folds
ISR, cosmic-ray, and `fitDetectorMap` statistics into the metrics table.

The task writes data only; plotting lives in `pfs.drp.qa.plotting.iqQa` and is driven after the fact by
`bin.src/plotIqQaTimeSeries.py` (see
[Command-line tools](#command-line-tools)).

##### Inputs

| Connection          | Dataset              | Required    | Description                                                                                                                                                                                                                 |
|---------------------|----------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `arcLines`          | `lines`              | Yes         | Per-line arc emission measurements from `fitDetectorMap`, including second moments (`ixx`/`iyy`), flux, flags, and species.                                                                                                 |
| `detectorMap`       | `detectorMap`        | Yes         | Adjusted mapping from `(fiberId, wavelength)` to detector `(x, y)`. Used to enrich arc lines with wavelength and trace position.                                                                                            |
| `fiberProfiles`     | `fiberProfiles`      | Yes (calib) | Stored fiber profile shapes. Used as a last-resort fallback to derive trace-width FWHM when neither arc lines nor a calexp are available.                                                                                   |
| `detectorMapCalib`  | `detectorMap_calib`  | Yes (calib) | Static, pre-adjustment detectorMap. Differencing its `xCenter` against the measured fiber positions gives the flexure diagnostic `dxCenter`.                                                                                |
| `calexp`            | `calexp`             | No          | Post-ISR calibrated image. When present, trace widths are measured directly from its pixels (each fiber fitted with its neighbours) — the primary path for trace/quartz visits and a fallback for sparse arc visits.        |
| `pfsConfig`         | `pfsConfig`          | No          | Fiber configuration for the visit. When provided alongside `calexp`, restricts FWHM measurement to `FLUXSTD+GOOD` fibers to avoid contamination from sky fibers. Also supplies `W_SEQTYP` metadata when `calexp` is absent. |
| `isrLog`            | `isr_log`            | No          | ISR task log. Parsed for bad-pixel counts and runtime.                                                                                                                                                                      |
| `cosmicrayLog`      | `cosmicray_log`      | No          | Cosmic-ray task log. Parsed for CR counts, affected pixels, and runtime.                                                                                                                                                    |
| `reduceExposureLog` | `reduceExposure_log` | No          | `reduceExposure` log. Parsed for `fitDetectorMap` chi², RMS, softening, per-species and per-fiber statistics.                                                                                                               |

All five optional connections are declared `minimum=0`, so the task still runs when they are absent from the input
collection.

##### Measurement Paths

The task classifies each visit from the `W_SEQTYP` FITS header and selects the best available measurement path in
descending preference order:

| Visit type                                        | Primary path                                                           | Fallback                                                               |
|---------------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------|
| Regular arc (`scienceArc`, all fibers)            | Arc-line second moments (`ixx`/`iyy` → FWHM)                           | `calexp` trace widths if `nGoodLines < minGoodLines`                   |
| IIS arc (16 engineering fibers)                   | —                                                                      | Reported as sparse (no FWHM); arc catalog does not match IIS positions |
| Regular trace/quartz (`scienceTrace`, all fibers) | `calexp` trace widths                                                  | Fiber profile calibration widths if the `calexp` is absent or unusable |
| IIS trace/quartz                                  | —                                                                      | Reported as sparse; too few fibers for a reliable measurement          |
| Science / all-sky (`scienceObject*`)              | `calexp` trace widths of `FLUXSTD+GOOD` fibers                         | Sparse if FLUXSTD good fraction < `minFluxstdGoodFrac`                 |
| Unknown (`W_SEQTYP` absent)                       | Heuristic: arc lines if sufficient, else `calexp`, else fiber profiles | —                                                                      |

For arc visits, FWHM is computed as the Gaussian-equivalent width `2√(2 ln 2) × σ` from the second moments of each
arc-line profile. On the calexp, every `profileYStride`-th row is fitted with `pfs.drp.qa.crossDispersion`: each fiber
jointly with its neighbours, as pixel-integrated Gaussians sharing one width, plus a background, with no aperture
(fibers are 6.17 px apart). The `fiberProfiles` fallback reads each profile's width at half its maximum. Every FWHM
includes the pixel response, as the arc-line moments do; `iqQaData.fwhmIntrinsic` leaves it out.

The PFS trace profile is not Gaussian, so these readings were chosen by comparing them on real quartz
([`docs/trace-widths.ipynb`](docs/trace-widths.ipynb), PIPE2D-1918): the joint fit recovers the profile's half-maximum
width, a Gaussian fitted to the profile understates it by 8–10 %, and drp_stella's second moment
(`calculateStatistics().width`) varies between detectors whose profiles agree.

##### What the Metrics Tell Us

- **`medFwhm`** — Median FWHM in pixels across all good measurements for the detector quantum. Reflects the
  cross-dispersion width of the fiber PSF. Typical good values (Run25) are 2.5–3.2 px on arcs and 3.1–3.4 px on quartz;
  values above 3.5 px indicate a focus or alignment problem (SM1 defocused in Run27 read 4.1–4.5 px on quartz).
- **`medDxCenter`** / **`dxCenterRms`** — Median and scatter of the spatial offset of the measured fiber positions
  from the static calibration detectorMap (measured minus calibration, as drp_stella's residuals), in pixels. This is a flexure diagnostic: a large `|medDxCenter|` means
  the whole detector has shifted relative to the calibration, while a large `dxCenterRms` at small median suggests a
  distortion rather than a bulk shift. `NaN` when no `detectorMap_calib` is available.
- **`pctFlagged`** — Percentage of arc lines flagged by `fitDetectorMap` (below S/N threshold or otherwise rejected).
  High flag rates on b-arm Ar/Xe/Kr arcs are expected (lamp physics — those species have very few bright lines in the
  blue) and do not indicate an optical problem. High flag rates on HgCd or Ne arcs, or on r/n/m arms, are more likely to
  reflect genuine issues.
  `pctFlagged` is set to `NaN` for IIS frames, sparse visits, and science/FLUXSTD paths where the flag rate reflects
  exposure depth rather than optical quality.
- **Flag breakdown** — When `pctFlagged` is finite, the same rejections are broken out by
  `ReferenceLineStatus` bit: `pctNotVisible`, `pctBlend`, `pctSuspect`, `pctRejected`,
  `pctBroad`. On the arc-line path two measurement-level categories are added:
  `pctLowSN` (flagged with no flux measured — the fit never got that far) and `pctMeasFail`
  (flagged despite a finite flux — a centroid or photometry failure). The split matters:
  `pctLowSN` is usually lamp physics, `pctMeasFail` usually is not.
- **`nLines`** — Rows of the `lines` dataset, whichever path measures the FWHM: a row for every line of the
  lamp's list on every `GOOD` fiber, flagged when not measured, and the trace centroids `fitDetectorMap` found,
  on an arc as on a quartz. Not judged: it follows how many fibers the design lights, and on an arc the trace
  centroids, whose number follows the illumination and the pipeline rather than the lamp (PIPE2D-1935).
- **`linesPerFiber`** — The judged count: unflagged lamp lines per lit fiber on an arc, unflagged trace centroids
  per lit fiber on a quartz (about 4,160, one per detector row); `NaN` on other frames or without a pfsConfig.
  A design that lights one fiber group (a *shaka* exposure: every fourth cobra, the others parked behind their black
  dots; see `docs/qa-domain-notes.md`) is judged like a full slit;
  an obstructed frame, lit by its design but dim, still fails. On Run30's daily neon, a one-group arc measures
  the same lines on each lit fiber as a fully lit one.
- **`nLitFibers`**, **`nFibers`** — The detector's fibers the pfsConfig lights (`fiberStatus` `GOOD`, engineering
  fibers excepted: the only ones `lines` has rows for), and those it lights or hides behind a black dot
  (`BLACKSPOT`). `pctFlagged` on the arc-line path is taken over the lamp lines of the lit fibers, and the calexp
  trace width over the lit fibers. **`nTraceRows`** counts the trace centroids in `lines`.
- **`traceOnly`** — `True` when FWHM comes from fiber profile calibration widths rather than live measurements; these
  values reflect the calibration epoch, not the current visit, so `medFwhm` is then neither judged nor used to
  derive thresholds, and `pctFlagged` is `NaN`. A quartz falls back to them whenever its calexp gives no usable
  measurement.
- **`obsType`** / **`seqName`** — Visit classification (`arc`, `trace`, `science`, `allsky`, `twilight`, `dark`,
  `unknown`) and the raw `W_SEQNAM` string (e.g. `"Arc: HgCd"`) it was derived from. Twilight and dark frames are
  not measured.
- **`qaStatus`** — `PASS`, `WARN`, or `FAIL`: the worst verdict of the metrics judged (see thresholds below), from
  `pfs.drp.qa.metrics.gate.gate`; `UNKNOWN` when no metric was judged. **`qaDecidedBy`** names the first metric with that verdict (empty for `PASS`), and
  **`qaReason`** gives every `WARN` and `FAIL` reason; it is also on the task log's `IQ QA` line.

##### Pass/Warn/Fail Thresholds

New thresholds are derived from the validation visits: see
[QA thresholds and the validation visit set](#qa-thresholds-and-the-validation-visit-set).

Verdicts come from `pfs.drp.qa.metrics.gate.gate`, the one gating path, which also re-judges stored `iqQaMetrics`.
Each metric is judged by the `thresholdsFile` entry for its population, and by the config fields below where the file
has none. `thresholdsFile` is an absolute path, or one relative to `pfs/drp/qa/metrics/data`; empty, the config fields
judge alone.

The default, `iqQaThresholds-run25.yaml`, holds the thresholds derived from the validation visits, as adopted
(PIPE2D-1917, PIPE2D-1918, PIPE2D-1942, PIPE2D-1938) by `makeRun25Thresholds.py` beside it; each entry records its
provenance. They were derived from `u/wtg/qa-thresholds/010`, every validation visit reduced afresh with the pipeline
the file records (datamodel w.2026.40, obs_pfs w.2026.40, drp_stella w.2026.41-6-ge56938d2). The reference is Run25,
except that b2 and n4 quartz take Run29's detectorMap set (see [QA thresholds](#qa-thresholds-and-the-validation-visit-set)).

- `medFwhm` per detector and lamp: WARN 3 % and FAIL 5 % above the reference known-good median of the image's arm,
  spectrograph, `obsType` and lamp (`species`), on arcs and quartz (from the calexp width), every arm included.
  The spectrographs of one arm differ from each other by more than any one of them varies (r quartz: r2 3.27 px,
  r4 3.39 px, each within 0.03 px), and the lamps of one arm differ even more (b arcs: Argon 2.5 px, Krypton 2.85 px).
  b-arm Argon is not judged: under a line per lit fiber, its median width rests on a handful of lines and the
  known-good exposures scatter −4 to +2 % about their median, so the SM1 defocus is caught on the other lamps.
- `medFwhm` per arm, judging only an image whose detector and lamp have no reference median: arcs b 2.872/2.89,
  r 2.988/2.993, n 3.05/3.1, m 3.18/3.21 px; quartz b 3.33/3.34 (Run25 and Run29's b2 pooled), r 3.391/3.392,
  n 3.269/3.272 px, and m not judged (16 known-good values, below the 20-sample floor).
- `linesPerFiber` per arm and sequence (`seqName`), lower is worse (PIPE2D-1935): a tolerance below the reference
  known-good median, 25 %/50 % on arcs and 2.5 %/5 % on quartz. Known-good arcs reach 19 % below their median
  (n HgCd) and quartz 1.1 %; the obstructed Run30 frames sit 60–98 % below on arcs and 9–56 % on quartz. b Argon,
  under a line per lit fiber, is not judged. `nLines` is not judged: it counts rows, mostly trace centroids on an
  arc, on only the fibers a design lights. Its Run25 thresholds also failed some known-good Run25 arcs.
- `pctFlagged` is not judged: one entry with neither WARN nor FAIL, which also overrides the config fields. Run25's
  derivation had 5 visits per population, and the config's n-arm 15/20 % sits below the known-good n-arm Neon sets
  of Run25 (34 %) and Run30 (26–35 %), as comparison mode showed (PIPE2D-1929). It stays unjudged until there is a
  valid reference.

The config fields:

| Metric            | WARN                               | FAIL                               |
|-------------------|------------------------------------|------------------------------------|
| `medFwhm`         | ≥ 3.2 px (`fwhmWarnThreshold`)     | ≥ 3.5 px (`fwhmFailThreshold`)     |
| `\|medDxCenter\|` | ≥ 1.0 px (`dxCenterWarnThreshold`) | ≥ 2.0 px (`dxCenterFailThreshold`) |
| `pctFlagged`      | per-arm / per-lamp, see below      | per-arm / per-lamp, see below      |

`pctFlagged` thresholds are `DictField`s whose keys are either an arm letter (`b`) or an
`arm:species` pair (`b:HgCd`), where the species is the part of `W_SEQNAM` after the colon. Lookup order is
`arm:species` → `arm` → a built-in default of 15 % (WARN) / 20 % (FAIL). Set a threshold to `100` to disable it.

| Key           | WARN | FAIL | Rationale                                               |
|---------------|------|------|---------------------------------------------------------|
| `b`           | 50 % | 60 % | Blue arm, no lamp identified                            |
| `b:HgCd`      | 15 % | 25 % | HgCd has excellent b-arm coverage, so flags are genuine |
| `b:Neon`      | 50 % | 60 % | Dense and crowded in the blue                           |
| `b:Krypton`   | 55 % | 65 % | Most b-arm Kr lines are extremely faint                 |
| `b:Xenon`     | 85 % | 92 % | Very faint in the blue — lamp physics, not optics       |
| `b:Argon`     | 93 % | 97 % | Almost no usable b-arm Ar lines                         |
| `r`, `n`, `m` | 15 % | 20 % | Full lamp coverage; high flag rates are real            |

The per-lamp b-arm values are deliberately permissive: several arc species have very few or very faint lines between 400
and 650 nm, so `fitDetectorMap`'s global S/N cut rejects almost all of them regardless of image quality. Treating that
as a failure would flag every Ar and Xe arc. See the "Arc Lamp Physics" section of
[`docs/qa-domain-notes.md`](docs/qa-domain-notes.md) for the underlying line counts. That section also describes a
per-species S/N option (`minSignalToNoisePerSpecies`) that would address the cause, but it is **proposed only** — it
does not exist in `drp_stella`, so the permissive thresholds above are the current mitigation.

##### Options

###### Measurement

- `imageQualityQa:minGoodLines`: Minimum good arc-line measurements to trust the arc path. Default `10`.
- `imageQualityQa:minPeakSN`: Minimum significance of a calexp trace (fitted flux over its error). Default `5.0`.
- `imageQualityQa:maxCalexpFlagRate`: Max fraction of bad calexp samples before rejecting the calexp path. Default
  `0.5`.
- `imageQualityQa:minFluxstdGoodFrac`: Min fraction of good FLUXSTD samples for the stellar calexp path. Default `0.10`.
- `imageQualityQa:profileHalfWidth`: Deprecated and ignored; the calexp width uses no aperture.
- `imageQualityQa:profileYStride`: Row sampling interval (pixels) for calexp trace widths. Default `100` (about 40
  rows); the fit's cost is proportional to the rows.

###### Pass/Fail Thresholds

- `imageQualityQa:fwhmWarnThreshold`: Median FWHM (px) above which status is `WARN`. Default `3.2`.
- `imageQualityQa:fwhmFailThreshold`: Median FWHM (px) above which status is `FAIL`. Default `3.5`.
- `imageQualityQa:dxCenterWarnThreshold`: `|medDxCenter|` (px) above which status is `WARN`. Default `1.0`.
- `imageQualityQa:dxCenterFailThreshold`: `|medDxCenter|` (px) above which status is `FAIL`. Default `2.0`.
- `imageQualityQa:flagRateWarnThreshold`: `pctFlagged` (%) threshold for `WARN`, keyed by `arm` or `arm:species`. See
  the threshold table above for defaults.
- `imageQualityQa:flagRateFailThreshold`: `pctFlagged` (%) threshold for `FAIL`, keyed by `arm` or `arm:species`. See
  the threshold table above for defaults.
- `imageQualityQa:thresholdsFile`: Thresholds file judged before the fields above. Default `iqQaThresholds-run25.yaml`; empty for the fields alone.

`DictField` values cannot be set with dot notation on the command line; assign the whole dict as a Python literal:

```bash
-c "imageQualityQa:flagRateWarnThreshold={'b': 50.0, 'r': 15.0, 'b:Argon': 93.0}"
```

##### Outputs

| DataSet Type  | Dimensions                             | Description                                                                                                                                                                       |
|---------------|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `iqQaData`    | `instrument, visit, arm, spectrograph` | Per-line or per-trace-sample measurements: `fiberId`, `x`, `y`, `lam`, `fwhm`, `fwhmIntrinsic` (calexp only), `theta`, `dxCenter`, `flux`, `fluxErr`, `flag`, `traceOnly`, `peakRatio`, `snr` (calexp only), `status`, plus `visit`/`arm`/`spectrograph`. |
| `iqQaMetrics` | `instrument, visit, arm, spectrograph` | Single-row summary; see below.                                                                                                                                                    |
| `iqQaSpeciesMetrics` | `instrument, visit, arm, spectrograph` | Per-species `fitDetectorMap` statistics from the `reduceExposure` log, one row per species and metric: `visit`, `arm`, `spectrograph`, `description` (line species, e.g. `NeI`, or `Trace`), `metric` (`fitXRms`, `fitYRms`: weighted RMS; `fitXSoften`, `fitYSoften`: softening), `value`. |

`iqQaMetrics` columns, by group:

- **Identity** — `visit`, `arm`, `spectrograph`, `obsType`, `seqName`, `calexpRun` (the RUN collection of the
  calexp judged, whose `pfsArm` records the reducing pipeline; null without a calexp)
- **Image quality** — `medFwhm`, `medDxCenter`, `dxCenterRms`, `pctFlagged`, `nLines`,
  `traceOnly`, `qaStatus`, `qaDecidedBy`, `qaReason`
- **Flag breakdown** (only when `pctFlagged` is finite) — `pctNotVisible`, `pctBlend`,
  `pctSuspect`, `pctRejected`, `pctBroad`, and on the arc-line path `pctLowSN`, `pctMeasFail`
- **Log-derived** (zero/`NaN` when the corresponding log is absent) — `isrBadPixels`,
  `isrTime`, `cosmicRayCount`, `cosmicRayPixels`, `cosmicRayTime`, `reduceExposureTime`
- **`fitDetectorMap` statistics** — `fitChi2`, `fitDof`, `fitXRms`, `fitYRms`, `fitXSoften`,
  `fitYSoften`, `fitNLines`, `fitTotalLines`, the matching `fitReserved*` columns for the reserved-line sample,
  `fitTraceXRms`, `fitTraceYRms`; the per-species values are in `iqQaSpeciesMetrics`
- **Per-fiber arrays** — `fiberIds`, `fiberXRms`, `fiberYRms`, `fiberNLines`

There is no `iqQaPlot` output: the task writes data only, and plots are produced separately by
`bin.src/plotIqQaTimeSeries.py`.

#### `extractionQa`

Determines the quality of the fiber extraction.

##### Options

- `extractionQa:fiberWidth`: Half width of a fiber region (pix), default `3`.
- `extractionQa:plotMinChiMed`: Minimum median Chi to plot, default `-1.5`.
- `extractionQa:plotMaxChiMed`: Maximum median Chi to plot, default `1.5`.
- `extractionQa:plotMinChiStd`: Minimum standard deviation of Chi to plot, default `0.0`.
- `extractionQa:plotMaxChiStd`: Maximum standard deviation of Chi to plot, default `3.5`.
- `extractionQa:plotMinChiAtPeak`: Minimum Chi at peak to plot, default `-1.5`.
- `extractionQa:plotMaxChiAtPeak`: Maximum Chi at peak to plot, default `1.5`.
- `extractionQa:plotMinResFrac`: Minimum residual fraction, default `-5.0`.
- `extractionQa:plotMaxResFrac`: Maximum residual fraction, default `5.0`.
- `extractionQa:plotHistRangeScale`: The scale factor for the Chi histogram range, default `1.5`.
- `extractionQa:plotHistNbin`: The number of bins for the Chi histogram, default `100`.
- `extractionQa:targetType`: Target type for which to calculate statistics, default `["^ENGINEERING"]`.
- `extractionQa:figureDpi`: Resolution of plot for residual, default `72`.

##### Outputs

| DataSet Type        | Dimensions                             | Description                                                                                                                                                 |
|---------------------|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `extQaStats`        | `instrument, visit, arm, spectrograph` | Results of the residual analysis of extraction against fiberId are plotted (Chi_median, Chi_stddev, Chi_atProfilePeak, residual_fraction).                  |
| `extQaImage`        | `instrument, visit, arm, spectrograph` | Summary plot for 2D residual, chi, etc. (p.1),  2D image of chi (p.2), zoom-in of the chi image (p.3), and the histogram of residual fraction and chi (p.4) |
| `extQaImage_pickle` | `instrument, visit, arm, spectrograph` | Statistics of the residual analysis.                                                                                                                        |

#### `extractionQaCombined`

##### Options

- `extractionQaCombined:plotMinChiMed`: Minimum median Chi to plot, default `-1.5`.
- `extractionQaCombined:plotMaxChiMed`: Maximum median Chi to plot, default `1.5`.
- `extractionQaCombined:plotMinChiStd`: Minimum standard deviation of Chi to plot, default `0.0`.
- `extractionQaCombined:plotMaxChiStd`: Maximum standard deviation of Chi to plot, default `3.5`.
- `extractionQaCombined:plotMinChiAtPeak`: Minimum Chi at peak to plot, default `-1.5`.
- `extractionQaCombined:plotMaxChiAtPeak`: Maximum Chi at peak to plot, default `1.5`.
- `extractionQaCombined:plotMinResFrac`: Minimum residual fraction, default `-5.0`.
- `extractionQaCombined:plotMaxResFrac`: Maximum residual fraction, default `5.0`.
- `extractionQaCombined:targetType`: Target type for which to calculate statistics, default `["^ENGINEERING"]`.
- `extractionQaCombined:figureDpi`: Resolution of plot for residual, default `72`.
- `extractionQaCombined:footnoteSize`: Fontsize of the footnote, default `9`.

##### Outputs

| DataSet Type         | Dimensions               | Description                                                                                                                    |
|----------------------|--------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `extQaStatsCombined` | `instrument, visit, arm` | Results of the residual analysis of extraction plotted against fiberId for all spectrographs (left) and PFI positions (right). |

#### `fiberNormsQa`

Plot the fiber normalization for the given detector and visit.

##### Options

- `fiberNormsQa:plotLower`: Lower bound for plot (standard deviations from median), default 2.5.
- `fiberNormsQa:plotUpper`: Upper bound for plot (standard deviations from median), default 2.5.

##### Outputs

| DataSet Type     | Dimensions               | Description                                   |
|------------------|--------------------------|-----------------------------------------------|
| `fiberNormsPlot` | `instrument, visit, arm` | Plot of the fiber normalizations for a visit. |

#### `fluxCalQa`

##### Options

- `fluxCalQa:filterSet`: Filter set to use, default `ps1`.
- `fluxCalQa:includeFakeJ`: Include the fake narrow J filter, default `True`.
- `fluxCalQa:fakeJoffset`: Offset from the ps1 bands for the fake narrow J, default `0.054`.
- `fluxCalQa:diffFilter`: Filter to use for the color magnitude difference, default `g_ps1`.
- `fluxCalQa:doAnonymize`: Mask sensitive information in the plot, default `True`.

##### Outputs

| DataSet Type         | Dimensions          | Description                                        |
|----------------------|---------------------|----------------------------------------------------|
| `fluxCalStats`       | `instrument, visit` | Statistics of the flux calibration analysis.       |
| `fluxCalMagDiffPlot` | `instrument, visit` | Plot of the flux calibration magnitude difference. |

#### `skySubtractionQa`

There are two tasks in the Sky Subtraction QA, the `skyArmSubtractionQa` and `skySubtractionQa`. The
`skyArmSubtractionQa` task is used to subtract the sky from the spectra of each arm and the `skySubtractionQa` task is
used to plot the results of the sky subtraction for the entire visit.

##### Options

If config options are not passed, the default values come from `mergeArms_config.fitSkyModel`.

- `skyArmSubtractionQa:blockSize`: Block size for the sky subtraction, default `None`.
- `skyArmSubtractionQa:rejIterations`: Number of rejection iterations, default `None`.
- `skyArmSubtractionQa:rejThreshold`: Rejection threshold, default `None`.
- `skyArmSubtractionQa:oversample`: Oversampling factor, default `None`.
- `skyArmSubtractionQa:mask`: Mask types to use, default `None`.

##### Outputs

| DataSet Type           | Dimensions               | Description                                                                                 |
|------------------------|--------------------------|---------------------------------------------------------------------------------------------| 
| `skySubtractionQaPlot` | `instrument, visit, arm` | PDF of various plots related to sky subtraction <br/>built from all the arms for the visit. |

## QA thresholds and the validation visit set

Thresholds are derived from data, by a fixed procedure, and checked against visits whose
verdict is known. The decisions behind the current set:

- **Run25 is the reference run.** Its calibrations were made for it, and it is the most complete stable set. Every
  threshold is derived from Run25's known-good visits.
- **The validation set holds Run25's known-good visits**, and Run29's for two detectors' quartz (below). From
  other runs it holds only known-bad visits, as fault examples: each is checked against the thresholds, and none
  derives one.
- **A set is never its own reference.** The gate decides whether a run's calibration set should be used to build its
  calibrations, so a set counts as known-good only once the calibrations built from it have proved out (used, with good
  reductions), and then only for later runs. Each detector's reference is chosen from such sets (PIPE2D-1938): reduced
  with one pipeline, Runs 25–30 agree with Run25 to within 3 % in `medFwhm` everywhere but two cameras, so Run25 stays
  the reference except for b2 quartz (SM2's slit was refocused on 2026-03-11 and its quartz width stepped from 3.12
  to 3.34 px, where it has stayed) and n4 quartz (Run25's 3.10 px is the outlier against a steady 3.22 px since), which
  take Run29's detectorMap set (`cameras`/`excludeCameras` in the YAML; `referenceRuns: [25, 29]`). The n-arm
  verdicts were provisional while Run25's n-arm reductions predated the pipeline fixes; reduced afresh, the n arm agrees
  with Run25 like the others. Adopted 2026-10-19 from `u/wtg/qa-thresholds/010`: re-judged with the new file, Runs
  28, 29 and 30 go from 41, 57 and 32 flagged images to 1, 3 and 3, the rest being single arcs and n4's Neon arcs,
  whose width has crept from +1.5 % (Run26) to +4.7 % (Run30) of Run25's while its quartz stayed put.
- **Other runs are compared, not used for derivation.** Comparison mode (PIPE2D-1929) judges every visit of a run
  against Run25's thresholds; Run30 is the first. Run27 is not a fair comparison: no
  calibrations were made from it, and its n-arm flag rates sit 2–5 points above Run25's.
- **`imageQualityQa` judges by the adopted Run25 file** (`iqQaThresholds-run25.yaml`): `medFwhm` on arcs and
  quartz, and `linesPerFiber`. Flag rates (`pctFlagged`) are not judged until there is a valid reference; quartz `medFwhm` is
  measured from the calexp width (PIPE2D-1918); twilight is not measured. Details under
  [Pass/Warn/Fail Thresholds](#passwarnfail-thresholds).
- **`medFwhm` is judged against each detector's own Run25 width, per lamp** (PIPE2D-1942): WARN 3 % and FAIL 5 %
  above the known-good median of its arm, spectrograph, `obsType` and lamp. Per-arm percentiles put each arm's widest
  spectrograph, or broadest lamp, at FAIL, and per detector and lamp there are only 4–14 known-good values, too few
  for a tail percentile but plenty for a median. The 3 % and 5 % are set from the data, not from an image-quality
  requirement: Run25's known-good values sit at most 1.4 % above their median (3.3 % for Argon arcs, whose b, r and m
  widths scatter most), healthy detectors moved −3 to +4 % between Run25 and Run30, and the SM1 defocus reads +24 to
  +41 %. On Run25 no known-good arc or quartz fails and one warns (135841 m1 Argon); every known-bad one fails.
- **`dmLineResiduals` judges drift by tolerance, scatter by derived thresholds** (PIPE2D-1921, PIPE2D-1956). The
  median offsets from `detectorMap_calib` are judged at WARN 1 px and FAIL 2 px, a physical limit short of
  drp_stella's 3 px line-search radius: offsets from a calibration sit near zero in the run it came from, so no
  percentile applies. The robust RMS of `RESERVED` lines against the visit's own detectorMap is derived in the
  notebook like the other percentile thresholds, per arm, species, camera set and observation type, and as a
  tolerance per detector, from the known-good visits of each arm's reference run, and adopted through the task's
  `thresholdsFile` (2026-10-20, from
  `u/wtg/qa-thresholds/010`): trace centroids on quartz frames and the Ar, Kr, Ne and Xe lines are judged in x per
  detector by a tolerance, WARN 20 % and FAIL 50 % above the detector's known-good median, as `medFwhm` is, and the
  lamp lines in y at WARN 50 % and FAIL 150 % (their y moves more between runs: the healthy runs' arcs reach 1.4×
  Run25's medians at the 95th percentile and 2.4× at worst, Run27's faults 3–70×). A
  detector's quartz trace scatter repeats to about 1 % within a run and differs 20 % between detectors, so a
  percentile over the pooled detectors is set by the worst one and cuts through the best one's cluster, and a
  percentile over one run's arcs does not transfer to the next (thresholds from Run29's arcs flagged 20–70 % of Runs
  25–28's healthy arcs); against the per-detector median, the regular trace sets and arcs of Runs 25, 26 and 28–30
  fail at most one image a run and warn on a few percent (b1 and r1 traces at +24 % at the end of Run26, b-arm
  Xenon arcs), and the Run27 SM1 defocus fails at 1.7–4×. The calibration's own input visits are kept in the scatter
  derivation (they sit at zero against the calibration, which is the drift's concern, not against the visit's own
  map). Not judged, with the reasons in each entry's provenance: the mercury and cadmium lines, whose y scatter is
  intrinsically variable; trace centroids on arcs and twilights, which scatter two to
  three times more on the faint continuum; and the twilights' sky species. `bin.src/judgeScatterThresholds.py` is
  the acceptance check, run on the comparison exports of Runs 25–30 before any adoption, engineering sequences set
  apart; its findings are in `docs/qa-domain-notes.md`. `dmResiduals`, which it replaced, was retired on 2026-10-22
  after a comparison on Run30's fresh reductions: medians agree to under 0.002 px and robust RMS to 5–14 %
  (`dmResiduals` clipped), and the two tasks measure the same RMS on every image `dmLineResiduals` flagged.
- **b-arm populations are not split by camera set** (PIPE2D-1927). The camera set (`cameraSet`, the arms a visit
  read) is stored with `dmLineResiduals`' rows and is a population column of the scatter thresholds, but its one effect
  is the b arm's offset from `detectorMap_calib` when read with m (−0.45 to −0.47 px in Runs 25 and 28–30), which
  `medianDeltaX` measures and the 1 px drift WARN absorbs. The b-arm trace scatter does not depend on it: on quartz
  frames, b read with m and with r agree within 3 % per detector within a run (Runs 25, 26, 28 and 29; Run30 +13 %, on
  5 images read with m), so the per-detector scatter tolerances are not split by it. Pooled over runs the two differ,
  because the scatter moves twofold between runs (0.025–0.054 px) and the camera sets are not spread evenly over them:
  compare them within a run. `iqQaMetrics`' `medDxCenter` is superseded by `dmLineResiduals`' `medianDeltaX`.
- **Nothing measured is `UNKNOWN`, not `PASS`.** A FWHM read from the `fiberProfiles` calibration is not a
  measurement and is never judged.
- **Every known-bad entry is caught, or says why not** (PIPE2D-1934). `pfs.drp.qa.metrics.coverage` checks each
  confirmed known-bad entry against the thresholds the gate uses: every image it covers, with its metric measured,
  must reach its verdict. An entry no check can catch carries `unchecked: <reason>` in the YAML, and one that is
  caught must not; `tests/metrics/test_coverage.py` runs it on the stored validation metrics. Unchecked today: the
  b-arm `pctFlagged` baselines (flag rates not judged), the Run27 unlit-spectrograph IIS arcs (no science fiber
  lit; PIPE2D-1954) and the Run25 cloudy twilight (nothing measured on twilights sees cloud; PIPE2D-1973).
- **Thresholds hold for the pipeline that reduced their visits** (PIPE2D-1938). Two reductions of the same Run25
  exposures put n-arm quartz widths up to 53 % apart, so a verdict can follow the reduction rather than the
  instrument. The thresholds file records that pipeline (`pipelineVersions`: datamodel, obs_pfs and drp_stella, as
  drp_stella's `getPfsVersions` writes them into each `pfsArm` header). Given it, the gate marks a verdict on a
  reduction made with another pipeline, or an unknown one, provisional (`qaProvisional`, with the reason in
  `qaPipeline`); comparison mode reads each reduction's versions from the `pfsArm` in the RUN of the calexp judged
  (`calexpRun`), so a later reduction of the same detector can't stand in for it, and counts the provisional
  verdicts in its report. Metrics written before `calexpRun` read as an unknown pipeline until judged again. A file that records no pipeline makes every verdict provisional.
- **Re-baseline at each summit stack upgrade.** The live gate judges `drpActor`'s reductions, made with the stack
  deployed at the summit, so a new stack needs thresholds of its own:
  1. Reduce the reference sets fresh with the new stack:
     `qaComparison.py run --period run25 --fresh --cadence set` (and the other reference runs).
  2. Re-derive the thresholds from those reductions with [`docs/qa-thresholds.ipynb`](docs/qa-thresholds.ipynb),
     which records their pipeline.
  3. Compare the new thresholds with the old, metric by metric, and explain any that moved.
  4. Adopt the new file; until then, verdicts on the new stack's reductions are provisional.

The detail:

- [`docs/qa-thresholds.ipynb`](docs/qa-thresholds.ipynb) — **the procedure, step by step**, as a notebook run top
  to bottom: run `imageQualityQa` over the validation visits, derive the thresholds, review them and adopt them.
- [`docs/validation-visits.md`](docs/validation-visits.md) — which visits are in the validation set, why, and
  how to change it.
- [`docs/qa-principles.md`](docs/qa-principles.md) — the rules every metric and threshold follows, and the checklist
  for a PR that adds one.

## Guider tools

`pfs.drp.qa.guiders` analyses the AG (guider) cameras' data in notebooks: guide errors, the boresight and
per-camera models, drift, focus and image sizes. It replaces `pfs.drp.stella.utils.guiders`, and needs `pfs_utils`
but not the LSST stack.

- `queries` reads the opdb (and, for INST-PA and early M2_OFF3, a butler). It is the only module that does.
- `analysis` fits and averages; `plotting` draws. Both take DataFrames and return their results, leaving their
  inputs alone.
- `coordinates` holds the conventions: positions in hardware coordinates, offsets as a center minus a named
  reference in microns, units in the column names. 1 AG pixel is 13 µm, or 0.14 arcsec.

```python
from pfs.drp.qa.guiders import analysis, plotting, queries
from pfs.utils.database.opdb import OpDB

opdb = OpDB(host="pfsa-db", user="public_user")
agcData = queries.readAgcData(opdb, range(148284, 148292))

plotting.showAgcErrorsForVisits(agcData)

fit = analysis.fitGuiderModel(agcData, analysis.GuiderFitConfig(maxGuideError_um=100))
plotting.showGuiderErrors(fit, plotting.GuiderPlotConfig(guideStarFrac=0.3))

plotting.plotFocus(agcData, plotBy="agc_exposure_id", showFocusSets=True)
```

Each plot draws on a new figure, or on the `fig`, `ax` or `axes` it is given, and returns a `GuiderPlot` holding
the figure, its panels, the artists and the colorbars. Given the `colorbars` of an earlier call, it updates them,
so a figure can be redrawn in place. The notebook [`docs/guider-plots.ipynb`](docs/guider-plots.ipynb) describes
each plot (what it shows, where its data come from, how to read it and what to look for) beside an annotated sample.
It runs on the tests' AG data or, with `USE_OPDB = True`, on the opdb, where it also checks the readers and fits the
plots don't use.

Coming from drp_stella ([`docs/guiders-migration.ipynb`](docs/guiders-migration.ipynb) runs each of its calls
beside the replacement):

- Fit, then plot. `GuiderConfig` is split into `analysis.GuiderFitConfig` and `plotting.GuiderPlotConfig`.
  drp_stella's `estimateGuideErrors(plot=True)`, `plotDriftRate` and `compareAGCPfsUtils` become
  `analysis.estimateGuideErrors`, `analysis.fitDriftRate` and `analysis.comparePfsUtilsPositions`, whose results
  `plotting.plotGuideErrors`, `plotting.plotDriftRate` and `plotting.plotPfsUtilsComparison` draw.
- The plots take AG data, not an opdb connection. Cameras are `agcCameraIds`, 0-5, rather than `AGC`, 1-6, and the
  cursor readout takes design names by visit (`designNames`).
- The readers become `queries.readAgcData`; `compat` keeps their names, deprecated.

A raster scan visit holds two pointings: once its shutters close, the telescope moves to the next position. The
`boresight` reference, and the plots using it, are only right for its shutter-open rows (`shutter_open == 1`)
(PIPE2D-1910).

## Comparison mode

`bin.src/qaComparison.py` gates every image of an observing run, or of the calibrations taken before it, against
the reference thresholds, and writes a report comparing the run with Run25 (`pfs.drp.qa.comparison`).

- **The visits come from the opdb**, not from the hand-written run summaries: every SpS visit between the
  period's dates, with its sequence, `cmd_str`, cameras, design, obslog notes and telescope status. The periods
  (each run's nights and its pre-run calibration nights, HST, a night running noon to noon) are in
  [`comparison/data/observingRuns.yaml`](python/pfs/drp/qa/comparison/data/observingRuns.yaml), derived from
  gaps in SpS activity.
- **Calibration or science is what the exposure is**, not when it was taken: every sequence type but
  `scienceObject` and `scienceObject_windowed` is a calibration, and those are science when their design has a
  science proposal. Twilight frames, dithers and telescope focus sweeps are on engineering designs, so they are
  calibrations.
- **Everything measurable is judged; only validated types are gated.** `scienceArc`, `scienceTrace` and
  `scienceObject` are *gated*: their thresholds were derived from and checked against such visits. Every other type
  `imageQualityQa` can measure (engineering arcs, flats, fiber profiles, focus sweeps, windowed readouts, new
  types, and the engineering sequences taken as a validated type: bootstraps, IIS `ImageQuality` frames, health
  checks, line checks and detectorMap tests, by sequence name) is judged the same way but labelled *unvalidated*, in its own pass and its own report section: a `FAIL`
  there may be what an engineering test expected. Biases and darks have no method yet; test exposures are listed
  only.
- **n-arm darks are listed with the last lit exposure before them**, from the opdb alone: the n detectors keep an image
  of a bright exposure for a while, so the darks taken soonest after an arc or quartz are the ones to check for
  persistence until darks are measured (PIPE2D-1925). The report shows each night's n-arm exposures on a
  timeline, colored by what lit them, beside the gap before each dark sequence; `darksLastLit-<period>.csv` has
  every dark.
- **Daily arcs and traces are reported apart from sets.** A calibration *set* is 3 exposures per arc lamp and 10
  traces; one-visit sequences of its sequence group (`group_id`) belong to it. Another one-visit sequence is *daily* when
  its name recurs on 4 or more nights (from Run28, a neon arc and a trace, and IIS frames, each day, for drift), and a
  one-off *single* otherwise (positioner scans, bootstrap exposures), judged like a set. All are
  judged against the thresholds derived from Run25's sets. Daily ones are often taken with one fiber group lit:
  `linesPerFiber` and `pctFlagged` are taken on the lit fibers, so they are judged like sets; metrics measured before
  that (no `nLitFibers`) skip `nLines` and `pctFlagged` on them (PIPE2D-1935). Run30's nightly `ImageQuality` frames are
  IIS, reported sparse until PIPE2D-1954.
- **Sky visits are labelled** from the telescope status: a *focus sweep* (10 or more focus offsets spanning at least
  0.5 mm on one field in one night) changes how much light enters the fibers but not the line widths, so its
  `nLines`, `linesPerFiber` and `pctFlagged` aren't judged; *dithered* visits are marked.
- **Findings are written from the data.** Each `WARN` or `FAIL` image gets the metrics that crossed, how far the
  problem spreads (detector, arm, spectrograph, most of the visit, the whole sequence), the setup and the opdb's
  notes beside it. Notes never decide a verdict.
- **Line residuals are judged beside the gate** (PIPE2D-1956). `dmLineResiduals` runs in the same passes as
  `imageQualityQa`, and the report's *Line residuals* section has its verdicts per detector and night, the
  recommended actions by arm, its problems and findings (the reference, species and metric of each crossing), and the
  scatter of the lines the fit held back, by arm, observation type and species (the b arm split by whether it was
  read with m or with r), beside the reference run's: what a scatter threshold is derived from. Tables stored before `obsType` was a column, or that stored `unknown`, take it from `iqQaMetrics`. `dmLineResidualSummary-<period>.parquet`, `dmLineResidualMetrics-<period>.parquet` and
  `lineResidualFindings-<period>.csv` are written beside the report. A collection judged before the task joined the
  pipeline has none, and the section says so.

Four steps, each safe to repeat. Files go under `--data-dir`: opdb reads in `opdb/`, everything else in
`<run>/qa/<drp_qa version>/`; the report shows science visits, so keep it local.

```bash
python bin.src/qaComparison.py fetch  --period run30 --data-dir ~/Projects/Subaru/PFS/data   # opdb, read-only
python bin.src/qaComparison.py plan   --period run30 --data-dir ~/Projects/Subaru/PFS/data   # coverage, pipetask commands
python bin.src/qaComparison.py run    --period run30 --data-dir ~/Projects/Subaru/PFS/data   # reduce what's missing, judge
python bin.src/qaComparison.py report --period run30 --data-dir ~/Projects/Subaru/PFS/data   # report-run30.html
```

`plan` prints the `pipetask` commands as Python lists, for running from a notebook instead; `-j` sets the
`pipetask` processes (default 8). `--cadence set` (or `daily`, `single`; repeatable) judges only the arcs and traces
of those cadences, e.g. the calibration sets that thresholds are derived from; the other visits stay listed as
`cadence not selected`. By default the reductions in `drpActor/reductions` are reused, each made with the
pipeline of its day, and so are those an earlier drp_qa version's comparison of the period made beside them, its
verdicts being judged again; `--fresh` reduces everything
with the current pipeline instead, so runs are compared on a level field (their calibrations are still each
run's own). Its collection and files are named after that pipeline, `<period>/<version>/pfs-<weekly>` (from
drp_stella's `getPfsVersions`, each product's version if they differ; `--pipeline-version` names it), so a new
pipeline starts afresh while a new drp_qa version reuses the same pipeline's reductions and only judges again. Since
the label comes from the stack set up in the shell, `run --fresh` stops when its label has no reductions of the
period yet while another label has (the whole reduction again, where a judge pass was meant); `--new-pipeline`
reduces anyway, `plan` and `--dry-run` only warn, and a run resumed under its own label is not stopped. Passes are
`calibration`,
`sky` and `unvalidated-calibration`, gated first, each split by sequence: sequences with an image to reduce get the
whole pipeline, and sequences already reduced by drpActor only the QA tasks, `imageQualityQa` and `dmLineResiduals`
(a `-judge` pass, e.g. `sky-judge`;
`--pass sky` selects both). Otherwise pipetask reruns `cosmicray` wherever a reduction lacks the `cosmicray_log`
that `imageQualityQa` optionally reads, though the reduction is reused; the judge-only images then have no
cosmic-ray counts, which are not gated. `run --dry-run` builds
each pass's graph (`pipetask qgraph`) and prints its quanta without writing anything; `--pass calibration`
or `--pass sky` limits `run` to one pass. Visits with no `pfsConfig` in the inputs can't be reduced and are
reported, not scheduled. A failed quantum fails its pass but not the others; `run` lists the failures at the end and
exits 1. Images whose quanta failed in an earlier run (read from the period's `pipetask` logs) are `failed`, listed
in the report and not rescheduled, since they fail the same way each time; `--retry-failed` retries them. The
output collection is
`u/$USER/comparison/<period>/<version>`, the version being `git describe` of the last commit that could change what
the pipeline writes (not the comparison driver, its report, tests or docs), so a change to the driver keeps the
collection; uncommitted changes to the rest are refused. Report the reference
(`--period run25`) first: its stored metrics are what each other run is compared with.

## Loading QA results into qadb

`pfs.drp.qa.qadb` copies what `imageQualityQa` and `dmLineResiduals` wrote to the Butler into qadb
(`spt_qa_database` 0.2.0), after the fact: the Butler is the record, and a database outage cannot fail a quantum.
One code path serves qaActor after each visit's pipetask (`loadVisit`; INSTRM-3040), comparison mode after a run,
and the backfill of whole collections (`bin.src/loadQadb.py`).

- **Tables.** `iqQaMetrics` → `image_quality_qa`, `iqQaSpeciesMetrics` → `image_quality_qa_species`,
  `dmLineResidualSummary` → `dm_line_residual_qa`, `dmLineResidualMetrics` → `dm_line_residual_qa_metric`: the same
  columns in snake_case (`pfs.drp.qa.qadb.tables`), keyed by the processing, the visit and the camera (`b1`). A
  Butler column reaches qadb once both the schema and the table's spec here name it; the fiber arrays and the
  per-flag percentages of `iqQaMetrics` stay in the Butler.
- **Processing.** Each Butler RUN found is a `data_processing_pipe2d` row (collection and RUN name), with the drp_qa
  that wrote it (the RUN's `packages` dataset) and the pipeline that reduced the images it judged (`datamodel`,
  `obs_pfs` and `drp_stella`, from the judged calexps' `pfsArm` headers as comparison mode reads them; null when the
  RUN judged reductions of more than one pipeline, or none is recorded).
- **Thresholds.** The thresholds files the tasks' stored configs name are registered in `qa_thresholds`, identified
  by content (sha256), with their entries in `qa_threshold_entry`; `qa_gate` cites them.
- **Gate.** `qa_gate` holds each image's worst verdict across the tasks (`FAIL` > `WARN` > `PASS` > `UNKNOWN`; a task
  that judged nothing never outranks one that did), the task and metric that decided it, every task's reason, the
  mode (`realtime`, `comparison`, `backfill`) and whether the verdict is provisional. PIPE2D-1930 reads this table.
- **Idempotent.** Every write is `INSERT ... ON CONFLICT DO UPDATE` on the row's key, and a long table's rows of each
  image are deleted before its reload: load a collection again after a re-judgement, or after a load that stopped,
  and the rows are replaced, not duplicated, and a species or metric the new judgement lacks is gone. Visits qadb's `pfs_visit`
  lacks get a row, filled from the opdb when one is given.

```bash
# What would be written, as parquet, touching no database
python bin.src/loadQadb.py -c u/wtg/comparison/run30/<version> --visit-range 148302 148348 --dry-run ~/reports/dry
# Load it (qadb user that may write; credentials in ~/.pgpass), with the opdb's visit descriptions
python bin.src/loadQadb.py -c qaActor/reductions --visits 150001 150002 --opdb-host pfsa-db
```

The tests (`tests/qadb/`) run the writer's statements on SQLite through SQLAlchemy, whose `ON CONFLICT` and
`RETURNING` match PostgreSQL's; set `PFS_QADB_TEST_DSN` to a scratch PostgreSQL to run them there too. The
schema the tests create (`tests/qadb/conftest.py`) mirrors `spt_qa_database`'s migration `a7c3e5d91b24`: change both
together.

## Command-line tools

Scripts live in `bin.src/` and are run directly — there is no SCons step to copy them into a `bin/` directory on `PATH`:

```bash
python bin.src/<script>.py --help
```

### `qaComparison.py`

Comparison mode; see [Comparison mode](#comparison-mode).

### `judgeScatterThresholds.py`

The acceptance check for adopting line-scatter thresholds (PIPE2D-1956): judges the reserved-line scatter that
comparison mode exported for each run (`dmLineResidualMetrics-<period>.parquet` beside the report, with
`iqQaMetrics-<period>.parquet` for the observation type of tables stored before `obsType` was a column, or that
stored `unknown`) with a
thresholds file, and prints WARN and FAIL counts per run, arm, observation type and species. Thresholds that fail a
run's healthy frames in bulk describe the reference visits' night, not the instrument. Stack-free.

### `fiberNormsQa.py`

Thin entry point for `pfs.drp.qa.fiberNormsQa.main`, which plots fiber normalizations from a Butler collection.

### `loadQadb.py`

Loads a collection's QA tables, thresholds and gate verdicts into qadb; see
[Loading QA results into qadb](#loading-qa-results-into-qadb). Requires the stack (it reads the Butler) and a qadb
user that may write; `--dry-run DIR` writes parquet files instead.

## Command-line tools

Scripts live in `bin.src/` and are run directly — there is no SCons step to copy them into a `bin/` directory on `PATH`:

```bash
python bin.src/<script>.py --help
```

### `fitDetectorMapLogQa.py`

Parses `pipetask run` log files from the detectorMap pipeline and reports a per-quantum
`OK` / `WARN` / `BAD` verdict for each `fitDetectorMap` execution: fit RMS, softening, line counts, per-species
statistics, and slit-offset failures. Exits non-zero if any quantum is `BAD`, so it can be used as a gate in a reduction
script.

**Requires only the Python standard library** — no LSST stack — with `matplotlib`
needed just for `--plot`. Quanta are keyed off the `(fitDetectorMap:{...})` label that
`pipetask` writes into the logger context, so logs must be captured with that context intact.

```bash
python bin.src/fitDetectorMapLogQa.py run28-dm-02.log run28-dm-03.log
python bin.src/fitDetectorMapLogQa.py --warn-yrms 0.12 --plot run28-dm-03.log
```

Thresholds default to `yRMS` 0.10 px (WARN) and `xRMS` 0.05 px (WARN), overridable with
`--warn-yrms` / `--bad-yrms`. A quantum is also flagged when its arc-line total falls below 25 % of the best peer in the
same spectrograph module.

> Note: this parser and `drp_stella`'s `bin.src/parse_detectormap_log.py` read the same
> log strings independently, so `fitDetectorMap`'s log format is effectively an API with
> two consumers. Both the old format and the current one — which carries an inline
> `arm=<a> spectrograph=<n>` prefix as of `drp_stella` `16174310` — are accepted here.

### `imageQualityLogQa.py`

Builds a per-visit image quality report from `reduceExposure` and `imageQualityQa` logs, or by querying the Butler
directly. Produces a diagnostic dashboard plot, a markdown report, and a JSON dump of the parsed state. `numpy` and
`matplotlib` are only needed for
`--plot-dir`.

```bash
# From log files
python bin.src/imageQualityLogQa.py run28.log --plot-dir plots/ --report-out qa.md

# From a butler collection
python bin.src/imageQualityLogQa.py \
    --butler-repo /path/to/butler --collection u/user/run28 \
    --visit 140005 --spectrograph 1 --arms b r
```

`--json-out` writes the parsed state, and `--json-in` replays it, so plots and reports can be regenerated without
reparsing.

### `plotIqQaTimeSeries.py`

Reads `iqQaMetrics` datasets across a collection and produces the multi-panel time-series figure (FWHM, flexure, flag
breakdown, pass/fail heatmap) implemented in
`pfs.drp.qa.plotting.iqQa.plotIqTimeSeries`. Requires the LSST stack for the Butler query; `--csv` bypasses it.

```bash
python bin.src/plotIqQaTimeSeries.py \
    -b /path/to/butler -c u/user/run28 -o iq_timeseries.png

python bin.src/plotIqQaTimeSeries.py --csv metrics.csv --arm b,r --obs-type arc
```

Filters: `--arm`, `--spectrograph`, `--obs-type` (all comma-separated), plus `--where`
for an arbitrary Butler query expression.

### `fiberNormsQa.py`

Thin entry point for `pfs.drp.qa.fiberNormsQa.main`, which plots fiber normalizations from a Butler collection.
