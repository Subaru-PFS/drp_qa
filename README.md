DRP QA
======

## Introduction

This repository contains the pipeline and corresponding tasks for the quality assurance (QA) of the Prime Focus
Spectrograph (PFS) Data Release Production (DRP)
pipeline. The QA tasks are implementations of the `PipelineTask` class in the LSST Science Pipelines. The tasks are run
on the output of the DRP pipeline to assess the quality of the data products.

## How QA works

QA is four steps:

1. **Tasks measure.** `imageQualityQa`, `dmLineResiduals` and `twilightFluxQa` write numbers per image
   (visit × detector) to the Butler: widths, offsets, scatter, line counts, sky-line residuals. No task holds a
   limit or gives a verdict.
2. **One file holds the limits.** `pfs.drp.qa.metrics.registry.CHECKS` holds 13 checks, each a WARN and a FAIL
   limit and the frame types it applies to. Nothing else in the package holds a limit.
3. **One table holds each camera's normal values.** qadb's `qa_normal_value`, made once per run from the run's
   calibration frames (see [Normal values](#normal-values)).
4. **One function gives the verdict.** `pfs.drp.qa.metrics.gate.gate` takes the numbers, the checks and the normal
   values, and returns `PASS`, `WARN`, `FAIL` or `UNKNOWN` per image. An image takes its worst check. A visit takes
   the worst of its images (`pfs.drp.qa.metrics.visitRollup`).

`drpQa.py load` copies a collection's numbers from the Butler into qadb (`spt_qa_database`), calls the gate, and
stores one verdict per image in `qa_gate`; see [Loading QA results into qadb](#loading-qa-results-into-qadb).

- **It judges quick reductions.** `drpActor` reduces each image during observing (`reduceExposure`, its `hilo`
  pipeline). That is not the reduction the science release uses. `qaActor` listens to `drpActor` and reads
  `drpActor/reductions`, so QA never reduces anything a second time. [Redoing a run](#redoing-a-run) is the
  one exception: it re-reduces a run with an updated pipeline, with `drpActor`'s per-sequence-type config,
  into its own collection.
- **A verdict is on an image, not on a science product.** Judging the quality of science products is a separate
  job, not designed here.
- **Not covered:** cosmic-ray processing.

### The checks

A check is *fixed* or *relative*. A fixed check has its limits in the metric's units and needs only the number
measured. A relative check has its limits in per cent from the camera's normal value for the frame type and lamp
or species, in the metric's bad direction; without a normal value it gives no verdict. A value at a limit has
crossed it.

The 13 checks, as `CHECKS` holds them:

| Check | Judges | Metric | Frames | Arms | WARN | FAIL | Compared with |
|---|---|---|---|---|---|---|---|
| `skyOffset` | offset from the calibration | `medianDeltaX`, `medianDeltaY` | science, all-sky | all | 0.5 px | 1 px | fixed limit |
| `skyLineResidual` | OH residual after the visit's adjustment | `medianDeltaYKms` | science, all-sky | r, n, m | 0.5 km/s | 1 km/s | fixed limit |
| `skyLineResidualBlue` | [O I] residual after the visit's adjustment | `medianDeltaYKms` | science, all-sky | b | 1.5 km/s | 2.5 km/s | fixed limit |
| `calibOffset` | offset from the calibration | `medianDeltaX`, `medianDeltaY` | arcs, quartz | all | 1 px | 2 px | fixed limit |
| `traceCentreOffset` | trace centre offset | `medDxCenter` | engineering-fiber arcs and quartz | all | 1 px | 2 px | fixed limit |
| `twilightLevel` | twilight level offset | `twilightLevelOffset` | twilights | b, r, m | 2 % | 3 % | fixed limit |
| `twilightScatter` | twilight residual scatter | `twilightResidRms` | twilights | b, r, m | 1.5 % | 2.5 % | fixed limit |
| `twilightPatch` | twilight patch | `twilightPatch` | twilights | b, r, m | 2.5 % | 4 % | fixed limit |
| `width` | line and trace width | `medFwhm` | arcs, quartz, engineering-fiber arcs and quartz | all | +3 % | +5 % | normal value per camera and lamp |
| `scatterX` | line scatter in x | `robustRmsDeltaX` | arcs, quartz | all | +20 % | +50 % | normal value per camera and species |
| `scatterY` | line scatter in y | `robustRmsDeltaY` | arcs | all | +50 % | +150 % | normal value per camera and species |
| `linesPerFiberArc` | lines per lit fiber | `linesPerFiber` | arcs, engineering-fiber arcs | all | −25 % | −50 % | normal value per arm and lamp |
| `linesPerFiberQuartz` | lines per lit fiber | `linesPerFiber` | quartz, engineering-fiber quartz | all | −2.5 % | −5 % | normal value per arm |

The frame types that are judged are `science`, `allsky`, `arc`, `quartz`, `twilight`, `iisArc` and `iisQuartz`.
The last two are lit through the 16 engineering fibers of each spectrograph by the IIS lamps.

What a frame is comes from its IIC sequence type (`W_SEQTYP`, stored as `seqType`): `registry.SEQUENCE_TYPES`
gives each sequence type its frame type, and `registry.frameType` adds the sequence name, which tells a twilight
and an all-sky frame from science, and the illumination. A row stored without `seqType` is typed by the tasks'
`obsType`, as before. Nothing else in the package decides what a frame is; both tasks read the same table for
the measurement they make (`registry.observationType`).

A frame type either has checks or is named in `registry.NOT_JUDGED`, with the reason no check applies to it:

| Frame type | Sequence types | Why it is not judged |
| --- | --- | --- |
| `dark` | `darks`, `masterDarks` (and `scienceDark`, which the IIC does not write) | No lamp or sky is exposed. |
| `bias` | `biases`, `masterBiases` | No lamp or sky is exposed. |
| `fiberProfile` | `fiberProfiles` | The slit is dithered in x between the exposures, and one group of fibers is lit at a time. |
| `dotRoach` | `dotRoach`, `dotRoachInit`, `dotRoachNew`, `dotScan` | Cobras are moved between the exposures, so the fibers lit change within the sequence. |
| `throughFocus` | `detThroughFocus`, `fpaThroughFocus`, `slitThroughFocus`, `defocusedArcs` | The focus is stepped between the exposures. |
| `dithered` | `ditheredArcs`, `ditheredFlats` | The slit is dithered between the exposures. |
| `engineering` | `arcs`, `flats`, and their `_binned` forms | Taken with the IIC's plain expose command, with any lamp, exposure time, cameras and row binning. |
| `domeFlat` | `domeFlat` | No normal value is kept for a dome flat. |
| `windowed` | `scienceTrace_windowed` | Only part of the detector is read. |

Their images are `UNKNOWN`, with the frame type and its reason as the verdict's reason. A sequence type the table
lacks gives no frame type, and the reason names it (`no frame type for sequence type rampObject`), so a new
sequence type shows up as untyped instead of being judged as something it is not. A sequence that resumes another
(`dotRoach_continued`) is the same kind of frame. To classify a sequence type, add it to `SEQUENCE_TYPES`; to name
a new kind that is not judged, add it to `NOT_JUDGED` with its reason.

What each check reads (`gate.measurements`):

- `skyOffset`, `calibOffset`: residuals against `detectorMap_calib`, over every good line (`BOTH`, or `UNFITTED`
  when the visit's fit marked none). In x, the median offset of the trace centroids. In y, the species furthest
  off: the sky species (OH, O2, OI, NaI) on a science or all-sky frame, the lamp species on an arc.
- `skyLineResidual`, `skyLineResidualBlue`: residuals against the visit's own detectorMap, the median of the
  held-back (`RESERVED`) lines in km/s: OH on r, n and m, [O I] on b.
- `scatterX`, `scatterY`: against the visit's detectorMap, the robust RMS of the `RESERVED` lines: each lamp
  species of an arc but mercury and cadmium, and in x the trace centroids of a quartz.
- `width`, `linesPerFiberArc`, `linesPerFiberQuartz`, `traceCentreOffset`: `iqQaMetrics` columns.
- `twilightLevel`, `twilightScatter`, `twilightPatch`: the twilight metrics, each exposure measured against the
  other exposures of its sequence (see [Twilights](#twilights)).

Only `iqQaMetrics` records a frame's illumination. An arc or quartz with line residuals and no `iqQaMetrics` row
is not judged on them.

### Not judged

An image with no check judged is `UNKNOWN`, never `PASS`. `registry.EXCLUSIONS` names what is measured and stored
but not judged, each with its reason:

- Width and line or trace scatter on science and all-sky frames. They follow fiber signal and exposure time: the
  width is 15 to 30 % wider on faint fibers, and trace scatter 15 to 20 % higher on 60 s exposures.
- Scatter of mercury and cadmium lines, in x and y. It varies by itself from frame to frame of a healthy run.
- Scatter of trace centroids on arcs and twilights. On a faint continuum they scatter two to three times more
  than on a quartz.
- Sky species on twilights, and the n arm on twilights. Its residuals exceed the noise model on clear nights.
- A width resting on fewer than 5 lines per lit fiber, and a line count whose normal value is below that. Too few
  lines for a median: this covers b-arm Argon.
- Line residuals on frames lit through the engineering fibers. No normal values are kept for them. Their width
  and lines per lit fiber are judged, against their own.
- A relative check whose normal value rests on fewer than 3 exposures, or that has none. A median of one or two
  exposures is not a normal value, however many detectors each gave.
- Flagged-line rate (`pctFlagged`). It is a property of the line list and the fit's flags, not of the frame.
- Saturation (`nSaturatedLines`, `pctSaturatedPixels`) and line flux. Stored for cross-visit comparison; no limit
  has been set.
- Every number measured on a frame type of `registry.NOT_JUDGED` (the table above), for the reason given there.
- Frames with no frame type. Their sequence type is missing or not in `SEQUENCE_TYPES`.

A width read back from the `fiberProfiles` calibration (`traceOnly`) counts as not measured.

### Science frames

A science or all-sky frame is judged on two things, both fixed limits, the same for short and long exposures:

- **Offset from the calibration** (`skyOffset`): the traces in x, and the sky species furthest off in y. WARN at
  0.5 px, FAIL at 1 px.
- **Sky-line residual after the visit's detectorMap adjustment**: the median of the held-back lines of the arm's
  species, in km/s. OH on r, n and m at 0.5 and 1 km/s (`skyLineResidual`); [O I] on b at 1.5 and 2.5 km/s
  (`skyLineResidualBlue`).

An image is judged on whichever was measured: one with no sky-line residual is judged on its offsets alone.

The calibration is made with the r arm. On a visit read with the m arm, the b arm sits a fraction of a pixel off
in x, and the n arm in y. That offset is subtracted before `skyOffset` is judged, and the verdict's reason says
so. The offsets are rows of the normal values (`mArmOffsetX` per b camera, `mArmOffsetY` per n camera): the median
with m minus the median with r, from the run's quartz (b, in x) and Neon arcs (n, in y), with at least 3 visits
in each mode.

### Normal values

A normal value is the median of one metric on one camera (or arm), for one frame type and lamp or species, over
the frames a run's calibrations are built from. Each row of `qa_normal_value` also holds the number of frames it
rests on, the collection, the visit it is valid from, the visits it was made from, and who accepted it, when and
why.

- The lamp of an arc is the one its header says was lit (`iqQaMetrics.lamps`), under one spelling per lamp
  (`registry.lampOf`: `Ne` for `Ne`, `Ne_eng` and `Neon`). A row stored without `lamps` takes it from the sequence
  name (`Arc: Neon`, `ImageQuality (Kr)`), which says nothing for a sequence named otherwise. A quartz has one
  lamp, so its values are kept per camera or arm alone. A value stored under another spelling of its lamp
  (`Neon`, or `Halogen` on a quartz) is read under the one above (`registry.lampName`).
- A frame is judged by the latest accepted row at or before its visit, so the previous run's values stay in force
  until the run's own are accepted.
- Normal values belong to a collection. A frame is judged by its own collection's values; in a collection with
  none, the relative checks are `UNKNOWN` and the fixed checks still apply.
- Cameras differ from each other by more than the tolerance, which is why the values are kept per camera.

#### At the start of a run

1. The run's calibration frames are taken, reduced and loaded. They are judged against the values in force, the
   previous run's.
2. `drpQa.py values` makes candidate values from the calibration visits, and prints each beside the value in
   force with its shift:

   ```bash
   python bin.src/drpQa.py values -c qaActor/reductions --visits 152652-152705
   ```

   Each candidate has a status: `NEW` (no value in force), `TOO_FEW` (fewer than 3 exposures), else `PASS`, `WARN`
   or `FAIL` by the limits of the check the value is for. Only the rows that are not `PASS` are listed; `--all`
   lists every row. `--visits` takes several ranges, and `--arms brn` keeps only those arms. An image that fails
   a fixed check is left out of the medians.
3. `--accept [NAME]` stores the values, accepted by `NAME` (default: the user running the command), valid from
   the visit after the last of `--visits` (`--valid-from` names another), and judges again every frame of the
   collection from that visit on. A value shifted beyond its check's FAIL is refused, and the value in force
   stays, unless `--reason` says why it is accepted. A `TOO_FEW` value is left out. The command exits 1 when a
   value was refused for its shift.
4. The rest of the run is judged against the stored values.

A change during a run is the same command on a named calibration set, giving rows valid from that visit. The
command reads and writes qadb only, and needs no LSST stack.

#### After a change of limit

Verdicts are stored, so a change to `CHECKS` does not reach frames already loaded. `drpQa.py judge` judges the
stored numbers of a visit range again, with the limits and normal values as they are, and replaces the rows of
`qa_gate`:

```bash
python bin.src/drpQa.py judge -c qaActor/reductions --visits 152594 155519 [--dry-run]
```

`--dry-run` prints the counts and writes nothing. It reads and writes qadb only.

### Twilights

A twilight is the sky flat behind the screen response, and is judged against the other exposures of its
sequence, minutes apart (`pfs.drp.qa.metrics.twilight`): no calibration enters. `twilightFluxQa` stores each lit
fiber's flux per image. Once the sequence has finished and every visit of it has that table, `drpQa.py load`
measures the sequence, stores the result in `twilight_qa` and judges every exposure of it, those loaded earlier
included. It finds the sequence in the opdb (`--opdb-host`); without an opdb, and until the sequence is complete,
a twilight image is `UNKNOWN`.

- The level of each detector (`twilightLevelOffset`) is judged on every exposure.
- The residual scatter beyond noise (`twilightResidRms`) and the deepest coherent patch (`twilightPatch`) are
  measured on exposures bright enough to be flats: a median b-arm S/N of 15 per pixel or more. A fainter one is
  `tooFaint` and judged on its level alone.
- The n arm is not judged.
- Anything that stays the same through a sequence cancels, whether a pattern on the focal plane or a colour
  change uniform across it.

The three limits are provisional, from 16 sequences of 2025-03 to 2026-09.

### Known faults

`tests/metrics/test_knownFaults.py` holds a table of known faults (a defocused spectrograph, offsets from a
calibration built later, obstructed fibers, twilight level steps, a wrong line list) on stored numbers of
calibration and engineering visits. Each must reach its verdict, and the frames known to be good must not fail.
The rules for adding a check are in [`docs/qa-principles.md`](docs/qa-principles.md).

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

`drpQA.yaml` currently wires up `dmLineResiduals`, `extractionQa`, `extractionQaCombined`, `imageQualityQa` and
`twilightFluxQa`. The
remaining tasks below are implemented and importable
but are not registered in the pipeline; run them by referencing their task class directly.

### Tasks

#### `dmLineResiduals`

Per detector, measures every line of `lines` against two references (`reference`): `detectorMap_calib` (`calib`), the
calibration the visit is reduced against, whose offset is drift; and the visit's own `detectorMap` (`visit`), adjusted
to these lines, whose `RESERVED` lines test how well it predicts lines it was not fitted to, as it must on a science
frame. The lines are those the visit's detectorMap fit marked `DETECTORMAP_USED` or `DETECTORMAP_RESERVED`, less
`BAD` ones, with no other selection or clipping; trace centroids are kept on arcs. When the fit marked none, every
good line is measured, as `UNFITTED`. Trace centroids measure x, lamp and sky lines x and y, measured minus reference.
Each row carries the frame's sequence type as read (`seqType`, the raw header's `W_SEQTYP`; the pfsConfig header
stands in when the raw's lacks the card), the observation type it gives (`obsType`, as `imageQualityQa` classifies
it: `trace` for a quartz, `arc`, `twilight`, ...) and the arms the visit read (`cameraSet`).

Per reference, line species and line set (`statusType`: `BOTH`, `USED`, `RESERVED`, or `UNFITTED`): the median and
robust RMS of the x and y residuals in pixels, the median y residual in km/s (`medianDeltaYKms`), `nLines` and
`linesPerFiber`. A species with fewer than `minLines` lines is not measured. In km/s, each line's y residual is
multiplied by `c × dispersion / wavelength`, with the reference detectorMap's dispersion at that line, before the
median is taken; the value is positive to the red. Trace centroids have none.

The task stores numbers only. Which of them are judged is under [The checks](#the-checks). The weighted RMS and
the softening are drp_stella's, in `iqQaMetrics` and `iqQaSpeciesMetrics`, not recomputed. The measurement is in
the stack-free `pfs.drp.qa.metrics.lineResiduals`. No plot is drawn as the task runs: plot `dmLineResidualData`
afterwards with `pfs.drp.qa.plotting.dmResiduals.plotDetectorResiduals` (the former `dmResiduals` 1D and 2D views,
per detector).

It also runs behind drp_stella's `reduceExposure`, measured by a separate `drpQA.yaml` pass, for
[redoing a run](#redoing-a-run) and `drpQa.py calib validate`.

##### Options

- `minLines`: fewest lines a species needs to be measured. Default `10`.
- `lineFlags`: `ReferenceLineStatus` names that exclude a line. Default `["BAD"]`.
- `maxTraceRowsPerFiber`: most `USED`, and most `RESERVED`, trace rows per fiber stored in `dmLineResidualData`,
  evenly spaced in y; `0` for all. Default `100`. Lamp and sky lines and `UNFITTED` rows are always stored; the
  statistics use every line, so a trace statistic recomputed from the table is an estimate.

##### Outputs

| DataSet Type            | Dimensions                             | Description                                                                                                       |
|-------------------------|----------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `dmLineResidualMetrics` | `instrument, visit, arm, spectrograph` | Long format: one row per reference, species, `statusType` and metric, with `cameraSet`, `obsType` and `seqType`.  |
| `dmLineResidualSummary` | `instrument, visit, arm, spectrograph` | One row: `cameraSet`, `obsType`, `seqType`, `fitted`, `nSpecies`, `nSpeciesMeasured`, and the calibration used.   |
| `dmLineResidualData`    | `instrument, visit, arm, spectrograph` | One row per line measured (`USED` and `RESERVED` traces sampled), with `dxCalib`, `dyCalib`, `dxVisit`, `dyVisit` in pixels and `dyKmsCalib`, `dyKmsVisit` in km/s. |

##### Coming from `dmResiduals`

`dmResiduals` and `dmCombinedResiduals` were retired on 2026-10-22. Where their outputs went:

| Before                                            | Now                                                                                                      |
|---------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `dmQaResidualStats`, a row per `status_type` and species | `dmLineResidualMetrics`, long format; `lineResiduals.wideResiduals` gives a column per metric. `reference` `visit` is what `dmResiduals` measured against. |
| `spatial.median`, `spatial.robustRms`             | `medianDeltaX`, `robustRmsDeltaX`                                                                        |
| `wavelength.median`, `wavelength.robustRms`       | `medianDeltaY`, `robustRmsDeltaY`                                                                        |
| `weightedRms`, `softenFit`                        | not recomputed: drp_stella's own, in `iqQaSpeciesMetrics`                                                |
| `num_lines`, per fiber                            | `nLines` (lines of the species) and `linesPerFiber`                                                      |
| `dmQaResidualData`                                | `dmLineResidualData`, with a residual against each reference                                             |
| `dmQaResidualPlot`                                | nothing per quantum: `plotting.dmResiduals.plotDetectorResiduals` draws it from `dmLineResidualData` after the fact |
| `dmQaDetectorStats`, `dmQaCombinedResidualPlot`   | none; `pfs.drp.qa.metrics.readers.exportMetrics` writes a collection's tables to parquet                 |

#### `imageQualityQa`

Measures image quality (FWHM), spatial flexure, and arc-line flag rates on a per-detector quantum
`(instrument, visit, arm, spectrograph)`. The task classifies each visit by reading FITS headers (`W_SEQTYP`,
`W_SEQNAM`) and routes it to the appropriate measurement path. When `reduceExposure` logs are available it also folds
ISR, cosmic-ray, and `fitDetectorMap` statistics into the metrics table.

The task writes numbers only: no plot and no verdict. Which of them are judged is under
[The checks](#the-checks).

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
| IIS arc (`scienceArc`, 16 engineering fibers)      | Arc-line second moments on the engineering fibers                      | `calexp` trace widths of those fibers if `nGoodLines < minGoodLines`   |
| Regular trace/quartz (`scienceTrace`, all fibers) | `calexp` trace widths                                                  | Fiber profile calibration widths if the `calexp` is absent or unusable |
| IIS trace/quartz (`scienceTrace`, 16 eng. fibers) | `calexp` trace widths of the engineering fibers                        | Fiber profile calibration widths if the `calexp` is absent or unusable |
| Science / all-sky (`scienceObject*`)              | `calexp` trace widths of `FLUXSTD+GOOD` fibers                         | Sparse if FLUXSTD good fraction < `minFluxstdGoodFrac`                 |
| Unknown (`W_SEQTYP` absent)                       | Heuristic: arc lines if sufficient, else `calexp`, else fiber profiles | —                                                                      |

For arc visits, FWHM is computed as the Gaussian-equivalent width `2√(2 ln 2) × σ` from the second moments of each
arc-line profile. On the calexp, every `profileYStride`-th row is fitted with `pfs.drp.qa.crossDispersion`: each fiber
jointly with its neighbours, as pixel-integrated Gaussians sharing one width, plus a background, with no aperture
(fibers are 6.17 px apart). The `fiberProfiles` fallback reads each profile's width at half its maximum. Every FWHM
includes the pixel response, as the arc-line moments do; `fwhmIntrinsic` in the per-line data leaves it out.

The PFS trace profile is not Gaussian, so these readings were chosen by comparing them on real quartz
([`docs/trace-widths.ipynb`](docs/trace-widths.ipynb)): the joint fit recovers the profile's half-maximum
width, a Gaussian fitted to the profile understates it by 8–10 %, and drp_stella's second moment
(`calculateStatistics().width`) varies between detectors whose profiles agree.

##### What the Metrics Tell Us

- **`medFwhm`** — Median FWHM in pixels across all good measurements for the detector quantum. Reflects the
  cross-dispersion width of the fiber PSF. Typical values (Run25) are 2.5–3.2 px on arcs and 3.1–3.4 px on quartz;
  SM1 defocused in Run27 read 4.1–4.5 px on quartz. Judged on arcs, quartz and engineering-fiber frames (`width`).
- **`medDxCenter`** / **`dxCenterRms`** — Median and scatter of the spatial offset of the measured fiber positions
  from the static calibration detectorMap (measured minus calibration, as drp_stella's residuals), in pixels. This is a flexure diagnostic: a large `|medDxCenter|` means
  the whole detector has shifted relative to the calibration, while a large `dxCenterRms` at small median suggests a
  distortion rather than a bulk shift. `NaN` when no `detectorMap_calib` is available. `medDxCenter` is judged on
  engineering-fiber frames (`traceCentreOffset`); on other frames `dmLineResiduals`' offsets are.
- **`pctFlagged`** — Percentage of arc lines flagged by `fitDetectorMap` (below S/N threshold or otherwise rejected).
  High flag rates on b-arm Ar/Xe/Kr arcs are expected (lamp physics — those species have very few bright lines in the
  blue) and do not indicate an optical problem.
  `pctFlagged` is set to `NaN` for sparse visits and science/FLUXSTD paths where the flag rate reflects
  exposure depth rather than optical quality. Not judged.
- **Flag breakdown** — When `pctFlagged` is finite, the same rejections are broken out by
  `ReferenceLineStatus` bit: `pctNotVisible`, `pctBlend`, `pctSuspect`, `pctRejected`,
  `pctBroad`. On the arc-line path two measurement-level categories are added:
  `pctLowSN` (flagged with no flux measured — the fit never got that far) and `pctMeasFail`
  (flagged despite a finite flux — a centroid or photometry failure). The split matters:
  `pctLowSN` is usually lamp physics, `pctMeasFail` usually is not.
- **`nLines`** — Rows of the `lines` dataset, whichever path measures the FWHM: a row for every line of the
  lamp's list on every `GOOD` fiber, flagged when not measured, and the trace centroids `fitDetectorMap` found,
  on an arc as on a quartz. Not judged: it follows how many fibers the design lights, and on an arc the trace
  centroids, whose number follows the illumination and the pipeline rather than the lamp.
- **Saturation and line flux** — Measured, not judged. `pctSaturatedPixels` is the percentage of `SAT` pixels
  among those the lit fibers' traces cover (7 columns about each trace centre, every row); `nSaturatedLines` counts
  the lamp or sky lines on lit fibers with a `SAT` pixel within one pixel of where the `detectorMap` puts them
  (not the measured centroid: drp_stella's line finder skips `SAT` pixels). It counts rows of `lines`, one per
  line-list entry and fiber, so two entries that fall on the same saturated pixels (Kr 810.625 and 810.659 nm) count
  as two. The reference is the detector's own
  saturation level, which ISR marks in the calexp's `SAT` plane; both are null without a calexp.
  Per species, in `iqQaSpeciesMetrics`: `nSaturatedLines` again, and `medLineFlux` and `robustRmsLineFlux` over
  the species' good lines (unflagged, no bad status bit, finite flux; `nFluxLines` of them). The flux is
  `lines`' own, in the calexp's units and not per second. Within one image a species' lines span orders of
  magnitude, so these say nothing about the lamp alone: they are the inputs to a comparison across visits of the
  same exposure time (`pfs.drp.qa.metrics.lineFlux`, `saturation`; both stack-free).
- **`linesPerFiber`** — The count that compares between frames: unflagged lamp lines per lit fiber on an arc,
  unflagged trace centroids per lit fiber on a quartz (about 4,160, one per detector row); `NaN` on other frames or
  without a pfsConfig.
  A design that lights one fiber group (a *shaka* exposure: every fourth cobra, the others parked behind their black
  dots; see `docs/qa-domain-notes.md`) reads like a full slit;
  an obstructed frame, lit by its design but dim, reads low. On Run30's daily neon, a one-group arc measures
  the same lines on each lit fiber as a fully lit one. Judged on arcs and quartz, those lit through the
  engineering fibers included (`linesPerFiberArc`, `linesPerFiberQuartz`).
- **`nLitFibers`**, **`nFibers`** — The detector's fibers the pfsConfig lights (`fiberStatus` `GOOD`, engineering
  fibers excepted: the only ones `lines` has rows for), and those it lights or hides behind a black dot
  (`BLACKSPOT`). `pctFlagged` on the arc-line path is taken over the lamp lines of the lit fibers, and the calexp
  trace width over the lit fibers. **`nTraceRows`** counts the trace centroids in `lines`.
- **`traceOnly`** — `True` when FWHM comes from fiber profile calibration widths rather than live measurements; these
  values reflect the calibration epoch, not the current visit, so `medFwhm` then counts as not measured: it is
  not judged and enters no normal value. `pctFlagged` is `NaN`. A quartz falls back to them whenever its calexp
  gives no usable measurement.
- **`obsType`** / **`seqType`** / **`seqName`** / **`illumination`** / **`lamps`** — Which measurement was made
  (`arc`, `trace`, `science`, `allsky`, `twilight`, `dark`, `bias`, or `unknown` for a frame with no measurement of
  its own), the raw `W_SEQTYP` and `W_SEQNAM` strings (e.g. `"scienceArc"`, `"Arc: HgCd"`) it was derived from,
  `full` (the science fibers) or `iis` (the 16 engineering fibers), and the lamps the lamp headers say were lit
  (obs_pfs's names joined by `+`: `Ne`, `Kr_eng`; empty when none). Twilight, dark and bias frames are not
  measured. The gate works out the frame type from `seqType`, `seqName` and `illumination`, and the lamp of a
  normal value from `lamps`, so a change of classification is applied to stored rows without the tasks.

##### Options

- `imageQualityQa:minGoodLines`: Minimum good arc-line measurements to trust the arc path. Default `10`.
- `imageQualityQa:minPeakSN`: Minimum significance of a calexp trace (fitted flux over its error). Default `5.0`.
- `imageQualityQa:maxCalexpFlagRate`: Max fraction of bad calexp samples before rejecting the calexp path. Default
  `0.5`.
- `imageQualityQa:minFluxstdGoodFrac`: Min fraction of good FLUXSTD samples for the stellar calexp path. Default `0.10`.
- `imageQualityQa:profileHalfWidth`: Deprecated and ignored; the calexp width uses no aperture.
- `imageQualityQa:profileYStride`: Row sampling interval (pixels) for calexp trace widths. Default `100` (about 40
  rows); the fit's cost is proportional to the rows.
- `imageQualityQa:doWriteLineData`: Write `iqQaData`. Default `False`.

##### Outputs

| DataSet Type  | Dimensions                             | Description                                                                                                                                                                       |
|---------------|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `iqQaData`    | `instrument, visit, arm, spectrograph` | Only with `doWriteLineData` (default off). Per-line or per-trace-sample measurements: `fiberId`, `x`, `y`, `lam`, `fwhm`, `fwhmIntrinsic` (calexp only), `theta`, `dxCenter`, `flux`, `fluxErr`, `flag`, `traceOnly`, `peakRatio`, `snr` (calexp only), `status`, plus `visit`/`arm`/`spectrograph`. |
| `iqQaMetrics` | `instrument, visit, arm, spectrograph` | Single-row summary; see below.                                                                                                                                                    |
| `iqQaSpeciesMetrics` | `instrument, visit, arm, spectrograph` | Per-species `fitDetectorMap` statistics from the `reduceExposure` log, one row per species and metric: `visit`, `arm`, `spectrograph`, `description` (line species, e.g. `NeI`, or `Trace`), `metric` (`fitXRms`, `fitYRms`: weighted RMS; `fitXSoften`, `fitYSoften`: softening), `value`. Also, measured from `lines` on lit fibers, each species' `medLineFlux`, `robustRmsLineFlux`, `nFluxLines` and (with a calexp) `nSaturatedLines`; see *Saturation and line flux*. |

`iqQaMetrics` columns, by group:

- **Identity** — `visit`, `arm`, `spectrograph`, `obsType`, `illumination`, `seqName`, `seqType`, `lamps`,
  `calexpRun` (the RUN
  collection of the calexp measured, whose `pfsArm` records the reducing pipeline; null without a calexp)
- **Image quality** — `medFwhm`, `medDxCenter`, `dxCenterRms`, `pctFlagged`, `nLines`, `linesPerFiber`,
  `nLitFibers`, `nFibers`, `nTraceRows`, `traceOnly`
- **Saturation** (null without a calexp) — `nSaturatedLines`, `pctSaturatedPixels`
- **Flag breakdown** (only when `pctFlagged` is finite) — `pctNotVisible`, `pctBlend`,
  `pctSuspect`, `pctRejected`, `pctBroad`, and on the arc-line path `pctLowSN`, `pctMeasFail`
- **Log-derived** (zero/`NaN` when the corresponding log is absent) — `isrBadPixels`,
  `isrTime`, `cosmicRayCount`, `cosmicRayPixels`, `cosmicRayTime`, `reduceExposureTime`
- **`fitDetectorMap` statistics** — `fitChi2`, `fitDof`, `fitXRms`, `fitYRms`, `fitXSoften`,
  `fitYSoften`, `fitNLines`, `fitTotalLines`, the matching `fitReserved*` columns for the reserved-line sample,
  `fitTraceXRms`, `fitTraceYRms`; the per-species values are in `iqQaSpeciesMetrics`
- **Per-fiber arrays** — `fiberIds`, `fiberXRms`, `fiberYRms`, `fiberNLines`

There is no verdict column and no `iqQaPlot` output.

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

#### `twilightFluxQa`

Per detector of a twilight (a `scienceObject` whose `W_SEQNAM` starts "Twilight"), stores `twilightFiberFlux`:
each lit fiber's median flux / norm over the arm's window, its S/N, good and saturated pixels, and focal-plane
position, with the sequence id (`W_SEQID`). It judges nothing: a twilight is judged against the other
exposures of its sequence, which one quantum cannot see. `drpQa.py load` measures and judges the sequence once it
is complete (see [Twilights](#twilights)). Any other visit is skipped from the raw header (the
pfsConfig's standing in) before its pfsArm is read, so running it on every visit costs the header and the pfsConfig.
Raws without those cards (before 2025-11) need `assumeTwilight` and a data query selecting the twilights.

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
`boresight` reference, and the plots using it, are only right for its shutter-open rows (`shutter_open == 1`).

### End-of-run report

`drpQa.py agReport` reports how the AG cameras and the guide loop did over an observing run, so
problems are caught once the run is over. The runs are the same ones `redo` uses
([`observingRuns.yaml`](python/pfs/drp/qa/comparison/data/observingRuns.yaml)).

```bash
python bin.src/drpQa.py agReport fetch --period run30   # the opdb, once: ~10 MB, a few seconds
python bin.src/drpQa.py agReport report --period run30 --title "Run 30" [--notes notes.md]
```

`fetch` reads seven aggregated tables (`queries.readAgRun`, the matches on a 4×4 grid over each detector and the
designed guide stars matched by magnitude among them) into `~/.cache/drp_qa/agRun/opdb/`; `report` writes
`<period>/report.md`, its figures and `metrics.json` from them, without the database. The report opens with its
key numbers (`runAnalysis.runSummary`: the size and the frame-to-frame jitter of each science visit's guide error in ΔRA,
ΔDec and ΔInR and on the sky, the distance of the matched stars from where the guider expected them, the share
of the designed guide stars matched per science visit, and the stars matched per AG exposure and per camera) and its flags, then a map of the camera states, AG1/3/5 against AG2/4/6, the
guiding of each science visit, focus and acquisition, the six detectors on the PFI (where on each matches are
rejected and the residual left there, each camera's focus against the others, and its FWHM per detector half), and
a section per camera: its flags, and its matched stars, designed guide stars matched by magnitude, the share of AG
exposures in which it matched a star, clusters of rejected matches, the three parts of its position-model error
(the slope of its residuals along detector x and along y, and the step between its detector's halves) and the share
of its detections ics_agccActor flagged (EDGE, SATURATED, BAD_SHAPE, BAD_ELLIP, BAD_SIZE), night by night. Every figure
has a caption saying what it shows (`runPlotting.captions`), which `pfs_qa_dashboard`'s AG page shows too.

A visit's guide error is measured from the guide offsets the guider accepted in AG exposures taken while the
spectrograph shutters were open (`runAnalysis.visitGuideErrors`, from `sps_exposure`'s times, at least 5 of them):
what the spectrograph saw. AG exposures go on after the shutters close, and the pointing may then be moved: in Run
31's 60 s windowed visits, six AG exposures are taken with the shutters open (median offset 0.11″) and the first
after they close reads about 1″. Nothing settles within a science visit (its first AG exposure reads as its
twelfth), so no frames are skipped at its start.

A good match is `agc_match.flags = GOOD_MATCH` on a clean detection (`agc_data.flags` NONE or RIGHT) of a camera
the guider had not left out of its fit. A camera's state in an AG exposure is one of matched, detections but no
match, no detections, or disabled (`agc_match.camera_enabled` false). The opdb records a camera's detections, not
whether it returned an image, so a dead camera and an empty image both read as no detections; a camera with none
in the whole run is **not in use**, named once and left out of every comparison between cameras (AG2 in Run 31).

Cameras are shown and compared by group, AG1/3/5 and AG2/4/6: they are wired and placed separately, and can fail
together (Run 30 lost AG1/3/5 after 2026-09-08). The guider treats all six alike.

What it flags (`runAnalysis.RunConfig` holds the numbers):

- **A camera that matched no star** in 30 or more consecutive AG exposures, among those in which some camera
  matched one and its design had 3 or more guide stars brighter than mag 20 there (fewer may well go
  undetected), and whether it detected anything at all.
- Guide frames rejected (INVALID_OFFSET), frames with no guide offset, and **offsets accepted from fewer than 3
  matched stars**, which the guider sends to the telescope while they are under `max_correction`.
- **Rejected matches clustered on one side of a detector**: detections paired with guide stars but not counted as
  good matches, whose mean position is far from the good ones', on that camera alone in its AG exposure. When a
  camera's position model is off, the matches far from where it fits best go first, always on the same side: AG3's
  top half in Run 29 visit 147728 (INSTRM-2982), a corner of AG4 in the same visit. Flagged at 2.5× the other
  cameras' median, with the clusters' mean direction and how consistent it is (Runs 29 and 30: AG3 toward ~115°,
  consistency 0.9; AG1 in Run 30).
- **A camera position model that is off**: a 2×2 fit, per night, of the good matches' residuals (measured minus
  expected position, less the camera's mean in each AG exposure, so the pointing drops out) against their position
  on the detector; quoted as the error 500 px from the centre and the direction it grows in. The residual is the guide fit's own: the detection through pfs_utils' `coordinates.det2dp` against the guide star through `Subaru_POPT2_PFS`, not the `CoordTransp` (`sky_pfi_ag`, `ag_pfimm_to_pixel`) chain that places guide stars in pfsDesign and pfsConfig. Flagged over 4 µm and
  1.5× the cameras' median (Runs 29 and 30: AG3 8 µm toward ~113°, the others 3–6 µm).
- Science visits whose guide jitter (frame-to-frame, with the shutters open) is over 3× the run's median, and
  cameras whose FWHM is 15% from the others'. The two detector halves focus differently by design, so their FWHM
  difference is plotted, not flagged.

The report holds science visit IDs and guiding statistics: keep it within the team.

## Redoing a run

`drpQa.py redo <run>` redoes an observing run (or its pre-run calibration period) with an updated pipeline, end
to end (`pfs.drp.qa.comparison`): it reads the run's visits from the opdb, reduces the raw frames, measures
them, loads the numbers into qadb, makes the run's normal values and compares them with the previous pipeline's,
judges every frame, and points at the dashboard for the run report. It runs on the processing host, where the
opdb, the Butler and qadb are all reachable, and writes no parquet and no report of its own.

- **The visits come from the opdb**, not from the hand-written run summaries: every SpS visit between the
  period's dates, with its sequence, `cmd_str`, cameras, design, obslog notes and telescope status. The periods
  (each run's nights and its pre-run calibration nights, HST, a night running from 14:00 to 14:00, which is 00:00 UTC, when the raw files' date changes) are in
  [`comparison/data/observingRuns.yaml`](python/pfs/drp/qa/comparison/data/observingRuns.yaml), derived from
  gaps in SpS activity.
- **Calibration or science is what the exposure is**, not when it was taken: every sequence type but
  `scienceObject` and `scienceObject_windowed` is a calibration, and those are science when their design has a
  science proposal. Twilight frames, dithers and telescope focus sweeps are on engineering designs, so they are
  calibrations.
- **Everything measurable is measured; the calibration recipe comes first.** `scienceArc` and `scienceTrace`
  visits are *gated*: their passes are planned and run first. Every other type the tasks can measure (science
  frames, engineering arcs, flats, fiber profiles, focus sweeps, windowed readouts, and the engineering
  sequences taken as an arc or trace: bootstraps, IIS `ImageQuality` frames, health checks, line checks and
  detectorMap tests, by sequence name) is *unvalidated*, in its own pass: a result out of the ordinary there
  may be what an engineering test expected. Biases and darks are not measured; test exposures are listed only.
- **Daily arcs and traces are told from sets.** A calibration *set* is 3 exposures per arc lamp and 10
  traces; one-visit sequences of its sequence group (`group_id`) belong to it. Another one-visit sequence is *daily* when
  its name recurs on 4 or more nights (from Run28, a neon arc and a trace, and IIS frames, each day, for drift), and a
  one-off *single* otherwise (positioner scans, bootstrap exposures).
- **Sky visits are labelled** from the telescope status: a *focus sweep* (10 or more focus offsets spanning at least
  0.5 mm on one field in one night), and *dithered* visits.

```bash
# Redo run31 with the current stack: reduce afresh, measure, load, make and compare normal values, judge
python bin.src/drpQa.py redo run31 --opdb-host pfsa-db --accept wtg
# Reuse an already-reduced collection (measure, load and judge only)
python bin.src/drpQa.py redo run31 --reductions drpActor/reductions --accept wtg
# Resume after the long reduce/measure: just load, make values and judge
python bin.src/drpQa.py redo run31 --only load,values,judge --accept wtg
```

The stages are `process` (reduce then measure), `load`, `values` and `judge`; `--only` runs a subset to resume
after an interruption. `-j` sets the `pipetask` processes (default 8); `--cadence set` (or `daily`, `single`;
repeatable) processes only the arcs and traces of those cadences; `--arm b r m` and `--spectrograph 1 2`
process only some detectors (e.g. `--arm b r m` skips the slow `n` arm); `--dry-run` builds each pass's graph
(`pipetask qgraph`) and writes nothing.

By default redo reduces every frame afresh with the current pipeline, into
`u/$USER/redo/<run>/<version>/<pipeline>`: the pipeline from drp_stella's `getPfsVersions` (each product's
version if they differ; `--pipeline-version` names it), so a different stack starts a fresh collection, and the
version is `git describe` of the last commit that could change what is written (not the driver, tests or docs),
so a change to the driver keeps the collection and uncommitted changes to the rest are refused. `--reductions
<collection>` reuses that collection's reductions instead, measuring only, into `u/$USER/redo/<run>/<version>`;
reductions an earlier drp_qa version of the same pipeline made are reused too. Since the pipeline label comes
from the stack set up in the shell, a fresh redo stops when its label has no reductions of the run yet while
another does (the whole reduction again, where a measure pass was meant); `--new-pipeline` reduces anyway, and
`--dry-run` only warns.

The reduction is drp_stella's `reduceExposure` (less `mergeArms`) with drpActor's per-sequence-type
configuration and cosmic-ray grouping; the measurement is `imageQualityQa`, `dmLineResiduals` and
`reductionStats`, a separate `pipetask` command that reads the reductions from the output chain. Passes are
`calibration`, `sky` and `unvalidated-calibration`, gated first, each split by sequence: a sequence with a frame
to reduce is reduced whole (so `cosmicray` combines it) and then measured, and one already reduced is measured
alone (a `-judge` pass). A failed quantum fails its pass but not the others; redo lists the failures at the end
and exits 1. Images whose quanta failed before (read from the run's `pipetask` logs) are not rescheduled, since
they fail the same way each time; `--retry-failed` retries them. Visits with no `pfsConfig` can't be reduced and
are reported, not scheduled.

Once the frames are loaded, redo makes the run's normal values from its calibration set and prints each beside
the value in force in the previous pipeline's collection (`--previous-collection`, default `qaActor/reductions`),
with its shift; `--accept NAME` stores them for the redo collection and judges every frame against them (a shift
beyond FAIL needs `--reason`; without `--accept` nothing is stored and the relative checks stay `UNKNOWN`).
`--calib-visits FIRST LAST` names the calibration frames; the default is the run's set-cadence arcs and traces.
The run report is the dashboard's: redo prints the `qaDashboard.py report --collection <redo collection> …`
command (`pfs_qa_dashboard` reads qadb and renders a stand-alone page; see [Loading QA results into
qadb](#loading-qa-results-into-qadb) and the dashboard's own README).

### Calibration inputs

`drpQa.py calib inputs` judges a run's calibration set exposure by exposure and detector by detector, for
whoever builds `detectorMap_calib` and the fiber profiles from it: which visits to use per detector,
which to leave out and why. It reads the set's arcs and traces (cadence `set`, or `--visits`) from the period's
visit listing (the opdb, `--opdb-host`) and what was measured on them (`iqQaMetrics`, `dmLineResidualMetrics`)
from a Butler collection (a redo's, `--collection`), and writes a Markdown report with its tables beside it as
CSV. It is blind: no calibration built from the set is used.

```bash
python bin.src/drpQa.py redo run31 --cadence set                                # measure the set
python bin.src/drpQa.py calib inputs --period run31 --opdb-host pfsa-db \
    --collection u/wtg/redo/run31/<version> -o calibInputs-run31.md
```

Each detector image is judged on its own detector against the other exposures of its sequence, with no outside
reference (`pfs.drp.qa.comparison.calibInputs`, stack-free):

- **out**: its reserved-line RMS is more than 2x the detector's median
  over the sequence; or its FWHM is more than 1.25x (either way) from that median *and* the RMS is also off (1.5x). A
  detector gone wrong mid-sequence shows in both (Run31: n4 from 152673); FWHM alone can be the measurement on a faint
  lamp's few lines, so it only says *check*.
- **check**: FWHM or RMS beyond 1.1x / 1.5x the detector's sequence median. Look before deciding.
- **weak**: a species with fewer than 0.8 of the lines of the detector's best exposure of the sequence: usable, with
  fewer lines. Kr and Xe fade through a set and Cd and Hg warm up; the *Lamps through their sequences*
  section shows it per arm. Line counts don't see the Cd warm-up (the same lines are matched); its flux does.

The gate judges each image too, on its width, lines per lit fiber and held-back line scatter, against the normal
values in force at the visit (`--normal-values FILE`, parquet or CSV with the columns of `qa_normal_value`): a
`FAIL` is *out* and a `WARN` is *check*. Every one of those checks needs a normal value, so without the table the
gate gives no verdict, the report says so, and the go/no-go rests on the sequence rules alone. A detector wrong
through a whole sequence agrees with itself, so only the gate catches it. The offset from the calibration is not
judged here: the set replaces that calibration.

The report is a go/no-go: the no-go detector images in one line, species a detector has no go exposure of, visits
with no data, the go/no-go grid (detectors down, the set's visits across, sequences named), and a per-detector table
of go and no-go visits; the same grid is written as CSV for the build. *Out* is no-go; every other
measured image is go. `--detail` adds the reasons for every no-go and check, the weak exposures, the lamp trends and
three more figures (every verdict, each detector's FWHM and reserved RMS through the set, the lamps through their
sequences); the per-image tables are always beside the report as CSV. The factors are `Thresholds` and the
`--fwhm-*`, `--rms-*` and `--weak-lines` options. With `--normal-values` the figures show the gate's limits, and
`<stem>-gate.csv` holds every number the gate judged. Judging per arm would hide a single bad detector (the medians over
spectrographs looked normal on Run31's n arm), so nothing aggregates over detectors except the lamp trends.

### Validating a built calibration

`drpQa.py calib validate` is the verdict after the build: it compares candidate calibrations on the same
visits. `run` reduces the period's set-cadence and daily arcs and traces (or `--visits`) once per candidate —
`reduceExposure` with drpActor's arc configuration, then `dmLineResiduals` — the candidate's collection first in
the inputs so its `detectorMap_calib` is the one used (an empty collection means the inputs as they are, e.g. the
certified calibration in `PFS/defaults`), into `<prefix>/<label>`. `report` lays the candidates'
`dmLineResidualMetrics` side by side per detector: the offsets and RMS of every line against the calibration (the
drift it leaves) and the reserved-line RMS against the visit's own map, medians over the set's images and over
the nights after it, with a figure per detector (`pfs.drp.qa.comparison.calibValidate`,
`plotting.calibInputs.plotCalibrationComparison`).

```bash
python bin.src/drpQa.py calib validate run --period run31 --opdb-host pfsa-db --detectors b1 r2 \
    --candidate asis= --candidate pick=u/wtg/pipe2d-1941/detectorMaps-pick --prefix u/wtg/pipe2d-1941/validate
python bin.src/drpQa.py calib validate report ... -o validate-run31.md
```

The set's own images read near zero against a calibration built from them, so the later nights are the fair
comparison. A candidate built from the go/no-go pick is made with the certified build's own task configurations,
stack and inputs (read from its collection) and only the visit list changed.

## Loading QA results into qadb

`drpQa.py load` copies what `imageQualityQa`, `dmLineResiduals` and `twilightFluxQa` wrote to the Butler into
qadb (`spt_qa_database`), after the fact, and judges it (`pfs.drp.qa.qadb.loader`). The Butler is the record of
what was measured, and a database outage cannot fail a quantum. One code path serves qaActor after each visit's
pipetask (`loadVisit`) and the backfill of whole collections (`loadCollection`).

- **Numbers.** `iqQaMetrics` → `image_quality_qa`, `iqQaSpeciesMetrics` → `image_quality_qa_species`,
  `dmLineResidualSummary` → `dm_line_residual_qa`, `dmLineResidualMetrics` → `dm_line_residual_qa_metric`: the same
  columns in snake_case (`pfs.drp.qa.qadb.tables`), keyed by the processing, the visit and the camera (`b1`). A
  Butler column reaches qadb once both the schema and the table's spec name it; the fiber arrays and the
  per-flag percentages of `iqQaMetrics` stay in the Butler. No numbers table holds a verdict.
- **Twilights.** `twilight_qa` holds the twilight metrics per image. The loader measures them from
  `twilightFiberFlux` once the sequence is complete; see [Twilights](#twilights).
- **Processing.** Each Butler RUN found is a `data_processing_pipe2d` row (collection and RUN name), with the drp_qa
  that wrote it (the RUN's `packages` dataset) and the pipeline that reduced its images (`datamodel`,
  `obs_pfs` and `drp_stella`, from their `pfsArm` headers; null when the RUN holds reductions of more than one
  pipeline, or none is recorded). `--no-pipeline-versions` skips the header reads.
- **Verdicts.** `qa_gate` holds one row per image: its frame type (`frame_type`), the verdict (`qa_status`), the
  check that decided it (`qa_decided_by`), the reasons of every `WARN` and `FAIL` (`qa_reason`; for `UNKNOWN`,
  why nothing was judged), the mode (`realtime`, `comparison`, `backfill`) and when it was decided. The gate
  judges with the collection's normal values in `qa_normal_value`.
- **Idempotent.** Every write is `INSERT ... ON CONFLICT DO UPDATE` on the row's key, and a long table's rows of each
  image are deleted before its reload: load a collection again after a load that stopped, or when more of a
  sequence has been reduced, and the rows are replaced, not duplicated. Visits qadb's `pfs_visit`
  lacks get a row, filled from the opdb when one is given.

```bash
# What would be written, as parquet, touching no database
python bin.src/drpQa.py load -c u/wtg/comparison/run30/<version> --visit-range 148302 148348 --dry-run ~/reports/dry
# Load and judge (qadb user that may write; credentials in ~/.pgpass), with the opdb's visit descriptions
python bin.src/drpQa.py load -c qaActor/reductions --visits 150001 150002 --opdb-host pfsa-db
```

`load` needs the stack, since it reads the Butler. With `--dry-run` and no database, the relative checks give no
verdict.

The tests (`tests/qadb/`) run the writer's statements on SQLite through SQLAlchemy, whose `ON CONFLICT` and
`RETURNING` match PostgreSQL's; set `PFS_QADB_TEST_DSN` to a scratch PostgreSQL to run them there too. They
empty its QA tables, so they refuse a database whose name does not end in `_test` or `_tests`: run them on a copy
(`CREATE DATABASE qadb_tests TEMPLATE qadb`). The
schema the tests create (`tests/qadb/conftest.py`) is the one `spt_qa_database`'s migration follows: change both
together.

## Reading qadb back

`pfs.drp.qa.qadb.reader` reads what the loader wrote, read-only, with one query per table and no Butler:

- `readQadb(qadb, collection, first, last, since=None)` returns a collection's numbers as the tables they came
  from (`iqQaMetrics`, `dmLineResidualSummary`, `dmLineResidualMetrics`, `twilightMetrics`) and qadb's visit
  listing. An image measured in more than one RUN of the collection is read, table by table, as the newest RUN
  that wrote that table; with `since`, only images whose verdict was decided at or after it are read.
- `readNormalValues(qadb, collection)` returns the collection's normal values.
- `readGate(qadb, collection, first, last)` returns each image's verdict.

`drpQa.py values` and `drpQa.py judge` are built on it, and so is the QA dashboard (`pfs_qa_dashboard`), with the
opdb's IIC sequences, times and notes from `comparison.queries.readVisitTimes` and `readNotes`.

### Verdicts per visit

`metrics.visitRollup.rollupVisits(gate, expected=None)` rolls `readGate`'s rows up to one row per
visit: the worst of its images' verdicts, the count of each, and for a `WARN` or `FAIL` the camera, check and
reason of the first image with that verdict (by spectrograph, then arm). An `UNKNOWN` image is
counted and does not lower a visit whose other images passed. Given the cameras that exposed
(`comparison.queries.readVisitCameras`, from the opdb), the rollup names those without a verdict (`missingCameras`);
they are reported, not judged. `drpQa.py summary` prints it:

```bash
python bin.src/drpQa.py summary -c qaActor/reductions --visit-range 153000 153534 --opdb-host pfsa-db --json summary.json
```

Its exit status is the worst visit verdict, for a nightly job: 0 `PASS`, 1 `WARN`, 2 `FAIL`, 3 when no visit was
judged. A summary that could not be made is none of those: 4 for a collection qadb holds no processing of (a mistyped
name is an error, not missing data) or a database that could not be read, 64 for bad arguments. Visits of the range
with SpS exposures and no verdict at all are counted on their own line (and listed in the JSON), not in the status:
not every exposure is one the QA tasks judge.

## Command-line tools

Scripts live in `bin.src/` and are run directly; there is no SCons step to copy them into a `bin/` directory on
`PATH`. One command, `drpQa.py`, has a subcommand for each job (`pfs.drp.qa.cli`). Each subcommand imports only
what it needs, so those that read only qadb run without the LSST stack.

```bash
python bin.src/drpQa.py --help                 # the subcommands
python bin.src/drpQa.py <subcommand> --help    # a subcommand's options
```

| Subcommand | Does | Needs the stack | See |
|---|---|---|---|
| `redo` | Redoes a run with an updated pipeline end to end: reduce, measure, load, make and compare normal values, judge, point at the dashboard | yes | [Redoing a run](#redoing-a-run) |
| `load` | Copies a collection's numbers from the Butler to qadb and judges them; measures and judges a complete twilight sequence | yes | [Loading QA results into qadb](#loading-qa-results-into-qadb) |
| `values` | Makes, compares and accepts a run's normal values, then judges the frames they cover | no | [At the start of a run](#at-the-start-of-a-run) |
| `judge` | Judges stored numbers again for a visit range of a collection | no | [After a change of limit](#after-a-change-of-limit) |
| `summary` | Prints the verdict of each visit of a collection, and exits with the worst | no | [Verdicts per visit](#verdicts-per-visit) |
| `calib inputs` | Says which exposures of a calibration set to build from, per detector | yes (the Butler) | [Calibration inputs](#calibration-inputs) |
| `calib validate` | Compares candidate calibrations on the same visits | yes | [Validating a built calibration](#validating-a-built-calibration) |
| `fiberNorms` | Plots fiber normalizations from a Butler collection (`pfs.drp.qa.fiberNormsQa.main`) | yes | |
| `agReport` | The end-of-run guider report | no | [End-of-run report](#end-of-run-report) |

The qadb subcommands (`load`, `values`, `judge`, `summary`) take `--host` (default `pfsa-db`), `--port` (default
`5436`), `--user` (default `pfs`) and `--dbname` (default `qadb`); credentials are read from `~/.pgpass`. `redo`
takes `--qadb-host`/`--qadb-port`/`--qadb-user` for qadb and `--opdb-host`/`--opdb-port` for the opdb.

The QA tasks themselves run under `pipetask`, from `pipelines/drpQA.yaml`.
