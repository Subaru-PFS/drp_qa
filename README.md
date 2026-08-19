DRP QA
======

## Introduction

This repository contains the pipeline and corresponding tasks for the quality assurance (QA) of the Prime Focus
Spectrograph (PFS) Data Release Production (DRP)
pipeline. The QA tasks are implementations of the `PipelineTask` class in the LSST Science Pipelines. The tasks are run
on the output of the DRP pipeline to assess the quality of the data products.

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

`drpQA.yaml` currently wires up `dmResiduals`, `dmCombinedResiduals`,
`extractionQa`, `extractionQaCombined`, and `imageQualityQa`. The remaining tasks below are implemented and importable
but are not registered in the pipeline; run them by referencing their task class directly.

### Tasks

#### `dmResiduals`

Measures the residuals in the spatial and wavelength directions between the detector map and the arcline centroids. The
residuals are measured for each visit.

##### Options

- `dmResiduals:useSigmaRange`: Use the sigma range for the color scale in the residual plots. Default is `False`.
- `dmResiduals:spatialRange`: The range of the x-center (i.e. spatial) in the residual plots. Default is `0.1`.
- `dmResiduals:wavelengthRange`: The range of the y-center (i.e. wavelength) in the residual plots. Default is `0.1`.
- `dmResiduals:binWavelength`: The bin size in wavelength for the residual plots in nm. Default is `0.1`.

##### Outputs

| DataSet Type        | Dimensions                             | Description                                                                                 |
|---------------------|----------------------------------------|---------------------------------------------------------------------------------------------|
| `dmQaResidualData`  | `instrument, visit, arm, spectrograph` | Residual data for the given detector and visit.                                             | 
| `dmQaResidualStats` | `instrument, visit, arm, spectrograph` | Summary statistics for the given detector and visit.                                        | 
| `dmQaResidualPlot`  | `instrument, visit, arm, spectrograph` | 1D and 2D plots of the residual between the detectormap and the arclines for a given visit. |

#### `dmCombinedResiduals`

Determines the aggregate statistics for all detectors across all given visits.

##### Options

N/A

##### Outputs

| DataSet Type               | Dimensions   | Description                                                                                       |
|----------------------------|--------------|---------------------------------------------------------------------------------------------------|
| `dmQaCombinedResidualPlot` | `instrument` | 1D and 2D plots of the residual between the detectormap and the arclines for the entire detector. |
| `dmQaDetectorStats`        | `instrument` | Statistics of the residual analysis per detector.                                                 |

#### `imageQualityQa`

Measures image quality (FWHM), spatial flexure, and arc-line flag rates on a per-detector quantum
`(instrument, visit, arm, spectrograph)`. The task classifies each visit by reading FITS headers (`W_SEQTYP`,
`W_SEQNAM`) and routes it to the appropriate measurement path. When `reduceExposure` logs are available it also folds
ISR, cosmic-ray, and `fitDetectorMap` statistics into the metrics table.

The task writes data only; plotting lives in `pfs.drp.qa.iqQaPlots` and is driven after the fact by
`bin.src/plotIqQaTimeSeries.py` (see
[Command-line tools](#command-line-tools)).

##### Inputs

| Connection          | Dataset              | Required    | Description                                                                                                                                                                                                                 |
|---------------------|----------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `arcLines`          | `lines`              | Yes         | Per-line arc emission measurements from `fitDetectorMap`, including second moments (`ixx`/`iyy`), flux, flags, and species.                                                                                                 |
| `detectorMap`       | `detectorMap`        | Yes         | Adjusted mapping from `(fiberId, wavelength)` to detector `(x, y)`. Used to enrich arc lines with wavelength and trace position.                                                                                            |
| `fiberProfiles`     | `fiberProfiles`      | Yes (calib) | Stored fiber profile shapes. Used as a last-resort fallback to derive trace-width FWHM when neither arc lines nor a calexp are available.                                                                                   |
| `detectorMapCalib`  | `detectorMap_calib`  | Yes (calib) | Static, pre-adjustment detectorMap. Differencing its `xCenter` against the measured fiber positions gives the flexure diagnostic `dxCenter`.                                                                                |
| `calexp`            | `calexp`             | No          | Post-ISR calibrated image. When present, cross-dispersion 2nd moments are measured directly from pixel data — the primary path for trace/quartz visits and a fallback for sparse arc visits.                                |
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
| Regular arc (`scienceArc`, all fibers)            | Arc-line second moments (`ixx`/`iyy` → FWHM)                           | `calexp` cross-dispersion moments if `nGoodLines < minGoodLines`       |
| IIS arc (16 engineering fibers)                   | —                                                                      | Reported as sparse (no FWHM); arc catalog does not match IIS positions |
| Regular trace/quartz (`scienceTrace`, all fibers) | `calexp` cross-dispersion moments                                      | Fiber profile calibration widths if no `calexp`                        |
| IIS trace/quartz                                  | —                                                                      | Reported as sparse; too few fibers for a reliable measurement          |
| Science / all-sky (`scienceObject*`)              | `calexp` moments on `FLUXSTD+GOOD` fibers                              | Sparse if FLUXSTD good fraction < `minFluxstdGoodFrac`                 |
| Unknown (`W_SEQTYP` absent)                       | Heuristic: arc lines if sufficient, else `calexp`, else fiber profiles | —                                                                      |

For arc visits, FWHM is computed as the Gaussian-equivalent width
`2√(2 ln 2) × σ` from the second moments of each arc-line profile. For calexp and fiber-profile paths, the same formula
is applied to the cross-dispersion intensity profile measured at regular row intervals.

##### What the Metrics Tell Us

- **`medFwhm`** — Median FWHM in pixels across all good measurements for the detector quantum. Reflects the
  cross-dispersion width of the fiber PSF. Typical good values are 2.5–3.2 px; values above 3.5 px indicate a focus or
  alignment problem.
- **`medDxCenter`** / **`dxCenterRms`** — Median and scatter of the spatial offset between the static calibration
  detectorMap and the measured fiber positions, in pixels. This is a flexure diagnostic: a large `|medDxCenter|` means
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
- **`nLines`** — Number of measurements used (arc lines, calexp samples, or profile swaths).
- **`traceOnly`** — `True` when FWHM comes from fiber profile calibration widths rather than live measurements; these
  values reflect the calibration epoch, not the current visit.
- **`obsType`** / **`seqName`** — Visit classification (`arc`, `trace`, `science`, `allsky`,
  `unknown`) and the raw `W_SEQNAM` string (e.g. `"Arc: HgCd"`) it was derived from.
- **`qaStatus`** — `PASS`, `WARN`, or `FAIL`; the worst of the `medFwhm`, `pctFlagged`, and
  `|medDxCenter|` checks (see thresholds below). The reason for anything other than `PASS` is written to the task log on
  the `IQ QA` line.

##### Pass/Warn/Fail Thresholds

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
as a failure would flag every Ar and Xe arc. See the "Arc Lamp Physics" section of `AGENTS.md` for the underlying line
counts and for the `minSignalToNoisePerSpecies` option in `drp_stella` that addresses the cause.

##### Options

###### Measurement

- `imageQualityQa:minGoodLines`: Minimum good arc-line measurements to trust the arc path. Default `10`.
- `imageQualityQa:minPeakSN`: Minimum peak S/N for calexp profile samples. Default `5.0`.
- `imageQualityQa:maxCalexpFlagRate`: Max fraction of bad calexp samples before rejecting the calexp path. Default
  `0.5`.
- `imageQualityQa:minFluxstdGoodFrac`: Min fraction of good FLUXSTD samples for the stellar calexp path. Default `0.10`.
- `imageQualityQa:profileHalfWidth`: Half-width (pixels) of the cross-dispersion aperture for calexp measurements.
  Default `7`.
- `imageQualityQa:profileYStride`: Row sampling interval (pixels) for calexp profile measurements. Default `50`.

###### Pass/Fail Thresholds

- `imageQualityQa:fwhmWarnThreshold`: Median FWHM (px) above which status is `WARN`. Default `3.2`.
- `imageQualityQa:fwhmFailThreshold`: Median FWHM (px) above which status is `FAIL`. Default `3.5`.
- `imageQualityQa:dxCenterWarnThreshold`: `|medDxCenter|` (px) above which status is `WARN`. Default `1.0`.
- `imageQualityQa:dxCenterFailThreshold`: `|medDxCenter|` (px) above which status is `FAIL`. Default `2.0`.
- `imageQualityQa:flagRateWarnThreshold`: `pctFlagged` (%) threshold for `WARN`, keyed by `arm` or `arm:species`. See
  the threshold table above for defaults.
- `imageQualityQa:flagRateFailThreshold`: `pctFlagged` (%) threshold for `FAIL`, keyed by `arm` or `arm:species`. See
  the threshold table above for defaults.

`DictField` values cannot be set with dot notation on the command line; assign the whole dict as a Python literal:

```bash
-c "imageQualityQa:flagRateWarnThreshold={'b': 50.0, 'r': 15.0, 'b:Argon': 93.0}"
```

##### Outputs

| DataSet Type  | Dimensions                             | Description                                                                                                                                                                       |
|---------------|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `iqQaData`    | `instrument, visit, arm, spectrograph` | Per-line measurements: `fiberId`, `x`, `y`, `lam`, `fwhm`, `theta`, `dxCenter`, `flux`, `fluxErr`, `flag`, `traceOnly`, `peakRatio`, `status`, plus `visit`/`arm`/`spectrograph`. |
| `iqQaMetrics` | `instrument, visit, arm, spectrograph` | Single-row summary; see below.                                                                                                                                                    |

`iqQaMetrics` columns, by group:

- **Identity** — `visit`, `arm`, `spectrograph`, `obsType`, `seqName`
- **Image quality** — `medFwhm`, `medDxCenter`, `dxCenterRms`, `pctFlagged`, `nLines`,
  `traceOnly`, `qaStatus`
- **Flag breakdown** (only when `pctFlagged` is finite) — `pctNotVisible`, `pctBlend`,
  `pctSuspect`, `pctRejected`, `pctBroad`, and on the arc-line path `pctLowSN`, `pctMeasFail`
- **Log-derived** (zero/`NaN` when the corresponding log is absent) — `isrBadPixels`,
  `isrTime`, `cosmicRayCount`, `cosmicRayPixels`, `cosmicRayTime`, `reduceExposureTime`
- **`fitDetectorMap` statistics** — `fitChi2`, `fitDof`, `fitXRms`, `fitYRms`, `fitXSoften`,
  `fitYSoften`, `fitNLines`, `fitTotalLines`, the matching `fitReserved*` columns for the reserved-line sample,
  `fitTraceXRms`, `fitTraceYRms`, and per-species
  `fitSpeciesXRms_<species>` / `fitSpeciesYRms_<species>`
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

## Command-line tools

Scripts live in `bin.src/` and are run directly — there is no SCons step to copy them into a `bin/` directory on `PATH`:

```bash
python bin.src/<script>.py --help
```

### `fiberNormsQa.py`

Thin entry point for `pfs.drp.qa.fiberNormsQa.main`, which plots fiber normalizations from a Butler collection.

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
`pfs.drp.qa.iqQaPlots.plotIqTimeSeries`. This is the plotting that used to live inside
`imageQualityQa` itself. Requires the LSST stack for the Butler query; `--csv` bypasses it.

```bash
python bin.src/plotIqQaTimeSeries.py \
    -b /path/to/butler -c u/user/run28 -o iq_timeseries.png

python bin.src/plotIqQaTimeSeries.py --csv metrics.csv --arm b,r --obs-type arc
```

Filters: `--arm`, `--spectrograph`, `--obs-type` (all comma-separated), plus `--where`
for an arbitrary Butler query expression.

### `fiberNormsQa.py`

Thin entry point for `pfs.drp.qa.fiberNormsQa.main`, which plots fiber normalizations from a Butler collection.
