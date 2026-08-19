# AGENTS.md — drp_qa

Quality Assurance tools for the **Subaru Prime Focus Spectrograph (PFS)** Data Reduction
Pipeline (DRP), built on the **LSST Science Pipelines** `PipelineTask` framework.

This is the single source of instructions for all AI coding assistants working in this
repository. `CLAUDE.md`, `GEMINI.md`, and `.github/copilot-instructions.md` are symlinks 
to this file — edit this file only.

Contents:

1. [Project Overview](#project-overview)
2. [Build, Test & Lint](#build-test--lint)
3. [Architecture](#architecture)
4. [Key Conventions](#key-conventions)
5. [How to Work in This Repo](#how-to-work-in-this-repo)
6. [Git Commit Convention](#git-commit-convention)
7. [Domain Knowledge: `imageQualityQa`](#domain-knowledge-imagequalityqa)
8. [Arc Lamp Physics and b-arm `pctFlagged` Failures](#arc-lamp-physics-and-b-arm-pctflagged-failures)
9. [Butler / Pipeline Data Flow for IQ QA](#butler--pipeline-data-flow-for-iq-qa)
10. [Common Failure Patterns](#common-failure-patterns)
11. [Cross-repo Dependency Notes](#cross-repo-dependency-notes)

---

## Project Overview

`drp_qa` implements QA analyses over PFS data products (detector map residuals, fiber
extraction, sky subtraction, flux calibration, fiber normalization, image quality) and
the pipeline/config glue to run them.

### Key components

- **QA pipeline (`pipelines/drpQA.yaml`)** — defines the sequence of QA tasks.
- **Tasks (`python/pfs/drp/qa/`)**:
  - `imageQualityQa.py` — image quality (FWHM, flag rates); plots in `iqQaPlots.py`
  - `dmResiduals.py`, `dmCombinedResiduals.py` — detector map residuals (per-detector
    and cross-visit combined)
  - `extractionQa.py` — fiber extraction quality
  - `skySubtractionQa.py` — sky subtraction accuracy
  - `fiberNormsQa.py` — fiber normalization plots
  - `fluxCalQa.py`, `fluxCal/fluxCalQA.py` — flux calibration validation
  - `storageClasses.py`, `formatters.py` — custom Butler storage classes / formatters
  - `utils/` — shared helpers (`math.py`, `plotting.py`)
  - `tasks/` — auxiliary task helpers (e.g. `overlapRegionLines.py`)
- **Command-line tools (`bin.src/`)** — run as `python bin.src/<name>.py`:
  - `fitDetectorMapLogQa.py` — stdlib-only OK/WARN/BAD gate over `fitDetectorMap` logs;
    exits non-zero on BAD. Keep it stack-free.
  - `imageQualityLogQa.py` — per-visit report from `reduceExposure`/`imageQualityQa`
    logs or a direct Butler query; dashboard plot, markdown report, JSON dump
  - `plotIqQaTimeSeries.py` — cross-visit `iqQaMetrics` time series via `iqQaPlots.py`
  - `fiberNormsQa.py` — entry point for `fiberNormsQa.main`
- **Inter-project relationship** — `drp_qa` depends on `drp_stella` (`../drp_stella`),
  which contains the core reduction logic and C++ primitives.

### Key files & directories

- `python/pfs/drp/qa/` — primary Python source for QA tasks
- `pipelines/` — YAML definitions for `pipetask` execution
- `bin.src/` — command-line scripts, run directly as `python bin.src/<name>.py`.
  There is no SCons `shebang()` step and no generated `bin/` directory.
- `ups/` — EUPS configuration (`drp_qa.table`, dependencies and `PATH`/`PYTHONPATH`)
- `tests/` — pytest-based tests (may rely on the LSST/PFS stack)
- `pyproject.toml` — build config (setuptools) and **all** tool configs (Ruff, pytest).
  Keep it that way: don't add standalone per-tool config files.
- `README.md` — project overview and usage notes
- `CHANGELOG.md` — Keep a Changelog format; add user-visible changes under
  `## [Unreleased]`. Released sections are keyed to the weekly tags (`w.2026.29`).

Package metadata: name `pfs-drp-qa`, requires Python >= 3.12.

---

## Build, Test & Lint

### Environment setup

Most tasks import `pfs.drp.stella`, so they need the LSST stack:

```bash
source /path/to/stack/loadLSST.bash
setup -r ../drp_stella   # drp_stella must be set up first
setup -r .
```

EUPS is still used for dependency resolution via `ups/drp_qa.table`, but **there is no
SCons build** — `SConstruct` and `bin.src/SConscript` were removed. `setup -r .` only
prepends `PATH` and `PYTHONPATH`; nothing is compiled or generated.

### Install

There are no compiled components, so a plain install is enough:

```bash
pip install -e .
# or, using the checked-in lockfile
uv sync
# or build a wheel/sdist
python -m pip install build && python -m build
```

### Tests

```bash
# Full suite
pytest tests/

# Single test file
pytest tests/test_fitDetectorMapLogQa.py

# Single test
pytest tests/test_dmResiduals.py::TestDetectorMapResiduals::testResiduals
```

`tests/test_fitDetectorMapLogQa.py` is stdlib-only and runs without the stack — keep it
that way when editing `bin.src/fitDetectorMapLogQa.py`. Other tests use
`lsst.utils.tests` and require the LSST/PFS stack; some are placeholders or rely on
external data. If a test cannot run without that environment, validate the logic via
unit-testable components and document the constraint in the test, the update note, or
the PR.

### Running pipelines

```bash
pipetask run -p pipelines/drpQA.yaml -b /path/to/butler -i input/collection -o output/collection
```

### Linting & formatting

**Ruff** replaces Black, isort and Flake8; all configuration lives in `pyproject.toml`
under `[tool.ruff]`. There is no `setup.cfg`.

```bash
ruff format .          # formatting (replaces black)
ruff check --fix .     # linting + import sorting (replaces flake8 + isort)
ruff check .           # lint without applying fixes
```

Ruff is the only style tool; there is no separate type checker.

- **Line length**: 110, `target-version = "py312"`
- **Rule sets selected**: `E`, `W`, `F`, `I`, `N`, `D`, `UP`, `B`, `C4`, `SIM`, `RUF`
- **Docstrings**: NumPy convention; the missing-docstring rules (`D100`–`D107`) are off
- **Naming**: LSST conventions — camelCase for modules, methods, arguments and
  variables. This conflicts with pep8-naming, so `N802`, `N803`, `N806`, `N812`, `N813`,
  `N815`, `N816` and `N999` are in the ignore list. **Do not "fix" camelCase names to
  satisfy pep8-naming** — re-enabling those rules flags ~730 intentional LSST-style
  names.
- **Excludes**: none configured; Ruff respects `.gitignore`, which already covers `bin/`
  and `tests/.tests/`.

The codebase is not currently Ruff-clean (~470 findings, mostly `E501`, `D4xx`, `UP`,
and `F401`). Lint your own changes; don't bulk-reformat unrelated files in a feature PR.

---

## Architecture

### PipelineTask pattern

Always use `lsst.pipe.base.PipelineTask` for new tasks. Every QA task follows the same
three-class structure:

1. **`*Connections`** — declares Butler inputs/outputs with `storageClass`,
   `dimensions`, and connection type (`Input`, `Output`, `PrerequisiteInput`). Inherits
   from `PipelineTaskConnections`.
2. **`*Config`** — declares configuration fields using `lsst.pex.config.Field`. Bound to
   its Connections class via `pipelineConnections=`.
3. **`*Task`** — the task class. `runQuantum()` fetches inputs from the Butler and calls
   `run()`. `run()` contains the actual analysis logic and returns an
   `lsst.pipe.base.Struct`.

```
Connections ←──── Config ←──── Task
                                 └── runQuantum() → butler.get/put
                                 └── run() → returns Struct
```

### Butler data flow

- `runQuantum` receives `QuantumContext`, `InputQuantizedConnection`,
  `OutputQuantizedConnection`.
- Inputs are fetched via `butlerQC.get(inputRefs)`.
- Outputs are stored via `butlerQC.put(outputs, outputRefs)`.
- `dataId` (visit, arm, spectrograph, instrument) is extracted from
  `inputRefs.<connection>.dataId.mapping` and passed into `run()` for plot labeling.

### Task dimensions

Tasks are scoped by their `dimensions`:

- Per-detector: `("instrument", "visit", "arm", "spectrograph")`
- Per-visit: `("instrument", "visit")`
- Combined/aggregate: `("instrument",)` — these use `multiple=True` inputs to consume
  outputs from per-detector tasks

### Pipeline YAML

`pipelines/drpQA.yaml` wires tasks together. Individual tasks can be run with the
`#taskName` fragment syntax:

```bash
pipetask run -p pipelines/drpQA.yaml#dmResiduals -b /path/to/butler -i input/coll -o output/coll
```

### Custom storage classes & formatters

- `storageClasses.py`: `MultipagePdfFigure` wraps `PdfPages` for multi-page PDF output
  via Butler — use `.append(fig)` to add pages, not `.savefig()`. `QaDict` is a plain
  `dict` subtype for Butler-serializable QA results.
- `formatters.py`: `PdfMatplotlibFormatter` — Butler formatter that saves figures as
  `.pdf` instead of `.png`.

### Dependency on `drp_stella`

The sibling repo `../drp_stella` provides the core data model types used as Butler
inputs:

- `ArcLineSet`, `DetectorMap`, `PfsArm`, `FiberProfileSet`, `PfsCalibratedSpectra`,
  `PfsConfig`
- Math utilities: `pfs.drp.stella.utils.math.robustRms`
- Sky/focal-plane fitting tasks used directly inside QA tasks (e.g.
  `FitBlockedOversampledSplineTask`, `subtractSky1d`)

`drp_stella` also contains:

- High-performance C++ implementations of detector models, fiber profiles, and spectral
  extraction
- Python wrappers for C++ classes using `lsst.utils.continueClass`
- Core DRP pipelines (e.g. `reduceExposure.yaml`, `science.yaml`)

Use `.pyi` files for C++ extensions in `drp_stella` to provide type hints.

---

## Key Conventions

### Task-level error handling

`runQuantum` wraps `self.run()` in a `try/except ValueError` — errors are logged but do
not crash the pipeline. Only write outputs when `run()` succeeds.

### Plot outputs

- Single-figure tasks: return a `matplotlib.figure.Figure` in the `Struct`; Butler stores
  it via `storageClass="Plot"` using `PdfMatplotlibFormatter`.
- Multi-page tasks: return a `MultipagePdfFigure`; call `.append(fig)` for each page.

### Shared plotting utilities (`utils/plotting.py`)

- `div_palette` — diverging colormap with over/under/bad colors for residual plots.
- `detector_palette` — arm color mapping: `{"b": blue, "r": red, "n": goldenrod, "m": pink}`.
- `spectrograph_plot_markers` — marker shapes per spectrograph: `{1: "s", 2: "o", 3: "X", 4: "P"}`.
- `scatterplot_with_outliers()` — standard scatter function used across residual plots.

### Shared math utilities (`utils/math.py`)

- `getChi2`, `getWeightedRMS`, `gaussianFixedWidth`, `gaussian_func` — used across tasks;
  prefer these over reimplementing.

### Typing

There is no static type checker configured for this repo, and none is run in CI. New code
in `pfs.drp.qa` should still include type hints where they aid readability, but they are
not verified. Note that `lsst.*` and most `pfs.*` packages ship no annotations, so hints
on Butler/LSST objects are documentation rather than something a tool will check.

---

## How to Work in This Repo

1. Prefer minimal, targeted code changes with a clear rationale in the update log.
2. Follow the existing patterns in `python/pfs/drp/qa/*` for new or modified files.
3. If changes affect runtime behavior, add or update tests under `tests/` when feasible.
4. Run style checks locally (`ruff format . && ruff check .`) before submitting, scoped
   to the files you touched.
5. If tests require the LSST/PFS environment and are not runnable in the current session,
   validate logic via unit-testable components and note the environment constraint.
6. Keep public APIs stable. If you must change one, update dependent code and add
   migration notes in `README.md` and/or docstrings.
7. Record user-visible changes — new tasks, new or removed config fields, new
   `iqQaMetrics` columns, changed defaults — under `## [Unreleased]` in `CHANGELOG.md`.
8. Include concise docstrings describing purpose, inputs, outputs, and any assumptions —
   especially about LSST data structures.
9. Prefer small, focused PRs with clear descriptions of the change and its QA impact.

---

## Git Commit Convention

Commits made with AI assistance must include trailers identifying the tool and the model,
in addition to the standard co-author trailer:

```bash
git commit \
  --trailer "Co-authored-by: <Assistant> <email>" \
  --trailer "AI-Tool: <Tool> (<Vendor>)" \
  --trailer "AI-Model: <model-id>" \
  -m "<commit message>"
```

Per-tool values:

| Tool | `Co-authored-by` | `AI-Tool` |
|---|---|---|
| Claude Code | `Claude Code <noreply@anthropic.com>` | `Claude Code (Anthropic)` |
| Junie | `Junie <junie@jetbrains.com>` | `Junie (JetBrains)` |

Set `AI-Model` to the model actually in use, e.g. `claude-opus-4-6` or
`claude-sonnet-4-6`.

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
2. **Calexp image moments** — direct cross-dispersion profile fit from post-ISR pixel
   data (`calexp` connection); used when arc lines are absent or sparse
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

- `obs_type`: one of `"arc"`, `"trace"`, `"science"`, `"allsky"`, `"unknown"`
- `is_iis`: `True` when illuminated by the 16 IIS engineering fibers (lamp header names
  from `getLamps()` end with `"_eng"`, e.g. `"Ar_eng"`)
- `seq_nam`: raw `W_SEQNAM` string

**IIS vs regular**: IIS frames illuminate only 16 engineering fibers rather than all 600
science fibers. Arc-line shape measurements are unreliable for IIS frames because the
line catalog doesn't match the sparse illumination; the calexp path or fiber-profile
fallback is preferred in that case.

### Key config fields

| Config field | Purpose |
|---|---|
| `minGoodLines` (default 10) | Min good arc-line measurements to trust the arc path |
| `minPeakSN` (default 5.0) | Min peak S/N for calexp profile samples |
| `maxCalexpFlagRate` (default 0.5) | Max fraction of bad calexp samples before rejecting calexp path |
| `minFluxstdGoodFrac` (default 0.10) | Min fraction of good FLUXSTD samples for stellar calexp path |
| `profileHalfWidth` (default 7) | Half-width (px) of the cross-dispersion aperture for calexp measurements |
| `profileYStride` (default 50) | Row sampling interval (px) for calexp profile measurements |
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

### Output metrics

`iqQaMetrics` DataFrame (one row per quantum). Core columns:

- `medFwhm`: median FWHM in pixels
- `medDxCenter` / `dxCenterRms`: median and scatter of the spatial offset from
  `detectorMap_calib`, a flexure diagnostic
- `pctFlagged`: percentage of arc lines flagged by `fitDetectorMap`
- `nLines`: number of measurements used
- `traceOnly`: True when falling back to fiber-profile widths
- `obsType` / `seqName`: visit classification and raw `W_SEQNAM`
- `qaStatus`: `"PASS"`, `"WARN"`, or `"FAIL"` — the worst of the FWHM, flag-rate, and
  `|medDxCenter|` checks

Additional columns are added dynamically: a per-status-bit flag breakdown
(`pctNotVisible`, `pctBlend`, `pctSuspect`, `pctRejected`, `pctBroad`, plus `pctLowSN` /
`pctMeasFail` on the arc-line path), and — when the optional `isr_log`, `cosmicray_log`,
and `reduceExposure_log` connections are present — ISR, cosmic-ray, and `fitDetectorMap`
statistics (`fitChi2`, `fitXRms`, `fitYRms`, `fitReserved*`, `fitSpecies*Rms_<species>`,
per-fiber arrays).

The task writes **data only**. Plotting lives in `iqQaPlots.py` and is driven after the
fact by `bin.src/plotIqQaTimeSeries.py`; there is no `iqQaPlot` dataset.

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

### Fix: `minSignalToNoisePerSpecies` in `fitDistortedDetectorMap`

A `DictField(keytype=str, itemtype=float, default={})` was added to
`FitDistortedDetectorMapConfig` in `drp_stella`. It allows per-species S/N thresholds to
be set independently of the global `minSignalToNoise` (default 10). Suggested values for
the b arm:

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

Fix (committed to `drp_stella`): guard `dof <= 0` in `calculateSoftening` → return `0.0`
early; also collapse `val < 0 or not isfinite(val)` into a single guard before the bisect
call.

---

## Butler / Pipeline Data Flow for IQ QA

```
detectorMap.yaml#fitDetectorMap
    → outputs: detectorMap, lines (=arcLines)

drpQA.yaml#imageQualityQa
    ← reads: arcLines, detectorMap,
             fiberProfiles, detectorMap_calib (calibrations),
             calexp, pfsConfig (optional),
             isr_log, cosmicray_log, reduceExposure_log (optional)
    → writes: iqQaData, iqQaMetrics

bin.src/plotIqQaTimeSeries.py
    ← reads: iqQaMetrics (all quanta in a collection)
    → writes: time-series PNG via pfs.drp.qa.iqQaPlots.plotIqTimeSeries
```

`reduceExposure` is **not** required between `fitDetectorMap` and `imageQualityQa`. The
`arcLines` (`lines`) and `detectorMap` outputs from `fitDetectorMap` are read directly.
When `reduceExposure` *has* run, its logs are picked up through the optional `*_log`
connections and its statistics are folded into `iqQaMetrics`.

Cross-visit QA is **not** a pipeline task. There is no `imageQualityQaSummary` and no
combined task — run `bin.src/plotIqQaTimeSeries.py` over the output collection after the
fact. This keeps aggregation off the critical path of a reduction and lets it be re-run
against a CSV without a Butler.

---

## Common Failure Patterns

From engineering run data:

| Symptom | Likely cause | Remedy |
|---|---|---|
| b-arm `pctFlagged` above the `b:Argon`/`b:Xenon`/`b:Krypton` thresholds | Lamp has very few/faint b-arm lines → global S/N cut flags almost all | Set `minSignalToNoisePerSpecies` for those species in `fitDetectorMap` |
| b-arm `pctFlagged` > 15 % for HgCd arcs (or > 50 % for Ne) | Genuine crowding or optical problem | Investigate `medFwhm`; if FWHM is also high → optics issue |
| High `pctMeasFail` with low `pctLowSN` | Centroid/photometry failures rather than faint lines — not lamp physics | Investigate the image; relaxing S/N thresholds will not help |
| All arms FWHM > 3.5 px for a single spectrograph module | Hardware/focus issue | Flag the entire SM as bad for that visit range |
| `\|medDxCenter\|` > 1 px across all arms of an SM | Flexure or a stale `detectorMap_calib` | Check the calib validity range before blaming the optics |
| `medDxCenter` ≈ 0 but `dxCenterRms` large | Distortion rather than bulk shift | Look at the `dxCenter` distribution in `iqQaData`, not just the summary |
| `traceOnly=True` for all arc visits | `arcLines` connection missing or `minGoodLines` not satisfied | Check `fitDetectorMap` ran and produced `lines` |
| Calexp path produces FWHM ~7 px for IIS arc frames | Scattered light passes S/N gate in neighbouring positions | Expected; `maxCalexpFlagRate` discards the calexp result and keeps sparse arc data |
| All `fit*` metric columns are `NaN`/0 | The `*_log` connections were absent from the input collection | Expected when `reduceExposure` hasn't run; the IQ metrics themselves are unaffected |
| `ValueError: f(a) = NaN` in bisect during `fitDetectorMap` | Per-fiber dof=0 when very few arc lines remain after S/N cut | Guard `dof <= 0` in `calculateSoftening` (see above) |

---

## Cross-repo Dependency Notes

- **`drp_stella`** must be set up before `drp_qa`
  (`setup -r ../drp_stella; setup -r .`).
- Key types from `drp_stella` used by `imageQualityQa`:
  - `ArcLineSet` — per-line measurements including `ixx`, `iyy`, `flux`, `fluxErr`,
    `flag`, `description` (species name), `status`
  - `DetectorMap` — maps `(fiberId, wavelength) → (x, y)`
  - `FiberProfileSet` — per-fiber cross-dispersion profile shapes
  - `addTraceLambdaToArclines()` — enriches ArcLineSet with wavelength column
- **`obs_pfs`**: `getLamps(metadata)` returns active lamp names; names ending in
  `"_eng"` indicate IIS illumination. Import is guarded so the task gracefully degrades
  if `obs_pfs` is unavailable.
- Line lists live in `obs_pfs/pfs/lineLists/{Ar,Xe,Kr,Ne,HgCd}.txt`; column format:
  `wavelength intensity species`.
