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
7. [Butler / Pipeline Data Flow for DM & IQ QA](#butler--pipeline-data-flow-for-dm--iq-qa)
8. [Cross-repo Dependency Notes](#cross-repo-dependency-notes)

This file holds durable rules and conventions only. Empirical QA findings — `imageQualityQa`
domain knowledge, arc-lamp physics, and the observed failure-pattern table — live in
[`docs/qa-domain-notes.md`](docs/qa-domain-notes.md); numbers there drift with the data and
should be re-checked, not trusted.

---

## Project Overview

`drp_qa` implements QA analyses over PFS data products (detector map residuals, fiber
extraction, sky subtraction, flux calibration, fiber normalization, image quality) and
the pipeline/config glue to run them.

### Key components

- **QA pipeline (`pipelines/drpQA.yaml`)** — defines the sequence of QA tasks. It
  currently registers **five** task labels, and only these run under `pipetask`:
  `dmResiduals`, `dmCombinedResiduals`, `extractionQa`, `extractionQaCombined`,
  `imageQualityQa`.
- **Tasks in the pipeline (`python/pfs/drp/qa/`)**:
  - `imageQualityQa.py` — image quality (FWHM, flag rates); plots in `iqQaPlots.py`
  - `dmResiduals.py`, `dmCombinedResiduals.py` — detector map residuals (per-detector
    and cross-visit combined)
  - `extractionQa.py`, `extractionQaCombined.py` — fiber extraction quality
    (per-detector, and combined per `(instrument, visit, arm)`)
- **QA modules *not* wired into `drpQA.yaml`** — these exist as `PipelineTask`s or CLIs
  but must be run via their own pipeline/entry point:
  - `skySubtractionQa.py` — sky subtraction accuracy (`SkyArmSubtractionTask`,
    `SkySubtractionQaTask`)
  - `fluxCalQa.py`, `fluxCal/fluxCalQA.py` — flux calibration validation
  - `fiberNormsQa.py` — **not** a `PipelineTask`; a standalone Butler-driven CLI module
    whose `main()` is invoked from `bin.src/fiberNormsQa.py`
- **Guider analysis (`python/pfs/drp/qa/guiders/`)** — notebook tools for the AG
  cameras, being moved from `pfs.drp.stella.utils.guiders` (epic PIPE2D-1891). Four
  layers: `coordinates`, `queries` (the only code touching the opdb or a butler),
  `analysis` and `plotting` (DataFrames in, never a database). Stack-free; tested in
  `tests/guiders/`.
- **Support modules (`python/pfs/drp/qa/`)**:
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
  which contains the core reduction logic and C++ primitives, and on `pfs_utils`. Both
  are `setupRequired` in `ups/drp_qa.table`; `pfs-utils` is also a hard `pyproject.toml`
  dependency (installed from git).

### Key files & directories

- `python/pfs/drp/qa/` — primary Python source for QA tasks
- `pipelines/` — YAML definitions for `pipetask` execution
- `bin.src/` — command-line scripts, run directly as `python bin.src/<name>.py`.
  There is no SCons `shebang()` step and no generated `bin/` directory.
- `ups/` — EUPS configuration (`drp_qa.table`, dependencies and `PATH`/`PYTHONPATH`)
- `tests/` — pytest-based tests (may rely on the LSST/PFS stack). `tests/SConscript` is
  the one surviving SCons file; it is vestigial and not exercised by `pytest`.
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

`ups/drp_qa.table` declares `setupRequired(drp_stella)` and `setupRequired(pfs_utils)`,
so both must be available.

EUPS is still used for dependency resolution via `ups/drp_qa.table`, but **there is no
SCons build** — `SConstruct` and `bin.src/SConscript` were removed. `setup -r .` only
prepends `PATH` and `PYTHONPATH`; nothing is compiled or generated. (`tests/SConscript`
still exists but is vestigial.)

### Install

There are no compiled components, so a plain install is enough:

```bash
pip install -e .
# or, with uv (the lockfile it writes isn't tracked)
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

`pyproject.toml` sets `addopts = "-ra --import-mode=importlib"`. Note the `importlib`
import mode: test modules are not added to `sys.path`, so they can't import each other
by bare module name.

`tests/test_fitDetectorMapLogQa.py` is stdlib-only and runs without the stack — keep it
that way when editing `bin.src/fitDetectorMapLogQa.py`. Other tests use
`lsst.utils.tests` and require the LSST/PFS stack; some are placeholders or rely on
external data. If a test cannot run without that environment, validate the logic via
unit-testable components and document the constraint in the test, the update note, or
the PR.

### Continuous integration

Two GitHub Actions workflows run on every pull request:

| Workflow | Blocking | What it does |
|---|---|---|
| `.github/workflows/tests.yml` | yes | On Python 3.12 and 3.13: `pytest -v` with only pytest installed, and `pytest -v tests/guiders` with pfs-utils and drp_pfs_data |
| `.github/workflows/lint.yml` | yes | `ruff check .` and `ruff format --check .` over the whole tree |

**The repository is Ruff-clean and both checks gate the whole tree.** Keep it that way:
fix findings in the code you touch rather than widening the ignore list in
`pyproject.toml`.

**The package is not installed in CI.** `pip install -e .` would pull `pfs-utils` (and
transitively `pfs-datamodel`, `pfs-instdata`) from GitHub, making every run depend on
three other repositories. Only stack-free tests run, in two jobs:

- **`stack-free`** installs only pytest, so its tests import nothing beyond the standard
  library.
- **`guiders`** runs `tests/guiders/` against `python/` on `PYTHONPATH`. It installs
  pfs-utils from git with `--no-deps`, plus the packages listed in the workflow, and
  clones drp_pfs_data at the PR's branch name (else `master`) with LFS smudging skipped,
  pulling only `guiders/`. `DRP_PFS_DATA_DIR` points at the clone. To give a PR new test
  data, push a drp_pfs_data branch with the same name. Keep the workflow's package list
  in step with `pyproject.toml`.

`tests/conftest.py` ignores `tests/guiders/` when the imports of its `conftest.py`
(numpy, pandas, pfs_utils) are missing, so the `stack-free` job skips it.

**Tests that need the stack** cannot be collected without it — a module-level
`import lsst.utils.tests` fails during collection and aborts the whole run, so an in-test
`try/except ImportError` never gets the chance to skip. `tests/conftest.py` lists those
modules in `_STACK_MODULES` and ignores them when the stack is absent. Add new ones there,
or better, keep the logic under test in pure functions that take arrays and DataFrames so
no stack is needed at all.

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
- **Excludes**: `examples/` — out-of-order imports (`E402`), unused imports and
  cross-cell names (`F401`/`F821`) are inherent to notebooks, not defects. Ruff also
  respects `.gitignore`, which covers `bin/` and `tests/.tests/`.
- **Also ignored**: `E501` (`ruff format` already enforces `line-length` for code; what
  E501 still catches is long regexes and report strings the formatter will not split),
  and `RUF001`–`RUF003` (Greek letters and typographic dashes are intentional here).

The codebase **is** Ruff-clean: `ruff check .` and `ruff format --check .` both pass, and
CI gates on them. Fix findings in the code you touch rather than adding to the ignore
list.

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
- `opaqueColorbar()` — context manager that draws a translucent mappable's colorbar
  opaque.

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

## Butler / Pipeline Data Flow for DM & IQ QA

```
detectorMap.yaml#fitDetectorMap
    → outputs: detectorMap, lines (=arcLines)

drpQA.yaml#imageQualityQa            dims: (instrument, visit, arm, spectrograph)
    ← reads: arcLines, detectorMap,
             fiberProfiles, detectorMap_calib (calibrations),
             calexp, pfsConfig (optional),
             isr_log, cosmicray_log, reduceExposure_log (optional)
    → writes: iqQaData, iqQaMetrics

drpQA.yaml#dmResiduals               dims: (instrument, visit, arm, spectrograph)
    ← reads: raw.visitInfo, detectorMap, lines, reduceExposure_config
    → writes: dmQaResidualData, dmQaResidualStats, dmQaResidualPlot

drpQA.yaml#dmCombinedResiduals       dims: (instrument,)   [multiple=True inputs]
    ← reads: detectorMap, dmQaResidualData, dmQaResidualStats
    → writes: dmQaDetectorStats, dmQaCombinedResidualPlot

drpQA.yaml#extractionQa              → extQaStats, extQaImage, extQaImage_pickle
drpQA.yaml#extractionQaCombined      ← extQaImage_pickle (multiple)
                                     → extQaStatsCombined

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

## Cross-repo Dependency Notes

- **`drp_stella`** must be set up before `drp_qa`
  (`setup -r ../drp_stella; setup -r .`).
- **`pfs_utils`** is the other `setupRequired` entry in `ups/drp_qa.table`, and
  `pfs-utils` is a hard `pyproject.toml` dependency pulled straight from GitHub —
  a plain `pip install -e .` will try to clone it.
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
