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
  currently registers **six** task labels, and only these run under `pipetask`:
  `extractionQa`, `extractionQaCombined`, `imageQualityQa`, `dmLineResiduals`,
  `twilightFluxQa`, `reductionStats`.
- **Reduce then measure (two `pipetask` commands)** — `drpQa.py redo` and `drpQa.py calib validate`
  reduce with drp_stella's `reduceExposure` (less `mergeArms`), applying `drpActor`'s per-sequence-type
  config, and then measure with `drpQA.yaml`'s tasks; the pipeline paths and labels are in
  `comparison.reduction` (`REDUCE_*`, `MEASURE_*`). There is no combined pipeline file.
- **Tasks in the pipeline (`python/pfs/drp/qa/`)**:
  - `imageQualityQa.py` — image quality (FWHM, flag rates); numbers only, no verdict
  - `extractionQa.py`, `extractionQaCombined.py` — fiber extraction quality
    (per-detector, and combined per `(instrument, visit, arm)`)
  - `dmLineResiduals.py` — per-species residuals of `lines` against `detectorMap_calib`
    (drift) and the visit's `detectorMap` (scatter on `RESERVED` lines, and their median in
    km/s); numbers only, the measurement stack-free in `metrics/lineResiduals.py`
  - `twilightFluxQa.py` — stores each lit fiber's median flux on a twilight
    (`twilightFiberFlux`; other visits stop at the header, before the pfsArm is read);
    `drpQa.py load` measures and judges a sequence's exposures against each other once the
    sequence is complete, with the stack-free `metrics/twilight.py`
  - `reductionStats.py` — detector health read from the reduction's task metadata
    (`isr_metadata`, `reduceExposure_metadata`) and the ISR header: ISR counts, per-amp
    gain/read-noise and the detectorMap fit scalars, long format; stored, not judged, with
    the stack-free measurement in `metrics/reductionStats.py`. Anchored on `raw.metadata`
    (the reduction's own inputs all optional), so it runs on every exposed detector and
    writes the coverage metrics `isrRan`, `reduceExposureRan` and `reductionStage` whatever
    ran — even a detector whose reduction never started. Loaded into `detector_health_qa`.
- **QA modules *not* wired into `drpQA.yaml`** — these exist as `PipelineTask`s or CLIs
  but must be run via their own pipeline/entry point:
  - `fiberProfilesQa.py` — measures a `fiberProfiles` calibration, per detector: one row per fiber
    (`fiberProfilesQaFibers`: width, wing light, `wingExcess` against the slit neighbours, and whether
    the profile is a copy of its neighbours, `replaced`) and the detector's metrics in long format
    (`fiberProfilesQaMetrics`). Numbers only. It has its own pipeline, `pipelines/fiberProfilesQa.yaml`,
    which `drpQa.py calib profiles` runs on a candidate calibration before it is certified; the
    stack-free measurement is `metrics/fiberProfiles.py`. It skips visits that are not quartzes unless
    `anyFrameType` is set, so that it can join `drpQA.yaml` once something reads its tables
  - `skySubtractionQa.py` — sky subtraction accuracy (`SkyArmSubtractionTask`,
    `SkySubtractionQaTask`)
  - `fluxCalQa.py`, `fluxCal/fluxCalQA.py` — flux calibration validation
  - `fiberNormsQa.py` — **not** a `PipelineTask`; a standalone Butler-driven CLI module
    whose `main()` is `drpQa.py fiberNorms`
- **Redo a run (`python/pfs/drp/qa/comparison/`)** — `drpQa.py redo <run>` (`comparison/redo.py`) redoes
  an observing run with an updated pipeline end to end: read the opdb, reduce the raw frames (unless
  `--reductions` reuses a reduced collection), measure, load qadb, make and compare normal values against
  the previous pipeline, judge, and point at `pfs_qa_dashboard` for the run report. It reads the opdb in
  memory and writes no parquet; `--only process,load,values,judge` resumes a subset. `queries` is the only
  code touching the opdb and `butlerQueries` the only code touching a Butler (both take it as an argument);
  `runs`, `classify`, `plan` and `reduction` are stack-free and tested in `tests/comparison/`. The observing
  runs are package data (`comparison/data/observingRuns.yaml`). Calibration vs science is decided by
  what an exposure is (sequence type, and the design's proposals for sky), never by when it was taken.
  `calibInputs` (which exposures of a calibration set to build from) and `calibValidate` (candidate
  calibrations compared on the same visits) are stack-free, with plots in `plotting/calibInputs.py`;
  `drpQa.py calib inputs` and `drpQa.py calib validate` drive them, reading the listing from the opdb
  (`--opdb-host`) and numbers from a Butler collection. The opdb reads list science visits: they stay local.
- **Guider analysis (`python/pfs/drp/qa/guiders/`)** — notebook tools for the AG
  cameras, being moved from `pfs.drp.stella.utils.guiders` (epic PIPE2D-1891). Four
  layers: `coordinates`, `queries` (the only code touching the opdb or a butler),
  `analysis` and `plotting` (DataFrames in, never a database). `compat` keeps drp_stella's
  four AG reader names as deprecated wrappers of `queries.readAgcData`; nothing in the
  package may use it. Stack-free; tested in `tests/guiders/` with a fake `OpDB`, a stub
  butler and real AG data in `tests/guiders/data/` (fixtures in its `conftest.py`).
  Each plot is described, with an annotated sample, in the notebook
  `docs/guider-plots.ipynb`, committed with its outputs: when you add or change a plot,
  update its section and rerun it (`jupyter nbconvert --to notebook --execute --inplace`).
  `docs/guiders-migration.ipynb` runs each drp_stella guiders call beside its replacement
  (needs the opdb; committed without outputs): keep it in step with renamed arguments.
  The end-of-run report adds `queries.readAgRun` (aggregates over a period's AG
  exposures), `runAnalysis` (stack-free: per-exposure camera states, dropouts, settled jitter,
  clusters of rejected matches, position-model errors, flags), `runPlotting` (figures from an `AgRun`) and `runReport` (fetch to parquet, then the
  Markdown report), driven by `drpQa.py agReport` with the same observing runs `redo` uses; tested in
  `tests/guiders/test_runAnalysis.py` and `test_runReport.py` on a synthetic run (`makeAgRunTables`). Its plots are not in `docs/guider-plots.ipynb`: a run
  holds science visits, which can't be committed, so each run's `report.md` is their description.
- **Plotting (`python/pfs/drp/qa/plotting/`)** — the QA plots: DataFrames in, figures
  out (`calibInputs`; `dmResiduals.plotDetectorResiduals` for one detector's line residuals
  — the former `dmResiduals` task's 1D and 2D views, drawn from `dmLineResidualData`; and
  `palettes` for the shared palettes and `scatterplot_with_outliers`). It imports no Butler,
  stack or task module, so plots can be drawn from stored data; `tests/plotting/test_imports.py`
  enforces this. `utils.plotting` re-exports the palettes.
- **Metrics (`python/pfs/drp/qa/metrics/`)** — stack-free: the limits, the judging, and the
  measurements that need no Butler.
  - `registry` declares each metric once (`METRIC_SPECS`: direction, units, reference) and holds
    every limit: `CHECKS`, 14 checks with a WARN and a FAIL each and the frame types they apply to,
    and `EXCLUSIONS`, what is measured but not judged, with the reason. Nothing else in the package
    holds a limit. It also says what a frame is, and nothing else does: `SEQUENCE_TYPES` gives each
    IIC sequence type (`seqType`, `W_SEQTYP` as the tasks store it) its frame type, `frameType` adds
    the sequence name and the illumination, and `NOT_JUDGED` names the frame types no check applies
    to, with the reason. A sequence type missing from the table has no frame type. `lampOf` gives
    the lamp a normal value is kept per, from the header's lamps (`lamps`) or else the sequence name.
  - `gate` is the only judging path (`measurements`, `judge`, `gate`): numbers, checks and normal
    values in, one verdict per image out.
  - `normalValues` makes each camera's normal values from a run's calibration frames (`candidates`),
    sets them beside those in force (`compare`) and returns the ones to store (`accepted`).
  - `lineResiduals` is `dmLineResiduals`' measurement; `longFormat` holds `iqQaSpeciesMetrics`;
    `saturation` (the calexp's `SAT` plane) and `lineFlux` (per-species flux of `lines`) are
    `imageQualityQa`'s unjudged measurements; `twilight` measures twilight sequences, each exposure
    against the sequence's others; `visitRollup` rolls verdicts up to one per visit; `readers` reads
    the tasks' tables from a Butler it is given.
  - `fiberProfiles` is `fiberProfilesQa`'s measurement: per-swath and per-fiber statistics of a
    `fiberProfiles` calibration, each fiber against its slit neighbours (`neighbourResidual`) and against
    another calibration (`compareFibers`), and the profiles drp_stella replaced with the mean of two to four
    neighbours' (`findReplaced`), which are left out of that reference. Tested in `tests/fiberProfiles/`, on
    synthetic profiles and on the Run30 and Run31 calibrations of r4 (`data/makeProfileFixtures.py`).
    `WING_EXCESS_LIMIT` and `PROFILE_MINIMUM_LIMIT` are the run-level listing limits of `drpQa.py calib profiles`
    (`comparison/calibProfiles.py`, `calibProfilesCli.py`), which measures candidate calibrations before one
    is certified and reports what to do, the fibers over the limit (a row per fiber, an arm per column) and
    the detectors, against the calibration in force, with `plotting/fiberProfiles.py`'s figure.
  - `slitOffsets` is the slit-offset measurement (PIPE2D-1677): per-fiber outliers within an MTP
    ferrule (`slitOffsetResiduals`/`flagOutliers`/`offsetSummary`), the cross-arm confirmed set
    (`crossArmConfirmed`) and the run-to-run change (`compareRuns`/`changeSummary`) are run-level, read
    from the calibration comparison; `offsetInvariance`/`invarianceRows` are `dmLineResiduals`'
    per-visit check that the visit's slit offsets still equal the calibration's.

  Read [`docs/qa-principles.md`](docs/qa-principles.md) (rules R1–R7, the procedure and the
  checklist) before adding a metric or a check. `tests/metrics/test_knownFaults.py` holds a table
  of known faults, on stored numbers of calibration and engineering visits, that every change must
  still catch.
- **qadb (`python/pfs/drp/qa/qadb/`)** — `loader` copies a collection's numbers from a Butler into
  qadb, calls the gate and stores one verdict per image in `qa_gate`; `judge` judges stored numbers
  again and makes, compares and stores normal values (`qa_normal_value`), from qadb alone; `reader`
  reads numbers, verdicts and normal values back; `tables` and `writer` are the row shapes and the
  idempotent writes; `visitSummary` prints the verdicts per visit; `cli` holds `drpQa.py`'s `load`,
  `values` and `judge`. Tested in `tests/qadb/` on SQLite; the schema there
  (`tests/qadb/conftest.py`) and `spt_qa_database`'s change together.
- **Support modules (`python/pfs/drp/qa/`)**:
  - `crossDispersion.py` — stack-free trace widths: the joint fit `imageQualityQa` runs on the calexp, and
    `profileFwhm` for `fiberProfiles`; the choice is recorded in `docs/trace-widths.ipynb`
  - `storageClasses.py`, `formatters.py` — custom Butler storage classes / formatters
  - `utils/` — shared helpers (`math.py`; `plotting.py` re-exports `plotting.palettes`;
    `provenance.py` reads which calibrations a dataset was made with from its Butler provenance)
- **Command-line tools (`bin.src/`)** — run as `python bin.src/<name>.py`:
  - `drpQa.py` — one command with a subcommand for each job (`pfs.drp.qa.cli.COMMANDS`): `redo`, `load`,
    `values`, `judge`, `summary`, `calib inputs`, `calib validate`, `calib profiles`, `fiberNorms`,
    `agReport`. A
    subcommand's module is imported only when it is asked for, so `values`, `judge` and `summary`,
    which read only qadb, run without the stack: keep them that way. Add a job as an entry of
    `COMMANDS`, not as a script. `summary`'s exit status is the worst visit verdict (0/1/2, 3 nothing
    judged, 4 could not be read), for a nightly job. `redo` (`comparison/redo.py`) redoes a run end to
    end: reduce, measure, load, make and compare normal values, judge, and point at the dashboard.
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
- `tests/` — pytest-based tests. Tests needing packages
  beyond the standard library live in subdirectories with their own `conftest.py`
  (`tests/guiders/`).
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
prepends `PATH` and `PYTHONPATH`; nothing is compiled or generated.

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
pytest tests/metrics/test_gate.py

# Single test
pytest tests/metrics/test_lineResiduals.py::testZeroOffset
```

`pyproject.toml` sets `addopts = "-ra --import-mode=importlib"`. Note the `importlib`
import mode: test modules are not added to `sys.path`, so they can't import each other
by bare module name.

No test module needs the stack (`_STACK_MODULES` in `tests/conftest.py` is empty).
`tests/test_connections.py` and
`tests/test_drpQaPipeline.py` read the task modules and the pipelines as text and need only
pytest. The tests in the subdirectories of `tests/` need the packages their `conftest.py`
imports (numpy, pandas, pfs_utils, ...). If logic cannot be tested without the LSST/PFS
environment, validate it via unit-testable components and document the constraint in the test,
the update note, or the PR.

### Continuous integration

Two GitHub Actions workflows run on every pull request:

| Workflow | Blocking | What it does |
|---|---|---|
| `.github/workflows/tests.yml` | yes | On Python 3.12 and 3.13: `pytest -v` with only pytest installed, and again with pfs-utils and the scientific packages |
| `.github/workflows/lint.yml` | yes | `ruff check .` and `ruff format --check .` over the whole tree |

**The repository is Ruff-clean and both checks gate the whole tree.** Keep it that way:
fix findings in the code you touch rather than widening the ignore list in
`pyproject.toml`.

**The package is not installed in CI.** `pip install -e .` would pull `pfs-utils` (and
transitively `pfs-datamodel`, `pfs-instdata`) from GitHub, making every run depend on
three other repositories. Only stack-free tests run, in two jobs:

- **`stack-free`** installs only pytest, so its tests import nothing beyond the standard
  library.
- **`deps`** runs every stack-free test against `python/` on `PYTHONPATH`. It installs
  pfs-utils from git with `--no-deps`, plus the packages listed in the workflow. Keep that
  list in step with `pyproject.toml`.

A test needing packages beyond the standard library goes in a subdirectory of `tests/`
whose `conftest.py` imports them at module scope, listed in `_OPTIONAL_DIRS` in
`tests/conftest.py`. That directory is ignored when those imports are missing, so the
`stack-free` job skips it and the `deps` job runs it. `tests/guiders/` is the example: its
`conftest.py` holds the fixtures and test doubles, and its real AG data are parquet files in
`tests/guiders/data/`, small enough for plain git, which `makeGuiderFixtures.py` there
remakes.

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
- Combined/aggregate: `("instrument", "visit", "arm")` (`extractionQaCombined`) — these use
  `multiple=True` inputs to consume outputs from per-detector tasks

### Pipeline YAML

`pipelines/drpQA.yaml` wires tasks together. Individual tasks can be run with the
`#taskName` fragment syntax:

```bash
pipetask run -p pipelines/drpQA.yaml#dmLineResiduals -b /path/to/butler -i input/coll -o output/coll
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
- Draw in `pfs.drp.qa.plotting`, never in the task module: the task binds the figures
  to the storage class, or, as `dmLineResiduals` does, writes data only and leaves the
  plots to be drawn after the fact.

### Shared plotting utilities (`plotting/palettes.py`)

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
7. Record QA decisions in the repository, where the next reader of the QA looks. A limit is one
   entry of `CHECKS` in `metrics/registry.py`; what is measured but not judged is one entry of
   `EXCLUSIONS` there, with its reason. Normal values are rows of qadb's `qa_normal_value`, each
   with who accepted it, when and why. `README.md` (*How QA works*) describes the design, with
   the detail in `docs/`. Not only in Jira or an assistant's memory: a ticket can hold the
   discussion; the repository holds the outcome.
8. Record user-visible changes — new tasks, new or removed config fields, new
   `iqQaMetrics` columns, changed defaults — under `## [Unreleased]` in `CHANGELOG.md`.
9. Include concise docstrings describing purpose, inputs, outputs, and any assumptions —
   especially about LSST data structures.
10. Prefer small, focused PRs with clear descriptions of the change and its QA impact.

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
             reduceExposure_metadata (optional)
    → writes: iqQaMetrics, iqQaSpeciesMetrics; iqQaData only with doWriteLineData   (numbers, no verdict;
              seqType, seqName, illumination and lamps say what the frame is, as read)

drpQA.yaml#dmLineResiduals           dims: (instrument, visit, arm, spectrograph)
    ← reads: lines, detectorMap, raw.metadata, detectorMap_calib (calibration), pfsConfig (optional)
    → writes: dmLineResidualMetrics (long format), dmLineResidualSummary (one row),
              dmLineResidualData (one row per line)   (numbers, no verdict)

drpQA.yaml#twilightFluxQa            dims: (instrument, visit, arm, spectrograph)
    ← reads: pfsArm, raw.metadata, pfsConfig (optional)
    → writes: twilightFiberFlux (twilights only)

fiberProfilesQa.yaml (drpQa.py calib profiles; not in drpQA.yaml)   dims: (instrument, visit, arm, spectrograph)
    ← reads: raw.metadata, fiberProfiles (calibration: the candidate's, first in the inputs), pfsConfig (optional)
    → writes: fiberProfilesQaFibers, fiberProfilesQaMetrics (numbers, no verdict)

drpQA.yaml#extractionQa              → extQaStats, extQaImage, extQaImage_pickle
drpQA.yaml#extractionQaCombined      ← extQaImage_pickle (multiple)
                                     → extQaStatsCombined

drpQa.py load / pfs.drp.qa.qadb.loader.loadCollection   (after pipetask, not a task)
    ← reads: iqQaMetrics, iqQaSpeciesMetrics, dmLineResidualSummary, dmLineResidualMetrics,
             packages (per RUN), pfsArm headers (the reducing pipeline);
             for a twilight visit, the opdb's visit_set and iic_sequence(_status), and once the
             sequence is complete the twilightFiberFlux of its visits;
             qa_normal_value (the collection's normal values)
    → pfs.drp.qa.metrics.gate.gate: one verdict per image
    → writes to qadb: image_quality_qa(_species), dm_line_residual_qa(_metric), twilight_qa,
             data_processing_pipe2d, pfs_visit, qa_gate (idempotent upserts)

drpQa.py values / pfs.drp.qa.qadb.judge.makeValues   (qadb only: no Butler, no stack)
    ← reads: the numbers of the run's calibration visits, qa_normal_value
    → writes, with --accept: qa_normal_value, then qa_gate for every frame the values cover

drpQa.py judge / pfs.drp.qa.qadb.judge.judgeStored   (qadb only)
    ← reads: the numbers of a visit range, qa_normal_value
    → writes: qa_gate
```

`reduceExposure` is **not** required between `fitDetectorMap` and `imageQualityQa`. The
`arcLines` (`lines`) and `detectorMap` outputs from `fitDetectorMap` are read directly.
When `reduceExposure` *has* run, its task metadata is picked up through the optional
`reduceExposure_metadata` connection and its per-species detectorMap fit statistics are
folded into `iqQaSpeciesMetrics`. The whole-detector fit scalars and ISR/cosmic-ray counts
are read from the same metadata by `reductionStats`, not here; no QA task parses a log.

Cross-visit QA is **not** a pipeline task. There is no `imageQualityQaSummary` and no
combined task: normal values, verdicts and the per-visit rollup are made from qadb after the
fact. This keeps them off the critical path of a reduction and lets a visit range be judged again
without a Butler.

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
  `"_eng"` indicate IIS illumination (the pfsConfig is the second witness: its engineering
  fibers all GOOD, `engineeringLit`). Import is guarded so the task gracefully
  degrades if `obs_pfs` is unavailable.
- Line lists live in `obs_pfs/pfs/lineLists/`: `{Ar,Xe,Kr,Ne,HgAr,HgCd}.txt` for the PFI lamps and
  `{Ar,Kr,Ne,HgAr,He,H}_IIS.txt` for the IIS lamps; columns
  `wavelength intensity species status transition source intensityErr`.
