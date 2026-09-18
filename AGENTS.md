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
6. [Golden Visit Set and Threshold Derivation](#golden-visit-set-and-threshold-derivation)
7. [Git Commit Convention](#git-commit-convention)
8. [Domain Knowledge: `imageQualityQa`](#domain-knowledge-imagequalityqa)
9. [Arc Lamp Physics and b-arm `pctFlagged` Failures](#arc-lamp-physics-and-b-arm-pctflagged-failures)
10. [Butler / Pipeline Data Flow for DM & IQ QA](#butler--pipeline-data-flow-for-dm--iq-qa)
11. [Common Failure Patterns](#common-failure-patterns)
12. [Cross-repo Dependency Notes](#cross-repo-dependency-notes)

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
  - `imageQualityQa.py` — image quality (FWHM, flag rates); plots in `plotting/iqQa.py`
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
- **Support modules (`python/pfs/drp/qa/`)**:
  - `storageClasses.py`, `formatters.py` — custom Butler storage classes / formatters
  - `utils/` — shared helpers (`math.py`, `plotting.py`)
  - `tasks/` — auxiliary task helpers (e.g. `overlapRegionLines.py`)
- **Command-line tools (`bin.src/`)** — run as `python bin.src/<name>.py`:
  - `fitDetectorMapLogQa.py` — stdlib-only OK/WARN/BAD gate over `fitDetectorMap` logs;
    exits non-zero on BAD. Keep it stack-free.
  - `imageQualityLogQa.py` — per-visit report from `reduceExposure`/`imageQualityQa`
    logs or a direct Butler query; dashboard plot, markdown report, JSON dump
  - `plotIqQaTimeSeries.py` — cross-visit `iqQaMetrics` time series via `plotting/iqQa.py`
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
- `tests/` — pytest-based tests. Most run without the LSST/PFS stack; the ones that need
  it guard themselves with `pytest.importorskip` (see below).
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
SCons build** — `SConstruct`, `bin.src/SConscript` and `tests/SConscript` were all
removed. `setup -r .` only prepends `PATH` and `PYTHONPATH`; nothing is compiled or
generated.

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
| `.github/workflows/tests.yml` | yes | `pytest -v` on Python 3.12 and 3.13 |
| `.github/workflows/lint.yml` | yes | `ruff check .` and `ruff format --check .` over the whole tree |

**The repository is Ruff-clean and both checks gate the whole tree.** Keep it that way:
fix findings in the code you touch rather than widening the ignore list in
`pyproject.toml`.

**The package is not installed in CI.** `pip install -e .` would pull `pfs-utils` (and
transitively `pfs-datamodel`, `pfs-instdata`) from GitHub, making every run depend on
three other repositories. CI installs only ordinary PyPI wheels (`pytest`, `numpy`,
`pandas`, `scipy`, `matplotlib`, `seaborn`, `pyyaml`) and `tests/conftest.py` puts `python/` on
`sys.path`, so the stack-free suite imports `pfs.drp.qa.*` straight from the checkout.

**Tests that need the stack** cannot be collected without it — a module-level
`import lsst.utils.tests` fails during collection and aborts the whole run, so an in-test
`try/except ImportError` never gets the chance to skip. Guard such a module at module
level instead:

```python
dmResiduals = pytest.importorskip("pfs.drp.qa.dmResiduals", reason="requires the LSST/PFS stack")
```

`tests/conftest.py` also keeps a `_STACK_MODULES` list that drops modules from collection
when their module-scope imports cannot be satisfied. It is empty today and is the escape
hatch for a module that genuinely cannot use `importorskip` — one subclassing
`lsst.utils.tests.TestCase`, say. Prefer `importorskip`, which keeps the guard next to the
import it guards.

**Better than either:** keep the logic under test in pure functions that take arrays and
DataFrames, so no stack is needed at all. That is what `pfs.drp.qa.metrics` and
`pfs.drp.qa.plotting` exist for.

**No `pass` test bodies.** A test that asserts nothing reports green and hides the defect
it was named after. A metric test should inject a defect of known size and assert the
metric recovers it.

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

### Metrics and gating: the registry

Tasks emit numbers; a separate layer turns numbers into `PASS`/`WARN`/`FAIL`. That layer
is `pfs.drp.qa.metrics.registry`, and it is stack-free and Butler-free so it can be
unit-tested in CI.

A metric declares itself once as a `MetricDef`: its `units`, the external `reference` it
is measured against (R1), `higherIsWorse`, its `thresholds` (with optional per-key
`overrides` for `arm` / `arm:species`), and the `provenance` of those thresholds (R2).
Gating is then `MetricRegistry.gate`, the same code for every metric, and the reason
string is generated rather than written per metric.

Two conventions that matter:

- **The thresholds live in the task's config, not in the registry module.** A task builds
  its registry from the config in force — see
  `pfs.drp.qa.metrics.definitions.buildImageQualityRegistry` — so `-c` overrides still
  take effect. The registry module holds the metadata; the config holds the numbers.
- **`gate` returns `None`, not `PASS`, when there is no verdict to give** — a NaN value,
  or a metric with no thresholds. `worstStatus` skips those. An unmeasured or ungated
  metric must never turn a bad quantum green. A task that wants to report "we could not
  measure at all" does so itself, when *every* gate returned `None`.

Do not add a per-metric if/elif ladder to a task. If gating needs something the registry
cannot express, extend `MetricDef`.

### Plot outputs

- Single-figure tasks: return a `matplotlib.figure.Figure` in the `Struct`; Butler stores
  it via `storageClass="Plot"` using `PdfMatplotlibFormatter`.
- Multi-page tasks: return a `MultipagePdfFigure`; call `.append(fig)` for each page.

### Plotting lives in `pfs.drp.qa.plotting`

Every plotting function takes DataFrames (and plain numbers) and returns a
`matplotlib.figure.Figure`. **None of them may import the Butler or a task class** —
`tests/test_plotting.py` checks this statically over the whole subpackage, so the rule
cannot quietly rot. That constraint is what gives three consumers (dashboard, on-demand
report, notebooks) one implementation, and what makes the plots testable at all.

| Module | Contents |
|---|---|
| `plotting/palettes.py` | `div_palette`, `detector_palette`, `description_palette`, `spectrograph_plot_markers`, `scatterplot_with_outliers` |
| `plotting/dmResiduals.py` | `DetectorGeometry`, `plot_detectormap_residuals`, `plot_residual` |
| `plotting/dmCombined.py` | `plot_detector_summary`, `plot_detector_summary_per_desc`, `plot_visits`, `plot_title`, `reportFigures` |
| `plotting/iqQa.py` | `plotIqTimeSeries` |

Two consequences worth knowing:

- **`plot_detectormap_residuals` takes a `DetectorGeometry`**, not a `DetectorMap`. A
  `DetectorMap` is still accepted and reduced to one via `DetectorGeometry.coerce`, so
  existing callers are unaffected — but passing the geometry is what lets the function be
  called without the stack.
- **Assembling figures into a Butler artifact is the task's job.** `reportFigures` yields
  pages; `dmCombinedResiduals.make_report` binds them to `MultipagePdfFigure`. Do not push
  a storage class back into the plotting package.

`pfs.drp.qa.utils.plotting` and `pfs.drp.qa.iqQaPlots` remain as re-export shims for
notebooks and external callers; prefer the new paths in new code.

### Shared plotting utilities (`utils/plotting.py` — re-export shim)

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

## Golden Visit Set and Threshold Derivation

**No threshold enters the codebase before it has been computed from a known-good visit
range.** A number invented at the keyboard cannot be defended when it fires at 03:00, and
a metric compared against a threshold derived from its own sample is pinned near a
constant and measures nothing. See `doc/qa-rebuild-plan.md` sections 1.1 and 1.2.

### The golden visit set

`tests/data/goldenVisits.yaml` is a small, fixed list of visits with known verdicts:
`known_good` entries must PASS every metric, `known_bad` entries must WARN or FAIL, and
each `known_bad` entry names the `metric` whose value identifies the fault. The file's
own header documents the entry schema.

Load it with `pfs.drp.qa.metrics.goldenVisits.loadGoldenVisits`, which is stack-free and
Butler-free:

```python
from pfs.drp.qa.metrics.goldenVisits import loadGoldenVisits

golden = loadGoldenVisits()
golden.expectationFor(140005, arm="b", spectrograph=1)   # -> "FAIL"
golden.expectationFor(999999)                            # -> None, i.e. no expectation
```

Two properties to keep in mind:

- **Absence is not a PASS.** `expectationFor` returns `None` for a visit the set does not
  cover. Never treat that as an expectation.
- **Placeholders are excluded by default.** Entries marked `placeholder: true` have no
  real visit number yet; the loader drops them unless `includePlaceholders=True` is
  passed, so an unfilled entry can never silently validate a threshold.
- **A suspected verdict is not a verdict.** `unconfirmed: true` marks a real visit that
  was flagged at the telescope but whose metrics nobody has checked. It is loaded and
  reported like any other entry — `calibrateQaThresholds.py` prints where its values land
  relative to the suggested FAIL — but `GoldenVisitSet.confirmedBad` leaves it out, so it
  never decides a pass/fail. A guess that fails the build is worse than no entry. Once the
  numbers have been compared, drop the flag and name the metric, or move the visit to
  `known_good` if it turns out fine.

**Every threshold and every new metric is validated against this set.** A metric that
flags a `known_good` visit, or passes a `known_bad` one, does not merge.

**Every entry carries a verdict, and only entries with verdicts belong.** A visit earns
its place because a metric that gets it wrong must not merge. Visits with no verdict — a
per-run calibration block, a nightly drift series, the inputs to a run-to-run comparison —
do not go here. Select those by querying the Butler on sequence type, lamp and date when
the job runs: that stays correct for the next run without anyone editing the file, whereas
a transcribed list is stale the moment observing continues.

### What the shipped thresholds actually rest on

Nothing written down. The `medFwhm` (3.2/3.5), `medDxCenter` (1.0/2.0) and `pctFlagged`
values arrived with `imageQualityQa` itself in commit `8a69c05` (2026-08-11), and no
derivation was recorded — not in the commit message, which explains the threshold
*structure* at length but not the numbers; not in the field docs beyond "Tuned for
arm-b"; not in `README.md` or this file, which restate them without justifying them.

The per-species *shape* of the flag-rate thresholds is well justified: Ar, Xe and Kr have
almost no usable blue lines, so a single b-arm threshold cannot work. The specific
percentages are not.

So treat all of them as R2 violations inherited rather than introduced, and re-derive
them. The registry's provenance strings say `ORIGIN UNRECORDED` rather than inventing a
plausible history — a guess dressed as a fact is worse than an admitted gap, because only
one of them prompts anyone to fix it.

### Threshold derivation procedure

Follow this for every threshold:

1. Run the metric over the `known_good` visits with **no gating**.
2. Take the distribution of the metric across all good detectors.
3. `WARN` at the 95th percentile, `FAIL` at the 99th, rounded to a readable value — or,
   where a physical limit exists (saturation, fiber pitch), use the physical limit.
4. Verify the `known_bad` visits exceed `FAIL`.
5. Record the visit range and the derivation date in the config field's `doc` string.

`bin.src/calibrateQaThresholds.py` does steps 1-4 and prints the provenance sentence for
step 5:

```bash
python bin.src/calibrateQaThresholds.py -b /path/to/butler -c u/you/qa/run12 \
    --metric medFwhm --metric pctFlagged --group-by arm
```

It exits non-zero when a suggestion rests on too few samples (< 20) or when the
known-bad data does not cross the suggested `FAIL` — either way, do not commit the
numbers *as though they were derived*.

It still prints them, which is deliberate: an underpowered number is worth having as long
as nobody mistakes it for a measurement. Its provenance sentence carries a `NOT RELIABLE`
clause naming the sample count and the floor, so the caveat travels with the value into
the config field rather than living only in the terminal output you saw once. The `m`-arm
trace threshold is the standing example — 16 detectors, and no more m-arm quartz in Run25
to be had. A known_bad entry that names `medFwhm` is only checked against `medFwhm`;
it asserts nothing about the other metrics.

Group with `--group-by arm` (and `--group-by description` once metrics are long-format)
wherever the physics differs per group. **Filter before grouping** when a group mixes
populations: `--group-by arm` alone puts arcs and quartz together, and quartz FWHM from the
fiber-profile fallback is read from a *calibration*, so it is the same value every visit and
lands in the tail. `--filter "traceOnly == False"` keeps the arc threshold an arc threshold. Blue-arm flag rates are the standing example: a
single blended threshold is dominated by the species mix, not by the instrument.

The pure functions behind the CLI live in `pfs.drp.qa.metrics.thresholds`
(`deriveThresholds`, `verifyKnownBad`, `formatProvenance`, `roundToReadable`) and are
unit-tested without the stack in `tests/test_thresholds.py`.

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
statistics (`fitChi2`, `fitXRms`, `fitYRms`, `fitReserved*`, per-fiber arrays).

**Per-species values are not columns here.** They go to the separate
`iqQaSpeciesMetrics` dataset in long format — one row per
`(visit, arm, spectrograph, description, metric)`, with columns `value` and `status`.
The old `fitSpeciesXRms_<species>` / `fitSpeciesYRms_<species>` columns made the column
set vary per quantum, so concatenating across quanta gave a ragged NaN-padded frame and
every `groupby` had to know the species in advance. Build the rows with
`pfs.drp.qa.metrics.longFormat.longRecords` / `toLongFrame`; `widen()` gives a wide view
when an operator wants one.

### DM residual metrics

`dmResiduals` writes `dmQaResidualData`, `dmQaResidualStats` and `dmQaResidualPlot`.
`get_fit_stats` builds `dmQaResidualStats` **per `(status_type, description)`** — that
is, separately for `RESERVED` and `USED` lines of each species — via the `FitStats` /
`FitStat` dataclasses. Each row carries `dof`, `chi2X`, `chi2Y`, and a `spatial.` and
`wavelength.` block of `median`, `robustRms`, `weightedRms`, `softenFit`, `dof`,
`num_fibers`, `num_lines`. The RMS values are **error-weighted** (`getWeightedRMS`) with
a robust variant (`robustRms`); `softenFit` is solved by bisection. Prefer these over
adding parallel unweighted metrics.

`dmCombinedResiduals` aggregates across detectors into `dmQaDetectorStats` and renders a
multi-page `dmQaCombinedResidualPlot` via `make_report`, which is now a thin wrapper that
binds the pages yielded by `pfs.drp.qa.plotting.dmCombined.reportFigures` to the Butler
storage class.

The task writes **data only**. Plotting lives in `pfs.drp.qa.plotting` and is driven after the
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

## Butler / Pipeline Data Flow for DM & IQ QA

```
detectorMap.yaml#fitDetectorMap
    → outputs: detectorMap, lines (=arcLines)

drpQA.yaml#imageQualityQa            dims: (instrument, visit, arm, spectrograph)
    ← reads: arcLines, detectorMap,
             fiberProfiles, detectorMap_calib (calibrations),
             calexp, pfsConfig (optional),
             isr_log, cosmicray_log, reduceExposure_log (optional)
    → writes: iqQaData, iqQaMetrics, iqQaSpeciesMetrics

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
    → writes: time-series PNG via pfs.drp.qa.plotting.plotIqTimeSeries
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
