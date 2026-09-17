# Changelog

All notable changes to `drp_qa` are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This package does not use semantic versioning. Released versions correspond to the LSST-style weekly tags applied to the
repository (`w.2026.29`, `w.2026.09`, …), so sections below are keyed to those tags rather than to `MAJOR.MINOR.PATCH`.

## [Unreleased]

### Added

- **Metric registry** (`pfs.drp.qa.metrics.registry`) — `MetricDef` declares a metric's units, the
  external reference it is measured against (R1), its direction, its thresholds and their provenance
  (R2); `MetricRegistry.gate` is the one gating path for every metric, replacing the per-metric
  if/elif ladders. A value that was not measured, or a metric with no thresholds, yields no verdict
  rather than a PASS.
- **`iqQaSpeciesMetrics`** — new `imageQualityQa` output holding per-species fit statistics in long
  format, one row per `(visit, arm, spectrograph, description, metric)`.

### Changed

- **`imageQualityQa` gates through the metric registry.** Thresholds still come from the config, so
  command-line overrides work as before, and the verdict boundaries and reason strings are unchanged.
- **Per-species metrics are long-format.** The ragged `fitSpeciesXRms_<species>` /
  `fitSpeciesYRms_<species>` columns are gone from `iqQaMetrics`; the same values are now rows in the
  new `iqQaSpeciesMetrics` dataset. Concatenating across quanta that saw different species no longer
  produces a NaN-padded frame, and `groupby("description")` no longer needs the species known in
  advance.

- **Golden visit set** (`tests/data/goldenVisits.yaml`) — a fixed list of visits with known verdicts,
  against which every threshold and every new metric is validated. Loaded with
  `pfs.drp.qa.metrics.goldenVisits.loadGoldenVisits`, which is stack-free and Butler-free. Entries
  marked `placeholder: true` are excluded by default so an unfilled entry cannot silently validate a
  threshold. The documented SM1 focus range (visits 140005-140138) is the anchor `known_bad` entry.
- **`pfs.drp.qa.metrics.thresholds`** — the threshold derivation procedure as pure functions
  (`deriveThresholds`, `verifyKnownBad`, `formatProvenance`, `roundToReadable`): WARN at p95 and
  FAIL at p99 of the known-good distribution, or a physical limit where one exists, with the
  provenance sentence required for the config field's `doc` string.
- **`bin.src/calibrateQaThresholds.py`** — CLI that runs the procedure over a Butler collection (or
  an exported CSV) and prints suggested config values. Exits non-zero when a suggestion rests on
  fewer than 20 samples or when the known-bad data does not cross the suggested FAIL.
- **AGENTS.md: "Golden Visit Set and Threshold Derivation"** — documents the procedure that every
  threshold must follow.

- **GitHub Actions CI** — `.github/workflows/tests.yml` runs the stack-free test suite on Python 3.12 and
  3.13; `.github/workflows/lint.yml` runs `ruff check .` and `ruff format --check .` over the whole tree.
  Both are blocking.
- **Repository is now Ruff-clean** — `ruff check .` and `ruff format --check .` both pass, and CI gates
  on them. `examples/` is excluded (out-of-order imports and cross-cell names are inherent to notebooks);
  `E501` is ignored because `ruff format` already enforces `line-length` for code and what remains are
  long regexes and report strings the formatter will not split; `RUF001`–`RUF003` are ignored because
  Greek letters and typographic dashes are intentional in a scientific package.
- **`tests/conftest.py`** — skips test modules that import the LSST/PFS stack at module scope when the
  stack is unavailable, so the stack-free suite can be collected and run in CI. Add new stack-dependent
  modules to `_STACK_MODULES`.
- Added a new `imageQualityQa` workflow that writes `iqQaData`/`iqQaMetrics` with per-quantum status and supports
  post-hoc time-series plotting via `iqQaPlots` and `bin.src/plotIqQaTimeSeries.py`.
- Added stack-free log QA/report tools (`bin.src/fitDetectorMapLogQa.py`, `bin.src/imageQualityLogQa.py`) and associated
  tests/documentation for the image-quality pipeline.
- **`AGENTS.md`** — single source of instructions for AI coding assistants, with
  `CLAUDE.md`, `GEMINI.md`, and `.github/copilot-instructions.md` as symlinks to it.

### Fixed

- **`dmResiduals` import** — `getDescriptionCounts` is now imported from
  `pfs.drp.stella.fitDetectorMap`. The former `pfs.drp.stella.fitDistortedDetectorMap` module no longer
  exists in `drp_stella`, so `DetectorMapResidualsTask` failed to import and the `dmResiduals` pipeline
  task could not run.

### Changed

- **Build and packaging** — `pyproject.toml` is now the single source of build, lint, and test configuration. Ruff
  replaces Black, isort, and Flake8; `uv.lock` pins the development environment. EUPS `setup -r .` still works via
  `ups/drp_qa.table`, but there is no longer a build step.
- **Lint and format sweep** — every pre-existing QA module reformatted under Ruff (`line-length = 110`,
  `target-version = "py312"`), including `typing.Union`/`Optional`
  → PEP 604 unions and `typing.Iterable` → `collections.abc.Iterable`. No behaviour changes. LSST camelCase naming is
  preserved; the corresponding pep8-naming rules are in the ignore list.

### Removed

- **Log-artifact tests** — the `TestRealLogs` class in `tests/test_fitDetectorMapLogQa.py` depended on
  `run28-dm-02.log` / `run28-dm-03.log`, which are not in the repository, so all eight tests always
  skipped and provided no coverage.

- **SCons build** — `SConstruct`, `bin.src/SConscript`, and `ups/drp_qa.cfg`. The `bin/`
  directory is no longer generated; scripts are run as `python bin.src/<name>.py`.
- **`setup.cfg`** and **`mypy.ini`** — superseded by `pyproject.toml`. No static type checker is configured for this
  repository.
- **Unused `database` pytest marker** — the marker declaration and the `-m 'not database'` filter in
  `addopts` are removed from `pyproject.toml`. No test in the repository was ever marked with it, so the
  filter was a no-op.
