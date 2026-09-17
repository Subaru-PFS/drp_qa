# Changelog

All notable changes to `drp_qa` are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This package does not use semantic versioning. Released versions correspond to the LSST-style weekly tags applied to the
repository (`w.2026.29`, `w.2026.09`, …), so sections below are keyed to those tags rather than to `MAJOR.MINOR.PATCH`.

## [Unreleased]

### Added

- **Golden visit set** (`tests/data/goldenVisits.yaml`) plus a stack-free loader,
  `pfs.drp.qa.metrics.goldenVisits`. `known_good` is the Run25 stable calibration sequence
  (two blocks on 2025-11-10 and one on 2025-12-01) plus two clear twilight-sky sets;
  `known_bad` is the SM1 focus range (140005-140138) and a cloudy twilight set.
  Placeholder entries are excluded by default, so an unfilled entry cannot validate a threshold.
  A third `reference` section holds visits that are tracked but assert no verdict — the Run30 per-run
  calibration block and nightly drift series, whose quality is the question rather than the premise.
  `role` and `epoch` group entries so a cross-run comparison can pair like with like.
- **`pfs.drp.qa.metrics.thresholds`** — the threshold derivation procedure as pure functions: WARN at
  p95 and FAIL at p99 of the known-good distribution, or a physical limit, plus the provenance
  sentence for the config field's `doc`.
- **`bin.src/calibrateQaThresholds.py`** — runs that procedure over a Butler collection or a CSV and
  prints suggested config values. Exits non-zero on too few samples, or when the known-bad data does
  not cross the suggested FAIL.
- **Metric registry** (`pfs.drp.qa.metrics.registry`) — one `MetricDef` per metric carrying its
  units, external reference, direction, thresholds and their provenance; `MetricRegistry.gate` is the
  single gating path. An unmeasured or ungated metric yields no verdict rather than a PASS.
- **`iqQaSpeciesMetrics`** — new `imageQualityQa` output: per-species fit statistics in long format.
- **`tests/test_fitStats.py`** — pins `FitStat`'s field order, which `FitStats.from_dataframe`
  unpacks positionally and which is therefore part of the stored `dmQaResidualStats` schema.
- **AGENTS.md: "Golden Visit Set and Threshold Derivation"**.
- **GitHub Actions CI** — `.github/workflows/tests.yml` runs the stack-free test suite on Python 3.12 and
  3.13; `.github/workflows/lint.yml` runs `ruff check .` and `ruff format --check .` over the whole tree.
  Both are blocking.
- **Repository is now Ruff-clean** — `ruff check .` and `ruff format --check .` both pass, and CI gates
  on them. `examples/` is excluded (out-of-order imports and cross-cell names are inherent to notebooks);
  `E501` is ignored because `ruff format` already enforces `line-length` for code and what remains are
  long regexes and report strings the formatter will not split; `RUF001`–`RUF003` are ignored because
  Greek letters and typographic dashes are intentional in a scientific package.
- **`tests/conftest.py`** — puts `python/` on `sys.path` so the suite imports `pfs.drp.qa.*` from the
  checkout, and drops test modules whose module-scope stack imports cannot be satisfied. New
  stack-dependent modules should use `pytest.importorskip` instead.
- **`imageQualityQa` workflow** writing `iqQaData`/`iqQaMetrics` with per-quantum status, with
  post-hoc time-series plotting via `bin.src/plotIqQaTimeSeries.py`.
- **Stack-free log QA/report tools** (`bin.src/fitDetectorMapLogQa.py`, `bin.src/imageQualityLogQa.py`)
  and associated tests and documentation.
- **`AGENTS.md`** — single source of instructions for AI coding assistants, with
  `CLAUDE.md`, `GEMINI.md`, and `.github/copilot-instructions.md` as symlinks to it.

### Changed

- **Plotting moved to `pfs.drp.qa.plotting`** (`palettes`, `dmResiduals`, `dmCombined`, `iqQa`).
  DataFrames in, `Figure` out; no Butler and no task class, checked statically by
  `tests/test_plotting.py`. `pfs.drp.qa.utils.plotting` and `pfs.drp.qa.iqQaPlots` remain as shims.
- **`plot_detectormap_residuals` takes a `DetectorGeometry`**; a `DetectorMap` is still accepted and
  reduced to one, so callers are unaffected.
- **`make_report` is a thin wrapper** over `plotting.dmCombined.reportFigures`; only the binding to
  `MultipagePdfFigure` stays with the task.
- **`FitStat`/`FitStats` moved to `pfs.drp.qa.metrics.fitStats`**, re-exported from `dmResiduals`.
- **`imageQualityQa` gates through the registry.** Thresholds still come from the config, so
  overrides work as before; verdict boundaries and reason strings are unchanged.
- **Per-species metrics are long-format.** The ragged `fitSpeciesXRms_<species>` columns are gone
  from `iqQaMetrics`; the values are rows in `iqQaSpeciesMetrics`, so quanta with different species
  mixes concatenate without NaN padding.
- **`tests/test_dmResiduals.py` tests something.** Was a `pass` body; now exercises `get_fit_stats`
  against injected defects of known size, guarded by `pytest.importorskip`.
- **CI installs the PyPI wheels the stack-free suite needs** (numpy, pandas, matplotlib, seaborn,
  pyyaml). None pulls in the stack or another PFS repository.
- **Build and packaging** — `pyproject.toml` is now the single source of build, lint, and test configuration. Ruff
  replaces Black, isort, and Flake8; `uv.lock` pins the development environment. EUPS `setup -r .` still works via
  `ups/drp_qa.table`, but there is no longer a build step.
- **Lint and format sweep** — every pre-existing QA module reformatted under Ruff (`line-length = 110`,
  `target-version = "py312"`), including `typing.Union`/`Optional`
  → PEP 604 unions and `typing.Iterable` → `collections.abc.Iterable`. No behaviour changes. LSST camelCase naming is
  preserved; the corresponding pep8-naming rules are in the ignore list.

### Fixed

- **`dmResiduals` import** — `getDescriptionCounts` is now imported from
  `pfs.drp.stella.fitDetectorMap`. The former `pfs.drp.stella.fitDistortedDetectorMap` module no longer
  exists in `drp_stella`, so `DetectorMapResidualsTask` failed to import and the `dmResiduals` pipeline
  task could not run.

### Removed

- **`tests/SConscript`** — the last SCons file; it imported `lsst.sconsUtils` and did nothing.
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
