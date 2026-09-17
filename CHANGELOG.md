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
  Every entry carries a verdict: visits with none — a per-run calibration block, a drift series, the
  inputs to a run-to-run comparison — are selected by querying the Butler when the job runs, not
  transcribed here. An entry marked `unconfirmed: true` carries a suspected verdict that nobody has
  checked: it is loaded and reported, but excluded from the check that a threshold separates the
  known-bad data, so a guess cannot fail a build.
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

- **Partial flag-rate overrides no longer change verdicts.** A species key present in only one of
  `flagRateWarnThreshold` / `flagRateFailThreshold` now resolves the missing side through its arm
  entry before the global fallback, as the task's original lookup did. Previously
  `warn={"b": 50, "b:Argon": 93}` with `fail={"b": 60}` gave `b:Argon` a FAIL of 20 rather than 60.
- **Threshold provenance records the quantiles actually used.** For a metric with
  `higherIsWorse=False` the thresholds come from p5/p1; the provenance said p95/p99.
- **`calibrateQaThresholds.py` no longer reports success without producing a threshold.** A metric
  named on the command line but absent from the data is an error, long-format input is pivoted so
  `iqQaSpeciesMetrics` can be calibrated against, a `verifyKnownBad` failure sets the exit status,
  and metrics for which step 4 never ran are listed explicitly instead of passing silently.
- **`imageQualityLogQa.py` reads per-species residuals again.** It reconstructed them from the
  `fitSpeciesXRms_*` columns that moved to `iqQaSpeciesMetrics`, so Butler-sourced reports silently
  omitted every species; it now merges that dataset, keeping the legacy path for older collections.
- **`get_fit_stats` survives `dof <= 0`.** The softening solve divided by zero and handed
  `scipy.optimize.bisect` a NaN endpoint, which raises. Matches the guard in `drp_stella`'s
  `calculateSoftening`.
- **The golden-visit loader rejects non-boolean `placeholder` / `unconfirmed`.** `placeholder:
  "false"` is a string, and truthiness silently dropped the entry.
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
