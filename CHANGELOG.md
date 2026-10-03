# Changelog

All notable changes to `drp_qa` are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This package does not use semantic versioning. Released versions correspond to the LSST-style weekly tags applied to the
repository (`w.2026.29`, `w.2026.09`, …), so sections below are keyed to those tags rather than to `MAJOR.MINOR.PATCH`.

## [Unreleased]

### Added

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
  replaces Black, isort, and Flake8. EUPS `setup -r .` still works via
  `ups/drp_qa.table`, but there is no longer a build step.
- **Lint and format sweep** — every pre-existing QA module reformatted under Ruff (`line-length = 110`,
  `target-version = "py312"`), including `typing.Union`/`Optional`
  → PEP 604 unions and `typing.Iterable` → `collections.abc.Iterable`. No behaviour changes. LSST camelCase naming is
  preserved; the corresponding pep8-naming rules are in the ignore list.
- **`pfs` and `pfs.drp` are PEP 420 namespace packages** — the `pkgutil`-style `python/pfs/__init__.py` and
  `python/pfs/drp/__init__.py` are removed (and no longer ship in the wheel), matching `drp_stella`, `pfs_utils` and
  `datamodel`. Module names are unchanged (PIPE2D-1904).
- **`FluxCalQA` takes the databases** — `FluxCalQA(butler, opdb, qadb, ...)` now requires a
  `pfs.utils.database` `OpDB` and `QaDB` and queries them with bound parameters, replacing its own
  `psycopg2.connect` calls. Needs `pfs_utils` 7.4.18 or later. `main()` still reads the opdb on `pfsa-db`
  (PIPE2D-1893).

### Removed

- **`uv.lock`** — it had gone stale (`uv lock --check` failed) and nothing used it: CI installs its packages
  without it, and the code runs in the LSST stack. It is now in `.gitignore`; `uv sync` still works and writes a
  local lockfile (PIPE2D-1906).
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
