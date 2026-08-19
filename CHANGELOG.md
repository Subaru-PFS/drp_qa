# Changelog

All notable changes to `drp_qa` are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This package does not use semantic versioning. Released versions correspond to the LSST-style weekly tags applied to the
repository (`w.2026.29`, `w.2026.09`, …), so sections below are keyed to those tags rather than to `MAJOR.MINOR.PATCH`.

## [Unreleased]

### Added

- Added a new `imageQualityQa` workflow that writes `iqQaData`/`iqQaMetrics` with per-quantum status and supports
  post-hoc time-series plotting via `iqQaPlots` and `bin.src/plotIqQaTimeSeries.py`.
- Added stack-free log QA/report tools (`bin.src/fitDetectorMapLogQa.py`, `bin.src/imageQualityLogQa.py`) and associated
  tests/documentation for the image-quality pipeline.
- **`AGENTS.md`** — single source of instructions for AI coding assistants, with
  `CLAUDE.md`, `GEMINI.md`, and `.github/copilot-instructions.md` as symlinks to it.

### Changed

- **Build and packaging** — `pyproject.toml` is now the single source of build, lint, and test configuration. Ruff
  replaces Black, isort, and Flake8; `uv.lock` pins the development environment. EUPS `setup -r .` still works via
  `ups/drp_qa.table`, but there is no longer a build step.
- **Lint and format sweep** — every pre-existing QA module reformatted under Ruff (`line-length = 110`,
  `target-version = "py312"`), including `typing.Union`/`Optional`
  → PEP 604 unions and `typing.Iterable` → `collections.abc.Iterable`. No behaviour changes. LSST camelCase naming is
  preserved; the corresponding pep8-naming rules are in the ignore list.

### Removed

- **SCons build** — `SConstruct`, `bin.src/SConscript`, and `ups/drp_qa.cfg`. The `bin/`
  directory is no longer generated; scripts are run as `python bin.src/<name>.py`.
- **`setup.cfg`** and **`mypy.ini`** — superseded by `pyproject.toml`. No static type checker is configured for this
  repository.
