# Changelog

All notable changes to `drp_qa` are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This package does not use semantic versioning. Released versions correspond to the LSST-style weekly tags applied to the
repository (`w.2026.29`, `w.2026.09`, …), so sections below are keyed to those tags rather than to `MAJOR.MINOR.PATCH`.

## [Unreleased]

### Added

- **`DmDriftMonitorTask`** (`dmDriftMonitor` PipelineTask) — new per-detector task that compares
  daily trace and neon arc centroids against the static detectormap (one per observation run) and
  writes `dmDriftMetrics` DataFrame with `deltaX`, `deltaY`, `driftMag`, `deltaWx`, `qaStatus`,
  and `recommendedAction` (NOMINAL / APPLY_SHIFT / RECALIBRATE).
- **`imageQualityQa` extended** — `iqQaMetrics` gains two new columns: `fluxJitterPct`
  (`fluxStd/medFlux*100` for non-flagged arc lines) and `nSaturated` (lines above the 95th-percentile
  flux threshold). Both feed into the existing `qaStatus` worst-of evaluation. New config fields:
  `maxFluxJitterPct` (default 15.0) and `maxSaturatedLines` (default 50).
- **`dmResiduals` extended** — `dmQaResidualStats` gains: `lineYieldFrac`, `spatialRms` (px),
  `wavelengthRms` (nm), `velocityRms` (km/s), `medResolution`, `minFiberPitch` (px), `maxCrossTalk`,
  and `qaStatus`. New config threshold fields added to `DetectorMapResidualsConfig`.
- **`dmCombinedResiduals` extended** — `dmQaDetectorStats` now aggregates the new per-detector DM
  columns and optionally joins `iqQaMetrics` (IQ gate results) via a new optional `multiple=True`
  input connection. The `dmQaCombinedResidualPlot` output has been removed (DataFrame-only outputs).
- **`drpQA.yaml`** — `dmDriftMonitor` task added to the pipeline graph.
- **`bin.src/qaVisitSummary.py`** — new CLI tool and Python API (`VisitSummary` / `VisitSummaryResult`)
  for querying per-visit QA results from the Butler. Prints a formatted summary of `dmQaDetectorStats`,
  `iqQaMetrics`, `dmQaResidualStats`, and `dmDriftMetrics`; supports multiple visits and file output.
  Exit code reflects the worst `qaStatus` across all datasets (0=PASS, 1=WARN, 2=FAIL, 3=no data).
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
