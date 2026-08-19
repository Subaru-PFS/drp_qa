# Changelog

All notable changes to `drp_qa` are recorded here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This package does not use semantic versioning. Released versions correspond to the
LSST-style weekly tags applied to the repository (`w.2026.29`, `w.2026.09`, …), so
sections below are keyed to those tags rather than to `MAJOR.MINOR.PATCH`.

## [Unreleased]

### Added

- **`imageQualityQa` PipelineTask** (`python/pfs/drp/qa/imageQualityQa.py`) —
  per-detector image quality QA on `(instrument, visit, arm, spectrograph)`. Replaces the
  interactive `showImageQuality` function from `drp_stella`. Classifies each visit from
  `W_SEQTYP` / `W_SEQNAM` / lamp headers and routes it to one of three measurement paths
  (arc-line second moments, `calexp` cross-dispersion moments, or fiber profile
  calibration widths), writing `iqQaData` and `iqQaMetrics`.
- **Per-quantum `PASS` / `WARN` / `FAIL` status** in `iqQaMetrics.qaStatus`, from absolute
  thresholds on `medFwhm`, `pctFlagged`, and `|medDxCenter|`. The reason for a non-`PASS`
  verdict is logged on the `IQ QA` line.
- **Flexure diagnostics** — `dxCenter` per line, and `medDxCenter` / `dxCenterRms` in the
  metrics, measured against the static `detectorMap_calib`.
- **Per-lamp flag rate thresholds** — `flagRateWarnThreshold` and `flagRateFailThreshold`
  accept `arm:species` keys (e.g. `b:Argon`) in addition to plain arm letters, so lamps
  with few blue lines are not failed for lamp physics. Lookup order is
  `arm:species` → `arm` → default.
- **Flag breakdown metrics** — `pctNotVisible`, `pctBlend`, `pctSuspect`, `pctRejected`,
  `pctBroad`, plus `pctLowSN` and `pctMeasFail` on the arc-line path, separating
  "never measured" from "measured and rejected".
- **Log-derived metrics** — optional `isr_log`, `cosmicray_log`, and `reduceExposure_log`
  connections are parsed for ISR bad pixels, cosmic-ray counts, runtimes, and
  `fitDetectorMap` chi²/RMS/softening statistics, including per-species and per-fiber
  breakdowns.
- **`pfs.drp.qa.iqQaPlots`** — time-series plotting for `iqQaMetrics` across visits:
  FWHM, flexure, flag breakdown, and a pass/fail status heatmap.
- **`bin.src/plotIqQaTimeSeries.py`** — CLI over `iqQaPlots.plotIqTimeSeries`, reading
  `iqQaMetrics` from a Butler collection or a CSV, with arm / spectrograph / obs-type
  filters.
- **`bin.src/fitDetectorMapLogQa.py`** — stdlib-only QA gate over `fitDetectorMap` log
  files. Reports `OK` / `WARN` / `BAD` per quantum from fit RMS, softening, line counts,
  and cross-spectrograph peer comparison; exits non-zero on `BAD`. Accepts both the old
  log format and the current `arm=`/`spectrograph=`-prefixed one.
- **`bin.src/imageQualityLogQa.py`** — per-visit QA report built from `reduceExposure` and
  `imageQualityQa` logs or from a direct Butler query, producing a diagnostic dashboard,
  a markdown report, and a replayable JSON dump.
- **`tests/test_fitDetectorMapLogQa.py`** — log-parser and threshold-assessment tests that
  run without the LSST stack.
- **`AGENTS.md`** — single source of instructions for AI coding assistants, with
  `CLAUDE.md`, `GEMINI.md`, and `.github/copilot-instructions.md` as symlinks to it.

### Changed

- **Build and packaging** — `pyproject.toml` is now the single source of build, lint, and
  test configuration. Ruff replaces Black, isort, and Flake8; `uv.lock` pins the
  development environment. EUPS `setup -r .` still works via `ups/drp_qa.table`, but
  there is no longer a build step.
- **Lint and format sweep** — every pre-existing QA module reformatted under Ruff
  (`line-length = 110`, `target-version = "py312"`), including `typing.Union`/`Optional`
  → PEP 604 unions and `typing.Iterable` → `collections.abc.Iterable`. No behaviour
  changes. LSST camelCase naming is preserved; the corresponding pep8-naming rules are in
  the ignore list.
- **`imageQualityQa` writes data only.** Plot generation was removed from the task and
  moved to `iqQaPlots` / `plotIqQaTimeSeries.py`, so QA output no longer depends on the
  pipeline run.

### Removed

- **`ImageQualityCombinedQaTask`** — the instrument-level aggregating task and its
  `iqQaCombinedPlot` output. Cross-visit views are now produced after the fact from
  `iqQaMetrics` by `plotIqQaTimeSeries.py`.
- **`iqQaPlot`** output and the 14 `show*` / colour-scale plot config fields on
  `imageQualityQa`, following the move to post-hoc plotting.
- **SCons build** — `SConstruct`, `bin.src/SConscript`, and `ups/drp_qa.cfg`. The `bin/`
  directory is no longer generated; scripts are run as `python bin.src/<name>.py`.
- **`setup.cfg`** and **`mypy.ini`** — superseded by `pyproject.toml`. No static type
  checker is configured for this repository.
