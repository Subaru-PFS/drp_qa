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
- **Validation visit set and threshold derivation** (PIPE2D-1914) — `pfs.drp.qa.metrics.validationVisits`
  loads `data/validationVisits.yaml`, visits with known verdicts, each assigned to an observing run;
  `metrics.calibration.calibrate` derives WARN/FAIL from the reference run (Run25) per population (arm and `obsType`; flag rates per species) from the known-good `iqQaMetrics`,
  rounded outwards to a step set by the scatter, with a distribution-free interval on FAIL, and checks
  each known-bad visit against its expected verdict; `compareRuns` judges the other runs' known-good data
  against those thresholds; `writeThresholds`/`readThresholds` keep them as a versioned YAML file for the
  judgement step; `metrics.readers.readMetrics` reads the metrics from
  a Butler; `plotting.plotThresholds` draws the result. `docs/qa-thresholds.ipynb` is the step-by-step
  procedure, from reducing the visits (`pipelines/qaThresholds.yaml`, with `drpActor`'s per-sequence-type
  config) to review; `docs/validation-visits.md` describes the set, and `docs/qa-principles.md` holds rules
  R1–R7 and the metric checklist. `pyyaml` is now a dependency.
- **One gating path for `imageQualityQa`** (PIPE2D-1915) — `pfs.drp.qa.metrics.gate.gate` judges `iqQaMetrics`
  rows against layered threshold tables: a `writeThresholds` file first, then the config. Each image gets
  `qaStatus` with the metric that decided it (`qaDecidedBy`, new column) and the reasons (`qaReason`, new
  column). A NaN value gets no verdict; an infinite one is judged. New config field `thresholdsFile` (default
  empty: verdicts unchanged). `pfs.drp.qa.metrics.registry` declares each metric's direction, units,
  reference and populations once, for derivation and gating; `MetricSpec` and `METRIC_SPECS` moved there from
  `calibration`, which re-exports them.
- **`imageQualityQa` can fail a quartz, and says when it judged nothing** (PIPE2D-1917) — `qaStatus` is
  `UNKNOWN`, not `PASS`, when no metric was judged. A quartz whose calexp measurement is unusable falls back to
  the `fiberProfiles` widths, as one without a calexp did; such a FWHM (`traceOnly`) is not a measurement, so
  `MetricSpec.notMeasuredWhen` keeps it out of gating and derivation, and its `pctFlagged` (zero by construction)
  is `NaN`. `medFwhm` on a quartz measured from its calexp is judged per arm by a thresholds file's
  `obsType=trace` entries. `nLines` now counts the rows of `lines` on every path. New `obsType` values
  `twilight` and `dark` (were `science`), neither measured. `imageQualityLogQa.py` tells `UNKNOWN` from a
  missing result (`NOT_RUN`).
- **`iqQaSpeciesMetrics`** (PIPE2D-1915) — the per-species `fitDetectorMap` statistics, in long format
  (`pfs.drp.qa.metrics.longFormat`), replacing the `fitSpeciesXRms_<species>`/`fitSpeciesYRms_<species>`
  columns of `iqQaMetrics`, which are gone. `imageQualityLogQa.py` reads it.
- **`pfs.drp.qa.guiders`** — empty `coordinates`, `queries`, `analysis` and `plotting` modules for the
  guider tools moving from `pfs.drp.stella.utils.guiders`, and a `guiders` CI job that runs
  `tests/guiders/` with pfs-utils (installed `--no-deps`).
  `tests/guiders/conftest.py` provides a `makeAgcData` fixture for synthetic AG data (PIPE2D-1895).
- **`opaqueColorbar`** in `pfs.drp.qa.utils.plotting`, from `pfs.drp.stella.utils.quality`. Unlike the
  original, it restores an alpha of 0 or `None` (PIPE2D-1895).
- **`pfs.drp.qa.guiders.coordinates`** — the guider tools' frame, sign and unit conventions. Positions are
  in hardware coordinates, converted from the opdb once by `opdbToHardware`; offsets are center minus a
  named reference (`addOffsets`, columns `dx_<reference>_um`); columns carry their units; and unit
  helpers replace bare factors. Also the AG constants, `rotXY`, and the zenith-frame conversions
  (`pfiToZenith`, `zenithToPfi`) from drp_stella's `ag_to_zenith_offset`, which take hardware
  coordinates rather than negated y. Constants are renamed with their units (`agcCameraCenters` is
  `AGC_CAMERA_CENTERS_MM`, ...) (PIPE2D-1896).
- **`pfs.drp.qa.guiders.queries`** — the guider tools' opdb and butler readers. Each takes a
  `pfs.utils.database.opdb.OpDB` and binds its parameters. `readAgcData` replaces drp_stella's four AG
  readers: one query for all the visits, one row per matched spot (every match flag; `agc_match_flags`
  says which are valid), positions in hardware coordinates, a fresh index, and the columns listed in
  `AGC_DATA_COLUMNS`. INST-PA and, before 2025-03-21, `m2_off3` come from the raw headers through a Gen3
  butler (`readInstPa`, `find_W_M2OFF3`). Also `readAGCStars`, `readPfsDesign`, `readSpSInfo` and
  `readTelStatus`. Compared with drp_stella: `readAGCStars` has its missing comma, the schema's
  `*_designed` column names, and a qualified `guide_star_ra`, and reads only the visit's own config;
  nothing calls `pd.set_option`; an exposure without a tel_status row keeps its stars; `shutter_open`
  doesn't depend on spectrograph camera 1; `readSpSInfo` keeps visits in no sequence; NULL
  `agc_data` flags get RIGHT from the centroid, as ics_agActor reads them; and
  `find_W_M2OFF3` names the right visit in its error
  (PIPE2D-1897).
- **`pfs.drp.qa.guiders.compat`** — drp_stella's `readAGCPositionsForVisitByAgcExposureId`,
  `readAgcDataFromOpdb`, `readAGCStarsForVisitByPfsVisitId` and `readAGCStarsForVisitSetByPfsVisitId`,
  as wrappers of `readAgcData` that keep their arguments and columns and raise a `DeprecationWarning`
  (PIPE2D-1897).
- **`pfs.drp.qa.guiders.analysis`** — the guider tools' fits and statistics, taken out of drp_stella's
  plotting routines. They take DataFrames, return their results, and modify neither their inputs nor
  shared state: `fitGuiderModel` (`showGuiderErrors`'s boresight and per-camera fits, configured by
  `GuiderFitConfig` and returning a `GuiderFit` in place of `GuiderConfig`'s cache of transforms),
  `fitGlobalModel` (`ag_to_zenith_offset`'s fits, in hardware coordinates), `selectStars`,
  `smoothAgcData`, `estimateGuideErrors`, `fitDriftRate`, `comparePfsUtilsPositions`
  (`compareAGCPfsUtils`), `addImageSizes`, `estimateFocusErrors` (one left/right focus estimate for
  `plotFocus` and `plotFocusByAG`), `averageByFocusPosition` and `correctAgActorFocus`. Fits predict
  the new `model` reference. Fits and averages use only valid matches (`agc_match_flags == 1`, see
  `selectValidMatches`), keeping the others in their output. Compared with drp_stella: `MeasureXYRot` gets offsets in microns, not
  positions in mm; each camera's transform is fitted to its own stars; `solveForAGTransforms` refits;
  the guide error cut is matched to stars by exposure; smoothing follows each star and leaves IDs
  and flags alone; the per-visit references and closed-shutter data work in `estimateGuideErrors`;
  drift rates keep x and y apart; `compareAGCPfsUtils` averages over the first N exposures, gives
  pfs_utils UTC rather than the opdb's HST, and wraps `delta_theta` (whose sign flips, as it is now
  in hardware coordinates); focus positions round correctly; the INSTRM-2501 correction applies to
  each visit before 122129 whatever the others; and the zenith fits no longer append to
  `pandas.DataFrame._metadata` (PIPE2D-1898).
- **`boresight` and `model` references** in `pfs.drp.qa.guiders.coordinates`: `boresight` is
  `estimateGuideErrors`'s `nominal0` moved by each visit's shift, from the stars seen most often in
  that visit; `model` is where a fit in `analysis` puts each star (PIPE2D-1898).
- **Guider regression tests on real AG data** — `tests/guiders/test_realData.py` runs the guider
  analysis on engineering visits of Run 30 (a focus sweep, a raster scan, an all-sky exposure), read
  through the `realAgcData` and `realAgcStars` fixtures from `tests/guiders/data/` (0.63 MB of parquet,
  made by `makeGuiderFixtures.py` there) (PIPE2D-1900).
- **`pfs.drp.qa.guiders.plotting`** — drp_stella's guider plots: `showAgcErrorsForVisits`,
  `showAgcErrorsForVisitsByCamera`, `showGuiderErrors` and `showGuiderErrorsByParams` (drawing a `GuiderFit`, with
  the plotting options of `GuiderConfig` in `GuiderPlotConfig`), `showTelescopeErrors`, `plotDriftRate` (a
  `DriftFit`), `plotGuideErrors` (`estimateGuideErrors(plot=True)`), `plotPfsUtilsComparison` (`compareAGCPfsUtils`),
  `plotFocus`, `plotFocusByAG`, and the helpers `FormatCoord`, `ShowFocusFit` and `showAGCameraCartoon`. They take
  DataFrames or `analysis` results, never the opdb; draw on given axes or figure, without pyplot state; return a
  `GuiderPlot` of their artists; and update the colorbars they are given, as ics_pfsPlotActor needs. Cameras are
  `agcCameraIds` (0-5) rather than `AGC` (1-6), and `plotFocus`'s `mmToMicrons` is gone. Compared with drp_stella:
  `showAgcErrorsForVisits` plots center minus nominal (its signs were flipped); 2-D axes work in every helper; axes
  can be given without the figure; `plotFrac` subsets every array; per-visit means reach every panel; label colours
  match the cameras; float camera IDs work; per-camera XY panels take any number of cameras; closed-shutter points
  are plotted, and open-shutter ones once; `plotFocusByAG` applies `onlyGuideStars`; titles give the visit range and
  say "closed shutter"; `showTheta` and an axes-only `showTelescopeErrors` work; the arrows' legend works with
  matplotlib 3.9; FWHM points are drawn once; the cursor readout reads correctly and needs no database; and the
  averages use only valid matches. `README.md` has a "Guider tools" section, and the notebook
  `docs/guider-plots.ipynb` describes each plot (what it shows, its data, how to read it, what to look for) beside
  an annotated sample drawn from the tests' AG data; with `USE_OPDB = True` it runs on the opdb and checks the
  readers and fits the plots don't use. `docs/guiders-migration.ipynb` runs each drp_stella guiders call beside its
  drp_qa replacement, on a real opdb (PIPE2D-1899).
- **`pfs.drp.qa.plotting`** — the QA plots, moved unchanged out of `dmResiduals`, `dmCombinedResiduals`,
  `iqQaPlots` and `utils.plotting` (into `plotting.dmResiduals`, `.dmCombined`, `.iqQa` and `.palettes`), which
  re-export them. The subpackage imports no Butler, stack or task module, so the plots can be drawn from stored
  data; `tests/plotting/` checks that and smoke-tests each plot. `plot_detectormap_residuals` takes a
  `DetectorGeometry` (a `DetectorMap` still works), and `dmCombinedResiduals.make_report` wraps the page
  generator `plotting.dmCombined.reportFigures`. `FitStat`/`FitStats` move to `pfs.drp.qa.metrics.fitStats`.
  Rendered output is unchanged (PIPE2D-1916).

### Fixed

- **`dmResiduals` import** — `getDescriptionCounts` is now imported from
  `pfs.drp.stella.fitDetectorMap`. The former `pfs.drp.stella.fitDistortedDetectorMap` module no longer
  exists in `drp_stella`, so `DetectorMapResidualsTask` failed to import and the `dmResiduals` pipeline
  task could not run.
- **`drpQA.yaml` builds as a whole** — `imageQualityQa` declared `pfsConfig` as a plain input while
  `extractionQa` and `extractionQaCombined` declare it as a prerequisite, so `pipetask build` of the whole
  pipeline failed with `ConnectionTypeConsistencyError`; running one task at a time with `#label` hid it. It
  is now a prerequisite there too, still optional (`minimum=0`), and in `skyArmSubtractionQa`, so a pipeline
  combining it with `extractionQa` or `fluxCalQa` builds as well. `tests/test_connections.py` reads every
  task's connections from source, without the stack, and checks that no dataset type is declared both
  ways (PIPE2D-1913).
- **`fluxCalQa` import** — `FilterCurve` and `TransmissionCurve` are imported from
  `pfs.drp.stella.fitFluxReference`; `pfs.drp.stella.fitReference` no longer exists (PIPE2D-1913).
- **`plot_residual` with pandas 3** — the binned wavelength panel called `.agg("median", robustRms)`.
  Older pandas took `robustRms` as `numeric_only` and computed the median; pandas 3 raises, which left both
  residual-by-wavelength panels of `plot_detectormap_residuals` empty. It now calls `.agg("median")`. Its
  per-fiber `groupby().apply()` no longer warns about operating on the grouping columns (PIPE2D-1916).
- **`get_fit_stats` at zero dof** — with zero degrees of freedom and zero residuals, the softening solve
  raised from `scipy.optimize.bisect` and `dmResiduals` wrote nothing for the quantum. `softenFit` is now NaN
  there, as it already was at zero dof with nonzero residuals (PIPE2D-1916).
- **`plot_detector_summary_per_desc` on matplotlib ≥ 3.10** — seaborn 0.13 passes `vert` to `Axes.bxp`,
  which matplotlib 3.10 pending-deprecates and 3.11 deprecates for removal in 3.13. The plot now passes
  `vert=None`, so matplotlib uses `orientation`, whose default is vertical. The output is pixel-identical.
  This needs matplotlib 3.10, now the minimum in `pyproject.toml` (PIPE2D-1916).

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
- **`scipy` is a declared dependency** in `pyproject.toml`, for the guider fits (PIPE2D-1895).
- **matplotlib 3.10 or later is required** (`pyproject.toml`), for `Axes.bxp`'s `orientation` (PIPE2D-1916).
- **CI's `guiders` job is now `deps`** and runs every stack-free test, not only `tests/guiders/`. A test
  needing packages beyond the standard library goes in a `tests/` subdirectory listed in `_OPTIONAL_DIRS`
  (PIPE2D-1913).
- **`comparePfsUtilsPositions` uses the AG actor's model** — pfs_utils's positions of the guide stars
  come from the chain ics_agActor uses for its nominal positions (the new `agActorPositions`: pfs_utils's
  `Subaru_POPT2_PFS` with each AG exposure's field center, PA, ADC and M2 positions and detector half),
  and match the guider's to 0.2 µm on Run 30 data. drp_stella's `compareAGCPfsUtils`, and so this function
  until now, used `CoordinateTransform`'s `sky_pfi` mode, pfs_utils's model for the cobras, which is about
  550 µm off. `agcStars` needs only the stars' catalogue columns. It no longer fails under pandas 3, where
  pfs_utils clamped a read-only parallax array (PIPE2D-1900).

- **`smoothAgcData` smooths only valid matches** — each valid match is averaged with the star's other valid
  matches, and invalid ones are left as they are. Smoothing every row averaged invalid matches, hundreds of microns
  off, into the valid ones: on the tests' all-sky exposure (148258) the valid matches' x scatter went from 19.7 to
  89.4 µm, where it now falls to 15.9 µm (`testRealDataSmoothValidMatches`). `estimateGuideErrors` and
  `fitDriftRate` already gave it only valid matches (PIPE2D-1899).

### Removed

- **`uv.lock`** — it had gone stale (`uv lock --check` failed) and nothing used it: CI installs its packages
  without it, and the code runs in the LSST stack. It is now in `.gitignore`; `uv sync` still works and writes a
  local lockfile (PIPE2D-1906).
- **Log-artifact tests** — the `TestRealLogs` class in `tests/test_fitDetectorMapLogQa.py` depended on
  `run28-dm-02.log` / `run28-dm-03.log`, which are not in the repository, so all eight tests always
  skipped and provided no coverage.

- **SCons build** — `SConstruct`, `bin.src/SConscript`, `tests/SConscript` and `ups/drp_qa.cfg`. The `bin/`
  directory is no longer generated; scripts are run as `python bin.src/<name>.py` (`tests/SConscript`:
  PIPE2D-1913).
- **`setup.cfg`** and **`mypy.ini`** — superseded by `pyproject.toml`. No static type checker is configured for this
  repository.
- **Unused `database` pytest marker** — the marker declaration and the `-m 'not database'` filter in
  `addopts` are removed from `pyproject.toml`. No test in the repository was ever marked with it, so the
  filter was a no-op.
