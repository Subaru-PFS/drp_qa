# QA domain notes — `imageQualityQa`, arc-lamp physics, failure patterns

Empirical findings accumulated through QA analysis on PFS engineering-run data. These are
**observations, not rules**: thresholds, line counts and failure signatures reflect the data they
were derived from and will drift as the instrument and the pipeline change. Re-check against
current data before acting on a number here.

Anything labelled **proposed** below is not implemented in any merged branch — check the code
before relying on it. Claims here were last verified against `drp_qa` `main` and `drp_stella`
`master` `c771d507` on 2026-10-02.

Conventions and build/test rules live in [`AGENTS.md`](../AGENTS.md); this file holds only the
domain knowledge that used to sit in sections 7, 8 and 10 of it.

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
statistics (`fitChi2`, `fitXRms`, `fitYRms`, `fitReserved*`, `fitSpecies*Rms_<species>`,
per-fiber arrays).

### DM residual metrics

`dmResiduals` writes `dmQaResidualData`, `dmQaResidualStats` and `dmQaResidualPlot`.
`get_fit_stats` builds `dmQaResidualStats` **per `(status_type, description)`** — that
is, separately for `RESERVED` and `USED` lines of each species — via the `FitStats` /
`FitStat` dataclasses (`pfs.drp.qa.metrics.fitStats`). Each row carries `dof`, `chi2X`, `chi2Y`, and a `spatial.` and
`wavelength.` block of `median`, `robustRms`, `weightedRms`, `softenFit`, `dof`,
`num_fibers`, `num_lines`. The RMS values are **error-weighted** (`getWeightedRMS`) with
a robust variant (`robustRms`); `softenFit` is solved by bisection, and is NaN when it
can't be (more than `maxSoften` needed, or zero dof). Prefer these over
adding parallel unweighted metrics.

`dmCombinedResiduals` aggregates across detectors into `dmQaDetectorStats` and renders a
multi-page `dmQaCombinedResidualPlot` via `make_report`, whose pages are drawn by
`pfs.drp.qa.plotting.dmCombined.reportFigures`.

The task writes **data only**. Plotting lives in `plotting/iqQa.py` and is driven after the
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

### Proposed fix: `minSignalToNoisePerSpecies` in `fitDetectorMap`

> **Not implemented.** The field does not exist in `drp_stella`: as of `master` `c771d507`
> (2026-10-02) nothing in `python/` matches `perSpecies`, and only the global
> `minSignalToNoise` is available (`FitDetectorMapConfig`, default 10.0, in
> `python/pfs/drp/stella/fitDetectorMap.py`). A working prototype exists only on the
> **local, unpushed** `drp_stella` branch `adjust-dm-fixes` (commits `e3d1a5db` field,
> `2ffbe7ba` logging; last touched 2026-08-11, no remote, no ticket). The CLI examples
> below describe that interface; against `master` they fail with an unknown-field error.

The proposal is to add a `DictField(keytype=str, itemtype=float, default={})` to
`FitDetectorMapConfig`, allowing per-species S/N thresholds to be set independently of the
global `minSignalToNoise` (default 10). Suggested values for the b arm:

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

**Proposed fix, not committed — the crash is live.** As of `drp_stella` `master` `c771d507`
(2026-10-02) `calculateSoftening` guards only `residuals.size == 0`; there is no `dof`
guard. Since NaN compares False both ways, `softenChi2(0.0) < 0` and
`softenChi2(maxSoften) > 0` both fall through and `scipy.optimize.bisect` still raises
`f(a) = NaN`.

The fix exists on the same local, unpushed `drp_stella` branch `adjust-dm-fixes`
(commit `5c35840d`): guard `residuals.size == 0 or dof <= 0` → return `0.0` early, and
collapse the pre-bisect check into `if not np.isfinite(val) or val < 0`.

---


## Common Failure Patterns

From engineering run data:

| Symptom | Likely cause | Remedy |
|---|---|---|
| b-arm `pctFlagged` above the `b:Argon`/`b:Xenon`/`b:Krypton` thresholds | Lamp has very few/faint b-arm lines → global S/N cut flags almost all | No remedy available yet — `minSignalToNoisePerSpecies` is only proposed (see above). Treat as known lamp physics rather than a detector fault |
| b-arm `pctFlagged` > 15 % for HgCd arcs (or > 50 % for Ne) | Genuine crowding or optical problem | Investigate `medFwhm`; if FWHM is also high → optics issue |
| High `pctMeasFail` with low `pctLowSN` | Centroid/photometry failures rather than faint lines — not lamp physics | Investigate the image; relaxing S/N thresholds will not help |
| All arms FWHM > 3.5 px for a single spectrograph module | Hardware/focus issue | Flag the entire SM as bad for that visit range |
| `\|medDxCenter\|` > 1 px across all arms of an SM | Flexure or a stale `detectorMap_calib` | Check the calib validity range before blaming the optics |
| `medDxCenter` ≈ 0 but `dxCenterRms` large | Distortion rather than bulk shift | Look at the `dxCenter` distribution in `iqQaData`, not just the summary |
| `traceOnly=True` for all arc visits | `arcLines` connection missing or `minGoodLines` not satisfied | Check `fitDetectorMap` ran and produced `lines` |
| No `medFwhm` (sparse result) for IIS arc or IIS quartz frames | Deliberate. The science arc catalog does not match the 16 engineering fibers, and scattered light from them gave a spurious ~7 px calexp FWHM, so both `is_iis` branches of `imageQualityQa` set `force_sparse = True` and never run the calexp measurement. `pctFlagged` is suppressed as well | Expected; no action. Were the calexp path re-enabled for IIS, the ~7 px figure would return for a second, independent reason — see the next row |
| Regular quartz/trace calexp logs "too sparse … FWHM will be sparse", or a calexp FWHM near the aperture width | `profileHalfWidth` (default 7 px) exceeds the 6.17 px fiber pitch on every arm, so the aperture always contains neighbouring traces. Here calexp *is* the primary path, so the bug bites | `maxCalexpFlagRate` discards the result and the visit reports sparse, which masks the cause rather than fixing it. Root-cause analysis and a proposed estimator are in `doc/tickets/drp_qa-calexp-width-estimator.md` on the unmerged branch |
| All `fit*` metric columns are `NaN`/0 | The `*_log` connections were absent from the input collection | Expected when `reduceExposure` hasn't run; the IQ metrics themselves are unaffected |
| `ValueError: f(a) = NaN` in bisect during `fitDetectorMap` | Per-fiber dof=0 when very few arc lines remain after S/N cut | Proposed `dof <= 0` guard in `calculateSoftening` is **not** committed (see above), so relaxing per-species S/N is not yet safe |

---
