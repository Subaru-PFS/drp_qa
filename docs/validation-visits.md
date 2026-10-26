# The validation visit set

The validation visit set is a fixed list of PFS visits whose QA verdict is known. Every
threshold and every metric is checked against it. A metric that flags a known-good visit,
or passes a known-bad one, does not merge.

- **Source of truth:**
  [`python/pfs/drp/qa/metrics/data/validationVisits.yaml`](../python/pfs/drp/qa/metrics/data/validationVisits.yaml),
  shipped with the package. Its header comment defines the entry schema.
- **Loader:** `pfs.drp.qa.metrics.validationVisits.loadValidationVisits()`.
- **Using it:** [`qa-thresholds.ipynb`](qa-thresholds.ipynb) is the step-by-step
  procedure: run `imageQualityQa` over these visits, then derive and check thresholds.
- **Known-bad coverage:** `pfs.drp.qa.metrics.coverage.knownBadCoverage` checks every
  known-bad entry against the thresholds the gate uses: every image it covers, with its
  metric measured, must reach its verdict, or the entry says why no check can
  (`unchecked`). `tests/metrics/test_coverage.py` runs it on the stored metrics of the
  validation visits, and the notebook prints it (PIPE2D-1934).

## Where the visits come from

Every entry was checked against the run's calibration summary (`calib_data.csv`, one row
per sequence: `sequence_type`, sequence name, cameras read, notes) for Runs 25, 27 and
30. Run27 took calibration data only, and no calibrations were made from it: it was
reduced with another run's, so its metrics are not comparable with Run25's (the n-arm
flag rates of its fault-free spectrographs sit 2–5 points higher). A `known_bad` reason
quotes the summary's note where there is one. The summaries are written by hand, so a sequence name there can
differ from the `W_SEQNAM` header that `seqType` is matched against. The threshold
notebook lists any entry that matches no data, which is how such a mismatch shows up.

## The reference run

Thresholds are derived from the known-good visits of the reference runs (`referenceRuns` in
the YAML), and the loader rejects a known-good entry from any other run. **Run25** is the
reference: its calibrations were made for it, and it is the most complete stable set. Two
detectors take their quartz reference from **Run29** instead (PIPE2D-1938): b2, whose slit was
refocused on 2026-03-11 and whose quartz width stepped from 3.12 to 3.34 px, where it has
stayed; and n4, whose Run25 quartz width (3.10 px) is the outlier against a steady 3.22 px in
every run since. Run25's trace entries leave those two cameras out (`excludeCameras`), and
Run29's detectorMap set holds them alone (`cameras`), so each population still has one
reference. For the line-scatter metrics Run25 is every camera's reference, b2 and n4 included
(PIPE2D-1956): Run29's trace entries carry `excludeMetrics: [robustRmsDeltaX, robustRmsDeltaY]`,
and Run25's trace sets have b2/n4 entries scoped to those two metrics (`metrics`), so a
known-good entry can be the reference for some metrics and not others. From other runs the set
holds only known-bad visits, as fault examples: each is
checked against the thresholds, and none derives one. The YAML's `runs` table gives each
run's visit range, so every entry belongs to one run.

Other runs are compared with Run25's thresholds by comparison mode (PIPE2D-1929), which
reads a run's calibration summary, not by this set; Run30 is the first. That is also where
a fault's other spectrographs, in the same exposures, are seen to pass.

## What belongs in the set

A visit belongs only if it has a **verdict**: someone has established that it should pass,
or that it should warn or fail for a stated reason.

- **`known_good`:** expected to pass every metric. These are the samples that thresholds
  are derived from, so they must span the populations being gated: every arm, arcs of every
  lamp, traces, and sky frames.
- **`known_bad`:** expected to `WARN` or `FAIL`. Each entry names the metric that should
  catch it, because a known-bad visit must fail *for the right reason*. An entry holds only
  for the metric it names; it says nothing about the others.
- **`unchecked: <reason>`:** a known-bad entry whose verdict no check gives today: its
  metric is not judged, or not measured, on those images. It keeps the fault on record and
  says why it is not caught. Every other confirmed known-bad entry must be caught, and an
  unchecked one must not be: once a check catches it, drop the field.
- **`unconfirmed: true`:** a suspected fault, noted in the run's calibration summary, whose numbers no one
  has checked yet. It is reported beside the thresholds but never decides anything. Once
  its metrics have been compared with the known-good data, either name the metric and drop
  the flag, or move the visit to `known_good`.
- **`placeholder: true`:** a reference case still to be found in the archive. The loader
  skips it.

Visits without a verdict don't belong: a run's calibration block, a drift series, the
inputs of a run-to-run comparison. Select those from the Butler by sequence type and date
when a job needs them; a transcribed list goes stale each run.

The file is in a public repository. Use engineering and calibration visits only,
identified by visit number, with no observer, target or design names.

## What is in it

### Known good

- **Run25 stable calibration set (2025-11-10/11):** the core of the reference run.
  Fixed pointing (alt 90, az 90, insrot 0); Ar, Xe, Ne, Kr and HgCd arcs and quartz traces.
  Block A read the b, r and n arms; block B read b and m.
- **Run25 block C (2025-12-01):** the same sequence at azimuth 168. It is a second epoch
  under a different gravity vector, and it brings b-arm HgCd above the 20-sample floor.
- **Run25 uniformity traces (2025-11-12 and 11-21):** extra quartz frames, so that trace
  widths can have thresholds of their own rather than borrowing the arcs'.
- **Twilight sky, Run25 (2025-11-25 and 11-28; one set at insrot 90):** the reference for
  a future sky-line check (PIPE2D-1925). Whether they should also be held to the FWHM and
  flag-rate gates is not yet established.

### Known bad

- **b-arm flag rates (Run25 Ar, Xe, Ne and Kr arcs):** most b-arm lines of these lamps are
  flagged, because `adjustDetectorMap` applies one S/N cut to all lines together. The
  exposures are good for every other metric; these entries hold for `pctFlagged` only, so
  today's rates are the baseline for the drp_stella work rather than the threshold.
  Unchecked: `pctFlagged` is not judged (PIPE2D-1929).
- **Cloudy twilight (Run25):** taken through cloud. Unchecked: nothing measured on a
  twilight sees cloud. FWHM is not measured on twilights (PIPE2D-1917), and cloud need not
  widen the lines; their `nLines` sit within the clear twilights' range. The check that
  would see it, the uniformity of the sky's flux across the fibers against `fiberNorms`,
  is not written yet (PIPE2D-1973), and the entry names it (`skyFluxUniformity`).
- **SM1 defocused (Run27, 2026-03-09/10, spectrograph 1):** the eight arc and trace
  sequences noted "SM1 defocused", with FWHM 3.83–4.86 px across every lamp type;
  `medFwhm`. The same visits are known-bad for the line scatter, `robustRmsDeltaX`, at
  1.7–4× each detector's Run25 median, except HgCd (scatter not judged) and b-arm Xenon (the
  one noisy population; two of its three defocused images pass). See "SM1 exception" in
  [`qa-domain-notes.md`](qa-domain-notes.md). The
  other visits in 140005–140138 carry no verdict and are not included: darks, the
  `sky1540+3500` frames, and visits the summary does not list (among them the raster
  scans 140106–140112 and 140115–140121).
- **Unlit spectrograph (Run27 nightly `ImageQuality` arcs):** exposures noted "No light on
  SM1" or "SM2". The unlit spectrograph's detectors have no lines; `nLines`, `FAIL`.
  Unchecked: they are IIS frames, lighting no `GOOD` science fiber, so `linesPerFiber` is not
  measured (PIPE2D-1935) and `nLines` is not judged. The raw `nLines` does separate them
  (r about 2.6k rows unlit against 16–19k lit). Judging IIS frames is PIPE2D-1954.
- **Obstructed frames (Run30):** taken with the M1 cover or the top screen closed. Less
  light reaches the detectors, so each lit fiber measures fewer lines (a quarter of the lamp
  lines on an arc, nine tenths of the trace centroids on a quartz); `linesPerFiber`, `FAIL`.
- **Partial slit illumination (Run30)** is no longer known-bad (PIPE2D-1935): with one fiber
  group lit, each lit fiber measures the lines a full slit's does, and `linesPerFiber`
  passes it. Comparison mode judges these daily arcs with every other.
- **149883 ("Home", Run30)** was a suspected fault and is removed: every metric passes it,
  and as a Run30 visit it cannot be known-good (PIPE2D-1934).
- **150661 (Run30, "Mirror cover open, but screen is wrong position")** was a suspected fault
  and is removed (PIPE2D-1934). Something is off: its trace scatter against its own
  detectorMap is 1.5–1.6× each b detector's Run25 median, r reaches WARN or FAIL on three of
  four and n is mixed. But no metric can be named for it: nobody has shown why a misplaced
  screen would scatter the trace centroids, its `linesPerFiber` is within 3 % of normal, and it
  is one visit.
- **Off-zenith Helium detectorMap tests (Run30, 150779 and 150782, alt 45–49)** were listed
  as the flexure case, `WARN` provisionally, and are removed: against `detectorMap_calib`
  they sit within 0.2 px in y and −0.33 to −0.51 px in x on b, as b read with m does at the
  zenith (PIPE2D-1921). On the n arm `reduceExposure` fails on Helium (PIPE2D-1960).
- **Run27 against a later calibration (PIPE2D-1921):** the Run27 `detectorMap_calib` was
  built from 140502–140526, after these visits. Up to 140148, SM3 sits +1.15 px off in x
  (`medianDeltaX`, `WARN`) and SM2 −3.15 px off in y on b and r (`medianDeltaY`, `FAIL`);
  from 140477 on, neither is. The reference case for a calibration from another epoch.
- **Run25 SM3 after its calibration block (unconfirmed):** every Run25 visit after
  133055 sits off `detectorMap_calib` in y, by the same pixels in every arm of a
  spectrograph (n with the opposite sign): SM1 0.45, SM2 0.65, SM3 1.40, SM4 0.70 px
  (PIPE2D-1921). The adjustment absorbs it, so they stay known-good, but SM3 reaches the
  1.0 px `medianDeltaY` WARN. Unconfirmed until the cause is known. Only SM3's arcs and
  twilights are listed: a quartz has trace centroids only, and no `medianDeltaY`.

### Gaps

- A saturated frame (for PIPE2D-1922's saturation metric).
- m-arm traces: only the short block B and C sequences exist, below the sample floor.

## Every entry

Generated from the YAML by `python -m pfs.drp.qa.metrics.validationVisits tables`.
`tests/metrics/test_validationVisits.py` fails if this section and the YAML disagree.

<!-- BEGIN GENERATED: validationVisits tables -->
### Known good

| Visits | Run | Type | Arms | Spectrographs | Sequence | Expect | Metric | Note |
|---|---|---|---|---|---|---|---|---|
| 133025–133027 | 25 | scienceArc | b, r, n | 1, 2, 3, 4 | Arc: Argon | PASS |  | Run25 stable set, block A. 10 s exposures. |
| 133028–133030 | 25 | scienceArc | b, r, n | 1, 2, 3, 4 | Arc: Xenon | PASS |  | Run25 stable set, block A. 45 s exposures. |
| 133031–133033 | 25 | scienceArc | b, r, n | 1, 2, 3, 4 | Arc: Neon | PASS |  | Run25 stable set, block A. 5 s exposures. |
| 133034–133036 | 25 | scienceArc | b, r, n | 1, 2, 3, 4 | Arc: Krypton | PASS |  | Run25 stable set, block A. 70 s exposures. |
| 133037–133039 | 25 | scienceArc | b, r, n | 1, 2, 3, 4 | Arc: HgCd | PASS |  | Run25 stable set, block A. 45 s exposures. HgCd gives dense, well-measured b-arm coverage, so b-arm flag rates here are genuine rather than lamp physics. |
| 133040–133041 | 25 | scienceTrace | b, r, n; not b2, n4 | 1, 2, 3, 4 | Trace | PASS |  | Run25 stable set, block A. 20 s scienceTrace; exercises the calexp trace-width path. |
| 133042–133044 | 25 | scienceArc | b, m | 1, 2, 3, 4 | Arc: Argon | PASS |  | Run25 stable set, block B. 10 s exposures. |
| 133045–133047 | 25 | scienceArc | b, m | 1, 2, 3, 4 | Arc: Xenon | PASS |  | Run25 stable set, block B. 45 s exposures. |
| 133048–133050 | 25 | scienceArc | b, m | 1, 2, 3, 4 | Arc: Neon | PASS |  | Run25 stable set, block B. 5 s exposures. |
| 133051–133053 | 25 | scienceArc | b, m | 1, 2, 3, 4 | Arc: Krypton | PASS |  | Run25 stable set, block B. 45 s exposures. |
| 133054–133055 | 25 | scienceTrace | b, m; not b2, n4 | 1, 2, 3, 4 | Trace | PASS |  | Run25 stable set, block B. 20 s scienceTrace. |
| 135828–135829 | 25 | scienceTrace | b, r, n; not b2, n4 | 1, 2, 3, 4 | Trace | PASS |  | Run25 block C, az 168. 20 s scienceTrace. |
| 135830–135831 | 25 | scienceArc | b, r, n | 1, 2, 3, 4 | Arc: Argon | PASS |  | Run25 block C, az 168. 10 s exposures. |
| 135832–135833 | 25 | scienceArc | b, r, n | 1, 2, 3, 4 | Arc: Xenon | PASS |  | Run25 block C, az 168. 45 s exposures. |
| 135834–135835 | 25 | scienceArc | b, r, n | 1, 2, 3, 4 | Arc: Neon | PASS |  | Run25 block C, az 168. 5 s exposures. |
| 135836–135837 | 25 | scienceArc | b, r, n | 1, 2, 3, 4 | Arc: Krypton | PASS |  | Run25 block C, az 168. 70 s exposures. |
| 135838–135840 | 25 | scienceArc | b, r, n | 1, 2, 3, 4 | Arc: HgCd | PASS |  | Run25 block C, az 168. 45 s exposures. The second HgCd block; see above. |
| 135841–135842 | 25 | scienceArc | b, m | 1, 2, 3, 4 | Arc: Argon | PASS |  | Run25 block C, az 168, m arm. 10 s exposures. |
| 135843–135844 | 25 | scienceArc | b, m | 1, 2, 3, 4 | Arc: Xenon | PASS |  | Run25 block C, az 168, m arm. 45 s exposures. |
| 135845–135846 | 25 | scienceArc | b, m | 1, 2, 3, 4 | Arc: Neon | PASS |  | Run25 block C, az 168, m arm. 5 s exposures. |
| 135847–135848 | 25 | scienceArc | b, m | 1, 2, 3, 4 | Arc: Krypton | PASS |  | Run25 block C, az 168, m arm. 45 s exposures. |
| 135849–135850 | 25 | scienceTrace | b, m; not b2, n4 | 1, 2, 3, 4 | Trace | PASS |  | Run25 block C, az 168, m arm. 20 s scienceTrace. |
| 134880–134883 | 25 | scienceObject | b, r, n | 1, 2, 3, 4 | Twilight sky | PASS |  | Run25 twilight sky, 180 s. Reference case for the O2/OH sky-line check. |
| 134884–134886 | 25 | scienceObject | b, r, n | 1, 2, 3, 4 | Twilight sky | PASS |  | Run25 twilight sky, 180 s. Reference case for the O2/OH sky-line check. |
| 133532–133533 | 25 | scienceTrace | b, r, n; not b2, n4 | 1, 2, 3, 4 | Trace | PASS |  | Run25 uniformity set, 2025-11-12, az 90. 20 s scienceTrace. |
| 133536–133537 | 25 | scienceTrace | b, r; not b2, n4 | 1, 2, 3, 4 | Trace | PASS |  | Run25 uniformity set, 2025-11-12, az 90. 20 s scienceTrace; n arm not read. |
| 134338–134339 | 25 | scienceTrace | b, r, n; not b2, n4 | 1, 2, 3, 4 | Trace | PASS |  | Run25 uniformity set, 2025-11-21, az 290. 20 s scienceTrace. |
| 135275–135279 | 25 | scienceObject | b, r, n | 1, 2, 3, 4 | Twilight sky | PASS |  | Run25 uniformity set, 2025-11-28, az 290, insrot 90. 180 s. Reference case for the O2/OH sky-line check, as for the other twilight entries. |
| 145025–145034 | 29 | scienceTrace | b, r, n; cameras b2, n4 | 1, 2, 3, 4 | Trace | PASS | not `robustRmsDeltaX`, `robustRmsDeltaY` | Run29 detectorMap set, 2026-07-07, b/r/n arms. b2 and n4 quartz reference (PIPE2D-1938). |
| 145062–145071 | 29 | scienceTrace | b, m; cameras b2 | 1, 2, 3, 4 | Trace | PASS | not `robustRmsDeltaX`, `robustRmsDeltaY` | Run29 detectorMap set, 2026-07-07, b/m arms. b2 quartz reference (PIPE2D-1938). |
| 146090–146092 | 29 | scienceTrace | b, r, n; cameras b2, n4 | 1, 2, 3, 4 | Trace | PASS | not `robustRmsDeltaX`, `robustRmsDeltaY` | Run29 detectorMap set, 2026-07-14. b2 and n4 quartz reference (PIPE2D-1938). |
| 147745–147747 | 29 | scienceTrace | b, r, n; cameras b2, n4 | 1, 2, 3, 4 | Trace | PASS | not `robustRmsDeltaX`, `robustRmsDeltaY` | Run29 detectorMap set, 2026-07-23. b2 and n4 quartz reference (PIPE2D-1938). |
| 133040–133041 | 25 | scienceTrace | b, n; cameras b2, n4 | 2, 4 | Trace | PASS | `robustRmsDeltaX`, `robustRmsDeltaY` |  |
| 133054–133055 | 25 | scienceTrace | b; cameras b2 | 2 | Trace | PASS | `robustRmsDeltaX`, `robustRmsDeltaY` |  |
| 133532–133533 | 25 | scienceTrace | b, n; cameras b2, n4 | 2, 4 | Trace | PASS | `robustRmsDeltaX`, `robustRmsDeltaY` |  |
| 133536–133537 | 25 | scienceTrace | b; cameras b2 | 2 | Trace | PASS | `robustRmsDeltaX`, `robustRmsDeltaY` |  |
| 134338–134339 | 25 | scienceTrace | b, n; cameras b2, n4 | 2, 4 | Trace | PASS | `robustRmsDeltaX`, `robustRmsDeltaY` |  |
| 135828–135829 | 25 | scienceTrace | b, n; cameras b2, n4 | 2, 4 | Trace | PASS | `robustRmsDeltaX`, `robustRmsDeltaY` |  |
| 135849–135850 | 25 | scienceTrace | b; cameras b2 | 2 | Trace | PASS | `robustRmsDeltaX`, `robustRmsDeltaY` |  |

### Known bad

| Visits | Run | Type | Arms | Spectrographs | Sequence | Expect | Metric | Reason |
|---|---|---|---|---|---|---|---|---|
| 133025–133027 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Argon | WARN | `pctFlagged` (unchecked) | b-arm Argon (Run25 block A): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 133028–133030 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Xenon | WARN | `pctFlagged` (unchecked) | b-arm Xenon (Run25 block A): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 133031–133033 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Neon | WARN | `pctFlagged` (unchecked) | b-arm Neon (Run25 block A): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 133034–133036 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Krypton | WARN | `pctFlagged` (unchecked) | b-arm Krypton (Run25 block A): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 133042–133044 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Argon | WARN | `pctFlagged` (unchecked) | b-arm Argon (Run25 block B): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 133045–133047 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Xenon | WARN | `pctFlagged` (unchecked) | b-arm Xenon (Run25 block B): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 133048–133050 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Neon | WARN | `pctFlagged` (unchecked) | b-arm Neon (Run25 block B): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 133051–133053 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Krypton | WARN | `pctFlagged` (unchecked) | b-arm Krypton (Run25 block B): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 135830–135831 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Argon | WARN | `pctFlagged` (unchecked) | b-arm Argon (Run25 block C): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 135832–135833 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Xenon | WARN | `pctFlagged` (unchecked) | b-arm Xenon (Run25 block C): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 135834–135835 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Neon | WARN | `pctFlagged` (unchecked) | b-arm Neon (Run25 block C): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 135836–135837 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Krypton | WARN | `pctFlagged` (unchecked) | b-arm Krypton (Run25 block C): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 135841–135842 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Argon | WARN | `pctFlagged` (unchecked) | b-arm Argon (Run25 block C): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 135843–135844 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Xenon | WARN | `pctFlagged` (unchecked) | b-arm Xenon (Run25 block C): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 135845–135846 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Neon | WARN | `pctFlagged` (unchecked) | b-arm Neon (Run25 block C): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 135847–135848 | 25 | scienceArc | b | 1, 2, 3, 4 | Arc: Krypton | WARN | `pctFlagged` (unchecked) | b-arm Krypton (Run25 block C): most lines flagged by adjustDetectorMap's single S/N cut. *Unchecked:* pctFlagged is not judged (PIPE2D-1929); this records the rate a drp_stella fix is measured against. |
| 134334–134337 | 25 | scienceObject | b, r, n | 1, 2, 3, 4 | Twilight sky | FAIL | `skyFluxUniformity` (unchecked) | Cloudy; 180 s twilight sky taken through cloud. *Unchecked:* No metric measured on twilights sees cloud; the sky-uniformity check is PIPE2D-1973. |
| 140005–140006 | 27 | scienceArc | r, n | 1 | Arc: Argon | FAIL | `medFwhm` | SM1 defocused (Run27). 15 s, az 90. |
| 140032 | 27 | scienceTrace | b, r, n | 1 | Trace | FAIL | `medFwhm` | SM1 defocused (Run27). 30 s, az 226. |
| 140035 | 27 | scienceTrace | b, r, n | 1 | Trace | FAIL | `medFwhm` | SM1 defocused (Run27). 30 s, az 226. |
| 140139–140148 | 27 | scienceTrace | b, r, n | 1 | Trace | FAIL | `medFwhm` | SM1 defocused (Run27). 30 s, az 336. |
| 140124–140126 | 27 | scienceArc | r, n | 1 | Arc: Argon | FAIL | `medFwhm` | SM1 defocused (Run27). 15 s, az 336. |
| 140127–140129 | 27 | scienceArc | b, r, n | 1 | Arc: Xenon | FAIL | `medFwhm` | SM1 defocused (Run27). 45 s, az 336. |
| 140130–140132 | 27 | scienceArc | b, r, n | 1 | Arc: Neon | FAIL | `medFwhm` | SM1 defocused (Run27). 5 s, az 336. |
| 140133–140135 | 27 | scienceArc | b, r, n | 1 | Arc: Krypton | FAIL | `medFwhm` | SM1 defocused (Run27). 70 s, az 336. |
| 140136–140138 | 27 | scienceArc | b, r, n | 1 | Arc: HgCd | FAIL | `medFwhm` | SM1 defocused (Run27). 45 s, az 336. |
| 140005–140006 | 27 | scienceArc | r, n | 1 | Arc: Argon | FAIL | `robustRmsDeltaX` | SM1 defocused (Run27). 15 s, az 90. |
| 140032 | 27 | scienceTrace | b, r, n | 1 | Trace | FAIL | `robustRmsDeltaX` | SM1 defocused (Run27). 30 s, az 226. |
| 140035 | 27 | scienceTrace | b, r, n | 1 | Trace | FAIL | `robustRmsDeltaX` | SM1 defocused (Run27). 30 s, az 226. |
| 140139–140148 | 27 | scienceTrace | b, r, n | 1 | Trace | FAIL | `robustRmsDeltaX` | SM1 defocused (Run27). 30 s, az 336. |
| 140124–140126 | 27 | scienceArc | r, n | 1 | Arc: Argon | FAIL | `robustRmsDeltaX` | SM1 defocused (Run27). 15 s, az 336. |
| 140127–140129 | 27 | scienceArc | r, n | 1 | Arc: Xenon | FAIL | `robustRmsDeltaX` | SM1 defocused (Run27). 45 s, az 336. |
| 140130–140132 | 27 | scienceArc | b, r, n | 1 | Arc: Neon | FAIL | `robustRmsDeltaX` | SM1 defocused (Run27). 5 s, az 336. |
| 140133–140135 | 27 | scienceArc | b, r, n | 1 | Arc: Krypton | FAIL | `robustRmsDeltaX` | SM1 defocused (Run27). 70 s, az 336. |
| 139971 | 27 | scienceArc | all | 3 | any | WARN | `medianDeltaX` | Run27 before 140477, reduced against a detectorMap_calib built later (140502-140526): SM3 lines +1.15 px off in x. |
| 140005–140006 | 27 | scienceArc | all | 3 | any | WARN | `medianDeltaX` | Run27 before 140477, reduced against a detectorMap_calib built later (140502-140526): SM3 lines +1.15 px off in x. |
| 140124–140138 | 27 | scienceArc | all | 3 | any | WARN | `medianDeltaX` | Run27 before 140477, reduced against a detectorMap_calib built later (140502-140526): SM3 lines +1.15 px off in x. |
| 140032 | 27 | scienceTrace | all | 3 | any | WARN | `medianDeltaX` | Run27 before 140477, reduced against a detectorMap_calib built later (140502-140526): SM3 lines +1.15 px off in x. |
| 140035 | 27 | scienceTrace | all | 3 | any | WARN | `medianDeltaX` | Run27 before 140477, reduced against a detectorMap_calib built later (140502-140526): SM3 lines +1.15 px off in x. |
| 140139–140148 | 27 | scienceTrace | all | 3 | any | WARN | `medianDeltaX` | Run27 before 140477, reduced against a detectorMap_calib built later (140502-140526): SM3 lines +1.15 px off in x. |
| 139971 | 27 | scienceArc | b, r | 2 | any | FAIL | `medianDeltaY` | Run27 before 140477, reduced against a detectorMap_calib built later (140502-140526): SM2 lines -3.15 px off in y, at drp_stella's 3 px line-search radius. |
| 140005–140006 | 27 | scienceArc | b, r | 2 | any | FAIL | `medianDeltaY` | Run27 before 140477, reduced against a detectorMap_calib built later (140502-140526): SM2 lines -3.15 px off in y, at drp_stella's 3 px line-search radius. |
| 140124–140138 | 27 | scienceArc | b, r | 2 | any | FAIL | `medianDeltaY` | Run27 before 140477, reduced against a detectorMap_calib built later (140502-140526): SM2 lines -3.15 px off in y, at drp_stella's 3 px line-search radius. |
| 139971 | 27 | scienceArc | b, r, n | 1 | ImageQuality | FAIL | `linesPerFiber` (unchecked) | SM1 lamp didn't turn on. (Run27, 1 s arc) *Unchecked:* IIS frame: linesPerFiber on the engineering fibers has no IIS thresholds yet (PIPE2D-1954). |
| 140477 | 27 | scienceArc | b, r, n | 1 | ImageQuality | FAIL | `linesPerFiber` (unchecked) | No light on SM1. (Run27, 1 s arc) *Unchecked:* IIS frame: linesPerFiber on the engineering fibers has no IIS thresholds yet (PIPE2D-1954). |
| 140640 | 27 | scienceArc | b, r, n | 2 | ImageQuality | FAIL | `linesPerFiber` (unchecked) | No light on SM2. (Run27, 1 s arc) *Unchecked:* IIS frame: linesPerFiber on the engineering fibers has no IIS thresholds yet (PIPE2D-1954). |
| 140649 | 27 | scienceArc | b, r, n | 1, 2 | ImageQuality | FAIL | `linesPerFiber` (unchecked) | No light on SM1 or SM2. (Run27, 1 s arc) *Unchecked:* IIS frame: linesPerFiber on the engineering fibers has no IIS thresholds yet (PIPE2D-1954). |
| 140651 | 27 | scienceArc | b, r, n | 2 | ImageQuality | FAIL | `linesPerFiber` (unchecked) | No light on SM2. (Run27, 1 s arc) *Unchecked:* IIS frame: linesPerFiber on the engineering fibers has no IIS thresholds yet (PIPE2D-1954). |
| 150115 | 30 | scienceArc | b, r, n | 1, 2, 3, 4 | any | FAIL | `linesPerFiber` | M1 cover closed. |
| 150116 | 30 | scienceTrace | b, r, n | 1, 2, 3, 4 | any | FAIL | `linesPerFiber` | M1 cover closed. |
| 150641 | 30 | scienceArc | b, r, n | 1, 2, 3, 4 | any | FAIL | `linesPerFiber` | Mirror cover and top screen both closed; noted in the calibration summary as to be ignored. |
| 150642 | 30 | scienceTrace | b, r, n | 1, 2, 3, 4 | any | FAIL | `linesPerFiber` | Mirror cover and top screen both closed; noted in the calibration summary as to be ignored. |

### Unconfirmed

| Visits | Run | Type | Arms | Spectrographs | Sequence | Expect | Metric | Reason |
|---|---|---|---|---|---|---|---|---|
| 135830–135848 | 25 | scienceArc | all | 3 | any | WARN | `medianDeltaY` | Run25 SM3, after the calibration block: lines 1.4 px off detectorMap_calib in y, cause unknown. |
| 134334–134337 | 25 | scienceObject | all | 3 | Twilight sky | WARN | `medianDeltaY` | Run25 SM3, after the calibration block: lines 1.4 px off detectorMap_calib in y, cause unknown. |
| 134880–134886 | 25 | scienceObject | all | 3 | Twilight sky | WARN | `medianDeltaY` | Run25 SM3, after the calibration block: lines 1.4 px off detectorMap_calib in y, cause unknown. |
| 135275–135279 | 25 | scienceObject | all | 3 | Twilight sky | WARN | `medianDeltaY` | Run25 SM3, after the calibration block: lines 1.4 px off detectorMap_calib in y, cause unknown. |

### Placeholders

| Visits | Run | Type | Arms | Spectrographs | Sequence | Expect | Metric | Reason |
|---|---|---|---|---|---|---|---|---|
| *to find* |  |  | all | all | any | FAIL | `pctSaturatedPixels` | Saturated exposure; peak pixels flagged SAT. |
<!-- END GENERATED: validationVisits tables -->

## Changing the set

1. Check the visits in the run's calibration summary: `sps_camera_name` says which arms
   and spectrographs were read, and the sequence `name` is normally the `W_SEQNAM` that
   `seqType` must equal, character for character. The summary is typed by hand: when in
   doubt, read `W_SEQNAM` from a header, or leave `seqType` out of an entry whose visits
   are a single sequence.
2. Edit the YAML. Every entry needs `visit` or `visitRange`, and the `sequenceType` from the
   summary (`scienceArc`, `scienceTrace` or `scienceObject`): it decides how the visit is
   reduced, so an entry covers visits of one type only. A `known_bad` entry needs
   `expect` and should name a `metric` and a `reason`. Restrict `arms`, `spectrographs` and
   `seqType` to what was actually read: an omitted selector means "every value".
3. Regenerate the tables above and check the loader still accepts the file:

   ```bash
   python -m pfs.drp.qa.metrics.validationVisits tables
   pytest tests/metrics/test_validationVisits.py tests/metrics/test_coverage.py
   ```

   A new known-bad entry must be caught by the shipped thresholds on the stored metrics
   (`test_coverage.py`), or carry `unchecked` with the reason. Until its visits are in the
   stored metrics (`tests/metrics/data`, remade by `makeGateFixtures.py` and
   `makeDriftFixture.py` there), the test fails it as `no rows`.

   Paste the output between the `GENERATED` markers in this file.
4. Update the prose in [What is in it](#what-is-in-it) if a new kind of visit was added.
5. Rerun the threshold derivation ([`qa-thresholds.ipynb`](qa-thresholds.ipynb)): a
   new known-bad visit has to be caught by the current thresholds, and a new known-good one
   changes them.
