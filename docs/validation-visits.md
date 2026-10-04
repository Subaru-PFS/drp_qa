# The validation visit set

The validation visit set is a fixed list of PFS visits whose QA verdict is known. Every
threshold and every metric is checked against it. A metric that flags a known-good visit,
or passes a known-bad one, does not merge.

- **Source of truth:**
  [`python/pfs/drp/qa/metrics/data/validationVisits.yaml`](../python/pfs/drp/qa/metrics/data/validationVisits.yaml),
  shipped with the package. Its header comment defines the entry schema.
- **Loader:** `pfs.drp.qa.metrics.validationVisits.loadValidationVisits()`.
- **Using it:** [`deriving-thresholds.md`](deriving-thresholds.md) is the step-by-step
  procedure: run `imageQualityQa` over these visits, then derive and check thresholds.

## What belongs in the set

A visit belongs only if it has a **verdict**: someone has established that it should pass,
or that it should warn or fail for a stated reason.

- **`known_good`:** expected to pass every metric. These are the samples that thresholds
  are derived from, so they must span the populations being gated: every arm, arcs of every
  lamp, traces, and sky frames.
- **`known_bad`:** expected to `WARN` or `FAIL`. Each entry names the metric that should
  catch it, because a known-bad visit must fail *for the right reason*. An entry holds only
  for the metric it names; it says nothing about the others.
- **`unconfirmed: true`:** a suspected fault, flagged at the telescope, whose numbers no one
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

- **Run25 stable calibration set (2025-11-10/11):** the reference for every threshold.
  Fixed pointing (alt 90, az 90, insrot 0); Ar, Xe, Ne, Kr and HgCd arcs and quartz traces.
  Block A read the b, r and n arms; block B read b and m.
- **Run25 block C (2025-12-01):** the same sequence at azimuth 168. It is a second epoch
  under a different gravity vector, and it brings b-arm HgCd above the 20-sample floor.
- **Run25 uniformity traces (2025-11-12 and 11-21):** extra quartz frames, so that trace
  widths can have thresholds of their own rather than borrowing the arcs'.
- **Run25 twilight sky (2025-11-25 and 11-28; one set at insrot 90):** the reference for a
  future sky-line check (PIPE2D-1925). Whether they should also be held to the FWHM and
  flag-rate gates is not yet established.

### Known bad

- **Cloudy twilight (Run25):** image quality degraded by observing conditions; `medFwhm`.
- **SM1 focus range (140005–140138, spectrograph 1):** a documented optics fault, with FWHM
  3.83–4.86 px across every lamp type; `medFwhm`. See "SM1 exception" in
  [`qa-domain-notes.md`](qa-domain-notes.md).
- **Obstructed frames (Run30):** taken with the M1 cover or the top screen closed;
  `pctFlagged`.
- **Partial slit illumination (Run30):** only one fiber group lit. Shapes are fine and the
  line count is not; `nLines`, `WARN`.
- **Off-zenith Helium detectorMap tests (Run30, alt 45–49):** a real flexure offset against
  a zenith-derived `detectorMap_calib`; `medDxCenter`, `WARN` (provisional). They are the
  reference case for drift monitoring (PIPE2D-1921).

### Gaps

- A visit reduced against a stale `detectorMap_calib` (for PIPE2D-1921).
- A saturated frame (for PIPE2D-1922's saturation metric).
- m-arm traces: only the short block B and C sequences exist, below the sample floor.

## Every entry

Generated from the YAML by `python -m pfs.drp.qa.metrics.validationVisits tables`.
`tests/metrics/test_validationVisits.py` fails if this section and the YAML disagree.

<!-- BEGIN GENERATED: validationVisits tables -->
### Known good

| Visits | Arms | Spectrographs | Sequence | Expect | Metric | Note |
|---|---|---|---|---|---|---|
| 133025–133027 | b, r, n | 1, 2, 3, 4 | Arc: Argon | PASS |  | Run25 stable set, block A. 10 s exposures. |
| 133028–133030 | b, r, n | 1, 2, 3, 4 | Arc: Xenon | PASS |  | Run25 stable set, block A. 45 s exposures. |
| 133031–133033 | b, r, n | 1, 2, 3, 4 | Arc: Neon | PASS |  | Run25 stable set, block A. 5 s exposures. |
| 133034–133036 | b, r, n | 1, 2, 3, 4 | Arc: Krypton | PASS |  | Run25 stable set, block A. 70 s exposures. |
| 133037–133039 | b, r, n | 1, 2, 3, 4 | Arc: HgCd | PASS |  | Run25 stable set, block A. 45 s exposures. HgCd gives dense, well-measured b-arm coverage, so b-arm flag rates here are genuine rather than lamp physics. |
| 133040–133041 | b, r, n | 1, 2, 3, 4 | Trace | PASS |  | Run25 stable set, block A. 20 s scienceTrace; exercises the calexp trace-width path. |
| 133042–133044 | b, m | 1, 2, 3, 4 | Arc: Argon | PASS |  | Run25 stable set, block B. 10 s exposures. |
| 133045–133047 | b, m | 1, 2, 3, 4 | Arc: Xenon | PASS |  | Run25 stable set, block B. 45 s exposures. |
| 133048–133050 | b, m | 1, 2, 3, 4 | Arc: Neon | PASS |  | Run25 stable set, block B. 5 s exposures. |
| 133051–133053 | b, m | 1, 2, 3, 4 | Arc: Krypton | PASS |  | Run25 stable set, block B. 45 s exposures. |
| 133054–133055 | b, m | 1, 2, 3, 4 | Trace | PASS |  | Run25 stable set, block B. 20 s scienceTrace. |
| 135828–135829 | b, r, n | 1, 2, 3, 4 | Trace | PASS |  | Run25 block C, az 168. 20 s scienceTrace. |
| 135830–135831 | b, r, n | 1, 2, 3, 4 | Arc: Argon | PASS |  | Run25 block C, az 168. 10 s exposures. |
| 135832–135833 | b, r, n | 1, 2, 3, 4 | Arc: Xenon | PASS |  | Run25 block C, az 168. 45 s exposures. |
| 135834–135835 | b, r, n | 1, 2, 3, 4 | Arc: Neon | PASS |  | Run25 block C, az 168. 5 s exposures. |
| 135836–135837 | b, r, n | 1, 2, 3, 4 | Arc: Krypton | PASS |  | Run25 block C, az 168. 70 s exposures. |
| 135838–135840 | b, r, n | 1, 2, 3, 4 | Arc: HgCd | PASS |  | Run25 block C, az 168. 45 s exposures. The second HgCd block; see above. |
| 135841–135842 | b, m | 1, 2, 3, 4 | Arc: Argon | PASS |  | Run25 block C, az 168, m arm. 10 s exposures. |
| 135843–135844 | b, m | 1, 2, 3, 4 | Arc: Xenon | PASS |  | Run25 block C, az 168, m arm. 45 s exposures. |
| 135845–135846 | b, m | 1, 2, 3, 4 | Arc: Neon | PASS |  | Run25 block C, az 168, m arm. 5 s exposures. |
| 135847–135848 | b, m | 1, 2, 3, 4 | Arc: Krypton | PASS |  | Run25 block C, az 168, m arm. 45 s exposures. |
| 135849–135850 | b, m | 1, 2, 3, 4 | Trace | PASS |  | Run25 block C, az 168, m arm. 20 s scienceTrace. |
| 134880–134883 | b, r, n | 1, 2, 3, 4 | Twilight sky | PASS |  | Run25 twilight sky, 180 s. Reference case for the O2/OH sky-line check. |
| 134884–134886 | b, r, n | 1, 2, 3, 4 | Twilight sky | PASS |  | Run25 twilight sky, 180 s. Reference case for the O2/OH sky-line check. |
| 133532–133533 | b, r, n | 1, 2, 3, 4 | Trace | PASS |  | Run25 uniformity set, 2025-11-12, az 90. 20 s scienceTrace. |
| 133536–133537 | b, r | 1, 2, 3, 4 | Trace | PASS |  | Run25 uniformity set, 2025-11-12, az 90. 20 s scienceTrace; n arm not read. |
| 134338–134339 | b, r, n | 1, 2, 3, 4 | Trace | PASS |  | Run25 uniformity set, 2025-11-21, az 290. 20 s scienceTrace. |
| 135275–135279 | b, r, n | 1, 2, 3, 4 | Twilight sky | PASS |  | Run25 uniformity set, 2025-11-28, az 290, insrot 90. 180 s. Reference case for the O2/OH sky-line check, as for the other twilight entries. |

### Known bad

| Visits | Arms | Spectrographs | Sequence | Expect | Metric | Reason |
|---|---|---|---|---|---|---|
| 134334–134337 | b, r, n | 1, 2, 3, 4 | Twilight sky | FAIL | `medFwhm` | Cloudy; 180 s twilight sky taken through cloud. |
| 140005–140138 | b, r, n, m | 1 | any | FAIL | `medFwhm` | SM1 focus/mirror alignment; FWHM 3.83-4.86 px across all lamp types. |
| 150115–150116 | b, r, n | 1, 2, 3, 4 | any | FAIL | `pctFlagged` | M1 cover closed. |
| 150641–150642 | b, r, n | 1, 2, 3, 4 | any | FAIL | `pctFlagged` | Mirror cover and top screen both closed; flagged at the telescope as to be ignored. |
| 149217 | b, m | 1, 2, 3, 4 | Trace | WARN | `nLines` | Only group 2 was illuminated. |
| 149398 | b, r, n | 1, 2, 3, 4 | Arc: Neon | WARN | `nLines` | Only group 3 was illuminated. |
| 149881 | b, m | 1, 2, 3, 4 | Arc: Neon | WARN | `nLines` | Only group 2 was illuminated. |
| 150367 | b, r, n | 1, 2, 3, 4 | Arc: Neon | WARN | `nLines` | Only group 4 was illuminated. |
| 150413 | b, r, n | 1, 2, 3, 4 | Arc: Neon | WARN | `nLines` | Only group 1 was illuminated. |
| 150779 | b, m, n | 1, 2, 3, 4 | DetectorMap test pre-exposure (Helium) | WARN | `medDxCenter` | Alt 45, az 54, insrot 9; flexure against a zenith-derived detectorMap_calib. |
| 150782 | b, m, n | 1, 2, 3, 4 | DetectorMap test post-exposure (Helium) | WARN | `medDxCenter` | Alt 49, az 53, insrot 5; flexure against a zenith-derived detectorMap_calib. |

### Unconfirmed

| Visits | Arms | Spectrographs | Sequence | Expect | Metric | Reason |
|---|---|---|---|---|---|---|
| 150661 | b, r, n | 1, 2, 3, 4 | Trace | FAIL |  | Tagged "Mirror cover open, but screen is wrong position". Probably a failure; neither the verdict nor the metric that should catch it has been established. |
| 149883 | b, r, n | 1, 2, 3, 4 | Arc: Neon | FAIL |  | Tagged "Home" at the telescope, a note whose meaning is not settled. Other visits carry the same tag. Recorded so its metrics can be compared against the Run25 set; probably a failure, but unestablished. |

### Placeholders

| Visits | Arms | Spectrographs | Sequence | Expect | Metric | Reason |
|---|---|---|---|---|---|---|
| *to find* | all | all | any | WARN | `medDxCenter` | Reduced against a stale detectorMap_calib; bulk spatial offset. |
| *to find* | all | all | any | FAIL | `pctSaturatedPixels` | Saturated exposure; peak pixels flagged SAT. |
<!-- END GENERATED: validationVisits tables -->

## Changing the set

1. Edit the YAML. Every entry needs `visit` or `visitRange`. A `known_bad` entry needs
   `expect` and should name a `metric` and a `reason`. Restrict `arms`, `spectrographs` and
   `seqType` to what was actually read: an omitted selector means "every value".
2. Regenerate the tables above and check the loader still accepts the file:

   ```bash
   python -m pfs.drp.qa.metrics.validationVisits tables
   pytest tests/metrics/test_validationVisits.py
   ```

   Paste the output between the `GENERATED` markers in this file.
3. Update the prose in [What is in it](#what-is-in-it) if a new kind of visit was added.
4. Rerun the threshold derivation ([`deriving-thresholds.md`](deriving-thresholds.md)): a
   new known-bad visit has to be caught by the current thresholds, and a new known-good one
   changes them.
