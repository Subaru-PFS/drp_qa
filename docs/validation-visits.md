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

## Where the visits come from

Every entry was checked against the run's calibration summary (`calib_data.csv`, one row
per sequence: `sequence_type`, sequence name, cameras read, notes) for Runs 25, 27 and
30. Run27 took calibration data only. A `known_bad` reason quotes the summary's note
where there is one. The summaries are written by hand, so a sequence name there can
differ from the `W_SEQNAM` header that `seqType` is matched against. The threshold
notebook lists any entry that matches no data, which is how such a mismatch shows up.

## What belongs in the set

A visit belongs only if it has a **verdict**: someone has established that it should pass,
or that it should warn or fail for a stated reason.

- **`known_good`:** expected to pass every metric. These are the samples that thresholds
  are derived from, so they must span the populations being gated: every arm, arcs of every
  lamp, traces, and sky frames.
- **`known_bad`:** expected to `WARN` or `FAIL`. Each entry names the metric that should
  catch it, because a known-bad visit must fail *for the right reason*. An entry holds only
  for the metric it names; it says nothing about the others.
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

- **Run25 stable calibration set (2025-11-10/11):** the reference for every threshold.
  Fixed pointing (alt 90, az 90, insrot 0); Ar, Xe, Ne, Kr and HgCd arcs and quartz traces.
  Block A read the b, r and n arms; block B read b and m.
- **Run25 block C (2025-12-01):** the same sequence at azimuth 168. It is a second epoch
  under a different gravity vector, and it brings b-arm HgCd above the 20-sample floor.
- **Run25 uniformity traces (2025-11-12 and 11-21):** extra quartz frames, so that trace
  widths can have thresholds of their own rather than borrowing the arcs'.
- **Same-exposure controls (Run27):** in each SM1-defocused and unlit-spectrograph
  exposure, the spectrographs the note doesn't name. A metric must fail the faulty
  spectrograph and pass the others in the same frames. The SM1 controls of 2026-03-15
  and 03-17 assume SM1 was refocused after the "SM1 defocused" notes of 03-09/10.
- **Twilight sky, Run25 (2025-11-25 and 11-28; one set at insrot 90) and Run30
  (2026-09-04 and 09-16):** the reference for a future sky-line check (PIPE2D-1925).
  Whether they should also be held to the FWHM and flag-rate gates is not yet
  established. The Run30 set of 2026-09-02 is left out: it was taken near the Moon.

### Known bad

- **Cloudy twilight (Run25):** image quality degraded by observing conditions; `medFwhm`.
- **SM1 defocused (Run27, 2026-03-09/10, spectrograph 1):** the seven arc and trace
  sequences noted "SM1 defocused", with FWHM 3.83–4.86 px across every lamp type;
  `medFwhm`. See "SM1 exception" in [`qa-domain-notes.md`](qa-domain-notes.md). The
  other visits in 140005–140138 carry no verdict and are not included: darks, the
  `sky1540+3500` frames, and visits the summary does not list (among them the raster
  scans 140106–140112 and 140115–140121).
- **Unlit spectrograph (Run27 nightly `ImageQuality` arcs):** exposures noted "No light on
  SM1" or "SM2". The unlit spectrograph's detectors have no lines; `nLines`, `FAIL`.
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
| 148908–148911 | b, r, n | 1, 2, 3, 4 | Twilight sky | PASS |  | Run30 twilight sky, 2026-09-04, alt 80, az 270. 180 s. |
| 150840–150844 | b, r, n | 1, 2, 3, 4 | Twilight sky | PASS |  | Run30 twilight sky, 2026-09-16, alt 80, az 267. 180 s. |
| 140005–140006 | b, r, n | 2, 3, 4 | Arc: Argon | PASS |  | Run27 control for SM1 defocused, same exposures. |
| 140032–140035 | b, r, n | 2, 3, 4 | Trace | PASS |  | Run27 control for SM1 defocused, same exposures. |
| 140124–140126 | b, r, n | 2, 3, 4 | Arc: Argon | PASS |  | Run27 control for SM1 defocused, same exposures. |
| 140127–140129 | b, r, n | 2, 3, 4 | Arc: Xenon | PASS |  | Run27 control for SM1 defocused, same exposures. |
| 140130–140132 | b, r, n | 2, 3, 4 | Arc: Neon | PASS |  | Run27 control for SM1 defocused, same exposures. |
| 140133–140135 | b, r, n | 2, 3, 4 | Arc: Krypton | PASS |  | Run27 control for SM1 defocused, same exposures. |
| 140136–140138 | b, r, n | 2, 3, 4 | Arc: HgCd | PASS |  | Run27 control for SM1 defocused, same exposures. |
| 139971 | b, r, n | 2, 3, 4 | ImageQuality | PASS |  | Run27 control for SM1 unlit, same exposure. |
| 140477 | b, r, n | 2, 3, 4 | ImageQuality | PASS |  | Run27 control for SM1 unlit, same exposure. |
| 140640 | b, r, n | 1, 3, 4 | ImageQuality | PASS |  | Run27 control for SM2 unlit; SM1 after its 03-11 slit home update, same exposure. |
| 140649 | b, r, n | 3, 4 | ImageQuality | PASS |  | Run27 control for SM1 and SM2 unlit, same exposure. |
| 140651 | b, r, n | 1, 3, 4 | ImageQuality | PASS |  | Run27 control for SM2 unlit; SM1 after its 03-11 slit home update, same exposure. |

### Known bad

| Visits | Arms | Spectrographs | Sequence | Expect | Metric | Reason |
|---|---|---|---|---|---|---|
| 134334–134337 | b, r, n | 1, 2, 3, 4 | Twilight sky | FAIL | `medFwhm` | Cloudy; 180 s twilight sky taken through cloud. |
| 140005–140006 | b, r, n | 1 | Arc: Argon | FAIL | `medFwhm` | SM1 defocused (Run27). 15 s, az 90. |
| 140032–140035 | b, r, n | 1 | Trace | FAIL | `medFwhm` | SM1 defocused (Run27). 30 s, az 226. |
| 140124–140126 | b, r, n | 1 | Arc: Argon | FAIL | `medFwhm` | SM1 defocused (Run27). 15 s, az 336. |
| 140127–140129 | b, r, n | 1 | Arc: Xenon | FAIL | `medFwhm` | SM1 defocused (Run27). 45 s, az 336. |
| 140130–140132 | b, r, n | 1 | Arc: Neon | FAIL | `medFwhm` | SM1 defocused (Run27). 5 s, az 336. |
| 140133–140135 | b, r, n | 1 | Arc: Krypton | FAIL | `medFwhm` | SM1 defocused (Run27). 70 s, az 336. |
| 140136–140138 | b, r, n | 1 | Arc: HgCd | FAIL | `medFwhm` | SM1 defocused (Run27). 45 s, az 336. |
| 139971 | b, r, n | 1 | ImageQuality | FAIL | `nLines` | SM1 lamp didn't turn on. (Run27, 1 s arc) |
| 140477 | b, r, n | 1 | ImageQuality | FAIL | `nLines` | No light on SM1. (Run27, 1 s arc) |
| 140640 | b, r, n | 2 | ImageQuality | FAIL | `nLines` | No light on SM2. (Run27, 1 s arc) |
| 140649 | b, r, n | 1, 2 | ImageQuality | FAIL | `nLines` | No light on SM1 or SM2. (Run27, 1 s arc) |
| 140651 | b, r, n | 2 | ImageQuality | FAIL | `nLines` | No light on SM2. (Run27, 1 s arc) |
| 150115–150116 | b, r, n | 1, 2, 3, 4 | any | FAIL | `pctFlagged` | M1 cover closed. |
| 150641–150642 | b, r, n | 1, 2, 3, 4 | any | FAIL | `pctFlagged` | Mirror cover and top screen both closed; noted in the calibration summary as to be ignored. |
| 149217 | b, m | 1, 2, 3, 4 | Trace | WARN | `nLines` | Only group 2 was illuminated. |
| 149398 | b, r, n | 1, 2, 3, 4 | Arc: Neon | WARN | `nLines` | Only group 3 was illuminated. |
| 149881 | b, m | 1, 2, 3, 4 | Arc: Neon | WARN | `nLines` | Only group 2 was illuminated. |
| 150367 | b, r, n | 1, 2, 3, 4 | Arc: Neon | WARN | `nLines` | Only group 4 was illuminated. |
| 150413 | b, r, n | 1, 2, 3, 4 | Arc: Neon | WARN | `nLines` | Only group 1 was illuminated. |
| 150779 | b, m, n | 1, 2, 3, 4 | any | WARN | `medDxCenter` | DetectorMap test pre-exposure (Helium). Alt 45, az 54, insrot 9; flexure against a zenith-derived detectorMap_calib. |
| 150782 | b, m, n | 1, 2, 3, 4 | any | WARN | `medDxCenter` | DetectorMap test post-exposure (Helium). Alt 49, az 53, insrot 5; flexure against a zenith-derived detectorMap_calib. |

### Unconfirmed

| Visits | Arms | Spectrographs | Sequence | Expect | Metric | Reason |
|---|---|---|---|---|---|---|
| 150661 | b, r, n | 1, 2, 3, 4 | Trace | FAIL |  | Noted "Mirror cover open, but screen is wrong position" in the calibration summary. Probably a failure; neither the verdict nor the metric that should catch it has been established. |
| 149883 | b, r, n | 1, 2, 3, 4 | Arc: Neon | FAIL |  | Noted "Home" in the calibration summary, a note whose meaning is not settled. Other visits carry the same tag. Recorded so its metrics can be compared against the Run25 set; probably a failure, but unestablished. |

### Placeholders

| Visits | Arms | Spectrographs | Sequence | Expect | Metric | Reason |
|---|---|---|---|---|---|---|
| *to find* | all | all | any | WARN | `medDxCenter` | Reduced against a stale detectorMap_calib; bulk spatial offset. |
| *to find* | all | all | any | FAIL | `pctSaturatedPixels` | Saturated exposure; peak pixels flagged SAT. |
<!-- END GENERATED: validationVisits tables -->

## Changing the set

1. Check the visits in the run's calibration summary: `sps_camera_name` says which arms
   and spectrographs were read, and the sequence `name` is normally the `W_SEQNAM` that
   `seqType` must equal, character for character. The summary is typed by hand: when in
   doubt, read `W_SEQNAM` from a header, or leave `seqType` out of an entry whose visits
   are a single sequence.
2. Edit the YAML. Every entry needs `visit` or `visitRange`. A `known_bad` entry needs
   `expect` and should name a `metric` and a `reason`. Restrict `arms`, `spectrographs` and
   `seqType` to what was actually read: an omitted selector means "every value".
3. Regenerate the tables above and check the loader still accepts the file:

   ```bash
   python -m pfs.drp.qa.metrics.validationVisits tables
   pytest tests/metrics/test_validationVisits.py
   ```

   Paste the output between the `GENERATED` markers in this file.
4. Update the prose in [What is in it](#what-is-in-it) if a new kind of visit was added.
5. Rerun the threshold derivation ([`qa-thresholds.ipynb`](qa-thresholds.ipynb)): a
   new known-bad visit has to be caught by the current thresholds, and a new known-good one
   changes them.
