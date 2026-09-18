# Jira ticket draft — calexp cross-dispersion measurement fails on quartz in every arm

> **Status (2026-09-18): resolved on `tickets/PIPE2D-1391-01`, not yet wired in.**
> `pfs.drp.qa.crossDispersion.measureRow` replaces the failing estimator. On real Run25
> quartz 133040 it raises the usable fraction from 0.157 % to 92 % (b2) and from 0.004 % to
> 95.7 % (r2). Measured fiber pitch is 6.17 px on every arm, confirming the aperture
> hypothesis below. Open: r2 reads 10 % wider than its calibration, b2 agrees to 0.5 %.


**Component:** drp_qa
**Affects:** all versions to date
**Related:** the ongoing b-arm investigation — though this turns out not to be b-specific
**Found:** verifying PIPE2D-1391-01 against Run25

---

## Summary

`ImageQualityQaTask._buildImageWidthData` rejects essentially every cross-dispersion sample
it takes on quartz frames — **in every arm, not only b**. Across the Run25 trace visits
(104 quanta, 10 visits, 2025-11-10 to 2025-11-21):

| Arm | Quanta | Median usable | Best quantum |
|---|---|---|---|
| m | 8 | 0.004 % | 0.164 % |
| r | 32 | 0.006 % | 0.168 % |
| b | 40 | 0.072 % | 0.228 % |
| n | 24 | 0.101 % | 0.232 % |

**No quantum anywhere reached 1 % usable.** Every one of the 104 fell back to the fiber
profile calibration. Denominators are ~51 700 samples per quantum (50 512 on n).

This was first spotted on b and filed as a b-arm problem. The full run shows the b arm is
not even the worst: r and m reject an order of magnitude more samples than b and n.

## It is not a data problem

The same detector, same visit, measured through the fiber-profile path instead, gives
`medFwhm = 3.03 px` from 15 522 samples — a perfectly ordinary trace width. The traces are
present, well-formed, and measurable. What fails is `_buildImageWidthData` specifically.

Until PIPE2D-1391-01 this was invisible: the fiber-profile fallback was unreachable when a
calexp existed, so the visit simply went sparse and reported `PASS` on a vacuous metric.
The fallback now covers it, which makes the QA verdict correct but **masks the underlying
measurement failure**. Hence this ticket.

## Mechanism, and the primary hypothesis

`_buildImageWidthData` cuts a strip of `2 * profileHalfWidth + 1` pixels (15 by default)
centred on each fiber, then:

1. estimates the background from the **outermost 2 pixels on each side** of the strip;
2. estimates per-pixel noise as `bg_rms = std(those 4 pixels)`;
3. accepts the sample only if `peak_val >= minPeakSN * bg_rms`, with `minPeakSN = 5.0`.

**Hypothesis: with `profileHalfWidth = 7`, the strip overlaps the neighbouring fibers, so
the "background" pixels are sampling adjacent traces rather than sky.**

If that is what is happening, every step compounds:

- `bg` is overestimated, because it is another fiber's flux;
- `bg_rms` is inflated, because those 4 pixels straddle the flank of a neighbouring trace
  rather than flat background;
- `peak_val = (strip - bg).max()` is correspondingly suppressed;
- the `peak_val >= 5 * bg_rms` gate then fails on a perfectly good trace.

**This predicts the arc/quartz asymmetry, and it does not predict a b-arm effect.** A
quartz frame illuminates every fiber simultaneously with continuum, so every neighbour is
bright at every row and contamination is maximal — on any arm whose fiber pitch is below
the aperture width. An arc illuminates discrete lines, so most rows have genuine dark
background between them and the same aperture behaves far better. The arc quanta in the
same run produced no calexp warnings at all.

What the hypothesis does *not* obviously explain is the arm ordering: r and m rejecting
roughly ten times more than b and n. Pitch differences between arms might; so might
differing trace widths, or a noise estimate that behaves differently per arm. That split
is the most informative thing to explain.

## Diagnostics to confirm or kill it

1. **Measure the actual inter-fiber spacing** and compare it against `profileHalfWidth`:
   ```python
   dm = butler.get("detectorMap", dataId={"instrument": "PFS", "visit": 133040,
                                          "arm": "b", "spectrograph": 2})
   import numpy as np
   y = dm.getBBox().getHeight() // 2
   x = np.sort([dm.getXCenter(fid, y) for fid in dm.fiberId])
   print("median fiber pitch:", np.median(np.diff(x)), "px  vs halfWidth 7")
   ```
   A median pitch below ~15 px means the 15-pixel strip reaches its neighbours, and a
   pitch below ~10 px means the background pixels are sitting on them.

2. **Compare arms.** Run the pitch check over b, r, n and m. If r and m have the tightest
   pitch relative to their trace width, that explains the ordering above and confirms the
   hypothesis. If pitch is similar across arms, the aperture is not the whole story.

3. **Sweep `profileHalfWidth`.** Re-run `imageQualityQa` on 133040/b2 with
   `-c imageQualityQa:profileHalfWidth=3` and `=5`. If the usable fraction jumps, the
   aperture is the cause.

4. **Sweep `minPeakSN`** with `-c imageQualityQa:minPeakSN=2`. If it barely moves while the
   aperture sweep does, that confirms the background estimate rather than the threshold is
   what is wrong.

## Candidate fixes, in order of preference

1. **Derive the aperture from the detector map** rather than configuring it flat. Half the
   median inter-fiber spacing, minus a margin, is the natural bound: an aperture that
   reaches a neighbour cannot measure a background.
2. **Estimate the background between traces**, at the inter-fiber midpoints, instead of from
   the edge of an aperture that may not clear the neighbour.
3. **Estimate the noise properly.** `std` of 4 pixels carries roughly 40 % uncertainty even
   when those pixels *are* clean background — it is a weak basis for a 5-sigma gate. The
   calexp variance plane is right there and is the better source.
4. Only then revisit `minPeakSN`. Lowering it without fixing the background trades silent
   rejection for silent acceptance of contaminated samples, which is worse.

## Acceptance

- The usable fraction on Run25 quartz is a sensible number in every arm (tens of percent or
  better), or there is a written explanation of why those frames genuinely cannot be
  measured this way.
- A regression test over a synthetic densely-packed frame, asserting the measurement
  survives fiber spacing at the b-arm pitch.
- `imageQualityQa` stops relying on the fiber-profile fallback for ordinary quartz, and
  trace `medFwhm` varies between visits that share a calibration.

## Consequence found downstream: trace FWHM is a calibration constant

Because every quartz quantum falls back to `fiberProfiles`, every trace `medFwhm` in the
collection is read from a **calibration product**, not measured on the exposure. Every
visit that shares a calib reports the same value per detector. It shows up plainly when
thresholds are derived: every trace group comes out with p95 equal to p99 to four
digits, because the distribution is a handful of repeated calib values.

So until this is fixed, trace-path image quality cannot detect a change in the exposure
at all, and a threshold cannot be derived for it.

## Why it matters beyond the warning

The calexp path is the **primary** measurement for trace/quartz visits and the documented
fallback for arcs with too few good lines. If it is unusable on the b arm, then b-arm image
quality rests entirely on the fiber-profile widths — which come from the *calibration*, not
from the exposure being assessed. That is a weaker claim than it appears: a QA metric
sourced from a calib cannot detect a change in the thing it is supposed to be monitoring.
