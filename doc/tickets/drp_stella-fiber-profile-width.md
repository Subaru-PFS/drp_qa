# Jira ticket draft — drp_stella: fiber-profile width issues

**Component:** drp_stella
**Found:** investigating trace-width QA in drp_qa (PIPE2D-1391-01)

Three issues, in decreasing order of confidence. The first is a regression with a
one-line fix; the other two are context for it and can be split out.

`fitSwathProfiles` itself is sound — a joint conjugate-gradient fit of every fiber's
oversampled profile across a swath, which handles overlapping neighbours properly.
Nothing here is a defect in it. #1 is in the loop that feeds it, and #2 is in how its
output is summarised.

---

## 1. `buildFiberProfiles` takes the square root of a sigma (regression)

**`python/pfs/drp/stella/buildFiberProfiles.py:296`**

```python
width = np.array([profiles[ff].calculateStatistics().width for ff in profiles])
select = np.isfinite(width) & (width > 0)
sigma = np.median(np.sqrt(width[select]))
```

`calculateStatistics().width` is already a sigma (`fiberProfile.py:304` takes the square
root of the second moment), so `sigma` here is √σ.

### How it happened

| Commit | Date | Change |
|---|---|---|
| `aa13ca96` | 2023-07-31 | Added `sigma = median(sqrt(width))` here. **Correct at the time**: `width` was then a variance, `((xx - centroid)**2 * profiles).sum()/norm`, with no square root. |
| `5b882ab3` | 2024-03-21 | "fiberProfiles: expand metrics and plotting" changed `width` to `np.sqrt(...)`, a sigma, and did not touch this caller. |

This is the only caller of `calculateStatistics().width` in drp_stella.

### Effect

`sigma` sets the Gaussian approximations (`fluxProfiles`) used to extract spectra on every
`extractIter` iteration of `runMultiple`. Those spectra are then the fixed per-fiber norms
that `fitSwathProfiles` solves profile shapes against.

√σ is narrower than σ for σ > 1 px:

| True sigma | Used | Error |
|---|---|---|
| 1.10 px | 1.049 | −5 % |
| 1.30 px | 1.140 | −12 % |
| 1.55 px | 1.245 | −20 % |

PFS trace sigma is about 1.1–1.55 px (FWHM 2.6–3.7), so the extraction Gaussian has been
5–20 % too narrow since 2024-03-21. A roughly constant per-fiber bias is probably absorbed
into profile normalisation rather than shape, so the downstream effect may be modest —
**that is inference, not a measurement**, and the first thing to check.

### Fix

```python
sigma = np.median(width[select])
```

### Acceptance

- `sigma` at `buildFiberProfiles.py:296` equals the median profile sigma, not its root.
- A test building profiles from a synthetic quartz of known sigma asserts the extraction
  Gaussian's sigma matches it. The regression survived 18 months because nothing checks it.
- Compare profiles built before and after on one quartz set, to size the real effect.

---

## 2. `calculateStatistics().width` is sensitive to residual background

**`python/pfs/drp/stella/fiberProfile.py:304`**

`width` is an unsmoothed second moment over the full ±`profileRadius` window. The x²
weighting puts most of the weight on the wings, so any residual background, scattered
light or leftover neighbour light under the profile inflates it. The centroid in the same
function is measured robustly (convolved, then `centroidPeak`); the width is not.

Measured on a synthetic trace, sigma = 1.3 px, with a *perfectly separated* profile — the
best case, as `fitSwathProfiles` would produce — and only a constant residual background
added:

| Residual background | Second-moment width | Gaussian fit with a background term |
|---|---|---|
| 0 % of peak | −0.1 % | +1.0 % |
| 1 % | +5.6 % | +1.3 % |
| 2 % | +10.7 % | +1.3 % |
| 5 % | +23.4 % | +1.2 % |

The second moment grows about 5 % per 1 % of residual background; the fit is flat. To be
fair to it, the second moment is slightly *more* accurate on a truly clean profile.

**Why it matters:** this width is what drp_qa's image-quality task falls back to when it
cannot measure the exposure, and it is how the calibration's trace width is reported. A
calib widened by residual background reads as a broader PSF.

**Suggested:** fit a Gaussian with a constant background over the central profile, or
restrict the moment to the core and correct for truncation. Either is a change to a
reported quantity, so it needs its own decision.

---

## 3. `runSingle` builds profiles without modelling neighbours (latent)

**`python/pfs/drp/stella/fiberProfile.py:144`**, via `BuildFiberProfilesTask.runSingle`

`FiberProfile.fromImage` stacks pixels within ±`profileRadius` of each fiber's center,
per swath, with no model of the neighbours. `profileRadius` defaults to **5 px**, but
measured PFS fiber pitch is **6.17 px** on every arm (Run25), so half the pitch is 3.08 px.
On a quartz frame, where every fiber is lit, each profile's window reaches onto both
neighbours' flanks. `calculateStatistics().width` then comes out about 2× the true value
(simulated: FWHM 5.9 px for a true 3.06 px).

`runMultiple` does not have this problem, because `fitSwathProfiles` fits the neighbours
jointly. And production calibs are evidently built that way: their trace FWHM is
2.6–3.7 px, not 5–6. So **nothing in current use is known to be affected** — but anything
building profiles from a single exposure through `run()` gets contaminated profiles
without warning.

**Suggested:** either route single exposures through `fitSwathProfiles` too, or warn when
`profileRadius` exceeds half the local fiber pitch.

---

## Minor: `fitAmplitudes` evaluates the Gaussian at pixel centers

**`src/profile.cc:675`** — `std::exp(-0.5*std::pow(radius, 2))` samples the Gaussian at
each pixel center rather than integrating it over the pixel. At sigma ≈ 1.3 px that is
about a 2.5 % effective-width mismatch against the data being fitted. An approximation
rather than a bug; noted for completeness.
