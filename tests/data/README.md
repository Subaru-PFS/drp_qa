# Test data

| File | What it is |
|---|---|
| `goldenVisits.yaml` | The golden visit set: visits with known QA verdicts. See its header and `AGENTS.md`. |
| `quartz_133040_b2.npz`, `quartz_133040_r2.npz` | Real quartz calexp rows from Run25 visit 133040, detectors b2 and r2. |

## Quartz exports

Exported 2026-09-18 from `/work/datastore`, collection `PFS/defaults`, to validate
`pfs.drp.qa.crossDispersion.measureRow` on real data. Each file holds the 84 rows that
`imageQualityQa` samples (`profileYStride=50`) for one detector:

| Key | Shape | Contents |
|---|---|---|
| `image`, `variance` | (84, 4096) float32 | calexp image and variance planes for those rows |
| `bad` | (84, 4096) bool | mask planes BAD, SAT, CR, NO_DATA |
| `xCenters` | (84, 616) float64 | detectorMap x-center of every fiber at each row |
| `fiberIds` | (616,) int32 | fiber IDs, matching `xCenters` columns |
| `rows` | (84,) int64 | the image rows exported |
| `calibWidth` | (616,) float64 | `fiberProfiles` `calculateStatistics().width` per fiber, median over swaths |
| `visit`, `arm`, `spectrograph` | scalars | identity |

`calibWidth` is **approximate**: the export took `np.nanmedian` of a masked array, which
ignores the mask, so masked swaths may be included.

Results with these files: the old estimator finds 0.157 % (b2) and 0.004 % (r2) of samples
usable; `measureRow` finds 92.0 % and 95.7 %. b2 FWHM 3.12 px against a calib 3.11 px;
r2 3.17 px against a calib 2.88 px, which is unexplained.
