# Deriving QA thresholds

Step by step: measure the [validation visits](validation-visits.md) with the current code,
derive `WARN`/`FAIL` thresholds from the known-good visits, check them against the
known-bad ones, and adopt them. Run it whenever the metrics code, the visit set or the
instrument changes. The rules behind it are in [`qa-principles.md`](qa-principles.md).

Steps 1–3 write one new collection to the Butler; everything after that only reads.

## 0. Set up

On a machine with the LSST stack and the Butler (`/work/datastore`):

```bash
source /path/to/stack/loadLSST.bash
setup -r /path/to/drp_stella
setup -r /path/to/drp_qa
cd /path/to/drp_qa
```

## 1. Name the run and list the visits

```bash
export QA_VALIDATION_COLLECTION="u/$USER/qa-validation/$(date +%Y%m%d)"
export QA_VALIDATION_VISITS="$(python -m pfs.drp.qa.metrics.validationVisits expression)"
echo "$QA_VALIDATION_COLLECTION"
echo "$QA_VALIDATION_VISITS"
```

The visit list comes from the YAML, so it is never typed by hand. Keep this shell open:
steps 2–4 use both variables.

## 2. Check the inputs exist

`imageQualityQa` reads the `fitDetectorMap` outputs (`lines`, `detectorMap`) for each
visit. Build the graph without running it:

```bash
pipetask qgraph -b /work/datastore \
    -p pipelines/drpQA.yaml#imageQualityQa \
    -i drpActor/reductions,PFS/defaults \
    -d "instrument = 'PFS' AND $QA_VALIDATION_VISITS"
```

The log reports how many quanta the graph has: one per detector and visit, up to about
2,000 for the current set (124 visits). If it is far fewer, some visits lack `fitDetectorMap` outputs in
those input collections. Find them, or reduce them first, before going on. Step 4 lists any
visit that ends up with no metrics.

## 3. Run `imageQualityQa`

```bash
pipetask --long-log --log-level PFS=INFO run -j 8 -b /work/datastore \
    -p pipelines/drpQA.yaml#imageQualityQa \
    -i drpActor/reductions,PFS/defaults \
    -o "$QA_VALIDATION_COLLECTION" \
    -d "instrument = 'PFS' AND $QA_VALIDATION_VISITS"
```

Use a new collection for every run. Never derive thresholds from a collection that mixes
code versions, such as `qaActor/reductions`.

From a notebook with a `run_pipetask` helper, the same command is:

```python
import os

cmd = ["pipetask", "--long-log", "--log-level", "PFS=INFO", "run", "-j", "8",
       "-b", "/work/datastore", "-p", "pipelines/drpQA.yaml#imageQualityQa",
       "-i", "drpActor/reductions,PFS/defaults",
       "-o", os.environ["QA_VALIDATION_COLLECTION"],
       "-d", f"instrument = 'PFS' AND {os.environ['QA_VALIDATION_VISITS']}"]
run_pipetask(cmd)
```

## 4. Run the threshold notebook

In the same shell:

```bash
jupyter nbconvert --to notebook --execute --inplace docs/qa-thresholds.ipynb
```

Or open [`qa-thresholds.ipynb`](qa-thresholds.ipynb) in Jupyter, from a shell where
`QA_VALIDATION_COLLECTION` is set, and run all cells. The notebook reads the Butler
read-only. It caches the metrics as `~/.cache/drp_qa/iqQaMetrics-<collection>.parquet`
(set `DRP_QA_CACHE` to change the directory), and later runs read the cache.

## 5. Read the results

For each metric the notebook shows a table, one row per population (arm and observation
type; flag rates also per lamp; line counts per lamp), and one plot panel per row. Go
through these in order:

1. **Missing visits and unmatched entries.** The notebook lists validation visits with no
   metrics, and entries that match no row. Either is a check that didn't run. An entry
   whose visits are present but which matches nothing has a wrong selector: compare its
   `seqType` with the `seqName` column, which must agree character for character.
2. **Known-bad checks.** In each population with known-bad rows, `badOk` must be true. If
   it isn't, the threshold doesn't separate the fault: look at the panel. A marker on the
   good side of `FAIL` (left of it; right of it for `nLines`) is a missed fault. Either the
   metric doesn't see this fault, or the visit isn't bad for the reason recorded. Fix
   whichever is wrong, not the threshold.
3. **Sample size.** `reliable` is false below 20 values. `nGoodVisits` is the more honest
   count: the detectors of one visit, and back-to-back exposures, are correlated.
4. **`failBounded`.** When false, the sample is too small to bound the FAIL percentile
   (about 370 values are needed for p99), so `FAIL` sits at the sample's extreme. Usable,
   but expect it to move as data accumulate. Say so in the config `doc`; the provenance
   sentence already does.
5. **`degenerate`.** `WARN` is not below `FAIL`, so `WARN` can never fire. This comes from
   ties in the tail; set the pair by hand from the plot.
6. **In-sample flag rates.** `goodFlaggedWarn` and `goodFlaggedFail` should be near 5 % and
   1 %. Much more means ties at the threshold.
7. **Unconfirmed visits.** `unconfirmedFlagged` counts the suspected faults the suggestion
   would flag. Use it to settle them in the YAML ([`validation-visits.md`](validation-visits.md)),
   not to move a threshold.
8. **Populations without good data.** Rows with `nGood` 0 have known-bad visits and nothing
   to compare them with. Add known-good visits for that population, or accept that it
   goes unchecked.

## 6. Adopt the thresholds

The thresholds are config fields of `ImageQualityQaConfig` in
[`python/pfs/drp/qa/imageQualityQa.py`](../python/pfs/drp/qa/imageQualityQa.py):

| Metric | Fields | Granularity | Populations it gates |
|---|---|---|---|
| `medFwhm` | `fwhmWarnThreshold`, `fwhmFailThreshold` | one value | every arm, except traces (not gated until PIPE2D-1917) |
| `medDxCenter` (absolute) | `dxCenterWarnThreshold`, `dxCenterFailThreshold` | one value | every arm and type |
| `pctFlagged` | `flagRateWarnThreshold`, `flagRateFailThreshold` | per `arm` or `arm:species` | as keyed |

1. Where a field holds one value but the notebook gives one per population, use the most
   lenient suggestion among the populations that field gates (the last column). Anything
   stricter fails good detectors in the widest population. Note the populations in the
   `doc`. The spread between populations is what a single value costs: a fault that
   reaches only the lenient population's threshold goes unflagged in the others.
2. Paste the notebook's provenance sentence (its last cell) into the field's `doc`.
3. Update the threshold tables in the README (`imageQualityQa` → *Pass/Warn/Fail
   Thresholds*) and add a line under `## [Unreleased]` in `CHANGELOG.md`.
4. Rerun steps 3–5 into a new collection with the new defaults, and check that every
   known-good visit passes and every known-bad one gets its expected verdict.

## 7. Commit

On the ticket branch, commit the config change and the notebook *with its outputs*. The
outputs are the record of what the thresholds were derived from.

```bash
git add python/pfs/drp/qa/imageQualityQa.py README.md CHANGELOG.md docs/qa-thresholds.ipynb
git commit
```

The notebook shows only validation visits, which are engineering and calibration visits,
so its outputs are fine to publish.
