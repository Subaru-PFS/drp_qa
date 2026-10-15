"""Compute the lit-fiber metrics from stored ``lines`` and pfsConfig, per image (PIPE2D-1935).

``u/wtg/qa-thresholds/004`` was measured before ``imageQualityQa`` counted the
fibers lit, so this computes ``linesPerFiber``, ``nLitFibers`` and ``nFibers``
from its ``lines`` and pfsConfig as the task does (`pfs.drp.qa.imageQualityQa.litFibers`),
without rerunning it. ``makeGateFixtures.py`` merges the result into the
validation fixture. Needs the stack and a Butler; run where the repository is:

- lit fibers: the detector's ``GOOD`` fibers, engineering fibers excepted;
  possible: those and the ones hidden behind a black dot (``BLACKSPOT``).
- ``linesPerFiber``: unflagged lamp-line rows (``description != "Trace"``) on lit
  fibers per lit fiber on an arc; unflagged trace-centroid rows on a quartz.
- ``pctFlagged``: flagged lamp-line rows on lit fibers, in percent.

obsType and seqName are read from the collection's ``iqQaMetrics``; an IIS frame,
which the task reports sparse, is not told apart (004 holds none with lit science
fibers).

    python computeLitFiberMetrics.py --repo /work/datastore --collections u/wtg/qa-thresholds/004 \
        --output litFiberMetrics-u_wtg_qa-thresholds_004.parquet
"""

import argparse
import time

import numpy as np
import pandas as pd
from lsst.daf.butler import Butler

GOOD, BLACKSPOT = 1, 4  # pfs.datamodel.FiberStatus
ENGINEERING = 5  # pfs.datamodel.TargetType
FIBERS_PER_SPECTROGRAPH = 651  # pfs.utils.fibers


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--collections", action="append", required=True)
    parser.add_argument("--visits", type=int, nargs="*", help="only these visits (default: every one)")
    parser.add_argument("--output", default="litFiberMetrics.parquet")
    args = parser.parse_args()

    butler = Butler(args.repo, collections=args.collections)
    where = f"visit IN ({', '.join(map(str, args.visits))})" if args.visits else ""
    refs = sorted(
        butler.registry.queryDatasets(
            "iqQaMetrics", collections=args.collections, where=where, findFirst=True
        ),
        key=lambda ref: tuple(ref.dataId[k] for k in ("visit", "arm", "spectrograph")),
    )
    print(f"computing lit-fiber metrics for {len(refs)} images", flush=True)
    start = time.time()
    rows = []
    pfsConfigs = {}
    for done, ref in enumerate(refs, 1):
        dataId = ref.dataId
        visit, arm, spectrograph = dataId["visit"], dataId["arm"], dataId["spectrograph"]
        stored = butler.get(ref).iloc[0]
        row = {"visit": visit, "arm": arm, "spectrograph": spectrograph}
        row |= {key: stored.get(key) for key in ("obsType", "seqName", "nLines", "pctFlagged", "qaStatus")}
        try:
            lines = butler.get("lines", dataId).data
            if visit not in pfsConfigs:
                pfsConfigs[visit] = butler.get("pfsConfig", visit=visit, instrument=dataId["instrument"])
        except Exception as error:  # record and carry on
            row["error"] = str(error)[:200]
            rows.append(row)
            continue
        pfsConfig = pfsConfigs[visit]
        fiberId = np.asarray(pfsConfig.fiberId, dtype=np.int64)
        status = np.asarray(pfsConfig.fiberStatus, dtype=np.int64)
        # The detector's fibers, as detectorMap.fiberId gives them: its spectrograph's.
        onDetector = (fiberId - 1) // FIBERS_PER_SPECTROGRAPH + 1 == spectrograph
        onDetector &= np.asarray(pfsConfig.targetType, dtype=np.int64) != ENGINEERING
        litMask = onDetector & (status == GOOD)
        lit = fiberId[litMask]
        nLit = int(litMask.sum())
        nFibers = int((litMask | (onDetector & (status == BLACKSPOT))).sum())

        isTrace = np.asarray(lines["description"]) == "Trace"
        onLit = np.isin(np.asarray(lines["fiberId"], dtype=np.int64), lit)
        flag = np.asarray(lines["flag"]).astype(bool)
        lamp = ~isTrace & onLit
        counted = isTrace if row["obsType"] == "trace" else ~isTrace
        row |= {
            "nLitFibers": nLit,
            "nFibers": nFibers,
            "nTraceRows": int(isTrace.sum()),
            "nLampRows": int(lamp.sum()),
            # As the task: on arcs and quartz only.
            "linesPerFiber": (
                float((counted & onLit & ~flag).sum()) / nLit
                if nLit and row["obsType"] in ("arc", "trace")
                else np.nan
            ),
            "pctFlaggedLamp": 100.0 * float((lamp & flag).sum()) / lamp.sum() if lamp.any() else np.nan,
        }
        rows.append(row)
        if done % 20 == 0 or done == len(refs):
            elapsed = time.time() - start
            print(
                f"  {done}/{len(refs)}  elapsed {elapsed / 60:.1f} min"
                f"  eta {elapsed / done * (len(refs) - done) / 60:.1f} min",
                flush=True,
            )

    result = pd.DataFrame(rows)
    result.to_parquet(args.output)
    errors = result["error"].notna().sum() if "error" in result else 0
    print(f"wrote {args.output} ({len(result)} images, {errors} without lines or pfsConfig)")


if __name__ == "__main__":
    main()
