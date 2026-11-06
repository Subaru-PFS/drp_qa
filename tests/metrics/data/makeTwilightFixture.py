#!/usr/bin/env python
"""Make ``twilightMetrics.parquet``: the twilight metrics of 16 sequences, 2025-03 to 2026-09.

The inputs are per-fiber tables extracted from drpActor's and the validation run's pfsArms (one row per
visit, arm, spectrograph and fiber, with the median flux / norm over the arm's window, its S/N, the
fiber's focal-plane position and pfsConfig flags), and a table mapping each visit to its IIC sequence.
They are not committed (about 35 MB); this records how the fixture was made from them. Only the
metrics, by visit ID, are committed: twilights are engineering frames.

    python makeTwilightFixture.py twilightFibers.parquet otherTwilights.parquet --sequences sequences.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from pfs.drp.qa.metrics.twilight import sequenceConsistency


def toFibers(extracted: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
    """Return the extracted rows as ``twilightFiberFlux`` rows of the lit fibers, as ``litFibers`` picks them."""
    use = (extracted["fiberStatus"] == "GOOD") & (extracted["targetType"] != "ENGINEERING")
    fibers = extracted[use].rename(columns={"medReduced": "flux", "medSnr": "snr"})
    fibers = fibers.assign(satFraction=0.0).merge(sequences, on="visit")
    columns = ["visit", "arm", "spectrograph", "sequence", "seqName", "fiberId", "x", "y", "flux", "snr"]
    return fibers[[*columns, "nGood", "satFraction"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tables", nargs="+", help="per-fiber parquet tables")
    parser.add_argument("--sequences", required=True, help="CSV of visit, sequence")
    parser.add_argument("--output", default=str(Path(__file__).parent / "twilightMetrics.parquet"))
    args = parser.parse_args()
    extracted = pd.concat([pd.read_parquet(path) for path in args.tables], ignore_index=True)
    fibers = toFibers(extracted, pd.read_csv(args.sequences))
    metrics = sequenceConsistency(fibers)
    metrics.to_parquet(args.output, index=False)
    print(f"Wrote {len(metrics)} rows for {metrics['visit'].nunique()} visits to {args.output}")


if __name__ == "__main__":
    main()
