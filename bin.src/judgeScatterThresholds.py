#!/usr/bin/env python3
r"""Judge comparison mode's exported line scatter with a thresholds file, run by run.

The acceptance check for adopting scatter thresholds (PIPE2D-1956): the reserved-line robust
RMS of every lamp arc and quartz of the runs compared must not fail in bulk, or the thresholds
describe the reference visits' night rather than the instrument. Reads the
``dmLineResidualMetrics-<period>.parquet`` and ``dmLineResidualSummary-<period>.parquet`` that
``qaComparison.py report`` writes beside each report (and ``iqQaMetrics-<period>.parquet`` there
for the observation type of tables stored before ``obsType`` was a column), judges the
``RESERVED`` rows against the visit's detectorMap with the file (and the task's config fallback),
and prints WARN and FAIL counts per run, arm, observation type and species. Stack-free.

Example::

    PYTHONPATH=python python bin.src/judgeScatterThresholds.py \\
        python/pfs/drp/qa/metrics/data/dmLineResidualThresholds-run25.yaml \\
        ~/qa-comparison/run2*/qa/<version>/<pipeline>
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pfs.drp.qa.metrics.gate import judge
from pfs.drp.qa.metrics.lineResiduals import RESERVED, VISIT, residualThresholds, wideResiduals

SCATTER = ("robustRmsDeltaX", "robustRmsDeltaY")


def scatterVerdicts(
    metrics: pd.DataFrame, summary: pd.DataFrame, thresholds, kinds: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return the scatter verdicts of one period's images, one row per image, species and metric.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        ``dmLineResidualMetrics`` rows.
    summary : `pandas.DataFrame`
        ``dmLineResidualSummary`` rows of the same images.
    thresholds
        A thresholds file path or table, judged before the task's config
        fallback (``pfs.drp.qa.metrics.lineResiduals.residualThresholds``).
    kinds : `pandas.DataFrame`, optional
        ``iqQaMetrics`` rows of the same images: their ``obsType``, for tables
        stored before it was a column.

    Returns
    -------
    `pandas.DataFrame`
        ``visit``, ``arm``, ``spectrograph``, ``cameraSet``, ``obsType``,
        ``description``, ``metric``, ``value``, ``status`` (``PASS``, ``WARN``,
        ``FAIL`` or empty for not judged).
    """
    wide = wideResiduals(metrics, VISIT, (RESERVED,), summary=summary, kinds=kinds)
    if "obsType" not in wide.columns:
        wide["obsType"] = None
    judged = judge(wide, [thresholds, residualThresholds(1.0, 2.0, 1.0, 2.0)])
    judged = judged[judged["metric"].isin(SCATTER)]
    keys = ["visit", "arm", "spectrograph", "cameraSet", "obsType", "description"]
    rows = wide[keys].iloc[judged["row"].to_numpy()].reset_index(drop=True)
    return rows.assign(
        metric=judged["metric"].to_numpy(),
        value=judged["value"].to_numpy(),
        status=judged["status"].to_numpy(),
    )


def summarize(verdicts: pd.DataFrame) -> pd.DataFrame:
    """Count verdicts per arm, observation type and species: ``judged``, ``WARN``, ``FAIL`` and ``FAIL%``."""
    columns = ["arm", "obsType", "description", "judged", "WARN", "FAIL", "FAIL%"]
    judged = verdicts[verdicts["status"] != ""]
    if judged.empty:
        return pd.DataFrame(columns=columns)
    counts = judged.groupby(["arm", "obsType", "description"], dropna=False)["status"].agg(
        judged="size", WARN=lambda s: int((s == "WARN").sum()), FAIL=lambda s: int((s == "FAIL").sum())
    )
    counts["FAIL%"] = (100 * counts["FAIL"] / counts["judged"]).round(1)
    return counts.reset_index()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("thresholds", help="the thresholds file to judge with")
    parser.add_argument("directories", nargs="+", help="directories holding dmLineResidual*-<period>.parquet")
    args = parser.parse_args(argv)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    totals = []
    for directory in args.directories:
        directory = Path(directory).expanduser()
        for path in sorted(directory.glob("dmLineResidualMetrics-*.parquet")):
            period = path.stem.split("-", 1)[1]
            summary = directory / f"dmLineResidualSummary-{period}.parquet"
            iqQa = directory / f"iqQaMetrics-{period}.parquet"
            kinds = pd.read_parquet(iqQa) if iqQa.exists() else None
            verdicts = scatterVerdicts(
                pd.read_parquet(path), pd.read_parquet(summary), args.thresholds, kinds
            )
            counts = summarize(verdicts)
            print(f"\n== {period} ({directory})")
            print(counts.to_string(index=False))
            judged = int(counts["judged"].sum())
            totals.append(
                {
                    "period": period,
                    "judged": judged,
                    "WARN": int(counts["WARN"].sum()),
                    "FAIL": int(counts["FAIL"].sum()),
                }
            )
    table = pd.DataFrame(totals)
    if not table.empty:
        table["FAIL%"] = (100 * table["FAIL"] / table["judged"]).round(1)
        print("\n== totals")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
