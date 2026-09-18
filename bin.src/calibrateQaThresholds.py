#!/usr/bin/env python3
r"""Derive QA thresholds from the golden visit set.

Implements steps 1-4 of the threshold derivation procedure in
``doc/qa-rebuild-plan.md`` section 1.2 and prints the config values to paste
into the task, complete with the provenance sentence rule R2 requires.

The rule this script exists to enforce: **no threshold enters the codebase
before it has been computed from a known-good visit range.** A threshold
invented at the keyboard cannot be defended when it fires at 03:00.

Example::

    python bin.src/calibrateQaThresholds.py \\
        -b /path/to/butler \\
        -c u/wtg/qa/run12 \\
        --metric medFwhm --metric dxCenterRms

    # Per-arm thresholds, which is what the b arm needs (see AGENTS.md):
    python bin.src/calibrateQaThresholds.py \\
        -b /path/to/butler -c u/wtg/qa/run12 \\
        --metric pctFlagged --group-by arm

    # Against a CSV exported earlier, with no Butler in the loop:
    python bin.src/calibrateQaThresholds.py \\
        --csv metrics.csv --metric medFwhm

Requires the LSST stack (``lsst.daf.butler``) unless ``--csv`` is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running straight from a source checkout, before `setup -r .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from pfs.drp.qa.metrics.goldenVisits import (
    GoldenVisitSet,
    defaultGoldenVisitsPath,
    loadGoldenVisits,
)
from pfs.drp.qa.metrics.longFormat import LONG_COLUMNS, widen
from pfs.drp.qa.metrics.thresholds import (
    deriveThresholds,
    verifyKnownBad,
)

#: Metrics that are gated today, used when --metric is not given.
DEFAULT_METRICS = ("medFwhm", "pctFlagged", "medDxCenter")

#: Metrics whose bad direction is downwards.
LOWER_IS_WORSE = frozenset({"nLines", "fitNLines", "minFiberPitch"})


def loadMetrics(args: argparse.Namespace) -> pd.DataFrame:
    """Load the metrics table, from a CSV or from a Butler collection.

    Parameters
    ----------
    args : `argparse.Namespace`
        Parsed command-line arguments.

    Returns
    -------
    `pandas.DataFrame`
        Concatenated metrics rows.

    Notes
    -----
    The Butler path issues one registry query rather than looping over the
    arm/spectrograph grid with speculative ``butler.get`` calls: a mistyped
    collection name and a genuinely absent detector must not look the same.
    """
    if args.csv:
        return _asWide(pd.read_csv(args.csv))

    try:
        from lsst.daf.butler import Butler
    except ImportError:
        print(
            "Error: lsst.daf.butler not available. Run this inside the LSST stack, or use --csv.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    butler = Butler(args.butler, collections=[args.collection])
    # findFirst: one dataset per data ID, taken from the first run in the chain
    # that has it. Without it a chained collection returns every run's copy, so
    # a detector reduced twice is counted twice -- inflating n and repeating
    # values, which is precisely what drives p95 and p99 together.
    refs = set(
        butler.registry.queryDatasets(
            args.dataset_type,
            collections=[args.collection],
            where=args.where,
            findFirst=True,
        )
    )
    if not refs:
        # An empty result is reported as an error, not as "no data": a bad
        # collection name and an empty collection are different problems.
        print(
            f"No {args.dataset_type} datasets in collection '{args.collection}'"
            + (f" with where='{args.where}'" if args.where else "")
            + ".",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Found {len(refs)} {args.dataset_type} datasets. Loading...", file=sys.stderr)
    return _asWide(pd.concat([butler.get(ref) for ref in refs], ignore_index=True))


def _asWide(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pivot a long-format metrics table, leaving a wide one untouched.

    ``iqQaSpeciesMetrics`` stores metric names in a ``metric`` column and the
    numbers in ``value``. Calibrating against it without this would look for a
    column named after the metric, fail to find one, and skip -- reporting
    success while producing no threshold at all.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        Either schema.

    Returns
    -------
    `pandas.DataFrame`
        One column per metric, with ``description`` preserved so that
        ``--group-by description`` works.
    """
    if not set(LONG_COLUMNS).issubset(metrics.columns):
        return metrics
    print("Long-format input detected; pivoting to one column per metric.", file=sys.stderr)
    return widen(metrics)


def selectVisits(
    metrics: pd.DataFrame,
    golden: GoldenVisitSet,
    which: str,
    metric: str | None = None,
    confirmedOnly: bool = True,
) -> pd.DataFrame:
    """Select the rows of ``metrics`` covered by one half of the golden set.

    Parameters
    ----------
    metrics : `pandas.DataFrame`
        Metrics rows, with at least a ``visit`` column and optionally ``arm``
        and ``spectrograph``.
    golden : `GoldenVisitSet`
        The golden visit set.
    which : `str`
        ``"good"`` or ``"bad"``.
    metric : `str`, optional
        When given, restrict ``known_bad`` entries to those that either name
        this metric or name none at all. A known_bad entry that names
        ``medFwhm`` asserts nothing about ``pctFlagged``, and holding it against
        every metric turns one real fault into a wall of spurious failures.
    confirmedOnly : `bool`, optional
        For ``known_bad``, drop entries marked ``unconfirmed``. Their verdict is
        a suspicion nobody has checked yet, so it must not decide a pass/fail.

    Returns
    -------
    `pandas.DataFrame`
        The matching rows. Empty when the collection holds none of the golden
        visits.
    """
    if which == "good":
        entries = golden.knownGood
    else:
        entries = golden.confirmedBad if confirmedOnly else golden.knownBad
    if metric is not None:
        entries = tuple(entry for entry in entries if entry.metric in (None, metric))
    if "visit" not in metrics.columns:
        raise SystemExit("Metrics table has no 'visit' column; cannot match against the golden set.")

    def rowMatches(row: pd.Series) -> bool:
        arm = row.get("arm")
        spectrograph = row.get("spectrograph")
        seqName = row.get("seqName")
        return any(
            entry.matches(
                int(row["visit"]),
                arm=str(arm) if isinstance(arm, str) else None,
                spectrograph=int(spectrograph) if pd.notna(spectrograph) else None,
                seqType=str(seqName) if isinstance(seqName, str) else None,
            )
            for entry in entries
        )

    if not entries:
        return metrics.iloc[0:0]
    return metrics[metrics.apply(rowMatches, axis=1)]


def describeVisits(frame: pd.DataFrame) -> str:
    """Return a compact ``first-last`` description of a frame's visits.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        Rows with a ``visit`` column.

    Returns
    -------
    `str`
        e.g. ``"140200-140260"``, or ``"none"`` for an empty frame.
    """
    if frame.empty:
        return "none"
    visits = frame["visit"].dropna().astype(int)
    return f"{visits.min()}-{visits.max()}" if len(visits) else "none"


def main() -> int:
    """Run the threshold calibration.

    Returns
    -------
    `int`
        Process exit status: 0 when every metric produced a suggestion that the
        known-bad data crosses, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Derive QA thresholds from the golden visit set.",
    )
    parser.add_argument("-b", "--butler", help="Path to the butler repository.")
    parser.add_argument("-c", "--collection", help="Butler collection holding the QA metrics.")
    parser.add_argument(
        "--dataset-type",
        default="iqQaMetrics",
        help="Metrics dataset type to read (default: iqQaMetrics).",
    )
    parser.add_argument("--csv", help="Read metrics from a CSV instead of the Butler.")
    parser.add_argument("-w", "--where", default="", help="Butler query expression to narrow the query.")
    parser.add_argument(
        "--filter",
        default=None,
        help=(
            "pandas query applied to the metrics before anything else, e.g. "
            '"traceOnly == False" to derive arc thresholds without the quartz rows. '
            "Unlike --where it can use any column, not just dimensions."
        ),
    )
    parser.add_argument(
        "--golden",
        default=None,
        help=f"Golden visit set YAML (default: {defaultGoldenVisitsPath()}).",
    )
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        help=f"Metric column to calibrate; repeatable (default: {', '.join(DEFAULT_METRICS)}).",
    )
    parser.add_argument(
        "--group-by",
        action="append",
        dest="groupBy",
        default=None,
        help="Column to derive separate thresholds for; repeatable (e.g. arm, description).",
    )
    parser.add_argument("--warn-percentile", type=float, default=95.0, help="WARN percentile (default: 95).")
    parser.add_argument("--fail-percentile", type=float, default=99.0, help="FAIL percentile (default: 99).")
    args = parser.parse_args()

    if not args.csv and (not args.butler or not args.collection):
        parser.error("Either --csv, or both --butler and --collection, must be given.")

    golden = loadGoldenVisits(args.golden)
    if not golden.knownGood:
        print(
            f"The golden visit set at {golden.path} has no usable known_good entries.\n"
            "Every entry is still a placeholder. Fill in real visit numbers before deriving\n"
            "thresholds -- see doc/qa-rebuild-plan.md section 1.1.",
            file=sys.stderr,
        )
        return 1

    metrics = loadMetrics(args)
    if args.filter:
        # Applied before the golden-set split so good and bad rows are filtered
        # alike. Without it, --group-by arm mixes arcs and quartz, and the quartz
        # rows -- read from a calibration, so identical visit to visit -- land in
        # the arc tail and become the arc threshold.
        before = len(metrics)
        try:
            metrics = metrics.query(args.filter)
        except Exception as exc:
            print(f"--filter {args.filter!r} is not a valid query: {exc}", file=sys.stderr)
            return 1
        print(f"--filter {args.filter!r}: kept {len(metrics)} of {before} rows", file=sys.stderr)
        if metrics.empty:
            print("The filter removed every row.", file=sys.stderr)
            return 1
    good = selectVisits(metrics, golden, "good")
    bad = selectVisits(metrics, golden, "bad")

    if good.empty:
        print(
            f"None of the {len(golden.goodVisits)} known_good visits are present in this collection.\n"
            "Run the QA pipeline over the golden visits first.",
            file=sys.stderr,
        )
        return 1

    visitRange = describeVisits(good)
    print(f"known_good rows: {len(good)} over visits {visitRange}")
    print(f"known_bad  rows: {len(bad)} over visits {describeVisits(bad)}")
    if bad.empty:
        print("WARNING: no known_bad rows found; step 4 of the procedure cannot be checked.")

    # Suspected faults nobody has checked yet. Shown, never gated on -- the
    # point of recording them is that somebody compares their numbers.
    unconfirmed = selectVisits(metrics, golden, "bad", confirmedOnly=False)
    unconfirmed = unconfirmed.drop(index=bad.index, errors="ignore")
    if not unconfirmed.empty:
        print(
            f"unconfirmed rows: {len(unconfirmed)} over visits {describeVisits(unconfirmed)} "
            "(reported below, not gated on)"
        )
    print()

    groupBy = [column for column in (args.groupBy or []) if column in good.columns]
    if args.groupBy and len(groupBy) != len(args.groupBy):
        missing = sorted(set(args.groupBy) - set(groupBy))
        print(f"WARNING: ignoring --group-by columns absent from the data: {', '.join(missing)}\n")

    ok = True
    requested = bool(args.metrics)
    unverified: list[str] = []
    for metric in args.metrics or DEFAULT_METRICS:
        if metric not in good.columns:
            # Asked for explicitly but absent: an error, not a skip. Skipping
            # would let the command exit 0 having produced no threshold.
            print(f"{metric}: not a column in {args.dataset_type}.", file=sys.stderr)
            if requested:
                ok = False
            continue

        higherIsWorse = metric not in LOWER_IS_WORSE
        # Only the known_bad entries that name this metric (or name none) say
        # anything about it; see selectVisits.
        badForMetric = selectVisits(metrics, golden, "bad", metric=metric)
        groups = good.groupby(groupBy, observed=True) if groupBy else [((), good)]
        for key, subset in groups:
            label = metric if not groupBy else f"{metric}[{'/'.join(str(k) for k in _asTuple(key))}]"
            try:
                suggestion = deriveThresholds(
                    # |medDxCenter| is gated on its absolute value, so calibrate
                    # on the same quantity the gate sees.
                    subset[metric].abs() if metric == "medDxCenter" else subset[metric],
                    metric=label,
                    higherIsWorse=higherIsWorse,
                    warnPercentile=args.warn_percentile,
                    failPercentile=args.fail_percentile,
                    visitRange=visitRange,
                )
            except ValueError as exc:
                print(f"{label}: {exc}")
                ok = False
                continue

            print(suggestion)
            print(f"    doc: {suggestion.provenance}")
            if not suggestion.reliable or suggestion.degenerate:
                # Degenerate is as disqualifying as underpowered: a pair whose
                # WARN can never fire is not a threshold, it is one threshold
                # wearing two names.
                ok = False

            badSubset = _matchGroup(badForMetric, groupBy, key)
            if badSubset.empty or metric not in badSubset.columns:
                # Step 4 of the procedure did not run for this metric. Most
                # metrics have no known_bad entry naming them, so this is not a
                # failure -- but it must be visible, or the run looks verified
                # when nothing was verified.
                print("    step 4 NOT CHECKED: no known-bad rows name this metric")
                unverified.append(label)
            else:
                values = badSubset[metric].abs() if metric == "medDxCenter" else badSubset[metric]
                try:
                    crossed, message = verifyKnownBad(values, suggestion)
                except ValueError as exc:
                    # Known-bad rows exist but carry no usable value. Unlike an
                    # absent entry this is a broken check, so it fails the run.
                    print(f"    {exc}")
                    ok = False
                else:
                    print(f"    {message}")
                    ok = ok and crossed

            # Suspected faults: show where they land relative to the suggestion
            # so somebody can settle them, but never let them decide the exit
            # code. That is the difference between a record and a verdict.
            suspect = _matchGroup(unconfirmed, groupBy, key)
            if not suspect.empty and metric in suspect.columns:
                values = suspect[metric].abs() if metric == "medDxCenter" else suspect[metric]
                for visit, value in zip(suspect["visit"], values, strict=False):
                    verdict = "over" if value >= suggestion.fail else "under"
                    print(f"    unconfirmed: visit {visit} {metric}={value:.4g} ({verdict} FAIL)")
            print()

    if unverified:
        print(
            f"Step 4 was not checked for: {', '.join(unverified)}.\n"
            "Those thresholds rest on the known-good distribution alone; nothing has shown\n"
            "that a bad detector crosses them. Add a known_bad entry naming the metric.",
            file=sys.stderr,
        )
    if not ok:
        print(
            "One or more thresholds are unreliable, could not be verified, or fail to\n"
            "separate the known-bad data. Do not commit them as-is.",
            file=sys.stderr,
        )
    return 0 if ok else 1


def _asTuple(key: object) -> tuple:
    """Return a groupby key as a tuple, whether it is scalar or already one.

    Parameters
    ----------
    key : `object`
        The key yielded by ``DataFrame.groupby``.

    Returns
    -------
    `tuple`
        The key's components.
    """
    return key if isinstance(key, tuple) else (key,)


def _matchGroup(frame: pd.DataFrame, groupBy: list[str], key: object) -> pd.DataFrame:
    """Select the rows of ``frame`` matching one groupby key.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        The frame to filter.
    groupBy : `list` [`str`]
        Grouping columns; empty means no grouping, so ``frame`` is returned.
    key : `object`
        The key to match.

    Returns
    -------
    `pandas.DataFrame`
        The matching rows.
    """
    if not groupBy or frame.empty:
        return frame
    mask = pd.Series(True, index=frame.index)
    for column, value in zip(groupBy, _asTuple(key), strict=False):
        if column not in frame.columns:
            return frame.iloc[0:0]
        mask &= frame[column] == value
    return frame[mask]


if __name__ == "__main__":
    sys.exit(main())
