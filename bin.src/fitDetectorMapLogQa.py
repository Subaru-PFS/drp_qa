#!/usr/bin/env python3
"""Parse fitDetectorMap pipeline log files and report QA status.

Reads one or more ``pipetask run`` log files produced by the detectorMap
pipeline and summarises each ``fitDetectorMap`` quantum's fit quality,
line counts, and flags any suspicious results.

No LSST stack is required — only the Python standard library (plus optional
``matplotlib`` for ``--plot``).

Both the old and the current ``fitDetectorMap`` log formats are accepted: as of
``drp_stella`` commit 16174310 the ``Final result`` and ``Stats for <species>``
messages carry an inline ``arm=<a> spectrograph=<n>`` prefix, which is optional
here. Quanta are keyed off the ``(fitDetectorMap:{...})`` label that ``pipetask``
puts in the logger context, so logs must be captured with that context intact.

Example::

    python bin.src/fitDetectorMapLogQa.py run28-dm-02.log run28-dm-03.log
    python bin.src/fitDetectorMapLogQa.py --plot run28-dm-03.log
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QuantumResult:
    """All parsed information for one fitDetectorMap quantum."""

    arm: str
    spectrograph: int
    log_file: str

    # "Final result" — slit-corrected quantities, the primary output
    final_chi2: float = float("nan")
    final_dof: int = 0
    final_xRMS: float = float("nan")
    final_yRMS: float = float("nan")
    final_xSoften: float = float("nan")
    final_ySoften: float = float("nan")
    final_nLines: int = 0
    final_species: dict[str, int] = field(default_factory=dict)

    # Per-species residual stats (from "Stats for <species>" lines)
    species_xRMS: dict[str, float] = field(default_factory=dict)
    species_yRMS: dict[str, float] = field(default_factory=dict)
    species_count: dict[str, int] = field(default_factory=dict)

    # Number of fibers for which slit offset measurement failed
    n_failed_slit_offsets: int = 0

    # Wall time for the full quantum
    exec_time_s: float = float("nan")

    # Whether a "Final result" line was ever found
    has_final_result: bool = False

    # Errors logged at ERROR/CRITICAL level
    errors: list[str] = field(default_factory=list)

    # QA verdict populated by assess()
    status: str = "OK"  # "OK", "WARN", "BAD"
    flags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def arc_line_total(self) -> int:
        """Total arc lines (everything except Trace)."""
        return sum(n for sp, n in self.final_species.items() if sp != "Trace")


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Quantum context embedded in the logger name, e.g.:
#   (fitDetectorMap:{instrument: 'PFS', arm: 'b', spectrograph: 2})
_QUANTUM_CTX = re.compile(r"\(fitDetectorMap:\{instrument: '(\w+)', arm: '(\w)', spectrograph: (\d+)\}\)")

# The "arm=b spectrograph=1 " prefix that newer drp_stella emits on its summary
# lines. Optional, so that logs from before that change still parse.
_ARM_SPEC = r"(?:arm=\S+ spectrograph=\d+ )?"

# "Final result" — the last line printed by fitDistortedDetectorMap per quantum.
# Format (no nm annotation on yRMS):
#   Final result: [arm=b spectrograph=1 ]chi2=X dof=N xRMS=X yRMS=X xSoften=X ySoften=X
#       from N lines (sp: N, ...)
_FINAL_RESULT = re.compile(
    r"Final result: " + _ARM_SPEC + r"chi2=(\S+) dof=(\d+) xRMS=(\S+) yRMS=(\S+) "
    r"xSoften=(\S+) ySoften=(\S+) from (\d+) lines \((.+)\)"
)

# Per-species summary stats (logged after Final result)
#   Stats for HgI: [arm=b spectrograph=1 ]chi2=X dof=N xRMS=X yRMS=X xSoften=X ySoften=X
#       from N lines (...)
# The \w+ species group cannot match the "fiberId=42" / "wavelength=555.0 (HgI)"
# variants of this message, which is what keeps them out of the species tables.
_SPECIES_STATS = re.compile(
    r"Stats for (\w+): " + _ARM_SPEC + r"chi2=\S+ dof=\d+ xRMS=(\S+) yRMS=(\S+) "
    r"xSoften=\S+ ySoften=\S+ from (\d+) lines"
)

# Execution time
_EXEC_TIME = re.compile(r"Execution of task 'fitDetectorMap' on quantum .+ took (\S+) seconds")

# Failed slit offsets
_SLIT_FAIL = re.compile(r"Unable to measure slit offsets for (\d+) fiberIds")

# Species count dict inside parentheses, e.g. "(CdI: 2086, HgI: 4591, ...)"
_SPECIES_COUNTS = re.compile(r"(\w+): (\d+)")

# Log level prefix
_LOG_LEVEL = re.compile(r"^(ERROR|CRITICAL|WARNING)\s")


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


def _parse_species_dict(s: str) -> dict[str, int]:
    return {m.group(1): int(m.group(2)) for m in _SPECIES_COUNTS.finditer(s)}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_log(log_path: Path) -> dict[tuple[str, int], QuantumResult]:
    """Parse *log_path* and return a mapping ``(arm, spectrograph) → QuantumResult``."""
    results: dict[tuple[str, int], QuantumResult] = {}

    with open(log_path) as fh:
        lines = fh.readlines()

    for line in lines:
        # Every relevant fitDetectorMap line contains the quantum context
        ctx_m = _QUANTUM_CTX.search(line)
        if ctx_m is None:
            continue

        arm = ctx_m.group(2)
        spec = int(ctx_m.group(3))
        key = (arm, spec)

        if key not in results:
            results[key] = QuantumResult(arm=arm, spectrograph=spec, log_file=log_path.name)
        qr = results[key]

        # Isolate the message portion (after " - ")
        sep = line.find(" - ")
        msg = line[sep + 3 :].strip() if sep != -1 else line.strip()

        # ---------------------------------------------------------------- #
        # Final result
        m = _FINAL_RESULT.search(msg)
        if m:
            qr.has_final_result = True
            qr.final_chi2 = _safe_float(m.group(1))
            qr.final_dof = int(m.group(2))
            qr.final_xRMS = _safe_float(m.group(3))
            qr.final_yRMS = _safe_float(m.group(4))
            qr.final_xSoften = _safe_float(m.group(5))
            qr.final_ySoften = _safe_float(m.group(6))
            qr.final_nLines = int(m.group(7))
            qr.final_species = _parse_species_dict(m.group(8))
            continue

        # ---------------------------------------------------------------- #
        # Per-species stats (come after Final result)
        m = _SPECIES_STATS.search(msg)
        if m:
            sp = m.group(1)
            qr.species_xRMS[sp] = _safe_float(m.group(2))
            qr.species_yRMS[sp] = _safe_float(m.group(3))
            qr.species_count[sp] = int(m.group(4))
            continue

        # ---------------------------------------------------------------- #
        # Slit offset failures (take the max across iterations)
        m = _SLIT_FAIL.search(msg)
        if m:
            qr.n_failed_slit_offsets = max(qr.n_failed_slit_offsets, int(m.group(1)))
            continue

        # ---------------------------------------------------------------- #
        # Execution time (in single_quantum_executor line, has quantum ctx)
        m = _EXEC_TIME.search(line)
        if m:
            qr.exec_time_s = _safe_float(m.group(1))
            continue

        # ---------------------------------------------------------------- #
        # Errors
        lv = _LOG_LEVEL.match(line)
        if lv and lv.group(1) in ("ERROR", "CRITICAL"):
            qr.errors.append(msg[:200])

    return results


# ---------------------------------------------------------------------------
# QA assessment
# ---------------------------------------------------------------------------

# Default thresholds
YRMS_WARN = 0.10  # pixels
YRMS_BAD = 0.30  # pixels
XRMS_WARN = 0.05  # pixels
XRMS_BAD = 0.10  # pixels
LOW_ARC_FRAC = 0.25  # flag if arc total < this fraction of peer max
LOW_ARC_PEER_MIN = 100  # only compare against peers that have ≥ this many arc lines total


def _flag(qr: QuantumResult, level: str, msg: str) -> None:
    if level == "BAD":
        qr.status = "BAD"
    elif level == "WARN" and qr.status == "OK":
        qr.status = "WARN"
    qr.flags.append(f"[{level}] {msg}")


def assess(
    results: dict[tuple[str, int], QuantumResult],
    yrms_warn: float = YRMS_WARN,
    yrms_bad: float = YRMS_BAD,
) -> None:
    """Apply QA rules and populate ``.status`` / ``.flags`` for each quantum."""
    # --- Cross-quantum rules (within the same log + arm) ---
    # Group by arm
    arm_groups: dict[str, list[QuantumResult]] = {}
    for (arm, _spec), qr in results.items():
        arm_groups.setdefault(arm, []).append(qr)

    for _arm, group in arm_groups.items():
        # Only compare quanta that have a Final result
        with_result = [qr for qr in group if qr.has_final_result]
        if not with_result:
            continue

        # Arc line totals (non-Trace) for cross-SM comparison
        arc_totals = {qr.spectrograph: qr.arc_line_total() for qr in with_result}
        max_arc = max(arc_totals.values(), default=0)

        # All species seen across any quantum (ignore Trace)
        all_species: set[str] = set()
        for qr in with_result:
            all_species.update(sp for sp in qr.final_species if sp != "Trace")

        # Max count per species across quanta (to identify "expected" species)
        max_by_species: dict[str, int] = {}
        for sp in all_species:
            max_by_species[sp] = max(qr.final_species.get(sp, 0) for qr in with_result)

        for qr in with_result:
            arc_n = arc_totals[qr.spectrograph]

            # Low total arc lines vs peers
            if max_arc >= LOW_ARC_PEER_MIN and arc_n < LOW_ARC_FRAC * max_arc:
                _flag(
                    qr,
                    "WARN",
                    f"Total arc lines {arc_n:,} is only "
                    f"{100.0 * arc_n / max_arc:.0f}% of peer max {max_arc:,}",
                )

            # Missing species that other SMs have
            for sp in all_species:
                peer_max = max_by_species[sp]
                this_n = qr.final_species.get(sp, 0)
                if peer_max >= 100 and this_n == 0:
                    _flag(
                        qr,
                        "WARN",
                        f"Species {sp} has 0 lines (peers have up to {peer_max:,})",
                    )

    # --- Per-quantum rules ---
    for qr in results.values():
        if not qr.has_final_result:
            _flag(qr, "BAD", "No 'Final result' logged — quantum may have failed")
            continue

        if qr.errors:
            _flag(qr, "BAD", f"Logged {len(qr.errors)} ERROR/CRITICAL message(s)")

        # yRMS thresholds
        yr = qr.final_yRMS
        if not math.isnan(yr):
            if yr > yrms_bad:
                _flag(qr, "BAD", f"yRMS={yr:.4f} px > {yrms_bad} px threshold")
            elif yr > yrms_warn:
                _flag(qr, "WARN", f"yRMS={yr:.4f} px > {yrms_warn} px threshold")

        # xRMS thresholds
        xr = qr.final_xRMS
        if not math.isnan(xr):
            if xr > XRMS_BAD:
                _flag(qr, "BAD", f"xRMS={xr:.4f} px > {XRMS_BAD} px threshold")
            elif xr > XRMS_WARN:
                _flag(qr, "WARN", f"xRMS={xr:.4f} px > {XRMS_WARN} px threshold")

        # NaN ySoften indicates the softening calculation blew up (dof ≤ 0)
        if math.isnan(qr.final_ySoften):
            _flag(qr, "WARN", "ySoften=nan in Final result (possible dof≤0 in calculateSoftening)")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_STATUS_ICONS = {"OK": "✓", "WARN": "⚠", "BAD": "✗"}

# Ordered list of arc species to show in the table (Trace omitted — too large)
_ARC_SPECIES = ["ArI", "CdI", "HgI", "KrI", "NeI", "XeI"]


def _fmt(value: float, decimals: int = 4, width: int = 8) -> str:
    if math.isnan(value):
        return "nan".rjust(width)
    return f"{value:.{decimals}f}".rjust(width)


def _col(s: str, w: int) -> str:
    return str(s).ljust(w)


def _print_summary_table(
    results: dict[tuple[str, int], QuantumResult],
    log_name: str,
) -> None:
    """Print a per-quantum summary table."""
    # Determine which species actually appeared (to avoid empty columns)
    present_species = [
        sp for sp in _ARC_SPECIES if any(qr.final_species.get(sp, 0) > 0 for qr in results.values())
    ]

    header_parts = [
        _col("arm", 3),
        _col("SM", 3),
        _col("arc_total", 9),
        *(f"{sp:>6}" for sp in present_species),
        _col("xRMS(px)", 9),
        _col("yRMS(px)", 9),
        _col("ySoften", 8),
        _col("exec(s)", 7),
        _col("status", 6),
    ]
    header_parts.append("notes")

    sep = "  "
    header = sep.join(header_parts)
    divider = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f"Log file: {log_name}")
    print(divider)
    print(header)
    print(divider)

    for arm, spec in sorted(results):
        qr = results[(arm, spec)]
        arc_total = qr.arc_line_total()
        row = [
            _col(arm, 3),
            _col(spec, 3),
            _col(f"{arc_total:,}", 9),
            *(f"{qr.final_species.get(sp, 0):>6}" for sp in present_species),
            _fmt(qr.final_xRMS, 4, 9),
            _fmt(qr.final_yRMS, 4, 9),
            _fmt(qr.final_ySoften, 4, 8),
            _fmt(qr.exec_time_s, 1, 7),
            _col(f"{_STATUS_ICONS[qr.status]} {qr.status}", 6),
        ]
        first_flag = qr.flags[0] if qr.flags else ""
        row.append(first_flag)
        print(sep.join(row))

    print(divider)


def _print_flag_report(results: dict[tuple[str, int], QuantumResult]) -> None:
    """Print the full flag list for every flagged quantum."""
    flagged = {k: qr for k, qr in results.items() if qr.flags}
    if not flagged:
        print("  No flags raised.\n")
        return
    for (arm, spec), qr in sorted(flagged.items()):
        print(f"  {arm}{spec} ({qr.log_file}):")
        for f in qr.flags:
            print(f"    {f}")
    print()


# ---------------------------------------------------------------------------
# Optional bar chart
# ---------------------------------------------------------------------------


def _make_plot(all_results: list[dict[tuple[str, int], QuantumResult]], log_names: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("matplotlib not available — skipping plot.", file=sys.stderr)
        return

    n_logs = len(all_results)
    fig, axes = plt.subplots(1, n_logs, figsize=(6 * n_logs, 5), squeeze=False)

    for col, (results, log_name) in enumerate(zip(all_results, log_names, strict=True)):
        ax = axes[0][col]
        quanta = sorted(results)
        x_labels = [f"{arm}{spec}" for arm, spec in quanta]

        present_species = [
            sp for sp in _ARC_SPECIES if any(results[k].final_species.get(sp, 0) > 0 for k in quanta)
        ]

        import numpy as np

        x = np.arange(len(quanta))
        width = 0.8 / max(len(present_species), 1)
        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

        for i, sp in enumerate(present_species):
            counts = [results[k].final_species.get(sp, 0) for k in quanta]
            ax.bar(x + i * width, counts, width, label=sp, color=colors[i % len(colors)])

        ax.set_xticks(x + width * (len(present_species) - 1) / 2)
        ax.set_xticklabels(x_labels)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_title(log_name, fontsize=9)
        ax.set_ylabel("Arc lines used")
        ax.legend(fontsize=8)
        ax.set_xlabel("Quantum (arm+SM)")

    fig.suptitle("fitDetectorMap — arc line counts per species", fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("logfiles", nargs="+", metavar="LOGFILE", help="One or more pipetask log files")
    p.add_argument(
        "--warn-yrms",
        type=float,
        default=YRMS_WARN,
        metavar="PX",
        help=f"yRMS WARN threshold in pixels (default: {YRMS_WARN})",
    )
    p.add_argument(
        "--bad-yrms",
        type=float,
        default=YRMS_BAD,
        metavar="PX",
        help=f"yRMS BAD threshold in pixels (default: {YRMS_BAD})",
    )
    p.add_argument("--plot", action="store_true", help="Show bar chart of arc line counts")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    all_results: list[dict[tuple[str, int], QuantumResult]] = []
    log_names: list[str] = []
    any_bad = False

    for logfile_str in args.logfiles:
        log_path = Path(logfile_str)
        if not log_path.exists():
            print(f"ERROR: file not found: {log_path}", file=sys.stderr)
            return 1

        results = parse_log(log_path)
        if not results:
            print(f"WARNING: no fitDetectorMap quanta found in {log_path}", file=sys.stderr)
            continue

        assess(results, yrms_warn=args.warn_yrms, yrms_bad=args.bad_yrms)

        _print_summary_table(results, log_path.name)

        print("Flagged quanta:")
        _print_flag_report(results)

        all_results.append(results)
        log_names.append(log_path.name)

        if any(qr.status == "BAD" for qr in results.values()):
            any_bad = True

    if args.plot and all_results:
        _make_plot(all_results, log_names)

    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
