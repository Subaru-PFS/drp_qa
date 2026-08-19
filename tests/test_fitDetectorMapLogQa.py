"""Tests for fitDetectorMapLogQa.py — no LSST stack required."""

import math
import sys
import tempfile
import textwrap
import unittest
from io import StringIO
from pathlib import Path

# The script lives in bin.src/ with no package wrapper, so add it to the path.
sys.path.insert(0, str(Path(__file__).parent.parent / "bin.src"))

import fitDetectorMapLogQa as qa

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOGGER_PREFIX = "2025-01-01T00:00:00.000 fitDetectorMap pfs.fitDetectorMap.fitDetectorMap "
_QE_PREFIX = "2025-01-01T00:00:00.000 fitDetectorMap lsst.pipe.base.single_quantum_executor "

_CTX_B1 = "(fitDetectorMap:{instrument: 'PFS', arm: 'b', spectrograph: 1})"
_CTX_B2 = "(fitDetectorMap:{instrument: 'PFS', arm: 'b', spectrograph: 2})"


def _line(msg: str, ctx: str = _CTX_B1, prefix: str = _LOGGER_PREFIX) -> str:
    return f"{prefix}{ctx} - {msg}\n"


def _exec_time_line(seconds: float, ctx: str = _CTX_B1) -> str:
    return (
        f"{_QE_PREFIX}{ctx} - "
        f"Execution of task 'fitDetectorMap' on quantum "
        f"{{instrument: 'PFS', arm: 'b', spectrograph: 1}} took {seconds} seconds\n"
    )


def _write_log(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content))


# ---------------------------------------------------------------------------
# Unit tests — parsing
# ---------------------------------------------------------------------------


class TestParsing(unittest.TestCase):
    def test_final_result_normal(self):
        """Parse a well-formed Final result line."""
        lines = [
            _line(
                "Final result: chi2=1.234 dof=100 xRMS=0.030 yRMS=0.040 "
                "xSoften=0.010 ySoften=0.020 from 1000 lines "
                "(CdI: 200, HgI: 400, NeI: 400)"
            )
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.writelines(lines)
            tmp = Path(f.name)
        try:
            results = qa.parse_log(tmp)
            self.assertIn(("b", 1), results)
            qr = results[("b", 1)]
            self.assertTrue(qr.has_final_result)
            self.assertAlmostEqual(qr.final_chi2, 1.234)
            self.assertEqual(qr.final_dof, 100)
            self.assertAlmostEqual(qr.final_xRMS, 0.030)
            self.assertAlmostEqual(qr.final_yRMS, 0.040)
            self.assertEqual(qr.final_nLines, 1000)
            self.assertEqual(qr.final_species, {"CdI": 200, "HgI": 400, "NeI": 400})
        finally:
            tmp.unlink()

    def test_final_result_nan_ySoften(self):
        """ySoften='nan' in log is parsed as float NaN."""
        lines = [
            _line(
                "Final result: chi2=1.000 dof=5 xRMS=0.029 yRMS=0.450 "
                "xSoften=0.017 ySoften=nan from 120 lines (HgI: 120)"
            )
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.writelines(lines)
            tmp = Path(f.name)
        try:
            results = qa.parse_log(tmp)
            qr = results[("b", 1)]
            self.assertTrue(math.isnan(qr.final_ySoften))
            self.assertAlmostEqual(qr.final_yRMS, 0.450)
        finally:
            tmp.unlink()

    def test_species_stats(self):
        """Per-species stats lines are captured (fiberId lines are ignored)."""
        lines = [
            _line(
                "Final result: chi2=1.0 dof=50 xRMS=0.03 yRMS=0.04 "
                "xSoften=0.01 ySoften=0.02 from 500 lines (HgI: 300, NeI: 200)"
            ),
            _line(
                "Stats for HgI: chi2=0.9 dof=30 xRMS=0.028 yRMS=0.038 "
                "xSoften=0.009 ySoften=0.018 from 300 lines (HgI: 300)"
            ),
            # fiberId stats must NOT be captured as a species
            _line(
                "Stats for fiberId=42: chi2=0.8 dof=10 xRMS=0.025 yRMS=0.035 "
                "xSoften=0.008 ySoften=0.016 from 10 lines (HgI: 10)"
            ),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.writelines(lines)
            tmp = Path(f.name)
        try:
            results = qa.parse_log(tmp)
            qr = results[("b", 1)]
            self.assertIn("HgI", qr.species_xRMS)
            self.assertAlmostEqual(qr.species_xRMS["HgI"], 0.028)
            # "fiberId" should not appear as a species key
            self.assertNotIn("fiberId", qr.species_xRMS)
        finally:
            tmp.unlink()

    def test_final_result_with_arm_spectrograph_prefix(self):
        """Parse the current drp_stella format, which inlines arm/spectrograph."""
        lines = [
            _line(
                "Final result: arm=b spectrograph=1 chi2=1.234 dof=100 "
                "xRMS=0.030 yRMS=0.040 xSoften=0.010 ySoften=0.020 from 1000 lines "
                "(CdI: 200, HgI: 400, NeI: 400)"
            )
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.writelines(lines)
            tmp = Path(f.name)
        try:
            results = qa.parse_log(tmp)
            qr = results[("b", 1)]
            self.assertTrue(qr.has_final_result)
            self.assertAlmostEqual(qr.final_chi2, 1.234)
            self.assertEqual(qr.final_dof, 100)
            self.assertAlmostEqual(qr.final_xRMS, 0.030)
            self.assertAlmostEqual(qr.final_yRMS, 0.040)
            self.assertEqual(qr.final_nLines, 1000)
            self.assertEqual(qr.final_species, {"CdI": 200, "HgI": 400, "NeI": 400})
        finally:
            tmp.unlink()

    def test_species_stats_with_arm_spectrograph_prefix(self):
        """Per-species stats parse in the current format; fiberId lines stay excluded."""
        lines = [
            _line(
                "Stats for HgI: arm=b spectrograph=1 chi2=0.9 dof=30 "
                "xRMS=0.028 yRMS=0.038 xSoften=0.009 ySoften=0.018 from 300 lines (HgI: 300)"
            ),
            _line(
                "Stats for fiberId=42: arm=b spectrograph=1 chi2=0.8 dof=10 "
                "xRMS=0.025 yRMS=0.035 xSoften=0.008 ySoften=0.016 from 10 lines (HgI: 10)"
            ),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.writelines(lines)
            tmp = Path(f.name)
        try:
            results = qa.parse_log(tmp)
            qr = results[("b", 1)]
            self.assertAlmostEqual(qr.species_xRMS["HgI"], 0.028)
            self.assertAlmostEqual(qr.species_yRMS["HgI"], 0.038)
            self.assertEqual(qr.species_count["HgI"], 300)
            self.assertNotIn("fiberId", qr.species_xRMS)
        finally:
            tmp.unlink()

    def test_exec_time_parsed(self):
        """Execution time is captured from the single_quantum_executor line."""
        lines = [_exec_time_line(423.7)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.writelines(lines)
            tmp = Path(f.name)
        try:
            results = qa.parse_log(tmp)
            qr = results[("b", 1)]
            self.assertAlmostEqual(qr.exec_time_s, 423.7, places=1)
        finally:
            tmp.unlink()

    def test_slit_offsets_failure_tracked(self):
        """Slit offset failure count is the max across iterations."""
        lines = [
            _line("Unable to measure slit offsets for 12 fiberIds: [1, 2]"),
            _line("Unable to measure slit offsets for 5 fiberIds: [3]"),
            _line("Unable to measure slit offsets for 12 fiberIds: [1, 2]"),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.writelines(lines)
            tmp = Path(f.name)
        try:
            results = qa.parse_log(tmp)
            qr = results[("b", 1)]
            self.assertEqual(qr.n_failed_slit_offsets, 12)
        finally:
            tmp.unlink()

    def test_multiple_quanta(self):
        """Multiple quanta in the same log are parsed into separate QuantumResults."""
        lines = [
            _line(
                "Final result: chi2=1.0 dof=50 xRMS=0.03 yRMS=0.04 "
                "xSoften=0.01 ySoften=0.02 from 500 lines (HgI: 500)",
                ctx=_CTX_B1,
            ),
            _line(
                "Final result: chi2=2.0 dof=80 xRMS=0.04 yRMS=0.05 "
                "xSoften=0.02 ySoften=0.03 from 800 lines (HgI: 800)",
                ctx=_CTX_B2,
            ),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.writelines(lines)
            tmp = Path(f.name)
        try:
            results = qa.parse_log(tmp)
            self.assertIn(("b", 1), results)
            self.assertIn(("b", 2), results)
            self.assertEqual(results[("b", 1)].final_nLines, 500)
            self.assertEqual(results[("b", 2)].final_nLines, 800)
        finally:
            tmp.unlink()

    def test_no_final_result(self):
        """A quantum with no Final result line has has_final_result=False."""
        lines = [
            _line(
                "Final fit: chi2=1.0 dof=50 xRMS=0.03 yRMS=0.04 "
                "xSoften=0.01 ySoften=0.02 from 10/12 lines (HgI: 10)"
            )
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.writelines(lines)
            tmp = Path(f.name)
        try:
            results = qa.parse_log(tmp)
            qr = results[("b", 1)]
            self.assertFalse(qr.has_final_result)
        finally:
            tmp.unlink()


# ---------------------------------------------------------------------------
# Unit tests — assess()
# ---------------------------------------------------------------------------


class TestAssess(unittest.TestCase):
    def _make_qr(self, spec: int, arc_total: int, yrms: float, ysoften: float = 0.03) -> qa.QuantumResult:
        qr = qa.QuantumResult(arm="b", spectrograph=spec, log_file="test.log")
        qr.has_final_result = True
        qr.final_xRMS = 0.030
        qr.final_yRMS = yrms
        qr.final_xSoften = 0.01
        qr.final_ySoften = ysoften
        # Spread lines across two species (HgI + NeI proportionally)
        qr.final_species = {"HgI": arc_total // 2, "NeI": arc_total - arc_total // 2}
        return qr

    def test_ok_quantum(self):
        results = {("b", 1): self._make_qr(1, 18000, 0.04)}
        qa.assess(results)
        self.assertEqual(results[("b", 1)].status, "OK")
        self.assertEqual(results[("b", 1)].flags, [])

    def test_yrms_warn(self):
        results = {("b", 1): self._make_qr(1, 18000, 0.15)}
        qa.assess(results)
        self.assertEqual(results[("b", 1)].status, "WARN")
        self.assertTrue(any("yRMS" in f for f in results[("b", 1)].flags))

    def test_yrms_bad(self):
        results = {("b", 1): self._make_qr(1, 18000, 0.45)}
        qa.assess(results)
        self.assertEqual(results[("b", 1)].status, "BAD")

    def test_xrms_warn(self):
        results = {("b", 1): self._make_qr(1, 18000, 0.04)}
        results[("b", 1)].final_xRMS = 0.07
        qa.assess(results)
        self.assertEqual(results[("b", 1)].status, "WARN")
        self.assertTrue(any("xRMS" in f for f in results[("b", 1)].flags))

    def test_nan_ysoften_flagged(self):
        results = {("b", 1): self._make_qr(1, 18000, 0.04, ysoften=float("nan"))}
        qa.assess(results)
        self.assertIn("WARN", results[("b", 1)].status)
        self.assertTrue(any("ySoften=nan" in f for f in results[("b", 1)].flags))

    def test_no_final_result_flagged_bad(self):
        qr = qa.QuantumResult(arm="b", spectrograph=1, log_file="test.log")
        qr.has_final_result = False
        results = {("b", 1): qr}
        qa.assess(results)
        self.assertEqual(results[("b", 1)].status, "BAD")

    def test_low_arc_lines_flagged(self):
        """A quantum with <<25% of peer arc lines is flagged WARN."""
        # SM1 and SM2 have ~18k lines; SM3 has ~1k
        results = {
            ("b", 1): self._make_qr(1, 18000, 0.04),
            ("b", 2): self._make_qr(2, 18000, 0.04),
            ("b", 3): self._make_qr(3, 1000, 0.04),  # low lines
        }
        qa.assess(results)
        self.assertEqual(results[("b", 3)].status, "WARN")
        self.assertTrue(any("arc lines" in f for f in results[("b", 3)].flags))
        # Other quanta should be OK
        self.assertEqual(results[("b", 1)].status, "OK")
        self.assertEqual(results[("b", 2)].status, "OK")

    def test_missing_species_flagged(self):
        """A species present in peers but absent here is flagged."""
        qr1 = qa.QuantumResult(arm="b", spectrograph=1, log_file="test.log")
        qr1.has_final_result = True
        qr1.final_xRMS = 0.03
        qr1.final_yRMS = 0.04
        qr1.final_ySoften = 0.02
        qr1.final_species = {"CdI": 200, "HgI": 400, "NeI": 500}

        qr3 = qa.QuantumResult(arm="b", spectrograph=3, log_file="test.log")
        qr3.has_final_result = True
        qr3.final_xRMS = 0.03
        qr3.final_yRMS = 0.04
        qr3.final_ySoften = 0.02
        qr3.final_species = {"HgI": 100, "NeI": 200}  # CdI missing!

        results = {("b", 1): qr1, ("b", 3): qr3}
        qa.assess(results)
        self.assertEqual(results[("b", 3)].status, "WARN")
        self.assertTrue(any("CdI" in f for f in results[("b", 3)].flags))

    def test_custom_yrms_thresholds(self):
        """Custom --warn-yrms and --bad-yrms thresholds are honoured."""
        results = {("b", 1): self._make_qr(1, 18000, 0.20)}
        qa.assess(results, yrms_warn=0.15, yrms_bad=0.50)
        self.assertEqual(results[("b", 1)].status, "WARN")

        results2 = {("b", 1): self._make_qr(1, 18000, 0.20)}
        qa.assess(results2, yrms_warn=0.25, yrms_bad=0.50)
        self.assertEqual(results2[("b", 1)].status, "OK")


# ---------------------------------------------------------------------------
# Integration tests — against the real log files
# ---------------------------------------------------------------------------


class TestRealLogs(unittest.TestCase):
    """Smoke tests against the repo's example log files.

    Skipped automatically when the log files are not present.
    """

    repo_root = Path(__file__).parent.parent

    def _skip_if_missing(self, name: str) -> Path:
        p = self.repo_root / name
        if not p.exists():
            self.skipTest(f"Log file not found: {p}")
        return p

    def test_dm02_sm3_flagged(self):
        """run28-dm-02.log: SM3 b-arm must be flagged BAD."""
        log = self._skip_if_missing("run28-dm-02.log")
        results = qa.parse_log(log)
        qa.assess(results)
        self.assertIn(("b", 3), results)
        self.assertEqual(results[("b", 3)].status, "BAD")

    def test_dm02_other_sms_ok(self):
        """run28-dm-02.log: SM1/2/4 b-arm must all be OK."""
        log = self._skip_if_missing("run28-dm-02.log")
        results = qa.parse_log(log)
        qa.assess(results)
        for sm in (1, 2, 4):
            with self.subTest(sm=sm):
                self.assertEqual(results[("b", sm)].status, "OK")

    def test_dm02_sm3_ysoften_nan(self):
        """run28-dm-02.log: SM3 Final result ySoften is NaN."""
        log = self._skip_if_missing("run28-dm-02.log")
        results = qa.parse_log(log)
        self.assertTrue(math.isnan(results[("b", 3)].final_ySoften))

    def test_dm03_sm3_flagged(self):
        """run28-dm-03.log: SM3 is still flagged BAD despite per-species S/N."""
        log = self._skip_if_missing("run28-dm-03.log")
        results = qa.parse_log(log)
        qa.assess(results)
        self.assertEqual(results[("b", 3)].status, "BAD")

    def test_dm02_four_quanta(self):
        """run28-dm-02.log: exactly four fitDetectorMap quanta should be found."""
        log = self._skip_if_missing("run28-dm-02.log")
        results = qa.parse_log(log)
        self.assertEqual(len(results), 4)

    def test_dm02_exec_time_present(self):
        """Execution times are captured for all quanta in dm-02."""
        log = self._skip_if_missing("run28-dm-02.log")
        results = qa.parse_log(log)
        for k, qr in results.items():
            with self.subTest(quantum=k):
                self.assertFalse(math.isnan(qr.exec_time_s))
                self.assertGreater(qr.exec_time_s, 0)

    def test_main_exit_code(self):
        """main() returns 1 (BAD found) for the real log files."""
        log = self._skip_if_missing("run28-dm-02.log")
        rc = qa.main([str(log)])
        self.assertEqual(rc, 1)

    def test_main_output_table(self):
        """main() produces a table with the expected arm/SM rows."""
        log = self._skip_if_missing("run28-dm-02.log")
        captured = StringIO()
        orig_stdout = sys.stdout
        sys.stdout = captured
        try:
            qa.main([str(log)])
        finally:
            sys.stdout = orig_stdout
        output = captured.getvalue()
        for sm in ("b    1", "b    2", "b    3", "b    4"):
            self.assertIn(sm, output)


if __name__ == "__main__":
    unittest.main()
