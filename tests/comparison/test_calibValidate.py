"""Tests for `pfs.drp.qa.comparison.calibValidate` on two synthetic candidates."""

import pandas as pd

from pfs.drp.qa.comparison.calibValidate import buildValidationReport, compareCalibrations, validationSummary


def _metrics(offset: float, rms: float, visits=(100, 101, 200)):
    rows = []
    for visit in visits:
        for sp in (1, 2):
            for species in ("NeI", "Trace"):
                for reference, statusType, metric, value in (
                    ("calib", "BOTH", "medianDeltaX", offset if visit >= 200 else 0.0),
                    ("calib", "BOTH", "medianDeltaY", -offset if visit >= 200 else 0.0),
                    ("calib", "BOTH", "robustRmsDeltaX", rms),
                    ("calib", "BOTH", "robustRmsDeltaY", rms),
                    ("calib", "BOTH", "nLines", 1000.0),
                    ("visit", "RESERVED", "robustRmsDeltaX", rms / 2),
                    ("visit", "RESERVED", "robustRmsDeltaY", rms / 2),
                    ("visit", "RESERVED", "nLines", 100.0),
                ):
                    if species == "Trace" and metric.endswith("Y"):
                        continue
                    rows.append((visit, "b", sp, reference, species, statusType, metric, value))
    return pd.DataFrame(
        rows,
        columns=["visit", "arm", "spectrograph", "reference", "description", "statusType", "metric", "value"],
    )


def testCompareAndSummarise():
    table = compareCalibrations(
        {"asis": _metrics(0.5, 0.10), "pick": _metrics(0.1, 0.08)}, setVisits=[100, 101]
    )
    assert set(table["calibration"]) == {"asis", "pick"}
    assert set(table["detector"]) == {"b1", "b2"}
    assert (
        table.loc[table["visit"] == 100, "inSet"].all()
        and not table.loc[table["visit"] == 200, "inSet"].any()
    )
    ne = table[(table["species"] == "NeI") & (table["metric"] == "reserved RMS x")]
    assert (ne["nLines"] == 100.0).all()
    summary = validationSummary(table).set_index(["detector", "metric", "calibration"])
    assert summary.loc[("b1", "offset x vs calib", "asis"), "later"] == 0.5
    assert summary.loc[("b1", "offset x vs calib", "pick"), "later"] == 0.1
    assert (
        summary.loc[("b1", "offset x vs calib", "asis"), "set"] == 0.0
    )  # the set reads zero against its own build
    assert summary.loc[("b1", "reserved RMS x", "pick"), "set"] == 0.04
    # Negative control: an offset's sign must not cancel in the summary (absolute values).
    table2 = compareCalibrations({"asis": _metrics(-0.5, 0.10)}, setVisits=[100, 101])
    assert (
        validationSummary(table2)
        .set_index(["detector", "metric", "calibration"])
        .loc[("b1", "offset y vs calib", "asis"), "later"]
        == 0.5
    )


def testUnfittedCalibrationResidualsAreCompared():
    """A visit whose fit marked no usable lines stores its calibration residuals as UNFITTED."""
    frame = _metrics(0.5, 0.10)
    unfitted = (frame["visit"] == 200) & (frame["reference"] == "calib")
    frame.loc[unfitted, "statusType"] = "UNFITTED"
    table = compareCalibrations({"asis": frame}, setVisits=[100, 101])
    rows = table[(table["visit"] == 200) & (table["metric"] == "offset x vs calib")]
    assert len(rows) == 4 and (rows["value"] == 0.5).all() and (rows["nLines"] == 1000.0).all()
    # Negative control: a line set the comparison does not read stays out.
    frame.loc[unfitted, "statusType"] = "USED"
    table = compareCalibrations({"asis": frame})
    assert table[(table["visit"] == 200) & (table["metric"] == "offset x vs calib")].empty


def testReport():
    table = compareCalibrations(
        {"asis": _metrics(0.5, 0.10), "pick": _metrics(0.1, 0.08)}, setVisits=[100, 101]
    )
    report = buildValidationReport(
        table,
        "Run99 b1, b2",
        {"asis": "PFS/calib/x", "pick": "u/me/pick"},
        figures={"b1": "v-b1.png"},
        visitsNote="Visits: 100–200.",
    )
    assert report.startswith("# Run99 b1, b2")
    assert "| pick | `u/me/pick` |" in report
    assert "## b1" in report and "![b1](v-b1.png)" in report
    assert "| offset x vs calib | 0.000 (n=2) | 0.500 (n=1) | 0.000 (n=2) | 0.100 (n=1) |" in report


# --- slit-offset health and its report section (PIPE2D-1677) ------------------

import numpy as np  # noqa: E402

from pfs.drp.qa.comparison.calibValidate import buildSlitOffsetSection, slitOffsetHealth  # noqa: E402
from pfs.drp.qa.metrics.slitOffsets import CONFIRMED_COLUMNS, crossArmConfirmed  # noqa: E402


def _offsets(arm, spectrograph, bad=None, seed=1):
    """Return a `readSlitOffsets`-shaped frame: 4 MTP blocks of 20 fibers, a trend, maybe one bad fiber."""
    rng = np.random.default_rng(seed)
    nBlocks, per = 4, 20
    n = nBlocks * per
    group = np.repeat([f"MTP{b}" for b in range(nBlocks)], per)
    spatial = np.repeat(np.arange(nBlocks) * 1.0, per) + rng.normal(0.0, 0.01, n)
    spectral = rng.normal(0.0, 0.01, n)
    if bad is not None:
        spatial[bad - 1] += 0.3
    return pd.DataFrame(
        {
            "arm": arm,
            "spectrograph": spectrograph,
            "fiberId": np.arange(1, n + 1),
            "spatial": spatial,
            "spectral": spectral,
            "group": group,
        }
    )


def _run(bad):
    return pd.concat([_offsets("b", 1, bad=bad), _offsets("r", 1, bad=bad)], ignore_index=True)


def testSlitOffsetHealthFlagsPerDetector():
    flagged = slitOffsetHealth(_run(bad=27))
    assert set(flagged["arm"]) == {"b", "r"}
    assert set(flagged.loc[flagged["outlier"], "fiberId"]) == {27}
    confirmed = crossArmConfirmed(flagged)
    assert confirmed["fiberId"].to_list() == [27] and confirmed.iloc[0]["nArms"] == 2


def testSlitOffsetSectionListsConfirmed():
    md = buildSlitOffsetSection(crossArmConfirmed(slitOffsetHealth(_run(bad=27))))
    assert "## Slit offsets" in md and "SM1" in md and "27" in md


def testSlitOffsetSectionEmpty():
    empty = pd.DataFrame({c: pd.Series(dtype=d) for c, d in CONFIRMED_COLUMNS.items()})
    assert "No confirmed slit-offset outliers." in buildSlitOffsetSection(empty)


def testSlitOffsetSectionChangeVsPrevious():
    current = crossArmConfirmed(slitOffsetHealth(_run(bad=27)))
    previous = crossArmConfirmed(slitOffsetHealth(_run(bad=50)))
    md = buildSlitOffsetSection(current, previous, previousLabel="run30")
    assert "1 new" in md and "cleared" in md and "run30" in md
    # The status column is a column of its own, not merged into the last header cell.
    header = next(line for line in md.splitlines() if line.startswith("| detector"))
    cells = [c.strip() for c in header.strip("|").split("|")]
    assert cells[-1] == "status" and len(cells) == 6
