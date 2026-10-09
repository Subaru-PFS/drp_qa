"""Tests for `pfs.drp.qa.comparison.cli` that need neither the opdb nor a Butler."""

import datetime

import pandas as pd
import pytest

from pfs.drp.qa.comparison.cli import fetch, fetchSummary, loadFetched


def testFetchAndLoad(tmp_path, periods, visit, listing, FakeOpDB):
    answer = listing(
        visit(1, "2026-01-03 18:00", "scienceArc"),
        visit(2, "2026-01-03 20:00", "scienceObject", design=10),
    )
    opdb = FakeOpDB(
        {
            "FROM pfs_visit": answer,
            "obslog_visit_set_note": pd.DataFrame({"iic_sequence_id": [1], "note": ["fine"]}),
            "obslog_visit_note": pd.DataFrame(
                columns=["pfs_visit_id", "camera", "data_flag", "note", "source"]
            ),
            "FROM tel_status": pd.DataFrame(
                {"pfs_visit_id": [2], "n_status": [9], "focus_offset_max": [3.2]}
            ),
            "pfs_design_fiber": pd.DataFrame(
                {"pfs_design_id": [10], "proposal_id": ["S25A-123QF"], "n_fibers": [5]}
            ),
        }
    )
    period = periods["run1"]
    now = datetime.datetime(2026, 1, 4, 9)  # during the run: read up to now
    paths = fetch(opdb, period, tmp_path, now=now)

    assert opdb.calls[0].params == {"start": period.start, "end": now}
    telCall = next(call for call in opdb.calls if "tel_status" in call.sql)
    assert telCall.params == {"visits": [2]}  # sky visits only
    frames, stamp = loadFetched(period, tmp_path)
    assert set(paths) == set(frames) == {"listing", "notes", "telStatus", "designs"}
    assert frames["listing"]["pfs_visit_id"].tolist() == [1, 2]
    assert frames["designs"]["category"].tolist() == ["QF"]
    assert "proposal_id" not in frames["designs"]
    assert stamp["readUntil"] == str(now)


def testLoadBeforeFetch(tmp_path, periods):
    with pytest.raises(FileNotFoundError, match="run 'fetch' first"):
        loadFetched(periods["run1"], tmp_path)


def testFetchSummary(tmp_path, periods, visit, listing, FakeOpDB):
    answer = listing(
        visit(1, "2026-01-03 18:00", "scienceArc", sequence=10),
        visit(2, "2026-01-03 20:00", "scienceObject", design=10),
        visit(3, "2026-01-04 15:00", "darks", sequence=11),
    )
    opdb = FakeOpDB(
        {
            "FROM pfs_visit": answer,
            "obslog_visit_set_note": pd.DataFrame({"iic_sequence_id": [10], "note": ["fine"]}),
            "obslog_visit_note": pd.DataFrame(
                columns=["pfs_visit_id", "camera", "data_flag", "note", "source"]
            ),
            "FROM tel_status": pd.DataFrame(
                {"pfs_visit_id": [2], "n_status": [9], "focus_offset_max": [3.2]}
            ),
            "pfs_design_fiber": pd.DataFrame(
                {"pfs_design_id": [10], "proposal_id": ["S25A-123QF"], "n_fibers": [5]}
            ),
        }
    )
    fetch(opdb, periods["run1"], tmp_path, now=datetime.datetime(2026, 2, 1))
    frames, stamp = loadFetched(periods["run1"], tmp_path)
    text = fetchSummary(frames, stamp, periods["run1"], periods)
    assert "3 visits in 3 sequences, on 2 nights" in text
    assert (
        "2 gated" in text
        and "0 unvalidated" in text
        and "1  calibration  scienceArc single" in text
        and "1  science      scienceObject" in text
    )
    assert "1  no method for darks" in text
    assert "sky: 1 visits (1 with telescope status), 1 science" in text
    assert "notes: 1 (1 on sequences)" in text


def testQgraphFromRun():
    from pfs.drp.qa.comparison.cli import _qgraph

    command = ["pipetask", "--long-log", "run", "-j", "8", "-b", "/repo", "-d", "visit IN (1)"]
    assert _qgraph(command) == ["pipetask", "--long-log", "qgraph", "-b", "/repo", "-d", "visit IN (1)"]


def testPipetaskFailures():
    from pfs.drp.qa.comparison.cli import pipetaskFailures

    log = "\n".join(
        [
            "INFO 2026-10-08T07:53:20 lsst.ctrl.mpexec ... Executing 40 quanta",
            "ERROR 2026-10-08T07:54:01 lsst.ctrl.mpexec ... Task <reduceExposure dataId={visit: 148144}> failed",
            "ERROR 2026-10-08T07:54:01 lsst.ctrl.mpexec ... Task <reduceExposure dataId={visit: 148144}> failed",
            "INFO 2026-10-08T07:54:09 lsst.ctrl.mpexec ... Executed 30 quanta successfully, 1 failed and 9 remain"
            " out of total 40 quanta.",
        ]
    )
    summary, lines = pipetaskFailures(log)
    assert summary == "Executed 30 quanta successfully, 1 failed and 9 remain out of total 40 quanta."
    assert len(lines) == 1 and "148144" in lines[0]  # repeated lines once
    traceback = (
        "ERROR ... Caught an exception\nTraceback:\n  File x.py\nValueError: Output CHAINED collection"
    )
    assert pipetaskFailures(traceback)[1][-1] == "ValueError: Output CHAINED collection"
    assert pipetaskFailures("INFO nothing to see") == (None, [])
    many = "\n".join(f"ERROR quantum {i} failed" for i in range(50))
    assert pipetaskFailures(many, limit=5)[1][0] == "ERROR quantum 45 failed"  # the last ones


def testResolveCollections():
    from pfs.drp.qa.comparison.cli import resolveCollections

    assert resolveCollections(False, "PFS/defaults", None, None) == (
        ["drpActor/reductions"],
        ["drpActor/reductions", "PFS/defaults"],
    )
    assert resolveCollections(True, "PFS/defaults", None, None) == ([], ["PFS/defaults"])
    # Negative control: --fresh with drpActor's reductions among the inputs would reuse them.
    with pytest.raises(ValueError, match="--fresh"):
        resolveCollections(True, "PFS/defaults", None, ["drpActor/reductions", "PFS/defaults"])
    with pytest.raises(ValueError, match="--fresh"):
        resolveCollections(True, "PFS/defaults", ["drpActor/reductions"], None)


def testPipelineVersion():
    from pfs.drp.qa.comparison.cli import pipelineVersion

    weekly = {"DATAMODEL": "w.2026.40", "OBS_PFS": "w.2026.40", "DRP_STELLA": "w.2026.40"}
    assert pipelineVersion(weekly) == "pfs-w.2026.40"
    mixed = weekly | {"HIERARCH OBS_PFS": "w.2026.41-2-gabc"}
    del mixed["OBS_PFS"]
    assert pipelineVersion(mixed) == "datamodel-w.2026.40+drp_stella-w.2026.40+obs_pfs-w.2026.41-2-gabc"
    with pytest.raises(RuntimeError, match="Unknown"):
        pipelineVersion(weekly | {"DRP_STELLA": "unknown"})
