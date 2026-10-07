"""Tests for `pfs.drp.qa.comparison.cli` that need neither the opdb nor a Butler."""

import datetime

import pandas as pd
import pytest

from pfs.drp.qa.comparison.cli import fetch, loadFetched


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
