"""Tests for `pfs.drp.qa.comparison.queries`, against a fake opdb."""

import datetime

import pandas as pd
import pytest

from pfs.drp.qa.comparison import queries
from pfs.drp.qa.comparison.queries import LISTING_COLUMNS, proposalCategory


@pytest.mark.parametrize(
    "proposalId, category",
    [
        ("S25B-EN16", "EN"),
        ("S25A-123QF", "QF"),
        ("S25B-045QN", "QN"),
        ("S26A-OT02", "OT"),
        ("S25A-UH018-A", "UH"),
        ("S25B-TE007", "TE"),
        ("N/A", "none"),
        ("", "none"),
        (None, "none"),
        ("S25A-123", "other"),  # semester and number only
        ("engineering", "other"),
    ],
)
def testProposalCategory(proposalId, category):
    assert proposalCategory(proposalId) == category


def testListingBindsTheInterval(FakeOpDB):
    answer = pd.DataFrame(
        {
            "pfs_visit_id": [141578, 141577],
            "time_exp_start": ["2026-05-04 22:10:47", "2026-05-04 22:08:25"],
            "iic_sequence_id": [61288, None],
            "pfs_design_id": [1, 1],
        }
    )
    opdb = FakeOpDB({"FROM pfs_visit": answer})
    start, end = datetime.datetime(2026, 5, 4, 12), datetime.datetime(2026, 5, 5, 12)
    result = queries.readVisitListing(opdb, start, end)

    call = opdb.calls[0]
    assert call.params == {"start": start, "end": end}
    assert ":start" in call.sql and ":end" in call.sql
    assert str(start) not in call.sql  # bound, not formatted in
    assert list(result.columns) == list(LISTING_COLUMNS)
    assert list(result["pfs_visit_id"]) == [141577, 141578]
    assert result["iic_sequence_id"].isna().tolist() == [True, False]  # a visit outside any sequence is kept
    assert pd.api.types.is_datetime64_any_dtype(result["time_exp_start"])


def testNotesFromVisitsAndSequences(FakeOpDB):
    visitNotes = pd.DataFrame(
        {
            "pfs_visit_id": [141581, 141590],
            "camera": [None, "b2"],
            "data_flag": [None, 1],
            "note": ["SM4 is missing", "ghost"],
            "source": ["obslog", "sps_annotation"],
        }
    )
    sequenceNotes = pd.DataFrame({"iic_sequence_id": [61294], "note": ["Dont use"]})
    opdb = FakeOpDB({"obslog_visit_set_note": sequenceNotes, "obslog_visit_note": visitNotes})
    notes = queries.readNotes(opdb, [141590, 141581, 141581], [61294])

    assert opdb.calls[0].params == {"visits": [141581, 141590]}
    assert opdb.calls[1].params == {"sequences": [61294]}
    assert list(notes["source"]) == ["obslog", "sps_annotation", "obslog_sequence"]
    assert notes["iic_sequence_id"].isna().tolist() == [True, True, False]


def testNothingToReadMakesNoQuery(FakeOpDB):
    opdb = FakeOpDB({})
    assert queries.readNotes(opdb, [], []).empty
    assert queries.readTelStatus(opdb, []).empty
    assert queries.readDesignCategories(opdb, []).empty
    assert opdb.calls == []


def testDesignCategoriesDropProposalIds(FakeOpDB):
    proposals = pd.DataFrame(
        {
            "pfs_design_id": [7, 7, 7, 8],
            "proposal_id": ["S25B-EN16", "S25B-EN17", "N/A", "S25A-123QF"],
            "n_fibers": [100, 20, 5, 400],
        }
    )
    opdb = FakeOpDB({"pfs_design_fiber": proposals})
    categories = queries.readDesignCategories(opdb, [8, 7])

    assert "target_type = 1" in opdb.calls[0].sql  # science fibers only
    assert opdb.calls[0].params == {"designs": [7, 8]}
    assert list(categories.columns) == ["pfs_design_id", "category", "n_fibers"]
    assert categories.to_dict("records") == [
        {"pfs_design_id": 7, "category": "EN", "n_fibers": 120},
        {"pfs_design_id": 7, "category": "none", "n_fibers": 5},
        {"pfs_design_id": 8, "category": "QF", "n_fibers": 400},
    ]
