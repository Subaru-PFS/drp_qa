"""Tests that the PostgreSQL fixture empties only a database named as a scratch one."""

import pytest


class RecordingDB:
    """A database that answers ``current_database()`` with ``name`` and records what is committed."""

    def __init__(self, name):
        self.name = name
        self.committed = []

    def query_scalar(self, sql, /, *, params=None, conn=None):
        assert sql == "SELECT current_database()", sql
        return self.name

    def commit(self, query, params=None):
        self.committed.append(query)


@pytest.mark.parametrize("name", ["qadb_tests", "qadb_pipe2d_2013_tests", "qadb_test"])
def testScratchDatabaseIsCleared(scratchGuard, name):
    db = RecordingDB(name)
    scratchGuard.clearTestRows(db)
    emptied = [query.split()[-1] for query in db.committed if "WHERE" not in query]
    assert emptied == list(reversed(scratchGuard.QA_TABLES))
    # The wholesale QA_TABLES empties, then the four scoped deletes (qa_normal_value,
    # slit_offset_confirmed, data_processing_pipe2d, pfs_visit).
    assert len(db.committed) == len(scratchGuard.QA_TABLES) + 4


@pytest.mark.parametrize("name", ["qadb", "opdb", "qadb_tests_copy", "tests"])
def testOtherDatabaseIsRefused(scratchGuard, name):
    # Negative control: the same call on a database not named as a scratch one deletes nothing.
    db = RecordingDB(name)
    with pytest.raises(scratchGuard.NotScratchError, match=f"'{name}'"):
        scratchGuard.clearTestRows(db)
    assert db.committed == []
