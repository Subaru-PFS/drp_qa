"""Tests for `pfs.drp.qa.comparison.persistence`."""

import pandas as pd

from pfs.drp.qa.comparison.classify import classifyVisits
from pfs.drp.qa.comparison.persistence import gapSummary, lastLitBefore


def testLastLitBefore(periods, visit, listing):
    visits = classifyVisits(
        listing(
            visit(1, "2026-01-03 17:00", "darks", cameras="n1,n2", exptime=300),  # nothing lit before
            visit(2, "2026-01-03 18:00", "scienceArc", cameras="n1,n2", exptime=60, name="Arc: Neon"),
            visit(3, "2026-01-03 18:03", "darks", cameras="n1,n2", exptime=300),  # 2 min after 2 ended
            visit(4, "2026-01-03 18:10", "scienceTrace", cameras="b1,b2", exptime=20),  # lights no n camera
            visit(5, "2026-01-03 18:20", "darks", cameras="n1,n2", exptime=300),  # still after 2, not 3 or 4
            visit(6, "2026-01-03 18:30", "scienceTrace", cameras="n2", exptime=20, name="Trace"),
            visit(7, "2026-01-03 18:31", "darks", cameras="n1,n2", exptime=300),  # n1 after 2, n2 after 6
        ),
        periods.values(),
    )
    result = lastLitBefore(visits)
    rows = {(row.visit, row.cameras): (row.litVisit, row.minutesSince) for row in result.itertuples()}
    assert rows[(3, "n1,n2")] == (2, 2.0)
    assert rows[(5, "n1,n2")] == (2, 19.0)
    assert rows[(7, "n2")][0] == 6 and rows[(7, "n2")][1] == 0.7
    assert rows[(7, "n1")] == (2, 30.0)
    assert pd.isna(rows[(1, "n1,n2")][0])
    assert result.loc[result["litVisit"] == 2, "litType"].iloc[0] == "scienceArc 'Arc: Neon'"
    assert result["minutesSince"].iloc[0] == 0.7  # shortest first

    summary = gapSummary(result).set_index("after the last lit exposure")["darks"].to_dict()
    assert summary == {"< 5 min": 2, "5-30 min": 1, "30-120 min": 0, "> 120 min": 0, "none earlier": 1}


def testNoDarks(periods, visit, listing):
    visits = classifyVisits(
        listing(visit(1, "2026-01-03 18:00", "scienceArc", cameras="n1")), periods.values()
    )
    assert lastLitBefore(visits).empty


def testDarkSequences(periods, visit, listing):
    from pfs.drp.qa.comparison.persistence import darkSequences

    visits = classifyVisits(
        listing(
            visit(
                1, "2026-01-03 17:00", "darks", sequence=10, cameras="n1", exptime=300
            ),  # nothing lit before
            visit(2, "2026-01-03 17:06", "darks", sequence=10, cameras="n1", exptime=300),
            visit(
                3, "2026-01-03 18:00", "scienceArc", sequence=11, cameras="n1", exptime=60, name="Arc: Neon"
            ),
            *(
                visit(
                    4 + i, f"2026-01-03 18:{2 + 6 * i:02d}", "darks", sequence=12, cameras="n1", exptime=300
                )
                for i in range(3)
            ),
        ),
        periods.values(),
    )
    sequences = darkSequences(lastLitBefore(visits)).set_index("iic_sequence_id")
    assert sequences.loc[12, ["firstVisit", "darks", "litVisit", "minutesSince"]].tolist() == [4, 3, 3, 1.0]
    # The first dark of sequence 10 had nothing lit before it; a later dark mustn't fill that in.
    assert sequences.loc[10, "firstVisit"] == 1 and pd.isna(sequences.loc[10, "litVisit"])
    assert sequences.index.tolist() == [12, 10]  # shortest gap first, none last
