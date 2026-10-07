"""Tests for `pfs.drp.qa.comparison.classify`."""

import numpy as np
import pandas as pd

from pfs.drp.qa.comparison.classify import classifyVisits, designKinds, focusSweeps


def testDesignKinds():
    categories = pd.DataFrame(
        {
            "pfs_design_id": [1, 1, 2, 2, 3, 4],
            "category": ["EN", "none", "EN", "QF", "UH", "other"],
        }
    )
    kinds = designKinds(categories)
    # 2 is shared by an engineering and a science proposal: science.
    assert kinds.to_dict() == {1: "engineering", 2: "science", 3: "science", 4: "engineering"}


def _sweepVisits(offsets, night="2026-05-04", name="sky1540+3500 pa=90"):
    return pd.DataFrame(
        {
            "night": [pd.Timestamp(night).date()] * len(offsets),
            "sequence_name": name,
            "focus_offset_max": offsets,
        }
    )


def testFocusSweepFound():
    # Run28's sweep: 0.075 mm steps from 2.705 to 3.755 mm.
    sweep = _sweepVisits(np.round(np.arange(2.705, 3.756, 0.075), 3))
    assert focusSweeps(sweep).all()


def testFocusCorrectionIsNotASweep():
    # The focus correction during a night: as many visits, about 0.1 mm.
    tracking = _sweepVisits(np.round(np.linspace(3.21, 3.30, 15), 3))
    assert not focusSweeps(tracking).any()


def testFewStepsAreNotASweep():
    # A large range in a few steps: a refocus, not a sweep.
    refocus = _sweepVisits([2.7, 2.7, 3.2, 3.2, 3.7, 3.7] * 3)
    assert not focusSweeps(refocus).any()


def testSweepIsPerFieldAndNight():
    # Ten steps, but split over two fields: neither is a sweep.
    offsets = np.round(np.arange(2.705, 3.456, 0.075), 3)
    split = pd.concat(
        [_sweepVisits(offsets[:5], name="a"), _sweepVisits(offsets[5:], name="b")], ignore_index=True
    )
    assert not focusSweeps(split).any()
    together = _sweepVisits(offsets, name="a")
    assert focusSweeps(together).all()


def testClassifyVisits(periods, visit, listing):
    visits = listing(
        visit(1, "2026-01-03 18:00", "scienceArc"),
        visit(2, "2026-01-03 18:05", "scienceTrace"),
        visit(3, "2026-01-03 20:00", "scienceObject", design=10),  # science
        visit(4, "2026-01-03 19:00", "scienceObject", design=11),  # twilight, engineering design
        visit(5, "2026-01-03 19:10", "scienceObject_windowed", design=11),
        visit(6, "2026-01-03 21:00", "scienceObject", design=12),  # a design with no science fibers
        visit(7, "2026-01-04 15:00", "darks"),
        visit(8, "2026-01-04 15:10", "darks", expType="test"),
        visit(9, "2026-01-04 15:20", "scienceArc", expType="test"),
        visit(10, "2026-01-04 15:30", None),  # no sequence
        visit(11, "2026-01-20 18:00", "scienceArc"),  # between runs
        visit(12, "2026-02-02 18:00", "scienceArc"),  # run2's pre-run period
    )
    designs = designKinds(pd.DataFrame({"pfs_design_id": [10, 11], "category": ["QF", "EN"]}))
    result = classifyVisits(visits, periods.values(), designs).set_index("pfs_visit_id")

    assert result.loc[3, "category"] == "science"
    for calibration in (1, 2, 4, 5, 6, 7):
        assert result.loc[calibration, "category"] == "calibration", calibration
    assert result["reason"].to_dict() == {
        1: "judged",
        2: "judged",
        3: "judged",
        4: "judged",
        5: "no method for scienceObject_windowed",
        6: "judged",
        7: "no method for darks",
        8: "test exposure",
        9: "test exposure",
        10: "no sequence",
        11: "outside every period",
        12: "judged",
    }
    assert result["judged"].sum() == 6
    assert result.loc[12, "period"] == "run2-pre"
    assert pd.isna(result.loc[11, "period"])


def testWithoutDesignsSkyIsScience(periods, visit, listing):
    visits = listing(visit(1, "2026-01-03 20:00", "scienceObject", design=10))
    assert classifyVisits(visits, periods.values())["category"].tolist() == ["science"]


def testSubLabels(periods, visit, listing):
    offsets = np.round(np.arange(2.705, 3.756, 0.075), 3)
    rows = [
        visit(100 + i, f"2026-01-03 22:{i:02d}", "scienceObject", name="M39", design=11)
        for i in range(len(offsets))
    ]
    rows.append(visit(200, "2026-01-03 23:30", "scienceObject", name="dither", design=11))
    rows.append(visit(300, "2026-01-03 23:40", "scienceArc"))
    telStatus = pd.DataFrame(
        {
            "pfs_visit_id": [*range(100, 100 + len(offsets)), 200, 300],
            "focus_offset_max": [*offsets, 3.2, 3.2],
            "dither_ra_max": [0.0] * len(offsets) + [1.0, 1.0],
            "dither_dec_max": 0.0,
        }
    )
    result = classifyVisits(listing(*rows), periods.values(), telStatus=telStatus).set_index("pfs_visit_id")
    assert result.loc[100 : 100 + len(offsets) - 1, "focusSweep"].all()
    assert not result.loc[[200, 300], "focusSweep"].any()
    assert result.loc[200, "dithered"]
    assert not result.loc[300, "dithered"]  # only sky visits are dithered
