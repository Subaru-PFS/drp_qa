"""`pfs.drp.qa.metrics.lineFlux`: per-species line flux statistics."""

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.metrics.lineFlux import speciesLineFlux
from pfs.drp.qa.metrics.longFormat import SPECIES_COLUMNS, metricsFrame, speciesFrame, speciesStats


def makeLines() -> pd.DataFrame:
    """Return lines of two species and trace centroids; NeI has a flagged, a blended and an unmeasured line."""
    return pd.DataFrame(
        {
            "description": ["NeI"] * 7 + ["ArI"] * 3 + ["Trace"] * 2,
            "flux": [100.0, 200.0, 300.0, 400.0, 1.0e6, 5.0e5, np.nan, 10.0, 20.0, 60.0, 7.0, 7.0],
            "flag": [False] * 4 + [True, False, True] + [False] * 5,
            "status": [0, 0x20, 0x40, 0, 0, 0x02, 0, 0, 0, 0, 0, 0],
        }
    )


def testSpeciesLineFluxIsPerSpeciesOverGoodLines():
    """Each species has its own median and robust RMS, over unflagged, unblended, measured lines."""
    stats = speciesLineFlux(makeLines())
    assert sorted(stats) == ["ArI", "NeI"]  # no Trace
    # USED (0x20) and RESERVED (0x40) lines count; the flagged, BLEND and NaN ones don't.
    assert stats["NeI"]["nFluxLines"] == 4
    assert stats["NeI"]["medLineFlux"] == pytest.approx(250.0)
    assert stats["NeI"]["robustRmsLineFlux"] == pytest.approx(0.741 * 150.0)
    assert stats["ArI"]["medLineFlux"] == pytest.approx(20.0)
    assert "nSaturatedLines" not in stats["NeI"]

    # Negative control: blended across species (R7), the median is neither species' own.
    blended = makeLines().assign(description="X")
    assert speciesLineFlux(blended)["X"]["medLineFlux"] not in (pytest.approx(250.0), pytest.approx(20.0))


def testSpeciesLineFluxSelectsRowsAndCountsSaturation():
    """``select`` limits the rows; saturated lines are counted whether or not they are flagged."""
    lines = makeLines()
    select = np.ones(len(lines), dtype=bool)
    select[0] = False  # NeI 100.0 is on an unlit fiber
    saturated = np.zeros(len(lines), dtype=bool)
    saturated[[0, 4, 10]] = True  # the unselected line, the flagged NeI line, a trace centroid
    stats = speciesLineFlux(lines, select=select, saturated=saturated)
    assert stats["NeI"]["nFluxLines"] == 3
    assert stats["NeI"]["medLineFlux"] == pytest.approx(300.0)
    assert stats["NeI"]["nSaturatedLines"] == 1
    assert stats["ArI"]["nSaturatedLines"] == 0


def testSpeciesWithNoGoodLineHasNanFlux():
    """A species whose lines are all flagged is reported, with no flux: its absence is the finding."""
    lines = makeLines()
    lines.loc[lines["description"] == "ArI", "flag"] = True
    stats = speciesLineFlux(lines)["ArI"]
    assert stats["nFluxLines"] == 0
    assert np.isnan(stats["medLineFlux"]) and np.isnan(stats["robustRmsLineFlux"])
    assert speciesLineFlux(lines.iloc[:0]) == {}
    assert speciesLineFlux(lines.drop(columns="description")) == {}


def testMetricsFrameIsLongFormat():
    """`metricsFrame` has the species table's schema, and its rows don't disturb `speciesStats`."""
    dataId = {"visit": 12345, "arm": "r", "spectrograph": 1}
    flux = metricsFrame(dataId, speciesLineFlux(makeLines()))
    assert list(flux.columns) == list(SPECIES_COLUMNS)
    assert dict(flux.dtypes.astype(str)) == dict(speciesFrame(dataId, {}).dtypes.astype(str))
    assert flux[flux["description"] == "NeI"]["metric"].tolist() == [
        "medLineFlux",
        "robustRmsLineFlux",
        "nFluxLines",
    ]
    assert len(metricsFrame(dataId, {})) == 0

    # The fit statistics read back as before with flux rows beside them; ArI, which has flux
    # rows alone, is not given NaN fit statistics.
    both = pd.concat([speciesFrame(dataId, {"NeI": (0.02, 0.03, 0.0, 0.0)}), flux], ignore_index=True)
    assert speciesStats(both) == {"NeI": (0.02, 0.03)}
