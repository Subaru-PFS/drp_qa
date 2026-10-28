"""Tests for reading the dashboard's Tier 1 table back from qadb (`pfs.drp.qa.dashboard.qadbSource`).

Each test loads the fixtures with the loader, as qaActor does, and reads them back.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pfs.drp.qa.dashboard.qadbSource import metricsTableFromQadb, readQadb
from pfs.drp.qa.dashboard.schema import VERDICT
from pfs.drp.qa.dashboard.serving import taskThresholds
from pfs.drp.qa.qadb.loader import loadCollection

COLLECTION = "u/test/qa"
VISITS = [133025, 133040]
NOW = datetime(2026, 10, 26, 12, 0, 0)
LATER = datetime(2026, 10, 26, 12, 5, 0)
IMAGE = ["visit", "arm", "spectrograph"]


def load(butler, qadb, visits=VISITS, now=NOW):
    loadCollection(
        butler, qadb, COLLECTION, visits=visits, pipelineVersions=False, log=lambda _: None, now=lambda: now
    )


@pytest.fixture
def butler(StubButler, fixtures, configs):
    return StubButler(fixtures, datasets={"packages": {"drp_qa": "w.2026.44"}, **configs})


def byImage(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(IMAGE).reset_index(drop=True)


def testRoundTrip(butler, fixtures, qadb):
    """What the loader wrote reads back as the Butler tables, camera split into arm and spectrograph."""
    load(butler, qadb)
    tables = readQadb(qadb, COLLECTION)
    iq, expected = byImage(tables["iqQaMetrics"]), byImage(fixtures["iqQaMetrics"])
    assert iq[IMAGE].equals(expected[IMAGE])
    assert iq["medFwhm"].to_numpy() == pytest.approx(expected["medFwhm"].to_numpy(), rel=1e-6)
    assert iq["qaStatus"].tolist() == expected["qaStatus"].tolist()
    # Negative control: arm and spectrograph come from the camera name the right way round.
    assert not (iq["arm"].astype(str) == iq["spectrograph"].astype(str)).any()
    assert len(tables["dmLineResidualSummary"]) == 4
    metrics = tables["dmLineResidualMetrics"]
    assert len(metrics) == len(fixtures["dmLineResidualMetrics"]) == 128
    assert set(metrics["arm"]) == {"b", "r"} and set(metrics["spectrograph"]) == {1}
    assert sorted(tables["visitTimes"]["pfs_visit_id"]) == VISITS


def testOtherCollectionsAndVisitsAreLeftOut(butler, qadb):
    load(butler, qadb)
    assert all(
        readQadb(qadb, "u/test/other")[name].empty for name in ("iqQaMetrics", "dmLineResidualMetrics")
    )
    iq = readQadb(qadb, COLLECTION, first=133030)["iqQaMetrics"]
    assert set(iq["visit"]) == {133040}


def testLatestRunWins(StubButler, fixtures, configs, qadb):
    """An image judged in two RUNs of the collection reads back once, as the later RUN judged it."""
    datasets = {"packages": {"drp_qa": "x"}, **configs}
    load(StubButler(fixtures, datasets=datasets), qadb)
    again = fixtures["iqQaMetrics"].assign(medFwhm=9.0)
    runs = {tuple(image): "u/test/qa/20261026T010000Z" for image in again[IMAGE].itertuples(index=False)}
    load(StubButler({**fixtures, "iqQaMetrics": again}, runs=runs, datasets=datasets), qadb, now=LATER)
    iq = readQadb(qadb, COLLECTION)["iqQaMetrics"]
    assert len(iq) == 4
    assert (iq["medFwhm"] == 9.0).all()


def testSinceReadsOnlyNewerLoads(butler, qadb):
    """``since`` is inclusive, and leaves out images not loaded again since."""
    load(butler, qadb)
    first = readQadb(qadb, COLLECTION)
    stamp = first["iqQaMetrics"]["decidedAt"].max()
    assert len(readQadb(qadb, COLLECTION, since=stamp)["iqQaMetrics"]) == 4
    load(butler, qadb, visits=[133040], now=LATER)
    stamp = readQadb(qadb, COLLECTION)["iqQaMetrics"]["decidedAt"].max()
    newer = readQadb(qadb, COLLECTION, since=stamp)
    assert set(newer["iqQaMetrics"]["visit"]) == {133040}
    assert set(newer["dmLineResidualMetrics"]["visit"]) == {133040}


def testMetricsTableHasEachTasksVerdict(butler, fixtures, qadb):
    load(butler, qadb)
    table = metricsTableFromQadb(readQadb(qadb, COLLECTION), taskThresholds()[0])
    verdicts = table[table["metric"] == VERDICT]
    assert len(verdicts) == 8  # four images, two tasks
    iq = byImage(verdicts[verdicts["source"] == "iqQaMetrics"])
    assert iq["status"].tolist() == byImage(fixtures["iqQaMetrics"])["qaStatus"].tolist()
    # Judged again with the shipped thresholds, for their limits: the quartz images have them.
    fwhm = table[(table["source"] == "iqQaMetrics") & (table["metric"] == "medFwhm")]
    traces = fwhm["visit"] == 133040
    assert np.isfinite(fwhm.loc[traces, "warn"]).all() and traces.sum() == 2


def testNothingReadIsAnEmptyTable(qadb):
    assert metricsTableFromQadb(readQadb(qadb, COLLECTION), taskThresholds()[0]).empty
