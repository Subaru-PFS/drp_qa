"""Tests for the long-format per-species schema.

The point of long format is that concatenating across quanta yields a stable
schema with no NaN padding, so that is what these tests assert.
"""

import pandas as pd
import pytest

from pfs.drp.qa.metrics.longFormat import (
    LONG_COLUMNS,
    longRecords,
    toLongFrame,
    widen,
)
from pfs.drp.qa.metrics.registry import MetricDef, MetricRegistry, Thresholds


@pytest.fixture
def registry():
    return MetricRegistry(
        [
            MetricDef(
                name="fitSpeciesXRms",
                units="pixels",
                reference="fitDetectorMap solution",
                thresholds=Thresholds(warn=0.05, fail=0.08),
            ),
            MetricDef(
                name="fitSpeciesYRms",
                units="pixels",
                reference="fitDetectorMap solution",
            ),
        ]
    )


@pytest.fixture
def dataId():
    return {"visit": 12345, "arm": "b", "spectrograph": 1}


class TestLongRecords:
    def testOneRowPerSpeciesAndMetric(self, dataId):
        records = longRecords(
            dataId,
            {"HgI": {"fitSpeciesXRms": 0.021}, "ArI": {"fitSpeciesXRms": 0.088}},
        )
        assert len(records) == 2
        assert all(tuple(record) == LONG_COLUMNS for record in records)
        assert [record["description"] for record in records] == ["ArI", "HgI"], "sorted for reproducibility"

    def testGatingPopulatesStatus(self, dataId, registry):
        records = longRecords(
            dataId,
            {"HgI": {"fitSpeciesXRms": 0.021}, "ArI": {"fitSpeciesXRms": 0.088}},
            registry=registry,
        )
        bySpecies = {record["description"]: record["status"] for record in records}
        assert bySpecies == {"HgI": "PASS", "ArI": "FAIL"}

    def testUngatedMetricHasNoStatus(self, dataId, registry):
        (record,) = longRecords(dataId, {"HgI": {"fitSpeciesYRms": 99.0}}, registry=registry)
        assert record["status"] is None

    def testMetricAbsentFromRegistryHasNoStatus(self, dataId, registry):
        (record,) = longRecords(dataId, {"HgI": {"somethingElse": 1.0}}, registry=registry)
        assert record["status"] is None

    def testNoRegistryMeansNoStatus(self, dataId):
        (record,) = longRecords(dataId, {"HgI": {"fitSpeciesXRms": 99.0}})
        assert record["status"] is None

    def testSpeciesOverrideKeyIsOffered(self, dataId):
        """Per-species thresholds are keyed ``arm:description`` (R7)."""
        registry = MetricRegistry(
            [
                MetricDef(
                    name="fitSpeciesXRms",
                    units="pixels",
                    reference="fitDetectorMap solution",
                    thresholds=Thresholds(warn=0.05, fail=0.08),
                    overrides={"b:ArI": Thresholds(warn=0.5, fail=0.9)},
                )
            ]
        )
        records = longRecords(
            dataId,
            {"ArI": {"fitSpeciesXRms": 0.09}, "HgI": {"fitSpeciesXRms": 0.09}},
            registry=registry,
        )
        bySpecies = {record["description"]: record["status"] for record in records}
        assert bySpecies == {"ArI": "PASS", "HgI": "FAIL"}

    def testPartialDataIdStillYieldsRows(self):
        (record,) = longRecords({"visit": 1}, {"HgI": {"fitSpeciesXRms": 0.01}})
        assert record["arm"] is None
        assert record["spectrograph"] is None

    def testNoSpeciesYieldsNoRows(self, dataId):
        assert longRecords(dataId, {}) == []


class TestToLongFrame:
    def testSchemaIsFixed(self, dataId):
        frame = toLongFrame(longRecords(dataId, {"HgI": {"fitSpeciesXRms": 0.02}}))
        assert tuple(frame.columns) == LONG_COLUMNS
        assert frame["value"].dtype == "float64"
        assert frame["visit"].dtype == "Int64"

    def testEmptyFrameKeepsTheSchema(self):
        frame = toLongFrame([])
        assert tuple(frame.columns) == LONG_COLUMNS
        assert frame.empty
        assert frame["value"].dtype == "float64"

    def testConcatenationAcrossQuantaIsRaggedFree(self):
        """The failure this format exists to prevent: a ragged, NaN-padded frame."""
        first = toLongFrame(
            longRecords(
                {"visit": 1, "arm": "b", "spectrograph": 1},
                {"HgI": {"fitSpeciesXRms": 0.02}, "CdI": {"fitSpeciesXRms": 0.03}},
            )
        )
        # A different visit that saw an entirely different species mix.
        second = toLongFrame(
            longRecords(
                {"visit": 2, "arm": "r", "spectrograph": 3},
                {"ArI": {"fitSpeciesXRms": 0.09}, "XeI": {"fitSpeciesXRms": 0.07}},
            )
        )
        combined = pd.concat([first, second], ignore_index=True)

        assert tuple(combined.columns) == LONG_COLUMNS
        assert not combined["value"].isna().any(), "no NaN padding from the species mix"
        assert len(combined) == 4
        assert set(combined.groupby("description").groups) == {"HgI", "CdI", "ArI", "XeI"}

    def testEmptyQuantumConcatenatesCleanly(self):
        """A quantum that measured nothing must not change the combined schema."""
        populated = toLongFrame(
            longRecords({"visit": 1, "arm": "b", "spectrograph": 1}, {"HgI": {"fitSpeciesXRms": 0.02}})
        )
        combined = pd.concat([toLongFrame([]), populated], ignore_index=True)
        assert tuple(combined.columns) == LONG_COLUMNS
        assert combined["value"].dtype == "float64"
        assert len(combined) == 1


class TestWiden:
    def testPivotsToOneColumnPerMetric(self, dataId):
        frame = toLongFrame(
            longRecords(
                dataId,
                {
                    "HgI": {"fitSpeciesXRms": 0.02, "fitSpeciesYRms": 0.05},
                    "ArI": {"fitSpeciesXRms": 0.09, "fitSpeciesYRms": 0.11},
                },
            )
        )
        wide = widen(frame)
        assert len(wide) == 2
        assert {"fitSpeciesXRms", "fitSpeciesYRms"} <= set(wide.columns)
        hgi = wide.query("description == 'HgI'").iloc[0]
        assert hgi["fitSpeciesXRms"] == pytest.approx(0.02)

    def testEmptyInputGivesEmptyOutput(self):
        wide = widen(toLongFrame([]))
        assert wide.empty
        assert "description" in wide.columns
