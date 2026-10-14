"""Tests for reading calibrations from Butler provenance, with a fake Butler."""

from types import SimpleNamespace

import pytest

from pfs.drp.qa.utils.provenance import CALIBRATION_COLUMNS, calibrationsUsed, provenanceInputs

FITS_HEADER = {
    "LSST BUTLER ID": "a0",
    "LSST BUTLER RUN": "u/someone/run",
    "LSST BUTLER DATASETTYPE": "calexp",
    "LSST BUTLER N_INPUTS": 3,
    "LSST BUTLER INPUT 0 ID": "b0",
    "LSST BUTLER INPUT 0 RUN": "u/someone/run",
    "LSST BUTLER INPUT 0 DATASETTYPE": "postISRCCD",
    "LSST BUTLER INPUT 2 ID": "d2",
    "LSST BUTLER INPUT 2 RUN": "PFS/calib/detectorMap/1",
    "LSST BUTLER INPUT 2 DATASETTYPE": "detectorMap_calib",
    "LSST BUTLER INPUT 1 ID": "c1",
    "LSST BUTLER INPUT 1 RUN": "PFS/calib/fiberProfiles/7",
    "LSST BUTLER INPUT 1 DATASETTYPE": "fiberProfiles",
    "EXPTIME": 30.0,
}


class TestProvenanceInputs:
    def testReadsFitsStyleKeysInInputOrder(self):
        table = provenanceInputs(FITS_HEADER)
        assert table["datasetType"].tolist() == ["postISRCCD", "fiberProfiles", "detectorMap_calib"]
        assert table.loc[1, "run"] == "PFS/calib/fiberProfiles/7"
        assert table.loc[1, "id"] == "c1"

    def testReadsDottedLowerCaseKeys(self):
        header = {
            "lsst.butler.input.0.id": "x",
            "lsst.butler.input.0.run": "r",
            "lsst.butler.input.0.datasettype": "t",
        }
        assert provenanceInputs(header).to_dict("records") == [{"datasetType": "t", "run": "r", "id": "x"}]

    def testIgnoresKeysWithoutTheButlerPrefix(self):
        """An unrelated ``... INPUT 0 ID`` key is not provenance."""
        assert provenanceInputs({"OTHER INPUT 0 ID": "x", "EXPTIME": 1.0}).empty

    def testNoProvenanceGivesAnEmptyTable(self):
        table = provenanceInputs({})
        assert table.empty
        assert list(table.columns) == ["datasetType", "run", "id"]


class FakeButler:
    """Holds dataset metadata, which dataset types are calibrations, and a lookup."""

    CALIBRATIONS = frozenset({"fiberProfiles", "detectorMap_calib", "bias"})

    def __init__(self, metadata, lookup=None):
        self.metadata = metadata
        self.lookup = lookup or {}

    def get(self, name, dataId):
        datasetType = name.removesuffix(".metadata")
        if datasetType not in self.metadata:
            raise LookupError(name)
        return SimpleNamespace(toDict=lambda: self.metadata[datasetType])

    def get_dataset_type(self, name):
        return SimpleNamespace(isCalibration=lambda: name in self.CALIBRATIONS)

    def find_dataset(self, name, dataId):
        return self.lookup.get(name)


DATA_ID = {"instrument": "PFS", "visit": 1, "arm": "r", "spectrograph": 1}


class TestCalibrationsUsed:
    def testKeepsOnlyCalibrationsFromProvenance(self):
        table = calibrationsUsed(FakeButler({"calexp": FITS_HEADER}), DATA_ID)
        assert list(table.columns) == list(CALIBRATION_COLUMNS)
        assert table["datasetType"].tolist() == ["fiberProfiles", "detectorMap_calib"]
        assert set(table["source"]) == {"provenance"}
        assert set(table["via"]) == {"calexp"}

    def testProvenanceWinsOverALookupThatDisagrees(self):
        """The reduction's calib, not the one a lookup finds today."""
        lookup = {"fiberProfiles": SimpleNamespace(run="PFS/calib/fiberProfiles/9", id="z9")}
        table = calibrationsUsed(FakeButler({"calexp": FITS_HEADER}, lookup), DATA_ID)
        assert table.set_index("datasetType").loc["fiberProfiles", "id"] == "c1"

    def testMissingDatasetsAreSkippedAndInputsMerged(self):
        postIsr = {
            "LSST BUTLER INPUT 0 ID": "e0",
            "LSST BUTLER INPUT 0 RUN": "PFS/calib/bias/1",
            "LSST BUTLER INPUT 0 DATASETTYPE": "bias",
        }
        butler = FakeButler({"calexp": FITS_HEADER, "postISRCCD": postIsr})
        table = calibrationsUsed(butler, DATA_ID, datasetTypes=("calexp", "postISRCCD", "missing"))
        assert table["datasetType"].tolist() == ["fiberProfiles", "detectorMap_calib", "bias"]
        assert table["via"].tolist() == ["calexp", "calexp", "postISRCCD"]

    def testFallsBackToALookupMarkedAsSuch(self):
        lookup = {"fiberProfiles": SimpleNamespace(run="PFS/calib/fiberProfiles/9", id="z9")}
        table = calibrationsUsed(FakeButler({"calexp": {"EXPTIME": 1.0}}, lookup), DATA_ID)
        assert table.to_dict("records") == [
            {
                "datasetType": "fiberProfiles",
                "run": "PFS/calib/fiberProfiles/9",
                "id": "z9",
                "source": "lookup",
                "via": "",
            }
        ]

    @pytest.mark.parametrize("metadata", [{}, {"calexp": {}}])
    def testNothingFoundGivesAnEmptyTable(self, metadata):
        table = calibrationsUsed(FakeButler(metadata), DATA_ID)
        assert table.empty
        assert list(table.columns) == list(CALIBRATION_COLUMNS)

    def testLooksUpOnlyTheCalibrationsProvenanceDoesNotName(self):
        """Provenance from postISRCCD alone names no fiberProfiles, so that one is looked up."""
        postIsr = {
            "LSST BUTLER INPUT 0 ID": "e0",
            "LSST BUTLER INPUT 0 RUN": "PFS/calib/bias/1",
            "LSST BUTLER INPUT 0 DATASETTYPE": "bias",
        }
        lookup = {"fiberProfiles": SimpleNamespace(run="PFS/calib/fiberProfiles/9", id="z9")}
        butler = FakeButler({"postISRCCD": postIsr}, lookup)
        table = calibrationsUsed(
            butler, DATA_ID, datasetTypes=("calexp", "postISRCCD"), fallback=("fiberProfiles",)
        )
        assert table[["datasetType", "source"]].values.tolist() == [
            ["bias", "provenance"],
            ["fiberProfiles", "lookup"],
        ]
