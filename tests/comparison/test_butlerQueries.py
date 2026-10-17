"""Tests for `pfs.drp.qa.comparison.butlerQueries`, with a stub Butler."""

from fnmatch import fnmatch
from types import SimpleNamespace

import pytest

from pfs.drp.qa.comparison.butlerQueries import collectionExists


class MissingCollectionError(Exception):
    """Stands in for ``lsst.daf.butler.MissingCollectionError``."""


def _butler(query):
    return SimpleNamespace(collections=SimpleNamespace(query=query))


def _missing(name):
    raise MissingCollectionError(f"No collection with name '{name}' found.")


def testCollectionExists():
    assert collectionExists(_butler(lambda name: [name]), "u/me/comparison/run25/v1")
    assert not collectionExists(_butler(lambda name: []), "u/me/comparison/run25/v1")
    # A missing explicit name raises in daf_butler rather than returning nothing.
    assert not collectionExists(_butler(_missing), "u/me/comparison/run25/v1")


def testOtherErrorsPropagate():
    def broken(name):
        raise RuntimeError("database gone")

    with pytest.raises(RuntimeError, match="database gone"):
        collectionExists(_butler(broken), "u/me/comparison/run25/v1")


def testNoCollectionsFindNothing():
    from pfs.drp.qa.comparison.butlerQueries import datasetDetectors

    def query(*args, **kwargs):
        raise AssertionError("a query with no collections searches the whole repository")

    butler = SimpleNamespace(query_datasets=query)
    assert datasetDetectors(butler, "calexp", [1, 2], []).empty  # --fresh: no reductions to reuse


def testEarlierReductions():
    from pfs.drp.qa.comparison.butlerQueries import earlierReductions

    names = [
        "u/me/comparison/run30/v1",  # drpActor's reductions
        "u/me/comparison/run30/v1/pfs-w.2026.40",
        "u/me/comparison/run30/v1/pfs-w.2026.40/20261009T120000Z",  # a run inside the chain
        "u/me/comparison/run30/v2/pfs-w.2026.40",  # the current output
        "u/me/comparison/run30/v2/pfs-w.2026.44",  # another pipeline
    ]
    butler = SimpleNamespace(collections=SimpleNamespace(query=lambda pattern: names))
    found = earlierReductions(
        butler,
        "u/me/comparison",
        "run30",
        "pfs-w.2026.40",
        "u/me/comparison/run30/v2/pfs-w.2026.40",
    )
    assert found == ["u/me/comparison/run30/v1/pfs-w.2026.40"]

    # The default mode: earlier versions' collections beside drpActor's, not fresh ones or runs.
    names.append("u/me/comparison/run30/v2")
    found = earlierReductions(butler, "u/me/comparison", "run30", "", "u/me/comparison/run30/v2")
    assert found == ["u/me/comparison/run30/v1"]


def testOwnRuns():
    """Only the RUN collections pipetask wrote directly under the output."""
    from pfs.drp.qa.comparison.butlerQueries import ownRuns

    def info(name, kind):
        return SimpleNamespace(name=name, type=SimpleNamespace(name=kind))

    infos = [
        info("u/me/comparison/run30/v2/20261013T105556Z", "RUN"),
        info("u/me/comparison/run30/v2/20261013T120326Z", "RUN"),
        info("u/me/comparison/run30/v2/pfs-w.2026.40", "CHAINED"),  # a --fresh variant
        info("u/me/comparison/run30/v2/pfs-w.2026.40/20261014T000000Z", "RUN"),  # its run
        info("u/me/comparison/run30/v20/20261013T000000Z", "RUN"),  # another version
    ]
    butler = SimpleNamespace(
        collections=SimpleNamespace(query_info=lambda pattern: [i for i in infos if fnmatch(i.name, pattern)])
    )
    assert ownRuns(butler, "u/me/comparison/run30/v2") == [
        "u/me/comparison/run30/v2/20261013T120326Z",
        "u/me/comparison/run30/v2/20261013T105556Z",
    ]
    assert ownRuns(butler, "u/me/comparison/run30/v3") == []


def testFreshPipelines():
    """The pipeline labels of a period's fresh collections: chains two levels under the period."""
    from pfs.drp.qa.comparison.butlerQueries import freshPipelines

    def info(name, kind):
        return SimpleNamespace(name=name, type=SimpleNamespace(name=kind))

    infos = [
        info("u/me/comparison/run25/v1/pfs-w.2026.39", "CHAINED"),
        info("u/me/comparison/run25/v1/pfs-w.2026.39/20261017T000000Z", "RUN"),
        info("u/me/comparison/run25/v1/datamodel-w.2026.40+drp_stella-w.2026.41", "CHAINED"),
        info("u/me/comparison/run25/v2/datamodel-w.2026.40+drp_stella-w.2026.41", "CHAINED"),
        info("u/me/comparison/run25/v1", "CHAINED"),  # default mode: one level
        info("u/me/comparison/run25/v1/20261013T000000Z", "RUN"),  # its run, not a label
        info("u/me/comparison/run26/v1/pfs-w.2026.39", "CHAINED"),  # another period
    ]
    butler = SimpleNamespace(
        collections=SimpleNamespace(query_info=lambda pattern: [i for i in infos if fnmatch(i.name, pattern)])
    )
    assert freshPipelines(butler, "u/me/comparison", "run25") == {
        "datamodel-w.2026.40+drp_stella-w.2026.41": [
            "u/me/comparison/run25/v2/datamodel-w.2026.40+drp_stella-w.2026.41",
            "u/me/comparison/run25/v1/datamodel-w.2026.40+drp_stella-w.2026.41",
        ],
        "pfs-w.2026.39": ["u/me/comparison/run25/v1/pfs-w.2026.39"],
    }
    assert freshPipelines(butler, "u/me/comparison", "run27") == {}
