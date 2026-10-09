"""Tests for `pfs.drp.qa.comparison.butlerQueries`, with a stub Butler."""

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
