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


def testEarlierRunsAreTheChainsOwnRuns():
    """A new comparison takes the earlier chains' own runs, never a chain, and never its own runs back."""
    from pfs.drp.qa.comparison.butlerQueries import earlierRuns

    def info(name, kind):
        return SimpleNamespace(name=name, type=SimpleNamespace(name=kind))

    infos = [
        info("u/me/comparison/run31/v259", "CHAINED"),
        info("u/me/comparison/run31/v259/20261031T200143Z", "RUN"),
        info("u/me/comparison/run31/v271", "CHAINED"),  # the newer version chained v259's runs as inputs
        info("u/me/comparison/run31/v271/20261102T175232Z", "RUN"),
        info("u/me/comparison/run31/v271/20261102T182929Z", "RUN"),
    ]
    butler = SimpleNamespace(
        collections=SimpleNamespace(
            query=lambda pattern: [i.name for i in infos if fnmatch(i.name, pattern)],
            query_info=lambda pattern: [i for i in infos if fnmatch(i.name, pattern)],
        )
    )
    # Writing to v259 again: v271's own runs are the inputs, not the v271 chain (which holds v259's runs).
    assert earlierRuns(butler, "u/me/comparison", "run31", "", "u/me/comparison/run31/v259") == [
        "u/me/comparison/run31/v271/20261102T182929Z",
        "u/me/comparison/run31/v271/20261102T175232Z",
    ]
    # Negative control: the chain itself is never among the inputs.
    assert "u/me/comparison/run31/v271" not in earlierRuns(
        butler, "u/me/comparison", "run31", "", "u/me/comparison/run31/v259"
    )
    assert earlierRuns(butler, "u/me/comparison", "run31", "", "u/me/comparison/run31/v271") == [
        "u/me/comparison/run31/v259/20261031T200143Z"
    ]


def testRebaseOutputChain():
    """The chain becomes its own runs, then the inputs flattened, without repeating its own runs."""
    from pfs.drp.qa.comparison.butlerQueries import rebaseOutputChain

    def info(name, kind):
        return SimpleNamespace(name=name, type=SimpleNamespace(name=kind))

    infos = [
        info("u/me/comparison/run31/v1/20261031T200143Z", "RUN"),
        info("u/me/comparison/run31/v1/20261031T224529Z", "RUN"),
    ]
    flat = {
        "drpActor/reductions": [
            "drpActor/reductions/20261104T000000Z",
            "drpActor/reductions/20261030T000000Z",
        ],
        "PFS/defaults": ["PFS/raw/sps", "PFS/calib/x"],
        "u/me/comparison/run31/v1/20261031T200143Z": ["u/me/comparison/run31/v1/20261031T200143Z"],
    }
    redefined = {}
    butler = SimpleNamespace(
        collections=SimpleNamespace(
            query_info=lambda pattern: [i for i in infos if fnmatch(i.name, pattern)],
            query=lambda names, flatten_chains=False: [c for n in names for c in flat[n]],
            redefine_chain=lambda parent, children: redefined.setdefault(parent, list(children)),
        )
    )
    children = rebaseOutputChain(
        butler,
        "u/me/comparison/run31/v1",
        ["u/me/comparison/run31/v1/20261031T200143Z", "drpActor/reductions", "PFS/defaults"],
    )
    assert children == [
        "u/me/comparison/run31/v1/20261031T224529Z",
        "u/me/comparison/run31/v1/20261031T200143Z",
        "drpActor/reductions/20261104T000000Z",
        "drpActor/reductions/20261030T000000Z",
        "PFS/raw/sps",
        "PFS/calib/x",
    ]
    assert redefined == {"u/me/comparison/run31/v1": children}
    # Negative control: a chain among the inputs is never a child itself.
    assert "drpActor/reductions" not in children and "PFS/defaults" not in children
