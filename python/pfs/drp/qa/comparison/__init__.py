"""Comparison mode: reduce every image of an observing run and run the QA tasks on it.

The visits come from the opdb, the reductions and QA results from a Butler. Nothing here gives a
verdict: an image is judged when its numbers are loaded into qadb (`pfs.drp.qa.metrics.gate`). ``queries`` is the
only module that touches the opdb and ``butlerQueries`` the only one that touches a Butler;
both take what they read from as an argument, so everything else here is stack-free and tested
in CI. ``bin.src/qaComparison.py`` drives it.
"""
