"""Redo a run: reduce every image of an observing run with an updated pipeline and run the QA tasks.

The visits come from the opdb, the reductions and QA results from a Butler. Nothing here gives a
verdict: an image is judged when its numbers are loaded into qadb (`pfs.drp.qa.metrics.gate`). ``queries`` is the
only module that touches the opdb and ``butlerQueries`` the only one that touches a Butler;
both take what they read from as an argument, so everything else here is stack-free and tested
in CI. ``redo`` (``drpQa.py redo``) drives it, and ``calibInputsCli``/``calibValidateCli`` reuse its
classification and command-building.
"""
