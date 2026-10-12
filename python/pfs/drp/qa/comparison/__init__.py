"""Comparison mode: gate every image of an observing run against the reference thresholds.

The visits come from the opdb, the reductions and verdicts from a Butler. ``queries`` is the
only module that touches the opdb and ``butlerQueries`` the only one that touches a Butler;
both take what they read from as an argument, so everything else here is stack-free and tested
in CI. ``bin.src/qaComparison.py`` drives it.
"""
