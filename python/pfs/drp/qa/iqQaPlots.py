"""Backwards-compatible alias for `pfs.drp.qa.plotting.iqQa`.

The plotting code moved to `pfs.drp.qa.plotting` so that it takes DataFrames
and returns figures without importing the Butler or a task class; see
``doc/qa-rebuild-plan.md`` section 1.5. ``bin.src/plotIqQaTimeSeries.py`` and the
notebooks under ``examples/`` import from here, so this module re-exports the
public entry point. Prefer importing from ``pfs.drp.qa.plotting`` in new code.
"""

from pfs.drp.qa.plotting.iqQa import plotIqTimeSeries

__all__ = ["plotIqTimeSeries"]
