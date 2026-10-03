"""Autoguider (AG) analysis tools, moved from ``pfs.drp.stella.utils.guiders``.

The package has four layers:

- `coordinates`: frame, sign and unit conventions.
- `queries`: the only code that touches the opdb or a butler.
- `analysis`: fits and statistics.
- `plotting`: figures.

`analysis` and `plotting` take DataFrames, never a database connection.

`compat` keeps the names of drp_stella's AG readers, as deprecated wrappers
of `queries.readAgcData`, while notebooks move to it.
"""
