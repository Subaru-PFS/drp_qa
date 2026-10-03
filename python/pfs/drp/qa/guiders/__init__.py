"""Autoguider (AG) analysis tools, moved from ``pfs.drp.stella.utils.guiders``.

The package has four layers:

- `coordinates`: frame, sign and unit conventions.
- `queries`: the only code that touches the opdb or a butler.
- `analysis`: fits and statistics.
- `plotting`: figures.

`analysis` and `plotting` take DataFrames, never a database connection.
"""
