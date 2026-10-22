"""The QA dashboard's serving layer and prototype app (PIPE2D-1924).

`serving` builds one long-format Tier 1 table for a collection, with its
provenance; `store` queries it with DuckDB. Neither imports the stack, Panel or
a Butler. `app` is the Panel prototype over a `store.Tier1Store`.
"""
