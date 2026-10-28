"""The QA dashboard's serving layer and prototype app (PIPE2D-1924).

`qadbSource` reads a collection's QA results from qadb (or a snapshot of them);
`serving` builds them into one long-format metrics table, and `store` queries it
with DuckDB; `live` keeps it up to date as visits are loaded. None imports the
stack, Panel or a Butler. `app` is the Panel prototype over a `store.MetricsStore`.
"""
