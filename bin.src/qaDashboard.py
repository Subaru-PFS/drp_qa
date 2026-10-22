#!/usr/bin/env python3
r"""Serve the QA dashboard prototype over a Tier 1 table (PIPE2D-1924).

Stack-free; needs panel, holoviews, datashader and duckdb. Build the table with
``bin.src/exportDashboardTier1.py`` first.

Example::

    python bin.src/qaDashboard.py tier1-u_wtg_qa-thresholds_010.parquet --detail-dir detail --port 5006

then open http://localhost:5006. To serve the local network, add
``--address 0.0.0.0 --allow-origin <this host's name or IP>``. ``--detail-dir/<visit>/`` may hold the
visit's fetched PDFs (``extQaStats*.pdf``, ...) and ``dmLineResidualData*.parquet``.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the QA dashboard prototype.")
    parser.add_argument("tier1", type=Path, help="Tier 1 parquet file")
    parser.add_argument("--detail-dir", type=Path, help="per-visit detail files")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument(
        "--address", default="localhost", help="interface to listen on; 0.0.0.0 for the local network"
    )
    parser.add_argument(
        "--allow-origin",
        nargs="*",
        default=[],
        help="extra host[:port] names browsers may use for this server ('*' for any), with --address 0.0.0.0",
    )
    parser.add_argument(
        "--start-visit", type=int, help="visit the Live replay starts at (or the first after it)"
    )
    parser.add_argument("--show", action="store_true", help="open a browser")
    args = parser.parse_args(argv)

    # One line per request, with the client's address: shows whether a browser reached the server.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("tornado.access").setLevel(logging.INFO)
    logging.getLogger("bokeh").setLevel(logging.INFO)

    import panel as pn

    from pfs.drp.qa.dashboard.app import createApp

    origins = [f"localhost:{args.port}", f"127.0.0.1:{args.port}"]
    origins += [origin if ":" in origin else f"{origin}:{args.port}" for origin in args.allow_origin]
    print(f"Serving {args.tier1} on {args.address}:{args.port}, for {', '.join(origins)}", flush=True)
    pn.serve(
        lambda: createApp(args.tier1, args.detail_dir, args.start_visit),
        address=args.address,
        port=args.port,
        show=args.show,
        title="PFS QA dashboard",
        websocket_origin=origins,
    )


if __name__ == "__main__":
    main()
