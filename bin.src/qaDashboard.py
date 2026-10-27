#!/usr/bin/env python3
r"""Serve the QA dashboard over a Tier 1 table, or live from qadb (PIPE2D-1924).

Stack-free; needs panel, holoviews, datashader and duckdb, and psycopg for qadb.
Over a file, build the table with ``bin.src/exportDashboardTier1.py`` first.

Examples::

    python bin.src/qaDashboard.py tier1-u_wtg_qa-thresholds_010.parquet --detail-dir detail --port 5006
    # A run as qaActor loads it: read-only, every 30 s (credentials in ~/.pgpass)
    python bin.src/qaDashboard.py --collection qaActor/reductions --first-visit 151000 \
        --qadb-host 127.0.0.1 --qadb-port 5436 --qadb-user public_user

then open http://localhost:5006. To serve the local network, add
``--address 0.0.0.0 --allow-origin <this host's name or IP>``. ``--detail-dir/<visit>/`` may hold the
visit's fetched PDFs (``extQaStats*.pdf``, ...) and ``dmLineResidualData*.parquet``.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the QA dashboard.")
    parser.add_argument("tier1", type=Path, nargs="?", help="Tier 1 parquet file (or --collection)")
    qadb = parser.add_argument_group("live from qadb, instead of a file")
    qadb.add_argument("--collection", help="the collection loaded into qadb, e.g. qaActor/reductions")
    qadb.add_argument("--first-visit", type=int, default=0, help="first visit shown")
    qadb.add_argument("--last-visit", type=int, default=2**31 - 1, help="last visit shown")
    qadb.add_argument("--qadb-host", default="127.0.0.1")
    qadb.add_argument("--qadb-port", type=int, default=5436)
    qadb.add_argument("--qadb-dbname", default="qadb")
    qadb.add_argument("--qadb-user", default="public_user", help="a read-only user (default public_user)")
    qadb.add_argument("--refresh", type=float, default=30.0, help="seconds between qadb reads (default 30)")
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
    if (args.tier1 is None) == (args.collection is None):
        parser.error("give a Tier 1 file or --collection, not both")

    # One line per request, with the client's address: shows whether a browser reached the server.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("tornado.access").setLevel(logging.INFO)
    logging.getLogger("bokeh").setLevel(logging.INFO)

    import panel as pn

    from pfs.drp.qa.dashboard.app import createApp

    if args.collection:
        from pfs.drp.qa.dashboard.live import QadbTier1Store
        from pfs.utils.database.qadb import QaDB

        db = QaDB(host=args.qadb_host, port=args.qadb_port, dbname=args.qadb_dbname, user=args.qadb_user)
        print(f"Reading {args.collection} from qadb at {args.qadb_host}:{args.qadb_port} ...", flush=True)
        # One store for every session: each refresh reads qadb once, whoever asks.
        source = QadbTier1Store.fromQadb(
            db, args.collection, args.first_visit, args.last_visit, minInterval=args.refresh / 2
        )
        print(f"{source.table['visit'].nunique()} visits so far", flush=True)
    else:
        source = args.tier1

    origins = [f"localhost:{args.port}", f"127.0.0.1:{args.port}"]
    origins += [origin if ":" in origin else f"{origin}:{args.port}" for origin in args.allow_origin]
    print(f"Serving on {args.address}:{args.port}, for {', '.join(origins)}", flush=True)
    pn.serve(
        lambda: createApp(source, args.detail_dir, args.start_visit, args.refresh),
        address=args.address,
        port=args.port,
        show=args.show,
        title="PFS QA dashboard",
        websocket_origin=origins,
    )


if __name__ == "__main__":
    main()
