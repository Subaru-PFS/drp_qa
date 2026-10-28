#!/usr/bin/env python3
r"""Serve the QA dashboard live from qadb, or over a snapshot of it (PIPE2D-1924).

Stack-free; needs panel, holoviews, datashader and duckdb, and psycopg for qadb.

Examples::

    # A run as qaActor loads it, with the opdb's sequences and notes: read-only, every 30 s
    # (credentials in ~/.pgpass)
    python bin.src/qaDashboard.py --collection qaActor/reductions --first-visit 151000 \
        --qadb-host 127.0.0.1 --qadb-port 5436 --opdb-host 127.0.0.1 --opdb-port 5432
    # Read it once into a snapshot (parquet and provenance.json), then serve that offline
    python bin.src/qaDashboard.py --collection qaActor/reductions --opdb-host 127.0.0.1 --write-snapshot run31-qadb
    python bin.src/qaDashboard.py --snapshot run31-qadb --detail-dir detail

then open http://localhost:5006. To serve the local network, add
``--address 0.0.0.0 --allow-origin <this host's name or IP>``. ``--detail-dir/<visit>/`` may hold the
visit's fetched PDFs (``extQaStats*.pdf``, ...) and ``dmLineResidualData*.parquet``.
"""

import argparse
import logging
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the QA dashboard.")
    parser.add_argument("--snapshot", type=Path, help="serve a snapshot written by --write-snapshot")
    qadb = parser.add_argument_group("from qadb")
    qadb.add_argument("--collection", help="the collection loaded into qadb, e.g. qaActor/reductions")
    qadb.add_argument(
        "--write-snapshot", type=Path, metavar="DIR", help="read the collection once into DIR and exit"
    )
    qadb.add_argument("--first-visit", type=int, default=0, help="first visit shown")
    qadb.add_argument("--last-visit", type=int, default=2**31 - 1, help="last visit shown")
    qadb.add_argument("--qadb-host", default="127.0.0.1")
    qadb.add_argument("--qadb-port", type=int, default=5436)
    qadb.add_argument("--qadb-dbname", default="qadb")
    qadb.add_argument("--qadb-user", default="public_user", help="a read-only user (default public_user)")
    qadb.add_argument("--refresh", type=float, default=30.0, help="seconds between qadb reads (default 30)")
    opdb = parser.add_argument_group("the opdb's IIC sequences, visit times and notes (default: none)")
    opdb.add_argument("--opdb-host")
    opdb.add_argument("--opdb-port", type=int, default=5432)
    opdb.add_argument("--opdb-dbname", default="opdb")
    opdb.add_argument("--opdb-user", default="public_user", help="a read-only user (default public_user)")
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
    if (args.snapshot is None) == (args.collection is None):
        parser.error("give --snapshot or --collection, not both")

    # One line per request, with the client's address: shows whether a browser reached the server.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("tornado.access").setLevel(logging.INFO)
    logging.getLogger("bokeh").setLevel(logging.INFO)

    from pfs.drp.qa.dashboard.live import QadbMetricsStore

    if args.collection:
        from pfs.utils.database.qadb import QaDB

        db = QaDB(host=args.qadb_host, port=args.qadb_port, dbname=args.qadb_dbname, user=args.qadb_user)
        opdb = None
        if args.opdb_host:
            from pfs.utils.database.opdb import OpDB

            opdb = OpDB(
                host=args.opdb_host, port=args.opdb_port, dbname=args.opdb_dbname, user=args.opdb_user
            )
        print(
            f"Reading {args.collection} from qadb at {args.qadb_host}:{args.qadb_port}"
            + (f", with the opdb at {args.opdb_host}:{args.opdb_port}" if opdb else ", without the opdb")
            + " ...",
            flush=True,
        )
        if args.write_snapshot:
            writeSnapshot(db, opdb, args)
            return
        # One store for every session: each refresh reads qadb (and the opdb) once, whoever asks.
        store = QadbMetricsStore.fromQadb(
            db, args.collection, args.first_visit, args.last_visit, minInterval=args.refresh / 2, opdb=opdb
        )
        print(f"{store.table['visit'].nunique()} visits so far", flush=True)
    else:
        store = QadbMetricsStore.fromSnapshot(args.snapshot)
        print(f"{store.table['visit'].nunique()} visits in {args.snapshot}", flush=True)

    import panel as pn

    from pfs.drp.qa.dashboard.app import createApp

    origins = [f"localhost:{args.port}", f"127.0.0.1:{args.port}"]
    origins += [origin if ":" in origin else f"{origin}:{args.port}" for origin in args.allow_origin]
    print(f"Serving on {args.address}:{args.port}, for {', '.join(origins)}", flush=True)
    pn.serve(
        lambda: createApp(store, args.detail_dir, args.start_visit, args.refresh),
        address=args.address,
        port=args.port,
        show=args.show,
        title="PFS QA dashboard",
        websocket_origin=origins,
    )


def writeSnapshot(db, opdb, args: argparse.Namespace) -> None:
    """Read the collection from qadb (and the opdb) once and write it, with its provenance, to ``--write-snapshot``."""
    from pfs.drp.qa.dashboard.live import HST, opdbReader, qadbProvenance
    from pfs.drp.qa.dashboard.qadbSource import readQadb
    from pfs.drp.qa.dashboard.qadbSource import writeSnapshot as write

    start = time.monotonic()
    tables = readQadb(db, args.collection, args.first_visit, args.last_visit)
    if opdb is not None:
        visits = sorted({int(visit) for visit in tables["iqQaMetrics"]["visit"]})
        tables["opdbVisitTimes"], tables["notes"] = opdbReader(opdb)(visits)
    provenance = qadbProvenance(
        f"{args.collection} (qadb snapshot)",
        builtAt=datetime.now(HST).isoformat(timespec="seconds"),
        qadb=f"{args.qadb_host}:{args.qadb_port}/{args.qadb_dbname}",
        opdb=f"{args.opdb_host}:{args.opdb_port}/{args.opdb_dbname}" if opdb is not None else None,
        visits=[args.first_visit, args.last_visit],
    )
    out = write(tables, args.write_snapshot, provenance)
    size = sum(path.stat().st_size for path in out.iterdir()) / 1e6
    rows = ", ".join(f"{name} {len(frame)}" for name, frame in tables.items())
    print(f"{rows} rows -> {out} ({size:.2f} MB) in {time.monotonic() - start:.1f} s", flush=True)


if __name__ == "__main__":
    main()
