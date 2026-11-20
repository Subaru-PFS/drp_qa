"""Load the QA tasks' numbers from a Butler collection into qadb, and judge them.

The Butler is the record of what was measured: ``imageQualityQa``,
``dmLineResiduals`` and ``twilightFluxQa`` write one table per detector image
there. This package copies those numbers into qadb (``spt_qa_database``) after
the fact, measures a twilight sequence once it is complete, and gives every
image its verdict (`pfs.drp.qa.metrics.gate`), stored in ``qa_gate``. One code
path serves qaActor after each visit's pipetask and the backfill of whole
collections, and a Butler outage or a database outage cannot fail a quantum.

- `pfs.drp.qa.qadb.tables`: numbers, verdicts and normal values as qadb rows, no database needed.
- `pfs.drp.qa.qadb.writer`: idempotent writes (``INSERT ... ON CONFLICT DO UPDATE``)
  through a `pfs.utils.database.qadb.QaDB`.
- `pfs.drp.qa.qadb.loader`: read a collection from a Butler, write it and judge it.
- `pfs.drp.qa.qadb.reader`: read a collection's numbers, verdicts and normal values back.
- `pfs.drp.qa.qadb.judge`: judge stored numbers again, and make, compare and accept normal values.
"""
