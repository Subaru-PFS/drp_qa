"""Load the QA tasks' tables and verdicts from a Butler collection into qadb (PIPE2D-1385).

The Butler is the record: ``imageQualityQa`` and ``dmLineResiduals`` write one
table per detector image there, and this package copies those tables, with the
thresholds they were judged by and a per-image gate verdict, into qadb
(``spt_qa_database``) after the fact. One code path serves qaActor after each
visit's pipetask, comparison mode after a run, and the backfill of whole
collections, and a Butler outage or a database outage cannot fail a quantum.

- `pfs.drp.qa.qadb.tables`: the Butler tables as qadb rows, no database needed.
- `pfs.drp.qa.qadb.writer`: idempotent writes (``INSERT ... ON CONFLICT DO UPDATE``)
  through a `pfs.utils.database.qadb.QaDB`.
- `pfs.drp.qa.qadb.loader`: read a collection from a Butler and write it.
"""

from .loader import *  # noqa: F403
from .tables import *  # noqa: F403
from .writer import *  # noqa: F403
