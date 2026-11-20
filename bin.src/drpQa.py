#!/usr/bin/env python3
"""``drpQa.py``: entry point for `pfs.drp.qa.cli.main`. ``drpQa.py --help`` lists the subcommands."""

import sys

from pfs.drp.qa.cli import main

if __name__ == "__main__":
    sys.exit(main())
