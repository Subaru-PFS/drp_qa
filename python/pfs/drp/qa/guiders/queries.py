"""Readers for AG data from the opdb and the butler.

This is the only module that touches a database or a butler. Each reader takes
a `pfs.utils.database.opdb.OpDB` as its first argument and uses bound
parameters.
"""
