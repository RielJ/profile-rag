"""PostgreSQL connection pool.

Opening a database connection takes a network round trip and authentication.
A pool keeps a few connections open and lends them out, so each request
borrows one instead of connecting from scratch.

Usage:

    with pool.connection() as conn:
        conn.execute("SELECT 1")

The connection returns to the pool when the `with` block ends. If the block
raised an exception, the transaction is rolled back first; otherwise it is
committed.
"""

from psycopg import Connection
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool

# The pool type used across the project: connections that return rows as
# tuples (psycopg's default).
Pool = ConnectionPool[Connection[TupleRow]]


def create_pool(conninfo: str, *, max_size: int = 5, open_now: bool = True) -> Pool:
    """Create a pool and, by default, start opening connections in the background.

    `wait=False` lets the application start even when the database is down.
    Requests that need the database then fail individually, and the health
    endpoint reports the problem, instead of the whole server refusing to
    start.

    `open_now=False` returns a closed pool, which tests use to simulate a pool
    that has been shut down.
    """
    pool: Pool = ConnectionPool(
        conninfo,
        connection_class=Connection[TupleRow],
        min_size=1,
        max_size=max_size,
        open=False,
        name="profile-rag",
        # Fail a single connection attempt quickly when the server is down.
        kwargs={"connect_timeout": 2},
    )
    if open_now:
        pool.open(wait=False)
    return pool
