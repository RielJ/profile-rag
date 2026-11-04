"""Numbered SQL migrations.

A migration is a SQL file in `db/migrations/` named `NNNN_description.sql`,
for example `0001_initial.sql`. The runner applies the files that have not
run yet, in version order, and records each one in the `schema_migrations`
table so it never runs twice.

Rules the runner enforces:

- Each file runs inside its own transaction. If any statement fails, the
  whole file is rolled back and is not recorded, so the database is never
  left half-migrated.
- A file that has been applied must not change. The runner stores each
  file's SHA-256 checksum and refuses to continue if an applied file was
  edited. To change the schema, add a new migration instead.
- Only one runner works at a time. A Postgres advisory lock makes a second
  runner wait instead of applying the same file twice.

Run it with `make migrate`, which calls `python -m profile_rag.db.migrate`.
"""

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.rows import TupleRow

from profile_rag.settings import REPO_ROOT, get_settings

MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"

# `0001_initial.sql`: four digits, an underscore, then lowercase words.
FILENAME_PATTERN = re.compile(r"^(?P<version>\d{4})_(?P<name>[a-z0-9_]+)\.sql$")

# Any fixed 64-bit number works; it only has to be the same for every runner.
ADVISORY_LOCK_KEY = 7_242_001


class MigrationError(Exception):
    """The migration files or the recorded history are inconsistent."""


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    path: Path
    sql: bytes

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql).hexdigest()


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Return the migration files in `directory`, sorted by version.

    Raises MigrationError for a `.sql` file with a malformed name or for two
    files that share a version number.
    """
    migrations: dict[str, Migration] = {}
    for path in sorted(directory.glob("*.sql")):
        match = FILENAME_PATTERN.match(path.name)
        if match is None:
            raise MigrationError(f"{path.name}: migration files must be named NNNN_description.sql")
        version = match["version"]
        if version in migrations:
            raise MigrationError(
                f"{path.name}: version {version} is also used by {migrations[version].path.name}"
            )
        migrations[version] = Migration(
            version=version, name=match["name"], path=path, sql=path.read_bytes()
        )
    return [migrations[v] for v in sorted(migrations)]


def applied_checksums(conn: psycopg.Connection[TupleRow]) -> dict[str, str]:
    """Return {version: checksum} for every migration already applied."""
    rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    return {str(version): str(checksum) for version, checksum in rows}


def migrate(conninfo: str, directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Apply every pending migration in `directory`. Return the ones applied.

    Running it again when nothing is pending applies nothing and returns [].
    """
    migrations = discover(directory)
    applied_now: list[Migration] = []

    # autocommit=True means each `conn.transaction()` block below is a real
    # BEGIN ... COMMIT, not a savepoint inside one long transaction.
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version     text PRIMARY KEY,
                    name        text NOT NULL,
                    checksum    text NOT NULL,
                    applied_at  timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            history = applied_checksums(conn)

            for migration in migrations:
                recorded = history.get(migration.version)
                if recorded is not None:
                    if recorded != migration.checksum:
                        raise MigrationError(
                            f"{migration.path.name} changed after it was applied. "
                            "Applied migrations are immutable: revert the edit and "
                            "add a new migration instead (or `make db-reset` to "
                            "start over while developing)."
                        )
                    continue

                with conn.transaction():
                    # Bytes, not str: psycopg only accepts literal strings as
                    # queries, and a file's contents are not a literal. With no
                    # parameters, one execute() may run several statements.
                    conn.execute(migration.sql)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, name, checksum) "
                        "VALUES (%s, %s, %s)",
                        (migration.version, migration.name, migration.checksum),
                    )
                applied_now.append(migration)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))

    return applied_now


def main() -> int:
    """Apply pending migrations to the database in DATABASE_URL."""
    try:
        applied = migrate(str(get_settings().database_url))
    except (MigrationError, psycopg.Error) as error:
        print(f"migration failed: {error}", file=sys.stderr)
        return 1
    if not applied:
        print("Database is up to date.")
    for migration in applied:
        print(f"Applied {migration.path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
