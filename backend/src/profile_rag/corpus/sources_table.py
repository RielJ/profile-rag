"""Keep the `sources` table equal to the approved entries of sources.yaml.

`sources` (db/migrations/0001_initial.sql) holds one row per approved
source. Candidate, in-review, and rejected sources never reach it, so
nothing that reads the database can return unapproved material. The index
command calls `sync_sources` on every run: it writes each approved entry
and deletes the rows of sources that are no longer approved.
"""

from dataclasses import dataclass

import psycopg
from psycopg.rows import TupleRow

from profile_rag.corpus.registry import Registry


@dataclass(frozen=True)
class SourcesSync:
    """What `sync_sources` changed."""

    upserted: list[str]  # IDs written, inserted or updated
    deleted: list[str]  # IDs removed because they are no longer approved


UPSERT = """
INSERT INTO sources (id, kind, title, url, visibility, attribution, revision,
                     content_sha256, approved_at, approved_by)
VALUES (%(id)s, %(kind)s, %(title)s, %(url)s, %(visibility)s, %(attribution)s,
        %(revision)s, %(content_sha256)s, %(approved_at)s, %(approved_by)s)
ON CONFLICT (id) DO UPDATE SET
    kind = EXCLUDED.kind,
    title = EXCLUDED.title,
    url = EXCLUDED.url,
    visibility = EXCLUDED.visibility,
    attribution = EXCLUDED.attribution,
    revision = EXCLUDED.revision,
    content_sha256 = EXCLUDED.content_sha256,
    approved_at = EXCLUDED.approved_at,
    approved_by = EXCLUDED.approved_by
"""


def sync_sources(conn: psycopg.Connection[TupleRow], registry: Registry) -> SourcesSync:
    """Upsert every approved entry and delete every other row, in one transaction.

    Commits on success and rolls back on error, so `sources` is never left
    half-updated.
    """
    approved = [entry for entry in registry.sources if entry.state == "approved"]
    with conn.transaction():
        for entry in approved:
            conn.execute(
                UPSERT,
                {
                    "id": entry.id,
                    "kind": entry.kind,
                    "title": entry.title,
                    "url": entry.url,
                    "visibility": entry.visibility,
                    "attribution": entry.attribution,
                    "revision": entry.revision,
                    "content_sha256": entry.content_sha256,
                    "approved_at": entry.approved_at,
                    "approved_by": entry.approved_by,
                },
            )
        ids = [entry.id for entry in approved]
        rows = conn.execute(
            "DELETE FROM sources WHERE NOT (id = ANY(%s)) RETURNING id", (ids,)
        ).fetchall()
    return SourcesSync(upserted=ids, deleted=sorted(row[0] for row in rows))
