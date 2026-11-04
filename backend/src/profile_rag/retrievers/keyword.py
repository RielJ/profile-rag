"""Keyword search that matches any word of a question, ranked by `ts_rank_cd`.

`PGVectorStore` 0.0.19 has a keyword search of its own, but it always builds
the query with `plainto_tsquery`, which joins the words with AND. A question
such as "What languages does Dana use at work?" then matches only a chunk
that contains every one of its words, which almost never happens. The
library has no setting to change that function, so this module runs the
keyword half of hybrid search itself:

1. Postgres parses the question with the same text search configuration
   that filled the table's tsvector column (`store.TS_CONFIG`). That drops
   stopwords ("what", "does", "at") and stems the rest ("jobs" becomes
   "job").
2. The lexemes are joined with OR (`|`) into one tsquery. A chunk that
   contains any of them matches; `ts_rank_cd` ranks chunks with more and
   closer matches higher.

A question of stopwords only gives no lexemes, so no keyword hits.

The SQL text is fixed. Every value (the question, the tsquery, filter
values, the limit) is a query parameter. The table name is not a value: it
must be one of `store.TABLE_NAMES` and is quoted with `sql.Identifier`, as
are the column names.

Tests: backend/tests/test_retrievers.py.
"""

from collections.abc import Mapping, Sequence

import psycopg
from langchain_core.documents import Document
from psycopg import sql
from psycopg.rows import dict_row

from profile_rag.vectorstore.models import EmbeddingModel
from profile_rag.vectorstore.store import (
    CONTENT_COLUMN,
    ID_COLUMN,
    METADATA_COLUMNS,
    METADATA_JSON_COLUMN,
    TABLE_NAMES,
    TS_CONFIG,
    TSV_COLUMN,
)


def question_lexemes(conn: psycopg.Connection, question: str) -> list[str]:
    """The distinct lexemes of `question` under `TS_CONFIG`, sorted. Stopwords are dropped."""
    row = conn.execute(
        "SELECT tsvector_to_array(to_tsvector(%s::regconfig, %s))", (TS_CONFIG, question)
    ).fetchone()
    return list(row[0]) if row and row[0] else []


def or_query(lexemes: Sequence[str]) -> str:
    """The tsquery text that matches any of `lexemes`: `'a' | 'b' | ...`.

    Each lexeme is quoted for the tsquery input format (a quote is doubled,
    a backslash escaped), so it is read as one lexeme, never as an operator.
    Cast the result with `::tsquery`, not `to_tsquery`, so Postgres does not
    stem the lexemes a second time.
    """
    quoted = ("'" + lexeme.replace("\\", "\\\\").replace("'", "''") + "'" for lexeme in lexemes)
    return " | ".join(quoted)


def allowed_values(filter: Mapping[str, object] | None) -> dict[str, list[str] | None]:
    """For each of `METADATA_COLUMNS`, the values a metadata filter allows (None: any).

    Supports the subset of the `PGVectorStore` filter syntax this app uses,
    such as `dense.VISIBLE`: `{column: value}`, `{column: {"$eq": value}}`,
    and `{column: {"$in": [values]}}`, with string values and `column` one of
    `METADATA_COLUMNS`. Several keys must all hold. Anything else raises
    ValueError rather than being ignored.
    """
    allowed: dict[str, list[str] | None] = dict.fromkeys(METADATA_COLUMNS)
    for column, condition in (filter or {}).items():
        if column not in allowed:
            raise ValueError(f"keyword search cannot filter on {column!r}")
        if not isinstance(condition, Mapping):
            condition = {"$eq": condition}
        if len(condition) != 1:
            raise ValueError(f"one operator per column, got {sorted(condition)}")
        [(operator, value)] = condition.items()
        if operator == "$eq":
            values = [value]
        elif operator == "$in" and isinstance(value, Sequence) and not isinstance(value, str):
            values = list(value)
        else:
            raise ValueError(f"keyword search does not support {operator!r} on {column!r}")
        if not all(isinstance(item, str) for item in values):
            raise ValueError(f"keyword search filters on strings only, got {values!r}")
        allowed[column] = [str(item) for item in values]
    return allowed


def keyword_search(
    conninfo: str,
    model: EmbeddingModel,
    question: str,
    *,
    k: int,
    filter: Mapping[str, object] | None = None,
) -> list[Document]:
    """At most `k` chunks of the model's table that contain any word of `question`.

    Best `ts_rank_cd` first; ties go by `chunk_id`. Rows that `filter`
    rejects are never returned. Each Document looks like one from
    `PGVectorStore`: the row's text, its JSON metadata plus the metadata
    columns, and the row ID as `id`.

    Raises ValueError for a table outside `TABLE_NAMES` or an unsupported
    filter.
    """
    if model.table_name not in TABLE_NAMES:
        raise ValueError(f"unknown chunk table {model.table_name!r}")
    allowed = allowed_values(filter)
    with psycopg.connect(conninfo) as conn:
        lexemes = question_lexemes(conn, question)
        if not lexemes:
            return []
        # One `(%s IS NULL OR column = ANY(%s))` per metadata column: NULL
        # means the filter does not restrict that column.
        params: list[object] = [or_query(lexemes)]
        for column in METADATA_COLUMNS:
            params += [allowed[column], allowed[column]]
        params.append(k)
        with conn.cursor(row_factory=dict_row) as cursor:
            rows = cursor.execute(
                sql.SQL(
                    "SELECT {id}, {content}, {source_id}, {chunk_id}, {attribution},"
                    " {visibility}, {metadata}, ts_rank_cd({tsv}, query) AS rank"
                    " FROM {table}, CAST(%s AS tsquery) AS query"
                    " WHERE {tsv} @@ query"
                    " AND (%s::text[] IS NULL OR {source_id} = ANY(%s::text[]))"
                    " AND (%s::text[] IS NULL OR {chunk_id} = ANY(%s::text[]))"
                    " AND (%s::text[] IS NULL OR {attribution} = ANY(%s::text[]))"
                    " AND (%s::text[] IS NULL OR {visibility} = ANY(%s::text[]))"
                    " ORDER BY rank DESC, {chunk_id}"
                    " LIMIT %s"
                ).format(
                    id=sql.Identifier(ID_COLUMN),
                    content=sql.Identifier(CONTENT_COLUMN),
                    source_id=sql.Identifier("source_id"),
                    chunk_id=sql.Identifier("chunk_id"),
                    attribution=sql.Identifier("attribution"),
                    visibility=sql.Identifier("visibility"),
                    metadata=sql.Identifier(METADATA_JSON_COLUMN),
                    tsv=sql.Identifier(TSV_COLUMN),
                    table=sql.Identifier(model.table_name),
                ),
                params,
            ).fetchall()
    found: list[Document] = []
    for row in rows:
        metadata = dict(row[METADATA_JSON_COLUMN] or {})
        metadata.update({column: row[column] for column in METADATA_COLUMNS})
        found.append(
            Document(page_content=row[CONTENT_COLUMN], metadata=metadata, id=row[ID_COLUMN])
        )
    return found
