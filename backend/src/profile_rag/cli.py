"""The `profile-rag` command line.

Corpus commands read `sources.yaml` (corpus/registry.py), change it, and
write it back:

    profile-rag corpus add ID --kind K --title T --attribution A --visibility V
    profile-rag corpus snapshot ID [--from PATH_OR_URL] [--private] [--revision R]
    profile-rag corpus review [ID] [--all] [--approve --checklist-confirmed | --reject REASON]
                          [--by NAME]
    profile-rag corpus status

The lifecycle of one source:

    add       creates the entry as `candidate`
    snapshot  copies the content into approved/ and moves it to `in_review`
    review    walks the sources waiting for review (corpus/review_session.py):
              view, edit, checklist, then `approved` (with the file's
              SHA-256) or `rejected`
    status    lists every entry and reports approved files that changed
              since approval (stale). Exit code 1 if anything is stale

The corpus root is `corpus/` at the repository root. Use `--root` or the
`CORPUS_DIR` environment variable for another one, for example
`corpus-example/`.

Read the chunks of one approved source:

    profile-rag chunks show ID [--root DIR] [--chunk-words N] [--overlap-words N]

It loads the source's snapshot (corpus/documents.py), splits it
(corpus/splitting.py), and prints each chunk under its heading path. It
reads files only: nothing is written and no database is used.

Index the corpus:

    profile-rag index [--model nomic-v15] [--root DIR]

It checks that Ollama has the model with the expected digest, loads and
splits every approved source, writes the approved entries to the `sources`
table, and indexes all chunks (vectorstore/indexing.py). It prints the
counts from `index()`. A second run with nothing changed skips every chunk.

Search the index:

    profile-rag search "QUESTION" [--model nomic-v15] [--mode dense|hybrid|mmr] [-k 8]

It opens the model's table (vectorstore/store.py), runs the retriever for
the mode (retrievers/dense.py or retrievers/hybrid.py) with the visibility
filter, and prints one line per chunk: the rank, the cosine distance
between the question and the chunk, the `chunk_id`, and the heading path.
The distance is the same measure in every mode, so it shows where MMR and
hybrid search leave the dense order. It is not the score a mode ranks by.

Answer a question:

    profile-rag ask "QUESTION" [--model nomic-v15] [--mode dense|hybrid|mmr] [-k 8]

It retrieves k chunks as `search` does, then runs the grounded answer chain
(generation/chain.py) through Pi (generation/pi_rpc.py, settings PROFILE_RAG_PI_BIN,
PROFILE_RAG_PI_MODEL, PROFILE_RAG_PI_TIMEOUT_SECONDS). It prints the answer and each cited chunk
with its source title and attribution, or `Unknown.` and the reason. Exit
codes: 0 answered or unknown, 2 setup error or failed search, 3 Pi failed, 4 the model's
reply was rejected (not a valid answer, or a citation of a chunk it was not
given). Error messages never contain the model's reply.

Evaluate retrieval:

    profile-rag eval --set dev|heldout|example [--model nomic-v15] [--mode dense|hybrid|mmr] [-k 8]

It loads the set's questions (evaluation/questions.py), resolves each label
to chunk IDs against a fresh split of the set's corpus (evaluation/labels.py)
and stops on any label error. It checks that the model's table holds exactly
that split, runs every question through the same retriever `search` uses,
and writes per-question detail to eval/results/ and the aggregate report and
comparison table to docs/eval/ (evaluation/runner.py).

`uv --directory backend run ...` runs in backend/, so relative paths are
relative to backend/. To use paths relative to the repository root, run
from there with `uv run --project backend profile-rag ...`.
"""

import getpass
from collections import Counter
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, NoReturn, get_args

import click
import psycopg
import typer
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_postgres import PGVectorStore
from pydantic import ValidationError

from profile_rag import rag, tracing
from profile_rag.agent.errors import AgentLoopError, GradeParseError, RewriteError
from profile_rag.agent.graph import build_graph, run_agent
from profile_rag.api.models import FailedCode
from profile_rag.corpus import documents, registry, review_session, sources_table, splitting
from profile_rag.corpus.registry import (
    REGISTRY_FILENAME,
    Attribution,
    Kind,
    Registry,
    RegistryError,
    SourceEntry,
    State,
    Visibility,
)
from profile_rag.corpus.snapshot import SnapshotError, take_snapshot
from profile_rag.corpus.splitting import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    SPLITTER_VERSION,
)
from profile_rag.evaluation import agent_compare, ann_compare
from profile_rag.evaluation.labels import ResolvedQuestion, resolve_labels
from profile_rag.evaluation.questions import QuestionFileError, load_questions
from profile_rag.evaluation.runner import RunConfig, run_evaluation
from profile_rag.generation import pi_chat_model
from profile_rag.generation.answer import InventedCitation, Unknown
from profile_rag.generation.chain import GroundedAnswer, build_chain
from profile_rag.generation.pi_rpc import PiError, PiProcessError, PiTimeout
from profile_rag.retrievers.factory import SearchMode, build_retriever, dense_candidates
from profile_rag.settings import REPO_ROOT, Settings, TracingSettings, get_settings
from profile_rag.vectorstore import hnsw, indexing, ollama_embeddings, store
from profile_rag.vectorstore.models import DEFAULT_MODEL, MODELS, EmbeddingModel
from profile_rag.vectorstore.prefixed import EmbeddingError

DEFAULT_CORPUS = REPO_ROOT / "corpus"

app = typer.Typer(help="Question answering over reviewed professional evidence.")
corpus_app = typer.Typer(help="Manage the corpus registry (sources.yaml) and reviews.")
app.add_typer(corpus_app, name="corpus")
chunks_app = typer.Typer(help="Read the chunks the splitter makes from a source.")
app.add_typer(chunks_app, name="chunks")
hnsw_app = typer.Typer(help="Manage the HNSW index of a model's chunk table.")
app.add_typer(hnsw_app, name="hnsw")


def main() -> None:
    """Entry point of the `profile-rag` console script."""
    app()


@app.callback()
def _root_callback() -> None:
    # Runs before every command, so before any LangChain object is made.
    # It also makes Typer treat every command as a subcommand.
    try:
        tracing.configure_tracing(TracingSettings(), tracing.corpus_dir_from_env())
    except tracing.TracingRefused as error:
        _fail(str(error))


# --- Shared helpers ---------------------------------------------------------


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(2)


def _describe(error: ValidationError) -> str:
    """One line per problem, without Pydantic's input dump and help links."""
    lines: list[str] = []
    for problem in error.errors():
        where = ".".join(str(part) for part in problem["loc"])
        message = problem["msg"].removeprefix("Value error, ")
        lines.append(f"  {where}: {message}" if where else f"  {message}")
    return "\n".join(lines)


def _root(context: typer.Context) -> Path:
    root = context.obj
    if not isinstance(root, Path):
        raise TypeError("the corpus callback did not set the root path")
    return root


# The registry is looked up through the `registry` module on every call, not
# imported by name, so tests can patch it.


def _load(root: Path, *, create: bool = False) -> Registry:
    path = root / REGISTRY_FILENAME
    if not path.exists():
        if create:
            return registry.Registry()
        _fail(f"{path} does not exist. Create it with `profile-rag corpus add`.")
    try:
        return registry.load_registry(path)
    except RegistryError as error:
        _fail(str(error))


def _save(root: Path, entries: list[SourceEntry]) -> None:
    """Validate the whole registry again, then write it."""
    try:
        updated = registry.Registry.model_validate(
            {"sources": [entry.model_dump() for entry in entries]}
        )
    except ValidationError as error:
        _fail(f"the change would make {REGISTRY_FILENAME} invalid:\n{_describe(error)}")
    root.mkdir(parents=True, exist_ok=True)
    registry.save_registry(updated, root / REGISTRY_FILENAME)


def _entry(current: Registry, source_id: str) -> SourceEntry:
    entry = current.get(source_id)
    if entry is None:
        _fail(f"no source with id {source_id!r} in {REGISTRY_FILENAME}")
    return entry


def _updated(entry: SourceEntry, **changes: object) -> SourceEntry:
    """Return `entry` with `changes` applied, validated by the model rules."""
    try:
        return registry.SourceEntry.model_validate({**entry.model_dump(), **changes})
    except ValidationError as error:
        _fail(f"{entry.id}: invalid change:\n{_describe(error)}")


def _replace(current: Registry, entry: SourceEntry) -> list[SourceEntry]:
    return [entry if existing.id == entry.id else existing for existing in current.sources]


# --- Commands ---------------------------------------------------------------


@corpus_app.callback()
def corpus(
    context: typer.Context,
    root: Annotated[
        Path,
        typer.Option(
            envvar="CORPUS_DIR",
            help="Corpus root: the folder with sources.yaml, approved/, private/, reviews/.",
            show_default="corpus/ in this repository",
        ),
    ] = DEFAULT_CORPUS,
) -> None:
    """Manage the corpus registry (sources.yaml) and reviews."""
    context.obj = root.expanduser().resolve()


@corpus_app.command()
def add(
    context: typer.Context,
    source_id: Annotated[str, typer.Argument(metavar="ID", help="e.g. resume-2026")],
    kind: Annotated[Kind, typer.Option()],
    title: Annotated[str, typer.Option()],
    attribution: Annotated[Attribution, typer.Option()],
    visibility: Annotated[Visibility, typer.Option()],
    url: Annotated[str | None, typer.Option(help="Public link shown to visitors.")] = None,
    origin: Annotated[
        str | None, typer.Option("--from", help="Local path or URL to snapshot from.")
    ] = None,
    notes: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Add a new source as a `candidate`."""
    root = _root(context)
    current = _load(root, create=True)
    if current.get(source_id) is not None:
        _fail(f"a source with id {source_id!r} already exists")
    try:
        entry = registry.SourceEntry.model_validate(
            {
                "id": source_id,
                "kind": kind,
                "title": title,
                "attribution": attribution,
                "visibility": visibility,
                "state": "candidate",
                "url": url,
                "origin": origin,
                "notes": notes,
            }
        )
    except ValidationError as error:
        _fail(f"{source_id}: invalid source:\n{_describe(error)}")
    _save(root, [*current.sources, entry])
    typer.echo(f"Added {source_id} as candidate. Next: profile-rag corpus snapshot {source_id}")


@corpus_app.command()
def snapshot(
    context: typer.Context,
    source_id: Annotated[str, typer.Argument(metavar="ID")],
    origin: Annotated[
        str | None,
        typer.Option("--from", help="Local path or http(s) URL. Defaults to the entry's origin."),
    ] = None,
    private: Annotated[
        bool, typer.Option(help="Save a raw copy in private/; the entry is not changed.")
    ] = False,
    revision: Annotated[
        str | None, typer.Option(help="Record this revision, e.g. a commit SHA.")
    ] = None,
) -> None:
    """Copy a source into the corpus and move it to `in_review`."""
    root = _root(context)
    current = _load(root)
    entry = _entry(current, source_id)
    origin = origin or entry.origin
    if origin is None:
        _fail(f"{source_id} has no origin. Pass --from PATH_OR_URL.")
    try:
        taken = take_snapshot(source_id, origin, root, private=private, revision=revision)
    except (SnapshotError, OSError) as error:
        _fail(str(error))

    if taken.private:
        typer.echo(f"Saved a raw copy to {taken.path} (never ingested). Write a sanitized version")
        typer.echo(f"to approved/, then run: profile-rag corpus snapshot {source_id} --from FILE")
        return

    # New content means a new review: the old approval no longer applies.
    changes: dict[str, object] = {
        "state": "in_review",
        "path": taken.path,
        "revision": taken.revision,
        "origin": taken.origin,
        "content_sha256": None,
        "approved_at": None,
        "approved_by": None,
        "rejection_reason": None,
    }
    _save(root, _replace(current, _updated(entry, **changes)))
    typer.echo(f"Snapshot {taken.path} (revision {taken.revision}); {source_id} is in_review.")
    typer.echo(f"Next: profile-rag corpus review {source_id}")


@corpus_app.command()
def review(
    context: typer.Context,
    source_id: Annotated[
        str | None,
        typer.Argument(metavar="[ID]", help="One source. Leave out to review the whole queue."),
    ] = None,
    approve: Annotated[
        bool,
        typer.Option("--approve", help="Approve ID without asking. Needs --checklist-confirmed."),
    ] = False,
    checklist_confirmed: Annotated[
        bool,
        typer.Option(
            "--checklist-confirmed", help="With --approve: you checked every checklist item."
        ),
    ] = False,
    reject: Annotated[
        str | None, typer.Option(metavar="REASON", help="Reject ID without asking.")
    ] = None,
    include_all: Annotated[
        bool, typer.Option("--all", help="Also re-review approved sources that are not stale.")
    ] = False,
    by: Annotated[str | None, typer.Option(help="Reviewer name. Defaults to your login.")] = None,
) -> None:
    """Review sources one by one: view, edit, checklist, then approve or reject.

    Without ID, the queue is every in_review source, then every stale
    approval, in sources.yaml order (corpus/review_session.py).
    """
    if approve and reject is not None:
        _fail("use --approve or --reject, not both")
    if reject is not None and not reject.strip():
        _fail("--reject needs a reason")
    if approve and not checklist_confirmed:
        _fail("--approve also needs --checklist-confirmed: you state that you checked every item")
    if checklist_confirmed and not approve:
        _fail("--checklist-confirmed only goes with --approve")
    if source_id is None and (approve or reject is not None):
        _fail("--approve and --reject need an ID")
    if source_id is not None and include_all:
        _fail("--all reviews the whole queue; leave out the ID")
    root = _root(context)
    current = _load(root)
    io = review_session.ReviewIO(prompt=_prompt, echo=typer.echo, show=typer.echo, edit=_edit)
    reviewer = by or getpass.getuser()
    try:
        if source_id is None:
            queue = review_session.build_queue(current.sources, root, include_all=include_all)
            for missing in queue.missing:
                typer.echo(f"{missing}: the snapshot file is missing; snapshot it again first.")
            if not queue.sources:
                typer.echo("Nothing to review. --all also lists approved sources.")
                return
            entries = queue.sources
        else:
            entry = _reviewable(root, _entry(current, source_id))
            if approve or reject is not None:
                review_session.decide_by_flag(
                    root=root, entry=entry, io=io, reviewer=reviewer, when=_now(), reject=reject
                )
                return
            entries = [entry]
        session = review_session.Session(root=root, io=io, reviewer=reviewer, clock=_now)
        review_session.print_summary(io, root, session.run(entries))
    except review_session.ReviewError as error:
        _fail(str(error))


def _reviewable(root: Path, entry: SourceEntry) -> SourceEntry:
    if entry.state not in ("in_review", "approved"):
        hint = "snapshot" if entry.state == "candidate" else "snapshot it again before reviewing"
        _fail(f"{entry.id} is {entry.state}; {hint}: profile-rag corpus snapshot {entry.id}")
    if entry.path is None or not (root / entry.path).is_file():
        _fail(f"{entry.id} has no snapshot file; run profile-rag corpus snapshot {entry.id}")
    return entry


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _prompt(text: str) -> str:
    try:
        return typer.prompt(text, default="", show_default=False)
    except typer.Abort:  # Click's prompt turns Ctrl-C and end of input into Abort
        raise EOFError from None


def _edit(path: Path) -> None:
    try:
        click.edit(filename=str(path))
    except click.ClickException as error:
        raise review_session.EditorFailed(error.format_message()) from error


@corpus_app.command()
def status(context: typer.Context) -> None:
    """List sources by state and report stale approvals. Exit 1 if any are stale."""
    root = _root(context)
    current = _load(root)
    entries = current.sources
    stale = registry.find_stale(entries, root)

    counts = Counter(entry.state for entry in entries)
    states: tuple[State, ...] = get_args(State)
    typer.echo(f"{root / REGISTRY_FILENAME}: {len(entries)} sources")
    typer.echo("  ".join(f"{state} {counts[state]}" for state in states))
    if entries:
        typer.echo("")
        width = max(len(entry.id) for entry in entries)
        for entry in entries:
            mark = "  STALE" if entry.id in stale else ""
            typer.echo(
                f"  {entry.id:<{width}}  {entry.state:<9}  {entry.kind:<9}  "
                f"{entry.attribution:<13}  {entry.visibility}{mark}"
            )

    referenced = {entry.path for entry in entries if entry.path is not None}
    approved_dir = root / registry.APPROVED_DIR
    unreferenced = sorted(
        path.relative_to(root).as_posix()
        for path in approved_dir.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    )
    unreferenced = [path for path in unreferenced if path not in referenced]
    if unreferenced:
        typer.echo("\nFiles in approved/ that no entry points to (they are never ingested):")
        for path in unreferenced:
            typer.echo(f"  {path}")

    if stale:
        typer.echo("\nStale: the file is missing or changed since it was approved.")
        typer.echo("Snapshot and review each one again:")
        for source_id in stale:
            typer.echo(f"  {source_id}")
        raise typer.Exit(1)


# --- chunks -------------------------------------------------------------------


@chunks_app.command("show")
def chunks_show(
    source_id: Annotated[str, typer.Argument(metavar="ID", help="e.g. resume-2026")],
    root: Annotated[
        Path,
        typer.Option(
            envvar="CORPUS_DIR",
            help="Corpus root: the folder with sources.yaml and approved/.",
            show_default="corpus/ in this repository",
        ),
    ] = DEFAULT_CORPUS,
    chunk_words: Annotated[
        int, typer.Option(help="Most words in one chunk.")
    ] = DEFAULT_CHUNK_WORDS,
    overlap_words: Annotated[
        int, typer.Option(help="Most words a chunk repeats from the previous one.")
    ] = DEFAULT_OVERLAP_WORDS,
) -> None:
    """Split one approved source now and print its chunks, each under its heading path.

    The text a chunk embeds is its heading path (the `[n]` line), a blank
    line, then the body (the `|` lines). The word count is the body's.
    """
    root = root.expanduser().resolve()
    entry = _entry(_load(root), source_id)
    if entry.state != "approved":
        _fail(f"{source_id} is {entry.state}, not approved. Only approved sources are split.")
    try:
        loaded = documents.load_documents(registry.Registry(sources=[entry]), root)
    except documents.ApprovedSnapshotError as error:  # the message starts with the ID
        _fail(str(error))
    except UnicodeDecodeError as error:
        _fail(f"{source_id}: not UTF-8 text: {error}")
    try:
        chunks = splitting.split_documents(
            loaded, chunk_words=chunk_words, overlap_words=overlap_words
        )
    except ValueError as error:
        _fail(str(error))
    typer.echo(
        f"{source_id}: {len(chunks)} chunks ({SPLITTER_VERSION}, at most {chunk_words} words,"
        f" overlap {overlap_words})"
    )
    for chunk in chunks:
        heading_path: list[str] = chunk.metadata["heading_path"]
        path = splitting.heading_line(heading_path) or "(before the first heading)"
        body = splitting.chunk_body(chunk)
        size = splitting.count_words(body)
        typer.echo(
            f"\n[{chunk.metadata['ordinal']}] {path}\n"
            f"    {size} words, id {chunk.metadata['chunk_id']}"
        )
        for line in body.split("\n"):
            typer.echo(f"    | {line}".rstrip())


# --- index --------------------------------------------------------------------
#
# The Ollama parts go through vectorstore/ollama_embeddings.py; tests replace
# its `ollama_digests` and `for_model` with fakes.


def _check_model(model: EmbeddingModel, base_url: str) -> None:
    if problem := ollama_embeddings.model_problem(model, base_url):
        _fail(problem)


def _name_chunk(chunks: list[Document], error: EmbeddingError) -> str:
    if error.text is None:
        return str(error)
    ids = [c.metadata.get("chunk_id", "?") for c in chunks if c.page_content == error.text]
    return f"chunk {', '.join(ids) or '(unknown)'}: {error.reason}"


@app.command("index")
def index_command(
    model_key: Annotated[
        str,
        typer.Option("--model", help=f"One of: {', '.join(MODELS)}."),
    ] = DEFAULT_MODEL,
    root: Annotated[
        Path,
        typer.Option(
            envvar="CORPUS_DIR",
            help="Corpus root: the folder with sources.yaml and approved/.",
            show_default="corpus/ in this repository",
        ),
    ] = DEFAULT_CORPUS,
) -> None:
    """Embed every approved chunk into the model's table, and sync `sources`."""
    model = MODELS.get(model_key)
    if model is None:
        _fail(f"unknown model {model_key!r}. Choose one of: {', '.join(MODELS)}.")
    settings = get_settings()
    database_url = str(settings.database_url)
    _check_model(model, settings.ollama_url)

    root = root.expanduser().resolve()
    current = _load(root)
    try:
        loaded = documents.load_documents(current, root)
    except documents.ApprovedSnapshotError as error:
        _fail(str(error))
    chunks = splitting.split_documents(
        loaded, chunk_words=DEFAULT_CHUNK_WORDS, overlap_words=DEFAULT_OVERLAP_WORDS
    )

    with psycopg.connect(database_url) as conn:
        synced = sources_table.sync_sources(conn, current)
    typer.echo(f"sources: {len(synced.upserted)} approved, {len(synced.deleted)} removed")
    for source_id in synced.deleted:
        typer.echo(f"  removed {source_id}")

    embeddings = ollama_embeddings.for_model(model, settings.ollama_url)
    with store.connected(database_url) as engine:
        if not store.table_exists(database_url, model.table_name):
            store.create_table(engine, model, embeddings)
            typer.echo(f"created table {model.table_name}")
        index_store = store.open_index_store(engine, model, embeddings)
        record_manager = indexing.open_record_manager(model, database_url)
        try:
            result = indexing.index_corpus(chunks, index_store, record_manager)
        except EmbeddingError as error:
            _fail(_name_chunk(chunks, error))
    typer.echo(
        f"{model.table_name}: {len(chunks)} chunks, {result['num_added']} added,"
        f" {result['num_updated']} updated, {result['num_skipped']} skipped,"
        f" {result['num_deleted']} deleted"
    )


# --- search -------------------------------------------------------------------


def _warn_low_ef_search(ef_search: int | None, mode: SearchMode, k: int) -> None:
    """Warn if an HNSW scan could return fewer rows than the dense query asks for."""
    wanted = dense_candidates(mode, k)
    if ef_search is not None and ef_search < wanted:
        typer.echo(
            f"warning: --ef-search {ef_search} is below the {wanted} rows that {mode} search"
            f" asks the dense query for. When Postgres uses the HNSW index, it returns at"
            f" most {ef_search}.",
            err=True,
        )


EfSearchOption = Annotated[
    int | None,
    typer.Option(
        "--ef-search",
        min=hnsw.EF_SEARCH_MIN,
        max=hnsw.EF_SEARCH_MAX,
        help="Set hnsw.ef_search for the dense queries (only matters if the HNSW index is used).",
    ),
]


def _distances(search_store: PGVectorStore, question: str, chunks: list[Document]) -> list[float]:
    """The cosine distance from the question to each chunk, in the order of `chunks`."""
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    scored = search_store.similarity_search_with_score(
        question, k=len(ids), filter={"chunk_id": {"$in": ids}}
    )
    by_id = {doc.metadata["chunk_id"]: distance for doc, distance in scored}
    return [by_id[chunk_id] for chunk_id in ids]


@app.command("search")
def search_command(
    question: Annotated[str, typer.Argument(help="The question, in quotes.")],
    model_key: Annotated[
        str,
        typer.Option("--model", help=f"One of: {', '.join(MODELS)}."),
    ] = DEFAULT_MODEL,
    mode: Annotated[SearchMode, typer.Option(help="Which retriever to run.")] = "dense",
    k: Annotated[int, typer.Option("-k", min=1, help="How many chunks to print.")] = 8,
    ef_search: EfSearchOption = None,
) -> None:
    """Retrieve the chunks for a question and print them, best first."""
    model = MODELS.get(model_key)
    if model is None:
        _fail(f"unknown model {model_key!r}. Choose one of: {', '.join(MODELS)}.")
    settings = get_settings()
    database_url = str(settings.database_url)
    _check_model(model, settings.ollama_url)
    _require_table(database_url, model)
    _require_traceable(database_url, model)

    _warn_low_ef_search(ef_search, mode, k)
    embeddings = ollama_embeddings.for_model(model, settings.ollama_url)
    with store.searching(database_url, model, embeddings, ef_search=ef_search) as search_store:
        try:
            retriever = build_retriever(mode, search_store, k, conninfo=database_url, model=model)
            chunks = retriever.invoke(question)
            distances = _distances(search_store, question, chunks) if chunks else []
        except EmbeddingError as error:
            _fail(f"the question: {error.reason}")

    typer.echo(f"{model.table_name}, {mode}: {len(chunks)} chunks")
    if chunks:
        typer.echo(f"{'rank':>4}  {'distance':>8}  chunk_id  heading path")
    for rank, (chunk, distance) in enumerate(zip(chunks, distances, strict=True), start=1):
        heading_path: list[str] = chunk.metadata.get("heading_path") or []
        path = " > ".join(heading_path) or "(before the first heading)"
        typer.echo(f"{rank:>4}  {distance:>8.4f}  {chunk.metadata['chunk_id']}  {path}")


# --- ask ----------------------------------------------------------------------
#
# Exit codes of `ask` and `agent`: 0 answered or unknown, 2 setup error or a
# failed evidence search (the `_fail` code), 3 Pi failed, 4 the model's reply
# was rejected, 5 the agent graph exceeded its recursion limit (a bug). A message never contains the
# model's reply: an OutputParserException's text does, so it is not printed.

EXIT_PI_FAILED = 3
EXIT_REPLY_REJECTED = 4
EXIT_AGENT_LOOP = 5

# Exit code of each failure of the chain. `rag.failure_code` decides which
# failure an error is, for the CLI and the API alike; this table and the
# API's FAILED_STATUS (api/routes.py) only translate it. `internal_error`
# has no exit code: the error is a bug, so it is raised with its traceback.
ANSWER_EXIT: dict[FailedCode, int] = {
    "retrieval_failed": 2,
    "pi_timeout": EXIT_PI_FAILED,
    "pi_failed": EXIT_PI_FAILED,
    "bad_reply": EXIT_REPLY_REJECTED,
    "invented_citation": EXIT_REPLY_REJECTED,
}

QuestionArgument = Annotated[str, typer.Argument(help="The question, in quotes.")]
AnswerModelOption = Annotated[
    str, typer.Option("--model", help=f"Embedding model, one of: {', '.join(MODELS)}.")
]
AnswerModeOption = Annotated[SearchMode, typer.Option(help="Which retriever to run.")]
AnswerKOption = Annotated[int, typer.Option("-k", min=1, help="How many chunks to give the model.")]


def _pi_model(settings: Settings) -> BaseChatModel:
    """The Pi chat model from the settings. Tests replace this function."""
    return pi_chat_model.from_settings(settings)


def _ask_failure(message: str, code: int) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


def _describe_pi_error(error: PiError) -> str:
    if isinstance(error, PiTimeout):
        return f"{error} (set PROFILE_RAG_PI_TIMEOUT_SECONDS in .env to wait longer)."
    if isinstance(error, PiProcessError):
        lines = [str(error)]
        if error.returncode is None:
            lines.append("Install Pi, or set PROFILE_RAG_PI_BIN in .env to its path.")
        if error.stderr_tail:
            lines.append("Pi's stderr ends with:")
            lines.extend(f"  {line}" for line in error.stderr_tail.splitlines()[-10:])
        return "\n".join(lines)
    return str(error)


@app.command("ask")
def ask_command(
    question: QuestionArgument,
    model_key: AnswerModelOption = DEFAULT_MODEL,
    mode: AnswerModeOption = "hybrid",
    k: AnswerKOption = 8,
    ef_search: EfSearchOption = None,
) -> None:
    """Answer a question from the indexed evidence, through Pi, with citations."""
    with _answer_setup(question, model_key, mode, k, ef_search) as (retriever, settings):
        chain = build_chain(retriever, _pi_model(settings))
        with _answer_errors():
            result = chain.invoke({"question": question})
    _print_answer(result)


@app.command("agent")
def agent_command(
    question: QuestionArgument,
    model_key: AnswerModelOption = DEFAULT_MODEL,
    mode: AnswerModeOption = "hybrid",
    k: AnswerKOption = 8,
    ef_search: EfSearchOption = None,
) -> None:
    """Answer a question with the agent graph: grade the evidence, rewrite once if needed."""
    with _answer_setup(question, model_key, mode, k, ef_search) as (retriever, settings):
        graph = build_graph(retriever, _pi_model(settings))
        with _answer_errors():
            run = run_agent(graph, question, on_node=_print_node)
    typer.echo("")
    _print_answer(run.result)
    typer.echo(f"\nPi calls: {run.pi_calls}, rewrites: {run.rewrites}")


def _print_node(node: str, update: dict[str, object]) -> None:
    """One line per node: its name and counts or labels, never text."""
    detail = ""
    if node == "retrieve":
        docs = update.get("docs")
        detail = f"{len(docs) if isinstance(docs, list) else 0} chunks"
    elif node == "grade":
        detail = str(update.get("grade"))
    elif node == "rewrite":
        detail = "question rewritten"
    elif node == "generate":
        result = update.get("result")
        detail = result.answer.root.status if isinstance(result, GroundedAnswer) else "?"
    pi_calls = update.get("pi_calls")
    suffix = f" (Pi calls so far: {pi_calls})" if pi_calls is not None else ""
    typer.echo(f"[{node}] {detail}{suffix}")


@contextmanager
def _answer_setup(
    question: str, model_key: str, mode: SearchMode, k: int, ef_search: int | None
) -> Generator[tuple[BaseRetriever, Settings]]:
    """Check the inputs, open the store, and yield the retriever and settings."""
    if not question.strip():
        _fail("the question is empty.")
    model = _model(model_key)
    settings = get_settings()
    database_url = str(settings.database_url)
    _check_model(model, settings.ollama_url)
    _require_table(database_url, model)
    _require_traceable(database_url, model)

    _warn_low_ef_search(ef_search, mode, k)
    embeddings = ollama_embeddings.for_model(model, settings.ollama_url)
    with store.searching(database_url, model, embeddings, ef_search=ef_search) as search_store:
        yield build_retriever(mode, search_store, k, conninfo=database_url, model=model), settings


@contextmanager
def _answer_errors() -> Generator[None]:
    """Turn the errors of an answer run into exit codes. See the exit codes above."""
    try:
        yield
    # The agent graph's own errors. The API runs the chain only, so these are
    # not in `rag.failure_code`.
    except (GradeParseError, RewriteError) as error:
        _ask_failure(str(error), EXIT_REPLY_REJECTED)
    except AgentLoopError as error:
        _ask_failure(str(error), EXIT_AGENT_LOOP)
    except Exception as error:
        code = rag.failure_code(error)
        if code == "internal_error":
            raise
        _ask_failure(_describe_failure(code, error), ANSWER_EXIT[code])


def _describe_failure(code: FailedCode, error: Exception) -> str:
    """The CLI message of a chain failure. Never the model's reply."""
    match code:
        case "pi_timeout" | "pi_failed" if isinstance(error, PiError):
            return _describe_pi_error(error)
        case "bad_reply":
            # An OutputParserException's text contains the reply.
            return "the model's reply is not a valid answer (not JSON of the required shape)."
        case "invented_citation" if isinstance(error, InventedCitation):
            return f"the model cited chunks it was not given: {', '.join(error.invented)}."
        case "retrieval_failed" if isinstance(error, EmbeddingError):
            return f"the question: {error.reason}"
        case "retrieval_failed":
            first_line = (str(error).splitlines() or [""])[0][:200]
            return (
                f"the evidence search failed ({type(error).__name__}: {first_line})."
                " Check that Postgres and Ollama are running."
            )
        case _:
            return str(error)


def _print_answer(result: GroundedAnswer) -> None:
    answer = result.answer.root
    if isinstance(answer, Unknown):
        typer.echo("Unknown.")
        typer.echo(f"Reason: {answer.reason}")
        typer.echo(f"({len(result.docs)} chunks retrieved)")
        return
    typer.echo(answer.answer)
    typer.echo("\nCitations:")
    for doc in result.cited_docs():
        metadata = doc.metadata
        typer.echo(
            f"  {metadata['chunk_id']}  {metadata.get('title', '?')}"
            f" ({metadata.get('attribution', '?')})"
        )


# --- eval ---------------------------------------------------------------------

EvalSet = Literal["dev", "heldout", "example"]

# Question file and corpus root of each set. The example set is fictional and
# committed; the others are private and gitignored.
EVAL_SETS: dict[str, tuple[Path, Path]] = {
    "dev": (REPO_ROOT / "eval" / "dev.jsonl", DEFAULT_CORPUS),
    "heldout": (REPO_ROOT / "eval" / "heldout.jsonl", DEFAULT_CORPUS),
    "example": (REPO_ROOT / "eval-example" / "dev.jsonl", REPO_ROOT / "corpus-example"),
}
EVAL_RESULTS_DIR = REPO_ROOT / "eval" / "results"
EVAL_DOCS_DIR = REPO_ROOT / "docs" / "eval"


def _eval_inputs(
    eval_set: str, model: EmbeddingModel
) -> tuple[list[ResolvedQuestion], int, str, str]:
    """The resolved questions of a set, its chunk count, the database URL, and the Ollama URL.

    Exits with an error if the questions do not resolve, the model is not
    pulled, or the model's table does not hold exactly the set's corpus.
    """
    questions_path, root = EVAL_SETS[eval_set]
    try:
        questions = load_questions(questions_path)
    except QuestionFileError as error:
        _fail(str(error))

    current = _load(root)
    try:
        loaded = documents.load_documents(current, root)
    except documents.ApprovedSnapshotError as error:
        _fail(str(error))
    chunks = splitting.split_documents(
        loaded, chunk_words=DEFAULT_CHUNK_WORDS, overlap_words=DEFAULT_OVERLAP_WORDS
    )
    resolution = resolve_labels(questions, chunks, loaded)
    if resolution.errors:
        for error in resolution.errors:
            typer.echo(error.describe(), err=True)
        _fail(f"{len(resolution.errors)} label errors in {questions_path.name}. Fix the spans.")

    settings = get_settings()
    database_url = str(settings.database_url)
    _check_model(model, settings.ollama_url)
    _require_table(database_url, model)
    _require_traceable(database_url, model)
    indexed = store.chunk_ids(database_url, model)
    split = {chunk.metadata["chunk_id"] for chunk in chunks}
    if indexed != split:
        _fail(
            f"{model.table_name} does not hold this set's corpus ({root.name}/):"
            f" {len(split - indexed)} chunks missing, {len(indexed - split)} extra."
            f" Run `make index CORPUS_DIR={root.name} MODEL={model.key}` first."
        )
    return resolution.questions, len(chunks), database_url, settings.ollama_url


@app.command("eval")
def eval_command(
    eval_set: Annotated[
        EvalSet,
        typer.Option("--set", help="dev or heldout (eval/), or example (eval-example/)."),
    ] = "dev",
    model_key: Annotated[
        str,
        typer.Option("--model", help=f"One of: {', '.join(MODELS)}."),
    ] = DEFAULT_MODEL,
    mode: Annotated[SearchMode, typer.Option(help="Which retriever to run.")] = "dense",
    k: Annotated[int, typer.Option("-k", min=1, help="How many chunks each question gets.")] = 8,
    ef_search: EfSearchOption = None,
) -> None:
    """Measure recall@k and MRR of a question set, and write the reports."""
    model = _model(model_key)
    questions, chunk_count, database_url, ollama_url = _eval_inputs(eval_set, model)

    config = RunConfig(eval_set, model.key, mode, k, chunk_count, ef_search)
    _warn_low_ef_search(ef_search, mode, k)
    embeddings = ollama_embeddings.for_model(model, ollama_url)
    with store.searching(database_url, model, embeddings, ef_search=ef_search) as search_store:
        retriever = build_retriever(mode, search_store, k, conninfo=database_url, model=model)
        try:
            output = run_evaluation(
                retriever,
                questions,
                config,
                results_dir=EVAL_RESULTS_DIR,
                docs_dir=EVAL_DOCS_DIR,
            )
        except EmbeddingError as error:
            _fail(f"a question: {error.reason}")

    summary = output.summary
    recall = "n/a" if summary.recall is None else f"{summary.recall:.3f}"
    mrr = "n/a" if summary.mrr is None else f"{summary.mrr:.3f}"
    typer.echo(
        f"{config.name}: recall@{k} {recall}, MRR {mrr}, n {summary.n},"
        f" unanswerable {summary.unanswerable}"
    )
    typer.echo(f"misses: {', '.join(summary.misses) or 'none'}")
    for path in (output.detail_path, output.report_path, output.comparison_path):
        typer.echo(f"wrote {_shown(path)}")


# --- compare-agent ------------------------------------------------------------

CompareSet = Literal["dev", "example"]


@app.command("compare-agent")
def compare_agent_command(
    eval_set: Annotated[
        CompareSet, typer.Option("--set", help="dev (eval/) or example (eval-example/).")
    ] = "dev",
    model_key: AnswerModelOption = DEFAULT_MODEL,
    mode: AnswerModeOption = "hybrid",
    k: AnswerKOption = 8,
    max_pi_calls: Annotated[
        int | None,
        typer.Option(
            "--max-pi-calls",
            min=agent_compare.WORST_CASE_PI_CALLS,
            help="Skip the remaining questions once the next one could exceed this many Pi calls.",
        ),
    ] = None,
) -> None:
    """Answer a question set with the chain and the agent graph, and compare them."""
    model = _model(model_key)
    questions, _chunk_count, database_url, ollama_url = _eval_inputs(eval_set, model)
    settings = get_settings()
    config = agent_compare.CompareConfig(
        eval_set, model.key, mode, k, settings.pi_model, max_pi_calls
    )
    worst = len(questions) * agent_compare.WORST_CASE_PI_CALLS
    typer.echo(
        f"{len(questions)} questions: at most {worst} Pi calls"
        + (f", capped at {max_pi_calls}" if max_pi_calls is not None else "")
    )

    def progress(result: agent_compare.QuestionComparison) -> None:
        typer.echo(
            f"{result.resolved.question.id}: chain {result.chain.outcome}"
            f" ({result.chain.pi_calls}), agent {result.agent.outcome}"
            f" ({result.agent.pi_calls}, rewrites {result.agent.rewrites})"
        )

    embeddings = ollama_embeddings.for_model(model, ollama_url)
    with store.searching(database_url, model, embeddings) as search_store:
        retriever = build_retriever(mode, search_store, k, conninfo=database_url, model=model)
        pi = _pi_model(settings)
        with _answer_errors():
            output = agent_compare.compare(
                questions,
                build_chain(retriever, pi),
                build_graph(retriever, pi),
                max_pi_calls=max_pi_calls,
                on_question=progress,
            )

    detail, report = agent_compare.write_outputs(
        config, output, results_dir=EVAL_RESULTS_DIR, docs_dir=EVAL_DOCS_DIR
    )
    for name, summary in (("chain", output.chain), ("agent", output.agent)):
        typer.echo(
            f"{name}: answered {summary.answered}, unknown {summary.unknown},"
            f" rejected {summary.rejected}, Pi calls {summary.pi_calls_total}"
        )
    if output.skipped:
        typer.echo(f"skipped (Pi call cap): {', '.join(output.skipped)}")
    for path in (detail, report):
        typer.echo(f"wrote {_shown(path)}")


def _shown(path: Path) -> Path:
    """`path` relative to the repository root, if it is inside it."""
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


# --- hnsw ---------------------------------------------------------------------

ModelOption = Annotated[str, typer.Option("--model", help=f"One of: {', '.join(MODELS)}.")]


def _model(model_key: str) -> EmbeddingModel:
    model = MODELS.get(model_key)
    if model is None:
        _fail(f"unknown model {model_key!r}. Choose one of: {', '.join(MODELS)}.")
    return model


def _require_table(database_url: str, model: EmbeddingModel) -> None:
    if not store.table_exists(database_url, model.table_name):
        _fail(f"no table {model.table_name}. Run `profile-rag index --model {model.key}` first.")


def _require_traceable(database_url: str, model: EmbeddingModel) -> None:
    """With tracing on, exit unless the model's table holds exactly corpus-example/.

    CORPUS_DIR=corpus-example passes the startup check even when the table
    was indexed from the real corpus; this checks the data itself.
    """
    if not tracing.tracing_enabled():
        return
    root = tracing.EXAMPLE_CORPUS
    try:
        loaded = documents.load_documents(_load(root), root)
    except documents.ApprovedSnapshotError as error:
        _fail(str(error))
    chunks = splitting.split_documents(
        loaded, chunk_words=DEFAULT_CHUNK_WORDS, overlap_words=DEFAULT_OVERLAP_WORDS
    )
    if store.chunk_ids(database_url, model) != {c.metadata["chunk_id"] for c in chunks}:
        _fail(
            f"LANGSMITH_TRACING=true, but {model.table_name} does not hold exactly the"
            " example corpus. Traces may only come from corpus-example/: run"
            f" `make index CORPUS_DIR=corpus-example MODEL={model.key}`, or turn tracing off."
        )


def _describe_index(status: hnsw.HnswStatus | None, model: EmbeddingModel) -> str:
    if status is None:
        return f"{model.table_name}: no HNSW index"
    valid = "valid" if status.valid else "INVALID"
    return (
        f"{status.name}: {valid}, m {status.m}, ef_construction {status.ef_construction},"
        f" {status.size_bytes} bytes"
    )


@hnsw_app.command("apply")
def hnsw_apply_command(
    model_key: ModelOption = DEFAULT_MODEL,
    m: Annotated[int, typer.Option("--m", min=2, max=100, help="Graph links per node.")] = (
        hnsw.DEFAULT_M
    ),
    ef_construction: Annotated[
        int,
        typer.Option("--ef-construction", min=4, max=1000, help="Candidates kept while building."),
    ] = hnsw.DEFAULT_EF_CONSTRUCTION,
) -> None:
    """Create the HNSW index on the model's table. Does nothing if it exists with these settings."""
    model = _model(model_key)
    settings = get_settings()
    database_url = str(settings.database_url)
    _require_table(database_url, model)
    if ef_construction < 2 * m:
        _fail(f"--ef-construction must be at least 2 * --m ({2 * m}).")
    embeddings = ollama_embeddings.for_model(model, settings.ollama_url)
    with store.searching(database_url, model, embeddings) as search_store:
        result = hnsw.apply_hnsw(
            search_store, model, conninfo=database_url, m=m, ef_construction=ef_construction
        )
    typer.echo(f"{result}: {_describe_index(hnsw.hnsw_status(database_url, model), model)}")


@hnsw_app.command("drop")
def hnsw_drop_command(model_key: ModelOption = DEFAULT_MODEL) -> None:
    """Drop the HNSW index of the model's table. Does nothing if there is none."""
    model = _model(model_key)
    settings = get_settings()
    database_url = str(settings.database_url)
    _require_table(database_url, model)
    existed = hnsw.hnsw_status(database_url, model) is not None
    embeddings = ollama_embeddings.for_model(model, settings.ollama_url)
    with store.searching(database_url, model, embeddings) as search_store:
        hnsw.drop_hnsw(search_store, model)
    action = "dropped" if existed else "absent"
    typer.echo(f"{action}: {hnsw.hnsw_index_name(model)}")


@hnsw_app.command("status")
def hnsw_status_command(model_key: ModelOption = DEFAULT_MODEL) -> None:
    """Show the HNSW index of the model's table, if any."""
    model = _model(model_key)
    database_url = str(get_settings().database_url)
    _require_table(database_url, model)
    status = hnsw.hnsw_status(database_url, model)
    typer.echo(_describe_index(status, model))
    if status is not None:
        typer.echo(status.definition)


def _ef_values(text: str) -> list[int]:
    """`"10,20,40"` as `[10, 20, 40]`, each within pgvector's range."""
    values: list[int] = []
    for part in text.split(","):
        try:
            value = int(part.strip())
        except ValueError:
            _fail(f"--ef-search: {part.strip()!r} is not a whole number.")
        if not hnsw.EF_SEARCH_MIN <= value <= hnsw.EF_SEARCH_MAX:
            _fail(f"--ef-search: {value} is outside {hnsw.EF_SEARCH_MIN}..{hnsw.EF_SEARCH_MAX}.")
        values.append(value)
    return sorted(set(values))


@hnsw_app.command("compare")
def hnsw_compare_command(
    model_key: ModelOption = DEFAULT_MODEL,
    eval_set: Annotated[
        EvalSet,
        typer.Option("--set", help="dev or heldout (eval/), or example (eval-example/)."),
    ] = "dev",
    k: Annotated[int, typer.Option("-k", min=1, help="How many chunks each question gets.")] = 8,
    ef_search: Annotated[
        str, typer.Option("--ef-search", help="Comma-separated hnsw.ef_search values.")
    ] = "10,20,40,100",
) -> None:
    """Compare HNSW and exact dense results; write docs/eval/hnsw-<set>-<model>.md."""
    model = _model(model_key)
    ef_values = _ef_values(ef_search)
    questions, _, database_url, ollama_url = _eval_inputs(eval_set, model)
    if hnsw.hnsw_status(database_url, model) is None:
        _fail(f"no HNSW index on {model.table_name}. Run `profile-rag hnsw apply` first.")
    try:
        comparison = ann_compare.run_ann_compare(
            database_url,
            model,
            ollama_embeddings.for_model(model, ollama_url),
            questions,
            set_name=eval_set,
            chunks=len(store.chunk_ids(database_url, model)),
            k=k,
            ef_values=ef_values,
        )
    except EmbeddingError as error:
        _fail(f"a question: {error.reason}")
    except ValueError as error:
        _fail(str(error))
    path = ann_compare.write_ann_report(comparison, EVAL_DOCS_DIR)
    for plan, line in comparison.plans.items():
        typer.echo(f"plan ({plan}): {line}")
    exact = comparison.exact
    typer.echo(
        f"exact: recall@{k} {ann_compare.number(exact.recall)}, MRR {ann_compare.number(exact.mrr)}"
    )
    for result in comparison.by_ef:
        typer.echo(
            f"ef_search {result.ef_search}: overlap@{k} mean {result.mean:.3f},"
            f" min {result.min:.3f}, short {result.short},"
            f" recall@{k} {ann_compare.number(result.summary.recall)},"
            f" MRR {ann_compare.number(result.summary.mrr)}"
        )
    typer.echo(f"wrote {_shown(path)}")


if __name__ == "__main__":
    main()
