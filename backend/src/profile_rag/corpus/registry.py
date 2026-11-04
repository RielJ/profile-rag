"""The corpus registry: `sources.yaml` as typed, validated Python objects.

`sources.yaml` lists every candidate source and its review state. Nothing
reaches the database unless this file says it is approved, so a mistake here
(a typo in a state, an approval without a recorded hash) must stop the program
with a clear error instead of being ignored. That is what the models are for.

A corpus root is a folder with this layout (see `corpus/README.md`):

    sources.yaml    this registry
    approved/       snapshots that may be ingested
    private/        raw private material, never ingested
    reviews/        review decisions (written by `profile-rag corpus review`)

Paths inside an entry are relative to the corpus root and use `/`.

Example entry:

    sources:
      - id: resume-2026
        kind: resume
        title: Resume
        attribution: self_report
        visibility: public
        state: approved
        path: approved/resume-2026.md
        revision: "2026-10-01"
        content_sha256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
        approved_at: 2026-10-15 09:00:00+00:00
        approved_by: riel

Quote values that YAML would otherwise read as another type: `"2026-10-01"`
unquoted is a date, not a string.

Tests: backend/tests/test_registry.py.
"""

import hashlib
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator

# The allowed values match the CHECK constraints of the `sources` table
# (db/migrations/), plus the review state, which
# only exists in the registry.
Kind = Literal["resume", "github", "portfolio", "blog", "linkedin", "note"]
Visibility = Literal["public", "sanitized"]
Attribution = Literal["self_authored", "self_report", "project_doc", "third_party"]
State = Literal["candidate", "in_review", "approved", "rejected"]

REGISTRY_FILENAME = "sources.yaml"
APPROVED_DIR = "approved"
PRIVATE_DIR = "private"
REVIEWS_DIR = "reviews"


class RegistryError(Exception):
    """`sources.yaml` is missing, unreadable, or invalid.

    The message always names the file, so the CLI can print it as is.
    """


class SourceEntry(BaseModel):
    """One candidate source in `sources.yaml`.

    Unknown fields are an error, not ignored.

    Field rules:

    - `id`: lowercase letters and digits in groups joined by single hyphens,
      for example `resume-2026` or `blog-postgres-queues`. It becomes the
      primary key of the `sources` table.
    - `title`: not empty.
    - `url`: the public link shown to visitors. None for private notes.
    - `origin`: where `corpus snapshot` copies or downloads the source from
      (a local path or an http(s) URL). Never shown to visitors.
    - `path`: the snapshot to ingest, relative to the corpus root, naming a
      file inside `approved/` (for example `approved/resume-2026.md`). It
      must not be absolute and must contain no `..` part, so an entry can
      never point into `private/` or outside the corpus. `approved-old/x.md`
      does not count as inside `approved/`.
    - `revision`: commit SHA, snapshot date, or version of the content.
    - `content_sha256`: SHA-256 of the file at `path` when it was approved,
      as 64 lowercase hexadecimal characters.
    - `approved_at`: when it was approved, with a time zone.
    - `rejection_reason`: why it was rejected.
    - `notes`: free text for review notes (YAML comments are lost when the
      CLI rewrites the file; this field is kept).

    State rules (docs/ARCHITECTURE.md, "Corpus policy"):

    - `approved` requires `path`, `revision`, `content_sha256`,
      `approved_at`, and `approved_by`; `approved_by` must not be blank.
    - `rejected` requires a `rejection_reason` that is not blank (empty or
      only whitespace).
    - `approved_at` and `approved_by` are set only when the state is
      `approved`.
    - `rejection_reason` is set only when the state is `rejected`.

    Every error message from a state rule names the source `id` and the
    field or fields concerned, for example
    `resume-2026: approved sources need content_sha256`.
    """

    model_config = ConfigDict(strict=True, extra="forbid")
    id: str
    kind: Kind
    title: str
    attribution: Attribution
    visibility: Visibility
    state: State
    url: str | None = None
    origin: str | None = None
    path: str | None = None
    revision: str | None = None
    content_sha256: str | None = None
    approved_at: AwareDatetime | None = None
    approved_by: str | None = None
    rejection_reason: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def check_rules(self) -> Self:
        """Enforce the state rules in the class docstring. Return `self`."""

        if not bool(re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$").fullmatch(self.id)):
            raise ValueError(f"{self.id}: Invalid id.")

        if not self.title.strip():
            raise ValueError(f"{self.id}: Invalid title.")

        if self.path is not None and not bool(
            re.compile(
                r"^(?!.*(?:^|/)\.\.(?:/|$))approved/(?:[^/\n]+/)*[^/\n]+\.[^/\n.]+$"
            ).fullmatch(self.path)
        ):
            raise ValueError(f"{self.id}: Invalid path.")

        if self.content_sha256 is not None and not bool(
            re.compile(r"^[a-f0-9]{64}$").fullmatch(self.content_sha256)
        ):
            raise ValueError(f"{self.id}: Invalid content_sha256.")

        if self.state == "approved":

            def is_missing(v: object) -> bool:
                return v is None or (isinstance(v, str) and not v.strip())

            fields_to_check = [
                self.path,
                self.revision,
                self.content_sha256,
                self.approved_at,
                self.approved_by,
            ]

            if any(is_missing(f) for f in fields_to_check):
                raise ValueError(
                    f"{self.id}: approved entries require path, revision,"
                    " content_sha256, approved_at, and approved_by."
                )

        if self.state == "rejected" and (
            self.rejection_reason is None or not self.rejection_reason.strip()
        ):
            raise ValueError(f"{self.id}: rejected entries requires rejection_reason.")

        if self.state != "approved" and (
            (self.approved_by is not None and self.approved_by.strip())
            or self.approved_at is not None
        ):
            raise ValueError(
                f"{self.id}: approved_by and approved_at should only be set when state is approved."
            )
        if self.state != "rejected" and (
            self.rejection_reason is not None and self.rejection_reason.strip()
        ):
            raise ValueError(
                f"{self.id}: rejection_reason should only be set when state is rejected."
            )

        return self


class Registry(BaseModel):
    """The whole `sources.yaml` file: a mapping with one key, `sources`.

    Unknown top-level keys are an error. No two entries share an `id`; the
    error message names every duplicated ID.
    """

    sources: list[SourceEntry] = Field(default_factory=list[SourceEntry])
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def check_unique_ids(self) -> Self:
        """Raise ValueError naming the duplicated IDs, if any. Return `self`."""
        seen = set()
        duplicates = set()
        for item in self.sources:
            if item.id in seen:
                duplicates.add(item.id)
            else:
                seen.add(item.id)

        if len(duplicates):
            raise ValueError(f"Duplicate {duplicates} found in the sources!")
        return self

    def get(self, source_id: str) -> SourceEntry | None:
        """Return the entry with this ID, or None."""
        return next((entry for entry in self.sources if entry.id == source_id), None)


def load_registry(path: Path) -> Registry:
    """Read and validate the registry file at `path`.

    - An empty file is a registry with no sources.
    - A missing or unreadable file, invalid YAML, or content that fails
      validation raises `RegistryError`. The message starts with `path` and
      includes the underlying error, so the user sees which file and which
      field or rule is wrong.
    """
    try:
        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise RegistryError(f"{path}: {error}") from error

    if data is None:
        return Registry(sources=[])

    if not isinstance(data, dict):
        raise RegistryError(f"{path}: Invalid yaml contents")

    try:
        return Registry.model_validate(data)
    except ValidationError as error:
        raise RegistryError(f"{path}: {error}") from error


_HEADER = """\
# Corpus registry: every candidate source and its review state.
# Format: backend/src/profile_rag/corpus/registry.py. Policy: docs/ARCHITECTURE.md,
# "Corpus policy". `profile-rag corpus ...` rewrites this file, so comments
# are not kept; use an entry's `notes` field instead.
"""


def save_registry(registry: Registry, path: Path) -> None:
    """Write `registry` to `path` as YAML.

    Fields that are None are left out. The file is written to a temporary
    name and then renamed over `path`, so a crash never leaves half a file.
    """
    data = registry.model_dump(mode="python", exclude_none=True)
    text = _HEADER + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of the file's bytes as 64 lowercase hex characters.

    Raises OSError (for example FileNotFoundError) if the file can't be read.
    """
    sha_256 = hashlib.sha256()
    with path.open("rb") as file:
        for byte_block in iter(lambda: file.read(65536), b""):
            sha_256.update(byte_block)
    return sha_256.hexdigest()


def find_stale(entries: Iterable[SourceEntry], root: Path) -> list[str]:
    """Return the IDs of approved entries whose file no longer matches.

    `root` is the corpus root; each entry's `path` is relative to it. An
    approved entry is stale when the file at `root / path` is missing, or
    when its SHA-256 differs from `content_sha256`. Entries in other states
    are never stale. IDs are returned in the order of `entries`.
    """
    staleIds = []
    for entry in entries:
        if entry.state != "approved":
            continue
        if entry.path is None or entry.content_sha256 is None:
            staleIds.append(entry.id)
            continue
        entryPath = Path(root / entry.path.lstrip("/"))
        if not entryPath.is_file() or file_sha256(entryPath) != entry.content_sha256:
            staleIds.append(entry.id)
    return staleIds
