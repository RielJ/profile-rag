"""Tests for corpus/registry.py.

Run them with:

    uv --directory backend run pytest tests/test_registry.py

Expected hashes are written out as constants (computed with
`printf 'abc' | shasum -a 256`), so these tests check the hashing against
an independent tool, not against the same code.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from profile_rag.corpus.registry import (
    Registry,
    RegistryError,
    SourceEntry,
    file_sha256,
    find_stale,
    load_registry,
    save_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "corpus-example"

# SHA-256 of a few byte strings, from `shasum -a 256`.
SHA_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SHA_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
SHA_A_LF = "87428fc522803d31065e7bce3cf03fe475096631e5e07bbd7a0fde60c4cf25c7"  # b"a\n"
SHA_A_CRLF = "8e4621379786ef42a4fec155cd525c291dd7db3c1fde3478522f4f61c03fd1bd"  # b"a\r\n"
SHA_NOT_UTF8 = "b3d510ef04275ca8e698e5b3cbb0ece3949ef9252f0cdc839e9ee347409a2209"  # b"\xff\xfe"

APPROVED_AT = datetime(2026, 10, 15, 9, 0, tzinfo=UTC)

CANDIDATE: dict[str, object] = {
    "id": "resume-2026",
    "kind": "resume",
    "title": "Resume",
    "attribution": "self_report",
    "visibility": "public",
    "state": "candidate",
}
IN_REVIEW = CANDIDATE | {
    "state": "in_review",
    "path": "approved/resume-2026.md",
    "revision": "2026-10-01",
}
APPROVED = IN_REVIEW | {
    "state": "approved",
    "content_sha256": SHA_ABC,
    "approved_at": APPROVED_AT,
    "approved_by": "riel",
}
REJECTED = CANDIDATE | {"state": "rejected", "rejection_reason": "Mostly a colleague's work."}


def without(data: dict[str, object], *names: str) -> dict[str, object]:
    return {key: value for key, value in data.items() if key not in names}


def rule_error(data: dict[str, object]) -> str:
    """Validate `data`, expect failure, and return the error messages."""
    with pytest.raises(ValidationError) as caught:
        SourceEntry.model_validate(data)
    return " | ".join(error["msg"] for error in caught.value.errors())


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --- one entry --------------------------------------------------------------


@pytest.mark.parametrize(
    "data", [CANDIDATE, IN_REVIEW, APPROVED, REJECTED], ids=lambda d: str(d["state"])
)
def test_valid_entry_in_each_state(data: dict[str, object]) -> None:
    entry = SourceEntry.model_validate(data)
    assert entry.model_dump(exclude_none=True) == data


def test_optional_fields_are_accepted() -> None:
    entry = SourceEntry.model_validate(
        APPROVED
        | {"url": "https://example.com/r", "origin": "~/cv.md", "notes": "Contact details removed."}
    )
    assert entry.url == "https://example.com/r"


def test_unknown_fields_are_rejected() -> None:
    # A misspelled key must not silently disappear.
    for extra in ({"atribution": "self_report"}, {"reviewed": True}):
        with pytest.raises(ValidationError) as caught:
            SourceEntry.model_validate(CANDIDATE | extra)
        assert [error["loc"] for error in caught.value.errors()] == [tuple(extra)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kind", "tweet"),
        ("visibility", "private"),
        ("attribution", "self"),
        ("state", "approve"),
        ("state", "Approved"),
    ],
)
def test_values_outside_the_allowed_set_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        SourceEntry.model_validate(CANDIDATE | {field: value})


@pytest.mark.parametrize("source_id", ["resume-2026", "a", "blog-postgres-queues", "x1", "2026"])
def test_valid_ids(source_id: str) -> None:
    assert SourceEntry.model_validate(CANDIDATE | {"id": source_id}).id == source_id


@pytest.mark.parametrize(
    "source_id",
    [
        "",
        "Resume-2026",
        "resume_2026",
        "resume 2026",
        "-resume",
        "resume-",
        "resume--2026",
        "approved/resume",
        "../resume",
        "résumé",
    ],
)
def test_invalid_ids(source_id: str) -> None:
    with pytest.raises(ValidationError):
        SourceEntry.model_validate(CANDIDATE | {"id": source_id})


def test_id_must_be_a_string() -> None:
    with pytest.raises(ValidationError):
        SourceEntry.model_validate(CANDIDATE | {"id": 2026})


def test_title_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        SourceEntry.model_validate(CANDIDATE | {"title": ""})


@pytest.mark.parametrize("path", ["approved/resume.md", "approved/github/lantern-queue/README.md"])
def test_paths_inside_approved_are_accepted(path: str) -> None:
    assert SourceEntry.model_validate(IN_REVIEW | {"path": path}).path == path


@pytest.mark.parametrize(
    "path",
    [
        "private/notes.md",
        "approved/../private/notes.md",
        "approved/sub/../../private/notes.md",
        "/approved/resume.md",
        "/etc/passwd",
        "resume.md",
        "approved-old/resume.md",
        "approved",
        "approved/",
        "../corpus/approved/resume.md",
    ],
)
def test_paths_outside_approved_are_rejected(path: str) -> None:
    with pytest.raises(ValidationError):
        SourceEntry.model_validate(IN_REVIEW | {"path": path})


@pytest.mark.parametrize(
    "digest",
    [SHA_ABC.upper(), SHA_ABC[:63], SHA_ABC + "0", "g" * 64, "", f"sha256:{SHA_ABC}"],
)
def test_content_sha256_must_be_64_lowercase_hex_characters(digest: str) -> None:
    with pytest.raises(ValidationError):
        SourceEntry.model_validate(APPROVED | {"content_sha256": digest})


def test_approved_at_needs_a_time_zone() -> None:
    with pytest.raises(ValidationError):
        SourceEntry.model_validate(APPROVED | {"approved_at": datetime(2026, 10, 15, 9, 0)})


# --- state rules ------------------------------------------------------------


@pytest.mark.parametrize(
    "missing", ["path", "revision", "content_sha256", "approved_at", "approved_by"]
)
def test_approved_needs_every_approval_field(missing: str) -> None:
    message = rule_error(without(APPROVED, missing))
    assert "resume-2026" in message
    assert missing in message


@pytest.mark.parametrize("blank", ["", "   "])
def test_approved_by_must_not_be_blank(blank: str) -> None:
    message = rule_error(APPROVED | {"approved_by": blank})
    assert "approved_by" in message


@pytest.mark.parametrize("reason", [None, "", "  \n"])
def test_rejected_needs_a_reason(reason: str | None) -> None:
    data = without(REJECTED, "rejection_reason")
    if reason is not None:
        data["rejection_reason"] = reason
    message = rule_error(data)
    assert "resume-2026" in message
    assert "rejection_reason" in message


@pytest.mark.parametrize("state", ["candidate", "in_review", "rejected"])
@pytest.mark.parametrize(
    ("field", "value"), [("approved_at", APPROVED_AT), ("approved_by", "riel")]
)
def test_approval_fields_only_on_approved_entries(state: str, field: str, value: object) -> None:
    base = REJECTED if state == "rejected" else CANDIDATE
    message = rule_error(base | {"state": state, field: value})
    assert "resume-2026" in message
    assert field in message


@pytest.mark.parametrize("base", [CANDIDATE, IN_REVIEW, APPROVED], ids=lambda d: str(d["state"]))
def test_rejection_reason_only_on_rejected_entries(base: dict[str, object]) -> None:
    message = rule_error(base | {"rejection_reason": "no"})
    assert "resume-2026" in message
    assert "rejection_reason" in message


def test_a_rejected_entry_may_keep_its_snapshot_path() -> None:
    entry = SourceEntry.model_validate(REJECTED | {"path": "approved/x.md", "revision": "r1"})
    assert entry.state == "rejected"


# --- the whole file ---------------------------------------------------------


def test_registry_can_be_empty() -> None:
    assert Registry.model_validate({}).sources == []
    assert Registry.model_validate({"sources": []}).sources == []


def test_registry_rejects_unknown_top_level_keys() -> None:
    with pytest.raises(ValidationError) as caught:
        Registry.model_validate({"version": 1, "sources": []})
    assert [error["loc"] for error in caught.value.errors()] == [("version",)]


def test_registry_rejects_duplicate_ids_and_names_them() -> None:
    sources = [
        CANDIDATE | {"id": "dup-one"},
        CANDIDATE | {"id": "unique-one"},
        CANDIDATE | {"id": "dup-two"},
        REJECTED | {"id": "dup-one"},
        CANDIDATE | {"id": "dup-two"},
    ]
    with pytest.raises(ValidationError) as caught:
        Registry.model_validate({"sources": sources})
    message = " | ".join(error["msg"] for error in caught.value.errors())
    assert "dup-one" in message
    assert "dup-two" in message
    assert "unique-one" not in message


def test_registry_get_finds_by_id() -> None:
    registry = Registry.model_validate({"sources": [CANDIDATE, REJECTED | {"id": "other"}]})
    found = registry.get("other")
    assert found is not None
    assert found.state == "rejected"
    assert registry.get("missing") is None


# --- loading sources.yaml ---------------------------------------------------

VALID_YAML = """\
# A comment
sources:
  - id: resume-2026
    kind: resume
    title: Resume
    attribution: self_report
    visibility: public
    state: approved
    path: approved/resume-2026.md
    revision: "2026-10-01"
    content_sha256: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    approved_at: 2026-10-15 09:00:00+00:00
    approved_by: riel
  - id: blog-queues
    kind: blog
    title: Queues in Postgres
    attribution: self_authored
    visibility: public
    state: candidate
"""


def test_load_a_valid_file(tmp_path: Path) -> None:
    registry = load_registry(write(tmp_path / "sources.yaml", VALID_YAML))

    assert [entry.id for entry in registry.sources] == ["resume-2026", "blog-queues"]
    resume = registry.sources[0]
    assert resume == SourceEntry.model_validate(APPROVED)
    assert resume.approved_at == APPROVED_AT


@pytest.mark.parametrize("text", ["", "\n", "# only a comment\n", "sources: []\n"])
def test_an_empty_file_has_no_sources(tmp_path: Path, text: str) -> None:
    assert load_registry(write(tmp_path / "sources.yaml", text)).sources == []


def test_a_missing_file_raises_registry_error(tmp_path: Path) -> None:
    path = tmp_path / "sources.yaml"
    with pytest.raises(RegistryError) as caught:
        load_registry(path)
    assert str(caught.value).startswith(str(path))


@pytest.mark.parametrize(
    "text",
    ["sources: [\n", "sources:\n  - id: a\n   kind: resume\n", "\tsources: []\n"],
    ids=["unclosed", "bad-indent", "tab"],
)
def test_invalid_yaml_raises_registry_error(tmp_path: Path, text: str) -> None:
    path = write(tmp_path / "sources.yaml", text)
    with pytest.raises(RegistryError) as caught:
        load_registry(path)
    assert str(caught.value).startswith(str(path))


@pytest.mark.parametrize(
    ("replace", "by", "named"),
    [
        ("kind: resume", "kind: cv", "kind"),
        ('revision: "2026-10-01"', "revision: 2026-10-01", "revision"),  # a date, not a string
        ("approved_by: riel", "approved_by: riel\n    reviewer: riel", "reviewer"),
        ("state: candidate", "state: rejected", "rejection_reason"),
    ],
)
def test_validation_errors_name_the_file_and_the_problem(
    tmp_path: Path, replace: str, by: str, named: str
) -> None:
    assert replace in VALID_YAML
    path = write(tmp_path / "sources.yaml", VALID_YAML.replace(replace, by, 1))
    with pytest.raises(RegistryError) as caught:
        load_registry(path)
    message = str(caught.value)
    assert message.startswith(str(path))
    assert named in message


@pytest.mark.parametrize(
    "text", ["- id: a\n", "just text\n", "sources: nope\n"], ids=["list", "scalar", "not-a-list"]
)
def test_the_wrong_shape_raises_registry_error(tmp_path: Path, text: str) -> None:
    with pytest.raises(RegistryError):
        load_registry(write(tmp_path / "sources.yaml", text))


def test_loading_never_runs_code_from_the_file(tmp_path: Path) -> None:
    # With yaml.load (not safe_load) this tag would create the marker file.
    marker = tmp_path / "pwned"
    text = f'sources: !!python/object/apply:builtins.open ["{marker}", "w"]\n'
    with pytest.raises(RegistryError):
        load_registry(write(tmp_path / "sources.yaml", text))
    assert not marker.exists()


def test_save_then_load_returns_the_same_registry(tmp_path: Path) -> None:
    registry = Registry.model_validate(
        {
            "sources": [
                APPROVED,
                IN_REVIEW | {"id": "repo-readme", "kind": "github", "revision": "0123456"},
                REJECTED | {"id": "old-post", "kind": "blog"},
                CANDIDATE | {"id": "linkedin", "kind": "linkedin", "notes": "Copy by hand."},
            ]
        }
    )
    path = tmp_path / "sources.yaml"
    save_registry(registry, path)
    assert load_registry(path) == registry


# --- hashing ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "digest"),
    [(b"", SHA_EMPTY), (b"abc", SHA_ABC), (b"a\n", SHA_A_LF), (b"a\r\n", SHA_A_CRLF)],
    ids=["empty", "abc", "lf", "crlf"],
)
def test_file_sha256_hashes_the_bytes(tmp_path: Path, content: bytes, digest: str) -> None:
    path = tmp_path / "file"
    path.write_bytes(content)
    assert file_sha256(path) == digest


def test_file_sha256_accepts_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    path = tmp_path / "binary"
    path.write_bytes(b"\xff\xfe")
    assert file_sha256(path) == SHA_NOT_UTF8


def test_file_sha256_of_a_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):  # any OSError subclass is fine
        file_sha256(tmp_path / "missing")


# --- staleness --------------------------------------------------------------


def approved_entry(source_id: str, path: str, digest: str = SHA_ABC) -> SourceEntry:
    return SourceEntry.model_validate(
        APPROVED | {"id": source_id, "path": path, "content_sha256": digest}
    )


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A corpus root. The working directory is elsewhere, so paths must use `root`."""
    corpus = tmp_path / "corpus"
    (corpus / "approved").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return corpus


def test_unchanged_files_are_not_stale(root: Path) -> None:
    (root / "approved" / "a.md").write_bytes(b"abc")
    assert find_stale([approved_entry("a", "approved/a.md")], root) == []


def test_a_changed_file_is_stale(root: Path) -> None:
    (root / "approved" / "a.md").write_bytes(b"abc, edited")
    assert find_stale([approved_entry("a", "approved/a.md")], root) == ["a"]


def test_a_line_ending_change_is_stale(root: Path) -> None:
    # Decoded as text, both files read as "a\n". As bytes they differ.
    (root / "approved" / "a.md").write_bytes(b"a\r\n")
    assert find_stale([approved_entry("a", "approved/a.md", SHA_A_LF)], root) == ["a"]


def test_a_missing_file_is_stale(root: Path) -> None:
    assert find_stale([approved_entry("gone", "approved/gone.md")], root) == ["gone"]


def test_only_approved_entries_can_be_stale(root: Path) -> None:
    entries = [
        SourceEntry.model_validate(CANDIDATE | {"id": "c"}),
        SourceEntry.model_validate(IN_REVIEW | {"id": "r", "path": "approved/missing.md"}),
        SourceEntry.model_validate(REJECTED | {"id": "x", "path": "approved/missing.md"}),
    ]
    assert find_stale(entries, root) == []


def test_stale_ids_keep_the_order_of_the_entries(root: Path) -> None:
    for name in ("one", "two", "three", "four"):
        (root / "approved" / f"{name}.md").write_bytes(b"abc")
    (root / "approved" / "three.md").write_bytes(b"changed")
    (root / "approved" / "one.md").unlink()
    entries = [
        approved_entry(name, f"approved/{name}.md") for name in ("four", "three", "two", "one")
    ]
    assert find_stale(iter(entries), root) == ["three", "one"]


# --- The committed example corpus -----------------------------------------


def test_the_example_corpus_is_valid_and_fresh() -> None:
    registry = load_registry(EXAMPLE_ROOT / "sources.yaml")

    states = {entry.state for entry in registry.sources}
    assert states >= {"approved", "candidate", "rejected"}
    approved = [entry for entry in registry.sources if entry.state == "approved"]
    assert {entry.kind for entry in approved} >= {"resume", "github", "blog"}
    assert find_stale(registry.sources, EXAMPLE_ROOT) == []

    on_disk = {
        p.relative_to(EXAMPLE_ROOT).as_posix() for p in (EXAMPLE_ROOT / "approved").iterdir()
    }
    assert on_disk == {entry.path for entry in approved}, "every approved/ file has an entry"
