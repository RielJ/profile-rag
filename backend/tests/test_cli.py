"""Tests for cli.py: the `profile-rag corpus` commands."""

from pathlib import Path
from typing import Self

import pytest
import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator
from typer.testing import CliRunner, Result

from profile_rag import cli
from profile_rag.corpus import registry, review_session
from profile_rag.corpus.registry import RegistryError

PROBE_ENTRY = {
    "id": "probe",
    "kind": "note",
    "title": "Probe",
    "attribution": "self_report",
    "visibility": "public",
    "state": "candidate",
}
PROBE_ARGS = (
    "probe",
    "--kind",
    "note",
    "--title",
    "Probe",
    "--attribution",
    "self_report",
    "--visibility",
    "public",
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return (tmp_path / "corpus").resolve()


@pytest.fixture
def resume(tmp_path: Path) -> Path:
    """A source file outside the corpus, as if exported to ~/Documents."""
    path = tmp_path / "outside" / "resume.md"
    path.parent.mkdir()
    path.write_text("# Resume\n\nBackend engineer.\n")
    return path


def run(root: Path, *args: str, input: str | None = None) -> Result:
    return CliRunner().invoke(cli.app, ["corpus", "--root", str(root), *args], input=input)


def ok(root: Path, *args: str, input: str | None = None) -> Result:
    result = run(root, *args, input=input)
    assert result.exit_code == 0, result.output
    return result


def add_resume(root: Path, origin: Path | None = None) -> None:
    extra = ["--from", str(origin)] if origin is not None else []
    ok(
        root,
        "add",
        "resume-2026",
        "--kind",
        "resume",
        "--title",
        "Resume",
        "--attribution",
        "self_report",
        "--visibility",
        "public",
        "--url",
        "https://example.com/resume",
        *extra,
    )


def entries(root: Path) -> dict[str, dict[str, object]]:
    """sources.yaml as plain data, read without the registry code."""
    data = yaml.safe_load((root / "sources.yaml").read_text())
    return {entry["id"]: entry for entry in data["sources"]}


def approve_resume(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")
    ok(root, "review", "resume-2026", "--approve", "--checklist-confirmed", "--by", "tester")


# --- add ----------------------------------------------------------------------


def test_add_creates_the_registry_with_a_candidate(root: Path) -> None:
    add_resume(root)
    assert entries(root)["resume-2026"] == {
        "id": "resume-2026",
        "kind": "resume",
        "title": "Resume",
        "attribution": "self_report",
        "visibility": "public",
        "state": "candidate",
        "url": "https://example.com/resume",
    }


def test_add_refuses_a_duplicate_id(root: Path) -> None:
    add_resume(root)
    result = run(root, "add", "resume-2026", "--kind", "resume", "--title", "Again",
                 "--attribution", "self_report", "--visibility", "public")  # fmt: skip
    assert result.exit_code == 2
    assert "already exists" in result.output


class _Shape(BaseModel):
    name: str = Field(pattern="^[a-z]+$")

    @model_validator(mode="after")
    def check_name(self) -> Self:
        if self.name == "bad":
            raise ValueError("demo: name must not be bad")
        return self


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"name": "Bad_Name"}, "  name: String should match pattern '^[a-z]+$'"),
        ({"name": "bad"}, "  demo: name must not be bad"),
    ],
)
def test_validation_errors_are_shown_without_the_input_dump(
    data: dict[str, object], expected: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        _Shape.model_validate(data)
    assert cli._describe(caught.value) == expected


def test_add_rejects_an_unknown_kind(root: Path) -> None:
    result = run(root, "add", "x", "--kind", "tweet", "--title", "X",
                 "--attribution", "self_report", "--visibility", "public")  # fmt: skip
    assert result.exit_code == 2
    assert "tweet" in result.output
    assert not (root / "sources.yaml").exists()


# --- snapshot -----------------------------------------------------------------


def test_snapshot_copies_the_file_and_moves_to_in_review(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    result = ok(root, "snapshot", "resume-2026", "--revision", "2026-10-01")

    entry = entries(root)["resume-2026"]
    assert entry["state"] == "in_review"
    assert entry["path"] == "approved/resume-2026.md"
    assert entry["revision"] == "2026-10-01"
    assert entry["origin"] == str(resume.resolve())
    assert (root / "approved" / "resume-2026.md").read_bytes() == resume.read_bytes()
    assert "corpus review resume-2026" in result.output


def test_snapshot_needs_an_origin(root: Path) -> None:
    add_resume(root)
    result = run(root, "snapshot", "resume-2026")
    assert result.exit_code == 2
    assert "--from" in result.output


def test_snapshot_reports_a_missing_file_in_one_line(root: Path, tmp_path: Path) -> None:
    add_resume(root, tmp_path / "nowhere.md")
    result = run(root, "snapshot", "resume-2026")
    assert result.exit_code == 2
    assert result.output.startswith("error:")
    assert "no such file" in result.output
    assert "Traceback" not in result.output


def test_private_snapshot_leaves_the_entry_alone(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    before = entries(root)
    result = ok(root, "snapshot", "resume-2026", "--private")
    assert entries(root) == before
    assert (root / "private" / "resume-2026.md").is_file()
    assert "sanitized" in result.output


def test_unknown_id_is_an_error(root: Path) -> None:
    add_resume(root)
    result = run(root, "snapshot", "nope", "--from", "x.md")
    assert result.exit_code == 2
    assert "no source with id 'nope'" in result.output


# --- review -------------------------------------------------------------------


YES_TO_ALL = "y\n" * len(review_session.CHECKLIST)


def test_approve_records_hash_reviewer_and_time(root: Path, resume: Path) -> None:
    approve_resume(root, resume)

    entry = entries(root)["resume-2026"]
    snapshot = root / "approved" / "resume-2026.md"
    assert entry["state"] == "approved"
    assert entry["content_sha256"] == registry.file_sha256(snapshot)
    assert entry["approved_by"] == "tester"
    assert entry["approved_at"] is not None
    record = (root / "reviews" / "resume-2026.md").read_text()
    assert " approve\n" in record
    assert "- reviewer: tester" in record
    assert "- checklist: confirmed by flag" in record


def test_approve_flag_needs_checklist_confirmed(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")

    result = run(root, "review", "resume-2026", "--approve", "--by", "tester")

    assert result.exit_code == 2
    assert "--checklist-confirmed" in result.output
    assert entries(root)["resume-2026"]["state"] == "in_review"
    assert not (root / "reviews").exists()


@pytest.mark.parametrize("flag", [["--approve", "--checklist-confirmed"], ["--reject", "why"]])
def test_decision_flags_need_an_id(root: Path, resume: Path, flag: list[str]) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")

    result = run(root, "review", *flag)

    assert result.exit_code == 2
    assert "need an ID" in result.output
    assert entries(root)["resume-2026"]["state"] == "in_review"


def test_review_shows_a_findings_summary_without_the_match(root: Path, resume: Path) -> None:
    resume.write_text("# Resume\n\nCall +63 917 555 0100.\n")
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")

    result = ok(root, "review", "resume-2026", input="n\ns\n")

    assert "[1/1] resume-2026 (in_review)" in result.output
    header = result.output.split("--- approved/resume-2026.md ---")[0]
    assert "findings: phone 3:6" in header
    # The header names the rule and position only. The file view that follows
    # shows personal data as-is (only secrets are masked) so you can judge it.
    assert "555 0100" not in header
    assert "Skipped; no change." in result.output
    assert "Approved 0, rejected 0, skipped 1. Remaining: in_review 1, stale 0." in result.output
    assert entries(root)["resume-2026"]["state"] == "in_review"
    assert not (root / "reviews").exists()


def test_review_record_never_contains_the_matched_secret(root: Path, resume: Path) -> None:
    token = "ghp_" + "Q" * 36
    resume.write_text(f"token {token}\n")
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")
    result = ok(root, "review", "resume-2026", "--reject", "contains a token", "--by", "tester")

    record = (root / "reviews" / "resume-2026.md").read_text()
    assert "github_token 1:7" in record
    assert token not in record
    assert token not in result.output


def test_interactive_approve_after_the_checklist(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")

    # Guided: the file, then the checklist, then "Approve resume-2026? [y/n]".
    ok(root, "review", "resume-2026", "--by", "tester", input=f"{YES_TO_ALL}y\n")

    assert entries(root)["resume-2026"]["state"] == "approved"
    record = (root / "reviews" / "resume-2026.md").read_text()
    for question in review_session.CHECKLIST:
        assert f"  - yes: {question}" in record
    assert "- edited during review: no" in record


def test_review_shows_the_file_before_the_checklist(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")

    result = ok(root, "review", "resume-2026", input="n\ns\n")

    start = result.output.index("--- approved/resume-2026.md ---")
    shown = result.output.index("3 | Backend engineer.")
    end = result.output.index("--- end of approved/resume-2026.md (3 lines) ---")
    assert start < shown < end < result.output.index("1/5 ")
    assert "Checklist for resume-2026:" in result.output


def test_interactive_view_prints_the_file_again(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")

    result = ok(root, "review", "resume-2026", input="n\nv\ns\n")

    assert result.output.count("3 | Backend engineer.") == 2


def test_interactive_edit_runs_the_editor(
    root: Path, resume: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")
    editor = tmp_path / "editor.sh"
    editor.write_text('#!/bin/sh\nprintf "# Resume\\n\\nEdited.\\n" > "$1"\n')
    editor.chmod(0o755)
    monkeypatch.setenv("VISUAL", str(editor))

    result = ok(root, "review", "resume-2026", "--by", "t", input=f"n\ne\n{YES_TO_ALL}y\n")

    assert "Content changed" in result.output
    snapshot = root / "approved" / "resume-2026.md"
    assert snapshot.read_text() == "# Resume\n\nEdited.\n"
    assert entries(root)["resume-2026"]["content_sha256"] == registry.file_sha256(snapshot)
    assert "- edited during review: yes" in (root / "reviews" / "resume-2026.md").read_text()


def test_end_of_input_stops_without_a_traceback(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")

    result = ok(root, "review", "resume-2026", input="y\n")

    assert "Stopped (interrupted). Decisions made so far are saved." in result.output
    assert entries(root)["resume-2026"]["state"] == "in_review"


def test_interactive_review_asks_again_after_an_unknown_key(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")

    result = ok(root, "review", "resume-2026", input="n\nx\n Skip \n")

    assert "Type one of: v, e, c, a, r, s, q." in result.output
    assert "Skipped; no change." in result.output


def test_interactive_reject_asks_again_for_a_blank_reason(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")

    result = ok(root, "review", "resume-2026", input="n\nr\n   \nnot my text\n")

    assert "A rejection needs a reason." in result.output
    assert entries(root)["resume-2026"]["rejection_reason"] == "not my text"


def test_reject_flag_needs_a_reason(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")

    result = run(root, "review", "resume-2026", "--reject", "  ")

    assert result.exit_code == 2
    assert "--reject needs a reason" in result.output
    assert entries(root)["resume-2026"]["state"] == "in_review"


def test_interactive_reject_asks_for_a_reason(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")
    ok(root, "review", "resume-2026", input="n\nreject\nnot my text\n")

    entry = entries(root)["resume-2026"]
    assert entry["state"] == "rejected"
    assert entry["rejection_reason"] == "not my text"
    assert "approved_at" not in entry
    assert "- reason: not my text" in (root / "reviews" / "resume-2026.md").read_text()


def test_review_needs_a_snapshot_first(root: Path) -> None:
    add_resume(root)
    result = run(root, "review", "resume-2026", "--approve", "--checklist-confirmed")
    assert result.exit_code == 2
    assert "corpus snapshot resume-2026" in result.output


def test_review_refuses_approve_and_reject_together(root: Path, resume: Path) -> None:
    add_resume(root, resume)
    ok(root, "snapshot", "resume-2026")
    result = run(root, "review", "resume-2026", "--approve", "--reject", "why")
    assert result.exit_code == 2
    assert "not both" in result.output


def test_review_without_an_id_walks_the_queue(root: Path, resume: Path) -> None:
    approve_resume(root, resume)
    ok(root, "add", *PROBE_ARGS)
    probe = resume.with_name("probe.md")
    probe.write_text("# Probe\n")
    ok(root, "snapshot", "probe", "--from", str(probe))
    (root / "approved" / "resume-2026.md").write_text("# Resume\n\nChanged.\n")

    result = ok(root, "review", "--by", "t", input="n\nr\nnot yet\nn\ns\n")

    assert "[1/2] probe (in_review)" in result.output
    assert "[2/2] resume-2026 (approved, but the file changed since approval)" in result.output
    assert "Approved 0, rejected 1, skipped 1. Remaining: in_review 0, stale 1." in result.output
    assert entries(root)["probe"]["state"] == "rejected"


def test_review_with_nothing_waiting(root: Path, resume: Path) -> None:
    approve_resume(root, resume)
    assert "Nothing to review." in ok(root, "review").output
    assert "[1/1] resume-2026 (approved)" in ok(root, "review", "--all", input="n\nq\n").output


# --- status and staleness -----------------------------------------------------


def test_status_lists_entries_and_counts(root: Path, resume: Path) -> None:
    approve_resume(root, resume)
    result = ok(root, "status")
    assert "1 sources" in result.output
    assert "candidate 0  in_review 0  approved 1  rejected 0" in result.output
    assert "resume-2026  approved" in result.output
    assert "STALE" not in result.output


def test_editing_an_approved_file_makes_it_stale(root: Path, resume: Path) -> None:
    approve_resume(root, resume)
    (root / "approved" / "resume-2026.md").write_text("# Resume\n\nEdited after review.\n")

    result = run(root, "status")

    assert result.exit_code == 1
    assert "resume-2026  approved   resume     self_report    public  STALE" in result.output


def test_a_missing_approved_file_is_stale(root: Path, resume: Path) -> None:
    approve_resume(root, resume)
    (root / "approved" / "resume-2026.md").unlink()
    assert run(root, "status").exit_code == 1


def test_snapshot_again_clears_the_approval(root: Path, resume: Path) -> None:
    approve_resume(root, resume)
    resume.write_text("# Resume\n\nNew job.\n")

    ok(root, "snapshot", "resume-2026")
    entry = entries(root)["resume-2026"]
    assert entry["state"] == "in_review"
    assert {"content_sha256", "approved_at", "approved_by"}.isdisjoint(entry)

    ok(root, "review", "resume-2026", "--approve", "--checklist-confirmed", "--by", "tester")
    assert ok(root, "status").exit_code == 0
    assert (root / "reviews" / "resume-2026.md").read_text().count(" approve\n") == 2


def test_status_lists_files_no_entry_points_to(root: Path, resume: Path) -> None:
    approve_resume(root, resume)
    (root / "approved" / "forgotten.md").write_text("never registered")
    result = ok(root, "status")
    assert "approved/forgotten.md" in result.output


def test_status_without_a_registry_explains_what_to_do(root: Path) -> None:
    result = run(root, "status")
    assert result.exit_code == 2
    assert "does not exist" in result.output
    assert "corpus add" in result.output


def test_registry_errors_are_printed_without_a_traceback(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root.mkdir()
    (root / "sources.yaml").write_text("sources: []\n")

    def broken(path: Path) -> registry.Registry:
        raise RegistryError(f"{path}: sources.0.kind: Input should be 'resume'")

    monkeypatch.setattr(registry, "load_registry", broken)
    result = run(root, "status")
    assert result.exit_code == 2
    expected = f"error: {root / 'sources.yaml'}: sources.0.kind: Input should be 'resume'\n"
    assert result.output == expected


def test_corpus_dir_environment_variable_selects_the_root(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root.mkdir()
    (root / "sources.yaml").write_text("sources: []\n")
    monkeypatch.setenv("CORPUS_DIR", str(root))
    result = CliRunner().invoke(cli.app, ["corpus", "status"])
    assert result.exit_code == 0, result.output
    assert f"{root / 'sources.yaml'}: 0 sources" in result.output
