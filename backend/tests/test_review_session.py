"""Tests for corpus/review_session.py: the queue and the interactive review.

Each test builds a corpus in tmp_path and drives a Session with scripted
answers in place of the terminal.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from profile_rag.corpus import registry, review_session, scan
from profile_rag.corpus.registry import Registry, SourceEntry
from profile_rag.corpus.review_session import CHECKLIST, ReviewIO, Session

NOW = datetime(2026, 11, 5, 8, 0, tzinfo=UTC)
TOKEN = "ghp_" + "Q" * 36
YES = ["y"] * len(CHECKLIST)


type Answer = str | type[BaseException] | Callable[[], str]


@dataclass
class FakeTerminal:
    """Answers prompts from a script: a string is typed, an exception class is
    raised (EOFError, KeyboardInterrupt), and a function runs then returns the
    answer. Each call to the editor runs the next function in `edits`."""

    answers: list[Answer]
    edits: list[Callable[[Path], object]] = field(default_factory=list[Callable[[Path], object]])
    out: list[str] = field(default_factory=list[str])
    paged: list[str] = field(default_factory=list[str])
    prompts: list[str] = field(default_factory=list[str])

    def prompt(self, text: str) -> str:
        self.prompts.append(text)
        if not self.answers:
            raise AssertionError(f"no scripted answer for {text!r}")
        answer = self.answers.pop(0)
        if isinstance(answer, str):
            return answer
        if isinstance(answer, type):
            raise answer()
        return answer()

    def edit(self, path: Path) -> None:
        self.edits.pop(0)(path)

    def io(self) -> ReviewIO:
        return ReviewIO(
            prompt=self.prompt, echo=self.out.append, show=self.paged.append, edit=self.edit
        )

    @property
    def text(self) -> str:
        return "\n".join(self.out)


def entry(source_id: str, state: str = "in_review", **extra: object) -> SourceEntry:
    return SourceEntry.model_validate(
        {
            "id": source_id,
            "kind": "note",
            "title": source_id.title(),
            "attribution": "self_report",
            "visibility": "public",
            "state": state,
            "path": f"approved/{source_id}.md",
            "revision": "2026-11-01",
            **extra,
        }
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "corpus"


def make_corpus(root: Path, files: dict[str, str], states: dict[str, str] | None = None) -> None:
    """Write each file to approved/ and an entry for it; `states` defaults to in_review.
    Approved entries get the hash of the file as written."""
    (root / "approved").mkdir(parents=True)
    entries: list[SourceEntry] = []
    for source_id, text in files.items():
        file = root / "approved" / f"{source_id}.md"
        file.write_text(text)
        state = (states or {}).get(source_id, "in_review")
        if state == "approved":
            entries.append(
                entry(
                    source_id,
                    "approved",
                    content_sha256=registry.file_sha256(file),
                    approved_at=NOW,
                    approved_by="earlier",
                )
            )
        elif state == "rejected":
            entries.append(entry(source_id, state, rejection_reason="earlier"))
        else:
            entries.append(entry(source_id, state))
    registry.save_registry(Registry(sources=entries), root / "sources.yaml")


def load(root: Path) -> dict[str, SourceEntry]:
    return {e.id: e for e in registry.load_registry(root / "sources.yaml").sources}


def run(
    root: Path, terminal: FakeTerminal, ids: list[str] | None = None, *, guided: bool = False
) -> review_session.Summary:
    """Most tests drive the menu directly, so the guided start is off unless asked for."""
    current = registry.load_registry(root / "sources.yaml").sources
    queue = [e for e in current if ids is None or e.id in ids]
    session = Session(
        root=root, io=terminal.io(), reviewer="tester", clock=lambda: NOW, guided=guided
    )
    return session.run(queue)


def record(root: Path, source_id: str) -> str:
    return (root / "reviews" / f"{source_id}.md").read_text()


# --- the queue ------------------------------------------------------------------


def test_queue_is_in_review_then_stale_in_registry_order(root: Path) -> None:
    make_corpus(
        root,
        {"stale-a": "a\n", "fresh": "f\n", "wait-b": "b\n", "gone": "x\n", "wait-c": "c\n"},
        {"stale-a": "approved", "fresh": "approved", "gone": "rejected"},
    )
    (root / "approved" / "stale-a.md").write_text("changed\n")
    sources = registry.load_registry(root / "sources.yaml").sources

    queue = review_session.build_queue(sources, root, include_all=False)

    assert [e.id for e in queue.sources] == ["wait-b", "wait-c", "stale-a"]
    assert queue.missing == []


def test_queue_with_all_adds_approved_sources_that_are_not_stale(root: Path) -> None:
    make_corpus(
        root,
        {"fresh": "f\n", "stale-a": "a\n", "wait-b": "b\n"},
        {"fresh": "approved", "stale-a": "approved"},
    )
    (root / "approved" / "stale-a.md").write_text("changed\n")
    sources = registry.load_registry(root / "sources.yaml").sources

    queue = review_session.build_queue(sources, root, include_all=True)

    assert [e.id for e in queue.sources] == ["wait-b", "stale-a", "fresh"]


def test_queue_reports_sources_without_a_file(root: Path) -> None:
    make_corpus(root, {"wait-a": "a\n", "stale-b": "b\n"}, {"stale-b": "approved"})
    (root / "approved" / "wait-a.md").unlink()
    (root / "approved" / "stale-b.md").unlink()
    sources = registry.load_registry(root / "sources.yaml").sources

    queue = review_session.build_queue(sources, root, include_all=False)

    assert queue.sources == []
    assert queue.missing == ["wait-a", "stale-b"]


# --- header and view ------------------------------------------------------------


def test_header_shows_position_metadata_and_findings_without_the_match(root: Path) -> None:
    make_corpus(root, {"note-a": f"# Note\n\ntoken {TOKEN}\nmail me@example.com\n"})
    terminal = FakeTerminal(["s"])

    run(root, terminal)

    assert "[1/1] note-a (in_review)" in terminal.text
    assert "kind: note  attribution: self_report  visibility: public" in terminal.text
    assert "findings: github_token 3:7, email 4:6" in terminal.text
    assert TOKEN not in terminal.text
    assert "me@example.com" not in terminal.text


def test_view_numbers_lines_marks_findings_and_masks_secrets(root: Path) -> None:
    make_corpus(root, {"note-a": f"# Note\n\ntoken {TOKEN} end\nmail me@example.com\n"})
    terminal = FakeTerminal(["v", "s"])

    run(root, terminal)

    [view] = terminal.paged
    assert "  1 | # Note" in view
    assert f"! 3 | token {TOKEN[:4]}… end    <- github_token" in view
    assert "! 4 | mail me@example.com    <- email" in view
    assert TOKEN not in view


def test_render_view_masks_each_secret_on_a_line() -> None:
    other = "AKIA" + "B" * 16
    text = f"a {TOKEN} b {other}\n"
    findings = scan.scan_text(text)

    view = review_session.render_view(text, findings)

    assert "! 1 | a ghp_… b AKIA…    <- github_token, aws_access_key" in view
    assert TOKEN not in view
    assert other not in view


# --- guided start ------------------------------------------------------------------


def test_guided_shows_the_file_then_asks_the_checklist_then_approves(root: Path) -> None:
    make_corpus(root, {"note-a": "Line one.\n"})
    terminal = FakeTerminal([*YES, "y"])

    summary = run(root, terminal, guided=True)

    assert len(terminal.paged) == 1
    assert "Line one." in terminal.paged[0]
    assert terminal.prompts == [
        *(f"{n}/{len(CHECKLIST)} {q} [y/n]" for n, q in enumerate(CHECKLIST, start=1)),
        "Approve note-a? [y/n]",
    ]
    assert summary.approved == 1
    assert load(root)["note-a"].state == "approved"


def test_guided_a_no_leads_to_the_menu(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    terminal = FakeTerminal(["y", "n", "s"])

    summary = run(root, terminal, guided=True)

    assert len(terminal.paged) == 1
    assert terminal.prompts[2].startswith("[v]iew [e]dit [c]hecklist [r]eject")
    assert summary.skipped == 1
    assert load(root)["note-a"].state == "in_review"


def test_guided_declining_approval_leaves_approve_in_the_menu(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    terminal = FakeTerminal([*YES, "n", "a"])

    summary = run(root, terminal, guided=True)

    assert terminal.prompts[len(CHECKLIST) + 1].startswith("[v]iew [e]dit [c]hecklist [a]pprove")
    assert summary.approved == 1


def test_guided_an_edit_that_changes_the_file_starts_the_walkthrough_again(root: Path) -> None:
    make_corpus(root, {"note-a": f"token {TOKEN}\n"})
    terminal = FakeTerminal(
        ["y", "n", "e", *YES, "y"], edits=[lambda path: path.write_text("clean\n")]
    )

    summary = run(root, terminal, guided=True)

    assert len(terminal.paged) == 2
    assert "clean" in terminal.paged[1]
    assert TOKEN not in "\n".join(terminal.paged)
    assert summary.approved == 1
    assert "- edited during review: yes" in record(root, "note-a")


def test_guided_an_edit_with_no_change_returns_to_the_menu(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    terminal = FakeTerminal(["n", "e", "s"], edits=[lambda path: None])

    summary = run(root, terminal, guided=True)

    assert len(terminal.paged) == 1
    assert "No change." in terminal.text
    assert summary.skipped == 1


# --- keys -------------------------------------------------------------------------


def test_unknown_key_asks_again(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    terminal = FakeTerminal(["x", "", "Skip"])

    summary = run(root, terminal)

    assert terminal.text.count("Type one of: v, e, c, a, r, s, q.") == 2
    assert summary.skipped == 1


def test_skip_moves_on_without_a_change(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n", "note-b": "b\n"})
    terminal = FakeTerminal(["s", "s"])

    summary = run(root, terminal)

    assert summary == review_session.Summary(skipped=2)
    assert "[2/2] note-b" in terminal.text
    assert {e.state for e in load(root).values()} == {"in_review"}
    assert not (root / "reviews").exists()


# --- checklist and approve --------------------------------------------------------


def test_approve_is_not_offered_before_the_checklist(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    terminal = FakeTerminal(["c", *YES, "a"])

    run(root, terminal)

    assert "[a]pprove" not in terminal.prompts[0]
    assert "[a]pprove" in terminal.prompts[-1]


def test_approve_after_every_yes_records_hash_and_answers(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    terminal = FakeTerminal(["c", *YES, "a"])

    summary = run(root, terminal)

    saved = load(root)["note-a"]
    assert summary.approved == 1
    assert saved.state == "approved"
    assert saved.content_sha256 == registry.file_sha256(root / "approved" / "note-a.md")
    assert saved.approved_by == "tester"
    audit = record(root, "note-a")
    assert f"- sha256: {saved.content_sha256}" in audit
    assert "- edited during review: no" in audit
    for question in CHECKLIST:
        assert f"  - yes: {question}" in audit


def test_approve_before_the_checklist_runs_it(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    terminal = FakeTerminal(["a", *YES])

    run(root, terminal)

    assert "Approval needs a yes to every checklist item" in terminal.text
    assert load(root)["note-a"].state == "approved"


def test_a_no_ends_the_checklist_and_blocks_approval(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    # The second item is answered no; approving then asks the checklist again.
    terminal = FakeTerminal(["c", "y", "n", "a", "y", "y", "n", "q"])

    run(root, terminal)

    assert "Answered no. Edit the file ([e]) or reject the source ([r])." in terminal.text
    checklist_prompts = [p for p in terminal.prompts if "[y/n]" in p]
    assert [p.split(" ")[0] for p in checklist_prompts] == ["1/5", "2/5", "1/5", "2/5", "3/5"]
    assert load(root)["note-a"].state == "in_review"


def test_checklist_asks_again_after_an_answer_that_is_not_y_or_n(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    terminal = FakeTerminal(["c", "maybe", *YES, "a"])

    run(root, terminal)

    assert "Type y or n." in terminal.text
    assert load(root)["note-a"].state == "approved"


def test_every_checklist_item_is_phrased_so_yes_is_safe() -> None:
    assert not any(question.startswith("Does it contain") for question in CHECKLIST)
    assert any("free of secrets" in question for question in CHECKLIST)


def test_a_file_changed_after_the_checklist_is_not_approved(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    file = root / "approved" / "note-a.md"

    def edit_elsewhere() -> str:
        file.write_text(f"a\ntoken {TOKEN}\n")  # another program edits the file
        return "a"

    terminal = FakeTerminal(["c", *YES, edit_elsewhere, "q"])

    run(root, terminal)

    assert "Not approved: the file changed after the checklist was answered." in terminal.text
    assert "findings: github_token 2:7" in terminal.text
    assert load(root)["note-a"].state == "in_review"


# --- edit -------------------------------------------------------------------------


def test_edit_that_changes_the_content_clears_the_checklist(root: Path) -> None:
    make_corpus(root, {"note-a": f"a\ntoken {TOKEN}\n"})
    file = root / "approved" / "note-a.md"
    terminal = FakeTerminal(
        ["c", *YES, "e", "a", *YES],
        edits=[lambda path: path.write_text("a\n")],
    )

    run(root, terminal)

    assert "Content changed" in terminal.text
    assert "findings: none." in terminal.text
    assert "Checklist answers cleared" in terminal.text
    # The second approve had to run the checklist again.
    assert "Approval needs a yes to every checklist item" in terminal.text
    saved = load(root)["note-a"]
    assert saved.state == "approved"
    assert saved.content_sha256 == registry.file_sha256(file)
    assert "- edited during review: yes" in record(root, "note-a")


def test_edit_without_a_change_keeps_the_checklist(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    terminal = FakeTerminal(["c", *YES, "e", "a"], edits=[lambda path: None])

    run(root, terminal)

    assert "No change." in terminal.text
    assert "- edited during review: no" in record(root, "note-a")


def test_a_failed_editor_is_reported(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})

    def fail(path: Path) -> None:
        raise review_session.EditorFailed("vi: not found")

    terminal = FakeTerminal(["e", "q"], edits=[fail])

    run(root, terminal)

    assert "The editor failed: vi: not found" in terminal.text


# --- reject -----------------------------------------------------------------------


def test_reject_needs_a_reason_that_is_not_blank(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n"})
    terminal = FakeTerminal(["r", "", "  ", "not mine"])

    summary = run(root, terminal)

    assert terminal.text.count("A rejection needs a reason.") == 2
    assert summary.rejected == 1
    saved = load(root)["note-a"]
    assert saved.state == "rejected"
    assert saved.rejection_reason == "not mine"
    audit = record(root, "note-a")
    assert "- reason: not mine" in audit
    assert "- checklist: not answered" in audit


def test_reject_records_the_answers_given(root: Path) -> None:
    make_corpus(root, {"note-a": f"token {TOKEN}\n"})
    terminal = FakeTerminal(["c", "y", "n", "r", "has a token"])

    run(root, terminal)

    audit = record(root, "note-a")
    assert f"  - yes: {CHECKLIST[0]}" in audit
    assert f"  - no: {CHECKLIST[1]}" in audit
    assert CHECKLIST[2] not in audit
    assert "github_token 1:7" in audit
    assert TOKEN not in audit
    assert TOKEN[4:] not in audit


# --- stopping ---------------------------------------------------------------------


@pytest.mark.parametrize("stop", ["q", EOFError, KeyboardInterrupt])
def test_stopping_keeps_earlier_decisions(root: Path, stop: Answer) -> None:
    make_corpus(root, {"note-a": "a\n", "note-b": "b\n", "note-c": "c\n"})
    # note-b: one checklist answer of each, then stop at the menu.
    terminal = FakeTerminal(["r", "old", "c", "y", "n", stop])

    summary = run(root, terminal)

    assert "Decisions made so far are saved." in terminal.text
    assert summary == review_session.Summary(rejected=1)
    states = {source_id: e.state for source_id, e in load(root).items()}
    assert states == {"note-a": "rejected", "note-b": "in_review", "note-c": "in_review"}
    yaml.safe_load((root / "sources.yaml").read_text())
    assert not list(root.glob(".sources.yaml*"))


@pytest.mark.parametrize(
    ("failure", "said"),
    [
        (KeyboardInterrupt, "Stopped (interrupted)."),
        (PermissionError, "Not saved: note-a: could not write sources.yaml"),
    ],
)
def test_a_failed_save_after_the_audit_record_is_marked_in_it(
    root: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException], said: str
) -> None:
    make_corpus(root, {"note-a": f"token {TOKEN}\n"})
    before = (root / "sources.yaml").read_text()
    seen_in_record: list[str] = []

    def fail(src: object, dst: object) -> None:
        # The record is already written when sources.yaml is replaced.
        seen_in_record.append(record(root, "note-a"))
        raise failure(f"mentions {TOKEN}")

    monkeypatch.setattr(os, "replace", fail)
    terminal = FakeTerminal(["r", "has a token", "q"])

    run(root, terminal)

    assert said in terminal.text
    assert (root / "sources.yaml").read_text() == before
    assert load(root)["note-a"].state == "in_review"
    assert " reject\n" in seen_in_record[0]
    audit = record(root, "note-a")
    assert audit.startswith(seen_in_record[0])
    assert audit.endswith(f"- decision not saved: {failure.__name__}\n\n")
    assert TOKEN not in audit
    assert TOKEN[4:] not in audit


def test_a_saved_decision_always_has_its_record(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_corpus(root, {"note-a": "a\n"})
    replace = os.replace
    record_when_saved: list[str] = []

    def spy(src: str, dst: str) -> None:
        record_when_saved.append(record(root, "note-a"))
        replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)

    run(root, FakeTerminal(["r", "why"]))

    assert " reject\n" in record_when_saved[0]
    assert "not saved" not in record(root, "note-a")
    assert load(root)["note-a"].state == "rejected"


def test_summary_counts_what_remains(root: Path) -> None:
    make_corpus(
        root,
        {"note-a": "a\n", "note-b": "b\n", "stale-c": "c\n"},
        {"stale-c": "approved"},
    )
    (root / "approved" / "stale-c.md").write_text("changed\n")
    terminal = FakeTerminal(["c", *YES, "a", "s"])

    summary = run(root, terminal, ids=["note-a", "note-b"])
    review_session.print_summary(terminal.io(), root, summary)

    assert terminal.out[-1] == (
        "Approved 1, rejected 0, skipped 1. Remaining: in_review 1, stale 1."
    )


def test_a_decision_that_cannot_be_saved_is_reported(root: Path) -> None:
    make_corpus(root, {"note-a": "a\n", "note-b": "b\n"})
    queue = registry.load_registry(root / "sources.yaml").sources
    # Another process removes note-a from sources.yaml during the review.
    registry.save_registry(Registry(sources=[queue[1]]), root / "sources.yaml")
    terminal = FakeTerminal(["r", "why", "s"])

    session = Session(
        root=root, io=terminal.io(), reviewer="tester", clock=lambda: NOW, guided=False
    )
    summary = session.run(queue)

    assert "Not saved: note-a is no longer in sources.yaml" in terminal.text
    assert summary.skipped == 2


# --- flags ------------------------------------------------------------------------


def test_decide_by_flag_records_the_confirmation(root: Path) -> None:
    make_corpus(root, {"note-a": f"token {TOKEN}\n"})
    terminal = FakeTerminal([])

    review_session.decide_by_flag(
        root=root, entry=load(root)["note-a"], io=terminal.io(), reviewer="t", when=NOW, reject=None
    )

    assert load(root)["note-a"].state == "approved"
    audit = record(root, "note-a")
    assert "- checklist: confirmed by flag" in audit
    assert TOKEN not in audit
    assert TOKEN not in terminal.text
