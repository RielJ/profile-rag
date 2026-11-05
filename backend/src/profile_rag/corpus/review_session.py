"""The review of corpus sources behind `profile-rag corpus review`.

With no ID, `corpus review` walks a queue (`build_queue`): every
`in_review` source, then every approved source whose file changed since
approval (stale), in `sources.yaml` order. `--all` adds the approved
sources that are not stale, for a second look. With an ID, the queue is that
one source.

For each source the reviewer picks an action by its key, until a decision:

    v  view       the file, printed in the terminal with line numbers. Lines with scan
                  findings carry a `!` mark and the rule names; secrets are
                  masked as `Finding.display` masks them
    e  edit       the snapshot in $VISUAL or $EDITOR, then hash and scan again.
                  New content clears the checklist answers
    c  checklist  each CHECKLIST item, y or n. A "no" ends it
    a  approve    only after a yes to every item for the current content.
                  The file is hashed again first: if it changed since the
                  checklist, the approval is refused
    r  reject     with a reason that is not blank
    s  skip       next source, nothing saved
    q  quit       stop. Ctrl-C and end of input do the same

Each decision is saved when it is made: `sources.yaml` is written to a
temporary file and renamed over the old one (`registry.save_registry`), so
quitting or an interrupt keeps earlier decisions and never leaves half a
file. Before that, `record_decision` appends the decision to
`reviews/<id>.md`: reviewer, hash, the rule and position of each scan
finding, the checklist answers, and whether the file was edited during the
review. So a saved decision always has its record; if the save then fails or
is interrupted, the record gets a `decision not saved: <exception class>`
line. It never writes matched text, which may be a secret.

All terminal I/O goes through `ReviewIO`, so tests drive a session with
plain functions. `cli.py` passes Typer's prompt and echo, and Click's editor.

Tests: backend/tests/test_review_session.py.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, assert_never

from pydantic import ValidationError

from profile_rag.corpus import registry, scan
from profile_rag.corpus.registry import REGISTRY_FILENAME, REVIEWS_DIR, SourceEntry

# Phrased so that "yes" is the safe answer to every item.
CHECKLIST: tuple[str, ...] = (
    "Is the content public, or sanitized into a version any visitor could see?",
    "Is it free of secrets, credentials, internal hostnames, and other people's personal data?",
    "Is the attribution category correct?",
    "Is the revision recorded (commit SHA, date, or hash)?",
    "Would you be comfortable with an answer quoting any sentence of it verbatim?",
)


class ReviewError(Exception):
    """A decision cannot be saved. The message is for the user and names the source."""


class EditorFailed(Exception):
    """The editor did not start or exited with an error. The message says why."""


@dataclass(frozen=True)
class ReviewIO:
    """The terminal. `prompt` returns one line and raises EOFError or
    KeyboardInterrupt when the reviewer stops; `edit` raises EditorFailed."""

    prompt: Callable[[str], str]
    echo: Callable[[str], None]
    # Prints the file view. Not a pager: a pager's screen is cleared when it
    # closes, and the file has to stay visible above the checklist questions.
    show: Callable[[str], None]
    edit: Callable[[Path], None]


# --- Decisions and the audit record -------------------------------------------


@dataclass(frozen=True)
class AllYes:
    """Every CHECKLIST item was answered yes in the session, for the approved hash."""


@dataclass(frozen=True)
class ConfirmedByFlag:
    """`--approve --checklist-confirmed`: the reviewer states each item was checked."""


@dataclass(frozen=True)
class Answers:
    """The answers given so far, in CHECKLIST order. Empty if none were asked."""

    given: tuple[tuple[str, bool], ...] = ()


@dataclass(frozen=True)
class Approve:
    digest: str
    checklist: AllYes | ConfirmedByFlag
    edited: bool


@dataclass(frozen=True)
class Reject:
    reason: str
    digest: str
    checklist: Answers
    edited: bool


type Decision = Approve | Reject


def record_decision(
    *,
    root: Path,
    entry: SourceEntry,
    decision: Decision,
    reviewer: str,
    when: datetime,
    findings: Sequence[scan.Finding],
) -> SourceEntry:
    """Append `decision` to reviews/, then save it in `sources.yaml`.

    The registry is read again first, so decisions saved earlier in the same
    session are kept. Raises ReviewError if the entry is gone, its snapshot
    path changed, the result would be invalid, or `sources.yaml` cannot be
    written. An interrupt during the save is re-raised after the record is
    marked as not saved.
    """
    path = root / REGISTRY_FILENAME
    try:
        current = registry.load_registry(path)
    except registry.RegistryError as error:
        raise ReviewError(str(error)) from error
    saved = current.get(entry.id)
    if saved is None:
        raise ReviewError(f"{entry.id} is no longer in {REGISTRY_FILENAME}; nothing saved")
    if saved.path != entry.path:
        raise ReviewError(f"{entry.id}: its snapshot path changed during the review; review again")

    changes: dict[str, object]
    match decision:
        case Approve(digest=digest):
            changes = {
                "state": "approved",
                "content_sha256": digest,
                "approved_at": when,
                "approved_by": reviewer,
                "rejection_reason": None,
            }
        case Reject(reason=reason):
            changes = {
                "state": "rejected",
                "rejection_reason": reason,
                "approved_at": None,
                "approved_by": None,
            }
        case _:
            assert_never(decision)
    try:
        updated = registry.SourceEntry.model_validate({**saved.model_dump(), **changes})
        new = registry.Registry.model_validate(
            {"sources": [updated if e.id == entry.id else e for e in current.sources]}
        )
    except ValidationError as error:
        raise ReviewError(f"{entry.id}: the decision is invalid: {error}") from error
    # The audit record goes first, so no saved decision is ever without one. A
    # record whose save failed or was interrupted is marked as not saved.
    _append_record(root, updated, decision, reviewer, when, findings)
    try:
        registry.save_registry(new, path)
    except BaseException as error:  # also KeyboardInterrupt, re-raised below
        _append_not_saved(root, entry.id, error)
        if isinstance(error, OSError):
            message = f"{entry.id}: could not write {REGISTRY_FILENAME}: {error}"
            raise ReviewError(message) from error
        raise
    return updated


def _append_not_saved(root: Path, source_id: str, error: BaseException) -> None:
    """Mark the last record as not saved. Only the exception class: its message
    could quote file content."""
    with (root / REVIEWS_DIR / f"{source_id}.md").open("a", encoding="utf-8") as record:
        record.write(f"- decision not saved: {type(error).__name__}\n\n")


def _append_record(
    root: Path,
    entry: SourceEntry,
    decision: Decision,
    reviewer: str,
    when: datetime,
    findings: Sequence[scan.Finding],
) -> None:
    kind = "approve" if isinstance(decision, Approve) else "reject"
    lines = [
        f"## {when.isoformat()} {kind}",
        "",
        f"- reviewer: {reviewer}",
        f"- revision: {entry.revision}",
        f"- path: {entry.path}",
        f"- sha256: {decision.digest}",
        f"- attribution: {entry.attribution}",
        f"- visibility: {entry.visibility}",
    ]
    if isinstance(decision, Reject):
        lines.append(f"- reason: {decision.reason}")
    # Only rule names and positions: the matched text may be a secret.
    lines.append(f"- scan findings: {findings_summary(findings) or 'none'}")
    lines.append(f"- edited during review: {'yes' if decision.edited else 'no'}")
    lines.extend(_checklist_lines(decision.checklist))
    lines.extend(("", ""))
    reviews = root / REVIEWS_DIR
    reviews.mkdir(parents=True, exist_ok=True)
    with (reviews / f"{entry.id}.md").open("a", encoding="utf-8") as record:
        record.write("\n".join(lines))


def _checklist_lines(checklist: AllYes | ConfirmedByFlag | Answers) -> list[str]:
    match checklist:
        case AllYes():
            return ["- checklist:", *(f"  - yes: {question}" for question in CHECKLIST)]
        case ConfirmedByFlag():
            return ["- checklist: confirmed by flag"]
        case Answers(given=()):
            return ["- checklist: not answered"]
        case Answers(given=given):
            answered = (f"  - {'yes' if ok else 'no'}: {question}" for question, ok in given)
            return ["- checklist:", *answered]
        case _:
            assert_never(checklist)


# --- What is shown --------------------------------------------------------------


def findings_summary(findings: Sequence[scan.Finding]) -> str:
    """`rule line:col, ...`, without the matched text. Empty if there are none."""
    return ", ".join(f"{f.rule} {f.line}:{f.column}" for f in findings)


def read_text(file: Path) -> str:
    """The file as `scan.scan_file` reads it, so finding positions line up."""
    return file.read_bytes().decode("utf-8", errors="replace")


def render_view(text: str, findings: Sequence[scan.Finding]) -> str:
    """The text with line numbers. A line with findings starts with `!` and ends
    with the rule names. Each run of secret text keeps its first 4 characters
    and the rest becomes `…`, as in `Finding.display`."""
    by_line: dict[int, list[scan.Finding]] = {}
    for finding in findings:
        by_line.setdefault(finding.line, []).append(finding)
    lines = text.splitlines()
    width = len(str(len(lines)))
    out = ["Lines marked ! have scan findings. Secrets are masked.", ""]
    for number, line in enumerate(lines, start=1):
        found = by_line.get(number, [])
        secrets = [f for f in found if f.category == "secret"]
        shown = _mask(line, secrets)
        if found:
            rules = ", ".join(dict.fromkeys(f.rule for f in found))
            out.append(f"! {number:>{width}} | {shown}    <- {rules}")
        else:
            out.append(f"  {number:>{width}} | {shown}")
    return "\n".join(out) + "\n"


def _mask(line: str, secrets: Sequence[scan.Finding]) -> str:
    hidden = [False] * len(line)
    for finding in secrets:
        start = finding.column - 1
        for index in range(start, min(start + len(finding.match), len(line))):
            hidden[index] = True
    parts: list[str] = []
    index = 0
    while index < len(line):
        if not hidden[index]:
            parts.append(line[index])
            index += 1
            continue
        end = index
        while end < len(line) and hidden[end]:
            end += 1
        parts.append(f"{line[index : index + 4]}…")
        index = end
    return "".join(parts)


def print_header(
    io: ReviewIO,
    entry: SourceEntry,
    digest: str,
    findings: Sequence[scan.Finding],
    position: tuple[int, int] | None = None,
) -> None:
    """The source's metadata and a findings summary, never its text."""
    state: str = entry.state
    if entry.state == "approved" and entry.content_sha256 != digest:
        state = "approved, but the file changed since approval"
    where = f"[{position[0]}/{position[1]}] " if position else ""
    io.echo("")
    io.echo(f"{where}{entry.id} ({state})")
    io.echo(
        f"  kind: {entry.kind}  attribution: {entry.attribution}"
        f"  visibility: {entry.visibility}  revision: {entry.revision or '(none)'}"
    )
    io.echo(f"  file: {entry.path}  sha256: {digest[:12]}…")
    _echo_findings(io, findings)


def _echo_findings(io: ReviewIO, findings: Sequence[scan.Finding]) -> None:
    summary = findings_summary(findings)
    if summary:
        io.echo(f"  findings: {summary}")
    else:
        io.echo("  findings: none. That does not make the file safe; read it in full.")


# --- The queue ------------------------------------------------------------------


@dataclass(frozen=True)
class Queue:
    sources: list[SourceEntry]
    missing: list[str]
    """IDs that belong in the queue but have no snapshot file: snapshot them again."""


def build_queue(entries: Sequence[SourceEntry], root: Path, *, include_all: bool) -> Queue:
    """In-review sources, then stale approvals, then (with `include_all`) the
    other approved sources. Each group keeps the order of `entries`."""
    stale = set(registry.find_stale(entries, root))
    groups: tuple[list[SourceEntry], list[SourceEntry], list[SourceEntry]] = ([], [], [])
    missing: list[str] = []
    for entry in entries:
        if entry.state == "in_review":
            group = groups[0]
        elif entry.state == "approved" and entry.id in stale:
            group = groups[1]
        elif entry.state == "approved" and include_all:
            group = groups[2]
        else:
            continue
        if entry.path is None or not (root / entry.path).is_file():
            missing.append(entry.id)
        else:
            group.append(entry)
    return Queue(sources=[*groups[0], *groups[1], *groups[2]], missing=missing)


# --- The session ----------------------------------------------------------------


@dataclass
class Summary:
    approved: int = 0
    rejected: int = 0
    skipped: int = 0


type Outcome = Literal["approved", "rejected", "skipped", "quit"]
type Action = Literal["view", "edit", "checklist", "approve", "reject", "skip", "quit"]

_KEYS: dict[str, Action] = {
    "v": "view",
    "e": "edit",
    "c": "checklist",
    "a": "approve",
    "r": "reject",
    "s": "skip",
    "q": "quit",
}
# The whole word works too: `approve` as well as `a`.
_WORDS: dict[str, Action] = {action: action for action in _KEYS.values()}


class _FileGone(Exception):
    pass


@dataclass
class _Source:
    """One source under review. `checked` is the hash every checklist item was
    answered yes for; None until then, and again after the content changes."""

    entry: SourceEntry
    file: Path
    digest: str
    findings: list[scan.Finding]
    checked: str | None = None
    answers: list[tuple[str, bool]] = field(default_factory=list[tuple[str, bool]])
    edited: bool = False

    def refresh(self) -> bool:
        """Hash the file again. If it changed, scan it and clear the checklist."""
        try:
            digest = registry.file_sha256(self.file)
        except OSError as error:
            raise _FileGone(str(error)) from error
        if digest == self.digest:
            return False
        self.digest = digest
        self.findings = scan.scan_file(self.file)
        self.checked = None
        self.answers = []
        return True


@dataclass(frozen=True)
class Session:
    root: Path
    io: ReviewIO
    reviewer: str
    clock: Callable[[], datetime]
    # Guided: each source opens with its contents, then the checklist, then an
    # offer to approve. The menu follows only if that does not end in approval.
    guided: bool = True

    def run(self, queue: Sequence[SourceEntry]) -> Summary:
        """Review each source in turn until the end or a quit. Decisions are saved
        as they are made; one that cannot be saved is reported and counted as
        skipped."""
        summary = Summary()
        try:
            for number, entry in enumerate(queue, start=1):
                outcome = self._review(entry, (number, len(queue)))
                if outcome == "quit":
                    self.io.echo("Stopped. Decisions made so far are saved.")
                    break
                if outcome == "approved":
                    summary.approved += 1
                elif outcome == "rejected":
                    summary.rejected += 1
                else:
                    summary.skipped += 1
        except EOFError, KeyboardInterrupt:
            self.io.echo("")
            self.io.echo("Stopped (interrupted). Decisions made so far are saved.")
        return summary

    def _review(self, entry: SourceEntry, position: tuple[int, int]) -> Outcome:
        if entry.path is None:
            self.io.echo(f"{entry.id} has no snapshot; skipped.")
            return "skipped"
        file = self.root / entry.path
        try:
            source = _Source(entry, file, registry.file_sha256(file), scan.scan_file(file))
            print_header(self.io, entry, source.digest, source.findings, position)
            return self._loop(source)
        except _FileGone:
            self.io.echo(
                f"{entry.path} is missing; skipped. Run: profile-rag corpus snapshot {entry.id}"
            )
            return "skipped"
        except ReviewError as error:
            self.io.echo(f"Not saved: {error}. Skipped.")
            return "skipped"

    def _loop(self, source: _Source) -> Outcome:
        if self.guided and self._walkthrough(source):
            return "approved"
        while True:
            action = self._ask_action(source)
            match action:
                case "view":
                    self._view(source)
                case "edit":
                    if self._edit(source) and self.guided and self._walkthrough(source):
                        return "approved"
                case "checklist":
                    self._checklist(source)
                case "approve":
                    if self._approve(source):
                        return "approved"
                case "reject":
                    self._reject(source)
                    return "rejected"
                case "skip":
                    self.io.echo("Skipped; no change.")
                    return "skipped"
                case "quit":
                    return "quit"
                case _:
                    assert_never(action)

    def _ask_action(self, source: _Source) -> Action:
        if source.checked is not None and source.checked == source.digest:
            menu = "[v]iew [e]dit [c]hecklist [a]pprove [r]eject [s]kip [q]uit"
        else:
            menu = "[v]iew [e]dit [c]hecklist [r]eject [s]kip [q]uit (approve after the checklist)"
        while True:
            answer = self.io.prompt(menu).strip().lower()
            action = _KEYS.get(answer) or _WORDS.get(answer)
            if action is not None:
                return action
            self.io.echo("Type one of: v, e, c, a, r, s, q.")

    def _walkthrough(self, source: _Source) -> bool:
        """Show the file, ask every checklist item, then offer approval.
        True if the source was approved."""
        self._view(source)
        self.io.echo(f"Checklist for {source.entry.id}:")
        if not self._checklist(source):
            return False
        if not self._ask_yes_no(f"Approve {source.entry.id}? [y/n]"):
            return False
        return self._approve(source)

    def _view(self, source: _Source) -> None:
        self._sync(source)
        text = read_text(source.file)
        path = source.entry.path
        self.io.echo(f"\n--- {path} ---")
        self.io.show(render_view(text, source.findings))
        lines = len(text.splitlines())
        self.io.echo(f"--- end of {path} ({lines} lines) ---\n")

    def _sync(self, source: _Source) -> None:
        """Hash the file again; say so if it changed since the last look."""
        if source.refresh():
            self.io.echo("The file changed outside this review; checklist answers cleared.")
            _echo_findings(self.io, source.findings)

    def _edit(self, source: _Source) -> bool:
        """Open the file in the editor. True if its content changed."""
        self._sync(source)
        try:
            self.io.edit(source.file)
        except EditorFailed as error:
            self.io.echo(f"The editor failed: {error}")
        had_answers = bool(source.answers)
        if not source.refresh():
            self.io.echo("No change.")
            return False
        source.edited = True
        self.io.echo(f"Content changed; sha256 {source.digest[:12]}…")
        _echo_findings(self.io, source.findings)
        if had_answers:
            self.io.echo("Checklist answers cleared: answer the checklist again.")
        return True

    def _checklist(self, source: _Source) -> bool:
        """Ask each item. True if every answer is yes; then `source.checked` is set."""
        self._sync(source)
        digest = source.digest
        source.answers = []
        source.checked = None
        for number, question in enumerate(CHECKLIST, start=1):
            ok = self._ask_yes_no(f"{number}/{len(CHECKLIST)} {question} [y/n]")
            source.answers.append((question, ok))
            if not ok:
                self.io.echo("Answered no. Edit the file ([e]) or reject the source ([r]).")
                return False
        source.checked = digest
        self.io.echo("Every item answered yes. [a]pprove is available.")
        return True

    def _ask_yes_no(self, question: str) -> bool:
        while True:
            answer = self.io.prompt(question).strip().lower()
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            self.io.echo("Type y or n.")

    def _approve(self, source: _Source) -> bool:
        if source.checked is None or source.checked != source.digest:
            self.io.echo("Approval needs a yes to every checklist item for the current content.")
            if not self._checklist(source):
                return False
        checked = source.checked
        if checked is None:  # the checklist sets it when it passes
            return False
        if source.refresh():
            self.io.echo(
                "Not approved: the file changed after the checklist was answered."
                " Answer the checklist again."
            )
            _echo_findings(self.io, source.findings)
            return False
        decision = Approve(digest=checked, checklist=AllYes(), edited=source.edited)
        self._save(source, decision)
        return True

    def _reject(self, source: _Source) -> None:
        self._sync(source)
        while True:
            reason = self.io.prompt("Reason").strip()
            if reason:
                break
            self.io.echo("A rejection needs a reason.")
        decision = Reject(
            reason=reason,
            digest=source.digest,
            checklist=Answers(tuple(source.answers)),
            edited=source.edited,
        )
        self._save(source, decision)

    def _save(self, source: _Source, decision: Decision) -> None:
        updated = record_decision(
            root=self.root,
            entry=source.entry,
            decision=decision,
            reviewer=self.reviewer,
            when=self.clock(),
            findings=source.findings,
        )
        self.io.echo(f"{updated.id} is {updated.state}.")


def decide_by_flag(
    *,
    root: Path,
    entry: SourceEntry,
    io: ReviewIO,
    reviewer: str,
    when: datetime,
    reject: str | None,
) -> SourceEntry:
    """`--approve --checklist-confirmed` (reject None) or `--reject REASON`: show the
    header and save the decision without asking. The snapshot must exist."""
    if entry.path is None:
        raise ReviewError(f"{entry.id} has no snapshot path")
    file = root / entry.path
    digest = registry.file_sha256(file)
    findings = scan.scan_file(file)
    print_header(io, entry, digest, findings)
    decision: Decision
    if reject is None:
        decision = Approve(digest=digest, checklist=ConfirmedByFlag(), edited=False)
    else:
        decision = Reject(reason=reject, digest=digest, checklist=Answers(), edited=False)
    updated = record_decision(
        root=root, entry=entry, decision=decision, reviewer=reviewer, when=when, findings=findings
    )
    io.echo(f"{updated.id} is {updated.state}.")
    return updated


def print_summary(io: ReviewIO, root: Path, summary: Summary) -> None:
    """The session's counts and what is still waiting in the whole registry."""
    try:
        current = registry.load_registry(root / REGISTRY_FILENAME)
    except registry.RegistryError as error:
        raise ReviewError(str(error)) from error
    in_review = sum(1 for entry in current.sources if entry.state == "in_review")
    stale = len(registry.find_stale(current.sources, root))
    io.echo("")
    io.echo(
        f"Approved {summary.approved}, rejected {summary.rejected}, skipped {summary.skipped}."
        f" Remaining: in_review {in_review}, stale {stale}."
    )
