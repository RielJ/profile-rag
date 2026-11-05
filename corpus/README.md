# corpus/

This folder holds the real corpus: the sources the application answers from. Git ignores everything here except this README, because the repository is public.

| Path | Contents |
| --- | --- |
| `sources.yaml` | Registry of every candidate source and its review state |
| `approved/` | Snapshots that may be ingested: public material, or sanitized derivatives of private material |
| `private/` | Raw private material. Never ingested |
| `reviews/` | Review decisions and redaction notes |

Back this folder up separately. A private Git repository cloned into this folder works: the outer repository ignores it, and it keeps its own history.

For a committed, fictional example of the same layout, see `../corpus-example/`.

## Add a source

Run these commands from `backend/`. Each one reads and writes `corpus/sources.yaml`. To use another folder, pass `--root <path>` after `corpus`, or set `CORPUS_DIR`.

1. Register the source. It starts as a `candidate`:

   ```sh
   uv run profile-rag corpus add resume-2026 --kind resume --title "Resume" \
     --attribution self_report --visibility public
   ```

2. Copy it into `approved/`. The source moves to `in_review`:

   ```sh
   uv run profile-rag corpus snapshot resume-2026 --from ~/Documents/resume.md
   ```

   `--from` takes a local path or an `http(s)` URL. For private material, add `--private`: the raw copy goes to `private/` and the source stays a `candidate`. Write a sanitized version, then snapshot that file without `--private`: `corpus snapshot <id> --from <sanitized file>`.

3. Review the snapshots in a terminal session:

   ```sh
   uv run profile-rag corpus review              # every source waiting for review
   uv run profile-rag corpus review resume-2026  # one source
   ```

   Without an ID, the session goes through every `in_review` source, then every stale approval, in `sources.yaml` order. Add `--all` to also review approved sources that are not stale. For each source it shows the position in the queue, the metadata, and where the scanner found email addresses, phone numbers, secrets, or internal hosts (rule and line:column, never the matched text).

   Each source then prints its contents in the terminal, with line numbers, between `--- <path> ---` and `--- end of <path> (N lines) ---`, so the file stays on screen above the questions; scroll back for a long one. Lines with findings are marked `!` and name the rule, and secrets are masked after their first 4 characters; everything else is shown as it is, so you can judge it. The session then asks the five checklist questions. If you answer `y` to all five, it asks `Approve <id>? [y/n]`. If you answer `n` to a question or to the approval, the menu follows. Type a key and press Enter:

   | Key | Action |
   | --- | --- |
   | `v` | Print the file again |
   | `e` | Edit the snapshot in `$VISUAL` or `$EDITOR` (one that waits until you close the file, such as `nvim` or `code --wait`). If the content changed, the file is scanned again, the checklist answers are cleared, and the session shows the new content and asks the checklist again |
   | `c` | Answer the checklist, one item at a time. Every item is phrased so that `y` means OK. An `n` ends the checklist: edit the file or reject the source |
   | `a` | Approve. Allowed only after `y` to every item for the current content. If the file changed since the checklist (in the editor or elsewhere), the approval is refused |
   | `r` | Reject, with a reason |
   | `s` | Skip to the next source; nothing changes |
   | `q` | Quit. Ctrl-C and Ctrl-D do the same |

   Each decision is saved when you make it, so quitting keeps the decisions made before. At the end, the session prints how many sources were approved, rejected, and skipped, and how many are still in review or stale. A short session, with the inputs after each prompt:

   ```text
   [1/2] lantern-queue-readme (in_review)
     kind: github  attribution: project_doc  visibility: public  revision: 3f9a1c2e7b4d…
     file: approved/lantern-queue-readme.md  sha256: 61ad5de68538…
     findings: credential_assignment 24:7, github_token 24:16

   --- approved/lantern-queue-readme.md ---
   Lines marked ! have scan findings. Secrets are masked.
     ...
     23 | queue = Queue("postgresql://localhost/app")
   ! 24 | queue.toke…"    <- credential_assignment, github_token
     25 | queue.enqueue("send_welcome_email", {"user_id": 42})
     ...
   --- end of approved/lantern-queue-readme.md (37 lines) ---

   Checklist for lantern-queue-readme:
   1/5 Is the content public, or sanitized into a version any visitor could see? [y/n]: y
   2/5 Is it free of secrets, credentials, internal hostnames, and other people's personal data? [y/n]: n
   Answered no. Edit the file ([e]) or reject the source ([r]).
   [v]iew [e]dit [c]hecklist [r]eject [s]kip [q]uit (approve after the checklist): e
   Content changed; sha256 b046b1f5507e…
     findings: none. That does not make the file safe; read it in full.
   Checklist answers cleared: answer the checklist again.

   --- approved/lantern-queue-readme.md ---
     (the edited file)
   --- end of approved/lantern-queue-readme.md (36 lines) ---

   Checklist for lantern-queue-readme:
   1/5 Is the content public, or sanitized into a version any visitor could see? [y/n]: y
   2/5 Is it free of secrets, credentials, internal hostnames, and other people's personal data? [y/n]: y
   3/5 Is the attribution category correct? [y/n]: y
   4/5 Is the revision recorded (commit SHA, date, or hash)? [y/n]: y
   5/5 Would you be comfortable with an answer quoting any sentence of it verbatim? [y/n]: y
   Every item answered yes. [a]pprove is available.
   Approve lantern-queue-readme? [y/n]: y
   lantern-queue-readme is approved.

   [2/2] resume-2026 (approved, but the file changed since approval)
     ...
   Checklist for resume-2026:
   1/5 Is the content public, or sanitized into a version any visitor could see? [y/n]: n
   Answered no. Edit the file ([e]) or reject the source ([r]).
   [v]iew [e]dit [c]hecklist [r]eject [s]kip [q]uit (approve after the checklist): q
   Stopped. Decisions made so far are saved.

   Approved 1, rejected 0, skipped 0. Remaining: in_review 0, stale 1.
   ```

   Approval records the SHA-256 hash of the file you checked. Each decision is appended to `reviews/<id>.md` with the reviewer, the hash, the rule and position of each finding, the checklist answers, and whether the file was edited during the review. It never contains the matched text. The record is written before `sources.yaml` is saved. If the save then fails or is interrupted, the record ends with `- decision not saved: <error type>`, and the source keeps its previous state. `sources.yaml` is always the source of truth.

   For scripts, decide without the session: `corpus review ID --reject REASON`, or `corpus review ID --approve --checklist-confirmed`. The `--checklist-confirmed` flag states that you checked every item; the record says `checklist: confirmed by flag`.

4. Check the corpus:

   ```sh
   uv run profile-rag corpus status
   ```

   An approved file that is missing, or whose hash no longer matches, is *stale*. `status` exits with code 1 when any approval is stale. If you edited the file, review the source again. If the file is missing or the original changed, snapshot it again first.

## States

| State | Meaning |
| --- | --- |
| `candidate` | Registered; nothing copied yet |
| `in_review` | Snapshot in `approved/`, waiting for a decision |
| `approved` | Reviewed. May be ingested while its hash matches |
| `rejected` | Will not be ingested. `rejection_reason` says why |

Only `approved` sources are ingested.
