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

3. Review the snapshot. The command lists what the scanner flags (email addresses, phone numbers, secrets, internal hosts) and a checklist, then asks you to approve or reject:

   ```sh
   uv run profile-rag corpus review resume-2026
   ```

   Edit the file in `approved/` and run `review` again until it is clean. Approval records the file's SHA-256 hash. The decision and the findings are saved in `reviews/`.

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
