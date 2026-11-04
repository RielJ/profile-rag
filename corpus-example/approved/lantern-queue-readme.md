# lantern-queue

A small job queue for Python applications that already use PostgreSQL.
Jobs are rows in a table; workers claim them with `SELECT ... FOR UPDATE
SKIP LOCKED`, so no separate broker is needed.

This is a fictional project used as example data for profile-rag.

## Features

- At-least-once delivery. A job stays claimed only while its worker's
  transaction is open, so a crashed worker releases its jobs.
- Retries with exponential backoff and a per-job attempt limit.
- Scheduled jobs: set `run_at` to delay a job.
- A dead-letter table for jobs that ran out of attempts.
- Metrics for queue depth and job latency, exported in Prometheus format.

## Quick start

```python
from lantern_queue import Queue

queue = Queue("postgresql://localhost/app")
queue.enqueue("send_welcome_email", {"user_id": 42})
```

Run a worker:

```bash
lantern-queue worker --concurrency 4
```

## Maintainers

lantern-queue is maintained by the Harborlight platform team. See
CONTRIBUTING.md for how to propose changes.
