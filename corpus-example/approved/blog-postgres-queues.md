# Why we moved our job queue into PostgreSQL

*Dana Whitlock, August 2026. Fictional post used as example data.*

For three years our background jobs ran on a Redis-based queue. It was fast,
and it lost jobs. Not many: a handful each week, almost always during a
deploy, when a worker was stopped between taking a job and finishing it.

## What we wanted

We wanted a job to be either done or still waiting, never lost. We also
wanted to enqueue a job in the same transaction as the data change that
caused it, so an order and its confirmation email either both exist or
neither does.

## How it works

Each job is a row. A worker claims one with:

```sql
SELECT id, payload FROM jobs
WHERE run_at <= now()
ORDER BY run_at
LIMIT 1
FOR UPDATE SKIP LOCKED;
```

`SKIP LOCKED` lets many workers poll the same table without waiting on each
other. The row stays locked until the worker commits, so if the worker
crashes, the lock is released and another worker picks the job up.

## What it cost

Throughput is lower than Redis: we measured about 2,000 jobs per second on
our database, against a need of about 50. Table bloat needs attention; we
delete finished jobs in batches and let autovacuum keep up.

## Result

Since the move, we have not lost a job during a deploy. The queue code is now
a small internal library, lantern-queue.
