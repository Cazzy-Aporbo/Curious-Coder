"""Bounded async processing with typed failures, task context, and cancellation."""

import argparse
import asyncio
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import json
import math
import random
from typing import Awaitable, Callable, Iterable


request_id = ContextVar("request_id", default=None)


class TransientFailure(Exception):
    pass


class PermanentFailure(Exception):
    pass


@dataclass(frozen=True)
class Job:
    key: str
    payload: dict


@dataclass(frozen=True)
class Outcome:
    index: int
    key: str
    attempts: int
    value: object = None
    error: str | None = None


@dataclass
class PipelineStats:
    admitted: int = 0
    completed: int = 0
    failed: int = 0
    retries: int = 0
    max_active: int = 0
    max_queued: int = 0


async def process_jobs(jobs: Iterable[Job], handler: Callable[[Job], Awaitable[object]], *,
                       workers=3, capacity=4, attempts=3, timeout=1.0, base_delay=.01):
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in (workers, capacity, attempts)):
        raise ValueError("workers, capacity, and attempts must be positive integers.")
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(base_delay) or base_delay < 0:
        raise ValueError("Timeout must be finite and positive; retry delay must be finite and nonnegative.")
    queue = asyncio.Queue(maxsize=capacity)
    stats, outcomes = PipelineStats(), []
    active = 0

    async def produce():
        for index, job in enumerate(jobs):
            if not isinstance(job, Job) or not job.key.strip():
                raise ValueError("Each job requires a nonempty request key.")
            await queue.put((index, job))
            stats.admitted += 1
            stats.max_queued = max(stats.max_queued, queue.qsize())
        for _ in range(workers):
            await queue.put(None)

    async def consume(worker_index):
        nonlocal active
        rng = random.Random(worker_index)
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            index, job = item
            token = request_id.set(job.key)
            active += 1
            stats.max_active = max(stats.max_active, active)
            try:
                for attempt in range(1, attempts + 1):
                    try:
                        async with asyncio.timeout(timeout):
                            result = await handler(job)
                        outcomes.append(Outcome(index, job.key, attempt, value=result))
                        stats.completed += 1
                        break
                    except (TransientFailure, TimeoutError) as error:
                        if attempt == attempts:
                            outcomes.append(Outcome(index, job.key, attempt, error=type(error).__name__))
                            stats.failed += 1
                        else:
                            stats.retries += 1
                            await asyncio.sleep(rng.uniform(0, min(1.0, base_delay * 2 ** min(attempt - 1, 20))))
                    except PermanentFailure:
                        outcomes.append(Outcome(index, job.key, attempt, error="PermanentFailure"))
                        stats.failed += 1
                        break
            finally:
                request_id.reset(token)
                active -= 1
                queue.task_done()

    async with asyncio.TaskGroup() as group:
        group.create_task(produce(), name="producer")
        for worker_index in range(workers):
            group.create_task(consume(worker_index), name=f"worker-{worker_index}")
    return sorted(outcomes, key=lambda outcome: outcome.index), stats


async def demonstration():
    receipts, attempts = {}, {}

    async def receiver(job):
        await asyncio.sleep(.002)
        attempts[job.key] = attempts.get(job.key, 0) + 1
        if job.key not in receipts:
            receipts[job.key] = {"squared": job.payload["value"] ** 2}
        if job.key == "job-0" and attempts[job.key] == 1:
            raise TransientFailure("The receiver committed, but its acknowledgement was lost.")
        return receipts[job.key]

    jobs = (Job(f"job-{index % 4}", {"value": index % 4}) for index in range(8))
    results, stats = await process_jobs(jobs, receiver)
    return {"stats": asdict(stats), "unique_receiver_mutations": len(receipts),
            "outcomes": [asdict(result) for result in results],
            "scope": "In-memory receiver simulation; no external effects or durable queue."}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(asyncio.run(demonstration()), indent=2))
