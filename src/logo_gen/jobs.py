"""In-memory background jobs with progress and cancel.

Jobs run as asyncio tasks on the server loop. The UI polls GET /api/jobs
for progress; results (image ids) are appended as they are produced so the
gallery fills in while a job is still running.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal

Status = Literal["running", "done", "error", "cancelled"]


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str
    title: str
    status: Status = "running"
    done: int = 0
    total: int = 0
    message: str = ""
    errors: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cost: float = 0.0
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)

    # Called from worker threads as well as the loop, so keep it simple.
    def progress(self, done: int | None = None, total: int | None = None, message: str | None = None) -> None:
        if done is not None:
            self.done = done
        if total is not None:
            self.total = total
        if message is not None:
            self.message = message

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "message": self.message,
            "errors": self.errors,
            "image_ids": self.image_ids,
            "result": self.result,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "cost": round(self.cost, 6),
        }


class JobManager:
    def __init__(self, keep: int = 50) -> None:
        self._jobs: dict[str, Job] = {}
        self._keep = keep

    def start(self, kind: str, title: str, fn: Callable[[Job], Awaitable[None]]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, title=title)
        self._jobs[job.id] = job

        async def runner() -> None:
            try:
                await fn(job)
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Cancelled"
                elif job.status == "running":
                    job.status = "done"
            except (Cancelled, asyncio.CancelledError):
                job.status = "cancelled"
                job.message = "Cancelled"
            except Exception as e:  # noqa: BLE001
                job.status = "error"
                job.message = getattr(e, "message", None) or str(e) or e.__class__.__name__
                job.errors.append(job.message)
                traceback.print_exc()
            finally:
                job.finished_at = time.time()
                self._trim()

        job._task = asyncio.get_running_loop().create_task(runner())
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job and job.status == "running":
            job._cancel.set()
            job.message = "Cancelling..."
            # Give cooperative code (local pipelines) a moment, then hard-cancel.
            loop = asyncio.get_running_loop()
            loop.call_later(3.0, lambda: job._task and not job._task.done() and job._task.cancel())
        return job

    def _trim(self) -> None:
        finished = [j for j in self.list() if j.status != "running"]
        for j in finished[self._keep:]:
            self._jobs.pop(j.id, None)


jobs = JobManager()
