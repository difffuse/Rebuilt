"""Cooperative background work with progress and cancellation."""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from concurrent.futures import Future, TimeoutError as FutureTimeout


class Cancelled(Exception):
    """Raised inside a worker when cancellation has been requested."""


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_finished(self) -> bool:
        return self in (TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED)


@dataclass(frozen=True)
class Progress:
    fraction: float
    message: str


class ProgressReporter:
    """The worker's handle back to the main thread."""

    def __init__(self, cancel_event: threading.Event) -> None:
        self._cancel_event = cancel_event
        self._lock = threading.Lock()
        self._progress = Progress(0.0, "")

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        """Bail out of the worker if cancellation was requested."""
        if self._cancel_event.is_set():
            raise Cancelled()

    def update(self, fraction: float, message: str = "") -> None:
        """Publish progress, and bail out if cancellation was requested."""
        self.check_cancelled()
        with self._lock:
            self._progress = Progress(min(1.0, max(0.0, float(fraction))), message)

    @property
    def progress(self) -> Progress:
        with self._lock:
            return self._progress


class MainThreadCalls:
    """Array requests from a worker, serviced one at a time by the job's tick."""

    def __init__(self):
        self._pending = queue.Queue()

    def call(self, payload: Any, report: ProgressReporter) -> Any:
        report.check_cancelled()
        future = Future()
        self._pending.put((payload, future))
        try:
            while True:
                report.check_cancelled()
                try:
                    return future.result(timeout=0.05)
                except FutureTimeout:
                    if future.done():
                        return future.result()
        except Cancelled:
            future.cancel()
            raise

    def poll(self, handler: Callable[[Any], Any], *, timeout: float = 0.0) -> bool:
        try:
            payload, future = self._pending.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return False
        if not future.set_running_or_notify_cancel():
            return True
        try:
            result = handler(payload)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)
        return True


class BackgroundTask:
    """Runs one function on a worker thread."""

    def __init__(self, func: Callable[[ProgressReporter], Any], name: str = "task") -> None:
        self.name = name
        self._func = func
        self._cancel_event = threading.Event()
        self._reporter = ProgressReporter(self._cancel_event)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._state = TaskState.PENDING
        self._result: Any = None
        self._error: BaseException | None = None
        self._traceback: str = ""

    def start(self) -> None:
        with self._lock:
            if self._state is not TaskState.PENDING:
                raise RuntimeError(f"task {self.name!r} was already started")
            self._state = TaskState.RUNNING
        try:
            self._thread = threading.Thread(target=self._run, name=f"rebuilt-{self.name}", daemon=True)
            self._thread.start()
        except Exception as error:
            # No worker launched. Joining an unstarted Thread would itself fail
            # and prevent the owning job from restoring its scene/files.
            self._thread = None
            with self._lock:
                self._state = TaskState.FAILED
                self._error = error
                self._traceback = traceback.format_exc()
            raise

    def _run(self) -> None:
        try:
            result = self._func(self._reporter)
        except Cancelled:
            with self._lock:
                self._state = TaskState.CANCELLED
        except BaseException as exc:  # noqa: BLE001 — must not lose worker errors
            with self._lock:
                self._state = TaskState.FAILED
                self._error = exc
                self._traceback = traceback.format_exc()
        else:
            with self._lock:
                if self._cancel_event.is_set():
                    self._state = TaskState.CANCELLED
                else:
                    self._state = TaskState.DONE
                    self._result = result

    def cancel(self) -> None:
        """Ask the worker to stop. Returns immediately; use `join` to wait."""
        self._cancel_event.set()

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the worker to finish. Returns True if it did."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def state(self) -> TaskState:
        with self._lock:
            return self._state

    @property
    def is_finished(self) -> bool:
        return self.state.is_finished

    @property
    def progress(self) -> Progress:
        return self._reporter.progress

    @property
    def result(self) -> Any:
        """The worker's return value."""
        with self._lock:
            if self._state is TaskState.DONE:
                return self._result
            if self._state is TaskState.FAILED:
                raise RuntimeError(
                    f"task {self.name!r} failed:\n{self._traceback}"
                ) from self._error
            raise RuntimeError(f"task {self.name!r} has no result (state: {self._state.value})")

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    @property
    def formatted_error(self) -> str:
        with self._lock:
            return self._traceback


@dataclass
class JobState:
    """A durable breadcrumb of what a job was doing."""

    asset: str
    stages: list[str] = field(default_factory=list)
    current_stage: str = ""
    completed: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "running"

    def mark_stage(self, stage: str) -> None:
        self.current_stage = stage
        self.updated_at = time.time()

    def mark_complete(self, stage: str) -> None:
        if stage not in self.completed:
            self.completed.append(stage)
        self.current_stage = ""
        self.updated_at = time.time()

    def finish(self, status: str) -> None:
        self.status = status
        self.current_stage = ""
        self.updated_at = time.time()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temp.replace(path)

    @classmethod
    def load(cls, path: Path) -> JobState:
        return cls(**json.loads(path.read_text(encoding="utf-8")))
