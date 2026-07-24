"""Shared GPU/CPU inference scheduling with frame-skip awareness."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional


@dataclass
class InferJob:
    camera_key: str
    frame: Any
    modules: list[str]
    submitted_at: float = field(default_factory=time.time)
    callback: Optional[Callable[[str, list], None]] = None


class InferScheduler:
    """Serializes model inference so many cameras share one device without thrashing."""

    def __init__(self, max_queue: int = 64):
        self._queue: Deque[InferJob] = deque()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._processor: Optional[Callable[[InferJob], list]] = None
        self.max_queue = max_queue
        self.processed = 0
        self.dropped = 0
        self.last_infer_ms = 0.0
        self.queue_depth = 0

    def set_processor(self, processor: Callable[[InferJob], list]) -> None:
        self._processor = processor

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="infer-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        with self._cv:
            self._cv.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def submit(self, job: InferJob) -> bool:
        with self._cv:
            # Drop oldest for same camera to keep latency low
            self._queue = deque(item for item in self._queue if item.camera_key != job.camera_key)
            if len(self._queue) >= self.max_queue:
                self._queue.popleft()
                self.dropped += 1
            self._queue.append(job)
            self.queue_depth = len(self._queue)
            self._cv.notify()
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queue_depth": len(self._queue),
                "processed": self.processed,
                "dropped": self.dropped,
                "last_infer_ms": round(self.last_infer_ms, 1),
                "running": self._running,
            }

    def _loop(self) -> None:
        while self._running:
            with self._cv:
                while self._running and not self._queue:
                    self._cv.wait(timeout=0.5)
                if not self._running:
                    break
                job = self._queue.popleft()
                self.queue_depth = len(self._queue)

            if self._processor is None:
                continue
            t0 = time.time()
            try:
                events = self._processor(job)
            except Exception as exc:
                events = []
                print(f"[infer] error camera={job.camera_key}: {exc}")
            self.last_infer_ms = (time.time() - t0) * 1000.0
            self.processed += 1
            if job.callback:
                try:
                    job.callback(job.camera_key, events)
                except Exception as exc:
                    print(f"[infer] callback error: {exc}")


# Global singleton used by shard manager / workers
SCHEDULER = InferScheduler()
