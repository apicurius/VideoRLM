"""
Logger for RLM iterations.

Captures run metadata and iterations in memory so they can be attached to
RLMChatCompletion.metadata. Optionally writes the same data to JSON-lines files.

Disk writes are performed on a background thread so they never block the main
RLM iteration loop.
"""

import json
import os
import queue
import threading
import uuid
from datetime import datetime

from rlm.core.types import RLMIteration, RLMMetadata

# Sentinel object signalling the writer thread to shut down.
_SHUTDOWN = object()


class _AsyncFileWriter:
    """Background thread that drains a queue of (path, json_str) pairs to disk."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                self._queue.task_done()
                break
            path, line = item
            try:
                with open(path, "a") as f:
                    f.write(line)
            except OSError:
                pass  # best-effort logging
            self._queue.task_done()

    def write(self, path: str, data: dict) -> None:
        """Enqueue a JSON dict to be written as a JSONL line."""
        line = json.dumps(data) + "\n"
        self._queue.put((path, line))

    def flush(self) -> None:
        """Block until all pending writes are flushed."""
        self._queue.join()

    def shutdown(self) -> None:
        """Flush and stop the writer thread."""
        self._queue.put(_SHUTDOWN)
        self._thread.join(timeout=5)


class RLMLogger:
    """
    Captures trajectory (run metadata + iterations) for each completion.
    By default only captures in memory; set log_dir to also save to disk.

    - log_dir=None: trajectory is available via get_trajectory() and can be
      attached to RLMChatCompletion.metadata (no disk write).
    - log_dir="path": same capture plus appends to a JSONL file per run.
      Disk writes are non-blocking (background thread).
    """

    def __init__(self, log_dir: str | None = None, file_name: str = "rlm"):
        self._save_to_disk = log_dir is not None
        self.log_dir = log_dir
        self.log_file_path: str | None = None
        self._writer: _AsyncFileWriter | None = None

        if self._save_to_disk and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = str(uuid.uuid4())[:8]
            self.log_file_path = os.path.join(log_dir, f"{file_name}_{timestamp}_{run_id}.jsonl")
            self._writer = _AsyncFileWriter()

        self._run_metadata: dict | None = None
        self._iterations: list[dict] = []
        self._iteration_count = 0
        self._metadata_logged = False

    def log_metadata(self, metadata: RLMMetadata) -> None:
        """Capture run metadata (and optionally write to file)."""
        if self._metadata_logged:
            return

        self._run_metadata = metadata.to_dict()
        self._metadata_logged = True

        if self._writer and self.log_file_path:
            entry = {
                "type": "metadata",
                "timestamp": datetime.now().isoformat(),
                **self._run_metadata,
            }
            self._writer.write(self.log_file_path, entry)

    def log(self, iteration: RLMIteration) -> None:
        """Capture one iteration (and optionally append to file)."""
        self._iteration_count += 1
        entry = {
            "type": "iteration",
            "iteration": self._iteration_count,
            "timestamp": datetime.now().isoformat(),
            **iteration.to_dict(),
        }
        self._iterations.append(entry)

        if self._writer and self.log_file_path:
            self._writer.write(self.log_file_path, entry)

    def clear_iterations(self) -> None:
        """Reset iterations for the next completion (trajectory is per completion)."""
        self._iterations = []
        self._iteration_count = 0

    def flush(self) -> None:
        """Flush pending disk writes (blocks until complete)."""
        if self._writer:
            self._writer.flush()

    def get_trajectory(self) -> dict | None:
        """Return captured run_metadata + iterations for the current completion, or None if no metadata yet."""
        if self._writer:
            self._writer.flush()
        if self._run_metadata is None:
            return None
        return {
            "run_metadata": self._run_metadata,
            "iterations": list(self._iterations),
        }

    @property
    def iteration_count(self) -> int:
        return self._iteration_count
