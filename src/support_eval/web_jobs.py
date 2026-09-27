from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from support_eval.checkpoint import CHECKPOINT_FILENAME, BatchCheckpoint
from support_eval.evaluator import Evaluator
from support_eval.groq import GroqEvaluator
from support_eval.ingestion import load_conversations
from support_eval.models import StrictModel
from support_eval.pipeline import analyze

JobState = Literal["queued", "running", "complete", "failed"]
EvaluatorFactory = Callable[[Callable[[str], None]], Evaluator]


class WebJob(StrictModel):
    id: str
    state: JobState = "queued"
    display_filename: str
    conversation_filename: str
    total: int = Field(ge=1)
    status_message: str = "Waiting to start"
    error_message: str | None = None
    created_at: str
    updated_at: str


def _default_evaluator_factory(notify: Callable[[str], None]) -> Evaluator:
    return GroqEvaluator(notify=notify)


def _atomic_write_json(path: Path, value: WebJob) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class JobManager:
    """Run one local evaluation at a time and persist safe UI metadata."""

    def __init__(
        self,
        root: Path,
        *,
        evaluator_factory: EvaluatorFactory = _default_evaluator_factory,
        max_workers: int = 1,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.evaluator_factory = evaluator_factory
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="support-eval-web",
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future[None]] = {}

    def job_dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise ValueError("Invalid job identifier")
        destination = (self.root / job_id).resolve()
        if destination.parent != self.root:
            raise ValueError("Invalid job identifier")
        return destination

    def create_job(
        self,
        job_id: str,
        *,
        display_filename: str,
        conversation_filename: str,
        total: int,
        prepared: bool = False,
    ) -> WebJob:
        destination = self.job_dir(job_id)
        if not prepared:
            destination.mkdir(parents=False, exist_ok=False)
        elif not destination.is_dir():
            raise ValueError("Prepared job directory does not exist")
        now = datetime.now(UTC).isoformat()
        job = WebJob(
            id=job_id,
            display_filename=display_filename,
            conversation_filename=conversation_filename,
            total=total,
            created_at=now,
            updated_at=now,
        )
        self.save(job)
        return job

    def load(self, job_id: str) -> WebJob:
        path = self.job_dir(job_id) / "job.json"
        try:
            with self._lock:
                content = path.read_text(encoding="utf-8")
            return WebJob.model_validate_json(content)
        except FileNotFoundError:
            raise
        except (OSError, ValidationError) as exc:
            raise ValueError("Job metadata is corrupted or unreadable") from exc

    def save(self, job: WebJob) -> None:
        job.updated_at = datetime.now(UTC).isoformat()
        with self._lock:
            _atomic_write_json(self.job_dir(job.id) / "job.json", job)

    def submit(self, job_id: str) -> bool:
        with self._lock:
            existing = self._futures.get(job_id)
            if existing and not existing.done():
                return False
            self._futures[job_id] = self._executor.submit(self._run, job_id)
        return True

    def progress(self, job: WebJob) -> int:
        checkpoint_path = self.job_dir(job.id) / CHECKPOINT_FILENAME
        if job.state == "complete":
            return job.total
        if not checkpoint_path.exists():
            return 0
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint = BatchCheckpoint.model_validate(payload)
        except (OSError, json.JSONDecodeError, ValidationError):
            return 0
        return min(len(checkpoint.completed), job.total)

    def _notify(self, job_id: str, message: str) -> None:
        job = self.load(job_id)
        job.status_message = self._safe_message(message)
        self.save(job)

    def _run(self, job_id: str) -> None:
        job = self.load(job_id)
        job.state = "running"
        job.error_message = None
        job.status_message = "Evaluating conversations with Groq"
        self.save(job)
        destination = self.job_dir(job_id)
        try:
            evaluator = self.evaluator_factory(lambda message: self._notify(job_id, message))
            analyze(
                destination / job.conversation_filename,
                destination / "rubric.json",
                destination,
                evaluator,
            )
        except Exception as exc:  # noqa: BLE001 - boundary converts failures into safe job state
            job = self.load(job_id)
            job.state = "failed"
            job.status_message = "Evaluation stopped"
            job.error_message = self._safe_error(exc)
            self.save(job)
            return
        job = self.load(job_id)
        job.state = "complete"
        job.status_message = "Evaluation complete"
        job.error_message = None
        self.save(job)

    @staticmethod
    def _safe_message(message: str) -> str:
        compact = " ".join(message.split())
        return compact[:300] if compact else "Evaluation in progress"

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        text = " ".join(str(exc).split())
        api_key = os.environ.get("GROQ_API_KEY", "")
        if api_key:
            text = text.replace(api_key, "[REDACTED]")
        text = re.sub(r"(?i)Bearer\s+[^\s,;\"']+", "Bearer [REDACTED]", text)
        text = re.sub(
            r"(?i)(api[_ -]?key|authorization|token|credential|secret)"
            r"(\s*[:=]\s*[\"']?)[^\s,;\"'}]+",
            r"\1\2[REDACTED]",
            text,
        )
        if not text or isinstance(exc, (KeyError, AssertionError)):
            return "Evaluation failed unexpectedly. Your completed work is preserved."
        return text[:800]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


def count_saved_conversations(job_dir: Path, filename: str) -> int:
    return len(load_conversations(job_dir / filename))
