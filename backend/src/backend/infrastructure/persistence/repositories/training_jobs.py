from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.backend.infrastructure.persistence.models import TrainingJobRecord


class TrainingJobRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert_queued(self, job_id: str, model_id: str, payload: dict) -> None:
        record = self._session.scalar(
            select(TrainingJobRecord).where(TrainingJobRecord.job_id == job_id),
        )
        if record is None:
            record = TrainingJobRecord(
                job_id=job_id,
                model_id=model_id,
                status="queued",
                payload=payload,
                result=None,
                error=None,
            )
            self._session.add(record)
        else:
            record.status = "queued"
            record.model_id = model_id
            record.payload = payload
            record.result = None
            record.error = None
            record.updated_at = datetime.now(timezone.utc)
        self._session.commit()

    def mark_running(self, job_id: str) -> None:
        record = self._find(job_id)
        if record is None:
            return
        record.status = "running"
        record.updated_at = datetime.now(timezone.utc)
        self._session.commit()

    def mark_succeeded(self, job_id: str, result: dict) -> None:
        record = self._find(job_id)
        if record is None:
            return
        record.status = "succeeded"
        record.result = result
        record.error = None
        record.updated_at = datetime.now(timezone.utc)
        self._session.commit()

    def mark_failed(self, job_id: str, error: str) -> None:
        record = self._find(job_id)
        if record is None:
            return
        record.status = "failed"
        record.error = error
        record.updated_at = datetime.now(timezone.utc)
        self._session.commit()

    def get(self, job_id: str) -> Optional[TrainingJobRecord]:
        return self._find(job_id)

    def _find(self, job_id: str) -> Optional[TrainingJobRecord]:
        return self._session.scalar(
            select(TrainingJobRecord).where(TrainingJobRecord.job_id == job_id),
        )

