from datetime import datetime, timezone
from typing import Optional

from celery.result import AsyncResult

from src.backend.celery_app import celery_app
from src.backend.infrastructure.persistence.database import get_session_factory
from src.backend.infrastructure.persistence.repositories.training_jobs import TrainingJobRepository
from src.backend.schemas import TrainJobStatusResponse, TrainRequest


def enqueue_training_job(job_id: str, payload: TrainRequest) -> None:
    payload_dict = payload.model_dump(mode="json")
    _persist_queued_job(job_id=job_id, model_id=payload.model_id, payload=payload_dict)
    celery_app.send_task(
        "exchange_predictor.train_model",
        kwargs={"job_id": job_id, "payload": payload_dict},
        task_id=job_id,
    )


def get_training_job_status(job_id: str) -> Optional[TrainJobStatusResponse]:
    session_factory = get_session_factory()
    with session_factory() as session:
        repository = TrainingJobRepository(session)
        record = repository.get(job_id)

    async_result = AsyncResult(job_id, app=celery_app)
    if record is None and async_result.state == "PENDING":
        return None

    status = record.status if record is not None else async_result.state.lower()
    result_payload = record.result if record is not None else _extract_result(async_result)
    error_payload = record.error if record is not None else _extract_error(async_result)
    updated_at = (
        record.updated_at
        if record is not None
        else datetime.now(timezone.utc)
    )
    return TrainJobStatusResponse(
        job_id=job_id,
        status=status,
        result=result_payload,
        error=error_payload,
        updated_at=updated_at,
    )


def _extract_result(result: AsyncResult) -> Optional[dict]:
    payload = result.result
    if isinstance(payload, dict):
        return payload
    return None


def _extract_error(result: AsyncResult) -> Optional[str]:
    if result.failed():
        return str(result.result)
    return None


def _persist_queued_job(job_id: str, model_id: str, payload: dict) -> None:
    session_factory = get_session_factory()
    with session_factory() as session:
        repository = TrainingJobRepository(session)
        repository.upsert_queued(job_id=job_id, model_id=model_id, payload=payload)

