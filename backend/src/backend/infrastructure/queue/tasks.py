from src.backend.infrastructure.legacy.prediction_gateway import LegacyPredictionGateway
from src.backend.infrastructure.persistence.database import get_session_factory
from src.backend.infrastructure.persistence.repositories.training_jobs import TrainingJobRepository
from src.backend.schemas import TrainRequest
from src.backend.shared.settings import get_app_settings, to_legacy_settings

from src.backend.celery_app import celery_app


@celery_app.task(name="exchange_predictor.train_model", bind=True)
def train_model_task(self, job_id: str, payload: dict) -> dict:  # pylint: disable=unused-argument
    session_factory = get_session_factory()
    with session_factory() as session:
        repository = TrainingJobRepository(session)
        repository.mark_running(job_id)

    try:
        request = TrainRequest(**payload)
        settings = get_app_settings()
        gateway = LegacyPredictionGateway.from_settings(
            settings=to_legacy_settings(settings),
        )
        response = gateway.train_model(request)
        response_payload = response.model_dump(mode="json")
        with session_factory() as session:
            repository = TrainingJobRepository(session)
            repository.mark_succeeded(job_id, response_payload)
        return response_payload
    except Exception as exc:  # pylint: disable=broad-except
        with session_factory() as session:
            repository = TrainingJobRepository(session)
            repository.mark_failed(job_id, str(exc))
        raise
