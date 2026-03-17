from celery import Celery

from src.backend.shared.settings import get_app_settings


def create_celery_app() -> Celery:
    settings = get_app_settings()
    app = Celery(
        "exchange_predictor",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["src.backend.infrastructure.queue.tasks"],
    )
    app.conf.update(
        task_default_queue="exchange_predictor",
        task_track_started=True,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = create_celery_app()

