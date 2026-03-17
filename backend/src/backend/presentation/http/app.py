import logging

from fastapi import FastAPI

from src.backend.presentation.http.routes import router
from src.backend.shared.settings import get_app_settings


def create_app() -> FastAPI:
    settings = get_app_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        description="Backend для загрузки биржевых котировок, обучения моделей и выдачи прогнозов.",
    )
    app.include_router(router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.service_name,
            "version": settings.service_version,
        }

    return app

