from src.backend.presentation.http.app import create_app
from src.backend.shared.settings import get_app_settings

app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_app_settings()

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
