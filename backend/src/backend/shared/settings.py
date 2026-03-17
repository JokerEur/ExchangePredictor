import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from src.backend.config import Settings as LegacySettings


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppSettings:
    service_name: str
    service_version: str
    api_host: str
    api_port: int
    default_symbol: str
    default_timeframe: str
    project_root: Path
    default_data_path: Path
    model_config_path: Path
    model_registry_path: Path
    default_model_id: str
    min_training_rows: int
    min_train_rows_after_split: int
    min_validation_rows_after_split: int
    tuning_cv_splits: int
    random_state: int
    database_url: str
    redis_url: str
    celery_broker_url: str
    celery_result_backend: str

    @staticmethod
    def from_env() -> "AppSettings":
        project_root = Path(
            os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[3])
        )
        default_data_path = Path(
            os.getenv("DEFAULT_DATA_PATH", project_root / "data.csv")
        )
        model_config_path = Path(
            os.getenv("MODEL_CONFIG_PATH", project_root / "config" / "model_params.yml")
        )
        model_registry_path = Path(
            os.getenv("MODEL_REGISTRY_PATH", project_root / "model" / "registry")
        )
        return AppSettings(
            service_name=os.getenv("SERVICE_NAME", "ExchangePredictor API"),
            service_version=os.getenv("SERVICE_VERSION", "3.0.0"),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=_env_int("API_PORT", 8000),
            default_symbol=os.getenv("DEFAULT_SYMBOL", "BTC/USD"),
            default_timeframe=os.getenv("DEFAULT_TIMEFRAME", "1d"),
            project_root=project_root,
            default_data_path=default_data_path,
            model_config_path=model_config_path,
            model_registry_path=model_registry_path,
            default_model_id=os.getenv("DEFAULT_MODEL_ID", "default"),
            min_training_rows=_env_int("MIN_TRAINING_ROWS", 120),
            min_train_rows_after_split=_env_int("MIN_TRAIN_ROWS_AFTER_SPLIT", 40),
            min_validation_rows_after_split=_env_int("MIN_VALIDATION_ROWS_AFTER_SPLIT", 20),
            tuning_cv_splits=_env_int("TUNING_CV_SPLITS", 3),
            random_state=_env_int("RANDOM_STATE", 42),
            database_url=os.getenv("DATABASE_URL", "sqlite:///./exchange_predictor.db"),
            redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
            celery_broker_url=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/1"),
            celery_result_backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/2"),
        )


@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    return AppSettings.from_env()


def to_legacy_settings(settings: AppSettings) -> LegacySettings:
    return LegacySettings(
        service_name=settings.service_name,
        service_version=settings.service_version,
        default_symbol=settings.default_symbol,
        default_timeframe=settings.default_timeframe,
        default_data_path=settings.default_data_path,
        model_config_path=settings.model_config_path,
        model_registry_path=settings.model_registry_path,
        default_model_id=settings.default_model_id,
        min_training_rows=settings.min_training_rows,
        min_train_rows_after_split=settings.min_train_rows_after_split,
        min_validation_rows_after_split=settings.min_validation_rows_after_split,
        tuning_cv_splits=settings.tuning_cv_splits,
        random_state=settings.random_state,
    )

