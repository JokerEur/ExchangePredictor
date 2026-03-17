from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    service_name: str = "ExchangePredictor API"
    service_version: str = "2.1.0"
    default_symbol: str = "BTC/USD"
    default_timeframe: str = "1d"
    default_data_path: Path = Path(__file__).resolve().parents[2] / "data.csv"
    model_config_path: Path = Path(__file__).resolve().parents[2] / "config" / "model_params.yml"
    model_registry_path: Path = Path(__file__).resolve().parents[2] / "model" / "registry"
    default_model_id: str = "default"
    min_training_rows: int = 120
    min_train_rows_after_split: int = 40
    min_validation_rows_after_split: int = 20
    tuning_cv_splits: int = 3
    random_state: int = 42


settings = Settings()
