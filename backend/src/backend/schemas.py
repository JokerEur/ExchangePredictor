from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

AllowedSource = Literal["local", "exchange"]
TrainMode = Literal["train", "finetune", "retrain"]
RequestedModelType = Literal["random_forest", "xgboost"]


class DataSourceRequest(BaseModel):
    source: AllowedSource = Field(default="local")
    symbol: str = Field(default="BTC/USD", min_length=3, max_length=30)
    symbols: Optional[list[str]] = Field(
        default=None,
        description="Список криптопар для универсального обучения (например BTC/USD, ETH/USD, SOL/USD).",
    )
    timeframe: str = Field(default="1d", min_length=2, max_length=8)
    forecast_days: int = Field(default=5, ge=1, le=30)
    prediction_scope: Optional[int] = Field(
        default=None,
        ge=0,
        le=10000,
        description="Legacy-параметр. Если задан, переопределяет forecast_days.",
    )
    validation_fraction: float = Field(default=0.2, gt=0.05, lt=0.5)
    exchange_limit: int = Field(default=1500, ge=300, le=5000)
    data_path: Optional[str] = Field(default=None)


class MarketCandlesRequest(BaseModel):
    source: AllowedSource = Field(default="exchange")
    symbol: str = Field(default="BTC/USD", min_length=3, max_length=30)
    timeframe: str = Field(default="1h", min_length=2, max_length=8)
    limit: int = Field(default=300, ge=20, le=5000)
    data_path: Optional[str] = Field(default=None)


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketCandlesResponse(BaseModel):
    source: str
    symbol: str
    timeframe: str
    candles: list[Candle]
    last_candle_at: datetime
    generated_at: datetime
    data_lag_minutes: float
    is_fresh: bool


class PredictionMetrics(BaseModel):
    mae: float
    mape: float
    rmse: float
    mse: float
class DataWindow(BaseModel):
    symbol: str
    rows: int
    start_at: datetime
    end_at: datetime


class ModelDescriptor(BaseModel):
    model_id: str
    model_type: str
    mode: TrainMode
    tuned: bool
    trained_at: datetime
    best_params: dict[str, Any]


class TrainRequest(DataSourceRequest):
    mode: TrainMode = Field(default="train")
    model_id: str = Field(default="default", min_length=1, max_length=100)
    model_type: RequestedModelType = Field(default="xgboost")
    tune: bool = Field(default=True)
    tune_trials: int = Field(default=24, ge=4, le=300)
    persist_model: bool = Field(default=True)
    full_refit: bool = Field(default=True)


class TrainResponse(BaseModel):
    source: str
    symbol: str
    symbols: list[str]
    timeframe: str
    forecast_days: int
    prediction_scope: int
    model: ModelDescriptor
    metrics: PredictionMetrics
    train_rows: int
    validation_rows: int
    feature_count: int
    data_windows: list[DataWindow]
    data_is_fresh: bool
    max_data_lag_minutes: float
    exchange_limit_used: Optional[int] = None
    loss_function: str
    tuning_scoring: str
    generated_at: datetime


class PredictRequest(DataSourceRequest):
    model_id: str = Field(default="default", min_length=1, max_length=100)
    use_saved_model: bool = Field(default=True)
    auto_train_if_missing: bool = Field(default=True)
    auto_train_mode: TrainMode = Field(default="train")
    auto_model_type: RequestedModelType = Field(default="xgboost")
    tune_on_auto_train: bool = Field(default=True)
    tune_trials: int = Field(default=16, ge=4, le=300)


class DailyForecastPoint(BaseModel):
    day_ahead: int
    predict_for_at: datetime
    predicted_price: float


class PredictionResponse(BaseModel):
    source: str
    symbol: str
    timeframe: str
    forecast_days: int
    prediction_scope: int
    model_id: str
    model_type: str
    used_saved_model: bool
    last_observation_at: datetime
    predict_for_at: datetime
    predicted_price: float
    daily_path: list[DailyForecastPoint]
    metrics: PredictionMetrics
    train_rows: int
    validation_rows: int
    feature_count: int
    model_trained_at: Optional[datetime] = None
    training_data_windows: list[DataWindow] = Field(default_factory=list)
    training_data_is_fresh: Optional[bool] = None
    training_max_data_lag_minutes: Optional[float] = None
    training_exchange_limit: Optional[int] = None
    loss_function: Optional[str] = None
    tuning_scoring: Optional[str] = None
    request_data_windows: list[DataWindow] = Field(default_factory=list)
    request_data_is_fresh: bool
    request_max_data_lag_minutes: float
    generated_at: datetime


class ModelSummary(BaseModel):
    model_id: str
    model_type: str
    timeframe: str
    symbols: Optional[list[str]] = None
    forecast_days: Optional[int] = None
    prediction_scope: int
    trained_at: datetime
    metrics: Optional[PredictionMetrics] = None


class ModelRegistryResponse(BaseModel):
    models: list[ModelSummary]

class FeaturePreviewRequest(DataSourceRequest):
    history_size: int = Field(default=72, ge=20, le=300)


class FeatureHistoryPoint(BaseModel):
    timestamp: datetime
    value: float


class FeatureDescriptor(BaseModel):
    name: str
    category: str
    scale_hint: str
    latest_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    history: list[FeatureHistoryPoint] = Field(default_factory=list)


class FeaturePreviewResponse(BaseModel):
    source: str
    symbol: str
    timeframe: str
    forecast_days: int
    prediction_scope: int
    feature_count: int
    history_size: int
    features: list[FeatureDescriptor]
    generated_at: datetime


class TrainJobEnqueueResponse(BaseModel):
    job_id: str
    status: str
    queued_at: datetime


class TrainJobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    updated_at: datetime
