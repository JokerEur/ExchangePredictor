import logging
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, TimeSeriesSplit
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from .config import Settings
from .data_provider import DataProvider
from .features import build_feature_frame
from .model_params import load_model_params
from .model_store import ModelStore, StoredModel
from .schemas import (
    Candle,
    DataWindow,
    DailyForecastPoint,
    DataSourceRequest,
    FeatureDescriptor,
    FeatureHistoryPoint,
    FeaturePreviewRequest,
    FeaturePreviewResponse,
    MarketCandlesRequest,
    MarketCandlesResponse,
    ModelDescriptor,
    ModelRegistryResponse,
    ModelSummary,
    PredictRequest,
    PredictionMetrics,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
)

TIMEFRAME_PATTERN = re.compile(r"^(\d+)\s*([mhdw])$", re.IGNORECASE)
UNIT_TO_SECONDS = {"m": 60, "h": 60 * 60, "d": 60 * 60 * 24, "w": 60 * 60 * 24 * 7}
MAX_FORECAST_DAYS = 30
XGBOOST_HUBER_MODE_VERSION = "scaled_target_median_v1"
ESTIMATOR_REGISTRY = {
    "random_forest": RandomForestRegressor,
    "xgboost": XGBRegressor,
}
LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedDataset:
    source: str
    primary_symbol: str
    symbols: list[str]
    timeframe: str
    forecast_days: int
    day_scope_map: dict[int, int]
    validation_fraction: float
    ohlcv_map: dict[str, pd.DataFrame]
    data_windows: list[DataWindow]
    data_is_fresh: bool
    max_data_lag_minutes: float
    exchange_limit_used: Optional[int]


@dataclass
class ScopeMatrices:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    feature_columns: list[str]


@dataclass
class TrainOutcome:
    model: Any
    metadata: dict[str, Any]
    response: TrainResponse


class PredictionService:
    def __init__(self, settings: Settings, data_provider: Optional[DataProvider] = None) -> None:
        self._settings = settings
        self._data_provider = data_provider or DataProvider()
        self._model_params = load_model_params(settings.model_config_path)
        self._model_store = ModelStore(settings.model_registry_path)
        LOGGER.info(
            "PredictionService initialized | model_config=%s | model_registry=%s",
            settings.model_config_path,
            settings.model_registry_path,
        )

    def get_market_candles(self, request: MarketCandlesRequest) -> MarketCandlesResponse:
        if request.source == "local":
            csv_path = Path(request.data_path).expanduser() if request.data_path else self._settings.default_data_path
            frame = self._data_provider.load_local_csv(csv_path)
            frame = frame.tail(request.limit)
            symbol = "LOCAL_DATA"
        else:
            frame = self._data_provider.fetch_exchange_ohlcv(
                symbol=request.symbol.strip().upper(),
                timeframe=request.timeframe.strip().lower(),
                limit=request.limit,
            )
            symbol = request.symbol.strip().upper()

        candles = [
            Candle(
                timestamp=index.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
            for index, row in frame.iterrows()
        ]

        last_candle_at = frame.index[-1].to_pydatetime()
        generated_at = datetime.now(timezone.utc)
        generated_at_naive = generated_at.replace(tzinfo=None)
        data_lag_minutes = max(0.0, (generated_at_naive - last_candle_at).total_seconds() / 60.0)

        timeframe_step_minutes = (
            _timeframe_to_timedelta(request.timeframe.strip().lower(), 1).total_seconds() / 60.0
        )
        freshness_threshold_minutes = max(3.0, timeframe_step_minutes * 2.0)
        is_fresh = data_lag_minutes <= freshness_threshold_minutes
        return MarketCandlesResponse(
            source=request.source,
            symbol=symbol,
            timeframe=request.timeframe.strip().lower(),
            candles=candles,
            last_candle_at=last_candle_at,
            generated_at=generated_at,
            data_lag_minutes=data_lag_minutes,
            is_fresh=is_fresh,
        )

    def get_feature_preview(self, request: FeaturePreviewRequest) -> FeaturePreviewResponse:
        prepared = self._prepare_dataset(request)
        prediction_scope = max(prepared.day_scope_map.values())
        symbol = prepared.primary_symbol
        dataset, _, _ = build_feature_frame(
            ohlcv=prepared.ohlcv_map[symbol],
            prediction_scope=prediction_scope,
        )
        dataset = self._add_symbol_features(dataset, symbol, prepared.symbols)
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        feature_columns = [column for column in dataset.columns if column != "target"]

        features: list[FeatureDescriptor] = []
        for column in feature_columns:
            series = pd.to_numeric(dataset[column], errors="coerce").dropna().tail(request.history_size)
            if series.empty:
                history: list[FeatureHistoryPoint] = []
                latest_value = None
                min_value = None
                max_value = None
                mean_value = None
            else:
                history = [
                    FeatureHistoryPoint(
                        timestamp=index.to_pydatetime(),
                        value=float(value),
                    )
                    for index, value in series.items()
                ]
                latest_value = float(series.iloc[-1])
                min_value = float(series.min())
                max_value = float(series.max())
                mean_value = float(series.mean())

            category, scale_hint = _feature_profile(column)
            features.append(
                FeatureDescriptor(
                    name=column,
                    category=category,
                    scale_hint=scale_hint,
                    latest_value=latest_value,
                    min_value=min_value,
                    max_value=max_value,
                    mean_value=mean_value,
                    history=history,
                )
            )

        return FeaturePreviewResponse(
            source=prepared.source,
            symbol=symbol,
            timeframe=prepared.timeframe,
            forecast_days=prepared.forecast_days,
            prediction_scope=prediction_scope,
            feature_count=len(feature_columns),
            history_size=request.history_size,
            features=features,
            generated_at=datetime.now(timezone.utc),
        )

    def train_model(self, request: TrainRequest) -> TrainResponse:
        LOGGER.info(
            "Training started | model_id=%s | model_type=%s | mode=%s | source=%s | timeframe=%s | forecast_days=%s",
            request.model_id,
            request.model_type,
            request.mode,
            request.source,
            request.timeframe,
            request.forecast_days,
        )
        outcome = self._run_training(request)
        if request.persist_model:
            self._model_store.save(
                model_id=request.model_id,
                model=outcome.model,
                metadata=outcome.metadata,
            )
            LOGGER.info("Model persisted | model_id=%s", request.model_id)
        LOGGER.info(
            "Training finished | model_id=%s | model_type=%s | train_rows=%s | validation_rows=%s | feature_count=%s",
            outcome.response.model.model_id,
            outcome.response.model.model_type,
            outcome.response.train_rows,
            outcome.response.validation_rows,
            outcome.response.feature_count,
        )
        return outcome.response

    def predict(self, request: PredictRequest) -> PredictionResponse:
        LOGGER.info(
            "Prediction started | model_id=%s | requested_model_type=%s | source=%s | symbol=%s | timeframe=%s | forecast_days=%s",
            request.model_id,
            request.auto_model_type,
            request.source,
            request.symbol,
            request.timeframe,
            request.forecast_days,
        )
        prepared = self._prepare_dataset(request)
        used_saved_model = False

        model_bundle: dict[str, Any]
        metadata: dict[str, Any]

        if request.use_saved_model:
            try:
                stored = self._model_store.load(request.model_id)
                self._validate_stored_model(stored, prepared)
                model_bundle, metadata = self._normalize_stored_bundle(stored.model, stored.metadata)
                requested_type = request.auto_model_type
                stored_type = str(metadata.get("model_type", "")).strip().lower()
                if stored_type != requested_type:
                    raise ValueError(
                        "Сохраненная модель имеет другой тип. "
                        "Будет запущено автообучение c запрошенным auto_model_type."
                    )
                used_saved_model = True
                LOGGER.info("Using saved model | model_id=%s | model_type=%s", request.model_id, stored_type)
            except ValueError:
                if not request.auto_train_if_missing:
                    raise
                LOGGER.info(
                    "Saved model unavailable or mismatched. Auto-training triggered | model_id=%s | model_type=%s",
                    request.model_id,
                    request.auto_model_type,
                )
                auto_train_request = TrainRequest(
                    mode=request.auto_train_mode,
                    model_id=request.model_id,
                    model_type=request.auto_model_type,
                    tune=request.tune_on_auto_train,
                    tune_trials=request.tune_trials,
                    persist_model=True,
                    full_refit=True,
                    source=request.source,
                    symbol=request.symbol,
                    symbols=request.symbols,
                    timeframe=request.timeframe,
                    forecast_days=request.forecast_days,
                    prediction_scope=request.prediction_scope,
                    validation_fraction=request.validation_fraction,
                    exchange_limit=request.exchange_limit,
                    data_path=request.data_path,
                )
                outcome = self._run_training(auto_train_request)
                self._model_store.save(
                    model_id=auto_train_request.model_id,
                    model=outcome.model,
                    metadata=outcome.metadata,
                )
                model_bundle = outcome.model
                metadata = outcome.metadata
        else:
            train_request = TrainRequest(
                mode="train",
                model_id=request.model_id,
                model_type=request.auto_model_type,
                tune=request.tune_on_auto_train,
                tune_trials=request.tune_trials,
                persist_model=False,
                full_refit=True,
                source=request.source,
                symbol=request.symbol,
                symbols=request.symbols,
                timeframe=request.timeframe,
                forecast_days=request.forecast_days,
                prediction_scope=request.prediction_scope,
                validation_fraction=request.validation_fraction,
                exchange_limit=request.exchange_limit,
                data_path=request.data_path,
            )
            outcome = self._run_training(train_request)
            model_bundle = outcome.model
            metadata = outcome.metadata

        feature_columns = metadata.get("feature_columns", [])
        if not feature_columns:
            raise ValueError("В metadata нет feature_columns, переобучите модель.")

        supported_symbols = metadata.get("symbols", [])
        if not isinstance(supported_symbols, list) or not supported_symbols:
            legacy_symbol = metadata.get("symbol")
            supported_symbols = [legacy_symbol] if legacy_symbol else [prepared.primary_symbol]
        supported_symbols = [str(symbol).upper() for symbol in supported_symbols]

        forecast_path: list[DailyForecastPoint] = []
        for day_ahead in range(1, prepared.forecast_days + 1):
            scope = prepared.day_scope_map[day_ahead]
            model = self._get_model_for_scope(model_bundle, scope)
            latest_vector = self._build_latest_feature_vector(
                prepared=prepared,
                symbol=prepared.primary_symbol,
                prediction_scope=scope,
                supported_symbols=supported_symbols,
                feature_columns=feature_columns,
            )
            predicted_price = float(model.predict(latest_vector)[0])
            predict_for_at = (
                prepared.ohlcv_map[prepared.primary_symbol].index[-1].to_pydatetime()
                + _timeframe_to_timedelta(prepared.timeframe, scope + 1)
            )
            forecast_path.append(
                DailyForecastPoint(
                    day_ahead=day_ahead,
                    predict_for_at=predict_for_at,
                    predicted_price=predicted_price,
                )
            )

        if not forecast_path:
            raise ValueError("Не удалось построить траекторию прогноза.")

        final_scope = prepared.day_scope_map[prepared.forecast_days]
        final_point = forecast_path[-1]

        metrics_by_scope = metadata.get("metrics_by_scope", {})
        metrics_data = metrics_by_scope.get(str(final_scope), metadata.get("metrics"))
        if not metrics_data:
            raise ValueError("В metadata модели отсутствуют метрики.")
        model_type = str(metadata.get("model_type", "unknown"))
        best_params_raw = metadata.get("best_params", {})
        best_params = best_params_raw if isinstance(best_params_raw, dict) else {}
        display_loss_function = _resolve_loss_function(model_type, best_params)
        display_tuning_scoring = _display_scoring_name(
            _normalize_scoring_name(
                metadata.get(
                    "tuning_scoring",
                    self._model_params.get("tuning", {}).get("scoring", "mae"),
                )
            )
        )

        response = PredictionResponse(
            source=prepared.source,
            symbol=prepared.primary_symbol if prepared.source == "exchange" else "LOCAL_DATA",
            timeframe=prepared.timeframe,
            forecast_days=prepared.forecast_days,
            prediction_scope=final_scope,
            model_id=metadata.get("model_id", request.model_id),
            model_type=model_type,
            used_saved_model=used_saved_model,
            last_observation_at=prepared.ohlcv_map[prepared.primary_symbol].index[-1].to_pydatetime(),
            predict_for_at=final_point.predict_for_at,
            predicted_price=final_point.predicted_price,
            daily_path=forecast_path,
            metrics=PredictionMetrics(**metrics_data),
            train_rows=int(metadata.get("train_rows", 0)),
            validation_rows=int(metadata.get("validation_rows", 0)),
            feature_count=int(metadata.get("feature_count", len(feature_columns))),
            model_trained_at=(
                datetime.fromisoformat(metadata["trained_at"])
                if isinstance(metadata.get("trained_at"), str)
                else None
            ),
            training_data_windows=[
                DataWindow(**item)
                for item in metadata.get("data_windows", [])
                if isinstance(item, dict)
            ],
            training_data_is_fresh=(
                bool(metadata["data_is_fresh"])
                if metadata.get("data_is_fresh") is not None
                else None
            ),
            training_max_data_lag_minutes=(
                float(metadata["max_data_lag_minutes"])
                if metadata.get("max_data_lag_minutes") is not None
                else None
            ),
            training_exchange_limit=(
                int(metadata["exchange_limit_used"])
                if metadata.get("exchange_limit_used") is not None
                else None
            ),
            loss_function=display_loss_function,
            tuning_scoring=display_tuning_scoring,
            request_data_windows=prepared.data_windows,
            request_data_is_fresh=prepared.data_is_fresh,
            request_max_data_lag_minutes=prepared.max_data_lag_minutes,
            generated_at=datetime.now(timezone.utc),
        )
        LOGGER.info(
            "Prediction finished | model_id=%s | model_type=%s | used_saved=%s | horizon_days=%s",
            response.model_id,
            response.model_type,
            response.used_saved_model,
            response.forecast_days,
        )
        return response

    def list_models(self) -> ModelRegistryResponse:
        raw_items = self._model_store.list_metadata()
        models: list[ModelSummary] = []

        for item in raw_items:
            trained_at_raw = item.get("trained_at")
            if not trained_at_raw:
                continue
            metrics = PredictionMetrics(**item["metrics"]) if item.get("metrics") else None
            models.append(
                ModelSummary(
                    model_id=item.get("model_id", "unknown"),
                    model_type=item.get("model_type", "unknown"),
                    timeframe=item.get("timeframe", "unknown"),
                    symbols=item.get("symbols"),
                    forecast_days=item.get("forecast_days"),
                    prediction_scope=int(item.get("prediction_scope", 0)),
                    trained_at=datetime.fromisoformat(trained_at_raw),
                    metrics=metrics,
                )
            )
        models.sort(key=lambda value: value.trained_at, reverse=True)
        return ModelRegistryResponse(models=models)

    def _run_training(self, request: TrainRequest) -> TrainOutcome:
        existing_metadata = self._load_existing_metadata_if_needed(request)
        effective_request = self._enrich_request_with_existing_symbols(request, existing_metadata)
        LOGGER.info(
            "Stage: prepare_dataset | model_id=%s | symbols=%s",
            request.model_id,
            effective_request.symbols or [effective_request.symbol],
        )
        prepared = self._prepare_dataset(effective_request)
        if prepared.source == "exchange" and not prepared.data_is_fresh:
            raise ValueError(
                "Обучение остановлено: рыночные данные устарели "
                f"(max lag={prepared.max_data_lag_minutes:.1f} мин). "
                "Запросите свежие данные и повторите обучение."
            )

        tune = request.tune
        if request.mode == "retrain":
            tune = False

        candidate_types = self._resolve_candidate_types(request, existing_metadata)
        best_result: Optional[dict[str, Any]] = None

        max_day = prepared.forecast_days
        max_scope = prepared.day_scope_map[max_day]
        max_scope_matrices = self._build_scope_matrices(prepared, max_scope)
        tuning_scoring = _normalize_scoring_name(
            self._model_params.get("tuning", {}).get("scoring", "mae")
        )
        tuning_scoring_label = _display_scoring_name(tuning_scoring)
        LOGGER.info(
            "Stage: train_search | model_id=%s | candidate_types=%s | tuning=%s | scoring=%s",
            request.model_id,
            candidate_types,
            tune,
            tuning_scoring_label,
        )

        for model_type in candidate_types:
            LOGGER.info("Stage: evaluate_candidate | model_id=%s | model_type=%s", request.model_id, model_type)
            base_params = self._resolve_base_params(model_type, request, existing_metadata)
            result = self._fit_and_evaluate(
                model_type=model_type,
                scope_matrices=max_scope_matrices,
                tune=tune,
                tune_trials=request.tune_trials,
                full_refit=request.full_refit,
                base_params=base_params,
                scoring=tuning_scoring,
                forecast_days=prepared.forecast_days,
            )
            if not best_result or result["selection_loss"] < best_result["selection_loss"]:
                best_result = result

        if not best_result:
            raise ValueError("Не удалось обучить модель.")

        best_params = _json_safe(best_result["best_params"])
        trained_model_type = best_result["model_type"]
        unique_scopes = sorted(set(prepared.day_scope_map.values()))

        models_by_scope: dict[str, Any] = {}
        metrics_by_scope: dict[str, dict[str, float]] = {}
        train_rows = 0
        validation_rows = 0
        feature_columns: list[str] = max_scope_matrices.feature_columns

        for scope in unique_scopes:
            LOGGER.info(
                "Stage: fit_scope_model | model_id=%s | model_type=%s | scope=%s",
                request.model_id,
                trained_model_type,
                scope,
            )
            scope_matrices = max_scope_matrices if scope == max_scope else self._build_scope_matrices(prepared, scope)
            scope_result = self._fit_and_evaluate(
                model_type=trained_model_type,
                scope_matrices=scope_matrices,
                tune=False,
                tune_trials=1,
                full_refit=request.full_refit,
                base_params=best_result["best_params"],
                scoring=tuning_scoring,
                forecast_days=prepared.forecast_days,
            )
            models_by_scope[str(scope)] = scope_result["model"]
            metrics_by_scope[str(scope)] = scope_result["metrics"].model_dump()
            train_rows = len(scope_matrices.x_train)
            validation_rows = len(scope_matrices.x_validation)

        trained_at = datetime.now(timezone.utc)
        model_bundle = {
            "bundle_version": 1,
            "models_by_scope": models_by_scope,
        }

        symbol_response = (
            prepared.primary_symbol if (prepared.source != "exchange" or len(prepared.symbols) == 1) else "UNIVERSAL"
        )
        day_scope_map_json = {str(day): int(scope) for day, scope in prepared.day_scope_map.items()}
        final_metrics = metrics_by_scope.get(str(max_scope), best_result["metrics"].model_dump())
        loss_function = _resolve_loss_function(trained_model_type, best_result["best_params"])

        metadata = {
            "model_id": request.model_id,
            "model_type": trained_model_type,
            "mode": request.mode,
            "tuned": bool(best_result["tuned"]),
            "source": prepared.source,
            "symbol": symbol_response,
            "symbols": prepared.symbols,
            "timeframe": prepared.timeframe,
            "forecast_days": prepared.forecast_days,
            "prediction_scope": max_scope,
            "day_scope_map": day_scope_map_json,
            "feature_columns": feature_columns,
            "best_params": best_params,
            "metrics": final_metrics,
            "metrics_by_scope": metrics_by_scope,
            "train_rows": train_rows,
            "validation_rows": validation_rows,
            "feature_count": len(feature_columns),
            "trained_at": trained_at.isoformat(),
            "data_windows": [window.model_dump(mode="json") for window in prepared.data_windows],
            "data_is_fresh": prepared.data_is_fresh,
            "max_data_lag_minutes": prepared.max_data_lag_minutes,
            "exchange_limit_used": prepared.exchange_limit_used,
            "loss_function": loss_function,
            "tuning_scoring": tuning_scoring_label,
            "xgboost_huber_mode": (
                XGBOOST_HUBER_MODE_VERSION if trained_model_type == "xgboost" else None
            ),
        }

        response = TrainResponse(
            source=prepared.source,
            symbol=symbol_response,
            symbols=prepared.symbols,
            timeframe=prepared.timeframe,
            forecast_days=prepared.forecast_days,
            prediction_scope=max_scope,
            model=ModelDescriptor(
                model_id=request.model_id,
                model_type=trained_model_type,
                mode=request.mode,
                tuned=bool(best_result["tuned"]),
                trained_at=trained_at,
                best_params=best_params,
            ),
            metrics=PredictionMetrics(**final_metrics),
            train_rows=train_rows,
            validation_rows=validation_rows,
            feature_count=len(feature_columns),
            data_windows=prepared.data_windows,
            data_is_fresh=prepared.data_is_fresh,
            max_data_lag_minutes=prepared.max_data_lag_minutes,
            exchange_limit_used=prepared.exchange_limit_used,
            loss_function=loss_function,
            tuning_scoring=tuning_scoring_label,
            generated_at=datetime.now(timezone.utc),
        )

        return TrainOutcome(model=model_bundle, metadata=metadata, response=response)

    def _prepare_dataset(self, request: DataSourceRequest) -> PreparedDataset:
        timeframe = request.timeframe.strip().lower()
        primary_symbol = request.symbol.strip().upper() if request.symbol else self._settings.default_symbol
        day_scope_map, forecast_days = _build_day_scope_map(
            timeframe=timeframe,
            forecast_days=request.forecast_days,
            legacy_prediction_scope=request.prediction_scope,
        )
        symbols = self._resolve_symbols(request, primary_symbol)

        ohlcv_map: dict[str, pd.DataFrame] = {}
        if request.source == "local":
            csv_path = Path(request.data_path).expanduser() if request.data_path else self._settings.default_data_path
            ohlcv_map["LOCAL_DATA"] = self._data_provider.load_local_csv(csv_path)
            primary_symbol = "LOCAL_DATA"
            symbols = ["LOCAL_DATA"]
        else:
            if primary_symbol not in symbols:
                symbols = [primary_symbol] + [symbol for symbol in symbols if symbol != primary_symbol]
            for symbol in symbols:
                ohlcv_map[symbol] = self._data_provider.fetch_exchange_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=request.exchange_limit,
                )

        data_windows: list[DataWindow] = []
        data_lag_minutes_values: list[float] = []
        now_naive = datetime.now(timezone.utc).replace(tzinfo=None)

        for symbol, ohlcv in ohlcv_map.items():
            if len(ohlcv) < self._settings.min_training_rows:
                raise ValueError(
                    f"Недостаточно данных для обучения по {symbol}: {len(ohlcv)} строк, "
                    f"минимум {self._settings.min_training_rows}."
                )
            start_at = ohlcv.index[0].to_pydatetime()
            end_at = ohlcv.index[-1].to_pydatetime()
            data_windows.append(
                DataWindow(
                    symbol=symbol,
                    rows=len(ohlcv),
                    start_at=start_at,
                    end_at=end_at,
                )
            )
            data_lag_minutes_values.append(max(0.0, (now_naive - end_at).total_seconds() / 60.0))

        timeframe_step_minutes = _timeframe_to_timedelta(timeframe, 1).total_seconds() / 60.0
        freshness_threshold_minutes = max(3.0, timeframe_step_minutes * 2.0)
        max_data_lag_minutes = max(data_lag_minutes_values) if data_lag_minutes_values else 0.0
        data_is_fresh = max_data_lag_minutes <= freshness_threshold_minutes
        exchange_limit_used = request.exchange_limit if request.source == "exchange" else None
        LOGGER.info(
            "Stage: dataset_ready | source=%s | symbols=%s | rows=%s | data_is_fresh=%s | max_lag_min=%.2f",
            request.source,
            list(ohlcv_map.keys()),
            {symbol: len(frame) for symbol, frame in ohlcv_map.items()},
            data_is_fresh,
            max_data_lag_minutes,
        )

        return PreparedDataset(
            source=request.source,
            primary_symbol=primary_symbol,
            symbols=symbols,
            timeframe=timeframe,
            forecast_days=forecast_days,
            day_scope_map=day_scope_map,
            validation_fraction=request.validation_fraction,
            ohlcv_map=ohlcv_map,
            data_windows=data_windows,
            data_is_fresh=data_is_fresh,
            max_data_lag_minutes=max_data_lag_minutes,
            exchange_limit_used=exchange_limit_used,
        )

    def _load_existing_metadata_if_needed(self, request: TrainRequest) -> Optional[dict[str, Any]]:
        if request.mode not in {"finetune", "retrain"}:
            return None
        stored = self._model_store.load(request.model_id)
        return stored.metadata

    def _resolve_candidate_types(
        self,
        request: TrainRequest,
        existing_metadata: Optional[dict[str, Any]],
    ) -> list[str]:
        supported_models = self._supported_models()
        if request.mode in {"finetune", "retrain"}:
            model_type = existing_metadata.get("model_type") if existing_metadata else None
            if model_type not in supported_models:
                raise ValueError("У существующей модели неподдерживаемый тип.")
            return [model_type]
        return [request.model_type]

    def _resolve_base_params(
        self,
        model_type: str,
        request: TrainRequest,
        existing_metadata: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if request.mode not in {"finetune", "retrain"} or not existing_metadata:
            return None
        params = dict(existing_metadata.get("best_params", {}))
        if not params:
            return None
        params.pop("random_state", None)
        params.pop("n_jobs", None)
        return params

    def _fit_and_evaluate(
        self,
        model_type: str,
        scope_matrices: ScopeMatrices,
        tune: bool,
        tune_trials: int,
        full_refit: bool,
        base_params: Optional[dict[str, Any]],
        scoring: str,
        forecast_days: int,
    ) -> dict[str, Any]:
        if model_type not in self._supported_models():
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        chosen_params = dict(base_params or self._default_params(model_type))
        tuned = False
        tuning_config = self._model_params.get("tuning", {})
        scoring = _normalize_scoring_name(tuning_config.get("scoring", scoring))
        huber_delta = float(tuning_config.get("huber_delta", 1.0))
        huber_slope_base = _resolve_huber_slope(
            forecast_days=forecast_days,
            tuning_config=tuning_config,
        )
        huber_slope = _resolve_effective_huber_slope(
            base_huber_slope=huber_slope_base,
            y_values=scope_matrices.y_train,
        )
        if model_type == "xgboost":
            chosen_params = _apply_xgboost_huber_params(
                params=chosen_params,
                huber_slope=huber_slope,
            )

        if tune:
            search_space = self._build_search_space(model_type, base_params)
            search_size = _estimate_search_space_size(search_space)
            n_iter = max(1, min(tune_trials, search_size))
            cv_splits = self._resolve_cv_splits(len(scope_matrices.x_train))
            search_n_jobs = int(tuning_config.get("search_n_jobs", -1))
            search_scoring = _resolve_search_scoring(
                scoring=scoring,
                huber_delta=huber_delta,
            )
            if model_type == "xgboost":
                search_space = _adjust_xgboost_search_space_for_horizon(
                    search_space=search_space,
                    forecast_days=forecast_days,
                    huber_slope=huber_slope,
                )
            LOGGER.info(
                "Stage: hyperparameter_search | model_type=%s | n_iter=%s | cv_splits=%s | scoring=%s | huber_slope_base=%.4f | huber_slope_effective=%.3f",
                model_type,
                n_iter,
                cv_splits,
                _display_scoring_name(scoring),
                huber_slope_base,
                huber_slope,
            )
            if model_type == "xgboost":
                chosen_params = self._manual_time_series_search(
                    model_type=model_type,
                    x_train=scope_matrices.x_train,
                    y_train=scope_matrices.y_train,
                    search_space=search_space,
                    cv_splits=cv_splits,
                    n_iter=n_iter,
                    scoring=scoring,
                    huber_delta=huber_delta,
                    huber_slope=huber_slope,
                )
            else:
                search = RandomizedSearchCV(
                    estimator=self._build_estimator(model_type),
                    param_distributions=search_space,
                    n_iter=n_iter,
                    scoring=search_scoring,
                    cv=TimeSeriesSplit(n_splits=cv_splits),
                    random_state=self._settings.random_state,
                    n_jobs=search_n_jobs,
                    verbose=0,
                )
                search.fit(scope_matrices.x_train, scope_matrices.y_train)
                chosen_params = dict(search.best_params_)
            if model_type == "xgboost":
                chosen_params = _apply_xgboost_huber_params(
                    params=chosen_params,
                    huber_slope=huber_slope,
                )
            tuned = True

        validation_model = self._build_estimator(
            model_type,
            chosen_params,
            huber_slope=huber_slope,
        )
        validation_model.fit(scope_matrices.x_train, scope_matrices.y_train)
        val_predictions = validation_model.predict(scope_matrices.x_validation)
        metrics = _calculate_metrics(scope_matrices.y_validation, val_predictions)
        selection_loss = _score_loss(
            scoring,
            scope_matrices.y_validation,
            val_predictions,
            huber_delta=huber_delta,
        )
        LOGGER.info(
            "Stage: evaluate_done | model_type=%s | tuned=%s | scoring=%s | selection_loss=%.6f | huber_slope_effective=%.3f",
            model_type,
            tuned,
            _display_scoring_name(scoring),
            selection_loss,
            huber_slope,
        )

        final_model = validation_model
        if full_refit:
            refit = clone(
                self._build_estimator(
                    model_type,
                    chosen_params,
                    huber_slope=huber_slope,
                )
            )
            full_x = np.vstack([scope_matrices.x_train, scope_matrices.x_validation])
            full_y = np.concatenate([scope_matrices.y_train, scope_matrices.y_validation])
            refit.fit(full_x, full_y)
            final_model = refit

        return {
            "model_type": model_type,
            "best_params": chosen_params,
            "metrics": metrics,
            "model": final_model,
            "tuned": tuned,
            "selection_loss": selection_loss,
        }

    def _build_estimator(
        self,
        model_type: str,
        params: Optional[dict[str, Any]] = None,
        huber_slope: Optional[float] = None,
    ) -> Any:
        if model_type not in ESTIMATOR_REGISTRY:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        if model_type == "xgboost" and XGBRegressor is None:
            raise ValueError("XGBoost не установлен. Установите зависимости из requirements.txt")

        model_entry = self._get_model_entry(model_type)
        defaults = dict(model_entry.get("default_params", {}))
        defaults.update(dict(params or {}))

        if "random_state" in defaults and defaults["random_state"] is None:
            defaults["random_state"] = self._settings.random_state
        if model_type == "xgboost":
            defaults = _apply_xgboost_huber_params(
                params=defaults,
                huber_slope=(float(huber_slope) if huber_slope is not None else None),
            )

        estimator_cls = ESTIMATOR_REGISTRY[model_type]
        if estimator_cls is None:
            raise ValueError(f"Estimator для модели '{model_type}' недоступен в текущем окружении.")
        return estimator_cls(**defaults)

    def _default_params(self, model_type: str) -> dict[str, Any]:
        model_entry = self._get_model_entry(model_type)
        params = dict(model_entry.get("default_params", {}))
        params.pop("random_state", None)
        params.pop("n_jobs", None)
        return params

    def _build_search_space(self, model_type: str, base_params: Optional[dict[str, Any]]) -> dict[str, list[Any]]:
        model_entry = self._get_model_entry(model_type)
        search_space = model_entry.get("search_space", {})
        if not isinstance(search_space, dict) or not search_space:
            raise ValueError(f"Для модели '{model_type}' не задан search_space в model_params.yml")

        space = {
            key: value if isinstance(value, list) else [value]
            for key, value in deepcopy(search_space).items()
        }

        if base_params:
            for key, value in base_params.items():
                if key not in space:
                    continue
                if value not in space[key]:
                    space[key].append(value)
                for nearby in _generate_nearby_values(value):
                    if nearby not in space[key]:
                        space[key].append(nearby)
        return space

    def _resolve_cv_splits(self, train_rows: int) -> int:
        tuning_config = self._model_params.get("tuning", {})
        cv_rules = tuning_config.get("cv_rules", {})

        small_threshold = int(cv_rules.get("small_train_threshold", 80))
        medium_threshold = int(cv_rules.get("medium_train_threshold", 160))
        small_splits = int(cv_rules.get("small_train_splits", 2))
        medium_splits = int(cv_rules.get("medium_train_splits", 3))
        default_splits = int(cv_rules.get("default_splits", self._settings.tuning_cv_splits))

        if train_rows < small_threshold:
            return max(2, small_splits)
        if train_rows < medium_threshold:
            return max(2, medium_splits)
        return max(2, default_splits)

    def _manual_time_series_search(
        self,
        model_type: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        search_space: dict[str, list[Any]],
        cv_splits: int,
        n_iter: int,
        scoring: str,
        huber_delta: float,
        huber_slope: float,
    ) -> dict[str, Any]:
        splitter = TimeSeriesSplit(n_splits=cv_splits)
        sampler = ParameterSampler(
            param_distributions=search_space,
            n_iter=n_iter,
            random_state=self._settings.random_state,
        )
        LOGGER.info(
            "Stage: manual_time_series_search | model_type=%s | n_iter=%s | cv_splits=%s | scoring=%s | huber_slope=%.3f",
            model_type,
            n_iter,
            cv_splits,
            _display_scoring_name(scoring),
            huber_slope,
        )

        best_params: Optional[dict[str, Any]] = None
        best_score = float("inf")
        for params in sampler:
            fold_scores: list[float] = []
            for train_idx, validation_idx in splitter.split(x_train):
                estimator = self._build_estimator(
                    model_type,
                    params,
                    huber_slope=huber_slope,
                )
                estimator.fit(x_train[train_idx], y_train[train_idx])
                predictions = estimator.predict(x_train[validation_idx])
                fold_scores.append(
                    _score_loss(
                        scoring,
                        y_train[validation_idx],
                        predictions,
                        huber_delta=huber_delta,
                    )
                )

            if not fold_scores:
                continue
            score = float(np.mean(fold_scores))
            if score < best_score:
                best_score = score
                best_params = dict(params)

        if best_params is None:
            raise ValueError(f"Не удалось подобрать гиперпараметры для модели '{model_type}'.")
        LOGGER.info(
            "Stage: manual_search_done | model_type=%s | best_score=%.6f",
            model_type,
            best_score,
        )
        return best_params

    def _supported_models(self) -> list[str]:
        configured_models = self._model_params.get("models", {})
        if not isinstance(configured_models, dict):
            return []
        return [name for name in configured_models.keys() if name in ESTIMATOR_REGISTRY]

    def _get_model_entry(self, model_type: str) -> dict[str, Any]:
        models = self._model_params.get("models", {})
        if not isinstance(models, dict):
            raise ValueError("Некорректная структура models в model_params.yml")

        model_entry = models.get(model_type)
        if not isinstance(model_entry, dict):
            raise ValueError(f"Модель '{model_type}' не найдена в model_params.yml")
        return model_entry

    def _validate_stored_model(self, stored_model: StoredModel, prepared: PreparedDataset) -> None:
        metadata = stored_model.metadata
        if metadata.get("source") != prepared.source:
            raise ValueError("Сохраненная модель обучалась на другом источнике данных.")
        if metadata.get("timeframe") != prepared.timeframe:
            raise ValueError("Сохраненная модель обучалась на другом timeframe.")
        self._validate_metadata_compatibility(metadata)

        max_days = int(metadata.get("forecast_days", 0))
        if max_days < prepared.forecast_days:
            raise ValueError(
                "Сохраненная модель обучена на меньший горизонт. "
                "Включите auto_train_if_missing или выполните train на больший forecast_days."
            )

        if prepared.source == "exchange":
            supported_symbols = metadata.get("symbols", [])
            if not isinstance(supported_symbols, list) or not supported_symbols:
                legacy_symbol = metadata.get("symbol")
                supported_symbols = [legacy_symbol] if legacy_symbol else []
            supported_symbols = [str(symbol).upper() for symbol in supported_symbols]
            if prepared.primary_symbol not in supported_symbols:
                raise ValueError(
                    "Сохраненная модель не поддерживает эту криптопару. "
                    "Используйте отдельный model_id или включите auto_train_if_missing."
                )

        model_bundle, _ = self._normalize_stored_bundle(stored_model.model, metadata)
        models_by_scope = model_bundle.get("models_by_scope", {})
        for scope in prepared.day_scope_map.values():
            if str(scope) not in models_by_scope:
                raise ValueError(
                    f"В сохраненной модели нет horizon scope={scope}. "
                    "Переобучите модель на нужный forecast_days."
                )

    def _validate_metadata_compatibility(self, metadata: dict[str, Any]) -> None:
        model_type = str(metadata.get("model_type", "")).strip().lower()
        tuning_config = self._model_params.get("tuning", {})
        expected_scoring = _display_scoring_name(
            _normalize_scoring_name(tuning_config.get("scoring", "mae"))
        )
        stored_scoring = _display_scoring_name(
            _normalize_scoring_name(metadata.get("tuning_scoring", expected_scoring))
        )
        if stored_scoring != expected_scoring:
            raise ValueError(
                "Сохраненная модель обучена с устаревшим tuning scoring. "
                "Будет запущено автообучение."
            )

        if model_type != "xgboost":
            return

        best_params_raw = metadata.get("best_params", {})
        best_params = best_params_raw if isinstance(best_params_raw, dict) else {}
        objective = str(best_params.get("objective", "")).strip().lower()
        if not objective:
            legacy_loss = str(metadata.get("loss_function", "")).strip().lower()
            if legacy_loss.startswith("xgboost:"):
                objective = legacy_loss.split(":", 1)[1]
        if objective != "reg:pseudohubererror":
            raise ValueError(
                "Сохраненная XGBoost-модель обучена с устаревшей loss function. "
                "Будет запущено автообучение."
            )
        stored_huber_mode = str(metadata.get("xgboost_huber_mode", "")).strip()
        if stored_huber_mode != XGBOOST_HUBER_MODE_VERSION:
            raise ValueError(
                "Сохраненная XGBoost-модель обучена со старой стратегией huber_slope. "
                "Будет запущено автообучение."
            )

    def _normalize_stored_bundle(
        self,
        stored_model: Any,
        metadata: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(stored_model, dict) and "models_by_scope" in stored_model:
            return stored_model, metadata

        legacy_scope = int(metadata.get("prediction_scope", -1))
        if legacy_scope < 0:
            raise ValueError("Невозможно определить scope сохраненной legacy-модели.")

        bundle = {
            "bundle_version": 1,
            "models_by_scope": {str(legacy_scope): stored_model},
        }
        return bundle, metadata

    def _get_model_for_scope(self, model_bundle: dict[str, Any], prediction_scope: int) -> Any:
        models_by_scope = model_bundle.get("models_by_scope", {})
        model = models_by_scope.get(str(prediction_scope))
        if model is None:
            raise ValueError(
                f"В сохраненной модели отсутствует горизонт со scope={prediction_scope}. "
                "Переобучите модель на больший forecast_days."
            )
        return model

    def _build_scope_matrices(self, prepared: PreparedDataset, prediction_scope: int) -> ScopeMatrices:
        x_train_parts: list[np.ndarray] = []
        y_train_parts: list[np.ndarray] = []
        x_validation_parts: list[np.ndarray] = []
        y_validation_parts: list[np.ndarray] = []
        feature_columns: Optional[list[str]] = None

        for symbol in prepared.symbols:
            dataset, _, _ = build_feature_frame(
                ohlcv=prepared.ohlcv_map[symbol],
                prediction_scope=prediction_scope,
            )
            dataset = self._add_symbol_features(dataset, symbol, prepared.symbols)

            current_feature_columns = [column for column in dataset.columns if column != "target"]
            if feature_columns is None:
                feature_columns = current_feature_columns
            else:
                missing = [column for column in feature_columns if column not in current_feature_columns]
                extra = [column for column in current_feature_columns if column not in feature_columns]
                if missing or extra:
                    raise ValueError(
                        "Несовместимые признаки между символами для универсальной модели. "
                        f"missing={missing}, extra={extra}"
                    )
                dataset = dataset.loc[:, feature_columns + ["target"]]

            split_index = int(len(dataset) * (1.0 - prepared.validation_fraction))
            if split_index < self._settings.min_train_rows_after_split:
                raise ValueError(
                    f"Слишком маленький train-набор по {symbol} для horizon scope={prediction_scope}. "
                    "Уменьшите validation_fraction или увеличьте exchange_limit."
                )
            if (len(dataset) - split_index) < self._settings.min_validation_rows_after_split:
                raise ValueError(
                    f"Слишком маленький validation-набор по {symbol} для horizon scope={prediction_scope}. "
                    "Уменьшите validation_fraction или увеличьте exchange_limit."
                )

            train_part = dataset.iloc[:split_index]
            validation_part = dataset.iloc[split_index:]

            x_train_parts.append(train_part.loc[:, feature_columns].to_numpy(dtype=float))
            y_train_parts.append(train_part["target"].to_numpy(dtype=float))
            x_validation_parts.append(validation_part.loc[:, feature_columns].to_numpy(dtype=float))
            y_validation_parts.append(validation_part["target"].to_numpy(dtype=float))

        if not feature_columns or not x_train_parts or not x_validation_parts:
            raise ValueError("Не удалось собрать train/validation матрицы для обучения.")

        return ScopeMatrices(
            x_train=np.vstack(x_train_parts),
            y_train=np.concatenate(y_train_parts),
            x_validation=np.vstack(x_validation_parts),
            y_validation=np.concatenate(y_validation_parts),
            feature_columns=feature_columns,
        )

    def _build_latest_feature_vector(
        self,
        prepared: PreparedDataset,
        symbol: str,
        prediction_scope: int,
        supported_symbols: list[str],
        feature_columns: list[str],
    ) -> np.ndarray:
        _, latest_features, _ = build_feature_frame(
            ohlcv=prepared.ohlcv_map[symbol],
            prediction_scope=prediction_scope,
        )
        latest_features = self._add_symbol_features(latest_features, symbol, supported_symbols)

        missing_columns = [column for column in feature_columns if column not in latest_features.columns]
        if missing_columns:
            raise ValueError(
                "Текущие данные несовместимы с моделью. "
                f"Отсутствуют признаки: {missing_columns}"
            )
        return latest_features.loc[:, feature_columns].to_numpy(dtype=float)

    def _resolve_symbols(self, request: DataSourceRequest, primary_symbol: str) -> list[str]:
        if request.source != "exchange":
            return ["LOCAL_DATA"]

        if request.symbols:
            normalized = [symbol.strip().upper() for symbol in request.symbols if str(symbol).strip()]
            unique: list[str] = []
            for symbol in normalized:
                if symbol not in unique:
                    unique.append(symbol)
            if not unique:
                return [primary_symbol]
            return unique

        return [primary_symbol]

    def _add_symbol_features(self, frame: pd.DataFrame, symbol: str, all_symbols: list[str]) -> pd.DataFrame:
        result = frame.copy()
        normalized_symbol = symbol.strip().upper()
        for item in all_symbols:
            feature_name = f"symbol__{_normalize_symbol_name(item)}"
            result[feature_name] = 1.0 if normalized_symbol == item.strip().upper() else 0.0
        return result

    def _enrich_request_with_existing_symbols(
        self,
        request: TrainRequest,
        existing_metadata: Optional[dict[str, Any]],
    ) -> TrainRequest:
        if request.source != "exchange":
            return request
        if request.symbols:
            return request
        if not existing_metadata:
            return request

        metadata_symbols = existing_metadata.get("symbols", [])
        if not isinstance(metadata_symbols, list) or not metadata_symbols:
            return request

        payload = request.model_dump()
        payload["symbols"] = metadata_symbols
        payload["symbol"] = metadata_symbols[0]
        return TrainRequest(**payload)


def _timeframe_to_timedelta(timeframe: str, steps: int) -> timedelta:
    return timedelta(seconds=_timeframe_to_seconds(timeframe) * steps)


def _timeframe_to_seconds(timeframe: str) -> int:
    match = TIMEFRAME_PATTERN.match(timeframe)
    if not match:
        raise ValueError("Некорректный timeframe. Используйте форматы вида 1m, 1h, 1d, 1w.")
    value = int(match.group(1))
    unit = match.group(2).lower()
    return value * UNIT_TO_SECONDS[unit]


def _resolve_prediction_scope_and_days(
    timeframe: str,
    forecast_days: int,
    legacy_prediction_scope: Optional[int],
) -> tuple[int, int]:
    timeframe_seconds = _timeframe_to_seconds(timeframe)

    if legacy_prediction_scope is not None:
        scope = int(legacy_prediction_scope)
        steps = scope + 1
        effective_days = int(math.ceil((steps * timeframe_seconds) / 86400))
        if effective_days > MAX_FORECAST_DAYS:
            raise ValueError(
                f"Горизонт прогноза превышает {MAX_FORECAST_DAYS} дней. "
                "Уменьшите prediction_scope или используйте более крупный timeframe."
            )
        return scope, effective_days

    requested_days = int(forecast_days)
    steps = max(1, int(math.ceil((requested_days * 86400) / timeframe_seconds)))
    effective_days = int(math.ceil((steps * timeframe_seconds) / 86400))
    if effective_days > MAX_FORECAST_DAYS:
        raise ValueError(
            f"С выбранным timeframe минимальный шаг прогноза превышает лимит {MAX_FORECAST_DAYS} дней. "
            "Выберите более мелкий timeframe."
        )
    return steps - 1, effective_days


def _build_day_scope_map(
    timeframe: str,
    forecast_days: int,
    legacy_prediction_scope: Optional[int],
) -> tuple[dict[int, int], int]:
    if legacy_prediction_scope is not None:
        _, effective_days = _resolve_prediction_scope_and_days(
            timeframe=timeframe,
            forecast_days=forecast_days,
            legacy_prediction_scope=legacy_prediction_scope,
        )
        day_count = effective_days
    else:
        day_count = int(forecast_days)

    day_scope_map: dict[int, int] = {}
    for day_ahead in range(1, day_count + 1):
        day_scope_map[day_ahead] = _scope_from_days(timeframe, day_ahead)
    return day_scope_map, day_count


def _scope_from_days(timeframe: str, day_ahead: int) -> int:
    timeframe_seconds = _timeframe_to_seconds(timeframe)
    steps = max(1, int(math.ceil((day_ahead * 86400) / timeframe_seconds)))
    return steps - 1


def _normalize_symbol_name(symbol: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", symbol.strip().upper())
    return normalized.strip("_")

def _normalize_scoring_name(scoring: Any) -> str:
    normalized = str(scoring or "mae").strip().lower()
    aliases = {
        "mae": "neg_mean_absolute_error",
        "mape": "neg_mean_absolute_percentage_error",
        "rmse": "neg_root_mean_squared_error",
        "mse": "neg_mean_squared_error",
        "huber_loss": "huber",
        "neg_huber": "huber",
    }
    return aliases.get(normalized, normalized)


def _display_scoring_name(scoring: str) -> str:
    normalized = _normalize_scoring_name(scoring)
    names = {
        "neg_mean_absolute_error": "mae",
        "neg_mean_absolute_percentage_error": "mape",
        "neg_root_mean_squared_error": "rmse",
        "neg_mean_squared_error": "mse",
        "huber": "huber",
    }
    return names.get(normalized, normalized)


def _resolve_huber_slope(forecast_days: int, tuning_config: dict[str, Any]) -> float:
    short_value = float(tuning_config.get("huber_slope_short", 1.0))
    long_value = float(tuning_config.get("huber_slope_long", 0.3))
    short_days = 3
    long_days = MAX_FORECAST_DAYS
    clamped_days = max(1, min(int(forecast_days), long_days))
    if clamped_days <= short_days:
        return round(short_value, 4)
    if clamped_days >= long_days:
        return round(long_value, 4)
    ratio = (clamped_days - short_days) / float(long_days - short_days)
    value = short_value + (long_value - short_value) * ratio
    return round(value, 4)

def _resolve_effective_huber_slope(base_huber_slope: float, y_values: np.ndarray) -> float:
    base_value = max(float(base_huber_slope), 1e-4)
    values = np.asarray(y_values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return round(base_value, 4)
    target_scale = float(np.median(np.abs(finite_values)))
    if not np.isfinite(target_scale) or target_scale <= 1.0:
        target_scale = 1.0
    effective = base_value * target_scale
    effective = min(max(effective, 1e-4), 1_000_000.0)
    return round(effective, 4)


def _apply_xgboost_huber_params(params: dict[str, Any], huber_slope: Optional[float]) -> dict[str, Any]:
    updated = dict(params)
    updated["objective"] = "reg:pseudohubererror"
    if huber_slope is not None:
        updated["huber_slope"] = float(huber_slope)
    elif "huber_slope" not in updated:
        updated["huber_slope"] = 1.0
    return updated


def _adjust_xgboost_search_space_for_horizon(
    search_space: dict[str, list[Any]],
    forecast_days: int,
    huber_slope: float,
) -> dict[str, list[Any]]:
    adjusted = {key: list(values) for key, values in search_space.items()}
    adjusted["objective"] = ["reg:pseudohubererror"]
    safe_slope = max(float(huber_slope), 1e-4)

    if forecast_days <= 3:
        slope_candidates = [safe_slope * 0.7, safe_slope, safe_slope * 1.3]
    elif forecast_days >= MAX_FORECAST_DAYS:
        slope_candidates = [safe_slope * 0.4, safe_slope * 0.6, safe_slope * 0.8, safe_slope]
    else:
        slope_candidates = [safe_slope * 0.5, safe_slope * 0.75, safe_slope, safe_slope * 1.2]

    for value in adjusted.get("huber_slope", []):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > 0:
            slope_candidates.append(numeric)

    normalized = sorted(
        {
            round(min(max(float(value), 1e-4), 1_000_000.0), 4)
            for value in slope_candidates
        }
    )
    adjusted["huber_slope"] = normalized
    return adjusted


def _score_loss(
    scoring: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    huber_delta: float = 1.0,
) -> float:
    normalized = _normalize_scoring_name(scoring)
    if normalized == "huber":
        return float(_huber_loss(y_true, y_pred, delta=huber_delta))
    if normalized == "neg_mean_absolute_percentage_error":
        return float(mean_absolute_percentage_error(y_true, y_pred))
    if normalized == "neg_root_mean_squared_error":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if normalized == "neg_mean_squared_error":
        return float(mean_squared_error(y_true, y_pred))
    return float(mean_absolute_error(y_true, y_pred))


def _resolve_search_scoring(scoring: str, huber_delta: float) -> Any:
    normalized = _normalize_scoring_name(scoring)
    if normalized == "huber":
        return make_scorer(
            partial(_huber_loss, delta=huber_delta),
            greater_is_better=False,
        )
    return normalized


def _huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    safe_delta = max(float(delta), 1e-8)
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, safe_delta)
    linear = abs_error - quadratic
    return float(np.mean(0.5 * quadratic**2 + safe_delta * linear))


def _resolve_loss_function(model_type: str, params: dict[str, Any]) -> str:
    if model_type == "xgboost":
        objective = params.get("objective", "reg:pseudohubererror")
        return f"xgboost:{objective}"
    return "huber_evaluation_metric"


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> PredictionMetrics:
    return PredictionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        mape=float(mean_absolute_percentage_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mse=float(mean_squared_error(y_true, y_pred)),
    )


def _estimate_search_space_size(space: dict[str, list[Any]]) -> int:
    result = 1
    for values in space.values():
        result *= max(1, len(values))
    return result


def _generate_nearby_values(value: Any) -> list[Any]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [max(1, int(round(value * 0.7))), max(1, int(round(value * 1.3)))]
    if isinstance(value, float):
        return [round(max(0.0001, value * 0.7), 6), round(value * 1.3, 6)]
    return []

def _feature_profile(feature_name: str) -> tuple[str, str]:
    name = feature_name.lower()
    if name.startswith("symbol__"):
        return "symbol", "binary"
    if any(key in name for key in ("day_of_", "month", "dow_")):
        return "calendar", "number"
    if any(key in name for key in ("rsi", "stoch", "williams", "cci", "mfi", "adx", "plus_di", "minus_di")):
        return "oscillator", "oscillator"
    if any(key in name for key in ("volume", "obv", "cmf")):
        return "volume", "volume"
    if any(
        key in name
        for key in (
            "return",
            "pct",
            "ratio",
            "position",
            "spread",
            "zscore",
            "momentum",
            "trix",
            "ppo",
            "log_",
        )
    ):
        return "relative", "percent"
    if any(
        key in name
        for key in (
            "open",
            "high",
            "low",
            "close",
            "ema",
            "vwap",
            "bb_",
            "atr",
            "macd",
            "candle_",
            "hl_",
            "oc_",
        )
    ):
        return "price", "price"
    return "other", "number"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
