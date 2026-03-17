import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from src.backend.presentation.http.dependencies import UseCaseContainer, get_container
from src.backend.schemas import (
    FeaturePreviewRequest,
    FeaturePreviewResponse,
    MarketCandlesRequest,
    MarketCandlesResponse,
    ModelRegistryResponse,
    PredictRequest,
    PredictionResponse,
    TrainJobEnqueueResponse,
    TrainJobStatusResponse,
    TrainRequest,
    TrainResponse,
)

router = APIRouter()

@router.get("/features/preview", response_model=FeaturePreviewResponse)
def feature_preview(
    source: str = "exchange",
    symbol: str = "BTC/USD",
    timeframe: str = "1d",
    forecast_days: int = 14,
    exchange_limit: int = 600,
    history_size: int = 72,
    data_path: Optional[str] = None,
    container: UseCaseContainer = Depends(get_container),
) -> FeaturePreviewResponse:
    try:
        payload = FeaturePreviewRequest(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            forecast_days=forecast_days,
            exchange_limit=exchange_limit,
            history_size=history_size,
            data_path=data_path,
        )
        return container.get_feature_preview.execute(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервиса: {exc}") from exc


@router.get("/market/candles", response_model=MarketCandlesResponse)
def market_candles(
    source: str = "exchange",
    symbol: str = "BTC/USD",
    timeframe: str = "1h",
    limit: int = 300,
    data_path: Optional[str] = None,
    container: UseCaseContainer = Depends(get_container),
) -> MarketCandlesResponse:
    try:
        payload = MarketCandlesRequest(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            data_path=data_path,
        )
        return container.get_market_candles.execute(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервиса: {exc}") from exc


@router.post("/train", response_model=TrainResponse)
def train(
    payload: TrainRequest,
    container: UseCaseContainer = Depends(get_container),
) -> TrainResponse:
    try:
        return container.train_model.execute(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервиса: {exc}") from exc


@router.post("/train/async", response_model=TrainJobEnqueueResponse)
def train_async(payload: TrainRequest) -> TrainJobEnqueueResponse:
    try:
        from src.backend.infrastructure.queue.training_jobs import enqueue_training_job
        job_id = f"train-{payload.model_id}-{uuid.uuid4().hex[:12]}"
        enqueue_training_job(job_id=job_id, payload=payload)
        return TrainJobEnqueueResponse(
            job_id=job_id,
            status="queued",
            queued_at=datetime.now(timezone.utc),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервиса: {exc}") from exc


@router.get("/train/jobs/{job_id}", response_model=TrainJobStatusResponse)
def train_job_status(job_id: str) -> TrainJobStatusResponse:
    try:
        from src.backend.infrastructure.queue.training_jobs import get_training_job_status
        status_payload = get_training_job_status(job_id=job_id)
        if status_payload is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' не найден.")
        return status_payload
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервиса: {exc}") from exc


@router.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictRequest,
    container: UseCaseContainer = Depends(get_container),
) -> PredictionResponse:
    try:
        return container.predict.execute(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервиса: {exc}") from exc


@router.get("/models", response_model=ModelRegistryResponse)
def models(container: UseCaseContainer = Depends(get_container)) -> ModelRegistryResponse:
    return container.list_models.execute()

