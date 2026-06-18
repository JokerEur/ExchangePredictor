import logging
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    from statsmodels.tsa.arima.model import ARIMA as _StatsmodelsARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    _StatsmodelsARIMA = None
    STATSMODELS_AVAILABLE = False

# Порядки ARIMA, которые пробуем при обучении (от предпочтительного к запасному)
_CANDIDATE_ORDERS: list[Tuple[int, int, int]] = [
    (5, 1, 0),
    (2, 1, 2),
    (1, 1, 1),
    (2, 1, 0),
    (1, 1, 0),
]


class ARIMAWrapper:
    """
    Обёртка над statsmodels ARIMA для использования в ансамблевом прогнозировании.

    В отличие от XGBoost/RandomForest, ARIMA работает непосредственно с рядом
    цен (univariate time series). При предсказании модель всегда перефитируется
    на актуальных данных, поэтому хранит только подобранный порядок (p, d, q),
    а не объект результатов statsmodels.
    """

    def __init__(self, order: Optional[Tuple[int, int, int]] = None):
        self._preferred_order: Tuple[int, int, int] = order or (5, 1, 0)
        self._validated_order: Optional[Tuple[int, int, int]] = None

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def fit(self, price_series: np.ndarray) -> "ARIMAWrapper":
        """
        Валидирует, какой порядок ARIMA работает на данном ряде.

        Сохраняет лучший (первый успешный) порядок для последующего
        использования при предсказании.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError(
                "statsmodels не установлен. Добавьте 'statsmodels' в requirements.txt "
                "и выполните pip install -r requirements.txt."
            )

        series = np.asarray(price_series, dtype=float)
        if len(series) < 20:
            raise ValueError(
                f"Для ARIMA нужно минимум 20 наблюдений, получено {len(series)}."
            )

        candidates = [self._preferred_order] + [
            o for o in _CANDIDATE_ORDERS if o != self._preferred_order
        ]

        for order in candidates:
            try:
                model = _StatsmodelsARIMA(series, order=order)
                result = model.fit()
                self._validated_order = order
                LOGGER.info(
                    "ARIMA fitted | order=%s | aic=%.2f | bic=%.2f",
                    order,
                    result.aic,
                    result.bic,
                )
                return self
            except Exception as exc:
                LOGGER.warning("ARIMA order=%s не подошёл: %s", order, exc)

        raise ValueError(
            "Не удалось подобрать порядок ARIMA. Проверьте качество исходных данных."
        )

    # ------------------------------------------------------------------
    # Предсказание
    # ------------------------------------------------------------------

    def predict_scope(self, price_series: np.ndarray, scope: int) -> float:
        """
        Перефитирует ARIMA на актуальном ряде цен и возвращает прогноз
        на `scope + 1` шагов вперёд.

        Args:
            price_series: актуальный ряд цен закрытия (полная история).
            scope: индекс горизонта прогноза (0 = следующий день при 1d-timeframe).

        Returns:
            Прогнозная цена.
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels не установлен.")

        order = self._validated_order or self._preferred_order
        series = np.asarray(price_series, dtype=float)

        try:
            model = _StatsmodelsARIMA(series, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=scope + 1)
            return float(forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1])
        except Exception as exc:
            LOGGER.warning(
                "ARIMA predict_scope failed | scope=%s | error=%s. "
                "Возвращаем последнюю известную цену.",
                scope,
                exc,
            )
            return float(series[-1])

    def get_order(self) -> Tuple[int, int, int]:
        return self._validated_order or self._preferred_order
