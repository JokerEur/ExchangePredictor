from pathlib import Path

import ccxt
import pandas as pd
import re

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
TIMEFRAME_PATTERN = re.compile(r"^(\d+)\s*([mhdw])$", re.IGNORECASE)
TIMEFRAME_TO_MS = {
    "m": 60 * 1000,
    "h": 60 * 60 * 1000,
    "d": 24 * 60 * 60 * 1000,
    "w": 7 * 24 * 60 * 60 * 1000,
}


class DataProvider:
    def load_local_csv(self, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise ValueError(f"CSV-файл не найден: {csv_path}")

        frame = pd.read_csv(csv_path, sep=None, engine="python")
        return self._normalize_ohlcv(frame)

    def fetch_exchange_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        exchange = ccxt.bitfinex()
        timeframe_ms = _timeframe_to_milliseconds(timeframe)
        safety_margin = max(32, int(limit * 0.25))
        target_rows = limit + safety_margin
        now_ms = exchange.milliseconds()
        since_ms = max(0, now_ms - target_rows * timeframe_ms)
        try:
            candles = self._fetch_recent_candles_paginated(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
                since_ms=since_ms,
                timeframe_ms=timeframe_ms,
                target_rows=target_rows,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(f"Ошибка загрузки данных с биржи: {exc}") from exc

        if not candles:
            raise ValueError("Биржа вернула пустой набор свечей.")

        frame = pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume"])
        frame["datetime"] = pd.to_datetime(frame["datetime"], unit="ms", utc=True).dt.tz_localize(None)
        normalized = self._normalize_ohlcv(frame)

        # Берем последние свечи после нормализации, чтобы гарантировать свежий хвост.
        normalized = normalized.tail(limit)
        if normalized.empty:
            raise ValueError("После фильтрации последних свечей не осталось данных.")
        return normalized

    def _fetch_recent_candles_paginated(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        timeframe: str,
        limit: int,
        since_ms: int,
        timeframe_ms: int,
        target_rows: int,
    ) -> list[list[float]]:
        all_candles: list[list[float]] = []
        cursor = since_ms
        max_iterations = 12
        per_call_limit = min(1000, max(120, target_rows))

        for _ in range(max_iterations):
            batch = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=cursor,
                limit=per_call_limit,
            )
            if not batch:
                break

            all_candles.extend(batch)
            last_ts = int(batch[-1][0])
            next_cursor = last_ts + timeframe_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(all_candles) >= target_rows:
                break

        if not all_candles:
            # Fallback на дефолтный вызов, если paginated-режим не дал данных.
            return exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        return all_candles

    def _normalize_ohlcv(self, frame: pd.DataFrame) -> pd.DataFrame:
        renamed = {}
        for column in frame.columns:
            normalized = str(column).strip().lower()
            if normalized in {"datetime", "date", "timestamp", "time"}:
                renamed[column] = "datetime"
            elif normalized in REQUIRED_COLUMNS:
                renamed[column] = normalized

        frame = frame.rename(columns=renamed)
        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
            frame = frame.dropna(subset=["datetime"]).set_index("datetime")
        elif not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Данные должны содержать колонку даты/времени.")

        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"В данных отсутствуют обязательные колонки: {missing}")

        ohlcv = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
        ohlcv = ohlcv.apply(pd.to_numeric, errors="coerce").dropna()
        ohlcv = ohlcv.sort_index()
        ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")]

        if ohlcv.empty:
            raise ValueError("После нормализации не осталось валидных OHLCV-строк.")
        return ohlcv


def _timeframe_to_milliseconds(timeframe: str) -> int:
    match = TIMEFRAME_PATTERN.match(timeframe.strip().lower())
    if not match:
        raise ValueError("Некорректный timeframe. Используйте форматы вида 1m, 1h, 1d, 1w.")

    value = int(match.group(1))
    unit = match.group(2).lower()
    return value * TIMEFRAME_TO_MS[unit]
