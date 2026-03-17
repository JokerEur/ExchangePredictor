from typing import List, Tuple

import numpy as np
import pandas as pd


def build_feature_frame(ohlcv: pd.DataFrame, prediction_scope: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if prediction_scope < 0:
        raise ValueError("prediction_scope не может быть отрицательным.")

    data = ohlcv.copy()
    close = data["close"]
    high = data["high"]
    low = data["low"]
    open_price = data["open"]
    volume = data["volume"]

    safe_close = close.replace(0, np.nan)
    safe_open = open_price.replace(0, np.nan)
    feature_map: dict[str, pd.Series] = {}

    feature_map["return_1"] = close.pct_change()
    feature_map["return_3"] = close.pct_change(3)
    feature_map["return_7"] = close.pct_change(7)
    feature_map["return_14"] = close.pct_change(14)
    feature_map["log_return_1"] = np.log(safe_close).diff()

    feature_map["hl_spread"] = (high - low) / safe_close
    feature_map["oc_spread"] = (close - open_price) / safe_open

    candle_range = (high - low).replace(0, np.nan)
    upper_body = np.maximum(open_price, close)
    lower_body = np.minimum(open_price, close)
    feature_map["candle_body"] = close - open_price
    feature_map["candle_range"] = high - low
    feature_map["body_to_range"] = feature_map["candle_body"] / candle_range
    feature_map["upper_wick"] = high - upper_body
    feature_map["lower_wick"] = lower_body - low
    feature_map["upper_wick_ratio"] = feature_map["upper_wick"] / candle_range
    feature_map["lower_wick_ratio"] = feature_map["lower_wick"] / candle_range

    for lag in (1, 2, 3, 5, 7, 14, 21, 30):
        feature_map[f"close_lag_{lag}"] = close.shift(lag)
        feature_map[f"volume_lag_{lag}"] = volume.shift(lag)
        feature_map[f"return_lag_{lag}"] = feature_map["return_1"].shift(lag)

    for window in (3, 7, 14, 30, 60):
        close_roll = close.rolling(window)
        close_mean = close_roll.mean()
        close_std = close_roll.std()
        close_min = close_roll.min()
        close_max = close_roll.max()

        feature_map[f"close_mean_{window}"] = close_mean
        feature_map[f"close_std_{window}"] = close_std
        feature_map[f"close_min_{window}"] = close_min
        feature_map[f"close_max_{window}"] = close_max
        feature_map[f"close_zscore_{window}"] = (close - close_mean) / close_std.replace(0, np.nan)
        feature_map[f"close_position_{window}"] = (
            close - close_min
        ) / (close_max - close_min).replace(0, np.nan)
        feature_map[f"price_to_mean_{window}"] = close / close_mean.replace(0, np.nan)

        volume_roll = volume.rolling(window)
        volume_mean = volume_roll.mean()
        volume_std = volume_roll.std()
        feature_map[f"volume_mean_{window}"] = volume_mean
        feature_map[f"volume_std_{window}"] = volume_std
        feature_map[f"volume_zscore_{window}"] = (volume - volume_mean) / volume_std.replace(0, np.nan)
        feature_map[f"volume_to_mean_{window}"] = volume / volume_mean.replace(0, np.nan)

        feature_map[f"volatility_{window}"] = feature_map["return_1"].rolling(window).std()
        feature_map[f"momentum_{window}"] = close.pct_change(window)

    for span in (5, 10, 20, 50):
        ema = close.ewm(span=span, adjust=False).mean()
        feature_map[f"close_ema_{span}"] = ema
        feature_map[f"price_to_ema_{span}"] = close / ema.replace(0, np.nan)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    feature_map["macd"] = ema_12 - ema_26
    feature_map["macd_signal"] = feature_map["macd"].ewm(span=9, adjust=False).mean()
    feature_map["macd_hist"] = feature_map["macd"] - feature_map["macd_signal"]
    feature_map["ppo"] = feature_map["macd"] / ema_26.replace(0, np.nan)
    feature_map["trix_15"] = _trix(close, 15)

    feature_map["rsi_14"] = _rsi(close, 14)
    feature_map["williams_r_14"] = _williams_r(high, low, close, 14)
    feature_map["cci_20"] = _cci(high, low, close, 20)

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    feature_map["stoch_k_14"] = 100.0 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
    feature_map["stoch_d_14"] = feature_map["stoch_k_14"].rolling(3).mean()

    true_range = _true_range(high=high, low=low, close=close)
    feature_map["atr_14"] = true_range.rolling(14).mean()
    feature_map["atr_pct_14"] = feature_map["atr_14"] / safe_close
    plus_di_14, minus_di_14, adx_14 = _adx(high, low, close, 14)
    feature_map["plus_di_14"] = plus_di_14
    feature_map["minus_di_14"] = minus_di_14
    feature_map["adx_14"] = adx_14
    feature_map["di_spread_14"] = plus_di_14 - minus_di_14
    feature_map["di_ratio_14"] = plus_di_14 / minus_di_14.replace(0, np.nan)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_band = (bb_upper - bb_lower).replace(0, np.nan)
    feature_map["bb_mid_20"] = bb_mid
    feature_map["bb_upper_20"] = bb_upper
    feature_map["bb_lower_20"] = bb_lower
    feature_map["bb_width_20"] = bb_band / bb_mid.replace(0, np.nan)
    feature_map["bb_position_20"] = (close - bb_lower) / bb_band

    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    feature_map["obv"] = obv
    feature_map["obv_ema_20"] = obv.ewm(span=20, adjust=False).mean()
    feature_map["obv_momentum_5"] = obv.diff(5)

    typical_price = (high + low + close) / 3.0
    vwap_20 = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    feature_map["vwap_20"] = vwap_20
    feature_map["price_to_vwap_20"] = close / vwap_20.replace(0, np.nan)
    feature_map["mfi_14"] = _mfi(high, low, close, volume, 14)
    feature_map["cmf_20"] = _cmf(high, low, close, volume, 20)

    day_of_week = pd.Series(data.index.dayofweek, index=data.index, dtype=float)
    day_of_month = pd.Series(data.index.day, index=data.index, dtype=float)
    month = pd.Series(data.index.month, index=data.index, dtype=float)
    feature_map["day_of_week"] = day_of_week
    feature_map["day_of_month"] = day_of_month
    feature_map["month"] = month
    feature_map["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    feature_map["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    feature_map["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    feature_map["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    data = pd.concat([data, pd.DataFrame(feature_map, index=data.index)], axis=1)

    prediction_offset = prediction_scope + 1
    data["target"] = close.shift(-prediction_offset)

    data = data.replace([np.inf, -np.inf], np.nan)

    feature_columns = [column for column in data.columns if column != "target"]
    training_frame = data.loc[:, feature_columns + ["target"]].dropna()
    latest_features = data.loc[:, feature_columns].dropna().tail(1)

    if training_frame.empty:
        raise ValueError("Недостаточно данных после feature engineering для обучения.")
    if latest_features.empty:
        raise ValueError("Недостаточно данных для актуального вектора признаков.")

    return training_frame, latest_features, feature_columns


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    high_low = high - low
    high_close = (high - prev_close).abs()
    low_close = (low - prev_close).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    return -100.0 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    sma = typical_price.rolling(window).mean()
    mad = typical_price.rolling(window).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))),
        raw=True,
    )
    return (typical_price - sma) / (0.015 * mad.replace(0, np.nan))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    true_range = _true_range(high, low, close)
    atr = true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean() / atr.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    return plus_di, minus_di, adx


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume
    price_delta = typical_price.diff()

    positive_flow = money_flow.where(price_delta > 0, 0.0)
    negative_flow = money_flow.where(price_delta < 0, 0.0).abs()

    positive_sum = positive_flow.rolling(window).sum()
    negative_sum = negative_flow.rolling(window).sum()
    money_ratio = positive_sum / negative_sum.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + money_ratio))


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    money_flow_volume = multiplier * volume
    return money_flow_volume.rolling(window).sum() / volume.rolling(window).sum().replace(0, np.nan)


def _trix(close: pd.Series, window: int) -> pd.Series:
    ema1 = close.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    return ema3.pct_change()
