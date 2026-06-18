# Фрагменты кода: теханализ + обучение ML-модели
## 1) Расчёт индикаторов технического анализа
Источник: `backend/src/backend/features.py`
```python
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
```

```python
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
```

## 2) Обучение модели машинного обучения
Источник: `backend/src/backend/service.py`
```python
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
        train_part = dataset.iloc[:split_index]
        validation_part = dataset.iloc[split_index:]

        x_train_parts.append(train_part.loc[:, feature_columns].to_numpy(dtype=float))
        y_train_parts.append(train_part["target"].to_numpy(dtype=float))
        x_validation_parts.append(validation_part.loc[:, feature_columns].to_numpy(dtype=float))
        y_validation_parts.append(validation_part["target"].to_numpy(dtype=float))

    return ScopeMatrices(
        x_train=np.vstack(x_train_parts),
        y_train=np.concatenate(y_train_parts),
        x_validation=np.vstack(x_validation_parts),
        y_validation=np.concatenate(y_validation_parts),
        feature_columns=feature_columns,
    )
```

```python
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
    chosen_params = dict(base_params or self._default_params(model_type))
    tuned = False

    if tune:
        search_space = self._build_search_space(model_type, base_params)
        search_size = _estimate_search_space_size(search_space)
        n_iter = max(1, min(tune_trials, search_size))
        cv_splits = self._resolve_cv_splits(len(scope_matrices.x_train))

        search = RandomizedSearchCV(
            estimator=self._build_estimator(model_type),
            param_distributions=search_space,
            n_iter=n_iter,
            scoring=scoring,
            cv=TimeSeriesSplit(n_splits=cv_splits),
            random_state=self._settings.random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(scope_matrices.x_train, scope_matrices.y_train)
        chosen_params = dict(search.best_params_)
        tuned = True

    validation_model = self._build_estimator(model_type, chosen_params)
    validation_model.fit(scope_matrices.x_train, scope_matrices.y_train)
    val_predictions = validation_model.predict(scope_matrices.x_validation)
    metrics = _calculate_metrics(scope_matrices.y_validation, val_predictions)

    final_model = validation_model
    if full_refit:
        refit = clone(self._build_estimator(model_type, chosen_params))
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
    }
```

## 3) Оркестрация обучения (выбор лучшей модели и обучение по горизонтам)
Источник: `backend/src/backend/service.py`
```python
def _run_training(self, request: TrainRequest) -> TrainOutcome:
    existing_metadata = self._load_existing_metadata_if_needed(request)
    effective_request = self._enrich_request_with_existing_symbols(request, existing_metadata)
    prepared = self._prepare_dataset(effective_request)

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

    for model_type in candidate_types:
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

    trained_model_type = best_result["model_type"]
    unique_scopes = sorted(set(prepared.day_scope_map.values()))
    models_by_scope: dict[str, Any] = {}
    metrics_by_scope: dict[str, dict[str, float]] = {}

    for scope in unique_scopes:
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
```

## 4) Дополнительный фрагмент: подбор гиперпараметров и full refit
Источник: `backend/src/backend/service.py`
```python
if tune:
    search_space = self._build_search_space(model_type, base_params)
    search_size = _estimate_search_space_size(search_space)
    n_iter = max(1, min(tune_trials, search_size))
    cv_splits = self._resolve_cv_splits(len(scope_matrices.x_train))

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
    tuned = True

validation_model = self._build_estimator(
    model_type,
    chosen_params,
    huber_slope=huber_slope,
)
validation_model.fit(scope_matrices.x_train, scope_matrices.y_train)

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
```

## 5) Точка входа обучения
Источник: `backend/src/backend/service.py`
```python
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
    return outcome.response
```

## 6) Ручной `TimeSeriesSplit` подбор параметров (ветка для XGBoost)
Источник: `backend/src/backend/service.py`
```python
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
    return best_params
```

## 7) Подготовка датасета перед обучением
Источник: `backend/src/backend/service.py`
```python
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
```

## 8) Формирование `target` и итоговых матриц признаков
Источник: `backend/src/backend/features.py`
```python
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
```
