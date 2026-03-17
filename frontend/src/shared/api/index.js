const API_BASE = import.meta.env.VITE_API_BASE || "/api";

function buildQuery(params) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") {
      return;
    }
    query.set(key, String(value));
  });
  return query.toString();
}

async function parseResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail =
      payload?.detail ||
      payload?.message ||
      `Request failed with status ${response.status}`;
    throw new Error(detail);
  }
  return payload;
}

export async function getCandles({ symbol, timeframe = "1d", limit = 60 }) {
  const query = buildQuery({
    source: "exchange",
    symbol,
    timeframe,
    limit,
  });
  const response = await fetch(`${API_BASE}/market/candles?${query}`);
  return parseResponse(response);
}

export async function getPrediction({
  symbol,
  forecastDays = 14,
  timeframe = "1d",
  modelType = "xgboost",
  modelId,
  symbols,
}) {
  const effectiveModelType = modelType === "random_forest" ? "random_forest" : "xgboost";
  const symbolKey = String(symbol || "BTC/USD")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  const timeframeKey = String(timeframe || "1d")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  const effectiveModelId = modelId || `single-${symbolKey}-${timeframeKey}-${effectiveModelType}`;
  const body = {
    model_id: effectiveModelId,
    use_saved_model: true,
    auto_train_if_missing: true,
    auto_train_mode: "train",
    auto_model_type: effectiveModelType,
    tune_on_auto_train: true,
    tune_trials: 20,
    source: "exchange",
    symbol,
    timeframe,
    forecast_days: forecastDays,
    exchange_limit: 450,
  };
  if (Array.isArray(symbols) && symbols.length > 0) {
    body.symbols = symbols;
  }

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  return parseResponse(response);
}

export async function getFeaturePreview({
  symbol,
  timeframe = "1d",
  forecastDays = 14,
  source = "exchange",
  exchangeLimit = 600,
  historySize = 72,
}) {
  const query = buildQuery({
    source,
    symbol,
    timeframe,
    forecast_days: forecastDays,
    exchange_limit: exchangeLimit,
    history_size: historySize,
  });
  const response = await fetch(`${API_BASE}/features/preview?${query}`);
  return parseResponse(response);
}
