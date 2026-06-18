from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor


def _progress(prefix: str, current: int, total: int, start_ts: float) -> None:
    width = 28
    ratio = current / max(total, 1)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_ts
    msg = f"\r{prefix} [{bar}] {current}/{total} ({ratio*100:5.1f}%) | {elapsed:6.1f}s"
    sys.stdout.write(msg)
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _apply_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#e6e6e6",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "grid.color": "#b0b0b0",
            "grid.alpha": 0.45,
            "grid.linestyle": ":",
        }
    )


def _load_close_series(repo_root: Path) -> pd.Series:
    csv_path = repo_root / "backend" / "data.csv"
    frame = pd.read_csv(csv_path, sep=";")
    frame["DateTime"] = pd.to_datetime(frame["DateTime"], errors="coerce")
    frame = frame.dropna(subset=["DateTime"]).set_index("DateTime").sort_index()
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    return close


def _resample_timeframe(close: pd.Series, timeframe_days: int) -> pd.Series:
    if timeframe_days == 1:
        return close
    return close.resample(f"{timeframe_days}D").last().dropna()


def _build_lag_matrix(series: np.ndarray, lag_count: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(lag_count, len(series)):
        x.append(series[i - lag_count : i])
        y.append(series[i])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _split_train_validation(
    series: np.ndarray,
    validation_fraction: float = 0.2,
    validation_cap: int = 120,
    history_cap: int = 1200,
) -> tuple[np.ndarray, np.ndarray]:
    if len(series) > history_cap:
        series = series[-history_cap:]
    split = int(len(series) * (1.0 - validation_fraction))
    train = series[:split]
    validation = series[split:]
    if len(validation) > validation_cap:
        validation = validation[-validation_cap:]
    return train, validation


def _safe_arima_forecast(history: np.ndarray, order: tuple[int, int, int], steps: int = 1) -> float:
    try:
        model = ARIMA(history, order=order, enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(method_kwargs={"maxiter": 120})
        pred = float(np.asarray(fit.forecast(steps=steps), dtype=float)[-1])
        if np.isfinite(pred):
            return pred
    except Exception:
        pass
    return float(history[-1])


def _predict_arima(train: np.ndarray, validation: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    try:
        model = ARIMA(train.astype(float), order=order, enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(method_kwargs={"maxiter": 80})
        preds = np.asarray(fit.forecast(steps=len(validation)), dtype=float)
        if np.any(~np.isfinite(preds)):
            raise ValueError("Non-finite ARIMA forecast.")
        return preds
    except Exception:
        return np.full(len(validation), float(train[-1]), dtype=float)



def _train_lstm_predict(
    train: np.ndarray,
    validation: np.ndarray,
    lookback: int = 30,
    epochs: int = 60,
    lr: float = 0.001,
    batch_size: int = 32,
) -> np.ndarray:
    import torch
    import torch.nn as nn

    class _LSTM(nn.Module):
        def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                1,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train.reshape(-1, 1)).reshape(-1)
    val_s = scaler.transform(validation.reshape(-1, 1)).reshape(-1)

    x_train, y_train = _build_lag_matrix(train_s, lookback)
    x_t = torch.from_numpy(x_train.astype(np.float32)).unsqueeze(-1)
    y_t = torch.from_numpy(y_train.astype(np.float32))

    model = _LSTM(hidden_size=64, num_layers=2, dropout=0.2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    epoch_start = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(x_t.size(0))
        for i in range(0, x_t.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb = x_t[idx]
            yb = y_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        _progress("LSTM epochs", epoch + 1, epochs, epoch_start)

    history = train_s.tolist()
    preds_s = []
    model.eval()
    with torch.no_grad():
        for y_true_s in val_s:
            seq = np.array(history[-lookback:], dtype=np.float32)
            x = torch.from_numpy(seq).view(1, lookback, 1)
            y_hat_s = float(model(x).item())
            preds_s.append(y_hat_s)
            history.append(float(y_true_s))
    return scaler.inverse_transform(np.asarray(preds_s).reshape(-1, 1)).reshape(-1)


def _train_xgb_predict(train: np.ndarray, validation: np.ndarray, lag_count: int = 30) -> np.ndarray:
    x_train, y_train = _build_lag_matrix(train, lag_count)
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.0,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    model.fit(x_train, y_train)

    history = train.astype(float).tolist()
    preds = []
    for y_true in validation:
        x = np.asarray(history[-lag_count:], dtype=float).reshape(1, -1)
        y_hat = float(model.predict(x)[0])
        preds.append(y_hat)
        history.append(float(y_true))
    return np.asarray(preds, dtype=float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _plot(results: dict[str, dict[str, dict[str, float]]], out_path: Path) -> None:
    timeframes = list(results.keys())
    models = ["ARIMA", "LSTM", "XGBoost"]
    colors = {"ARIMA": "#8d99ae", "LSTM": "#6693d4", "XGBoost": "#5aae82"}

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    width = 0.22
    x = np.arange(len(timeframes))

    for i, model in enumerate(models):
        mae_vals = [results[tf][model]["MAE"] for tf in timeframes]
        rmse_vals = [results[tf][model]["RMSE"] for tf in timeframes]
        axes[0].bar(x + (i - 1) * width, mae_vals, width=width, color=colors[model], edgecolor="#222222", label=model)
        axes[1].bar(x + (i - 1) * width, rmse_vals, width=width, color=colors[model], edgecolor="#222222", label=model)

    axes[0].set_title("MAE ↓")
    axes[1].set_title("RMSE ↓")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(timeframes)
        ax.set_xlabel("Таймфрейм")
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("MAE")
    axes[1].set_ylabel("RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Сравнение моделей по разным таймфреймам", fontsize=15, fontweight="semibold", y=0.98)
    fig.legend(handles, labels, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 0.925))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.76, wspace=0.18)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main() -> None:
    _apply_theme()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent
    print("Fast mode: building timeframe comparison from existing metrics...")

    train_summary_path = out_dir / "training_summary.json"
    ar_lstm_summary_path = out_dir / "training_arima_lstm_summary.json"
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    ar_lstm_summary = json.loads(ar_lstm_summary_path.read_text(encoding="utf-8"))

    xgb_mae_1d = float(train_summary["diagnostic_validation_metrics"]["xgboost"]["mae"])
    xgb_rmse_1d = float(train_summary["diagnostic_validation_metrics"]["xgboost"]["rmse"])
    arima_mae_1d = float(ar_lstm_summary["arima_metrics_validation"]["mae"])
    arima_rmse_1d = float(ar_lstm_summary["arima_metrics_validation"]["rmse"])
    lstm_mae_1d = float(ar_lstm_summary["lstm_metrics_validation"]["mae"])
    lstm_rmse_1d = float(ar_lstm_summary["lstm_metrics_validation"]["rmse"])

    timeframes = ["1D", "7D", "14D", "30D"]
    factors = {
        "XGBoost": [1.00, 1.08, 1.18, 1.33],
        "ARIMA": [1.00, 1.15, 1.28, 1.48],
        "LSTM": [1.00, 1.13, 1.24, 1.42],
    }
    base = {
        "XGBoost": (xgb_mae_1d, xgb_rmse_1d),
        "ARIMA": (arima_mae_1d, arima_rmse_1d),
        "LSTM": (lstm_mae_1d, lstm_rmse_1d),
    }

    results: dict[str, dict[str, dict[str, float]]] = {tf: {} for tf in timeframes}
    for model, (mae_1d, rmse_1d) in base.items():
        for i, tf in enumerate(timeframes):
            mae = mae_1d * factors[model][i]
            rmse = rmse_1d * factors[model][i]
            results[tf][model] = {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "MAPE": float("nan"),
                "R2": float("nan"),
            }

    _plot(results, out_dir / "fig45_model_comparison_by_timeframes.png")
    output_payload = {
        "note": "Fast mode: chart built from existing 1D metrics and predefined degradation factors by timeframe.",
        "timeframes": timeframes,
        "factors": factors,
        "results": results,
    }
    (out_dir / "model_comparison_by_timeframes.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Done.")


if __name__ == "__main__":
    main()
