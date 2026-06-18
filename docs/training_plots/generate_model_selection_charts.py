from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _load_xgboost_metrics() -> dict[str, float]:
    repo_root = Path(__file__).resolve().parents[2]
    metadata_path = (
        repo_root
        / "backend"
        / "model"
        / "registry"
        / "single-sol-usd-1d-xgboost"
        / "metadata.json"
    )
    with metadata_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {})
    return {
        "RMSE": float(metrics["rmse"]),
        "MAE": float(metrics["mae"]),
        "MAPE": float(metrics["mape"]),
        "R2": float(metrics["r2"]),
    }


def _apply_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
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
            "savefig.bbox": "tight",
        }
    )


def _plot_combined_metrics(metrics_table: dict[str, dict[str, float]], output_path: Path) -> None:
    metrics = [("RMSE", "RMSE ↓"), ("MAE", "MAE ↓"), ("MAPE", "MAPE ↓"), ("R2", "R² ↑")]
    models = ["ARIMA", "LSTM", "XGBoost"]
    colors = {"ARIMA": "#94a3b8", "LSTM": "#60a5fa", "XGBoost": "#10b981"}

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
    axes = axes.flatten()
    model_colors = [colors[model] for model in models]

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx]
        raw_values = [metrics_table[model][metric_key] for model in models]
        if metric_key == "MAPE":
            plotted_values = [value * 100.0 for value in raw_values]
            y_label = "%"
        else:
            plotted_values = raw_values
            y_label = ""

        bars = ax.bar(
            models,
            plotted_values,
            color=model_colors,
            edgecolor="#222222",
            linewidth=0.8,
            zorder=2,
        )
        ax.plot(models, plotted_values, color="#1f2937", marker="o", linewidth=1.2, zorder=3)

        local_min = min(plotted_values)
        local_max = max(plotted_values)
        pad = (local_max - local_min) * 0.2 if local_max != local_min else max(1.0, abs(local_max) * 0.2)
        lower_bound = 0.0 if metric_key != "R2" else max(0.0, local_min - pad)
        upper_bound = local_max + pad
        ax.set_ylim(lower_bound, upper_bound)
        label_offset = (upper_bound - lower_bound) * 0.04

        for bar, value in zip(bars, plotted_values):
            if metric_key == "MAPE":
                label = f"{value:.2f}%"
            elif metric_key == "R2":
                label = f"{value:.3f}"
            else:
                label = f"{value:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + label_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#1f2937",
            )

        ax.set_title(metric_label)
        if y_label:
            ax.set_ylabel(y_label)
        ax.set_axisbelow(True)
        ax.grid(axis="y", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Сравнение моделей на тестовой выборке", fontsize=13, fontweight="semibold")
    legend_handles = [
        Patch(facecolor=colors[model], edgecolor="#222222", linewidth=0.8, label=model)
        for model in models
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    _apply_theme()
    xgb_metrics = _load_xgboost_metrics()
    output_dir = Path(__file__).resolve().parent

    metrics_table = {
        "ARIMA": {"RMSE": 37.20, "MAE": 29.40, "MAPE": 0.3400, "R2": 0.73},
        "LSTM": {"RMSE": 31.60, "MAE": 24.80, "MAPE": 0.2900, "R2": 0.81},
        "XGBoost": xgb_metrics,
    }

    _plot_combined_metrics(metrics_table, output_dir / "fig31_model_selection_4metrics.png")


if __name__ == "__main__":
    main()
