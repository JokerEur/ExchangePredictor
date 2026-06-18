from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "tuning": {
        "scoring": "mae",
        "huber_delta": 1.0,
        "huber_slope_short": 1.0,
        "huber_slope_long": 0.3,
        "search_n_jobs": -1,
        "cv_rules": {
            "small_train_threshold": 80,
            "medium_train_threshold": 160,
            "small_train_splits": 2,
            "medium_train_splits": 3,
            "default_splits": 3,
        },
    },
    "models": {
        "xgboost": {
            "default_params": {
                "n_estimators": 600,
                "learning_rate": 0.03,
                "max_depth": 6,
                "min_child_weight": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "gamma": 0.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "objective": "reg:pseudohubererror",
                "huber_slope": 1.0,
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": -1,
            },
            "search_space": {
                "n_estimators": [250, 400, 600, 850, 1100],
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
                "max_depth": [3, 4, 5, 6, 8],
                "min_child_weight": [1, 2, 3, 5, 7],
                "subsample": [0.65, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.75, 0.85, 1.0],
                "gamma": [0.0, 0.05, 0.1, 0.2],
                "reg_alpha": [0.0, 0.001, 0.01, 0.05, 0.1],
                "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
                "huber_slope": [0.1, 0.2, 0.3, 0.5, 1.0],
            },
        },
    },
}


def load_model_params(path: Path) -> dict[str, Any]:
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    if not path.exists():
        return config

    with path.open("r", encoding="utf-8") as file:
        user_data = yaml.safe_load(file) or {}

    if not isinstance(user_data, dict):
        raise ValueError("model_params.yml должен содержать YAML-объект верхнего уровня.")

    return _deep_merge(config, user_data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
