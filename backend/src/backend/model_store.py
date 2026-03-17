import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass
class StoredModel:
    model_id: str
    model: Any
    metadata: dict[str, Any]
    model_path: Path
    metadata_path: Path


class ModelStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, model_id: str, model: Any, metadata: dict[str, Any]) -> StoredModel:
        model_dir = self._resolve_model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.joblib"
        metadata_path = model_dir / "metadata.json"

        joblib.dump(model, model_path)
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        return StoredModel(
            model_id=self._normalize_model_id(model_id),
            model=model,
            metadata=metadata,
            model_path=model_path,
            metadata_path=metadata_path,
        )

    def load(self, model_id: str) -> StoredModel:
        model_dir = self._resolve_model_dir(model_id)
        model_path = model_dir / "model.joblib"
        metadata_path = model_dir / "metadata.json"

        if not model_path.exists() or not metadata_path.exists():
            raise ValueError(f"Модель '{model_id}' не найдена в registry.")

        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        model = joblib.load(model_path)
        return StoredModel(
            model_id=self._normalize_model_id(model_id),
            model=model,
            metadata=metadata,
            model_path=model_path,
            metadata_path=metadata_path,
        )

    def list_metadata(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self._root.glob("*/metadata.json")):
            with path.open("r", encoding="utf-8") as file:
                items.append(json.load(file))
        return items

    def _resolve_model_dir(self, model_id: str) -> Path:
        return self._root / self._normalize_model_id(model_id)

    @staticmethod
    def _normalize_model_id(model_id: str) -> str:
        normalized = re.sub(r"[^0-9A-Za-z_-]+", "-", model_id.strip()).strip("-")
        if not normalized:
            raise ValueError("model_id должен содержать буквенно-цифровые символы.")
        return normalized
