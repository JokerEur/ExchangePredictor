import logging
from typing import Any, Optional

import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Внутренняя архитектура сети
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:
    class _LSTMNet(nn.Module):  # type: ignore[misc]
        def __init__(self, input_size: int, hidden_size: int, num_layers: int):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)
else:
    _LSTMNet = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Публичный класс-обёртка
# ---------------------------------------------------------------------------

class LSTMWrapper:
    """
    Обёртка над PyTorch LSTM для использования в ансамблевом прогнозировании.

    При обучении строит скользящие окна из матрицы признаков X.
    При предсказании принимает последовательность из `sequence_len` строк признаков.

    Для совместимости с pipeline-обработкой validation-набора
    метод `predict(X)` использует последние `sequence_len - 1` обучающих строк
    в качестве контекста для каждой предсказываемой строки.
    """

    def __init__(
        self,
        sequence_len: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 30,
        lr: float = 0.001,
        batch_size: int = 32,
    ):
        self.sequence_len = sequence_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self._model: Optional[Any] = None
        self._input_size: Optional[int] = None
        # Нормализация признаков
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        # Нормализация таргета
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        # Контекст для predict(): последние sequence_len строк из обучения
        self._train_context: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Нормализация
    # ------------------------------------------------------------------

    def _norm_x(self, X: np.ndarray) -> np.ndarray:
        return (X - self._x_mean) / (self._x_std + 1e-8)

    def _denorm_y(self, y_norm: float) -> float:
        return float(y_norm * self._y_std + self._y_mean)

    # ------------------------------------------------------------------
    # Построение последовательностей
    # ------------------------------------------------------------------

    def _build_sequences(
        self, X_norm: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Строит sliding-window последовательности из нормализованного X."""
        seqs, targets = [], []
        for i in range(self.sequence_len, len(X_norm)):
            seqs.append(X_norm[i - self.sequence_len : i])
            targets.append(y[i])
        return np.array(seqs, dtype=np.float32), np.array(targets, dtype=np.float32)

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMWrapper":
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch не установлен. Добавьте 'torch' в requirements.txt "
                "и выполните pip install -r requirements.txt."
            )

        n_samples, n_features = X.shape
        if n_samples <= self.sequence_len + 1:
            raise ValueError(
                f"Недостаточно данных для LSTM: нужно > {self.sequence_len + 1} строк, "
                f"получено {n_samples}."
            )

        self._input_size = n_features
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        self._y_mean = float(y.mean())
        self._y_std = max(float(y.std()), 1e-8)

        X_norm = self._norm_x(X)
        y_norm = (y - self._y_mean) / self._y_std

        # Сохраняем контекст для предсказания
        self._train_context = X_norm[-self.sequence_len :]

        X_seq, y_seq = self._build_sequences(X_norm, y_norm)

        self._model = _LSTMNet(n_features, self.hidden_size, self.num_layers)
        optimizer_obj = optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_tensor = torch.from_numpy(X_seq)
        y_tensor = torch.from_numpy(y_seq)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self._model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer_obj.zero_grad()
                pred = self._model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer_obj.step()
            if (epoch + 1) % 10 == 0:
                LOGGER.debug("LSTM training | epoch=%d/%d", epoch + 1, self.epochs)

        LOGGER.info(
            "LSTM trained | n_samples=%d | n_features=%d | seq_len=%d | epochs=%d",
            n_samples,
            n_features,
            self.sequence_len,
            self.epochs,
        )
        return self

    # ------------------------------------------------------------------
    # Предсказание
    # ------------------------------------------------------------------

    def predict_from_sequence(self, feature_sequence: np.ndarray) -> float:
        """
        Предсказание из готовой последовательности признаков.

        Args:
            feature_sequence: массив формы (sequence_len, n_features).

        Returns:
            Прогнозная цена.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch не установлен.")
        if self._model is None:
            raise ValueError("LSTMWrapper не обучена. Вызовите fit() сначала.")

        seq_norm = self._norm_x(feature_sequence)
        x = torch.from_numpy(seq_norm.astype(np.float32)).unsqueeze(0)

        self._model.eval()
        with torch.no_grad():
            pred_norm = self._model(x).item()

        return self._denorm_y(pred_norm)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание для матрицы признаков (совместимость с sklearn).

        Для каждой строки X[i] строит последовательность:
        [контекст из обучения / предыдущие строки X] + [X[i]].
        """
        if self._model is None or self._train_context is None:
            raise ValueError("LSTMWrapper не обучена.")

        results = []
        context = self._train_context  # (sequence_len, n_features) – нормализовано

        for i in range(len(X)):
            # Нормализуем текущую строку
            row_norm = self._norm_x(X[i : i + 1])  # (1, n_features)

            if i == 0:
                # Берём последние (seq_len - 1) строк контекста + текущую строку
                seq = np.vstack([context[-(self.sequence_len - 1) :], row_norm])
            else:
                # Заменяем первую строку контекста
                prev_norm = self._norm_x(X[:i])  # все предыдущие строки val-набора
                tail = np.vstack([context, prev_norm])[-(self.sequence_len - 1) :]
                seq = np.vstack([tail, row_norm])

            # Дополняем/обрезаем до точной длины sequence_len
            if len(seq) < self.sequence_len:
                padding = np.zeros((self.sequence_len - len(seq), seq.shape[1]), dtype=np.float32)
                seq = np.vstack([padding, seq])
            else:
                seq = seq[-self.sequence_len :]

            results.append(self.predict_from_sequence(seq))

        return np.array(results, dtype=float)
