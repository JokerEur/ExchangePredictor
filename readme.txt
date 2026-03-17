# ExchangePredictor (Production-grade skeleton)

## Архитектура

### Frontend (FSD)
Frontend разделен по FSD-слоям:
- `frontend/src/app` — app entrypoint и глобальные стили;
- `frontend/src/pages` — page composition;
- `frontend/src/widgets` — крупные UI-блоки (торговый терминал);
- `frontend/src/shared/api` — API-клиент и сетевые утилиты;
- `frontend/src/App.jsx`, `frontend/src/api.js` — backward-compatible wrappers.

### Backend (DDD-style layering)
Backend разделен на DDD-слои:
- `backend/src/backend/domain` — порты/доменные контракты;
- `backend/src/backend/application` — use-cases;
- `backend/src/backend/infrastructure` — адаптеры (legacy ML service, queue, persistence);
- `backend/src/backend/presentation` — HTTP-слой FastAPI;
- `backend/src/backend/shared` — настройки и cross-cutting.

Текущая ML-логика сохранена через legacy-адаптер, поэтому контракты основных endpoint-ов остаются совместимыми.

## Инфраструктура
- Docker compose orchestration: `docker-compose.yml`
- PostgreSQL: хранение метаданных async training jobs
- Redis: брокер/бекенд очередей
- Celery worker: фоновые train-задачи
- Alembic: миграции БД

## Локальный запуск (без Docker)
1. Установить зависимости backend:
   - `cd backend`
   - `pip install -r requirements.txt`
2. Запустить backend:
   - `cd backend`
   - `uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload`
3. Запустить frontend:
   - `cd frontend`
   - `npm install`
   - `npm run dev`
4. Swagger:
   `http://127.0.0.1:8000/docs`

## Запуск через Docker
1. Скопировать env:
   `cp backend/.env.example backend/.env`
2. Поднять стек:
   `docker compose up --build`
3. Backend:
   `http://127.0.0.1:8000/docs`
4. Frontend:
   `http://127.0.0.1:5173`

## Основные endpoint-ы
- `GET /health`
- `GET /market/candles`
- `POST /train` (sync training)
- `POST /train/async` (enqueue training job)
- `GET /train/jobs/{job_id}` (async status/result)
- `POST /predict`
- `GET /models`

## Async training flow
1. Клиент отправляет `POST /train/async`.
2. API создает `job_id`, сохраняет queued job в БД и отправляет задачу в Celery.
3. Worker выполняет обучение, сохраняет `succeeded`/`failed` + payload/error.
4. Клиент опрашивает `GET /train/jobs/{job_id}`.

## Конфигурация
Основные переменные окружения:
- `DATABASE_URL`
- `REDIS_URL`
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `MODEL_CONFIG_PATH`
- `MODEL_REGISTRY_PATH`

Гиперпараметры моделей и search spaces:
- `backend/config/model_params.yml`

## Миграции
- Применить:
  `cd backend && alembic upgrade head`
- Откатить на шаг:
  `cd backend && alembic downgrade -1`
