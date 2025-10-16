# Smart Support Backend

Инструкция по запуску проекта и переходу в интерфейс Swagger.

## 1. Требования
- Python 3.10+
- `pip`
- Docker 24+ и Docker Compose v2 (для контейнерного запуска)
- Виртуальное окружение (рекомендуется для локального запуска без Docker)
- Доступ к API генерации эмбеддингов (замените тестовый ключ в `data/vector.py` на рабочий)

## 2. Запуск через Docker
1. Соберите и поднимите сервисы:
   ```bash
   docker compose up --build
   ```
   При необходимости можно добавить флаг `-d`, чтобы запустить контейнеры в фоне.
2. Дождитесь, пока билд завершится и в логах появятся сообщения о старте `uvicorn` и `vite preview`.
3. Откройте сервисы:
   - Backend: `http://localhost:8000`
   - Swagger UI: `http://localhost:8000/docs`
   - Frontend: `http://localhost:5173`
4. Остановите работу контейнеров:
   ```bash
   docker compose down
   ```

## 3. Установка зависимостей (локальный запуск)
```bash
python -m venv venv
venv\Scripts\activate                # Windows PowerShell
pip install -r requirements.txt
```

После выполнения появятся `data/knowledge_base.index` и `data/knowledge_base_metadata.json`.

## 4. Запуск backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Сервис будет доступен на `http://localhost:8000`.

## 5. Swagger UI
- Откройте `http://localhost:8000/docs` в браузере.
- Для проверки ручки `POST /dialog` отправьте JSON:
```json
{
  "text": "Ваш вопрос"
}
```
