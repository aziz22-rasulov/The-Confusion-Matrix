# Smart Support Backend

Инструкция по запуску проекта и переходу в интерфейс Swagger.

## 1. Требования
- Python 3.10+
- `pip`
- Виртуальное окружение (рекомендуется)
- Доступ к API генерации эмбеддингов (замените тестовый ключ в `data/vector.py` на рабочий)

## 2. Установка зависимостей
```bash
python -m venv venv
venv\Scripts\activate                # Windows PowerShell
pip install -r requirements.txt
```

После выполнения появятся `data/knowledge_base.index` и `data/knowledge_base_metadata.json`.

## 3. Запуск backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Сервис будет доступен на `http://localhost:8000`.

## 4. Swagger UI
- Откройте `http://localhost:8000/docs` в браузере.
- Для проверки ручки `POST /dialog` отправьте JSON:
```json
{
  "text": "Ваш вопрос"
}
```
