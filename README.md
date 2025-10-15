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

## 3. Подготовка базы знаний
1. Поместите `smart_support_vtb_belarus_faq_final.xlsx` в корень проекта (`c:\MINSK_WORK`).  
2. Убедитесь, что в `data/vector.py` корректно указан API-ключ и URL.  
3. Сгенерируйте векторный индекс и метаданные:
```bash
python -m data.vector
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
При наличии FAISS-индекса ответ вернётся из векторного поиска, иначе сработает запасной fuzzy-поиск.

## 6. Полезные команды
- Проверка файлов Python на синтаксис:
```bash
python -m compileall backend/support/support.py data/vector.py
```
- Перегенерация индекса после обновления Excel:
```bash
python -m data.vector
```

При необходимости можно добавить инструкции по запуску фронтенда или Docker позднее.
