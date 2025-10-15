import pandas as pd
import requests
import json

# Загружаем Excel-файл
df = pd.read_excel('smart_support_vtb_belarus_faq_final.xlsx')

# Преобразуем таблицу в список строк (например, объединяя столбцы)
# Если у тебя есть конкретная колонка с текстом для эмбеддингов — используй её:
texts = df.astype(str).agg(' '.join, axis=1).tolist()

# Формируем JSON-запрос
data = {
    "model": "bge-m3",
    "input": texts
}
print(data)

# Отправляем POST-запрос
response = requests.post(
    "https://llm.t1v.scibox.tech/v1/embeddings",
    headers={
        "Authorization" : "Bearer sk-dgKROD7rG4yPAo7bTOtatA",
        "Content-Type" : "application/json"
    },
    data=json.dumps(data, ensure_ascii=False).encode('utf-8')
)

# Проверяем ответ

inf_for_std = response.json()

vectors = inf_for_std["data"][0]["embedding"]
print(vectors)