import pandas as pd
import requests
import json
import numpy as np
import faiss
import os
from tqdm import tqdm

# ==============================
# Конфигурация
# ==============================
API_URL = "https://llm.t1v.scibox.tech/v1/embeddings"
API_KEY = "sk-dgKROD7rG4yPAo7bTOtatA"  # замените на ваш ключ
MODEL = "bge-m3"
BATCH_SIZE = 128
DATA_DIR = "data"

# ==============================
# Загрузка данных
# ==============================
df = pd.read_excel('smart_support_vtb_belarus_faq_final.xlsx')

required_cols = ['Пример вопроса', 'Шаблонный ответ']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Не найдена колонка '{col}' в Excel-файле")

df['combined_text'] = df['Пример вопроса'].astype(str) + ". " + df['Шаблонный ответ'].astype(str)
texts = df['combined_text'].tolist()
print(f"Загружено {len(texts)} записей для обработки")

# ==============================
# Функция получения эмбеддингов
# ==============================
def get_embeddings_batch(text_batch):
    data = {"model": MODEL, "input": text_batch}
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        data=json.dumps(data, ensure_ascii=False).encode('utf-8')
    )
    if response.status_code != 200:
        print(f"Ошибка API ({response.status_code}): {response.text[:200]}")
        return []
    result = response.json()
    return [item["embedding"] for item in result.get("data", [])]

# ==============================
# Генерация эмбеддингов батчами
# ==============================
embeddings = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Генерация эмбеддингов"):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_embs = get_embeddings_batch(batch_texts)
    embeddings.extend(batch_embs)

if not embeddings:
    raise RuntimeError("Не удалось получить эмбеддинги")

print(f"Получено {len(embeddings)} эмбеддингов")

# ==============================
# Построение FAISS индекса
# ==============================
os.makedirs(DATA_DIR, exist_ok=True)

vectors = np.array(embeddings).astype('float32')
dimension = vectors.shape[1]
print(f"Размерность эмбеддингов: {dimension}")

# Нормализуем векторы для корректного косинусного сходства
faiss.normalize_L2(vectors)

index = faiss.IndexFlatIP(dimension)
index.add(vectors)
print(f"Добавлено {index.ntotal} векторов в FAISS индекс")

index_path = os.path.join(DATA_DIR, "knowledge_base.index")
faiss.write_index(index, index_path)
print(f"FAISS индекс сохранен в {index_path}")

# ==============================
# Сохранение метаданных
# ==============================
metadata = {
    'ids': list(range(len(texts))),
    'main_categories': df.get('Основная категория', [''] * len(df)).tolist(),
    'sub_categories': df.get('Подкатегория', [''] * len(df)).tolist(),
    'example_questions': df['Пример вопроса'].tolist(),
    'target_audiences': df.get('Целевая аудитория', [''] * len(df)).tolist(),
    'template_answers': df['Шаблонный ответ'].tolist(),
    'combined_texts': texts
}
metadata_path = os.path.join(DATA_DIR, "knowledge_base_metadata.json")
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"Метаданные сохранены в {metadata_path}")

# ==============================
# Тестовый поиск
# ==============================
print("\nВыполняем тестовый поиск...")

test_query = "Как восстановить доступ к интернет-банку?"

test_data = {"model": MODEL, "input": [test_query]}
test_response = requests.post(
    API_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    data=json.dumps(test_data, ensure_ascii=False).encode('utf-8')
)

if test_response.status_code == 200:
    test_embedding = np.array([test_response.json()['data'][0]['embedding']]).astype('float32')
    faiss.normalize_L2(test_embedding)

    k = 3
    distances, indices = index.search(test_embedding, k=k)

    print(f"\nТоп-{k} результатов для запроса: '{test_query}'\n")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. [{metadata['main_categories'][idx]} → {metadata['sub_categories'][idx]}]")
        print(f"   Схожесть: {distances[0][i]:.4f}")
        print(f"   Вопрос: {metadata['example_questions'][idx]}")
        print(f"   Ответ: {metadata['template_answers'][idx][:150]}...\n")
else:
    print(f"Ошибка при тестовом поиске: {test_response.status_code}")
    print(test_response.text)
