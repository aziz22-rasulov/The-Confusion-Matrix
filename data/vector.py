import pandas as pd
import requests
import json
import numpy as np
import faiss
import os

# Загружаем Excel-файл
df = pd.read_excel('smart_support_vtb_belarus_faq_final.xlsx')

# Преобразуем таблицу в список строк (объединяя "Пример вопроса" и "Шаблонный ответ")
df['combined_text'] = df['Пример вопроса'].astype(str) + ". " + df['Шаблонный ответ'].astype(str)
texts = df['combined_text'].tolist()

# Формируем JSON-запрос
data = {
    "model": "bge-m3",
    "input": texts
}

# Отправляем POST-запрос
response = requests.post(
    "https://llm.t1v.scibox.tech/v1/embeddings",
    headers={
        "Authorization": "Bearer sk-dgKROD7rG4yPAo7bTOtatA",
        "Content-Type": "application/json"
    },
    data=json.dumps(data, ensure_ascii=False).encode('utf-8')
)

# Проверяем ответ
if response.status_code == 200:
    inf_for_std = response.json()

    # Правильно извлекаем все эмбеддинги из ответа
    embeddings = [item['embedding'] for item in inf_for_std['data']]

    print(f"Получено {len(embeddings)} эмбеддингов")
    print(f"Размерность одного эмбеддинга: {len(embeddings[0])}")

    # ================================================
    # ДОБАВЛЕННЫЙ КОД ДЛЯ РАБОТЫ С FAISS (начало)
    # ================================================

    # Создаем директорию для сохранения, если её нет
    os.makedirs('data', exist_ok=True)

    # Преобразуем список эмбеддингов в массив numpy
    vectors = np.array(embeddings).astype('float32')

    # Определяем размерность эмбеддингов
    dimension = vectors.shape[1]
    print(f"Размерность эмбеддингов: {dimension}")

    # Создаем FAISS индекс (используем внутреннее произведение)
    index = faiss.IndexFlatIP(dimension)

    # Добавляем векторы в индекс
    index.add(vectors)
    print(f"Добавлено {index.ntotal} векторов в FAISS индекс")

    # Сохраняем индекс
    index_path = "data/knowledge_base.index"
    faiss.write_index(index, index_path)
    print(f"FAISS индекс сохранен в {index_path}")

    # Сохраняем метаданные
    metadata = {
        'ids': list(range(len(texts))),
        'main_categories': df['Основная категория'].tolist(),
        'sub_categories': df['Подкатегория'].tolist(),
        'example_questions': df['Пример вопроса'].tolist(),
        'target_audiences': df['Целевая аудитория'].tolist(),
        'template_answers': df['Шаблонный ответ'].tolist(),
        'combined_texts': texts
    }

    metadata_path = "data/knowledge_base_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Метаданные сохранены в {metadata_path}")

    # Проверка: тестовый поиск
    print("\nВыполняем тестовый поиск...")

    test_query = "Я очень хочу есть"

    # Создаем эмбеддинг для тестового запроса
    test_data = {
        "model": "bge-m3",
        "input": [test_query]
    }

    test_response = requests.post(
        "https://llm.t1v.scibox.tech/v1/embeddings",
        headers={
            "Authorization": "Bearer sk-dgKROD7rG4yPAo7bTOtatA",
            "Content-Type": "application/json"
        },
        data=json.dumps(test_data, ensure_ascii=False).encode('utf-8')
    )

    if test_response.status_code == 200:
        test_embedding = test_response.json()['data'][0]['embedding']
        test_vector = np.array([test_embedding]).astype('float32')

        # Ищем ближайшие векторы (k=3)
        distances, indices = index.search(test_vector, k=3)

        # Выводим результаты
        print(f"\nТоп-3 результата для запроса: '{test_query}'")
        for i, idx in enumerate(indices[0]):
            print(indices[0][i])
            print(f"{i + 1}. [{metadata['main_categories'][idx]} → {metadata['sub_categories'][idx]}]")
            print(f"   Уверенность: {1 / (1 + distances[0][i]):.4f}")
            print(f"   Вопрос: {metadata['example_questions'][idx]}")
            print(f"   Ответ: {metadata['template_answers'][idx][:100]}...\n")
    else:
        print(f"Ошибка при тестовом поиске: {test_response.status_code}")
        print(test_response.text)

    # ================================================
    # ДОБАВЛЕННЫЙ КОД ДЛЯ РАБОТЫ С FAISS (конец)
    # ================================================

else:
    print(f"Ошибка при запросе: {response.status_code}")
    print(response.text)
