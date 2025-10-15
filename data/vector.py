import json
import logging
import os
from typing import Dict, Any, List, Tuple

import faiss
import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm


logger = logging.getLogger(__name__)

# ==============================
# Конфигурация
# ==============================
API_URL = "https://llm.t1v.scibox.tech/v1/embeddings"
API_KEY = "sk-dgKROD7rG4yPAo7bTOtatA"  # замените на ваш ключ
MODEL = "bge-m3"
CLASSIFIER_MODEL = "Qwen2.5-72B-Instruct-AWQ"
BATCH_SIZE = 128
DATA_DIR = "data"
SOURCE_EXCEL = "smart_support_vtb_belarus_faq_final.xlsx"

# Инициализация клиента OpenAI для использования с API Scibox
client = OpenAI(
    api_key=API_KEY,
    base_url="https://llm.t1v.scibox.tech/v1"
)


def _load_dataframe(excel_path: str = SOURCE_EXCEL) -> pd.DataFrame:
    df = pd.read_excel(excel_path)

    required_cols = ['Пример вопроса', 'Шаблонный ответ']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Не найдена колонка '{col}' в Excel-файле")

    df = df.copy()
    df['combined_text'] = df['Пример вопроса'].astype(str) + ". " + df['Шаблонный ответ'].astype(str)
    logger.info("Загружено %s записей для обработки", len(df))
    return df


def get_embeddings_batch(text_batch: List[str]) -> List[List[float]]:
    """
    Возвращает эмбеддинги для батча текстов через API Scibox.
    """
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
        logger.warning("Ошибка API (%s): %s", response.status_code, response.text[:200])
        return []
    result = response.json()
    return [item["embedding"] for item in result.get("data", [])]


def generate_embeddings(texts: List[str], batch_size: int = BATCH_SIZE, show_progress: bool = False) -> List[List[float]]:
    """
    Запрашивает эмбеддинги для списка текстов батчами.
    """
    embeddings: List[List[float]] = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Генерация эмбеддингов")

    for i in iterator:
        batch_texts = texts[i:i + batch_size]
        batch_embs = get_embeddings_batch(batch_texts)
        embeddings.extend(batch_embs)

    if not embeddings:
        raise RuntimeError("Не удалось получить эмбеддинги")

    logger.info("Получено %s эмбеддингов", len(embeddings))
    return embeddings


def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatIP:
    vectors = np.array(embeddings).astype('float32')
    if vectors.ndim != 2:
        raise ValueError("Ожидается двумерный массив эмбеддингов")

    dimension = vectors.shape[1]
    logger.info("Размерность эмбеддингов: %s", dimension)

    # Нормализуем векторы для корректного косинусного сходства
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    logger.info("Добавлено %s векторов в FAISS индекс", index.ntotal)
    return index


def build_metadata(df: pd.DataFrame, texts: List[str]) -> Dict[str, Any]:
    return {
        'ids': list(range(len(texts))),
        'main_categories': df.get('Основная категория', [''] * len(df)).tolist(),
        'sub_categories': df.get('Подкатегория', [''] * len(df)).tolist(),
        'example_questions': df['Пример вопроса'].tolist(),
        'target_audiences': df.get('Целевая аудитория', [''] * len(df)).tolist(),
        'template_answers': df['Шаблонный ответ'].tolist(),
        'priorities': df.get('Приоритет', ['средний'] * len(df)).tolist(),
        'combined_texts': texts
    }


def save_resources(index: faiss.IndexFlatIP, metadata: Dict[str, Any], data_dir: str = DATA_DIR) -> Tuple[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    index_path = os.path.join(data_dir, "knowledge_base.index")
    metadata_path = os.path.join(data_dir, "knowledge_base_metadata.json")

    faiss.write_index(index, index_path)
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("FAISS индекс сохранен в %s", index_path)
    logger.info("Метаданные сохранены в %s", metadata_path)
    return index_path, metadata_path


def prepare_knowledge_base(excel_path: str = SOURCE_EXCEL, data_dir: str = DATA_DIR, show_progress: bool = False) -> Tuple[Dict[str, Any], faiss.IndexFlatIP]:
    df = _load_dataframe(excel_path)
    texts = df['combined_text'].tolist()
    embeddings = generate_embeddings(texts, show_progress=show_progress)
    index = build_faiss_index(embeddings)
    metadata = build_metadata(df, texts)
    save_resources(index, metadata, data_dir=data_dir)
    return metadata, index


# ==============================
# Функция классификации запроса
# ==============================
def classify_query(query, metadata):
    """
    Классифицирует запрос по основной категории и подкатегории
    с использованием Qwen2.5-72B-Instruct-AWQ
    """
    # Собираем уникальные категории и подкатегории из метаданных
    categories = list(set(metadata['main_categories']))
    sub_categories = list(set(metadata['sub_categories']))

    # Формируем промпт для классификации
    prompt = f"""
    Ты - эксперт по классификации запросов клиентов банка ВТБ (Беларусь).
    Проанализируй следующий запрос и определи его основную категорию и подкатегорию.

    Доступные основные категории: {', '.join(categories)}
    Доступные подкатегории: {', '.join(sub_categories)}

    Запрос клиента: "{query}"

    Верни ответ в строго форматированном JSON:
    {{
        "main_category": "название основной категории",
        "sub_category": "название подкатегории",
        "confidence": "уверенность от 0.0 до 1.0"
    }}

    Если запрос не относится ни к одной из категорий, верни:
    {{
        "main_category": "Техническая поддержка",
        "sub_category": "Проблемы и решения",
        "confidence": 0.8
    }}
    """

    try:
        # Отправляем запрос к модели
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": "Ты - эксперт по классификации запросов клиентов банка."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Низкая температура для более предсказуемой классификации
            max_tokens=300
        )

        # Парсим ответ
        import json as json_module
        result = json_module.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        logger.warning("Ошибка при классификации: %s", e)
        # Fallback на случай некорректного JSON или ошибки API
        return {
            "main_category": "Техническая поддержка",
            "sub_category": "Проблемы и решения",
            "confidence": 0.5
        }


# ==============================
# Функция поиска с ранжированием
# ==============================
def search_and_rank(query, metadata, index, target_audience="все клиенты", top_k=10, rerank_k=3):
    """
    Находит релевантные ответы с учетом категории, приоритета, аудитории и с ранжированием
    """
    # 1. Классифицируем запрос
    classification = classify_query(query, metadata)
    main_category = classification['main_category']
    sub_category = classification['sub_category']
    try:
        confidence = float(classification.get('confidence', 0))
    except (TypeError, ValueError):
        confidence = 0.0
    logger.debug(
        "Классификация: %s -> %s (уверенность: %.2f)",
        main_category,
        sub_category,
        confidence,
    )

    # 2. Создаем эмбеддинг для запроса
    test_data = {"model": MODEL, "input": [query]}
    test_response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },
        data=json.dumps(test_data, ensure_ascii=False).encode('utf-8')
    )

    if test_response.status_code != 200:
        logger.warning("Ошибка при создании эмбеддинга запроса: %s", test_response.status_code)
        return []

    query_embedding = np.array([test_response.json()['data'][0]['embedding']]).astype('float32')
    faiss.normalize_L2(query_embedding)

    # 3. Ищем ближайшие вектора
    distances, indices = index.search(query_embedding, k=top_k)

    # 4. Фильтруем и ранжируем результаты
    candidates = []
    for i, idx in enumerate(indices[0]):
        item_main_cat = metadata['main_categories'][idx]
        item_sub_cat = metadata['sub_categories'][idx]

        # Фильтрация по категории (с возможностью частичного совпадения)
        category_match = (item_main_cat == main_category and item_sub_cat == sub_category)

        # Если категория не совпадает, снижаем вес, но не исключаем полностью
        category_boost = 1.5 if category_match else 0.5

        # Усиление веса для высокоприоритетных вопросов
        priority_boost = 1.3 if metadata['priorities'][idx] == 'высокий' else 1.0

        # Усиление веса для совпадения целевой аудитории
        audience_boost = 1.2 if metadata['target_audiences'][idx] == target_audience else 1.0

        # Рассчитываем финальный вес
        # Используем косинусное сходство (distance в FAISS IP близко к косинусному сходству для нормализованных векторов)
        similarity_score = distances[0][i]
        final_score = similarity_score * category_boost * priority_boost * audience_boost

        candidates.append({
            'score': final_score,
            'idx': idx,
            'distance': distances[0][i],
            'category_match': category_match,
            'priority_boost': priority_boost,
            'audience_boost': audience_boost
        })

    # 5. Сортируем по финальному весу
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # 6. Возвращаем топ-N результатов
    ranked_results = []
    for candidate in candidates[:rerank_k]:
        idx = candidate['idx']
        result = {
            'main_category': metadata['main_categories'][idx],
            'sub_category': metadata['sub_categories'][idx],
            'example_question': metadata['example_questions'][idx],
            'template_answer': metadata['template_answers'][idx],
            'target_audience': metadata['target_audiences'][idx],
            'priority': metadata['priorities'][idx],
            'similarity_score': float(candidate['distance']),
            'final_score': float(candidate['score']),
            'category_match': candidate['category_match']
        }
        ranked_results.append(result)

    return ranked_results


def demo_search(metadata: Dict[str, Any], index: faiss.IndexFlatIP) -> None:
    """
    Помогает вручную протестировать поиск при запуске из CLI.
    """
    logger.info("Выполняем тестовый поиск с категоризацией и ранжированием...")

    test_queries = [
        "Как стать клиентом банка онлайн, если я нахожусь за границей?",
        "Забыл пароль от мобильного приложения, что делать?",
        "Могу ли я оформить кредит, находясь в декретном отпуске?"
    ]

    for test_query in test_queries:
        logger.info("=" * 60)
        logger.info("ТЕСТОВЫЙ ЗАПРОС: '%s'", test_query)

        results = search_and_rank(
            query=test_query,
            metadata=metadata,
            index=index,
            target_audience="новые клиенты" if "стать клиентом" in test_query.lower() else "все клиенты",
            top_k=10,
            rerank_k=3
        )

        if results:
            logger.info("Найдено %s релевантных ответов:", len(results))
            for i, result in enumerate(results, 1):
                logger.info(
                    "%s. [%s → %s] приоритет=%s аудитория=%s (сходство=%.4f финальный вес=%.4f)",
                    i,
                    result['main_category'],
                    result['sub_category'],
                    result['priority'],
                    result['target_audience'],
                    result['similarity_score'],
                    result['final_score'],
                )
        else:
            logger.info("Не найдено подходящих результатов.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kb_metadata, kb_index = prepare_knowledge_base(show_progress=True)
    demo_search(kb_metadata, kb_index)
