import json
import logging
import os
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .config import API_KEY, API_URL, BATCH_SIZE, DATA_DIR, MODEL, SOURCE_EXCEL

logger = logging.getLogger(__name__)


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


__all__ = [
    "get_embeddings_batch",
    "generate_embeddings",
    "build_faiss_index",
    "build_metadata",
    "save_resources",
    "prepare_knowledge_base",
]
