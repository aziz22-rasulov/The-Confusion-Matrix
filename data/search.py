import json
import logging
from typing import Any, Dict, List

import faiss
import numpy as np
import requests

from .classifier import classify_query, verify_candidate_with_llm
from .config import API_KEY, API_URL, MODEL
from .embeddings import prepare_knowledge_base

logger = logging.getLogger(__name__)


def search_and_rank(query, metadata, index, target_audience="все клиенты", top_k=10, rerank_k=3):
    """
    Находит релевантные ответы с учетом категории, приоритета, аудитории и с ранжированием.
    """
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

    distances, indices = index.search(query_embedding, k=top_k)

    candidates = []
    for i, idx in enumerate(indices[0]):
        item_main_cat = metadata['main_categories'][idx]
        item_sub_cat = metadata['sub_categories'][idx]

        category_match = (item_main_cat == main_category and item_sub_cat == sub_category)
        category_boost = 1.5 if category_match else 0.5
        priority_boost = 1.3 if metadata['priorities'][idx] == 'высокий' else 1.0
        audience_boost = 1.2 if metadata['target_audiences'][idx] == target_audience else 1.0

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

    candidates.sort(key=lambda x: x['score'], reverse=True)

    ranked_results = []
    for candidate in candidates[:top_k]:
        idx = candidate['idx']
        raw_answer = metadata['template_answers'][idx]

        is_relevant, refined_answer = verify_candidate_with_llm(query, raw_answer)
        if not is_relevant:
            continue

        result = {
            'main_category': metadata['main_categories'][idx],
            'sub_category': metadata['sub_categories'][idx],
            'example_question': metadata['example_questions'][idx],
            'template_answer': refined_answer,
            'target_audience': metadata['target_audiences'][idx],
            'priority': metadata['priorities'][idx],
            'similarity_score': float(candidate['distance']),
            'final_score': float(candidate['score']),
            'category_match': candidate['category_match']
        }
        ranked_results.append(result)

        if len(ranked_results) >= rerank_k:
            break

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
