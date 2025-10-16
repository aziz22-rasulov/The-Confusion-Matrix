import difflib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

# Предполагается, что search_and_rank определен и умеет принимать exclude_idx
from data import search_and_rank 

logger = logging.getLogger(__name__)


def _load_metadata(metadata: Dict[str, Any] = None):
    """Если metadata не переданы, попробуем загрузить их из data/data/knowledge_base_metadata.json"""
    if metadata:
        return metadata
    base = os.getcwd()
    metadata_path = os.path.join(base, 'data', 'data', 'knowledge_base_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata not found: {metadata_path}")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _score_match(query: str, candidate: str) -> float:
    """Простая функция оценки совпадения между строками (0..1)"""
    if not candidate:
        return 0.0
    # Используем сочетание SequenceMatcher и простого пересечения токенов
    s = difflib.SequenceMatcher(None, query.lower(), candidate.lower()).ratio()
    q_tokens = set(re.findall(r"\w+", query.lower()))
    c_tokens = set(re.findall(r"\w+", candidate.lower()))
    if not q_tokens or not c_tokens:
        token_score = 0.0
    else:
        token_score = len(q_tokens & c_tokens) / max(len(q_tokens | c_tokens), 1)
    # Взвешиваем: 0.7*seq + 0.3*tokens
    return 0.7 * s + 0.3 * token_score


def validate_and_process_query(
    query: str, 
    metadata: Dict[str, Any] = None, 
    index=None, 
    target_audience: str = "все клиенты", 
    top_k: int = 5, 
    rerank_k: int = 3,
    exclude_idx: Optional[List[int]] = None # <-- НОВЫЙ АРГУМЕНТ
):
    """
    Проверяет вопрос на корректность и ищет наиболее подходящие ответы в метаданных.
    Поддерживает исключение индексов для повторного поиска.

    Аргументы:
        query: текст запроса
        metadata: словарь с полями
        index: опционально — faiss индекс
        target_audience: фильтр по аудитории
        top_k: сколько кандидатов вернуть для ранжирования
        rerank_k: сколько элементов вернуть в ответе
        exclude_idx: список индексов (int) ответов, которые нужно исключить из поиска.
    
    Возвращает:
        dict с ключом 'status' и 'results' — списком рекомендованных ответов
    """
    if not query or len(query.strip()) < 3:
        raise ValueError("Вопрос слишком короткий или отсутствует. Пожалуйста, сформулируйте более подробный запрос.")

    clean_query = re.sub(r'[^\w\s]', '', query).strip()
    if not clean_query:
        raise ValueError("Вопрос содержит только специальные символы или недопустимые символы. Пожалуйста, сформулируйте корректный запрос.")

    if len(query) > 1000:
        raise ValueError("Вопрос слишком длинный. Пожалуйста, сформулируйте запрос короче.")

    metadata = _load_metadata(metadata)
    
    if exclude_idx is None:
        exclude_idx = []

    vector_results = []
    if index is not None:
        try:
            # ПЕРЕДАЧА ИНДЕКСОВ ДЛЯ ИСКЛЮЧЕНИЯ В search_and_rank
            # Ищем больше, чтобы компенсировать исключенные
            vector_payload = search_and_rank(
                query=query,
                metadata=metadata,
                index=index,
                target_audience=target_audience,
                top_k=top_k + len(exclude_idx),
                rerank_k=rerank_k,
                exclude_idx=exclude_idx if exclude_idx else None,
            )
            vector_results = vector_payload.get('results', []) if isinstance(vector_payload, dict) else vector_payload
            classification = vector_payload.get('classification', {}) if isinstance(vector_payload, dict) else {}
            strategy = vector_payload.get('strategy', 'vector') if isinstance(vector_payload, dict) else 'vector'
        
        except Exception as vector_error:
            logger.warning("Vector search failed, fallback to fuzzy matching: %s", vector_error, exc_info=True)
            vector_results = []
            classification = {}
            strategy = 'fuzzy'

    if vector_results:
        return {
            'status': 'success',
            'query': query,
            'classification': classification,
            'strategy': strategy,
            'results': vector_results
        }

    # Логика фаззи-поиска:
    
    example_questions: List[str] = metadata.get('example_questions', [])
    combined_texts: List[str] = metadata.get('combined_texts', [])
    template_answers: List[str] = metadata.get('template_answers', [])
    main_categories = metadata.get('main_categories', [])
    sub_categories = metadata.get('sub_categories', [])
    target_audiences = metadata.get('target_audiences', [])
    priorities = metadata.get('priorities', [])

    candidates = []

    # Оцениваем по примеру вопроса и по combined_texts
    for idx, (ex_q, comb) in enumerate(zip(example_questions, combined_texts)):
        
        # ИСКЛЮЧЕНИЕ КАНДИДАТОВ В ФАЗЗИ-ПОИСКЕ
        if idx in exclude_idx:
            continue
        # <---

        score_q = _score_match(query, ex_q or '')
        score_c = _score_match(query, comb or '')
        score = max(score_q, score_c)

        # Бонусы по приоритету / аудитории
        priority = priorities[idx] if idx < len(priorities) else None
        audience = target_audiences[idx] if idx < len(target_audiences) else None
        if priority and isinstance(priority, str) and priority.lower() == 'высокий':
            score *= 1.1
        if audience and target_audience and audience == target_audience:
            score *= 1.05

        candidates.append((score, idx))

    # Сортируем и берём топ_k
    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:top_k]

    results = []
    for score, idx in top[:rerank_k]:
        ans = {
            'idx': int(idx),
            'similarity_score': float(round(score, 4)),
            'example_question': example_questions[idx] if idx < len(example_questions) else None,
            'template_answer': template_answers[idx] if idx < len(template_answers) else None,
            'main_category': main_categories[idx] if idx < len(main_categories) else None,
            'sub_category': sub_categories[idx] if idx < len(sub_categories) else None,
            'priority': priorities[idx] if idx < len(priorities) else None,
            'target_audience': target_audiences[idx] if idx < len(target_audiences) else None,
        }
        results.append(ans)

    return {
        'status': 'success',
        'query': query,
        'results': results
    }
