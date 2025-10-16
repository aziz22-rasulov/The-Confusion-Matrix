import difflib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

# Предполагается, что search_and_rank определен и умеет принимать exclude_idx
from data import search_and_rank, verify_candidate_with_llm

logger = logging.getLogger(__name__)

MIN_CLASSIFIER_CONFIDENCE = 0.55
MIN_FUZZY_SIMILARITY = 0.45
MIN_LLM_CONFIDENCE = 0.6


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




def _build_category_path(main_category: Optional[str], sub_category: Optional[str]) -> Optional[str]:
    parts = [part for part in (main_category, sub_category) if part]
    if not parts:
        return None
    return " > ".join(parts)


def _format_response(
    query: str,
    items: List[Dict[str, Any]],
    classification: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[str]]:
    top_item = items[0] if items else None
    classifier_confidence: Optional[float] = None
    if classification:
        try:
            classifier_confidence = float(classification.get('confidence', 0.0))
        except (TypeError, ValueError):
            classifier_confidence = None

    classification_payload: Optional[Dict[str, Optional[str]]] = None
    if classification:
        classification_payload = {
            'main_category': classification.get('main_category'),
            'sub_category': classification.get('sub_category'),
            'confidence': classifier_confidence,
        }

    if top_item:
        category_path = _build_category_path(
            top_item.get('main_category'),
            top_item.get('sub_category'),
        )

        if category_path is None and classification:
            category_path = _build_category_path(
                classification.get('main_category'),
                classification.get('sub_category'),
            )

        recommended_answer = top_item.get('template_answer')
        llm_confidence = top_item.get('llm_confidence')

        if recommended_answer and llm_confidence is None:
            raw_answer = top_item.get('raw_answer') or recommended_answer
            try:
                is_relevant, refined_answer, confidence = verify_candidate_with_llm(query, raw_answer)
                if is_relevant:
                    recommended_answer = refined_answer
                    llm_confidence = confidence
                else:
                    recommended_answer = raw_answer
            except Exception as refine_error:
                logger.warning(
                    "Classifier refinement failed, returning original answer: %s",
                    refine_error,
                    exc_info=True,
                )
                recommended_answer = raw_answer

        similarity_score = top_item.get('similarity_score')
        try:
            similarity_score = float(similarity_score)
        except (TypeError, ValueError):
            similarity_score = None

        try:
            llm_confidence = float(llm_confidence) if llm_confidence is not None else None
        except (TypeError, ValueError):
            llm_confidence = None

        should_return_answer = False
        if llm_confidence is not None and llm_confidence >= MIN_LLM_CONFIDENCE:
            should_return_answer = True
        elif similarity_score is not None and similarity_score >= MIN_FUZZY_SIMILARITY:
            if classifier_confidence is None or classifier_confidence >= MIN_CLASSIFIER_CONFIDENCE:
                should_return_answer = True

        if not should_return_answer:
            logger.debug(
                "Suppressing template answer due to low confidence: query='%s', similarity=%s, classifier_conf=%s, llm_conf=%s",
                query,
                similarity_score,
                classifier_confidence,
                llm_confidence,
            )
            recommended_answer = None
            llm_confidence = None

        return {
            'category_path': category_path,
            'recommended_answer': recommended_answer,
            'llm_confidence': llm_confidence,
            'match_main_category': top_item.get('main_category'),
            'match_sub_category': top_item.get('sub_category'),
            'match_similarity': similarity_score,
            'classification': classification_payload,
        }

    fallback_path = None
    if classification:
        fallback_path = _build_category_path(
            classification.get('main_category'),
            classification.get('sub_category'),
        )

    return {
        'category_path': fallback_path,
        'recommended_answer': None,
        'llm_confidence': None,
        'match_main_category': None,
        'match_sub_category': None,
        'match_similarity': None,
        'classification': classification_payload,
    }


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
        dict with keys 'category_path', 'recommended_answer', 'llm_confidence'
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
    classification = None
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
        
        except Exception as vector_error:
            logger.warning("Vector search failed, fallback to fuzzy matching: %s", vector_error, exc_info=True)
            vector_results = []
            classification = {}

    if vector_results:
        return _format_response(query, vector_results, classification)

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

    return _format_response(query, results, classification)
