import json
import logging
import re
from typing import Tuple

from .config import CLASSIFIER_MODEL, client

logger = logging.getLogger(__name__)


def _parse_llm_json(raw_content: str) -> dict:
    """Parse JSON returned by the LLM, tolerating minor format deviations."""
    if raw_content is None:
        raise ValueError("Empty response from LLM")

    content = raw_content.strip()
    if not content:
        raise ValueError("Empty response from LLM")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            normalized = snippet.replace("'", '"')
            try:
                return json.loads(normalized)
            except json.JSONDecodeError:
                logger.debug("Failed to parse snippet as JSON: %s", snippet)

    if content.count('"') == 0 and content.count("'") >= 2:
        try:
            return json.loads(content.replace("'", '"'))
        except json.JSONDecodeError:
            pass

    logger.debug("LLM response that failed to parse: %s", content)
    raise ValueError("LLM returned non-JSON content")


def classify_query(query: str, metadata) -> dict:
    """Classify the client query into main and sub categories via LLM."""
    categories = list({cat for cat in metadata["main_categories"] if cat})
    sub_categories = list({sub for sub in metadata["sub_categories"] if sub})

    prompt = f"""
Ты — эксперт контакт-центра банка. Подбери основную и дополнительную категорию из списка
и оцени уверенность модели (0.0–1.0) для вопроса клиента.

Основные категории: {", ".join(categories)}
Подкатегории: {", ".join(sub_categories)}

Вопрос клиента: "{query}"

Ответь строго JSON-объектом:
{{
  "main_category": "одна из перечисленных категорий или \\"Другое\\"",
  "sub_category": "одна из перечисленных подкатегорий или \\"Общее\\"",
  "confidence": число от 0.0 до 1.0
}}

Если подходящей категории нет, верни:
{{
  "main_category": "Другое",
  "sub_category": "Общее",
  "confidence": 0.5
}}
""".strip()

    try:
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Ты классифицируешь обращения клиентов банка.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )

        result = _parse_llm_json(response.choices[0].message.content)
        return result
    except Exception as exc:
        logger.warning("Classification via LLM failed: %s", exc)
        return {
            "main_category": "Другое",
            "sub_category": "Общее",
            "confidence": 0.5,
        }


def verify_candidate_with_llm(
    query: str,
    answer: str,
    refinement_temperature: float = 0.2,
    relevance_threshold: float = 0.6,
) -> Tuple[bool, str, float]:
    """
    Проверяет, подходит ли шаблонный ответ к вопросу. При необходимости улучшает формулировку.
    Возвращает кортеж (is_relevant, refined_answer, confidence).
    """
    prompt = f"""
Ты — контролёр качества FAQ банка. Определи, подходит ли шаблонный ответ к вопросу клиента.
Если ответ корректен, улучши формулировку, сохранив факты. Ответь строго JSON-объектом.

Вопрос: "{query}"
Шаблонный ответ: "{answer}"

Формат:
{{
  "relevant": true или false,
  "refined_answer": "улучшенный ответ или пустая строка",
  "confidence": число от 0 до 1
}}
""".strip()

    try:
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Ты проверяешь качество FAQ банка. Отвечай строго JSON-объектом.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=refinement_temperature,
            max_tokens=300,
        )

        verdict = _parse_llm_json(response.choices[0].message.content)

        is_relevant = bool(verdict.get("relevant"))
        confidence = float(verdict.get("confidence", 0.0))
        refined_answer = verdict.get("refined_answer") or answer

        if not is_relevant:
            return False, answer, confidence

        if confidence < relevance_threshold:
            return False, answer, confidence

        return True, refined_answer, confidence
    except Exception as exc:
        logger.warning("LLM verification failed, using original answer: %s", exc)
        return False, answer, 0.0


__all__ = ["classify_query", "verify_candidate_with_llm"]
