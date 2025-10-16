import json
import logging
from typing import Tuple

# Предполагается, что эти импорты определены в вашем проекте
from .config import CLASSIFIER_MODEL, client

logger = logging.getLogger(__name__)


def classify_query(query, metadata):
    """
    Классифицирует запрос по основной категории и подкатегории
    с использованием Qwen2.5-72B-Instruct-AWQ
    """
    categories = list({cat for cat in metadata['main_categories'] if cat})
    sub_categories = list({sub for sub in metadata['sub_categories'] if sub})

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
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": "Ты - эксперт по классификации запросов клиентов банка."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as exc:
        logger.warning("Ошибка при классификации: %s", exc)
        return {
            "main_category": "Техническая поддержка",
            "sub_category": "Проблемы и решения",
            "confidence": 0.5
        }


def verify_candidate_with_llm(query: str, answer: str, refinement_temperature: float = 0.2, relevance_threshold: float = 0.6) -> Tuple[bool, str, float]:
    """
    Проверяет релевантность кандидатного ответа и при необходимости переформулирует его.
    
    ВАЖНО: Если ответ релевантен, LLM должен переформулировать его, начиная с 
    фразы, которая прямо отвечает на вопрос клиента, а затем продолжаться фактами.

    Возвращает (is_relevant, refined_answer, confidence). В случае ошибки ЛЛМ считаем ответ релевантным и возвращаем исходный текст.
    """
    # ОБНОВЛЕННЫЙ ПРОМПТ
    prompt = f"""
Ты — помощник, который проверяет ответ базы знаний и адаптирует его под конкретный вопрос клиента.
Проанализируй вопрос и ответ.

Если ответ **релевантен**, ты должен **переформулировать** его, чтобы он **начинался с фразы-реакции, которая прямо отвечает на вопрос** клиента, а затем продолжался фактологической информацией из базы знаний.
Сохраняй все факты и цифры из исходного ответа.

Пример желаемой формулировки:
Вопрос: "Как взять кредит в декрете?"
Ответ базы знаний: "Гражданам, находящимся в отпуске по уходу за ребенком, кредиты не выдаются."
Твой refined_answer: "Взять кредит в декрете не получится. Это связано с тем, что..." (далее факты из исходного ответа).

Вопрос: "{query}"
Ответ базы знаний: "{answer}"

Ответь строго в формате JSON:
{{
  "relevant": true или false,
  "refined_answer": "переформулированный ответ (если релевантен, иначе пустая строка)",
  "confidence": число от 0 до 1
}}
"""

    try:
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": "Ты помощник, проверяющий релевантность ответов FAQ банка. Говоришь по-русски."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=refinement_temperature,
            max_tokens=300 # Увеличен для лучшей переформулировки
        )

        verdict = json.loads(response.choices[0].message.content.strip())

        is_relevant = bool(verdict.get("relevant"))
        confidence = float(verdict.get("confidence", 0.0))
        refined_answer = verdict.get("refined_answer") or answer

        if not is_relevant:
            return False, answer, confidence

        if confidence < relevance_threshold:
            return True, answer, confidence

        return True, refined_answer, confidence
    except Exception as exc:
        logger.warning("LLM verification failed, using original answer: %s", exc)
        return True, answer, 0.0


__all__ = ["classify_query", "verify_candidate_with_llm"]
