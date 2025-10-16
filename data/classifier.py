import json
import logging
from typing import Tuple

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


# Файл: llm_utils.py

# ... (Остальная часть файла остается без изменений, включая classify_query)

def verify_candidate_with_llm(query: str, answer: str, refinement_temperature: float = 0.2, relevance_threshold: float = 0.6) -> Tuple[bool, str]:
    """
    Проверяет релевантность кандидатного ответа и при необходимости переформулирует его.
    
    ВАЖНО: Если ответ релевантен, LLM должен переформулировать его, начиная с 
    фразы, которая прямо отвечает на вопрос клиента, а затем продолжаться фактами.

    Возвращает (is_relevant, refined_answer). В случае ошибки ЛЛМ считаем ответ релевантным и возвращаем исходный текст.
    """
    # *** СИЛЬНО ИЗМЕНЕННЫЙ ПРОМПТ ДЛЯ ПРИНУДИТЕЛЬНОЙ ПЕРЕФОРМУЛИРОВКИ ***
    prompt = f"""
Ты — помощник, который проверяет ответ базы знаний и адаптирует его под конкретный вопрос клиента.
Твоя задача:
1. Определить, релевантен ли ответ (Ответ базы знаний) запросу (Вопрос).
2. Если релевантен, **переформулировать** ответ таким образом, чтобы он **начинался с прямого ответа на Вопрос**, сохраняя при этом все факты и цифры из исходного текста.

Пример желаемого результата:
Вопрос: "Как взять кредит в декрете?"
Ответ базы знаний: "По условиям кредитных программ, заявитель не должен находиться в социальном отпуске по уходу за ребенком до достижения им возраста 3-х лет."
Твой refined_answer должен выглядеть так: "К сожалению, взять кредит, находясь в декретном отпуске, невозможно. По условиям кредитных программ, заявитель не должен находиться в социальном отпуске по уходу за ребенком до достижения им возраста 3-х лет."

Вопрос: "{query}"
Ответ базы знаний: "{answer}"

Ответь **СТРОГО** в формате JSON:
{{
  "relevant": true или false,
  "refined_answer": "переформулированный ответ, начинающийся с прямого ответа (если релевантен, иначе пустая строка)",
  "confidence": число от 0 до 1
}}
"""

    try:
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                # Используем более строгую системную инструкцию
                {"role": "system", "content": "Ты - строгий и точный помощник, который адаптирует ответы базы знаний под вопросы клиента, начиная ответ с прямой фразы-реакции."},
                {"role": "user", "content": prompt.strip()}
            ],
            temperature=refinement_temperature,
            max_tokens=300 
        )

        verdict = json.loads(response.choices[0].message.content.strip())

        is_relevant = bool(verdict.get("relevant"))
        confidence = float(verdict.get("confidence", 0.0))
        # Используем refined_answer, а если он пуст - исходный answer
        refined_answer = verdict.get("refined_answer") or answer 

        if not is_relevant or confidence < relevance_threshold:
            return False, answer

        return True, refined_answer
    except Exception as exc:
        logger.warning("LLM verification failed, using original answer: %s", exc)
        return True, answer


# ... (Остальная часть файла остается без изменений)


__all__ = ["classify_query", "verify_candidate_with_llm"]
