import os

from openai import OpenAI

# ==============================
# Конфигурация API и путей
# ==============================
API_URL = "https://llm.t1v.scibox.tech/v1/embeddings"
API_KEY = os.getenv("SCIBOX_API_KEY", "sk-dgKROD7rG4yPAo7bTOtatA")  # замените на ваш ключ
MODEL = "bge-m3"
CLASSIFIER_MODEL = "Qwen2.5-72B-Instruct-AWQ"
BATCH_SIZE = 128
DATA_DIR = "data"
SOURCE_EXCEL = "smart_support_vtb_belarus_faq_final.xlsx"


# Единый клиент OpenAI для использования с API Scibox
client = OpenAI(
    api_key=API_KEY,
    base_url="https://llm.t1v.scibox.tech/v1"
)


__all__ = [
    "API_URL",
    "API_KEY",
    "MODEL",
    "CLASSIFIER_MODEL",
    "BATCH_SIZE",
    "DATA_DIR",
    "SOURCE_EXCEL",
    "client",
]
