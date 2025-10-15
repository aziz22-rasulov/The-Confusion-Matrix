from backend.models.models import Dialog
from fastapi import APIRouter
from fastapi import HTTPException

from backend.support.support import validate_and_process_query

import os
import json

import faiss 

app = APIRouter()

_metadata = None
_index = None


def _load_metadata_and_index():
    global _metadata, _index
    if _metadata is not None and _index is not None:
        return _metadata, _index
    
    base = os.getcwd()
    metadata_path = os.path.join(base, 'data', 'data', 'knowledge_base_metadata.json')
    index_path = os.path.join(base, 'data', 'data', 'knowledge_base.index')

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata not found: {metadata_path}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"faiss index not found: {index_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        _metadata = json.load(f)

    if faiss is None:
        raise RuntimeError("faiss library is not available in the environment")

    _index = faiss.read_index(index_path)
    return _metadata, _index


@app.post('/dialog')
async def dialog(text: Dialog):
    '''
    Ручка для получения от пользователя текста. Проверяет и обрабатывает запрос через support.validate_and_process_query
    '''
    try:
        metadata, index = _load_metadata_and_index()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        result = validate_and_process_query(text.text, metadata, index)
        return result
    except ValueError as e:
        # Ошибки валидации — клиентская ошибка
        raise HTTPException(status_code=400, detail=str(e))
    except NameError as e:
        # Возможна ситуация, когда support.validate_and_process_query ссылается на отсутствующие функции
        raise HTTPException(status_code=500, detail=f"Не завершена конфигурация поиска: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

