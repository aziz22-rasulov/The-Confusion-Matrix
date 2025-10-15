from backend.main import app
from backend.models.models import dialog

from fastapi import  APIRouter

app = APIRouter()

app.post('/dialog')
async def dialog(text:dialog):
    '''
    ручка для получения от пользователя текста.
    '''
