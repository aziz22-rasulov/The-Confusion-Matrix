from fastapi import FastAPI
from backend.point.dialog_user import app as point

app = FastAPI()

app.include_router(point)