FROM python:3.11-slim

WORKDIR /app

# system deps for pandas, faiss and other packages
RUN apt-get update --fix-missing && \
    apt-get install -y --fix-missing build-essential git curl libsndfile1 libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# copy app
COPY backend /app/backend
COPY data /app/data

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
