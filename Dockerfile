FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY demo ./demo
COPY scripts ./scripts
COPY .env.example ./.env.example

RUN python -m scripts.ingest_documents \
    && python -m scripts.create_demo_versions \
    && python -m scripts.import_eval_dataset

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
