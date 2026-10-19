# Optional container deployment of the same text-only inference service.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    EMBEDDINGGEMMA_MODEL_PATH=/app/model \
    EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY=768 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py prepare_model.py /app/
RUN python prepare_model.py

EXPOSE 8080
CMD ["python", "main.py"]
