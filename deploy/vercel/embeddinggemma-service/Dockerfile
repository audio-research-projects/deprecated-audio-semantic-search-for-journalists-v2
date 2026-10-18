# EmbeddingGemma 2 query-side inference service (Cloud Run / Modal / VM).
# Model: google/embeddinggemma-2 (Apache 2.0) via sentence-transformers>=6.1.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    EMBEDDINGGEMMA_OUTPUT_DIMENSIONALITY=768

WORKDIR /app

RUN pip install --no-cache-dir \
    "sentence-transformers[audio]>=6.1" \
    "transformers>=4.57" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "httpx>=0.27" \
    "soundfile>=0.12" \
    "torchcodec>=0.5" \
    "torch>=2.2" \
    "numpy>=1.24"

COPY main.py /app/main.py

# Cloud Run listens on $PORT (default 8080).
EXPOSE 8080
CMD ["python", "main.py"]
