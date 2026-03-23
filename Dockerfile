FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/app/data/.cache \
    HF_HOME=/app/data/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/data/.cache/huggingface/hub \
    SENTENCE_TRANSFORMERS_HOME=/app/data/.cache/sentence-transformers

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.runtime.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.runtime.txt

COPY backend ./backend
COPY frontend ./frontend
COPY data/faq.json ./data/faq.json
COPY data/service_areas.json ./data/service_areas.json
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
