FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY smart_factory_anomaly_explainer.py ./
COPY data ./data

ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=llama3

CMD ["python", "smart_factory_anomaly_explainer.py"]
