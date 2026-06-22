# Dockerfile CPU multi-stage para ALPR System
FROM python:3.12-slim AS builder

WORKDIR /build

# Dependencias del sistema para compilar wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY alpr ./alpr
COPY README.md ./

# Instalar dependencias en un directorio para copiar después
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e "."

# Stage final
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY alpr ./alpr
COPY models ./models
COPY pyproject.toml ./

ENV ALPR_MODELS_DIR=/app/models
ENV ALPR_VEHICLE_MODEL_PATH=/app/models/yolov8n.pt
ENV ALPR_PLATE_MODEL_PATH=/app/models/license_plate_detector.pt
ENV ALPR_LOG_LEVEL=INFO
ENV ALPR_JSON_LOGS=true
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "alpr.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
