# ALPR System — Reconocimiento de Placas Mercosur

Sistema de reconocimiento automático de placas vehiculares (ALPR/ANPR) orientado a producción, con pipeline basado en **YOLOv8** y **PaddleOCR PP-OCRv4**, optimizado para **CPU**.

## Características

- Detección de vehículos con YOLOv8n (COCO).
- Detección de placas con detector YOLOv8 fine-tuned (modelo base incluido).
- OCR robusto con PaddleOCR PP-OCRv4.
- Validación sintáctica de formatos Mercosur.
- API REST con FastAPI, métricas Prometheus y logs estructurados.
- Docker reproducible y CI/CD con GitHub Actions.
- Scripts de evaluación end-to-end y métricas de producción.

## Requisitos

- Python ≥ 3.10
- Docker (opcional)
- ~4 GB de RAM libres para inferencia CPU

## Instalación local

```bash
python -m venv .venv
source .venv/bin/activate

# Instalar primero PyTorch CPU y PaddleOCR compatible
pip install torch==2.12.0+cpu torchvision==0.27.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
pip install "paddlepaddle>=2.6.0,<4.0.0" "paddleocr>=2.9.0,<3.0.0"

# Instalar el proyecto
pip install -e ".[dev]"
```

> **Nota sobre PaddleOCR:** `paddleocr>=3.x` arrastra `paddlex` y modelos `PP-OCRv6` que, en el entorno CPU-only probado, lanzan un error de OneDNN (`ConvertPirAttribute2RuntimeAttribute not support [...]`). El repositorio fija `paddleocr<3.0.0` para usar PP-OCRv3/v4 estables. Ver [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) para más detalles.

## Uso

### API REST + UI web

```bash
python -m alpr.api.main
```

Abre en el navegador:

```
http://localhost:8000
```

La interfaz web incluye pestañas para subir imágenes, procesar batch, controlar streams de video en vivo y ver logs.

Endpoints:

- `GET /health` — estado del servicio.
- `GET /metrics` — métricas Prometheus.
- `POST /detect` — detecta placas en una imagen.
- `POST /batch/detect` — procesa múltiples imágenes.
- `POST /stream/start` — inicia stream de webcam/cámara IP/archivo.
- `GET /stream/frame` — último frame anotado (JPEG).
- `GET /stream/plates` — placas detectadas recientemente.

Ejemplo:

```bash
curl -X POST -F "file=@data/samples/auto.jpg" http://localhost:8000/detect
```

### Streaming de video

```bash
# Webcam local
curl -X POST http://localhost:8000/stream/start \
  -H "Content-Type: application/json" \
  -d '{"source": 0, "process_every_n_frames": 3, "max_fps": 2}'

# Cámara IP Hikvision por RTSP
curl -X POST http://localhost:8000/stream/start \
  -H "Content-Type: application/json" \
  -d '{
    "source": "rtsp://usuario:pass@192.168.1.100:554/Streaming/Channels/101",
    "process_every_n_frames": 3,
    "max_fps": 2,
    "roi": [200, 300, 1400, 900]
  }'

curl http://localhost:8000/stream/frame -o frame.jpg
curl http://localhost:8000/stream/plates
curl -X POST http://localhost:8000/stream/stop
```

Ver [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) para recomendación de cámaras y optimización de streams.

### Docker

```bash
docker compose -f alpr/docker/docker-compose.yml up --build
```

Con observabilidad (Prometheus + Grafana):

```bash
docker compose -f alpr/docker/docker-compose.yml --profile observability up --build
```

### Evaluación

```bash
python -m alpr.eval.evaluate \
  --images data/datasets/test \
  --annotations data/datasets/test_annotations.json \
  --output outputs/eval_results.json
```

### Entrenamiento / Fine-tuning

Para fine-tuning con datasets propios (UFPR-ALPR, RodoSol-ALPR):

```bash
python -m alpr.training.train_yolo --data data/datasets/plates.yaml --epochs 50
```

> Nota: el entrenamiento requiere GPU. Los scripts están listos para ejecutarse en un entorno con CUDA.

## Métricas objetivo

| Métrica | Objetivo |
|---------|----------|
| Exact Match (día) | ≥ 95 % |
| Recall detección | ≥ 98 % |
| Latencia p95 CPU | ≤ 300 ms |

## Estructura del proyecto

```text
alpr/
  api/          # FastAPI
  core/         # Pipeline de inferencia
  training/     # Scripts de entrenamiento
  eval/         # Métricas y evaluación
  docker/       # Dockerfiles y compose
tests/          # Tests unitarios, integración y estrés
models/         # Modelos descargados (no versionados)
data/           # Datasets y muestras
```

## Documentación adicional

- [`ABOUT.md`](ABOUT.md) — información general, arquitectura, alcance y roadmap.
- [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) — requerimientos, datasets y decisiones.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — guía de despliegue, variables de entorno y troubleshooting.
- [`docs/UVP.md`](docs/UVP.md) — Utility Validation Protocol de componentes.

## Licencia

MIT — ver [`LICENSE`](LICENSE).


Ver video para instalar 

https://www.youtube.com/watch?v=wQ7wTtUL1BY