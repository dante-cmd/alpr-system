# AGENTS.md — ALPR System

## Alcance
Sistema de reconocimiento automático de placas vehiculares (ALPR/ANPR) orientado a producción, optimizado para CPU y placas tipo Mercosur.

## Stack aprobado (Opción A)
- Detección de vehículos: YOLOv8n preentrenado en COCO.
- Detección de placas: YOLOv8n/s (fine-tuning recomendado en UFPR-ALPR/RodoSol).
- OCR: PaddleOCR PP-OCRv4 (modelo inglés/latino).
- Serving: FastAPI + Uvicorn.
- Observabilidad: logs JSON (`structlog`), métricas Prometheus, trazas OpenTelemetry.
- Contenerización: Docker CPU (multi-stage).
- CI/CD: GitHub Actions.

## Restricciones
- No introducir componentes experimentales sin evidencia documentada (benchmark, paper, uso industrial).
- No subir datasets con acuerdo de licencia ni modelos pesados (>50 MB) al repositorio; usar scripts de descarga.
- Mantener latencia p95 ≤ 300 ms en CPU para entrada 720p.

## Convenciones de código
- Python 3.10+ con type hints.
- Formato con `ruff`, tipado con `mypy`.
- Configuración vía variables de entorno (`pydantic-settings`).
- Logs estructurados en JSON en producción.

## Estructura
- `alpr/core/` — pipeline de inferencia.
- `alpr/api/` — FastAPI.
- `alpr/training/` — scripts de fine-tuning.
- `alpr/eval/` — métricas y evaluación robusta.
- `alpr/docker/` — Dockerfiles.
- `tests/` — tests unitarios, integración y estrés.

## Cómo ejecutar localmente
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m alpr.api.main
```

## Tests
```bash
pytest
```

## Docker
```bash
docker compose -f alpr/docker/docker-compose.yml up --build
```
