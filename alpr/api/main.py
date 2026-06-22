"""FastAPI application para ALPR."""

import logging
import threading
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import FileResponse, PlainTextResponse, Response

from alpr import __version__
from alpr.api.schemas import (
    DetectionResponse,
    HealthResponse,
    PlateResult,
    StreamEventResponse,
    StreamStartRequest,
    StreamStatusResponse,
)
from alpr.core.config import settings
from alpr.core.pipeline import ALPRPipeline
from alpr.core.stream_processor import StreamConfig, StreamProcessor

# Métricas Prometheus
# Se crean solo cuando el módulo se importa (no como __main__), para evitar
# doble registro cuando uvicorn vuelve a importar la aplicación.
if __name__ != "__main__":
    REQUEST_COUNT = Counter(
        "alpr_requests_total", "Total de peticiones", ["endpoint", "status"]
    )
    REQUEST_LATENCY = Histogram(
        "alpr_request_duration_seconds", "Latencia de peticiones", ["endpoint"]
    )
    PLATES_DETECTED = Counter("alpr_plates_detected_total", "Placas detectadas")
else:
    REQUEST_COUNT = None  # type: ignore[assignment]
    REQUEST_LATENCY = None  # type: ignore[assignment]
    PLATES_DETECTED = None  # type: ignore[assignment]

logger = structlog.get_logger()


class PipelineStore:
    """Singleton para mantener el pipeline cargado en memoria."""

    def __init__(self) -> None:
        self.pipeline: ALPRPipeline | None = None

    def load(self) -> ALPRPipeline:
        if self.pipeline is None:
            self.pipeline = ALPRPipeline(
                detect_vehicle=settings.detect_vehicle_first,
                device="cpu",
            )
        return self.pipeline


store = PipelineStore()

# Stream state (protegido por lock para evitar carreras entre endpoints)
_stream_lock = threading.Lock()
_stream_processor: StreamProcessor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            (
                structlog.processors.JSONRenderer()
                if settings.json_logs
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logger.info("alpr_api_startup", version=__version__)
    pipeline = store.load()
    # Warm-up OCR para descargar modelos PaddleOCR en el arranque y evitar
    # timeout en la primera petición.
    dummy = np.full((64, 200, 3), 255, dtype=np.uint8)
    pipeline.ocr.read(dummy)
    logger.info("alpr_api_warmup_complete")
    yield
    # Shutdown
    logger.info("alpr_api_shutdown")


app = FastAPI(
    title="ALPR API",
    description="Sistema de reconocimiento automático de placas (Mercosur).",
    version=__version__,
    lifespan=lifespan,
)

# Servir la interfaz web estática
app.mount("/static", StaticFiles(directory=str(Path(__file__).with_name("static"))), name="static")


@app.get("/")
async def root() -> FileResponse:
    """Sirve la interfaz web principal."""
    return FileResponse(str(Path(__file__).with_name("static") / "index.html"))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.get("/metrics")
async def metrics() -> Response:
    return PlainTextResponse(content=generate_latest().decode("utf-8"))


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)) -> DetectionResponse:
    start = time.perf_counter()
    endpoint = "/detect"
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Se requiere una imagen")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

        result = store.load().run(image)
        elapsed_ms = (time.perf_counter() - start) * 1000

        PLATES_DETECTED.inc(result["count"])
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed_ms / 1000.0)
        REQUEST_COUNT.labels(endpoint=endpoint, status="200").inc()

        return DetectionResponse(
            success=result["success"],
            plates=[PlateResult(**p) for p in result["plates"]],
            count=result["count"],
            processing_time_ms=round(elapsed_ms, 2),
        )
    except HTTPException:
        REQUEST_COUNT.labels(endpoint=endpoint, status="400/500").inc()
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, status="500").inc()
        logger.error("detect_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/batch/detect")
async def batch_detect(files: list[UploadFile]) -> dict[str, Any]:
    """Procesa múltiples imágenes secuencialmente (batch sincrónico)."""
    results = []
    for f in files:
        try:
            resp = await detect(f)
            results.append({"filename": f.filename, "result": resp.model_dump()})
        except Exception as exc:
            results.append({"filename": f.filename, "error": str(exc)})
    return {"results": results}


@app.post("/stream/start", response_model=StreamStatusResponse)
async def stream_start(req: StreamStartRequest) -> StreamStatusResponse:
    """Inicia la captura y procesamiento de video desde una fuente."""
    global _stream_processor
    with _stream_lock:
        if _stream_processor is not None and _stream_processor.is_running:
            _stream_processor.stop()

        config = StreamConfig(
            source=req.source,
            process_every_n_frames=req.process_every_n_frames,
            max_fps=req.max_fps,
            roi=req.roi,
            min_confidence=req.min_confidence,
            detect_vehicle=req.detect_vehicle,
        )
        _stream_processor = StreamProcessor(config)
        if not _stream_processor.start():
            raise HTTPException(
                status_code=503,
                detail=f"No se pudo abrir la fuente de video: {req.source}",
            )
        info = _stream_processor.info()

    REQUEST_COUNT.labels(endpoint="/stream/start", status="200").inc()
    return StreamStatusResponse(
        running=info["running"],
        source=str(config.source),
        source_info=info["source_info"],
        config=info["config"],
        events_count=info["events_count"],
    )


@app.post("/stream/stop")
async def stream_stop() -> dict[str, bool]:
    """Detiene el stream de video."""
    global _stream_processor
    with _stream_lock:
        if _stream_processor is not None:
            _stream_processor.stop()
            _stream_processor = None
    REQUEST_COUNT.labels(endpoint="/stream/stop", status="200").inc()
    return {"stopped": True}


@app.get("/stream/status", response_model=StreamStatusResponse)
async def stream_status() -> StreamStatusResponse:
    """Devuelve el estado del stream activo."""
    with _stream_lock:
        if _stream_processor is None:
            raise HTTPException(status_code=404, detail="No hay stream activo")
        info = _stream_processor.info()

    return StreamStatusResponse(
        running=info["running"],
        source=info["config"]["source"],
        source_info=info["source_info"],
        config=info["config"],
        events_count=info["events_count"],
    )


@app.get("/stream/frame")
async def stream_frame() -> Response:
    """Devuelve el último frame procesado con anotaciones como JPEG."""
    with _stream_lock:
        if _stream_processor is None:
            raise HTTPException(status_code=404, detail="No hay stream activo")
        frame = _stream_processor.get_latest_frame()

    if frame is None:
        raise HTTPException(
            status_code=503, detail="Stream activo pero aún sin frames procesados"
        )

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="Error al codificar frame")

    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.get("/stream/plates", response_model=StreamEventResponse)
async def stream_plates(limit: int = 50) -> StreamEventResponse:
    """Devuelve las placas detectadas recientemente."""
    with _stream_lock:
        if _stream_processor is None:
            raise HTTPException(status_code=404, detail="No hay stream activo")
        events = _stream_processor.get_events(limit=limit)

    return StreamEventResponse(plates=events)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "alpr.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
