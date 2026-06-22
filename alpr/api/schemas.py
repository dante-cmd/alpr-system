"""Esquemas Pydantic para la API."""

from pydantic import BaseModel, Field


class PlateResult(BaseModel):
    plate: str = Field(..., description="Texto de la placa normalizado")
    raw_text: str = Field(..., description="Texto crudo del OCR")
    confidence: float = Field(..., ge=0.0, le=1.0)
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    valid_format: bool
    bbox: dict


class DetectionResponse(BaseModel):
    success: bool
    plates: list[PlateResult]
    count: int
    processing_time_ms: float | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class StreamStartRequest(BaseModel):
    source: str | int = Field(
        ...,
        description="Índice de webcam (0,1...), URL RTSP o ruta a archivo de video",
    )
    process_every_n_frames: int = Field(default=3, ge=1, description="Procesa 1 de cada N frames")
    max_fps: float = Field(default=5.0, ge=0.1, le=60.0, description="Máximo de FPS a procesar")
    roi: tuple[int, int, int, int] | None = Field(
        default=None,
        description="Región de interés (x1, y1, x2, y2)",
    )
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    detect_vehicle: bool = Field(default=True)


class StreamStatusResponse(BaseModel):
    running: bool
    source: str
    source_info: dict
    config: dict
    events_count: int


class StreamEventResponse(BaseModel):
    plates: list[dict]
