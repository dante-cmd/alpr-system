"""Procesador de stream de video para ALPR."""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from alpr.core.pipeline import ALPRPipeline
from alpr.core.video_source import VideoSource

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuración de procesamiento de stream."""

    source: str | int
    process_every_n_frames: int = 3
    max_fps: float = 5.0
    roi: tuple[int, int, int, int] | None = None
    min_confidence: float = 0.5
    dedup_ttl_seconds: float = 5.0
    detect_vehicle: bool = True


@dataclass
class PlateEvent:
    """Evento de placa detectada."""

    plate: str
    confidence: float
    detection_confidence: float
    valid_format: bool
    bbox: dict[str, int]
    timestamp: float = field(default_factory=time.perf_counter)
    frame_index: int = 0


class StreamProcessor:
    """Orquesta captura + ALPR pipeline en un hilo worker.

    Estrategias de eficiencia:
      - Hilo de captura independiente para no perder frames RTSP.
      - Procesa solo 1 de cada N frames o respeta max_fps.
      - ROI configurable para limitar zona de búsqueda.
      - Deduplicación temporal de placas para evitar OCR repetido.
    """

    def __init__(
        self,
        config: StreamConfig,
        pipeline: ALPRPipeline | None = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline or ALPRPipeline(
            detect_vehicle=config.detect_vehicle,
            device="cpu",
        )
        self.source = VideoSource(config.source)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_index = 0
        self._latest_annotated: np.ndarray | None = None
        self._events: deque[PlateEvent] = deque(maxlen=100)
        self._seen_plates: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _should_process(self, now: float) -> bool:
        """Decide si procesar este frame según skip y max_fps."""
        self._frame_index += 1
        if self._frame_index % self.config.process_every_n_frames != 0:
            return False
        if self.config.max_fps > 0:
            min_interval = 1.0 / self.config.max_fps
            if now - getattr(self, "_last_process_time", 0.0) < min_interval:
                return False
        return True

    def _dedupe(self, plate: str) -> bool:
        """Retorna True si la placa es nueva o expiró su TTL."""
        now = time.perf_counter()
        last_seen = self._seen_plates.get(plate)
        if last_seen is not None and (now - last_seen) < self.config.dedup_ttl_seconds:
            return False
        self._seen_plates[plate] = now
        # Limpia entradas antiguas periódicamente
        if len(self._seen_plates) > 500:
            cutoff = now - self.config.dedup_ttl_seconds
            self._seen_plates = {p: t for p, t in self._seen_plates.items() if t > cutoff}
        return True

    def _annotate(self, frame: np.ndarray, result: dict[str, Any]) -> np.ndarray:
        """Dibuja bounding boxes y texto sobre el frame."""
        annotated = frame.copy()
        for plate in result.get("plates", []):
            bbox = plate["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            color = (0, 255, 0) if plate["valid_format"] else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{plate['plate']} {plate['confidence']:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return annotated

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """Recorta la región de interés si está configurada."""
        if self.config.roi is None:
            return frame
        x1, y1, x2, y2 = self.config.roi
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    def _process_loop(self) -> None:
        """Bucle principal de procesamiento."""
        if not self.source.start():
            logger.error(
                "stream_processor_source_failed: %s", str(self.config.source)
            )
            return

        self._last_process_time = 0.0
        while not self._stop_event.is_set():
            frame = self.source.latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            if not self._should_process(now):
                continue

            self._last_process_time = now
            input_frame = self._crop_roi(frame)

            try:
                result = self.pipeline.run(input_frame)
            except Exception:
                logger.exception("stream_processor_pipeline_error")
                continue

            annotated = self._annotate(input_frame, result)
            # Si se recortó ROI, volver a incrustar en el frame original para visualización
            if self.config.roi is not None:
                x1, y1, _, _ = self.config.roi
                display = frame.copy()
                rh, rw = annotated.shape[:2]
                display[y1 : y1 + rh, x1 : x1 + rw] = annotated
                annotated = display

            with self._lock:
                self._latest_annotated = annotated
                for plate in result.get("plates", []):
                    text = plate["plate"]
                    if plate["confidence"] < self.config.min_confidence:
                        continue
                    if not self._dedupe(text):
                        continue
                    self._events.append(
                        PlateEvent(
                            plate=text,
                            confidence=plate["confidence"],
                            detection_confidence=plate["detection_confidence"],
                            valid_format=plate["valid_format"],
                            bbox=plate["bbox"],
                            timestamp=now,
                            frame_index=self._frame_index,
                        )
                    )
                    logger.info(
                        "plate_detected: %s confidence=%.3f",
                        text,
                        plate["confidence"],
                    )

        self.source.stop()

    def start(self) -> bool:
        if self.is_running:
            return True
        # Abre la fuente de forma síncrona para poder reportar fallo inmediatamente
        if not self.source.start():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        self.source.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_annotated.copy() if self._latest_annotated is not None else None

    def get_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)[-limit:]
        return [
            {
                "plate": e.plate,
                "confidence": e.confidence,
                "detection_confidence": e.detection_confidence,
                "valid_format": e.valid_format,
                "bbox": e.bbox,
                "timestamp": e.timestamp,
                "frame_index": e.frame_index,
            }
            for e in events
        ]

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.is_running,
                "source_info": self.source.info(),
                "events_count": len(self._events),
                "config": {
                    "source": str(self.config.source),
                    "process_every_n_frames": self.config.process_every_n_frames,
                    "max_fps": self.config.max_fps,
                    "roi": self.config.roi,
                    "min_confidence": self.config.min_confidence,
                    "dedup_ttl_seconds": self.config.dedup_ttl_seconds,
                    "detect_vehicle": self.config.detect_vehicle,
                },
            }
