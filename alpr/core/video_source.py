"""Fuente de video genérica: webcam local, RTSP/IP o archivo."""

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoSource:
    """Captura continua de video en un hilo independiente.

    Soporta:
      - Índices de webcam local: 0, 1, ...
      - URLs RTSP de cámaras IP (Hikvision, Dahua, etc.)
      - Rutas a archivos de video
    """

    def __init__(
        self,
        source: str | int,
        buffer_size: int = 3,
        reconnect_interval: float = 5.0,
    ) -> None:
        self.source = source
        self.buffer_size = buffer_size
        self.reconnect_interval = reconnect_interval

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._last_connected: float = 0.0
        self._fps: float = 0.0
        self._width: int = 0
        self._height: int = 0

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _open(self) -> bool:
        """Abre el capturador y configura buffer reducido para RTSP."""
        try:
            if isinstance(self.source, str) and self.source.isdigit():
                source = int(self.source)
            else:
                source = self.source
        except ValueError:
            source = self.source

        cap = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not cap.isOpened():
            return False

        # Para RTSP reduce el tamaño de buffer interno de OpenCV para bajar latencia
        if isinstance(source, str) and source.lower().startswith("rtsp"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        self._cap = cap
        self._last_connected = time.perf_counter()
        logger.info(
            "video_source_opened: source=%s width=%d height=%d fps=%.2f",
            str(source),
            self._width,
            self._height,
            self._fps,
        )
        return True

    def _read_loop(self) -> None:
        """Bucle de lectura continua. Reconecta automáticamente si falla."""
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                if time.perf_counter() - self._last_connected < self.reconnect_interval:
                    time.sleep(0.1)
                    continue
                if not self._open():
                    logger.warning(
                        "video_source_reconnect_failed: %s", str(self.source)
                    )
                    time.sleep(self.reconnect_interval)
                    continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("video_source_frame_lost: %s", str(self.source))
                self._release()
                time.sleep(0.1)
                continue

            with self._lock:
                self._frames.append((time.perf_counter(), frame))

            # Para archivos locales evita leer a máxima velocidad y saturar CPU
            src_str = str(self.source)
            is_file = not src_str.isdigit() and not src_str.lower().startswith("rtsp")
            if self._fps > 0 and is_file:
                time.sleep(1.0 / self._fps)

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def start(self) -> bool:
        if self.is_running:
            return True
        if not self._open():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        self._release()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._frames.clear()

    def get_frame(self, timeout: float = 5.0) -> np.ndarray | None:
        """Devuelve el frame más reciente o espera hasta `timeout` segundos."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                if self._frames:
                    return self._frames[-1][1]
            time.sleep(0.01)
        return None

    def latest_frame(self) -> np.ndarray | None:
        """Devuelve el frame más reciente sin bloquear."""
        with self._lock:
            return self._frames[-1][1] if self._frames else None

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "source": str(self.source),
                "running": self.is_running,
                "buffered_frames": len(self._frames),
                "width": self._width,
                "height": self._height,
                "fps": self._fps,
            }
