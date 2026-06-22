"""Configuración del sistema vía variables de entorno."""

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings cargadas desde variables de entorno."""

    # Paths
    project_root: Path = Path(__file__).resolve().parents[2]
    models_dir: Path = Field(default=Path("models"))
    vehicle_model_path: Path = Field(default=Path("models/yolov8n.pt"))
    plate_model_path: Path = Field(default=Path("models/license_plate_detector.pt"))

    # Pipeline
    detect_vehicle_first: bool = Field(default=True) # True
    vehicle_classes: List[str] = Field(default=["car", "motorcycle", "bus", "truck"])
    conf_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    plate_min_area: int = Field(default=800, ge=0)

    # OCR
    ocr_lang: str = Field(default="en")
    ocr_use_angle_cls: bool = Field(default=True)
    ocr_drop_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # only for fast-plate-ocr
    ocr_model_name: str = Field(default='cct-s-v2-global-model')

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1)
    log_level: str = Field(default="INFO")
    json_logs: bool = Field(default=False)

    # Producción
    enable_prometheus: bool = Field(default=True)
    enable_opentelemetry: bool = Field(default=False)

    model_config = SettingsConfigDict(
        env_prefix="ALPR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
