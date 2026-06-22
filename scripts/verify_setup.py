#!/usr/bin/env python3
"""Verifica que modelos y dependencias del sistema ALPR estén disponibles."""

import importlib
import sys
from pathlib import Path


def check_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"

    required_models = [
        "yolov8n.pt",
        "license_plate_detector.pt",
    ]

    required_modules = [
        "ultralytics",
        "paddleocr",
        "cv2",
        "fastapi",
        "uvicorn",
        "prometheus_client",
        "structlog",
    ]

    ok = True
    print("=== ALPR System Setup Verification ===")

    for model in required_models:
        path = models_dir / model
        status = "OK" if path.exists() else "MISSING"
        if status == "MISSING":
            ok = False
        print(f"  Model {model}: {status}")

    for mod in required_modules:
        status = "OK" if check_module(mod) else "MISSING"
        if status == "MISSING":
            ok = False
        print(f"  Module {mod}: {status}")

    if ok:
        print("\nAll checks passed. You can run the API with: python -m alpr.api.main")
        return 0
    print("\nSome checks failed. Run: make install")
    return 1


if __name__ == "__main__":
    sys.exit(main())
