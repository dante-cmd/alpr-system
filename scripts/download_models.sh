#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${1:-models}"
mkdir -p "$MODELS_DIR"

echo "Descargando YOLOv8n COCO..."
curl -L -o "$MODELS_DIR/yolov8n.pt" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"

echo "Descargando detector de placas base..."
curl -L -o "$MODELS_DIR/license_plate_detector.pt" \
  "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"

echo "Modelos listos en $MODELS_DIR"
