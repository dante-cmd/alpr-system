"""Detector de vehículos usando YOLOv8 COCO."""

from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from alpr.core.config import settings


class VehicleDetector:
    """Envuelve YOLOv8 entrenado en COCO para detectar vehículos."""

    COCO_VEHICLE_IDS = {
        "car": 2,
        "motorcycle": 3,
        "bus": 5,
        "truck": 7,
    }

    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        self.model_path = model_path or str(settings.vehicle_model_path)
        self.device = device
        self._model: YOLO | None = None

    def _load(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def detect(
        self,
        image: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
    ) -> List[Tuple[int, int, int, int, float, str]]:
        """Retorna lista de (x1, y1, x2, y2, score, clase)."""
        conf = conf or settings.conf_threshold
        iou = iou or settings.iou_threshold
        model = self._load()
        results = model.predict(image, conf=conf, iou=iou, verbose=False, device=self.device)
        
        detections = []
        allowed_ids = {self.COCO_VEHICLE_IDS[c] for c in settings.vehicle_classes}
        id_to_class = {v: k for k, v in self.COCO_VEHICLE_IDS.items()}
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, classes):
                if cls_id in allowed_ids:
                    detections.append(
                        (int(x1), int(y1), int(x2), int(y2), float(score), id_to_class[cls_id])
                    )
        return detections
