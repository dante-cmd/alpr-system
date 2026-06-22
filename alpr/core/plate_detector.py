"""Detector de placas usando YOLOv8 fine-tuned."""

from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from alpr.core.config import settings


class PlateDetector:
    """Envuelve un detector YOLOv8 entrenado para placas vehiculares."""

    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        self.model_path = model_path or str(settings.plate_model_path)
        self.device = device
        self._model: YOLO | None = None

    def _load(self) -> YOLO:
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def detect(
        self,
        image: np.ndarray,
        region_of_interest: Tuple[int, int, int, int] | None = None,
        conf: float | None = None,
        iou: float | None = None,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Retorna lista de (x1, y1, x2, y2, score) en coordenadas absolutas."""
        conf = conf or settings.conf_threshold
        iou = iou or settings.iou_threshold
        model = self._load()

        if region_of_interest is not None:
            x0, y0, x1, y1 = region_of_interest
            crop = image[y0:y1, x0:x1]
        else:
            x0 = y0 = 0
            crop = image

        results = model.predict(crop, conf=conf, iou=iou, verbose=False, device=self.device)
        plates = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            for (bx1, by1, bx2, by2), score in zip(boxes, scores):
                area = (bx2 - bx1) * (by2 - by1)
                if area < settings.plate_min_area:
                    continue
                plates.append(
                    (int(bx1 + x0), int(by1 + y0), int(bx2 + x0), int(by2 + y0), float(score))
                )
        return plates
