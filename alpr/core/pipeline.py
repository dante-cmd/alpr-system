"""Pipeline end-to-end ALPR."""

from typing import Any, Dict, List

import cv2
import numpy as np

from alpr.core.config import settings
from alpr.core.plate_detector import PlateDetector
from alpr.core.plate_ocr import PlateOCR
from alpr.core.postprocess import post_process
from alpr.core.vehicle_detector import VehicleDetector


class ALPRPipeline:
    """Orquesta detección de vehículo, placa, OCR y postprocesamiento."""

    def __init__(
        self,
        detect_vehicle: bool | None = None,
        vehicle_model_path: str | None = None,
        plate_model_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.detect_vehicle = (
            detect_vehicle if detect_vehicle is not None else settings.detect_vehicle_first
        )
        self.vehicle_detector = VehicleDetector(vehicle_model_path, device=device)
        self.plate_detector = PlateDetector(plate_model_path, device=device)
        self.ocr = PlateOCR()

    def preprocess(self, image: np.ndarray, target_width: int = 1280) -> np.ndarray:
        """Escala manteniendo relación de aspecto para acelerar inferencia."""
        h, w = image.shape[:2]
        if w <= target_width:
            return image
        scale = target_width / w
        new_size = (target_width, int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    def _enhance_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """Mejora local de contraste para OCR."""
        if len(plate_crop.shape) == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_crop
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def run(self, image: np.ndarray) -> Dict[str, Any]:
        """Ejecuta el pipeline completo sobre una imagen BGR."""
        image = self.preprocess(image)
        scale_x = image.shape[1] / image.shape[1]  # 1.0 si no se escala
        scale_y = image.shape[0] / image.shape[0]

        plates_output: List[Dict[str, Any]] = []

        if self.detect_vehicle:
            vehicles = self.vehicle_detector.detect(image)
            search_regions = [(x1, y1, x2, y2) for x1, y1, x2, y2, _, _ in vehicles]
        else:
            search_regions = [(0, 0, image.shape[1], image.shape[0])]

        for region in search_regions:
            plates = self.plate_detector.detect(image, region_of_interest=region)
            
            for x1, y1, x2, y2, det_score in plates:
                plate_crop = image[y1:y2, x1:x2]
                enhanced = self._enhance_plate(plate_crop)
                raw_text, ocr_score = self.ocr.read(enhanced)
                # print("raw_text", raw_text)
                # print("ocr_score", ocr_score)
                processed = post_process(raw_text, ocr_score)
                plates_output.append(
                    {
                        "plate": processed["text"],
                        "raw_text": processed["raw_text"],
                        "confidence": processed["confidence"],
                        "detection_confidence": float(det_score),
                        "valid_format": processed["valid_format"],
                        "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                    }
                )

        # Si no se detectó vehículo, de-duplicar por bbox superpuesto
        plates_output = self._deduplicate(plates_output)

        return {
            "success": True,
            "plates": plates_output,
            "count": len(plates_output),
        }

    @staticmethod
    def _deduplicate(plates: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """NMS simple por IoU; conserva el de mayor confianza OCR."""
        if not plates:
            return plates

        def iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
            ax1, ay1, ax2, ay2 = a["bbox"].values()
            bx1, by1, bx2, by2 = b["bbox"].values()
            xi1, yi1 = max(ax1, bx1), max(ay1, by1)
            xi2, yi2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return inter / (area_a + area_b - inter + 1e-6)

        plates_sorted = sorted(plates, key=lambda p: p["confidence"], reverse=True)
        kept = []
        for p in plates_sorted:
            if all(iou(p, k) < iou_threshold for k in kept):
                kept.append(p)
        return kept
