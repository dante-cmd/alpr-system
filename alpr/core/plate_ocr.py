"""OCR de placas usando PaddleOCR."""

from typing import List, Tuple

import cv2
import numpy as np
from paddleocr import PaddleOCR
from fast_plate_ocr import LicensePlateRecognizer

from alpr.core.config import settings


class PlateOCR:
    """Reconocimiento de caracteres sobre imagen de placa con PaddleOCR."""

    def __init__(
        self,
        lang: str | None = None,
        use_angle_cls: bool | None = None,
        drop_score: float | None = None,
    ) -> None:
        self.lang = lang or settings.ocr_lang
        self.use_angle_cls = use_angle_cls if use_angle_cls is not None else settings.ocr_use_angle_cls
        self.drop_score = drop_score if drop_score is not None else settings.ocr_drop_score
        # self._ocr: PaddleOCR | None = None
        self._ocr: LicensePlateRecognizer| None = None
        self.ocr_model_name = settings.ocr_model_name

    # def _load(self) -> PaddleOCR:
    #     if self._ocr is None:
    #         self._ocr = PaddleOCR(
    #             use_angle_cls=self.use_angle_cls,
    #             lang=self.lang,
    #             show_log=False,
    #             use_gpu=False,
    #         )
    #     return self._ocr
    def _load(self) -> LicensePlateRecognizer:
        if self._ocr is None:
            self._ocr = LicensePlateRecognizer(
                self.ocr_model_name
            )
        return self._ocr

    def recognize(
        self,
        plate_image: np.ndarray,
    ) -> Tuple:
        # List[Tuple[str, float]]:
        """Retorna lista de (texto, score) ordenados de izquierda a derecha."""
        if plate_image.size == 0:
            return []
        # PaddleOCR espera RGB
        if len(plate_image.shape) == 2 or plate_image.shape[2] == 1:
            plate_rgb = cv2.cvtColor(plate_image, cv2.COLOR_GRAY2RGB)
        else:
            plate_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)

        ocr = self._load()
        # result = ocr.ocr(plate_rgb, cls=self.use_angle_cls)
        result = ocr.run(plate_rgb, return_confidence=True)
        
        # if result is None or result[0] is None:
        #     return []
        result_item = result[0]
        plate = result_item.plate
        char_probs = result_item.char_probs
        avg_prob = np.mean(char_probs)

        if len(plate) != 6 and np.all(char_probs<0.9):
            return ()
        
        text, score = plate, avg_prob
        
        # readings = []
        # 
        # for line in result_item:
        #     if line is None:
        #         continue
        #     bbox, (text, score) = line
        #     if score < self.drop_score:
        #         continue
        #     # Centro x para ordenar
        #     center_x = sum(p[0] for p in bbox) / 4.0
        #     readings.append((center_x, text, float(score)))
# 
        # readings.sort(key=lambda x: x[0])
        # return [(text, score) for _, text, score in readings]
        return (text, score)

    def read(
        self,
        plate_image: np.ndarray,
    ) -> Tuple[str, float]:
        """Retorna el texto concatenado y el score promedio."""
        text, score = self.recognize(plate_image)
        # readings = self.recognize(plate_image)
        # if not readings:
        #     return "", 0.0
        # text = "".join(t for t, _ in readings)
        # score = sum(s for _, s in readings) / len(readings)
        return text, score
