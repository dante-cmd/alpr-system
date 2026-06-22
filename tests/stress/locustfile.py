"""Test de estrés con Locust para el endpoint /detect."""

import io

import cv2
import numpy as np
from locust import HttpUser, between, task


class ALPRUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def _make_image(self) -> bytes:
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", image)
        return encoded.tobytes()

    @task
    def detect(self) -> None:
        data = self._make_image()
        self.client.post(
            "/detect",
            files={"file": ("stress.jpg", io.BytesIO(data), "image/jpeg")},
            timeout=30,
        )

    @task(3)
    def health(self) -> None:
        self.client.get("/health")
