"""Tests de API REST."""

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from alpr.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_metrics_endpoint(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_detect_valid_image(client: TestClient) -> None:
    # Imagen sintética 640x480 con fondo gris
    image = np.full((480, 640, 3), 128, dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", image)
    file = io.BytesIO(encoded.tobytes())
    response = client.post("/detect", files={"file": ("test.jpg", file, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "plates" in data
    assert "processing_time_ms" in data
