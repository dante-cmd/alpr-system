"""Tests de endpoints de streaming de video."""

import pytest
from fastapi.testclient import TestClient

from alpr.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_stream_status_without_active_stream(client: TestClient) -> None:
    response = client.get("/stream/status")
    assert response.status_code == 404


def test_stream_frame_without_active_stream(client: TestClient) -> None:
    response = client.get("/stream/frame")
    assert response.status_code == 404


def test_stream_plates_without_active_stream(client: TestClient) -> None:
    response = client.get("/stream/plates")
    assert response.status_code == 404


def test_stream_stop_without_active_stream(client: TestClient) -> None:
    response = client.post("/stream/stop")
    assert response.status_code == 200
    assert response.json()["stopped"] is True


def test_stream_start_invalid_source(client: TestClient) -> None:
    response = client.post(
        "/stream/start",
        json={"source": "/nonexistent/video.mp4"},
    )
    # La apertura falla y el endpoint devuelve 503
    assert response.status_code == 503
