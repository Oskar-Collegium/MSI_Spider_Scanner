"""
Tests for Spider Scanner.
Run from project root:
    pytest tests/ -v
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def make_jpeg_bytes(width=224, height=224) -> bytes:
    """Create a minimal valid JPEG image in memory."""
    img = Image.new("RGB", (width, height), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_png_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(60, 80, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Route tests ---

class TestIndexRoute:
    def test_returns_200(self):
        res = client.get("/")
        assert res.status_code == 200

    def test_returns_html(self):
        res = client.get("/")
        assert "text/html" in res.headers["content-type"]

    def test_contains_app_title(self):
        res = client.get("/")
        assert "Spider" in res.text


class TestPredictRoute:
    def test_rejects_non_image(self):
        res = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert res.status_code == 415

    def test_rejects_oversized_image(self):
        large_bytes = b"0" * (6 * 1024 * 1024)  # 6 MB
        res = client.post(
            "/predict",
            files={"file": ("big.jpg", large_bytes, "image/jpeg")},
        )
        assert res.status_code == 413

    def test_accepts_jpeg(self):
        """Should accept a valid JPEG (503 expected until model is trained)."""
        res = client.post(
            "/predict",
            files={"file": ("spider.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        assert res.status_code in (200, 503)

    def test_accepts_png(self):
        res = client.post(
            "/predict",
            files={"file": ("spider.png", make_png_bytes(), "image/png")},
        )
        assert res.status_code in (200, 503)

    def test_predict_response_shape(self, monkeypatch):
        """Mock model to verify response JSON shape."""
        import app.routes as routes

        class FakeModel:
            pass

        def fake_predict(image_bytes, model):
            return {"species": "Araneus diadematus", "confidence": 91.5}

        monkeypatch.setattr(routes, "model", FakeModel())
        monkeypatch.setattr(routes, "predict", fake_predict)

        res = client.post(
            "/predict",
            files={"file": ("spider.jpg", make_jpeg_bytes(), "image/jpeg")},
        )
        assert res.status_code == 200
        data = res.json()
        assert "species" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 100
