import pytest
from PIL import Image
import io
import numpy as np
import requests
import time


def _generate_dummy_image() -> bytes:
    img = Image.new("RGB", (224, 224), color=(155, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_predict_endpoint_success():
    """Test the predict endpoint with a valid image."""
    img_bytes = _generate_dummy_image()
    files = {"file": ("dummy.jpg", img_bytes, "image/jpeg")}
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", files=files, timeout=10)
        assert response.status_code == 200
        data = response.json()

        assert "estimated_net_worth" in data
        assert "top_matches" in data
        assert len(data["top_matches"]) == 3
        for match in data["top_matches"]:
            assert "name" in match and "similarity" in match
            assert -1.0 <= match["similarity"] <= 1.0
    except requests.exceptions.ConnectionError:
        pytest.skip("Server not running on localhost:8000")


def test_predict_endpoint_invalid_file():
    """Test the predict endpoint with an invalid file."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", files=files, timeout=10)
        assert response.status_code == 400
    except requests.exceptions.ConnectionError:
        pytest.skip("Server not running on localhost:8000")


if __name__ == "__main__":
    # Run a simple test if called directly
    test_predict_endpoint_success()
    test_predict_endpoint_invalid_file()
    print("All tests passed!") 