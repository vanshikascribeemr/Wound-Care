from fastapi.testclient import TestClient
from app import app
import os

client = TestClient(app)

def test_read_main():
    """Basic health check - ensure index page loads."""
    response = client.get("/")
    assert response.status_code == 200

def test_api_status():
    """Verify appointments endpoint is reachable."""
    response = client.get("/appointments")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
