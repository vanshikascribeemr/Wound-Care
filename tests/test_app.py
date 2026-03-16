from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    """Health endpoint must return 200 and status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_status():
    """Verify appointments endpoint is reachable and returns a list."""
    response = client.get("/appointments")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
