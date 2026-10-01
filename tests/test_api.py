from fastapi.testclient import TestClient

from commerce_ai.api import app


def test_api_end_to_end():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    search = client.post("/v1/search", json={"query": "降噪耳机", "user_id": "u001"})
    assert search.status_code == 200 and search.json()["items"]
    agent = client.post("/v1/agent", json={"message": "推荐耳机", "user_id": "u001"})
    assert agent.status_code == 200 and agent.json()["trace"]

