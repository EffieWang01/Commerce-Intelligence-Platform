from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from commerce_ai import api


ROOT = Path(__file__).parents[1]


@pytest.fixture()
def client(monkeypatch):
    # API 端到端验证使用 10 条中文样例，而不是默认的 5 万条英文演示目录：
    # 既保证查询语义匹配，又避免测试依赖外部大数据集。
    monkeypatch.setenv("COMMERCE_DATA_DIR", str(ROOT / "data" / "sample"))
    api.service.cache_clear()
    yield TestClient(api.app)
    api.service.cache_clear()


def test_api_end_to_end(client):
    assert client.get("/health").status_code == 200
    search = client.post("/v1/search", json={"query": "降噪耳机", "user_id": "u001"})
    assert search.status_code == 200 and search.json()["items"]
    agent = client.post("/v1/agent", json={"message": "推荐耳机", "user_id": "u001"})
    assert agent.status_code == 200 and agent.json()["trace"]
    invalid_image = client.post("/v1/search", json={"query": "相似商品", "image_data": "not-base64"})
    assert invalid_image.status_code == 422
