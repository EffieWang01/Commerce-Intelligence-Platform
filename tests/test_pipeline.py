from pathlib import Path

from commerce_ai.agent import CommerceAgent
from commerce_ai.config import Settings
from commerce_ai.data import read_jsonl
from commerce_ai.evaluation import evaluate_queries, ndcg_at_k, recall_at_k
from commerce_ai.service import CommerceService


ROOT = Path(__file__).parents[1]


def settings():
    return Settings("local", ROOT / "data" / "sample", ROOT / "artifacts", "heuristic", "hash",
                    "heuristic-local", "hash-384", "weighted-local", 10, 30, 4)


def test_search_is_grounded_and_respects_price():
    result = CommerceService(settings()).search("预算 500 的降噪耳机", "u001", top_k=5)
    assert result.items
    assert all(item.price <= 500 for item in result.items)
    assert result.items[0].item_id in {"p001", "p002"}
    assert all(item.reason for item in result.items)


def test_recommendation_and_agent_tools():
    service = CommerceService(settings())
    recommendation = service.recommend("u001", top_k=3)
    assert recommendation.items
    agent = CommerceAgent(service)
    search = agent.run("帮我找预算 500 的降噪耳机", "u001")
    assert search["items"] and any(step["action"] == "search_products" for step in search["trace"])
    constrained_recommendation = agent.run("帮我找预算 500 的降噪耳机，并说明推荐理由", "u001")
    assert constrained_recommendation["items"]
    assert all(item["price"] <= 500 for item in constrained_recommendation["items"])
    assert all(item["category"] == "耳机" for item in constrained_recommendation["items"])
    comparison = agent.run("对比 p001 和 p002")
    assert len(comparison["items"]) == 2


def test_offline_evaluation_contract():
    service = CommerceService(settings())
    report = evaluate_queries(service, read_jsonl(ROOT / "data" / "sample" / "queries.jsonl"))
    assert report.queries == 4
    assert 0 <= report.ndcg_at_10 <= 1
    assert report.recall_at_10 > 0
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1
    assert recall_at_k(["a", "b"], {"a", "c"}, 2) == .5

