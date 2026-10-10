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


def test_queries_are_distinct_and_irrelevant_query_is_empty():
    service = CommerceService(settings())
    headphones = service.search("降噪耳机", top_k=5)
    keyboards = service.search("机械键盘", top_k=5)
    irrelevant = service.search("完全不存在的商品", user_id="u001", top_k=5)
    assert headphones.items[0].category == "耳机"
    assert keyboards.items[0].category == "键盘"
    assert not irrelevant.items


def test_hash_retrieval_does_not_return_collision_only_matches():
    service = CommerceService(settings())
    result = service.search("完全不存在的商品xyz987", top_k=5)
    assert not result.items
    english_result = service.search("xyz987-unseen-catalog-query", top_k=5)
    assert not english_result.items


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
    assert report.mrr_at_10 > 0
    assert report.p95_latency_ms >= 0
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1
    assert recall_at_k(["a", "b"], {"a", "c"}, 2) == .5

