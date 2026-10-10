import requests

from commerce_ai.evaluation import mrr_at_k, percentile
from commerce_ai.llm import HeuristicLLM, ResilientLLM
from commerce_ai.ranking import Candidate, MultiObjectiveRanker
from commerce_ai.recommendation import GorseRecommender
from commerce_ai.schemas import Interaction, Product
from scripts.import_abo_catalog import category_for, localized


def product(item_id: str = "p100") -> Product:
    return Product(
        item_id=item_id,
        title="轻量无线耳机",
        category="耳机",
        description="蓝牙连接，适合通勤",
        price=299,
        brand="示例",
        attributes=["无线", "轻量"],
    )


def test_llm_runtime_failure_uses_validated_fallback():
    class BrokenProvider:
        provider_name = "broken"

        def understand(self, text):
            raise requests.Timeout()

    llm = ResilientLLM(BrokenProvider(), HeuristicLLM())
    intent = llm.understand("预算 500 的无线耳机")
    assert intent["max_price"] == 500
    assert intent["rewritten_query"] == "预算 500 的无线耳机"
    assert llm.last_error == "Timeout"


def test_heuristic_llm_maps_natural_chinese_category_aliases():
    llm = HeuristicLLM(["厨房餐饮", "家具", "电子数码", "鞋靴"])
    assert llm.understand("咖啡杯")["category"] == "厨房餐饮"
    intent = llm.understand("想买一双鞋子")
    assert intent["category"] == "鞋靴"
    assert "鞋靴" in intent["rewritten_query"]
    assert llm.understand("办公椅")["category"] == "家具"
    assert llm.understand("电视机")["category"] == "电子数码"


def test_gorse_failure_falls_back_to_itemcf(monkeypatch):
    rows = [
        Interaction(user_id="u1", item_id="p100", event="purchase", timestamp="2026-01-01T00:00:00Z")
    ]
    recommender = GorseRecommender([product()], rows, "http://gorse.invalid")

    def fail(*args, **kwargs):
        raise requests.ConnectionError()

    monkeypatch.setattr(requests, "request", fail)
    result = recommender.recommend("u1", 5)
    assert result
    assert recommender.last_error == "ConnectionError"


def test_ranker_enforces_negative_and_price_constraints():
    allowed = product()
    blocked = product("p101").model_copy(update={"title": "有线监听耳机", "price": 899})
    ranked = MultiObjectiveRanker().rank(
        [Candidate(allowed, retrieval=.8), Candidate(blocked, retrieval=1.0)],
        {
            "max_price": 500,
            "min_price": 200,
            "category": "耳机",
            "brand": None,
            "attributes": ["无线"],
            "exclude_terms": ["有线"],
        },
        HeuristicLLM(),
        include_ads=False,
        top_k=10,
    )
    assert [item.item_id for item in ranked] == ["p100"]


def test_ranker_excludes_unknown_price_only_when_budget_is_requested():
    unknown = product().model_copy(update={"price": None})
    ranker = MultiObjectiveRanker()
    base_intent = {
        "category": "耳机", "brand": None, "attributes": [], "exclude_terms": [],
        "max_price": None, "min_price": None,
    }
    assert ranker.rank([Candidate(unknown, retrieval=.8)], base_intent, HeuristicLLM(), False, 10)
    constrained = {**base_intent, "max_price": 500}
    assert not ranker.rank([Candidate(unknown, retrieval=.8)], constrained, HeuristicLLM(), False, 10)


def test_extended_metrics():
    assert mrr_at_k(["x", "a", "b"], {"a"}, 3) == .5
    assert percentile([1, 2, 3, 100], .95) == 100


def test_abo_localization_and_leaf_category_mapping():
    assert localized([
        {"language_tag": "fr_FR", "value": "Tasse"},
        {"language_tag": "zh_CN", "value": "杯子"},
        {"language_tag": "en_US", "value": "Cup"},
    ]) == ["杯子"]
    assert category_for({
        "product_type": [{"value": "HOME"}],
        "node": [{"node_name": "/Home & Garden/Home & Kitchen/Categories/Artwork"}],
        "item_name": [{"value": "Typewriter Blueprint"}],
    }) == "家居装饰"
    assert category_for({
        "product_type": [{"value": "HOME"}],
        "node": [{"node_name": "/Categories/Kitchen & Dining"}],
        "item_name": [{"value": "Bed in a Bag"}],
    }) == "床品家纺"
    assert category_for({"product_type": [{"value": "DRINKING_CUP"}]}) == "厨房餐饮"
