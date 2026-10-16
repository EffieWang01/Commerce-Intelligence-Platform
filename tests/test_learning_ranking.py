import pytest

from commerce_ai.ranking import FEATURE_NAMES, ranking_features
from commerce_ai.schemas import Product
from scripts.train_lambdamart import load_groups, metrics


def test_partial_title_overlap_is_not_counted_as_full_coverage():
    product = Product(item_id="p", title="wireless headphones", category="audio", description="music")
    partial = ranking_features("wireless charger", product)
    exact = ranking_features("wireless headphones", product)
    assert len(partial) == len(FEATURE_NAMES)
    assert partial[1] == 0.5
    assert exact[1] == 1.0


def test_query_split_leakage_is_rejected_after_normalizing_case_and_spaces():
    rows = [
        {"query": "Wireless  Headphones", "item_id": "p", "relevance": 3, "split": "train"},
        {"query": "wireless headphones", "item_id": "q", "relevance": 0, "split": "test"},
    ]
    with pytest.raises(ValueError, match="multiple splits"):
        load_groups(rows, {"p": object(), "q": object()})


def test_esci_ndcg_uses_published_gains_and_candidate_order():
    groups = {"query": [{"item_id": "a", "relevance": 0}, {"item_id": "b", "relevance": 3}]}
    assert metrics(groups, [0.0, 1.0])["ndcg_at_10"] == 1.0
    assert metrics(groups, [1.0, 0.0])["ndcg_at_10"] == pytest.approx(0.63092975)
