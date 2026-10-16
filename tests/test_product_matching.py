from pathlib import Path

import pytest

from commerce_ai.config import Settings
from commerce_ai.product_matching import keyboard_product_type, matches_product_target, query_product_target
from commerce_ai.ranking import MultiObjectiveRanker
from commerce_ai.schemas import Product
from commerce_ai.service import CommerceService


@pytest.mark.parametrize("query,expected", [
    ("mechanical keyboard", "mechanical_keyboard"),
    ("keyboard", "keyboard"),
    ("mechanical keyboard keycaps", "keycaps"),
    ("keyboard cable", "cable"),
    ("keyboard wrist rest", "wrist_rest"),
    ("keyboard keycap puller", "keycap_tool"),
    ("keyboard switch opener", "switch_tool"),
    ("keyboard switches", "switches"),
    ("mechanical keyboard without keycaps", "mechanical_keyboard"),
    ("running shoes", None),
    ("phone cable", None),
])
def test_query_target_respects_requested_and_excluded_accessories(query, expected):
    assert query_product_target(query) == expected


@pytest.mark.parametrize("title,expected", [
    ("Mechanical Keyboard with PBT Keycaps and USB Cable", "mechanical_keyboard"),
    ("Gaming Mechanical Keyboard, Blue Switches, Keycap Puller Included", "mechanical_keyboard"),
    ("Gaming Keyboard and Mouse Combo with RGB Lights", "keyboard"),
    ("Mechanical Feeling Gaming Keyboard", "keyboard"),
    ("PBT Keycaps for Mechanical Keyboard (Only Keycap)", "keycaps"),
    ("Coiled Keyboard Cable for Mechanical Gaming Keyboards", "cable"),
    ("Keyboard Keycaps Set", "keycaps"),
    ("Wooden Wrist Rest - Mechanical Keyboard Palm Rest", "wrist_rest"),
    ("Keycap Puller Tool for Mechanical Keyboard", "keycap_tool"),
    ("Mechanical Keyboard Switch Opener", "switch_tool"),
    ("Keyboard Stabilizers for MX Switches", "stabilizer"),
    ("Replacement Keyboard Switches 90 Pack", "switches"),
])
def test_product_head_is_distinguished_from_bundled_and_compatible_parts(title, expected):
    assert keyboard_product_type(title) == expected


def test_description_can_explicitly_exclude_a_complete_keyboard():
    assert not matches_product_target("mechanical_keyboard", "Mechanical Keyboard DIY Parts", "Keyboard not included")
    assert matches_product_target("keyboard", "Gaming Mechanical Keyboard")


@pytest.mark.parametrize("learned", [False, True])
def test_service_filters_wrong_product_types_before_rule_or_learned_ranking(learned):
    root = Path(__file__).parents[1]
    settings = Settings("local", root / "data/sample", root / "artifacts", "heuristic", "hash",
                        "heuristic-local", "hash-384", "weighted-local", 10, 30, 4)
    service = CommerceService(settings)
    products = [Product(item_id=item_id, title=title, category="电子数码", description=title, price=20)
                for item_id, title in [
                    ("full", "Gaming Mechanical Keyboard with PBT Keycaps and USB Cable"),
                    ("caps", "PBT Keycaps for Mechanical Keyboard"),
                    ("wire", "Coiled Keyboard Cable for Mechanical Keyboard"),
                    ("puller", "Keycap Puller Tool for Mechanical Keyboard"),
                ]]
    service.retriever.search = lambda *_: [(product, 0.9) for product in products]
    if learned:
        class FakeModel:
            def predict(self, rows):
                return [0.9] * len(rows)
        service.ranker = MultiObjectiveRanker(FakeModel())
    for query, expected_id in [("mechanical keyboard", "full"),
                               ("mechanical keyboard keycaps", "caps"),
                               ("keyboard cable", "wire"),
                               ("keyboard keycap puller", "puller")]:
        result = service.search(query, top_k=5)
        assert [item.item_id for item in result.items] == [expected_id]
        match = next(event for event in result.trace if event["stage"] == "product_match")
        assert match["excluded"] == 3
