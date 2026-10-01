from collections import Counter, defaultdict

from .schemas import Interaction, Product


EVENT_WEIGHT = {"view": .2, "click": 1.0, "like": 1.5, "cart": 2.0, "purchase": 3.0}


class ItemCF:
    def __init__(self, products: list[Product], interactions: list[Interaction]):
        self.products = {product.item_id: product for product in products}
        self.user_items: dict[str, Counter] = defaultdict(Counter)
        self.popularity = Counter()
        for row in interactions:
            weight = EVENT_WEIGHT.get(row.event, 0)
            self.user_items[row.user_id][row.item_id] += weight
            self.popularity[row.item_id] += weight
        self.neighbors: dict[str, Counter] = defaultdict(Counter)
        for items in self.user_items.values():
            keys = list(items)
            for left in keys:
                for right in keys:
                    if left != right:
                        self.neighbors[left][right] += min(items[left], items[right])

    def recommend(self, user_id: str | None, limit: int = 30) -> list[tuple[Product, float]]:
        seen = self.user_items.get(user_id or "", Counter())
        scores = Counter()
        for item_id, weight in seen.items():
            for candidate, similarity in self.neighbors[item_id].items():
                if candidate not in seen:
                    scores[candidate] += weight * similarity
        if not scores:
            scores.update(self.popularity)
        maximum = max(scores.values(), default=1)
        return [(self.products[item_id], score / maximum) for item_id, score in scores.most_common(limit)
                if item_id in self.products]

