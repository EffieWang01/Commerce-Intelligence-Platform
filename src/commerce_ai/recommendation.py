from collections import Counter, defaultdict

import requests

from .schemas import Interaction, Product


EVENT_WEIGHT = {"view": .2, "click": 1.0, "like": 1.5, "cart": 2.0, "purchase": 3.0}
GORSE_FEEDBACK = {"view": "read", "click": "like", "like": "like", "cart": "like", "purchase": "like"}


class ItemCF:
    provider_name = "itemcf"

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


class GorseRecommender:
    """Gorse REST adapter with an explicit ItemCF fallback."""

    provider_name = "gorse"

    def __init__(self, products: list[Product], interactions: list[Interaction], url: str,
                 api_key: str = "", sync: bool = False, timeout: int = 5):
        self.products = {product.item_id: product for product in products}
        self.fallback = ItemCF(products, interactions)
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
        self.last_error: str | None = None
        if sync:
            self._sync(products, interactions)

    def _request(self, method: str, path: str, **kwargs):
        response = requests.request(
            method, self.url + path, headers=self.headers, timeout=self.timeout, **kwargs
        )
        response.raise_for_status()
        return response

    def _sync(self, products: list[Product], interactions: list[Interaction]) -> None:
        for product in products:
            self._request("POST", "/api/item", json={
                "ItemId": product.item_id,
                "IsHidden": False,
                "Categories": [product.category],
                "Timestamp": "2000-01-01T00:00:00Z",
                "Labels": [product.brand, *product.attributes] if product.brand else product.attributes,
                "Comment": product.description,
            })
        feedback = [{
            "FeedbackType": GORSE_FEEDBACK.get(row.event, row.event),
            "UserId": row.user_id,
            "ItemId": row.item_id,
            "Timestamp": row.timestamp,
        } for row in interactions]
        if feedback:
            self._request("POST", "/api/feedback", json=feedback)

    def recommend(self, user_id: str | None, limit: int = 30) -> list[tuple[Product, float]]:
        if not user_id:
            return self.fallback.recommend(user_id, limit)
        try:
            payload = self._request("GET", f"/api/recommend/{user_id}", params={"n": limit}).json()
            ids = payload.get("Items", payload) if isinstance(payload, dict) else payload
            item_ids = [
                row.get("Id", row.get("ItemId")) if isinstance(row, dict) else row
                for row in ids
            ]
            ranked = [self.products[item_id] for item_id in item_ids if item_id in self.products]
            self.last_error = None
            denominator = max(1, len(ranked))
            return [(product, 1.0 - index / denominator) for index, product in enumerate(ranked)]
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            self.last_error = type(exc).__name__
            return self.fallback.recommend(user_id, limit)

