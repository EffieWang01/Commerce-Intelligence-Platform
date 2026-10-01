import json
from collections import defaultdict
from pathlib import Path

from .schemas import Interaction, Product


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class CommerceData:
    def __init__(self, data_dir: Path):
        self.products = [Product.model_validate(row) for row in read_jsonl(data_dir / "products.jsonl")]
        self.interactions = [Interaction.model_validate(row) for row in read_jsonl(data_dir / "interactions.jsonl")]
        self.products_by_id = {product.item_id: product for product in self.products}
        self.user_history: dict[str, list[str]] = defaultdict(list)
        for event in sorted(self.interactions, key=lambda row: row.timestamp):
            if event.item_id in self.products_by_id and event.event in {"click", "cart", "purchase", "like"}:
                self.user_history[event.user_id].append(event.item_id)

    def history(self, user_id: str | None) -> list[Product]:
        return [self.products_by_id[item_id] for item_id in self.user_history.get(user_id or "", [])]

