import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from commerce_ai.data import read_jsonl
from commerce_ai.schemas import Interaction, Product


def main():
    parser = argparse.ArgumentParser(description="Validate server datasets before renting GPU time")
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    errors = []
    products = []
    try:
        products = [Product.model_validate(row) for row in read_jsonl(args.data_dir / "products.jsonl")]
        interactions = [Interaction.model_validate(row) for row in read_jsonl(args.data_dir / "interactions.jsonl")]
    except (FileNotFoundError, ValidationError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
        interactions = []
    product_ids = {product.item_id for product in products}
    missing = sorted({row.item_id for row in interactions} - product_ids)
    if missing:
        errors.append(f"interactions reference {len(missing)} missing items")
    report = {"products": len(products), "interactions": len(interactions), "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()

