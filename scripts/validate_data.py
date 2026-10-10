import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from commerce_ai.data import read_jsonl
from commerce_ai.schemas import Interaction, Product


def main():
    parser = argparse.ArgumentParser(description="Validate server datasets before renting GPU time")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--require-complete-products", action="store_true")
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
    local_images = [
        args.data_dir / product.image_url
        for product in products
        if product.image_url and not product.image_url.startswith(("http://", "https://", "data:"))
    ]
    missing_images = [str(path) for path in local_images if not path.is_file()]
    if missing_images:
        errors.append(f"{len(missing_images)} local product images are missing")
    required_fields = ("title", "category", "attributes", "price", "description", "image_url")
    missing_fields = {
        field: sum(getattr(product, field) in (None, "") for product in products)
        for field in required_fields
    }
    if args.require_complete_products and any(missing_fields.values()):
        errors.append(f"incomplete product fields: {missing_fields}")
    report = {
        "products": len(products),
        "interactions": len(interactions),
        "products_with_images": sum(bool(product.image_url) for product in products),
        "missing_product_fields": missing_fields,
        "missing_images": missing_images[:20],
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()

