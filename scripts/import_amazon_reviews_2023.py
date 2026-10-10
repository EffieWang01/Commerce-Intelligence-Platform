"""Build a field-complete demo catalog from Amazon Reviews 2023 metadata."""

import argparse
import gzip
import io
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
SOURCE_NAME = "Amazon Reviews 2023 (McAuley Lab)"
SOURCE_URL = "https://amazon-reviews-2023.github.io/main.html"
CATEGORY_SOURCES = {
    "All_Beauty": "美妆个护",
    "Automotive": "汽车用品",
    "Clothing_Shoes_and_Jewelry": "服装鞋包",
    "Electronics": "电子数码",
    "Home_and_Kitchen": "家居厨具",
    "Office_Products": "办公文具",
    "Sports_and_Outdoors": "运动户外",
    "Tools_and_Home_Improvement": "工具五金",
}


def clean_text(value) -> str:
    if isinstance(value, list):
        value = "；".join(str(item).strip() for item in value if str(item).strip())
    return " ".join(str(value or "").split())


def parse_price(value) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return round(price, 2) if price > 0 else None


def best_image(images) -> str | None:
    if isinstance(images, dict):
        for key in ("hi_res", "large", "thumb"):
            for value in images.get(key) or []:
                if isinstance(value, str) and value.startswith("https://"):
                    return value
        return None
    if not isinstance(images, list):
        return None
    for key in ("hi_res", "large", "thumb"):
        for image in images:
            value = image.get(key) if isinstance(image, dict) else None
            if isinstance(value, str) and value.startswith("https://"):
                return value
    return None


def product_from_row(row: dict, category: str) -> dict | None:
    source_id = clean_text(row.get("parent_asin"))
    title = clean_text(row.get("title"))
    price = parse_price(row.get("price"))
    image_url = best_image(row.get("images"))
    features = [clean_text(value) for value in row.get("features") or [] if clean_text(value)]
    source_categories = [
        clean_text(value) for value in row.get("categories") or [] if clean_text(value)
    ]
    attributes = list(dict.fromkeys(features + source_categories))[:8] or [category]
    description_parts = [
        clean_text(row.get("description")),
        "；".join(features[:6]),
        "；".join(source_categories),
    ]
    description = "；".join(part for part in description_parts if part)
    if not all((source_id, title, category, price, image_url, description)):
        return None
    return {
        "item_id": f"ar23_{source_id}",
        "title": title[:500],
        "category": category,
        "description": description[:3000],
        "price": price,
        "brand": clean_text(row.get("store")) or None,
        "image_url": image_url,
        "attributes": attributes,
        "source": SOURCE_NAME,
        "source_id": source_id,
        "source_price": price,
        "source_currency": "USD",
        "sponsored": False,
        "ctr_prior": 0.03,
        "cvr_prior": 0.01,
    }


def stream_category_once(source_category: str, category: str, limit: int) -> list[dict]:
    url = f"{BASE_URL}/meta_{source_category}.jsonl.gz"
    products = []
    seen = set()
    session = requests.Session()
    session.trust_env = False
    with session.get(url, stream=True, timeout=(20, 180), headers={"Accept-Encoding": "identity"}) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    product = product_from_row(row, category)
                    if not product or product["source_id"] in seen:
                        continue
                    seen.add(product["source_id"])
                    products.append(product)
                    if len(products) >= limit:
                        break
    if len(products) < limit:
        raise RuntimeError(f"{source_category}: only found {len(products)} complete products")
    return products


def stream_category(source_category: str, category: str, limit: int) -> list[dict]:
    last_error = None
    for attempt in range(1, 4):
        try:
            return stream_category_once(source_category, category, limit)
        except (requests.RequestException, ConnectionError, OSError, EOFError) as exc:
            last_error = exc
            if attempt < 3:
                print(f"{source_category}: attempt {attempt} failed ({type(exc).__name__}), retrying")
                time.sleep(attempt * 2)
    raise last_error


def stream_huggingface_category(source_category: str, category: str, limit: int) -> list[dict]:
    products = []
    seen = set()
    offset = 0
    session = requests.Session()
    while len(products) < limit:
        response = session.get(HF_ROWS_API, params={
            "dataset": "McAuley-Lab/Amazon-Reviews-2023",
            "config": f"raw_meta_{source_category}",
            "split": "full",
            "offset": offset,
            "length": 100,
        }, timeout=120)
        response.raise_for_status()
        rows = response.json().get("rows", [])
        if not rows:
            break
        for wrapped in rows:
            product = product_from_row(wrapped["row"], category)
            if not product or product["source_id"] in seen:
                continue
            seen.add(product["source_id"])
            products.append(product)
            if len(products) >= limit:
                break
        offset += len(rows)
    if len(products) < limit:
        raise RuntimeError(f"{source_category}: only found {len(products)} complete products via Hugging Face")
    return products


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def complete_product(row: dict) -> bool:
    required = ("item_id", "title", "category", "attributes", "price", "description", "image_url")
    return all(row.get(field) not in (None, "", []) for field in required)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--max-products", type=int, default=1000)
    parser.add_argument("--source", choices=("auto", "ucsd", "huggingface"), default="auto")
    parser.add_argument("--cache", type=Path, default=Path("data/.cache/amazon-reviews-2023"))
    args = parser.parse_args()
    if args.max_products < len(CATEGORY_SOURCES):
        raise SystemExit(f"--max-products must be at least {len(CATEGORY_SOURCES)}")

    base, remainder = divmod(args.max_products, len(CATEGORY_SOURCES))
    products = []
    category_counts = {}
    args.cache.mkdir(parents=True, exist_ok=True)
    for index, (source_category, category) in enumerate(CATEGORY_SOURCES.items()):
        limit = base + int(index < remainder)
        cache_path = args.cache / f"{source_category}_{limit}.jsonl"
        cached = read_jsonl(cache_path)
        reused_cache = False
        if len(cached) >= limit and all(complete_product(row) for row in cached[:limit]):
            selected = cached[:limit]
            reused_cache = True
            print(f"{source_category}: reused {len(selected)} cached products")
        elif args.source == "huggingface":
            selected = stream_huggingface_category(source_category, category, limit)
        else:
            try:
                selected = stream_category(source_category, category, limit)
            except (requests.RequestException, ConnectionError, OSError, EOFError) as exc:
                if args.source == "ucsd":
                    raise
                print(f"{source_category}: UCSD unavailable ({type(exc).__name__}), using Hugging Face")
                selected = stream_huggingface_category(source_category, category, limit)
        if not reused_cache:
            write_jsonl(cache_path, selected)
        products.extend(selected)
        category_counts[category] = len(selected)
        print(f"{source_category}: {len(selected)}")

    if len(products) != args.max_products:
        raise RuntimeError(f"expected {args.max_products} products, got {len(products)}")
    if len({row["item_id"] for row in products}) != len(products):
        raise RuntimeError("duplicate product IDs found across source categories")
    if not all(complete_product(row) for row in products):
        raise RuntimeError("incomplete product passed the import filter")

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "products.jsonl", products)
    write_jsonl(args.output / "interactions.jsonl", [])
    grouped: dict[str, list[str]] = defaultdict(list)
    for product in products:
        grouped[product["category"]].append(product["item_id"])
    queries = [
        {"query": category, "user_id": None, "relevant_item_ids": item_ids[:20]}
        for category, item_ids in sorted(grouped.items())
    ]
    write_jsonl(args.output / "queries.jsonl", queries)

    required = ("title", "category", "attributes", "price", "description", "image_url")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "products": len(products),
        "source_categories": list(CATEGORY_SOURCES),
        "categories": category_counts,
        "source_currency": "USD",
        "complete_products": sum(all(row.get(field) not in (None, "") for field in required) for row in products),
        "missing_by_field": {
            field: sum(row.get(field) in (None, "") for row in products) for field in required
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "ATTRIBUTION.md").write_text(
        "# Dataset attribution\n\n"
        "This demo subset is derived from [Amazon Reviews 2023]"
        "(https://amazon-reviews-2023.github.io/main.html), collected by the McAuley Lab at UCSD. "
        "Only records containing title, category, price, description, and an official product image URL "
        "are retained. Prices preserve the source USD currency.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
