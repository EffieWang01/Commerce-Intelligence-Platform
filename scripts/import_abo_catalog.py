"""Build a small, balanced demo catalog from Amazon Berkeley Objects metadata."""

import argparse
import csv
import gzip
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests


BASE_URL = "https://amazon-berkeley-objects.s3.amazonaws.com"
SOURCE_NAME = "Amazon Berkeley Objects (CC BY 4.0)"
LANGUAGE_ORDER = ("zh_CN", "zh_TW", "en_US", "en_GB")
CATEGORY_RULES = (
    ("电子数码", ("electronic", "computer", "monitor", "keyboard", "mouse", "headphone", "audio", "camera", "phone", "tablet", "speaker", "projector", "printer")),
    ("家用电器", ("appliance", "vacuum", "heater", "air conditioner", "refrigerator", "washer", "blender", "coffee maker", "toaster", "electric fan")),
    ("灯具照明", ("lighting", "light fixture", "lamp", "chandelier", "lantern")),
    ("床品家纺", ("bedding", "bed in a bag", "bed linen", "comforter", "pillow", "blanket", "sheet", "towel", "duvet", "mattress")),
    ("卫浴清洁", ("bath", "bathroom", "shower", "toilet", "cleaning", "laundry")),
    ("收纳整理", ("storage", "organizer", "basket", "container", "holder", "rack")),
    ("家居装饰", ("home decor", "decoration", "artwork", "wall art", "blueprint", "rug", "curtain", "picture frame", "mirror", "vase", "clock", "candle")),
    ("厨房餐饮", ("kitchen", "cookware", "bakeware", "dish", "bowl", "cup", "mug", "glassware", "utensil", "knife", "dining")),
    ("家具", ("furniture", "chair", "table", "desk", "sofa", "cabinet", "shelf", "bed frame", "stool", "bench")),
    ("服装", ("clothing", "shirt", "dress", "pants", "jacket", "coat", "underwear", "apparel")),
    ("鞋靴", ("shoe", "boot", "sandal", "slipper", "footwear")),
    ("箱包", ("bag", "luggage", "wallet", "backpack", "handbag", "suitcase")),
    ("珠宝配饰", ("jewelry", "necklace", "ring", "bracelet", "watch", "eyewear", "sunglasses")),
    ("美妆个护", ("beauty", "cosmetic", "skin care", "hair care", "perfume", "makeup", "personal care")),
    ("运动户外", ("sport", "fitness", "exercise", "outdoor", "camping", "cycling", "golf", "yoga")),
    ("玩具游戏", ("toy", "game", "puzzle", "doll", "playset")),
    ("母婴用品", ("baby", "infant", "diaper", "nursery")),
    ("宠物用品", ("pet", "dog", "cat", "aquarium")),
    ("汽车用品", ("automotive", "car accessory", "vehicle", "motorcycle")),
    ("工具五金", ("tool", "hardware", "drill", "wrench", "screw", "work safety")),
    ("办公文具", ("office", "stationery", "pen", "paper", "notebook", "binder")),
    ("园艺", ("garden", "lawn", "plant pot", "patio")),
    ("乐器", ("musical instrument", "guitar", "piano", "drum", "violin")),
    ("食品饮料", ("grocery", "food", "beverage", "coffee", "tea", "snack")),
    ("医疗健康", ("health", "medical", "mobility aid", "first aid")),
)


def localized(values, limit: int = 1) -> list[str]:
    if not values:
        return []
    if isinstance(values, dict):
        values = [values]
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    rows = [row for row in values if isinstance(row, dict) and str(row.get("value", "")).strip()]
    ordered = sorted(
        rows,
        key=lambda row: LANGUAGE_ORDER.index(row.get("language_tag"))
        if row.get("language_tag") in LANGUAGE_ORDER else len(LANGUAGE_ORDER),
    )
    result = []
    for row in ordered:
        value = str(row["value"]).strip()
        if value not in result:
            result.append(value)
        if len(result) >= limit:
            break
    return result


def category_for(row: dict) -> str:
    searchable = []
    product_types = row.get("product_type") or []
    if isinstance(product_types, dict):
        product_types = [product_types]
    searchable.extend(
        str(value.get("value", "")) if isinstance(value, dict) else str(value)
        for value in product_types
    )
    nodes = row.get("node") or []
    if isinstance(nodes, dict):
        nodes = [nodes]
    for node in nodes:
        if isinstance(node, dict):
            path = str(node.get("node_name") or node.get("node_path") or "")
            searchable.append(next((part for part in reversed(path.split("/")) if part), path))
    searchable.extend(localized(row.get("item_name"), 2))
    text = re.sub(r"[^a-z0-9]+", " ", " ".join(searchable).replace("_", " ").lower())
    for category, keywords in CATEGORY_RULES:
        if any(re.search(rf"\b{re.escape(keyword)}\w*\b", text) for keyword in keywords):
            return category
    return "其他商品"


def download(relative_path: str, cache_dir: Path) -> Path:
    target = cache_dir / relative_path.replace("/", "_")
    if target.is_file() and target.stat().st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(f"{BASE_URL}/{relative_path}", stream=True, timeout=90) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return target


def iter_listings(paths: list[Path]):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def image_paths(metadata_path: Path, wanted: set[str]) -> dict[str, str]:
    found = {}
    with gzip.open(metadata_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("image_id", "")
            if image_id in wanted:
                found[image_id] = row.get("path", "")
                if len(found) == len(wanted):
                    break
    return found


def product_from_abo(row: dict, image_path: str | None) -> dict | None:
    item_id = str(row.get("item_id", "")).strip()
    titles = localized(row.get("item_name"))
    if not item_id or not titles:
        return None
    descriptions = localized(row.get("product_description"), 2)
    bullets = localized(row.get("bullet_point"), 4)
    brands = localized(row.get("brand"))
    attributes = []
    for field in ("color", "material", "style", "product_type"):
        attributes.extend(localized(row.get(field), 2))
    attributes = list(dict.fromkeys(value for value in attributes if value))[:8]
    description = "；".join(descriptions + bullets) or "；".join(attributes) or titles[0]
    image_url = None
    if image_path:
        image_url = f"{BASE_URL}/{quote(image_path, safe='/')}"
    return {
        "item_id": f"abo_{item_id}",
        "title": titles[0],
        "category": category_for(row),
        "description": description[:2000],
        "price": None,
        "brand": brands[0] if brands else None,
        "image_url": image_url,
        "attributes": attributes,
        "source": SOURCE_NAME,
        "source_id": item_id,
        "source_price": None,
        "source_currency": None,
        "sponsored": False,
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/demo"))
    parser.add_argument("--cache", type=Path, default=Path("data/.cache/abo"))
    parser.add_argument("--shards", default="0,1,2,3", help="Hex shard suffixes from 0 to f")
    parser.add_argument("--max-products", type=int, default=500)
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--without-sample", action="store_true")
    args = parser.parse_args()

    suffixes = [value.strip().lower() for value in args.shards.split(",") if value.strip()]
    if not suffixes or any(value not in "0123456789abcdef" for value in suffixes):
        raise SystemExit("--shards must be a comma-separated subset of 0..f")
    if args.max_products < 1 or args.per_category < 1:
        raise SystemExit("--max-products and --per-category must be positive")

    listing_paths = [download(f"listings/metadata/listings_{suffix}.json.gz", args.cache) for suffix in suffixes]
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in iter_listings(listing_paths):
        category = category_for(row)
        if len(buckets[category]) < args.per_category:
            buckets[category].append(row)

    selected = []
    categories = sorted(buckets, key=lambda name: (name == "其他商品", name))
    while len(selected) < args.max_products:
        added = False
        for category in categories:
            if buckets[category]:
                selected.append(buckets[category].pop(0))
                added = True
                if len(selected) >= args.max_products:
                    break
        if not added:
            break

    wanted_images = {str(row.get("main_image_id")) for row in selected if row.get("main_image_id")}
    images_metadata = download("images/metadata/images.csv.gz", args.cache)
    paths_by_id = image_paths(images_metadata, wanted_images)
    imported = []
    for row in selected:
        image_id = str(row.get("main_image_id", ""))
        product = product_from_abo(row, paths_by_id.get(image_id))
        if product:
            imported.append(product)

    root = Path(__file__).resolve().parents[1]
    sample_dir = root / "data" / "sample"
    sample_products = [] if args.without_sample else read_jsonl(sample_dir / "products.jsonl")
    for product in sample_products:
        product.setdefault("source", "Local synthetic sample")

    args.output.mkdir(parents=True, exist_ok=True)
    products = sample_products + imported
    write_jsonl(args.output / "products.jsonl", products)
    if args.without_sample:
        write_jsonl(args.output / "interactions.jsonl", [])
        sample_queries = []
    else:
        shutil.copyfile(sample_dir / "interactions.jsonl", args.output / "interactions.jsonl")
        sample_queries = read_jsonl(sample_dir / "queries.jsonl")
    category_items: dict[str, list[str]] = defaultdict(list)
    for product in imported:
        category_items[product["category"]].append(product["item_id"])
    generated_queries = [
        {"query": category, "user_id": None, "relevant_item_ids": item_ids[:20]}
        for category, item_ids in sorted(category_items.items())
    ]
    write_jsonl(args.output / "queries.jsonl", sample_queries + generated_queries)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE_NAME,
        "source_url": "https://amazon-berkeley-objects.s3.amazonaws.com/index.html",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "listing_shards": suffixes,
        "sample_products": len(sample_products),
        "imported_products": len(imported),
        "products_with_images": sum(bool(row.get("image_url")) for row in imported),
        "categories": {name: len(rows) for name, rows in sorted(category_items.items())},
        "price_note": "ABO does not provide prices; imported products use null rather than fabricated values.",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "ATTRIBUTION.md").write_text(
        "# Dataset attribution\n\n"
        "This demo subset is derived from the [Amazon Berkeley Objects dataset]"
        "(https://amazon-berkeley-objects.s3.amazonaws.com/index.html), licensed under "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The subset selects "
        "listing metadata, maps product types into broad Chinese demo categories, and keeps "
        "the official image URLs. Amazon and UC Berkeley do not endorse this demo.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
