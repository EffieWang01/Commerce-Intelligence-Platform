"""Prepare an official ESCI text-only ranking subset, separate from the demo catalog."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from commerce_ai.schemas import Product


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/esci"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/esci_text"))
    parser.add_argument("--train-queries", type=int, default=3000)
    parser.add_argument("--validation-queries", type=int, default=500)
    parser.add_argument("--test-queries", type=int, default=500)
    args = parser.parse_args()
    data = pd.read_parquet(args.raw_dir / "examples.parquet")
    data = data[(data.product_locale == "us") & (data.small_version == 1)].copy()
    data["key"] = data["query"].map(lambda query: " ".join(query.casefold().split()))
    test_keys = set(data.loc[data.split == "test", "key"])
    data = data[~((data.split == "train") & data.key.isin(test_keys))]
    data = data.drop_duplicates(["key", "product_id", "esci_label"])
    data = data[~data.duplicated(["key", "product_id"], keep=False)]
    data = data[data.key != ""]
    digest = {key: hashlib.sha256(("42:" + key).encode()).hexdigest() for key in data.key.unique()}
    data["target_split"] = [
        "test" if row.split == "test" else
        "validation" if int(digest[row.key][:8], 16) % 5 == 0 else "train"
        for row in data.itertuples(index=False)
    ]
    limits = {"train": args.train_queries, "validation": args.validation_queries, "test": args.test_queries}
    selected_keys = set()
    for split, limit in limits.items():
        eligible = data.loc[data.target_split == split].groupby("key").esci_label.nunique()
        ordered = sorted(eligible[eligible >= 2].index, key=lambda key: digest[key])
        selected_keys.update(ordered[:limit])
    data = data[data.key.isin(selected_keys)].copy()
    wanted = set(data.product_id)
    products = {}
    columns = ["product_id", "product_locale", "product_title", "product_description",
               "product_bullet_point", "product_brand"]
    for batch in pq.ParquetFile(args.raw_dir / "products.parquet").iter_batches(batch_size=50000, columns=columns):
        frame = batch.to_pandas()
        frame = frame[(frame.product_locale == "us") & frame.product_id.isin(wanted)]
        for row in frame.to_dict("records"):
            def text(name):
                value = row[name]
                return str(value).strip() if pd.notna(value) else ""
            title, description, bullet = text("product_title"), text("product_description"), text("product_bullet_point")
            if not title or not (description or bullet):
                continue
            product = Product(
                item_id="esci_" + row["product_id"], title=title, category="",
                description="\n".join(filter(None, [description, bullet])),
                brand=text("product_brand") or None, attributes=[bullet] if bullet else [],
                source="Amazon ESCI text-only training subset", source_id=row["product_id"],
            )
            products[row["product_id"]] = product
    data = data[data.product_id.isin(products)]
    eligible = data.groupby("key").esci_label.nunique()
    data = data[data.key.isin(eligible[eligible >= 2].index)].sort_values(["target_split", "key", "product_id"])
    used_ids = set(data.product_id)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "products.jsonl").open("w", encoding="utf-8") as handle:
        for asin in sorted(used_ids):
            handle.write(products[asin].model_dump_json() + "\n")
    grades = {"I": 0, "C": 1, "S": 2, "E": 3}
    with (args.output / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in data.itertuples(index=False):
            handle.write(json.dumps({
                "query": row.key, "item_id": "esci_" + row.product_id,
                "relevance": grades[row.esci_label], "split": row.target_split,
            }) + "\n")
    (args.output / "interactions.jsonl").touch()
    report = {
        "source": "https://github.com/amazon-science/esci-data", "seed": 42,
        "products": len(used_ids), "pairs": len(data),
        "pairs_by_split": dict(Counter(data.target_split)),
        "queries_by_split": data.groupby("target_split").key.nunique().to_dict(),
        "text_filter": "nonempty original title and description or bullet points",
        "modalities": "text only; official ESCI does not provide product images, prices or categories",
        "scope": "Separate ranking experiment; does not replace or alter the complete 50k demo catalog",
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
