"""Join official Amazon ESCI judgments to the existing complete product catalog."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

from commerce_ai.data import read_jsonl


SOURCE_URL = (
    "https://media.githubusercontent.com/media/amazon-science/esci-data/main/"
    "shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
)
GRADES = {"I": 0, "C": 1, "S": 2, "E": 3}
SOURCE_SHA256 = "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a"
SOURCE_SIZE = 51286808


def query_key(query: str) -> str:
    return " ".join(query.casefold().split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/esci/examples.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/esci/labels.jsonl"))
    parser.add_argument("--download-url", default=SOURCE_URL,
                        help="Optional download mirror; official SHA-256 is always enforced")
    args = parser.parse_args()
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    if not args.raw.exists():
        partial = args.raw.with_suffix(".part")
        with requests.get(args.download_url, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
        if hashlib.sha256(partial.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise ValueError("Official ESCI download failed its Git LFS SHA-256 check")
        partial.replace(args.raw)
    with args.raw.open("rb") as handle:
        source_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if source_digest != SOURCE_SHA256:
        raise ValueError("ESCI parquet does not match the pinned official Git LFS object")

    catalog = read_jsonl(args.data_dir / "products.jsonl")
    asin_to_id = {row["source_id"]: row["item_id"] for row in catalog if row.get("source_id")}
    columns = ["query", "query_id", "product_id", "product_locale", "esci_label",
               "small_version", "split"]
    source = pd.read_parquet(args.raw, columns=columns)
    source = source[(source.product_locale == "us") & (source.small_version == 1)].copy()
    official_pairs = len(source)
    matched = source[source.product_id.isin(asin_to_id)].copy()
    matched = matched.dropna(subset=["query", "esci_label"])
    matched["query_key"] = matched["query"].map(query_key)
    matched = matched[matched.query_key != ""]
    # Reserve every occurrence of an official test query, including spelling/case variants.
    test_keys = set(source.loc[source.split == "test", "query"].dropna().map(query_key))
    matched = matched[~((matched.split == "train") & matched.query_key.isin(test_keys))]
    matched = matched.drop_duplicates(["query_key", "product_id", "esci_label"])
    if matched.duplicated(["query_key", "product_id"]).any():
        raise ValueError("Conflicting ESCI labels for the same normalized query/product")
    rows = []
    dropped_groups = 0
    for key, group in matched.groupby("query_key", sort=True):
        if len(group) < 2 or group.esci_label.nunique() < 2:
            dropped_groups += 1
            continue
        split = str(group["split"].iloc[0])
        if split == "train" and int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 5 == 0:
            split = "validation"
        for row in group.itertuples(index=False):
            rows.append({
                "query": key, "query_id": str(row.query_id),
                "item_id": asin_to_id[row.product_id], "relevance": GRADES[row.esci_label],
                "esci_label": row.esci_label, "split": split,
                "source": "amazon-science/esci-data", "source_asin": row.product_id,
            })
    if not rows:
        raise SystemExit("No informative query groups overlap this catalog; labels were not fabricated.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "source": SOURCE_URL,
        "sha256": source_digest,
        "catalog_products": len(catalog), "official_us_small_pairs": official_pairs,
        "overlapping_pairs_before_group_filter": len(matched),
        "overlapping_products": int(matched.product_id.nunique()),
        "dropped_noninformative_query_groups": dropped_groups,
        "eligible_group_policy": "at least 2 matched products and 2 distinct relevance labels",
        "pairs_by_split": dict(Counter(row["split"] for row in rows)),
        "queries_by_split": {split: len({r["query"] for r in rows if r["split"] == split})
                             for split in ("train", "validation", "test")},
        "grade_mapping": GRADES, "ndcg_gains": [0.0, 0.01, 0.1, 1.0],
        "scope": "ASIN-intersection subset; not the full official ESCI benchmark",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
