"""Evaluate deterministic query rules on real EcomMMMU relevance examples."""

import argparse
import json
from collections import Counter
from pathlib import Path

import requests

from commerce_ai.query_rules import classify_query_product, parse_query


DATASET_API = "https://datasets-server.huggingface.co/rows"
DATASET_NAME = "NingLab/EcomMMMU"
TASK = "multiclass_product_classification"


def fetch_examples(sample_size: int, split: str, start_offset: int, cache: Path) -> list[dict]:
    if cache.exists():
        cached = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line]
        usable = [row for row in cached if row.get("split") == split]
        if len(usable) >= sample_size:
            return usable[:sample_size]

    rows: list[dict] = []
    offset = start_offset
    session = requests.Session()
    while len(rows) < sample_size:
        response = session.get(DATASET_API, params={
            "dataset": DATASET_NAME,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": 100,
        }, timeout=120)
        response.raise_for_status()
        page = response.json().get("rows", [])
        if not page:
            break
        for wrapped in page:
            row = wrapped["row"]
            if row.get("task") != TASK or row.get("split") != split:
                continue
            payload = json.loads(row["input"])
            rows.append({
                "row_idx": wrapped["row_idx"],
                "split": row["split"],
                "query": payload["query"],
                "product_title": payload["product title"],
                "label": row["output"].strip(),
                "has_image": bool(row.get("images")),
                "is_vc": bool(row.get("is_vc")),
            })
            if len(rows) >= sample_size:
                break
        offset += 100

    if len(rows) < sample_size:
        raise RuntimeError(f"Only found {len(rows)} usable rows from offset {start_offset}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return rows


def macro_f1(labels: list[str], predictions: list[str]) -> float:
    scores = []
    for label in "ABCD":
        tp = sum(gold == pred == label for gold, pred in zip(labels, predictions))
        fp = sum(gold != label and pred == label for gold, pred in zip(labels, predictions))
        fn = sum(gold == label and pred != label for gold, pred in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def evaluate(rows: list[dict]) -> dict:
    labels, predictions = [], []
    constrained = 0
    token_covered = 0
    errors = []
    for row in rows:
        features = parse_query(row["query"])
        prediction, signals = classify_query_product(row["query"], row["product_title"])
        labels.append(row["label"])
        predictions.append(prediction)
        constrained += int(features.constrained)
        token_covered += int(bool(signals["matched_tokens"]))
        if prediction != row["label"] and len(errors) < 10:
            errors.append({
                "query": row["query"],
                "title": row["product_title"],
                "gold": row["label"],
                "pred": prediction,
                **signals,
            })
    correct = sum(gold == pred for gold, pred in zip(labels, predictions))
    majority_label = Counter(labels).most_common(1)[0][0]
    majority_predictions = [majority_label] * len(labels)
    confusion = {gold: Counter(pred for g, pred in zip(labels, predictions) if g == gold) for gold in "ABCD"}
    return {
        "samples": len(rows),
        "rows_with_images": sum(row["has_image"] for row in rows) / len(rows),
        "visual_context_rate": sum(row["is_vc"] for row in rows) / len(rows),
        "label_distribution": dict(Counter(labels)),
        "prediction_distribution": dict(Counter(predictions)),
        "query_constraint_coverage": constrained / len(rows),
        "query_product_token_hit_rate": token_covered / len(rows),
        "proxy_accuracy": correct / len(rows),
        "proxy_macro_f1": macro_f1(labels, predictions),
        "majority_baseline_label": majority_label,
        "majority_baseline_accuracy": sum(label == majority_label for label in labels) / len(labels),
        "majority_baseline_macro_f1": macro_f1(labels, majority_predictions),
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--start-offset", type=int, default=390_000)
    parser.add_argument("--cache", type=Path, default=Path("data/ecommmmu/relevance_sample.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/ecommmmu/rule_report.json"))
    args = parser.parse_args()
    rows = fetch_examples(args.sample_size, args.split, args.start_offset, args.cache)
    report = evaluate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "errors"}, ensure_ascii=False, indent=2))
    print(f"error examples: {args.output}")


if __name__ == "__main__":
    main()
