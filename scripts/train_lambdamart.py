"""Train and evaluate LambdaMART on query-disjoint, explicitly judged candidates."""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from commerce_ai.data import CommerceData, read_jsonl
from commerce_ai.llm import HeuristicLLM
from commerce_ai.ranking import FEATURE_NAMES, Candidate, MultiObjectiveRanker, ranking_features
from commerce_ai.retrieval import HybridRetriever


GAINS = [0.0, 0.01, 0.1, 1.0]


def load_groups(rows: list[dict], products: dict) -> dict:
    groups = {split: defaultdict(list) for split in ("train", "validation", "test")}
    query_splits = {}
    seen = set()
    for row in rows:
        split = row["split"]
        if split not in groups:
            raise ValueError(f"Invalid split: {split}")
        query = " ".join(row["query"].casefold().split())
        if not query or row["item_id"] not in products:
            raise ValueError("Every label must have a nonempty query and a catalog item ID")
        if query in query_splits and query_splits[query] != split:
            raise ValueError(f"Query appears in multiple splits: {query}")
        query_splits[query] = split
        key = (query, row["item_id"])
        if key in seen:
            raise ValueError(f"Duplicate query-item judgment: {key}")
        seen.add(key)
        relevance = row["relevance"]
        if type(relevance) is not int or not 0 <= relevance < len(GAINS):
            raise ValueError("relevance must be an integer from 0 to 3 (I, C, S, E)")
        groups[split][query].append(row)
    for split in groups:
        groups[split] = {
            query: sorted(group, key=lambda row: row["item_id"])
            for query, group in sorted(groups[split].items())
            if len(group) >= 2 and len({row["relevance"] for row in group}) >= 2
        }
        if not groups[split]:
            raise ValueError(f"No informative query groups in {split}")
    return groups


def matrix(groups: dict, products: dict) -> tuple:
    features, labels, sizes = [], [], []
    for query, rows in groups.items():
        sizes.append(len(rows))
        features.extend(ranking_features(query, products[row["item_id"]]) for row in rows)
        labels.extend(row["relevance"] for row in rows)
    return np.asarray(features), np.asarray(labels), sizes


def metrics(groups: dict, scores: list[float]) -> dict:
    ndcg_scores, mrr_scores = [], []
    offset = 0
    for rows in groups.values():
        predictions = scores[offset:offset + len(rows)]
        offset += len(rows)
        order = sorted(range(len(rows)), key=lambda i: (-predictions[i], rows[i]["item_id"]))
        observed = [rows[i]["relevance"] for i in order[:10]]
        ideal = sorted((row["relevance"] for row in rows), reverse=True)[:10]
        def dcg(grades):
            return sum(GAINS[grade] / math.log2(index + 2) for index, grade in enumerate(grades))
        ndcg_scores.append(dcg(observed) / dcg(ideal))
        mrr_scores.append(next((1 / rank for rank, grade in enumerate(observed, 1) if grade == 3), 0))
    return {"queries": len(groups), "pairs": offset,
            "ndcg_at_10": float(np.mean(ndcg_scores)),
            "mrr_exact_at_10": float(np.mean(mrr_scores))}


def rule_scores(groups: dict, products: dict) -> list[float]:
    scores = []
    explainer = HeuristicLLM()
    for query, rows in groups.items():
        items = [products[row["item_id"]] for row in rows]
        retrieved = dict((product.item_id, score)
                         for product, score in HybridRetriever(items).search(query, len(items)))
        candidates = [Candidate(product, retrieval=retrieved.get(product.item_id, 0)) for product in items]
        ranked = MultiObjectiveRanker().rank(
            candidates, {"intent": "search", "query": query}, explainer, False, len(items), query
        )
        by_id = {item.item_id: item.score for item in ranked}
        scores.extend(by_id.get(row["item_id"], -1.0) for row in rows)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--labels", type=Path, default=Path("data/processed/esci/labels.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/lambdamart/model.txt"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.labels.is_file():
        raise SystemExit(f"Missing relevance labels: {args.labels}; import official judgments first.")
    import lightgbm as lgb

    started = perf_counter()
    products = CommerceData(args.data_dir).products_by_id
    groups = load_groups(read_jsonl(args.labels), products)
    matrices = {split: matrix(group, products) for split, group in groups.items()}
    print(json.dumps({split: {"queries": len(groups[split]), "pairs": len(value[1])}
                      for split, value in matrices.items()}), flush=True)
    train_x, train_y, train_sizes = matrices["train"]
    val_x, val_y, val_sizes = matrices["validation"]
    model = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=GAINS,
        n_estimators=500, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, reg_lambda=1.0, random_state=42,
        n_jobs=args.threads, deterministic=True, force_col_wise=True, verbosity=-1,
    )
    model.fit(
        train_x, train_y, group=train_sizes, feature_name=FEATURE_NAMES,
        eval_set=[(val_x, val_y)], eval_group=[val_sizes], eval_at=[10],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.output))
    report = {
        "model": "LightGBM LambdaMART", "feature_names": FEATURE_NAMES,
        "seed": 42, "threads": args.threads, "label_gain": GAINS,
        "best_iteration": model.best_iteration_, "train_queries": len(groups["train"]),
        "train_pairs": len(train_y), "query_split_overlap": 0,
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "scope": "Rerank a sampled set of judged ESCI candidates; not end-to-end recall or full ESCI",
        "baseline_scope": "Existing weighted rules with BM25/hash fitted only to each judged candidate group",
        "modalities": "text and structured metadata; image embeddings not enabled",
        "evaluation": {},
    }
    manifest_path = args.data_dir / "manifest.json"
    if manifest_path.exists():
        report["data_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split in ("validation", "test"):
        features = matrices[split][0]
        report["evaluation"][split] = {
            "lambdamart": metrics(groups[split], model.predict(features).tolist()),
            "weighted_rules": metrics(groups[split], rule_scores(groups[split], products)),
        }
    report["elapsed_seconds"] = round(perf_counter() - started, 2)
    report["feature_importance_gain"] = dict(zip(
        FEATURE_NAMES, model.booster_.feature_importance(importance_type="gain").tolist()
    ))
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
