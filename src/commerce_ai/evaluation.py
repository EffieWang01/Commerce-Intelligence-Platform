import math
from time import perf_counter
from dataclasses import dataclass

from .service import CommerceService


def dcg(relevances: list[int]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(relevances))


def ndcg_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    observed = [1 if item in relevant else 0 for item in predicted[:k]]
    ideal = [1] * min(len(relevant), k)
    return dcg(observed) / dcg(ideal) if ideal else 0.0


def recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    return len(set(predicted[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    return next((1 / rank for rank, item in enumerate(predicted[:k], 1) if item in relevant), 0.0)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


@dataclass
class EvaluationReport:
    queries: int
    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    empty_result_rate: float
    p95_latency_ms: float


def evaluate_queries(service: CommerceService, rows: list[dict]) -> EvaluationReport:
    ndcg_scores, recall_scores, mrr_scores, latencies, empty = [], [], [], [], 0
    for row in rows:
        started = perf_counter()
        result = service.search(row["query"], row.get("user_id"), top_k=10, include_ads=False)
        latencies.append((perf_counter() - started) * 1000)
        predicted = [item.item_id for item in result.items]
        relevant = set(row["relevant_item_ids"])
        ndcg_scores.append(ndcg_at_k(predicted, relevant, 10))
        recall_scores.append(recall_at_k(predicted, relevant, 10))
        mrr_scores.append(mrr_at_k(predicted, relevant, 10))
        empty += not predicted
    count = len(rows)
    return EvaluationReport(
        count,
        sum(ndcg_scores) / max(count, 1),
        sum(recall_scores) / max(count, 1),
        sum(mrr_scores) / max(count, 1),
        empty / max(count, 1),
        percentile(latencies, .95),
    )

