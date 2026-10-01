import math
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


@dataclass
class EvaluationReport:
    queries: int
    ndcg_at_10: float
    recall_at_10: float
    empty_result_rate: float


def evaluate_queries(service: CommerceService, rows: list[dict]) -> EvaluationReport:
    ndcg_scores, recall_scores, empty = [], [], 0
    for row in rows:
        result = service.search(row["query"], row.get("user_id"), top_k=10, include_ads=False)
        predicted = [item.item_id for item in result.items]
        relevant = set(row["relevant_item_ids"])
        ndcg_scores.append(ndcg_at_k(predicted, relevant, 10))
        recall_scores.append(recall_at_k(predicted, relevant, 10))
        empty += not predicted
    count = len(rows)
    return EvaluationReport(count, sum(ndcg_scores) / max(count, 1),
                            sum(recall_scores) / max(count, 1), empty / max(count, 1))

