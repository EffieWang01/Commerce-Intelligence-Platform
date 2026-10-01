from dataclasses import dataclass

from .schemas import Product, RankedItem


@dataclass
class Candidate:
    product: Product
    retrieval: float = 0.0
    recommendation: float = 0.0


class MultiObjectiveRanker:
    """Transparent local baseline; replace with LambdaRank/ESMM in server profile."""

    def rank(self, candidates: list[Candidate], intent: dict, explain, include_ads: bool, top_k: int) -> list[RankedItem]:
        ranked = []
        for candidate in candidates:
            product = candidate.product
            if intent.get("max_price") is not None and product.price > intent["max_price"]:
                continue
            category_match = 1.0 if intent.get("category") and intent["category"] in product.category else 0.0
            if intent.get("category") and not category_match:
                continue
            relevance = .5 * candidate.retrieval + .25 * candidate.recommendation + .1 * category_match
            ad_score = product.ctr_prior * product.cvr_prior * max(product.price, 1) if product.sponsored else 0.0
            ad_eligible = category_match > 0 or candidate.retrieval >= .15 or candidate.recommendation >= .4
            sponsored_bonus = min(.08, ad_score) if include_ads and ad_eligible else 0.0
            score = relevance + sponsored_bonus
            signals = {"retrieval": candidate.retrieval, "recommendation": candidate.recommendation,
                       "ad": ad_score, "category_match": category_match}
            ranked.append(RankedItem(
                item_id=product.item_id, title=product.title, category=product.category,
                price=product.price, score=round(score, 6), retrieval_score=round(candidate.retrieval, 6),
                recommendation_score=round(candidate.recommendation, 6), ad_score=round(ad_score, 6),
                sponsored=product.sponsored, reason=explain(product, signals),
            ))
        return sorted(ranked, key=lambda item: (-item.score, item.item_id))[:top_k]

