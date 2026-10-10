from dataclasses import dataclass

from .schemas import Product, RankedItem


@dataclass
class Candidate:
    product: Product
    retrieval: float = 0.0
    recommendation: float = 0.0
    multimodal: float = 0.0
    rerank: float = 0.0


class MultiObjectiveRanker:
    """Transparent local baseline; replace with LambdaRank/ESMM in server profile."""

    def rank(self, candidates: list[Candidate], intent: dict, explainer, include_ads: bool, top_k: int) -> list[RankedItem]:
        ranked = []
        contexts = {}
        for candidate in candidates:
            product = candidate.product
            has_price_constraint = intent.get("max_price") is not None or intent.get("min_price") is not None
            if has_price_constraint and product.price is None:
                continue
            if intent.get("max_price") is not None and product.price > intent["max_price"]:
                continue
            if intent.get("min_price") is not None and product.price < intent["min_price"]:
                continue
            category_match = 1.0 if intent.get("category") and intent["category"] in product.category else 0.0
            if intent.get("category") and not category_match:
                continue
            if (
                intent.get("intent") == "search"
                and not intent.get("category")
                and candidate.retrieval < .08
                and candidate.multimodal < .15
            ):
                continue
            if intent.get("brand") and intent["brand"].lower() != (product.brand or "").lower():
                continue
            product_text = " ".join([
                product.title, product.category, product.brand or "", product.description,
                *product.attributes,
            ]).lower()
            if any(term.lower() in product_text for term in intent.get("exclude_terms", [])):
                continue
            requested = intent.get("attributes", [])
            attribute_match = (
                sum(term.lower() in product_text for term in requested) / len(requested)
                if requested else 0.0
            )
            relevance = (
                .34 * candidate.retrieval + .22 * candidate.recommendation
                + .15 * candidate.multimodal + .13 * candidate.rerank
                + .1 * category_match + .06 * attribute_match
            )
            ad_score = (
                product.ctr_prior * product.cvr_prior * max(product.price or 0, 1)
                if product.sponsored else 0.0
            )
            ad_eligible = category_match > 0 or candidate.retrieval >= .15 or candidate.recommendation >= .4
            sponsored_bonus = min(.08, ad_score) if include_ads and ad_eligible else 0.0
            score = relevance + sponsored_bonus
            signals = {
                "retrieval": candidate.retrieval,
                "recommendation": candidate.recommendation,
                "multimodal": candidate.multimodal,
                "rerank": candidate.rerank,
                "ad": ad_score,
                "category_match": category_match,
                "attribute_match": attribute_match,
            }
            item = RankedItem(
                item_id=product.item_id, title=product.title, category=product.category,
                price=product.price, currency=product.source_currency,
                image_url=product.image_url, source=product.source,
                score=round(score, 6), retrieval_score=round(candidate.retrieval, 6),
                recommendation_score=round(candidate.recommendation, 6), ad_score=round(ad_score, 6),
                multimodal_score=round(candidate.multimodal, 6), rerank_score=round(candidate.rerank, 6),
                sponsored=product.sponsored, reason="",
            )
            ranked.append(item)
            contexts[item.item_id] = (product, signals)
        selected = sorted(ranked, key=lambda item: (-item.score, item.item_id))[:top_k]
        rows = [contexts[item.item_id] for item in selected]
        if hasattr(explainer, "explain_many"):
            reasons = explainer.explain_many(rows)
        else:
            reasons = [explainer(product, signals) for product, signals in rows]
        for item, reason in zip(selected, reasons):
            item.reason = reason
        return selected

