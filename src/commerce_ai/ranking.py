from dataclasses import dataclass
from pathlib import Path

from .schemas import Product, RankedItem
from .retrieval import tokenize


@dataclass
class Candidate:
    product: Product
    retrieval: float = 0.0
    recommendation: float = 0.0
    multimodal: float = 0.0
    rerank: float = 0.0


FEATURE_NAMES = [
    "all_coverage", "title_coverage", "description_coverage", "category_coverage",
    "attribute_coverage", "brand_coverage", "title_length", "description_length",
    "has_price", "price_scaled", "title_phrase_match", "query_terms",
]


def load_ranking_model(path: Path):
    from lightgbm import Booster

    # Python handles Unicode paths that the native Windows file API may reject.
    model = Booster(model_str=path.read_text(encoding="utf-8"))
    if model.feature_name() != FEATURE_NAMES:
        raise ValueError("Ranking model feature schema does not match this service")
    return model


def ranking_features(query: str, product: Product) -> list[float]:
    query_terms = set(tokenize(query))
    fields = [set(tokenize(text)) for text in (
        product.title, product.description, product.category,
        " ".join(product.attributes), product.brand or "",
    )]
    term_count = max(1, len(query_terms))
    return [
        len(query_terms & set.union(*fields)) / term_count,
        *(len(query_terms & field) / term_count for field in fields),
        min(len(product.title), 500) / 500,
        min(len(product.description), 4000) / 4000,
        float(product.price is not None),
        min(product.price or 0.0, 10000.0) / 10000,
        float(bool(query.strip()) and query.strip().casefold() in product.title.casefold()),
        min(len(query_terms), 100) / 100,
    ]


class MultiObjectiveRanker:
    """Transparent local baseline; replace with LambdaRank/ESMM in server profile."""

    def __init__(self, model=None):
        self.model = model

    def rank(self, candidates: list[Candidate], intent: dict, explainer, include_ads: bool,
             top_k: int, query: str = "") -> list[RankedItem]:
        ranked = []
        contexts = {}
        model_features = []
        model_offsets = []
        ad_bonuses = []
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
            if self.model is not None:
                score = sponsored_bonus
                model_offsets.append(len(ranked))
                model_features.append(ranking_features(query, product))
                ad_bonuses.append(sponsored_bonus)
            else:
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
        if self.model is not None and model_features:
            predictions = self.model.predict(model_features)
            for offset, prediction, bonus in zip(model_offsets, predictions, ad_bonuses):
                ranked[offset].score = round(float(prediction) + bonus, 6)
        selected = sorted(ranked, key=lambda item: (-item.score, item.item_id))[:top_k]
        rows = [contexts[item.item_id] for item in selected]
        if hasattr(explainer, "explain_many"):
            reasons = explainer.explain_many(rows)
        else:
            reasons = [explainer(product, signals) for product, signals in rows]
        for item, reason in zip(selected, reasons):
            item.reason = reason
        return selected

