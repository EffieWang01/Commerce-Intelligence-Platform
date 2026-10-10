import numpy as np


def _product_text(product) -> str:
    return " ".join(filter(None, [
        product.title, product.category, product.brand or "", product.description,
        " ".join(product.attributes),
    ]))


class NoopReranker:
    provider_name = "weighted"

    def score(self, query: str, products: list) -> dict[str, float]:
        return {}


class CrossEncoderReranker:
    provider_name = "cross-encoder"

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def score(self, query: str, products: list) -> dict[str, float]:
        if not products:
            return {}
        raw = np.asarray(self.model.predict([(query, _product_text(p)) for p in products]), dtype=float)
        if raw.max() > raw.min():
            raw = (raw - raw.min()) / (raw.max() - raw.min())
        else:
            raw = np.full_like(raw, .5)
        return {product.item_id: float(value) for product, value in zip(products, raw)}
