import hashlib
import math
import re
from collections import Counter

import numpy as np

from .schemas import Product


TOKEN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    words = TOKEN.findall(text)
    chars = [text[index:index + 2] for index in range(max(0, len(text) - 1)) if not text[index:index + 2].isspace()]
    return words + chars


class HashEncoder:
    """Deterministic CPU fallback. Server profile replaces this with BGE."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest, "little") % self.dimensions
                matrix[row, index] += 1
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix


class SentenceTransformerEncoder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)


class HybridRetriever:
    def __init__(self, products: list[Product], encoder=None):
        self.products = products
        self.encoder = encoder or HashEncoder()
        self.documents = [f"{p.title} {p.category} {p.description}" for p in products]
        self.tokens = [tokenize(text) for text in self.documents]
        self.document_frequency = Counter(token for tokens in self.tokens for token in set(tokens))
        self.avg_length = sum(map(len, self.tokens)) / max(1, len(self.tokens))
        self.vectors = self.encoder.encode(self.documents)

    def _bm25(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.products), dtype=np.float32)
        for index, tokens in enumerate(self.tokens):
            counts = Counter(tokens)
            for token in tokenize(query):
                df = self.document_frequency.get(token, 0)
                if not df:
                    continue
                idf = math.log(1 + (len(self.products) - df + .5) / (df + .5))
                tf = counts[token]
                denom = tf + 1.5 * (1 - .75 + .75 * len(tokens) / max(1, self.avg_length))
                scores[index] += idf * tf * 2.5 / max(denom, 1e-6)
        if scores.max() > 0:
            scores /= scores.max()
        return scores

    def search(self, query: str, limit: int = 30) -> list[tuple[Product, float]]:
        query_vector = self.encoder.encode([query])[0]
        dense = self.vectors @ query_vector
        scores = .55 * self._bm25(query) + .45 * dense
        order = np.argsort(-scores)[:limit]
        return [(self.products[index], float(scores[index])) for index in order if scores[index] > 0]

