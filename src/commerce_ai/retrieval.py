import hashlib
import math
import re
from collections import Counter

import numpy as np

from .schemas import Product


TOKEN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
ANCHOR_TOKEN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.I)
HAN_RUN = re.compile(r"[\u4e00-\u9fff]+")
ANCHOR_STOPWORDS = {
    "best", "catalog", "item", "items", "product", "products", "query", "search",
    "商品", "产品", "查询", "搜索",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    words = TOKEN.findall(text)
    chars = [
        run[index:index + 2]
        for run in HAN_RUN.findall(text)
        for index in range(len(run) - 1)
    ]
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

    def dimension(self) -> int:
        return self.dimensions


class SentenceTransformerEncoder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                          dtype=np.float32)

    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


class HybridRetriever:
    def __init__(self, products: list[Product], encoder=None):
        self.products = products
        self.encoder = encoder or HashEncoder()
        self.documents = [
            " ".join(filter(None, [
                p.title, p.category, p.brand or "", p.description, " ".join(p.attributes),
            ]))
            for p in products
        ]
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

    def _word_coverage(self, query: str) -> float:
        words = set()
        for word in ANCHOR_TOKEN.findall(query.lower()):
            if word in ANCHOR_STOPWORDS:
                continue
            words.add(word)
            if any("\u4e00" <= char <= "\u9fff" for char in word):
                words.update(
                    pair for index in range(len(word) - 1)
                    if (pair := word[index:index + 2]) not in ANCHOR_STOPWORDS
                )
        if not words:
            return 0.0
        known = sum(self.document_frequency.get(word, 0) > 0 for word in words)
        return known / len(words)

    def _english_anchor_coverage(self, query: str) -> tuple[int, int]:
        # 纯数字是价格等数值约束，不是语义锚点，不能计入覆盖率分母，
        # 否则“预算 500 的降噪耳机”会因 500 不在词表而被整体清零。
        words = {
            word.lower() for word in re.findall(r"[a-z0-9]+", query.lower())
            if word.lower() not in ANCHOR_STOPWORDS and not word.isdigit()
        }
        return sum(self.document_frequency.get(word, 0) > 0 for word in words), len(words)

    def search(self, query: str, limit: int = 30) -> list[tuple[Product, float]]:
        query_vector = self.encoder.encode([query])[0]
        dense = self.vectors @ query_vector
        lexical = self._bm25(query)
        scores = .55 * lexical + .45 * dense
        if isinstance(self.encoder, HashEncoder):
            # Hash collisions are not semantic evidence; local mode requires a lexical anchor.
            english_known, english_total = self._english_anchor_coverage(query)
            english_coverage = english_known / english_total if english_total else 1.0
            if self._word_coverage(query) < .15 or english_coverage < .6:
                scores = np.zeros_like(scores)
            else:
                scores = np.where(lexical > 0, scores, 0.0)
        order = np.argsort(-scores)[:limit]
        return [(self.products[index], float(scores[index])) for index in order if scores[index] > 0]


class QdrantHybridRetriever(HybridRetriever):
    """BM25 locally plus dense retrieval from a Qdrant collection."""

    def __init__(self, products: list[Product], encoder, url: str, api_key: str,
                 collection: str, sync: bool = False):
        super().__init__(products, encoder)
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client = QdrantClient(url=url, api_key=api_key or None)
        self.collection = collection
        exists = self.client.collection_exists(collection)
        if exists and sync:
            self.client.delete_collection(collection_name=collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=encoder.dimension(), distance=Distance.COSINE),
            )
            sync = True
        if sync:
            points = [
                PointStruct(
                    id=index,
                    vector=self.vectors[index].tolist(),
                    payload={"item_id": product.item_id},
                )
                for index, product in enumerate(products)
            ]
            self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, limit: int = 30) -> list[tuple[Product, float]]:
        lexical = self._bm25(query)
        hits = self.client.query_points(
            collection_name=self.collection,
            query=self.encoder.encode([query])[0].tolist(),
            limit=limit,
            with_payload=True,
        ).points
        dense = {hit.payload.get("item_id"): max(0.0, float(hit.score)) for hit in hits}
        scores = [
            .55 * float(lexical[index]) + .45 * dense.get(product.item_id, 0.0)
            for index, product in enumerate(self.products)
        ]
        order = np.argsort(-np.asarray(scores))[:limit]
        return [(self.products[index], scores[index]) for index in order if scores[index] > 0]

