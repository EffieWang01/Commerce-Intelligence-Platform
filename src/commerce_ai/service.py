from time import perf_counter

from .config import Settings
from .data import CommerceData
from .llm import HeuristicLLM, OpenAICompatibleLLM, ResilientLLM, TransformersLLM
from .multimodal import ChineseClipMultimodalRetriever, DisabledMultimodalRetriever
from .ranking import Candidate, MultiObjectiveRanker
from .recommendation import GorseRecommender, ItemCF
from .reranking import CrossEncoderReranker, NoopReranker
from .retrieval import HybridRetriever, QdrantHybridRetriever, SentenceTransformerEncoder
from .schemas import ResultSet


class CommerceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data = CommerceData(settings.data_dir)
        self.init_warnings: list[str] = []
        self.llm = self._build_llm()
        self.encoder = self._build_encoder()
        self.retriever = self._build_retriever()
        self.recommender = self._build_recommender()
        self.reranker = self._build_reranker()
        self.multimodal = self._build_multimodal()
        self.ranker = MultiObjectiveRanker()

    def _fallback(self, component: str, exc: Exception) -> None:
        self.init_warnings.append(f"{component}:{type(exc).__name__}")

    def _build_llm(self):
        fallback = HeuristicLLM([product.category for product in self.data.products])
        try:
            if self.settings.llm_backend == "openai-compatible":
                return ResilientLLM(OpenAICompatibleLLM(
                    self.settings.llm_model, self.settings.llm_base_url, self.settings.llm_api_key
                ), fallback)
            if self.settings.llm_backend == "transformers":
                return ResilientLLM(
                    TransformersLLM(self.settings.llm_model), fallback
                )
        except (ImportError, OSError, ValueError) as exc:
            self._fallback("llm", exc)
        return fallback

    def _build_encoder(self):
        if self.settings.embedding_backend == "sentence-transformers":
            try:
                return SentenceTransformerEncoder(self.settings.embedding_model)
            except (ImportError, OSError) as exc:
                self._fallback("embedding", exc)
        return None

    def _build_retriever(self):
        if self.settings.vector_backend == "qdrant" and self.encoder is not None:
            try:
                return QdrantHybridRetriever(
                    self.data.products, self.encoder, self.settings.qdrant_url,
                    self.settings.qdrant_api_key, self.settings.qdrant_collection,
                    self.settings.sync_external,
                )
            except Exception as exc:
                # External clients use version-specific transport exception types.
                self._fallback("qdrant", exc)
        return HybridRetriever(self.data.products, self.encoder)

    def _build_recommender(self):
        if self.settings.recommendation_backend == "gorse":
            try:
                return GorseRecommender(
                    self.data.products, self.data.interactions, self.settings.gorse_url,
                    self.settings.gorse_api_key, self.settings.sync_external,
                )
            except Exception as exc:
                self._fallback("gorse", exc)
        return ItemCF(self.data.products, self.data.interactions)

    def _build_reranker(self):
        if self.settings.reranker_backend == "cross-encoder":
            try:
                return CrossEncoderReranker(self.settings.reranker_model)
            except Exception as exc:
                self._fallback("reranker", exc)
        return NoopReranker()

    def _build_multimodal(self):
        if self.settings.multimodal_backend == "chinese-clip":
            try:
                return ChineseClipMultimodalRetriever(
                    self.data.products,
                    self.settings.multimodal_model,
                    self.settings.qdrant_url if self.settings.vector_backend == "qdrant" else "",
                    self.settings.qdrant_api_key,
                    self.settings.qdrant_image_collection,
                    self.settings.sync_external,
                )
            except Exception as exc:
                self._fallback("multimodal", exc)
        return DisabledMultimodalRetriever()

    def health(self) -> dict:
        return {
            "status": "degraded" if self.init_warnings else "ok",
            "profile": self.settings.profile,
            "providers": {
                "llm": self.llm.provider_name,
                "retrieval": type(self.retriever).__name__,
                "recommendation": self.recommender.provider_name,
                "reranker": self.reranker.provider_name,
                "multimodal": self.multimodal.provider_name,
            },
            "warnings": self.init_warnings,
            "catalog": {
                "products": len(self.data.products),
                "categories": sorted({product.category for product in self.data.products}),
            },
        }

    def _merge(self, retrieval, recommendation, multimodal=()) -> list[Candidate]:
        merged: dict[str, Candidate] = {}
        for product, score in retrieval:
            merged.setdefault(product.item_id, Candidate(product)).retrieval = score
        for product, score in recommendation:
            merged.setdefault(product.item_id, Candidate(product)).recommendation = score
        for product, score in multimodal:
            merged.setdefault(product.item_id, Candidate(product)).multimodal = score
        return list(merged.values())

    def _apply_reranker(self, candidates: list[Candidate], query: str) -> None:
        scores = self.reranker.score(query, [candidate.product for candidate in candidates])
        for candidate in candidates:
            candidate.rerank = scores.get(candidate.product.item_id, 0.0)

    def search(self, query: str, user_id: str | None = None, top_k: int = 10,
               include_ads: bool = True, image_data: str | None = None,
               parsed_intent: dict | None = None) -> ResultSet:
        started = perf_counter()
        vision_text = self.llm.describe_image(image_data) if image_data else ""
        intent = parsed_intent or self.llm.understand(" ".join(filter(None, [query, vision_text])))
        if parsed_intent and vision_text:
            intent = dict(intent)
            intent["rewritten_query"] = " ".join(filter(None, [
                intent.get("rewritten_query") or intent.get("query"), vision_text,
            ]))
        query_text = intent.get("rewritten_query") or intent["query"]
        understand_ms = (perf_counter() - started) * 1000

        stage = perf_counter()
        retrieval = self.retriever.search(query_text, self.settings.candidate_k)
        recommendation = self.recommender.recommend(user_id, self.settings.candidate_k) if user_id else []
        multimodal = self.multimodal.search(query_text, image_data, self.settings.candidate_k)
        candidates = self._merge(retrieval, recommendation, multimodal)
        recall_ms = (perf_counter() - stage) * 1000

        stage = perf_counter()
        self._apply_reranker(candidates, query_text)
        items = self.ranker.rank(candidates, intent, self.llm, include_ads, top_k)
        rank_ms = (perf_counter() - stage) * 1000
        gorse_error = getattr(self.recommender, "last_error", None)
        llm_error = getattr(self.llm, "last_error", None)
        return ResultSet(intent=intent, items=items, trace=[
            {"stage": "understand", "provider": self.llm.provider_name, "output": intent,
             "vision_text": vision_text or None, "fallback": llm_error,
             "latency_ms": round(understand_ms, 2)},
            {"stage": "retrieve", "provider": type(self.retriever).__name__,
             "query": query_text, "candidates": len(retrieval)},
            {"stage": "personalize", "provider": self.recommender.provider_name,
             "candidates": len(recommendation), "fallback": gorse_error},
            {"stage": "multimodal", "provider": self.multimodal.provider_name,
             "candidates": len(multimodal)},
            {"stage": "rank", "provider": self.reranker.provider_name,
             "candidates": len(candidates), "returned": len(items), "latency_ms": round(rank_ms, 2)},
            {"stage": "latency", "recall_ms": round(recall_ms, 2),
             "total_ms": round((perf_counter() - started) * 1000, 2)},
        ])

    def recommend(self, user_id: str, top_k: int = 10, include_ads: bool = True) -> ResultSet:
        history = self.data.history(user_id)
        query = "推荐 " + " ".join(product.category + " " + product.title for product in history[-5:])
        result = self.search(query, user_id, top_k, include_ads)
        result.intent["intent"] = "recommend"
        result.trace.insert(0, {"stage": "history", "items": [p.item_id for p in history]})
        return result

    def compare(self, item_ids: list[str]) -> dict:
        products = [self.data.products_by_id[item_id] for item_id in item_ids if item_id in self.data.products_by_id]
        def price_text(product) -> str:
            if product.price is None:
                return "价格未知"
            symbol = {"USD": "$", "CNY": "¥"}.get(product.source_currency or "CNY", "")
            return f"{symbol}{product.price:.2f}"
        return {
            "items": [product.model_dump() for product in products],
            "summary": "；".join(f"{p.title}：{p.category}，{price_text(p)}" for p in products),
        }
