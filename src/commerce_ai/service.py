from .config import Settings
from .data import CommerceData
from .llm import HeuristicLLM, TransformersLLM
from .ranking import Candidate, MultiObjectiveRanker
from .recommendation import ItemCF
from .retrieval import HybridRetriever, SentenceTransformerEncoder
from .schemas import ResultSet


class CommerceService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data = CommerceData(settings.data_dir)
        self.llm = TransformersLLM(settings.llm_model) if settings.llm_backend == "transformers" else HeuristicLLM()
        encoder = SentenceTransformerEncoder(settings.embedding_model) if settings.embedding_backend == "sentence-transformers" else None
        self.retriever = HybridRetriever(self.data.products, encoder)
        self.recommender = ItemCF(self.data.products, self.data.interactions)
        self.ranker = MultiObjectiveRanker()

    def _merge(self, retrieval, recommendation) -> list[Candidate]:
        merged: dict[str, Candidate] = {}
        for product, score in retrieval:
            merged.setdefault(product.item_id, Candidate(product)).retrieval = score
        for product, score in recommendation:
            merged.setdefault(product.item_id, Candidate(product)).recommendation = score
        return list(merged.values())

    def search(self, query: str, user_id: str | None = None, top_k: int = 10,
               include_ads: bool = True) -> ResultSet:
        intent = self.llm.understand(query)
        retrieval = self.retriever.search(intent["query"], self.settings.candidate_k)
        recommendation = self.recommender.recommend(user_id, self.settings.candidate_k) if user_id else []
        candidates = self._merge(retrieval, recommendation)
        items = self.ranker.rank(candidates, intent, self.llm.explain, include_ads, top_k)
        return ResultSet(intent=intent, items=items, trace=[
            {"stage": "understand", "output": intent},
            {"stage": "retrieve", "candidates": len(retrieval)},
            {"stage": "personalize", "candidates": len(recommendation)},
            {"stage": "rank", "returned": len(items)},
        ])

    def recommend(self, user_id: str, top_k: int = 10, include_ads: bool = True) -> ResultSet:
        history = self.data.history(user_id)
        query = "推荐 " + " ".join(product.category + " " + product.title for product in history[-5:])
        intent = self.llm.understand(query)
        intent["intent"] = "recommend"
        retrieval = self.retriever.search(query, self.settings.candidate_k)
        recommendation = self.recommender.recommend(user_id, self.settings.candidate_k)
        candidates = self._merge(retrieval, recommendation)
        items = self.ranker.rank(candidates, intent, self.llm.explain, include_ads, top_k)
        return ResultSet(intent=intent, items=items, trace=[
            {"stage": "history", "items": [p.item_id for p in history]},
            {"stage": "recall", "retrieval": len(retrieval), "itemcf": len(recommendation)},
            {"stage": "rank", "returned": len(items)},
        ])

    def compare(self, item_ids: list[str]) -> dict:
        products = [self.data.products_by_id[item_id] for item_id in item_ids if item_id in self.data.products_by_id]
        return {"items": [product.model_dump() for product in products],
                "summary": "；".join(f"{p.title}：{p.category}，{p.price:.0f}元" for p in products)}

