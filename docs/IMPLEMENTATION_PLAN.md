# Low-resource implementation plan

## Implemented composition

| Stage | Upstream component | Integration boundary |
|---|---|---|
| Query understanding and visual attributes | Qwen-compatible VLM | `OpenAICompatibleLLM` |
| Lexical and semantic recall | BM25, BGE-M3, Qdrant | `HybridRetriever` / `QdrantHybridRetriever` |
| Visual recall | Chinese-CLIP | `ChineseClipMultimodalRetriever` |
| Personalization | Gorse | `GorseRecommender` with ItemCF fallback |
| Semantic reranking | BGE reranker | `CrossEncoderReranker` |
| Business ranking | Project-owned code | `MultiObjectiveRanker` |

## Delivery gates

- Gate 1, completed: provider contracts, fallbacks, constraints, trace and local tests.
- Gate 2, requires formal data: build Qdrant/Gorse indexes and save baseline metrics.
- Gate 3, requires product images: evaluate text-to-image and image-to-image retrieval.
- Gate 4, requires labeled explanations: measure unsupported claims and citation accuracy.
- Gate 5, optional by JD: train ESMM or a two-tower recommender instead of adding more orchestration frameworks.

## Experiment matrix

| Experiment | Control | Treatment | Primary metrics |
|---|---|---|---|
| Semantic recall | BM25 | BM25 + BGE-M3 | Recall@10, NDCG@10, P95 |
| Reranking | Hybrid recall | Hybrid + BGE reranker | NDCG@10, MRR@10, P95 |
| Query rewrite | Original query | Validated LLM rewrite | Zero-result rate, Recall@10 |
| Personalization | Popularity/ItemCF | Gorse | Recall@10, coverage, novelty |
| Multimodal | Text only | Text + Chinese-CLIP | Image Recall@10, NDCG@10 |
| Explanation | Template | Grounded VLM | Unsupported-claim rate, latency, cost |
