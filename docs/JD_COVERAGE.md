# JD coverage and interview evidence

This table distinguishes implemented code from results that still require a formal dataset. Interface-only
work must not be described as an offline quality improvement.

| JD requirement | Status | Verifiable evidence | Remaining experiment |
|---|---|---|---|
| LLM / Query Understanding | Implemented | Validated intent schema, query rewrite, price/category/brand/attribute/negative constraints, domestic OpenAI-compatible API and runtime fallback | Report parse success and rewrite ablation on labeled queries |
| Hybrid Retrieval / Embedding | Implemented | BM25, BGE-compatible encoder, Qdrant adapter and app-side lexical/dense fusion | Build formal catalog index and compare BM25/BGE/hybrid Recall@K |
| Ranking / Rerank | Implemented | BGE cross-encoder adapter followed by transparent business ranker | Record NDCG/MRR/P95 before and after reranking |
| Recommendation | Implemented with fallback | Gorse REST provider, interaction sync, ItemCF fallback and traceable degradation | Time-split interactions; report Recall/NDCG/coverage and cold-start results |
| Multimodal Retrieval | Implemented, data pending | Product image schema, Chinese-CLIP text-to-image/image-to-image provider, image upload API | Add licensed product images and report image/text fusion Recall@K |
| Grounded Generation | Implemented | Explanations receive only product fields and ranking signals; API response is cached and falls back locally | Build attribution set and measure unsupported-claim rate |
| Agent / Tool Calling | Implemented | Bounded search, recommendation and comparison tools with replayable trace; parsed intent is reused to avoid duplicate LLM calls | Report tool selection and argument accuracy |
| Evaluation | Implemented | NDCG@10, Recall@10, MRR@10, empty-result rate and P95 latency | Add dataset-scale ablations and confidence intervals |
| Engineering | Implemented | FastAPI, Streamlit image input, Docker profiles, provider health, circuit breaker, schema validation and tests | Load test concurrency and record CPU/RAM/VRAM |
| Advertising / ESMM | Baseline only | Sponsored score and organic-relevance eligibility guardrail | Train ESMM only if the target JD explicitly requires ads |

## Interview-safe claims

- The repository implements pluggable search, recommendation, reranking, multimodal and LLM providers.
- The bundled sample data proves contracts and end-to-end behavior only.
- Model quality claims require a time/query-split public dataset and a saved experiment report.
- Gorse, Qdrant or API outages are visible in the trace and fall back to local behavior.

## Acceptance gates

1. All unit and API tests pass in the zero-model local profile.
2. Formal search data reports BM25, dense, hybrid and reranked metrics separately.
3. Recommendation data uses a temporal split and excludes future interactions from training.
4. Multimodal evaluation includes text-to-image and image-to-image queries.
5. Every reported improvement includes P95 latency and peak resource usage.
