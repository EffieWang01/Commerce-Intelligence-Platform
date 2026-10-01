# JD coverage and interview evidence

| JD requirement | Project evidence |
|---|---|
| LLM / Query Understanding | Structured intent contract, price/category extraction, swappable Qwen provider |
| Retrieval / Embedding | BM25 and dense recall behind one retriever interface |
| Ranking / Rerank | Multi-signal ranker with explicit retrieval, recommendation and ad scores |
| Recommendation | ItemCF baseline and user-history recall; extension point for two-tower/OpenP5 |
| Advertising | Sponsored candidate objective using CTR × CVR × value; extension point for ESMM |
| RAG | Explanations use product metadata and expose their evidence signals |
| Agent / Tool Calling | Bounded search, recommendation and comparison tools with replayable trace |
| Evaluation | NDCG, Recall, empty-result rate and API-level regression tests |
| Engineering | FastAPI, Streamlit, Docker, profiles, health endpoint and schema validation |

The interview story should use measured improvements from the server dataset. Do not report the bundled
sample metrics as model quality: the sample exists only to prove the whole system runs.

