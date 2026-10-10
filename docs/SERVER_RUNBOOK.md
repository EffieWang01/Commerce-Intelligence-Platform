# Server runbook

## Profiles

- `local`: no model download, GPU or external service. Used for tests and contract validation.
- `server`: BGE, Qdrant, Gorse, cross-encoder, Chinese-CLIP and a domestic OpenAI-compatible VLM.
- Every external provider degrades to a local implementation and exposes the reason in `/health` or response trace.

## Before using the 4090

1. Run `pytest -q`.
2. Run `python scripts/validate_data.py <data-dir>`.
3. Split queries by query identity and interactions by time.
4. Start with 100k interactions and record index size, RAM, VRAM and P95.
5. Never report bundled sample metrics as model quality.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[server,multimodal,test]'
cp .env.server.example .env
```

Set `COMMERCE_LLM_API_KEY`. DashScope is the default, but any provider with an OpenAI-compatible
`/chat/completions` endpoint can be selected through `COMMERCE_LLM_BASE_URL` and `COMMERCE_LLM_MODEL`.

## Start dependencies

Development containers:

```bash
docker compose --profile full up -d qdrant gorse
```

The bundled Gorse container uses its official playground mode for integration testing. For formal
experiments, deploy Gorse with persistent PostgreSQL/Redis according to its upstream documentation and
point `COMMERCE_GORSE_URL` at that service.

On the first formal-data startup set `COMMERCE_SYNC_EXTERNAL=true`. After Qdrant and Gorse are populated,
set it back to `false` to avoid re-uploading data on every API restart.

```bash
set -a && source .env && set +a
pytest -q
python -m commerce_ai.cli evaluate
uvicorn commerce_ai.api:app --host 0.0.0.0 --port 8008
```

## Single RTX 4090 allocation

| Component | Placement | Policy |
|---|---|---|
| Qdrant and Gorse | CPU/RAM | Always on |
| BGE-M3 | GPU or CPU | Batch catalog embeddings; online query encoding only |
| BGE reranker | GPU | Rerank top 30-100 only |
| Chinese-CLIP | GPU | Offline image embeddings; online query/image encoding only |
| Qwen-VL | Domestic API by default | Avoid competing with retrieval models for VRAM |
| Local Qwen3-VL 4B/8B | vLLM, optional | Use quantization and do not run batch indexing concurrently |

## Acceptance run

1. `/health` reports the intended providers and no unexpected warning.
2. Search traces contain text, personalization, multimodal and rerank candidate counts.
3. Gorse outage produces an ItemCF fallback marker without failing the request.
4. LLM timeout produces a heuristic fallback marker without failing the request.
5. Save search/recommendation/multimodal ablation tables with P95 latency.
