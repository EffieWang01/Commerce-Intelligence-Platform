# Server runbook

## Before renting a server

1. Run `pytest -q` locally.
2. Run `python -m commerce_ai.cli evaluate` locally.
3. Convert the chosen public dataset to `products.jsonl`, `interactions.jsonl`, and `queries.jsonl`.
4. Run `python scripts/validate_data.py <data-dir>`.
5. Upload only validated data and this repository.

## Suggested first server

- 1 NVIDIA GPU with 16–24 GB VRAM
- 8–16 vCPU
- 64 GB RAM
- 200 GB SSD
- Ubuntu 22.04 or newer

Install the server extras in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[server,test]'
cp .env.server.example .env
set -a && source .env && set +a
pytest -q
python -m commerce_ai.cli smoke
uvicorn commerce_ai.api:app --host 0.0.0.0 --port 8008
```

Start with 100k interactions. Increase to 1M only after recording latency, peak RAM/VRAM and index size.
Do not run LLM inference, embedding construction and model training concurrently on a 16 GB card.

