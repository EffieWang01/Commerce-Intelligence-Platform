# Server44 training workspace

All server-side files for this project live in `/home/moxu/commerce-searchrec`.
The system Python, shared CUDA installation, shell startup files, and other projects are not modified.

| Path inside workspace | Contents |
| --- | --- |
| `src`, `scripts`, `tests`, `docs` | Source and reproducible commands |
| `python`, `tools`, `.venv` | Private Python runtime, uv, dependencies |
| `cache`, `tmp` | Package downloads, model caches, temporary files |
| `data/demo` | Existing 50,000 complete product records |
| `data/raw/esci` | Official Amazon ESCI relevance judgments |
| `data/processed/esci` | ASIN-matched labels and provenance manifest |
| `data/processed/esci_text` | Separate official text-only ranking experiment |
| `artifacts/lambdamart` | Trained model and evaluation report |
| `logs` | Environment, import, training and verification logs |

Connect and initialize:

```bash
ssh server44-moxu
cd /home/moxu/commerce-searchrec
source scripts/server_env.sh
bash scripts/bootstrap_server.sh
python scripts/import_esci_labels.py
python scripts/download_esci_products.py
python scripts/prepare_esci_training.py
python scripts/train_lambdamart.py --data-dir data/processed/esci_text \
    --labels data/processed/esci_text/labels.jsonl --threads 4
```

The importer uses US examples from the official ESCI ranking task and joins only exact ASINs from
the existing catalog. It retains query groups with at least two matched products and two relevance
levels. Official test queries stay held out; 20% of train queries become validation queries by a
deterministic query hash. Results describe this ASIN-intersection subset, not the full benchmark.

Grades are I=0, C=1, S=2, E=3; NDCG gains are 0, 0.01, 0.1, 1. Validation chooses the number of
trees via early stopping. Test labels are used only after fitting. Baseline evaluation reranks the
same judged candidates using the existing rule ranker, with BM25/hash statistics built within each
candidate group. It does not measure retrieval recall over the 50,000-product index.

The catalog join produced only 405 overlapping judgments and six informative queries. This is
insufficient for training. The separate `esci_text` experiment samples up to 3,000 train, 500
validation and 500 test queries from the official US ranking task, preserving the official test
split. It reads the matching product text from the official ESCI product table. Products must have
an original title and description or bullet points. Price, category and image fields are absent
from the original ESCI table and remain absent in this separate text-only experiment. These
records are never inserted into `data/demo`.

Downloads are verified against the SHA-256 identifiers in the official Git LFS pointer files.
If direct GitHub downloads are slow, the download scripts accept a mirror URL while enforcing
the same official checksum. Dependencies can use `COMMERCE_PYPI_INDEX` in the bootstrap script.

This first LambdaMART model uses text and structured metadata. Image URLs remain in the product
catalog, but this training run does not extract visual embeddings and is not a multimodal result.

The completed run retained 2,898 train queries (54,336 pairs), 482 validation queries (9,213
pairs), and 486 test queries (9,191 pairs). Validation selected 63 trees. Test NDCG@10 was 0.7237
versus 0.7013 for the rule baseline. Detailed scope and results are in
`docs/RANKING_BASELINE_RESULTS_ZH.md`. All 19 regression tests and the trained-model API smoke
check passed against the complete 50,000-product demo catalog.

To load a trained model for a service process:

```bash
source scripts/server_env.sh
export COMMERCE_RANKING_MODEL_PATH="$COMMERCE_ROOT/artifacts/lambdamart/model.txt"
python -m commerce_ai.cli smoke --query 'wireless headphones'
```

Unset `COMMERCE_RANKING_MODEL_PATH` to use the rule baseline. No background training service or
scheduled job is installed. After project processes have stopped, this single directory contains
the files needed for cleanup; no project-specific environment or model cache is stored elsewhere.
