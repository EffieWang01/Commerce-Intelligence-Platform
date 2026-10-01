import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    profile: str
    data_dir: Path
    artifacts_dir: Path
    llm_backend: str
    embedding_backend: str
    llm_model: str
    embedding_model: str
    reranker_model: str
    top_k: int
    candidate_k: int
    max_agent_steps: int

    @classmethod
    def from_env(cls) -> "Settings":
        profile = os.getenv("COMMERCE_PROFILE", "local")
        defaults = {
            "local": dict(
                llm_backend="heuristic", embedding_backend="hash",
                llm_model="heuristic-local", embedding_model="hash-384",
                reranker_model="weighted-local", candidate_k=30,
            ),
            "server": dict(
                llm_backend="transformers", embedding_backend="sentence-transformers",
                llm_model="Qwen/Qwen3-4B", embedding_model="BAAI/bge-m3",
                reranker_model="BAAI/bge-reranker-v2-m3", candidate_k=100,
            ),
        }
        if profile not in defaults:
            raise ValueError("COMMERCE_PROFILE must be local or server")
        values = defaults[profile]
        return cls(
            profile=profile,
            data_dir=Path(os.getenv("COMMERCE_DATA_DIR", ROOT / "data" / "sample")),
            artifacts_dir=Path(os.getenv("COMMERCE_ARTIFACTS_DIR", ROOT / "artifacts")),
            llm_backend=os.getenv("COMMERCE_LLM_BACKEND", values["llm_backend"]),
            embedding_backend=os.getenv("COMMERCE_EMBEDDING_BACKEND", values["embedding_backend"]),
            llm_model=os.getenv("COMMERCE_LLM_MODEL", values["llm_model"]),
            embedding_model=os.getenv("COMMERCE_EMBEDDING_MODEL", values["embedding_model"]),
            reranker_model=os.getenv("COMMERCE_RERANKER_MODEL", values["reranker_model"]),
            top_k=int(os.getenv("COMMERCE_TOP_K", "10")),
            candidate_k=int(os.getenv("COMMERCE_CANDIDATE_K", str(values["candidate_k"]))),
            max_agent_steps=int(os.getenv("COMMERCE_MAX_AGENT_STEPS", "4")),
        )

