from functools import lru_cache

from fastapi import FastAPI

from .agent import CommerceAgent
from .config import Settings
from .schemas import AgentRequest, RecommendRequest, SearchRequest
from .service import CommerceService


app = FastAPI(title="Commerce SearchRec Agent", version="0.1.0")


@lru_cache
def service() -> CommerceService:
    return CommerceService(Settings.from_env())


@app.get("/health")
def health():
    return service().health()


@app.post("/v1/search")
def search(request: SearchRequest):
    return service().search(
        request.query, request.user_id, request.top_k, request.include_ads, request.image_data
    )


@app.post("/v1/recommend")
def recommend(request: RecommendRequest):
    return service().recommend(request.user_id, request.top_k, request.include_ads)


@app.post("/v1/agent")
def agent(request: AgentRequest):
    return CommerceAgent(service()).run(request.message, request.user_id, request.image_data)

