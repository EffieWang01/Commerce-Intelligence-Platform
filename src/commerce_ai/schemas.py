from pydantic import BaseModel, Field


class Product(BaseModel):
    item_id: str
    title: str
    category: str
    description: str
    price: float = Field(ge=0)
    sponsored: bool = False
    ctr_prior: float = Field(default=0.03, ge=0, le=1)
    cvr_prior: float = Field(default=0.01, ge=0, le=1)


class Interaction(BaseModel):
    user_id: str
    item_id: str
    event: str
    timestamp: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    user_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    include_ads: bool = True


class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = Field(default=10, ge=1, le=50)
    include_ads: bool = True


class AgentRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str | None = None


class RankedItem(BaseModel):
    item_id: str
    title: str
    category: str
    price: float
    score: float
    retrieval_score: float
    recommendation_score: float
    ad_score: float
    sponsored: bool
    reason: str


class ResultSet(BaseModel):
    intent: dict
    items: list[RankedItem]
    trace: list[dict] = []

