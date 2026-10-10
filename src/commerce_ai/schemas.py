import base64
import binascii
import re
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field


DATA_URL = re.compile(r"^data:image/(?:jpeg|png|webp);base64,", re.I)


def validate_image_data(value: str) -> str:
    encoded = DATA_URL.sub("", value, count=1) if value.startswith("data:") else value
    if value.startswith("data:") and encoded == value:
        raise ValueError("image_data must be a JPEG, PNG or WebP base64 data URL")
    try:
        base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_data is not valid base64") from exc
    return value


ImageData = Annotated[str, AfterValidator(validate_image_data)]


class SearchIntent(BaseModel):
    intent: str = "search"
    query: str
    rewritten_query: str | None = None
    max_price: float | None = Field(default=None, ge=0)
    min_price: float | None = Field(default=None, ge=0)
    category: str | None = None
    brand: str | None = None
    attributes: list[str] = Field(default_factory=list)
    exclude_terms: list[str] = Field(default_factory=list)


class Product(BaseModel):
    item_id: str
    title: str
    category: str
    description: str
    price: float | None = Field(default=None, ge=0)
    brand: str | None = None
    image_url: str | None = None
    attributes: list[str] = Field(default_factory=list)
    source: str | None = None
    source_id: str | None = None
    source_price: float | None = Field(default=None, ge=0)
    source_currency: str | None = None
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
    image_data: ImageData | None = Field(
        default=None, max_length=8_000_000, description="Optional data URL or base64 encoded image"
    )
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
    image_data: ImageData | None = Field(default=None, max_length=8_000_000)


class RankedItem(BaseModel):
    item_id: str
    title: str
    category: str
    price: float | None
    currency: str | None = None
    image_url: str | None = None
    source: str | None = None
    score: float
    retrieval_score: float
    recommendation_score: float
    multimodal_score: float = 0.0
    rerank_score: float = 0.0
    ad_score: float
    sponsored: bool
    reason: str


class ResultSet(BaseModel):
    intent: dict
    items: list[RankedItem]
    trace: list[dict] = Field(default_factory=list)

