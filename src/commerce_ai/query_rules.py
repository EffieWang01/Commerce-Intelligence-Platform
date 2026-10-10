import re
from dataclasses import dataclass, field


TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*|[\u4e00-\u9fff]+", re.I)
MAX_PRICE = re.compile(
    r"(?:\b(?:under|below|less\s+than|up\s+to|max(?:imum)?)\b|预算|低于|不超过|最多)"
    r"\s*(?:usd|rmb|cny|[$¥￥])?\s*(\d+(?:\.\d+)?)",
    re.I,
)
MIN_PRICE = re.compile(
    r"(?:\b(?:over|above|more\s+than|at\s+least|min(?:imum)?)\b|高于|超过|不少于|至少)"
    r"\s*(?:usd|rmb|cny|[$¥￥])?\s*(\d+(?:\.\d+)?)",
    re.I,
)
PRICE_RANGE = re.compile(
    r"(?:usd|rmb|cny|[$¥￥])?\s*(\d+(?:\.\d+)?)\s*(?:-|~|to|到|至)\s*"
    r"(?:usd|rmb|cny|[$¥￥])?\s*(\d+(?:\.\d+)?)",
    re.I,
)
EXCLUSION = re.compile(
    r"(?:\b(?:without|excluding|except|not|no)\b\s+|(?:不要|不含|排除)\s*)"
    r"([a-z0-9-]+|[\u4e00-\u9fff]{1,8})",
    re.I,
)

STOPWORDS = {
    "a", "an", "and", "are", "best", "buy", "for", "from", "in", "is", "me",
    "of", "on", "or", "please", "show", "the", "to", "want", "with",
}
COLORS = {
    "black", "blue", "brown", "gold", "gray", "green", "grey", "orange", "pink",
    "purple", "red", "silver", "white", "yellow", "黑色", "蓝色", "棕色", "金色",
    "灰色", "绿色", "橙色", "粉色", "紫色", "红色", "银色", "白色", "黄色",
}
ATTRIBUTE_TERMS = {
    "bluetooth", "portable", "wireless", "waterproof", "rechargeable", "foldable",
    "lightweight", "mechanical", "noise", "cancelling", "organic", "refurbished",
    "降噪", "无线", "蓝牙", "防水", "便携", "可充电", "折叠", "轻量", "机械", "有机",
}
MATERIALS = {
    "aluminum", "ceramic", "cotton", "glass", "leather", "metal", "plastic", "silicone",
    "stainless", "steel", "wood", "wool", "铝", "陶瓷", "棉", "玻璃", "皮革", "金属",
    "塑料", "硅胶", "不锈钢", "木", "羊毛",
}
ACCESSORY_TERMS = {
    "adapter", "bag", "cable", "case", "charger", "cover", "holder", "mount", "protector",
    "replacement", "screen", "sleeve", "stand", "strap",
}


def _normalized_tokens(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN.findall(text)]
    normalized = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("es"):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        normalized.append(token)
    return normalized


def _present_terms(text: str, tokens: set[str], vocabulary: set[str]) -> list[str]:
    return sorted(term for term in vocabulary if term in tokens or (not term.isascii() and term in text))


@dataclass(frozen=True)
class QueryFeatures:
    tokens: list[str]
    attributes: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    min_price: float | None = None
    max_price: float | None = None

    @property
    def constrained(self) -> bool:
        return bool(
            self.attributes or self.colors or self.materials or self.exclude_terms
            or self.min_price is not None or self.max_price is not None
        )


def parse_query(text: str) -> QueryFeatures:
    """Extract deterministic constraints without inventing missing information."""
    lowered = text.lower()
    tokens = _normalized_tokens(lowered)
    token_set = set(tokens)
    price_range = PRICE_RANGE.search(lowered)
    min_price = float(price_range.group(1)) if price_range else None
    max_price = float(price_range.group(2)) if price_range else None
    if not price_range:
        minimum = MIN_PRICE.search(lowered)
        maximum = MAX_PRICE.search(lowered)
        min_price = float(minimum.group(1)) if minimum else None
        max_price = float(maximum.group(1)) if maximum else None
    excludes = [match.group(1).lower() for match in EXCLUSION.finditer(lowered)]
    return QueryFeatures(
        tokens=tokens,
        attributes=_present_terms(lowered, token_set, ATTRIBUTE_TERMS),
        colors=_present_terms(lowered, token_set, COLORS),
        materials=_present_terms(lowered, token_set, MATERIALS),
        exclude_terms=excludes,
        min_price=min_price,
        max_price=max_price,
    )


def classify_query_product(query: str, product_title: str) -> tuple[str, dict]:
    """Transparent text-only proxy for EcomMMMU A/B/C/D relevance labels."""
    features = parse_query(query)
    query_tokens = set(features.tokens)
    title_tokens = set(_normalized_tokens(product_title))
    matched = query_tokens & title_tokens
    coverage = len(matched) / max(len(query_tokens), 1)

    query_colors = set(features.colors)
    title_colors = title_tokens & COLORS
    color_conflict = bool(query_colors and title_colors and not query_colors & title_colors)
    excluded = bool(set(features.exclude_terms) & title_tokens)
    accessory = bool(title_tokens & ACCESSORY_TERMS)
    query_accessory = bool(query_tokens & ACCESSORY_TERMS)

    if excluded or color_conflict:
        label = "D" if coverage < 0.65 else "B"
    elif coverage >= 0.70:
        label = "A"
    elif accessory and not query_accessory and coverage >= 0.18:
        label = "C"
    elif coverage >= 0.30:
        label = "B"
    else:
        label = "D"
    return label, {
        "coverage": round(coverage, 4),
        "matched_tokens": sorted(matched),
        "color_conflict": color_conflict,
        "excluded_term_present": excluded,
    }
