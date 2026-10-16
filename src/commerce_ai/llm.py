import json
import re
from functools import lru_cache

import requests

from .query_rules import parse_query
from .schemas import Product, SearchIntent


PRICE = re.compile(r"(?:预算|价格|低于|不超过|under)\s*[¥￥$]?\s*(\d+(?:\.\d+)?)", re.I)
JSON_OBJECT = re.compile(r"\{.*\}", re.S)
CATEGORY_ALIASES = {
    "咖啡杯": "厨房餐饮", "餐具": "厨房餐饮", "厨具": "厨房餐饮", "杯子": "厨房餐饮",
    "办公椅": "家具", "椅子": "家具", "桌子": "家具", "沙发": "家具", "柜子": "家具",
    "电视机": "电子数码", "电视": "电子数码", "电脑": "电子数码", "手机": "电子数码",
    "相机": "电子数码", "音箱": "电子数码", "打印机": "办公文具",
    "鞋子": "鞋靴", "运动鞋": "鞋靴", "靴子": "鞋靴", "拖鞋": "鞋靴",
    "衣服": "服装", "衬衫": "服装", "裙子": "服装", "裤子": "服装",
    "行李箱": "箱包", "背包": "箱包", "钱包": "箱包",
    "化妆品": "美妆个护", "护肤": "美妆个护", "香水": "美妆个护",
    "帐篷": "运动户外", "露营": "运动户外", "健身": "运动户外", "瑜伽": "运动户外",
    "床单": "床品家纺", "被子": "床品家纺", "枕头": "床品家纺", "毛巾": "床品家纺",
    "浴室": "卫浴清洁", "淋浴": "卫浴清洁", "马桶": "卫浴清洁",
    "台灯": "灯具照明", "吊灯": "灯具照明", "灯泡": "灯具照明",
    "宠物": "宠物用品", "猫用品": "宠物用品", "狗用品": "宠物用品",
    "车载": "汽车用品", "汽车配件": "汽车用品", "电钻": "工具五金",
    "花盆": "园艺", "园林": "园艺", "文具": "办公文具",
}


def _validated_intent(payload: dict, original: str) -> dict:
    payload.setdefault("query", original.strip())
    intent = SearchIntent.model_validate(payload)
    if not intent.rewritten_query:
        intent.rewritten_query = intent.query
    return intent.model_dump()


def _json_object(text: str) -> dict:
    match = JSON_OBJECT.search(text)
    if not match:
        raise ValueError("model response does not contain a JSON object")
    return json.loads(match.group(0))


class HeuristicLLM:
    """Offline contract-compatible fallback used for tests and local smoke runs."""

    provider_name = "heuristic"

    def __init__(self, categories: list[str] | None = None):
        defaults = ["耳机", "键盘", "显示器", "咖啡", "运动"]
        self.categories = sorted(set(categories or defaults), key=len, reverse=True)

    def understand(self, text: str) -> dict:
        rules = parse_query(text)
        price = PRICE.search(text)
        intent = "recommend" if any(word in text for word in ("推荐", "适合我", "猜你喜欢")) else "search"
        if any(word in text for word in ("对比", "比较", "区别")):
            intent = "compare"
        alias_category = next((
            category for alias, category in sorted(CATEGORY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)
            if alias in text and category in self.categories
        ), None)
        category = alias_category or next((name for name in self.categories if name in text), None)
        attributes = rules.attributes or [
            word for word in ("降噪", "无线", "蓝牙", "防水", "机械", "便携") if word in text
        ]
        rewritten_query = text.strip()
        if category and category not in rewritten_query:
            rewritten_query = f"{rewritten_query} {category}"
        return _validated_intent({
            "intent": intent,
            "query": text.strip(),
            # Preserve discriminative tokens such as 4K and 跑步 in local mode.
            "rewritten_query": rewritten_query,
            "max_price": rules.max_price if rules.max_price is not None else (
                float(price.group(1)) if price else None
            ),
            "min_price": rules.min_price,
            "category": category,
            "attributes": attributes,
            "exclude_terms": rules.exclude_terms,
        }, text)

    def describe_image(self, image_data: str) -> str:
        return ""

    def explain(self, product: Product, signals: dict) -> str:
        price = f"价格 {product.price:.0f}" if product.price is not None else "价格未知"
        evidence = [product.category, price]
        if signals.get("retrieval", 0) > .4:
            evidence.append("与查询文本匹配")
        if signals.get("multimodal", 0) > .4:
            evidence.append("外观与图片需求相近")
        if signals.get("recommendation", 0) > .4:
            evidence.append("与历史偏好相近")
        return "；".join(evidence)

    def explain_many(self, rows: list[tuple[Product, dict]]) -> list[str]:
        return [self.explain(product, signals) for product, signals in rows]


class ResilientLLM:
    """Runtime circuit breaker for paid APIs and local GPU model failures."""

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback or HeuristicLLM()
        self.provider_name = primary.provider_name
        self.last_error: str | None = None

    def _call(self, method: str, *args):
        try:
            value = getattr(self.primary, method)(*args)
            self.last_error = None
            return value
        except (requests.RequestException, ValueError, KeyError, IndexError, RuntimeError) as exc:
            self.last_error = type(exc).__name__
            return getattr(self.fallback, method)(*args)

    def understand(self, text: str) -> dict:
        return self._call("understand", text)

    def describe_image(self, image_data: str) -> str:
        return self._call("describe_image", image_data)

    def explain(self, product: Product, signals: dict) -> str:
        return self._call("explain", product, signals)

    def explain_many(self, rows: list[tuple[Product, dict]]) -> list[str]:
        return self._call("explain_many", rows)


class OpenAICompatibleLLM:
    """Thin adapter for DashScope, Zhipu, Doubao and compatible local servers."""

    provider_name = "openai-compatible"

    def __init__(self, model_name: str, base_url: str, api_key: str, timeout: int = 45):
        if not api_key and not base_url.startswith(("http://127.0.0.1", "http://localhost")):
            raise ValueError("COMMERCE_LLM_API_KEY is required for a remote LLM endpoint")
        self.model_name = model_name
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout

    def _chat(self, messages: list[dict], max_tokens: int, json_mode: bool = False) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model_name, "messages": messages, "temperature": 0,
                   "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        if json_mode and response.status_code == 400:
            payload.pop("response_format", None)
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def understand(self, text: str) -> dict:
        schema = SearchIntent.model_json_schema()
        prompt = (
            "解析电商请求。提取硬约束并进行简洁搜索改写，只返回符合此 schema 的 JSON。"
            "不要臆造用户未表达的品牌、价格或属性。schema=" + json.dumps(schema, ensure_ascii=False)
        )
        output = self._chat([{"role": "system", "content": prompt},
                             {"role": "user", "content": text}], 256, json_mode=True)
        return _validated_intent(_json_object(output), text)

    def describe_image(self, image_data: str) -> str:
        url = image_data if image_data.startswith("data:") else f"data:image/jpeg;base64,{image_data}"
        content = [
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": "提取商品类别、颜色、材质、形状、可见文字和关键属性，用一行中文关键词回答。"},
        ]
        return self._chat([{"role": "user", "content": content}], 160)

    @lru_cache(maxsize=2048)
    def _explain_cached(self, product_json: str, signals_json: str) -> str:
        prompt = "仅依据给定商品字段和排序信号生成一句推荐理由，不得添加未提供的规格或承诺。"
        return self._chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"商品={product_json}\n信号={signals_json}"},
        ], 100)

    def explain(self, product: Product, signals: dict) -> str:
        return self._explain_cached(product.model_dump_json(), json.dumps(signals, sort_keys=True))

    @lru_cache(maxsize=512)
    def _explain_many_cached(self, rows_json: str) -> str:
        prompt = (
            "依据商品字段和排序信号为每个商品生成一句简短中文理由。"
            "不得添加未提供的规格或承诺。只返回 {reasons:{item_id:reason}} JSON。"
        )
        return self._chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": rows_json},
        ], 512, json_mode=True)

    def explain_many(self, rows: list[tuple[Product, dict]]) -> list[str]:
        payload = [
            {"item_id": product.item_id, "product": product.model_dump(), "signals": signals}
            for product, signals in rows
        ]
        output = _json_object(self._explain_many_cached(json.dumps(payload, ensure_ascii=False, sort_keys=True)))
        reasons = output.get("reasons", output)
        return [str(reasons.get(product.item_id, "")) for product, _ in rows]


class TransformersLLM:
    provider_name = "transformers"

    def __init__(self, model_name: str):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=model_name, device_map="auto")

    def understand(self, text: str) -> dict:
        prompt = "将电商请求解析为JSON，字段为intent,query,rewritten_query,max_price,min_price,category,brand,attributes,exclude_terms：" + text
        output = self.pipe(prompt, max_new_tokens=192, do_sample=False)[0]["generated_text"]
        return _validated_intent(_json_object(output[len(prompt):]), text)

    def describe_image(self, image_data: str) -> str:
        return ""

    def explain(self, product: Product, signals: dict) -> str:
        prompt = f"仅依据商品信息生成一句推荐理由：{product.model_dump_json()}，信号={signals}"
        return self.pipe(prompt, max_new_tokens=80, do_sample=False)[0]["generated_text"][len(prompt):].strip()

    def explain_many(self, rows: list[tuple[Product, dict]]) -> list[str]:
        return [self.explain(product, signals) for product, signals in rows]
