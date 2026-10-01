import json
import re

from .schemas import Product


PRICE = re.compile(r"(?:预算|价格|低于|不超过|under)\s*[¥￥$]?\s*(\d+(?:\.\d+)?)", re.I)


class HeuristicLLM:
    """Offline contract-compatible fallback used for tests and local smoke runs."""

    def understand(self, text: str) -> dict:
        price = PRICE.search(text)
        lowered = text.lower()
        intent = "recommend" if any(word in text for word in ("推荐", "适合我", "猜你喜欢")) else "search"
        if any(word in text for word in ("对比", "比较", "区别")):
            intent = "compare"
        return {
            "intent": intent,
            "query": text.strip(),
            "max_price": float(price.group(1)) if price else None,
            "category": next((name for name in ("耳机", "键盘", "显示器", "咖啡", "运动") if name in text), None),
        }

    def explain(self, product: Product, signals: dict) -> str:
        evidence = [product.category, f"价格 {product.price:.0f}"]
        if signals.get("retrieval", 0) > .4:
            evidence.append("与查询语义匹配")
        if signals.get("recommendation", 0) > .4:
            evidence.append("与历史偏好相近")
        return "；".join(evidence)


class TransformersLLM:
    def __init__(self, model_name: str):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=model_name, device_map="auto")

    def understand(self, text: str) -> dict:
        prompt = "将电商请求解析为JSON，字段为intent,query,max_price,category：" + text
        output = self.pipe(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        return json.loads(output[output.index("{"):output.rindex("}") + 1])

    def explain(self, product: Product, signals: dict) -> str:
        prompt = f"仅依据商品信息生成一句推荐理由：{product.model_dump_json()}，信号={signals}"
        return self.pipe(prompt, max_new_tokens=80, do_sample=False)[0]["generated_text"][len(prompt):].strip()

