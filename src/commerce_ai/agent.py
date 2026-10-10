import re

from .schemas import ResultSet
from .service import CommerceService


ITEM_ID = re.compile(r"\b(?:p\d{3})\b", re.I)


class CommerceAgent:
    """Bounded, observable tool orchestration rather than an unconstrained chatbot."""

    def __init__(self, service: CommerceService):
        self.service = service

    def run(self, message: str, user_id: str | None = None, image_data: str | None = None) -> dict:
        intent = self.service.llm.understand(message)
        trace = [{"step": 1, "action": "understand", "output": intent}]
        ids = ITEM_ID.findall(message)
        if intent["intent"] == "compare" and len(ids) >= 2:
            output = self.service.compare(ids[:4])
            trace.append({"step": 2, "action": "compare_products", "args": ids[:4]})
            return {"answer": output["summary"], "items": output["items"], "trace": trace}
        # Pass the parsed intent into search to avoid a second paid LLM call.
        result = self.service.search(
            message, user_id=user_id, top_k=5, image_data=image_data, parsed_intent=intent
        )
        action = "recommend_for_user" if intent["intent"] == "recommend" and user_id else "search_products"
        trace.append({"step": 2, "action": action, "args": {"user_id": user_id, "query": message}})
        trace.extend({"step": index + 3, **event} for index, event in enumerate(result.trace))
        answer = "；".join(f"{item.title}（{item.reason}）" for item in result.items[:3])
        return {"answer": answer, "items": [item.model_dump() for item in result.items], "trace": trace}

