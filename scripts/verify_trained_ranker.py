"""Smoke-check the trained model through the API against the complete demo catalog."""

import json
import os
import argparse
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", help="Verify a running API instead of an in-process test client")
    parser.add_argument("--output", type=Path, default=root / "artifacts/lambdamart/api-smoke.json")
    args = parser.parse_args()
    os.environ["COMMERCE_PROFILE"] = "local"
    os.environ["COMMERCE_DATA_DIR"] = str(root / "data/demo")
    os.environ["COMMERCE_RANKING_MODEL_PATH"] = str(root / "artifacts/lambdamart/model.txt")
    if args.api_url:
        import requests

        class LiveClient(requests.Session):
            def request(self, method, url, **kwargs):
                return super().request(method, args.api_url.rstrip("/") + url, timeout=180, **kwargs)

        client_context = LiveClient()
    else:
        from fastapi.testclient import TestClient
        from commerce_ai.api import app, service

        service.cache_clear()
        client_context = TestClient(app)
    queries = ["wireless headphones", "mechanical keyboard", "running shoes", "nail polish", "yoga mat"]
    report = {}
    with client_context as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["providers"]["ranking"] == "lightgbm-lambdarank"
        assert health.json()["catalog"]["products"] == 50000
        report["health"] = health.json()
        tops = []
        for query in queries:
            response = client.post("/v1/search", json={"query": query, "top_k": 3, "include_ads": False})
            response.raise_for_status()
            result = response.json()
            ids = [item["item_id"] for item in result["items"]]
            assert ids, f"No results for smoke query: {query}"
            assert all(item["image_url"] for item in result["items"])
            scores = [item["score"] for item in result["items"]]
            assert scores == sorted(scores, reverse=True)
            tops.append(tuple(ids))
            report[query] = {"item_ids": ids, "titles": [item["title"] for item in result["items"]],
                             "trace": result["trace"]}
        assert len(set(tops)) == len(queries), "Different query categories returned identical results"
        keyboard_ids = report["mechanical keyboard"]["item_ids"]
        assert "ar23_B07BRX2XMV" in keyboard_ids, "Known keyboard body missing from top results"
        assert not {"ar23_B0BKCFNSTK", "ar23_B06VWDBM2M"} & set(keyboard_ids)
        for query, expected_id in [("mechanical keyboard keycaps", "ar23_B06VWDBM2M"),
                                   ("keyboard cable", "ar23_B0BKCFNSTK")]:
            response = client.post("/v1/search", json={"query": query, "top_k": 3, "include_ads": False})
            response.raise_for_status()
            assert response.json()["items"][0]["item_id"] == expected_id
        report["keyboard_body_and_accessory_regression"] = True
        response = client.post("/v1/search", json={"query": "xyz987-unseen-catalog-query"})
        response.raise_for_status()
        assert not response.json()["items"], "Unmatched query unexpectedly returned default results"
        report["unmatched_query_empty"] = True
        response = client.post("/v1/search", json={"query": "电子数码 wireless headphones under 50", "top_k": 5})
        response.raise_for_status()
        constrained = response.json()
        assert constrained["items"]
        assert all(item["price"] <= 50 and item["category"] == "电子数码" for item in constrained["items"])
        report["price_and_category_constraints"] = True
        response = client.post("/v1/agent", json={"message": "wireless headphones under 50"})
        response.raise_for_status()
        agent_result = response.json()
        assert agent_result["items"]
        assert all(item["price"] <= 50 for item in agent_result["items"])
        assert any(event.get("stage") == "rank" and event["provider"] == "lightgbm-lambdarank"
                   for event in agent_result["trace"])
        compare_ids = report[queries[0]]["item_ids"][:2]
        response = client.post("/v1/agent", json={"message": "对比 " + " 和 ".join(compare_ids)})
        response.raise_for_status()
        assert [item["item_id"] for item in response.json()["items"]] == compare_ids
        report["agent_search_and_compare"] = True
    if not args.api_url:
        service.cache_clear()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"queries": len(queries), "unmatched_query_empty": True,
                      "model": "lightgbm-lambdarank", "catalog_products": 50000}), flush=True)


if __name__ == "__main__":
    main()
