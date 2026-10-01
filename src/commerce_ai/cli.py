import argparse
import json

from .config import Settings
from .data import read_jsonl
from .evaluation import evaluate_queries
from .service import CommerceService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "evaluate"])
    parser.add_argument("--query", default="预算 500 的降噪耳机")
    args = parser.parse_args()
    settings = Settings.from_env()
    service = CommerceService(settings)
    if args.command == "smoke":
        result = service.search(args.query, user_id="u001", top_k=5)
        print(result.model_dump_json(indent=2))
    else:
        rows = read_jsonl(settings.data_dir / "queries.jsonl")
        print(json.dumps(evaluate_queries(service, rows).__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

