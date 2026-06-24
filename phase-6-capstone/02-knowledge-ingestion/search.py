from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge import LocalKnowledgeIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a local Phase6 knowledge index.")
    parser.add_argument("--index", required=True, help="Input JSON index path.")
    parser.add_argument("--query", required=True, help="Search query.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = LocalKnowledgeIndex.load(Path(args.index))
    results = index.search(args.query, limit=args.limit)

    payload = [
        {
            "rank": rank,
            "title": result.chunk.title,
            "path": result.chunk.path,
            "score": result.score,
            "lexical_score": result.lexical_score,
            "vector_score": result.vector_score,
            "snippet": result.chunk.content[:240].replace("\n", " "),
        }
        for rank, result in enumerate(results, start=1)
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
