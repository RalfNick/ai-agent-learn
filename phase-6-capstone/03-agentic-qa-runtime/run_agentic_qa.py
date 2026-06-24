from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_qa import build_runtime_from_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase6 Agentic QA runtime.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Document file or directory. Can be passed multiple times.",
    )
    parser.add_argument("--question", required=True, help="Question to answer.")
    parser.add_argument("--session-id", default="cli", help="Session id for trace continuity.")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum number of retrieved chunks.")
    parser.add_argument(
        "--min-context-score",
        type=float,
        default=0.25,
        help="Minimum context score required to answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = build_runtime_from_sources(
        [Path(source) for source in args.source],
        min_context_score=args.min_context_score,
        top_k=args.top_k,
    )
    response = runtime.answer(args.question, session_id=args.session_id)
    print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
