from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parent
CAPSTONE_ROOT = EVAL_ROOT.parents[0]
for path in [
    EVAL_ROOT,
    CAPSTONE_ROOT / "02-knowledge-ingestion",
    CAPSTONE_ROOT / "03-agentic-qa-runtime",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from release_eval import EvalCase, evaluate_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase6 capstone golden-set eval.")
    parser.add_argument("--source", action="append", required=True, help="Document source path.")
    parser.add_argument("--cases", required=True, help="Eval cases JSON file.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-context-score", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = [EvalCase.from_dict(item) for item in raw_cases]
    summary = evaluate_cases(
        cases=cases,
        source_paths=[Path(source) for source in args.source],
        min_context_score=args.min_context_score,
        top_k=args.top_k,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    if summary.passed != summary.total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
