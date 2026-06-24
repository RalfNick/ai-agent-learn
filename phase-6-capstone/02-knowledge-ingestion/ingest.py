from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge import build_index_from_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local Phase6 knowledge index.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Document file or directory. Can be passed multiple times.",
    )
    parser.add_argument(
        "--index",
        default=".local/phase6-knowledge-index.json",
        help="Output JSON index path.",
    )
    parser.add_argument("--max-chars", type=int, default=900, help="Maximum characters per chunk.")
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=120,
        help="Character overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--extension",
        action="append",
        help="Allowed extension such as .md or .pdf. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extensions = set(args.extension) if args.extension else None
    index = build_index_from_paths(
        paths=[Path(source) for source in args.source],
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        extensions=extensions,
    )
    index_path = Path(args.index)
    index.save(index_path)

    payload = {
        "index_path": str(index_path),
        "stats": index.stats().to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
