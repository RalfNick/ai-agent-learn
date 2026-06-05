from __future__ import annotations

import argparse
from pathlib import Path

from runtime import IntegratedAgentRuntime


DEFAULT_QUESTION = "请结合 Phase4 Memory 的代码、文章和测试证据，说明是否可以进入 Phase5"
DEFAULT_MEMORY = ".memory/runtime_memory.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase4 integrated Agent runtime demo.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--memory-path", default=DEFAULT_MEMORY)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    memory_path = Path(args.memory_path)
    if not memory_path.is_absolute():
        memory_path = Path(__file__).resolve().parent / memory_path

    runtime = IntegratedAgentRuntime(project_root=project_root, memory_path=memory_path)

    warmup = runtime.answer("以后回答我问题时，代码示例尽量用 Python。")
    result = runtime.answer(args.question)

    print("== Warmup Memory ==")
    print(warmup.written_memory.content if warmup.written_memory else "no memory written")

    print("\n== Answer ==")
    print(result.answer)

    print("\n== Memory Context ==")
    for memory in result.memory_context:
        print(f"- {memory.memory_type.value}:{memory.subject} -> {memory.content}")

    print("\n== Tool Results ==")
    for tool_result in result.tool_results:
        print(f"- {tool_result.tool_name}: count={tool_result.count}, query={tool_result.query}")
        for item in tool_result.evidence[:3]:
            print(f"  - {item}")

    print("\n== Review ==")
    print(f"status={result.review.status.value}, score={result.review.score:.2f}")
    for comment in result.review.comments:
        print(f"- {comment}")

    print("\n== Trace ==")
    for step in result.trace:
        print(f"- {step}")


if __name__ == "__main__":
    main()
