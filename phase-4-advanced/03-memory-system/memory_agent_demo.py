from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from long_term_memory import JsonMemoryStore, MemoryRecord
from memory_policy import MemoryPolicy
from short_term_state import ShortTermState


DEFAULT_MEMORY_PATH = Path(__file__).parent / ".memory" / "agent_memory.json"


@dataclass
class AgentReply:
    answer: str
    retrieved_memories: list[MemoryRecord]
    written_memory: MemoryRecord | None
    short_term_state: dict


class MemoryAwareAgent:
    def __init__(self, memory_path: Path | str = DEFAULT_MEMORY_PATH) -> None:
        self.store = JsonMemoryStore(memory_path)
        self.policy = MemoryPolicy()

    def reply(self, message: str) -> AgentReply:
        state = ShortTermState(goal="回答当前用户问题")
        state.add_step("解析用户输入")

        written = self.policy.extract(message)
        if written is not None:
            self.store.upsert(written)
            state.add_step("根据写入策略更新长期记忆")

        retrieved = self.store.search(message, limit=3)
        state.add_step("按当前问题召回相关长期记忆")
        answer = self._compose_answer(message, retrieved, written)
        state.add_observation("长期记忆只影响回答上下文，不保存本轮执行步骤")

        return AgentReply(
            answer=answer,
            retrieved_memories=retrieved,
            written_memory=written,
            short_term_state=state.snapshot(),
        )

    def _compose_answer(
        self,
        message: str,
        memories: list[MemoryRecord],
        written: MemoryRecord | None,
    ) -> str:
        if written is not None:
            return (
                f"已写入长期记忆：{written.content}\n"
                "这类信息会跨会话保留；本轮推理步骤仍只放在短期状态里。"
            )

        hints = "\n".join(f"- {memory.content}" for memory in memories)
        style = "默认用简洁步骤说明"
        if any("Python" in memory.content for memory in memories):
            style = "优先给 Python 代码示例"

        return (
            "我会把 Phase4 Memory 当成长期记忆系统来学，而不是再做一次 RAG。\n"
            f"回答风格：{style}。\n"
            "建议关注三件事：长期记忆写入策略、相关记忆召回、记忆更新与冲突处理。\n"
            f"本次召回的长期记忆：\n{hints or '- 暂无相关长期记忆'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small memory-aware Agent demo.")
    parser.add_argument(
        "messages",
        nargs="*",
        help="User messages. If omitted, the demo runs a two-turn conversation.",
    )
    parser.add_argument(
        "--memory-path",
        default=str(DEFAULT_MEMORY_PATH),
        help="Path to the JSON memory file.",
    )
    args = parser.parse_args()

    messages = args.messages or [
        "以后回答我问题时，代码示例尽量用 Python。",
        "我准备继续学习 Phase4 Memory，应该关注什么？",
    ]
    agent = MemoryAwareAgent(memory_path=Path(args.memory_path))

    for index, message in enumerate(messages, start=1):
        reply = agent.reply(message)
        print(f"\n--- Turn {index} ---")
        print(f"User: {message}")
        print(f"Agent:\n{reply.answer}")
        print(f"Short-term state: {reply.short_term_state}")


if __name__ == "__main__":
    main()
