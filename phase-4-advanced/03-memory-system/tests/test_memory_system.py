import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_term_memory import JsonMemoryStore, MemoryType
from memory_agent_demo import MemoryAwareAgent
from memory_policy import MemoryPolicy
from short_term_state import ShortTermState


class MemorySystemTests(unittest.TestCase):
    def test_policy_extracts_stable_preference_and_skips_sensitive_content(self) -> None:
        policy = MemoryPolicy()

        preference = policy.extract("以后回答我问题时，代码示例尽量用 Python。")
        sensitive = policy.extract("记住我的 API key 是 sk-test-secret。")
        chinese_secret = policy.extract("记住：我的高德地图密钥是 test-secret。")

        self.assertIsNotNone(preference)
        assert preference is not None
        self.assertEqual(preference.memory_type, MemoryType.PREFERENCE)
        self.assertEqual(preference.subject, "response_style")
        self.assertIn("Python", preference.content)
        self.assertIsNone(sensitive)
        self.assertIsNone(chinese_secret)

    def test_policy_extracts_chinese_project_entity(self) -> None:
        policy = MemoryPolicy()

        entity = policy.extract("记住：我的项目叫 智能客服助手。")

        self.assertIsNotNone(entity)
        assert entity is not None
        self.assertEqual(entity.memory_type, MemoryType.ENTITY)
        self.assertEqual(entity.subject, "project_name")
        self.assertIn("智能客服助手", entity.content)

    def test_policy_does_not_store_task_questions_as_task_state(self) -> None:
        policy = MemoryPolicy()

        task_question = policy.extract("Phase4 当前任务是什么？")

        self.assertIsNone(task_question)

    def test_store_updates_existing_memory_instead_of_duplicating(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonMemoryStore(Path(tmp_dir) / "memory.json")
            policy = MemoryPolicy()

            first = policy.extract("以后回答我问题时，代码示例尽量用 Python。")
            second = policy.extract("以后回答我问题时，代码示例优先用 TypeScript。")

            assert first is not None
            assert second is not None

            store.upsert(first)
            store.upsert(second)

            memories = store.list_all()
            self.assertEqual(len(memories), 1)
            self.assertEqual(memories[0].subject, "response_style")
            self.assertIn("TypeScript", memories[0].content)

    def test_memory_search_returns_relevant_user_and_task_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonMemoryStore(Path(tmp_dir) / "memory.json")
            policy = MemoryPolicy()

            for message in [
                "以后回答我问题时，代码示例尽量用 Python。",
                "记住：Phase4 当前任务是实现 Agent Memory System。",
                "记住：我的项目叫 ai-agent-learn。",
            ]:
                candidate = policy.extract(message)
                self.assertIsNotNone(candidate)
                assert candidate is not None
                store.upsert(candidate)

            results = store.search("Phase4 memory Python 示例", limit=2)

            self.assertEqual([item.subject for item in results], ["phase4_current_task", "response_style"])

    def test_typed_search_does_not_drop_other_memory_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonMemoryStore(Path(tmp_dir) / "memory.json")
            policy = MemoryPolicy()

            for message in [
                "以后回答我问题时，代码示例尽量用 Python。",
                "记住：Phase4 当前任务是实现 Agent Memory System。",
                "记住：我的项目叫 ai-agent-learn。",
            ]:
                candidate = policy.extract(message)
                self.assertIsNotNone(candidate)
                assert candidate is not None
                store.upsert(candidate)

            store.search("Phase4 Memory", memory_type=MemoryType.TASK)

            self.assertEqual(
                sorted(memory.memory_type for memory in store.list_all()),
                [MemoryType.ENTITY, MemoryType.PREFERENCE, MemoryType.TASK],
            )

    def test_chinese_explicit_memories_get_distinct_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = JsonMemoryStore(Path(tmp_dir) / "memory.json")
            policy = MemoryPolicy()

            first = policy.extract("记住：用户喜欢短回答。")
            second = policy.extract("记住：项目使用向量数据库。")

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            assert first is not None
            assert second is not None

            store.upsert(first)
            store.upsert(second)

            memories = store.list_all()
            self.assertEqual(len(memories), 2)
            self.assertNotEqual(memories[0].memory_id, memories[1].memory_id)

    def test_short_term_state_tracks_execution_without_persisting_as_memory(self) -> None:
        state = ShortTermState(goal="学习 Agent Memory")

        state.add_step("理解短期状态和长期记忆的边界")
        state.add_observation("LangGraph checkpoint 更像执行状态，不等于长期记忆")
        snapshot = state.snapshot()

        self.assertEqual(snapshot["goal"], "学习 Agent Memory")
        self.assertEqual(snapshot["steps"], ["理解短期状态和长期记忆的边界"])
        self.assertIn("LangGraph checkpoint", snapshot["observations"][0])
        self.assertNotIn("memories", snapshot)

    def test_agent_uses_memory_to_shape_later_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            agent = MemoryAwareAgent(memory_path=Path(tmp_dir) / "memory.json")

            first = agent.reply("以后回答我问题时，代码示例尽量用 Python。")
            second = agent.reply("我准备继续学习 Phase4 Memory，应该关注什么？")

            self.assertIsNotNone(first.written_memory)
            self.assertIn("已写入长期记忆", first.answer)
            self.assertIsNone(second.written_memory)
            self.assertIn("Python", second.answer)
            self.assertIn("长期记忆", second.answer)


if __name__ == "__main__":
    unittest.main()
