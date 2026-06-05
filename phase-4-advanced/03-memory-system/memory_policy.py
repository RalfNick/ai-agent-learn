from __future__ import annotations

import re

from long_term_memory import MemoryRecord, MemoryType


class MemoryPolicy:
    """Decides whether a user message deserves long-term memory."""

    SENSITIVE_PATTERNS = [
        re.compile(r"\bapi[_ -]?key\b", re.IGNORECASE),
        re.compile(r"\btoken\b", re.IGNORECASE),
        re.compile(r"\bpassword\b", re.IGNORECASE),
        re.compile(r"\bsecret\b", re.IGNORECASE),
        re.compile(r"\bbearer\s+[a-z0-9._-]+", re.IGNORECASE),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"\bsk-[a-z0-9_-]+", re.IGNORECASE),
        re.compile(r"身份证"),
        re.compile(r"银行卡"),
        re.compile(r"密码"),
        re.compile(r"密钥"),
    ]

    def extract(self, message: str) -> MemoryRecord | None:
        text = message.strip()
        if not text or self._looks_sensitive(text):
            return None

        if "以后回答" in text and ("代码示例" in text or "示例" in text):
            return MemoryRecord(
                memory_type=MemoryType.PREFERENCE,
                subject="response_style",
                content=text,
                confidence=0.9,
                tags=["user_preference", "response_style", "code_example"],
            )

        phase_task = re.search(r"(Phase\d+)\s*当前任务是(.+?)(?:。|$)", text, re.IGNORECASE)
        if phase_task:
            if "?" in text or "？" in text:
                return None
            phase = phase_task.group(1).lower()
            task = phase_task.group(2).strip(" ：:")
            return MemoryRecord(
                memory_type=MemoryType.TASK,
                subject=f"{phase}_current_task",
                content=f"{phase_task.group(1)} 当前任务是{task}",
                confidence=0.85,
                tags=["task_state", phase, "memory"],
            )

        project_name = re.search(r"我的项目叫\s*([A-Za-z0-9_\-\u4e00-\u9fff]+)", text)
        if project_name:
            return MemoryRecord(
                memory_type=MemoryType.ENTITY,
                subject="project_name",
                content=f"用户当前项目叫 {project_name.group(1)}",
                confidence=0.85,
                tags=["entity", "project"],
            )

        if text.startswith("记住：") or text.startswith("记住:"):
            content = text.split("：", 1)[-1] if "：" in text else text.split(":", 1)[-1]
            return MemoryRecord(
                memory_type=MemoryType.ENTITY,
                subject=self._subject_from_content(content),
                content=content.strip(),
                confidence=0.7,
                tags=["explicit_memory"],
            )

        return None

    def _looks_sensitive(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.SENSITIVE_PATTERNS)

    def _subject_from_content(self, content: str) -> str:
        compact = re.sub(r"\s+", "_", content.strip().lower())
        compact = re.sub(r"[^a-z0-9_\u4e00-\u9fff-]+", "", compact)
        return compact[:40] or "explicit_memory"
