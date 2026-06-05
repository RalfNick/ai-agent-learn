from __future__ import annotations

import json
import re
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    ENTITY = "entity"
    TASK = "task"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    memory_type: MemoryType
    subject: str
    content: str
    confidence: float = 0.8
    source: str = "user_message"
    tags: list[str] = field(default_factory=list)
    memory_id: str | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_accessed_at: str | None = None
    access_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.memory_type, str):
            self.memory_type = MemoryType(self.memory_type)
        if self.memory_id is None:
            self.memory_id = make_memory_id(self.memory_type, self.subject)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryRecord":
        return cls(
            memory_type=MemoryType(data["memory_type"]),
            subject=data["subject"],
            content=data["content"],
            confidence=data.get("confidence", 0.8),
            source=data.get("source", "user_message"),
            tags=list(data.get("tags", [])),
            memory_id=data.get("memory_id"),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
            last_accessed_at=data.get("last_accessed_at"),
            access_count=int(data.get("access_count", 0)),
        )


def make_memory_id(memory_type: MemoryType, subject: str) -> str:
    raw = subject.strip().lower()
    slug = re.sub(r"\s+", "_", raw)
    slug = re.sub(r"[^a-z0-9_:\-\u4e00-\u9fff]+", "_", slug).strip("_")
    slug = slug[:32] or "memory"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{memory_type.value}:{slug}:{digest}"


class JsonMemoryStore:
    """Small file-backed memory store for learning the memory lifecycle."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def upsert(self, record: MemoryRecord) -> MemoryRecord:
        records = self._load()
        now = utc_now()
        record.updated_at = now

        for index, existing in enumerate(records):
            if existing.memory_id == record.memory_id:
                record.created_at = existing.created_at
                record.access_count = existing.access_count
                record.last_accessed_at = existing.last_accessed_at
                records[index] = record
                self._save(records)
                return record

        record.created_at = now
        records.append(record)
        self._save(records)
        return record

    def list_all(self) -> list[MemoryRecord]:
        return self._load()

    def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 5,
    ) -> list[MemoryRecord]:
        if limit <= 0:
            return []

        all_records = self._load()
        records = [
            record
            for record in all_records
            if memory_type is None or record.memory_type == memory_type
        ]
        scored = [
            (self._score(query, record), record)
            for record in records
        ]
        results = [
            record
            for score, record in sorted(scored, key=lambda item: item[0], reverse=True)
            if score > 0
        ][:limit]

        if results:
            now = utc_now()
            for result in results:
                result.last_accessed_at = now
                result.access_count += 1
            self._save(all_records)

        return results

    def _score(self, query: str, record: MemoryRecord) -> float:
        query_tokens = tokenize(query)
        text_tokens = tokenize(" ".join([record.subject, record.content, " ".join(record.tags)]))
        overlap = query_tokens & text_tokens
        if not overlap:
            return 0.0

        score = float(len(overlap))
        if record.memory_type == MemoryType.TASK:
            score += 0.2
            if "memory" in query_tokens or any(token.startswith("phase") for token in query_tokens):
                score += 1.5
        score += min(record.access_count, 3) * 0.05
        score += record.confidence * 0.1
        return score

    def _load(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [MemoryRecord.from_dict(item) for item in raw]

    def _save(self, records: Iterable[MemoryRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [record.to_dict() for record in records]
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def tokenize(text: str) -> set[str]:
    lower = text.lower()
    ascii_tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]*", lower))
    chinese_chars = {char for char in lower if "\u4e00" <= char <= "\u9fff"}
    return ascii_tokens | chinese_chars
