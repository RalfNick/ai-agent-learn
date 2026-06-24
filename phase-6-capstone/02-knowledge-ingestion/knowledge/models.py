from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Document:
    document_id: str
    path: str
    title: str
    content: str
    extension: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        return cls(
            document_id=str(data["document_id"]),
            path=str(data["path"]),
            title=str(data["title"]),
            content=str(data["content"]),
            extension=str(data["extension"]),
        )


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    path: str
    title: str
    content: str
    ordinal: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            document_id=str(data["document_id"]),
            path=str(data["path"]),
            title=str(data["title"]),
            content=str(data["content"]),
            ordinal=int(data["ordinal"]),
        )


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    lexical_score: float
    vector_score: float

    def to_dict(self) -> dict:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "lexical_score": self.lexical_score,
            "vector_score": self.vector_score,
        }


@dataclass(frozen=True)
class IndexStats:
    document_count: int
    chunk_count: int
    token_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IndexStats":
        return cls(
            document_count=int(data["document_count"]),
            chunk_count=int(data["chunk_count"]),
            token_count=int(data["token_count"]),
        )
