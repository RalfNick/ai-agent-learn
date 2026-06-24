from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

from .models import Document


DEFAULT_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}


def load_documents(
    paths: Sequence[Path | str],
    extensions: set[str] | None = None,
) -> list[Document]:
    allowed_extensions = {extension.lower() for extension in (extensions or DEFAULT_EXTENSIONS)}
    files = _iter_files(paths, allowed_extensions)
    documents: list[Document] = []

    for file_path in files:
        content = _read_file(file_path)
        if not content.strip():
            continue
        documents.append(
            Document(
                document_id=_document_id(file_path),
                path=str(file_path),
                title=_extract_title(file_path, content),
                content=content,
                extension=file_path.suffix.lower(),
            )
        )

    return documents


def _iter_files(paths: Sequence[Path | str], extensions: set[str]) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file() and path.suffix.lower() in extensions:
            discovered.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in extensions:
                    discovered.append(child)
    return sorted(discovered)


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF loading requires installing pypdf.") from exc

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_title(path: Path, content: str) -> str:
    heading = re.search(r"^\s*#\s+(.+?)\s*$", content, flags=re.MULTILINE)
    if heading:
        return heading.group(1).strip()
    return path.stem.replace("-", " ").replace("_", " ").strip().title()


def _document_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
