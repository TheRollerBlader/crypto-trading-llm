"""Load strategy files and either include them whole or retrieve chunks.

Retrieval uses hashed token embeddings plus token overlap. It does not call
an external embedding API, so backtests and tests run without credentials.
`auto` uses the full text when it fits in `full_max_chars`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import AppConfig
from core.models import StrategyContext

_TAG = re.compile(r"(?=\[(?:TREND|MOM|MR|BRK|RNG|SENT)-\d+\])")


@dataclass
class Chunk:
    source: str
    text: str


class ContextLoader:
    def __init__(self, config: AppConfig, root: Path):
        self.config = config
        self.directory = root / config.strategy.directory
        self._cached_text: str | None = None
        self._cached_sources: list[str] | None = None

    def load(self, query: str) -> StrategyContext:
        text, sources = self._read_all()
        mode = self.config.strategy.mode
        if mode == "auto":
            mode = "full" if len(text) <= self.config.strategy.full_max_chars else "retrieve"
        chunks = _chunk(text, self.config.strategy.chunk_chars, sources)
        if mode == "retrieve":
            chosen = retrieve(chunks, query, self.config.strategy.top_k)
            body = "\n\n".join(chunk.text for chunk in chosen)
            return StrategyContext(mode="retrieve", text=body, sources=sources, chunk_count=len(chosen))
        return StrategyContext(mode="full", text=text, sources=sources, chunk_count=len(chunks))

    def _read_all(self) -> tuple[str, list[str]]:
        if self._cached_text is not None and self._cached_sources is not None:
            return self._cached_text, self._cached_sources
        if not self.directory.exists():
            self._cached_text = ""
            self._cached_sources = []
            return "", []
        parts: list[str] = []
        sources: list[str] = []
        for path in sorted(self.directory.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
                continue
            parts.append(f"# Source: {path.name}\n\n{read_document(path)}")
            sources.append(path.name)
        text = "\n\n".join(parts).strip()
        self._cached_text = text
        self._cached_sources = sources
        return text, sources


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        raw = "\n".join((page.extract_text() or "") for page in reader.pages)
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
    return normalize_document(raw)


def normalize_document(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    nonempty = [line for line in lines if line]
    if nonempty:
        average_words = sum(len(line.split()) for line in nonempty) / len(nonempty)
        if average_words < 1.5:
            text = " ".join(nonempty)
        else:
            text = "\n".join(lines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _chunk(text: str, chunk_chars: int, sources: list[str]) -> list[Chunk]:
    if not text:
        return []
    source = sources[0] if sources else "strategy"
    pieces = [piece.strip() for piece in _TAG.split(text) if piece.strip()]
    if len(pieces) <= 1:
        pieces = [piece.strip() for piece in re.split(r"(?=^#{1,3} )", text, flags=re.M) if piece.strip()]
    chunks: list[Chunk] = []
    buffer = ""
    for piece in pieces:
        if len(buffer) + len(piece) + 2 <= chunk_chars:
            buffer = f"{buffer}\n\n{piece}".strip()
            continue
        if buffer:
            chunks.append(Chunk(source=source, text=buffer))
        if len(piece) <= chunk_chars:
            buffer = piece
            continue
        for start in range(0, len(piece), chunk_chars):
            chunks.append(Chunk(source=source, text=piece[start : start + chunk_chars]))
        buffer = ""
    if buffer:
        chunks.append(Chunk(source=source, text=buffer))
    return chunks


def embed(text: str, dim: int = 256) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float64)
    for token in re.findall(r"[a-z0-9$]+", text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9$]+", text.lower()))


def retrieve(chunks: list[Chunk], query: str, top_k: int) -> list[Chunk]:
    if not chunks or top_k <= 0:
        return []
    query_vector = embed(query)
    query_tokens = _tokens(query)
    scored: list[tuple[float, int, Chunk]] = []
    for index, chunk in enumerate(chunks):
        chunk_vector = embed(chunk.text)
        cosine = float(np.dot(query_vector, chunk_vector))
        tokens = _tokens(chunk.text)
        overlap = len(query_tokens & tokens) / len(query_tokens) if query_tokens else 0.0
        scored.append((0.7 * cosine + 0.3 * overlap, index, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:top_k]]
