"""Shared test fixtures.

The real embedding model downloads weights and is slow to load, so unit tests
use a deterministic, hash-seeded fake embedder instead. Identical text always
maps to the identical (normalized) vector, which lets us assert exact-match
ranking without any network access or GPU.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest


class FakeEmbedder:
    """Drop-in stand-in for ccrag.embedder.Embedder used in tests.

    Matches the interface (`embed`, `embed_one`, `dim`) and is deterministic:
    the same string always yields the same unit vector.
    """

    def __init__(self, model_name: str | None = None, cache_folder: str | None = None,
                 dim: int = 32):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self._dim = dim
        self.embedded_texts: list[str] = []  # every text ever embedded, for assertions

    def embed(self, texts: list[str]) -> np.ndarray:
        self.embedded_texts.extend(texts)
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.stack([self._vec(t) for t in texts])

    def embed_one(self, text: str) -> np.ndarray:
        return self._vec(text)

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_embedder_cls() -> type[FakeEmbedder]:
    """The class itself, for monkeypatching ccrag.embedder.Embedder."""
    return FakeEmbedder


@pytest.fixture
def make_record(fake_embedder: FakeEmbedder):
    """Factory for a store record dict with a real (fake) embedding vector."""

    def _make(
        id: str,
        text: str,
        file_path: str = "f.py",
        start: int = 1,
        end: int = 5,
        lang: str = "python",
    ) -> dict:
        return {
            "id": id,
            "file_path": file_path,
            "start_line": start,
            "end_line": end,
            "language": lang,
            "content": text,
            "vector": fake_embedder.embed_one(text).tolist(),
        }

    return _make
