from __future__ import annotations

from functools import cached_property

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        try:
            return self._model.get_embedding_dimension()
        except AttributeError:
            return self._model.get_sentence_embedding_dimension()
