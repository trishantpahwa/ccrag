from __future__ import annotations

from functools import cached_property
from pathlib import Path

import numpy as np

# Best retrieval quality among models that load natively (no remote code):
# 1024-dim, BERT-large, top of its tier on MTEB. Heavier than bge-base but a
# clear quality win. Code-specialized models (e.g. jinaai/jina-embeddings-v2-base-code)
# can score higher on code search but ship custom modeling code that requires
# trust_remote_code=True and breaks on mismatched transformers versions; use
# --model to opt into one if your environment supports it.
DEFAULT_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
MODELS_DIRNAME = "models"


def models_cache_dir(root) -> str:
    """Directory inside the project's .ccrag/ where model weights are cached.

    Keeping the cache in-repo means the model is downloaded once and reused on
    every later `index`/`serve`/`watch`, with no network access required.
    """
    return str(Path(root) / ".ccrag" / MODELS_DIRNAME)


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, cache_folder: str | None = None):
        self.model_name = model_name
        self.cache_folder = cache_folder

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if self.cache_folder:
            kwargs["cache_folder"] = self.cache_folder
        # Prefer the local cache: once the model lives in .ccrag/models it loads
        # offline with no Hub round-trip. Fall back to downloading on first use.
        try:
            return SentenceTransformer(self.model_name, local_files_only=True, **kwargs)
        except Exception:
            return SentenceTransformer(self.model_name, local_files_only=False, **kwargs)

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
