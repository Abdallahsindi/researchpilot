"""Embedding providers: text -> L2-normalized float32 vectors.

Because vectors are normalized, inner product equals cosine similarity, which is what the
vector stores rank by.
"""

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from app.rag.deps import require

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    model_id: str

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbeddings:
    """Semantic embeddings from a sentence-transformers model, loaded on first use."""

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            st = require("sentence_transformers", "semantic document embeddings")
            try:
                self._model = st.SentenceTransformer(self.model_id, device=self.device)
            except Exception as exc:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_id}': {exc}. "
                    "Downloading it needs internet access (on Kaggle: Settings > Internet > On), "
                    "or point the model id at a local copy of the model files."
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        model = self.model
        getter = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        return int(getter())

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


_WORD_RE = re.compile(r"[a-z0-9]+")


class HashingEmbeddings:
    """Lexical bag-of-words vectors via the hashing trick. No model download.

    This is not a neural model and captures no meaning beyond shared words. It exists so
    tests and offline development can exercise the full pipeline deterministically.
    """

    def __init__(self, dimension: int = 512) -> None:
        self.model_id = f"hashing-bow-{dimension}"
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for word in _WORD_RE.findall(text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % self._dimension] += 1.0 if (value >> 63) & 1 else -1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)
