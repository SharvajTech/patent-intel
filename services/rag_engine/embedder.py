"""
Patent Intel Agent - Embeddings (Part 4)

Wraps BGE-M3 (BAAI/bge-m3) for dense vector encoding.

Dense retrieval only. BGE-M3 also produces sparse and ColBERT vectors, but
those feed BM25/hybrid retrieval which is explicitly a later part.

The model is loaded lazily on first use so the API can boot (and /health can
answer) without waiting on a ~2GB model download.
"""
import logging
import os
import threading
from typing import List, Optional, Sequence

logger = logging.getLogger("patent-intel-agent.embedder")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
# BGE-M3 produces 1024-dimensional dense vectors.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

# Set EMBEDDING_BACKEND=stub to run the pipeline without downloading weights.
# Useful for CI/smoke tests. Stub vectors are deterministic but meaningless.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "bge-m3").lower()

# BGE-M3 is trained so that queries and passages share one embedding space;
# unlike bge-v1.5 it does NOT require an instruction prefix on queries.


class EmbedderError(Exception):
    """Raised when the embedding model cannot be loaded or run."""


class _StubEmbedder:
    """
    Deterministic hash-based embedder. NOT semantic - it exists only so the
    ingest/search plumbing can be exercised without the real model.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def encode(self, texts: Sequence[str], **_) -> List[List[float]]:
        import hashlib
        import struct

        out = []
        for text in texts:
            # Expand a digest into `dim` floats, then L2-normalize.
            raw = b""
            counter = 0
            need = self.dim * 4
            while len(raw) < need:
                raw += hashlib.sha256(
                    text.encode("utf-8") + counter.to_bytes(4, "little")
                ).digest()
                counter += 1
            vals = struct.unpack(f"<{self.dim}f", raw[:need])
            vals = [0.0 if (v != v or v in (float("inf"), float("-inf"))) else v
                    for v in vals]
            norm = sum(v * v for v in vals) ** 0.5 or 1.0
            out.append([v / norm for v in vals])
        return out


class Embedder:
    """Thread-safe lazy wrapper around the BGE-M3 sentence-transformers model."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
        backend: str = EMBEDDING_BACKEND,
    ):
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self._model: Optional[object] = None
        self._lock = threading.Lock()

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model

            if self.backend == "stub":
                logger.warning(
                    "EMBEDDING_BACKEND=stub - using non-semantic hash vectors. "
                    "Do not use for real retrieval."
                )
                self._model = _StubEmbedder(EMBEDDING_DIM)
                return self._model

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbedderError(
                    "sentence-transformers is not installed. "
                    "Install it or set EMBEDDING_BACKEND=stub."
                ) from exc

            logger.info(
                "Loading embedding model '%s' on %s (first run downloads ~2GB)",
                self.model_name, self.device,
            )
            try:
                self._model = SentenceTransformer(
                    self.model_name, device=self.device
                )
            except Exception as exc:
                raise EmbedderError(
                    f"Failed to load model '{self.model_name}': {exc}"
                ) from exc
            logger.info("Embedding model loaded.")
            return self._model

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """Encode texts into L2-normalized dense vectors."""
        if not texts:
            return []
        model = self._load()

        if isinstance(model, _StubEmbedder):
            return model.encode(texts)

        try:
            vectors = model.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=True,  # cosine == dot product
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbedderError(f"Encoding failed: {exc}") from exc
        return [v.tolist() for v in vectors]

    def encode_query(self, query: str) -> List[float]:
        """Encode a single search query."""
        vectors = self.encode([query])
        if not vectors:
            raise EmbedderError("Query produced no embedding.")
        return vectors[0]

    def warmup(self) -> None:
        """Force model load (useful at ingest start to fail fast)."""
        self._load()


# Process-wide singleton so the model is loaded at most once.
_embedder: Optional[Embedder] = None
_singleton_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _singleton_lock:
            if _embedder is None:
                _embedder = Embedder()
    return _embedder
