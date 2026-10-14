"""
Patent Intel Agent - Qdrant vector store (Part 4)

Creates the collection, upserts chunk vectors + metadata, and runs dense
top-K semantic search. Dense vectors only; BM25/hybrid comes later.
"""
import logging
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger("patent-intel-agent.vector_store")

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "patent_chunks")


class VectorStoreError(Exception):
    """Raised on Qdrant connection or operation failure."""


def _point_id(chunk_id: str) -> str:
    """
    Qdrant point IDs must be an unsigned int or a UUID. Chunk IDs are readable
    strings, so map them deterministically onto UUIDv5 - same chunk_id always
    yields the same point ID, which makes re-ingest idempotent.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"patent-chunk:{chunk_id}"))


class VectorStore:
    """Thin wrapper over QdrantClient for the patent chunk collection."""

    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection: str = COLLECTION_NAME,
        location: Optional[str] = None,
    ):
        """
        `location` overrides host/port - pass ':memory:' for an in-process
        store (used by tests) or a path for on-disk local mode.
        """
        self.collection = collection
        try:
            if location:
                self.client = QdrantClient(location=location)
            else:
                self.client = QdrantClient(host=host, port=port, timeout=30.0)
        except Exception as exc:
            raise VectorStoreError(f"Could not create Qdrant client: {exc}") from exc

    # -- collection management ------------------------------------------------

    def collection_exists(self) -> bool:
        try:
            existing = {c.name for c in self.client.get_collections().collections}
            return self.collection in existing
        except Exception as exc:
            raise VectorStoreError(f"Qdrant unreachable: {exc}") from exc

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        """Create the collection if missing (or recreate it when asked)."""
        try:
            exists = self.collection_exists()
            if exists and recreate:
                logger.info("Deleting existing collection '%s'", self.collection)
                self.client.delete_collection(self.collection)
                exists = False
            if not exists:
                logger.info(
                    "Creating collection '%s' (dim=%d, cosine)", self.collection, dim
                )
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to ensure collection: {exc}") from exc

    def count(self) -> int:
        try:
            return self.client.count(self.collection, exact=True).count
        except Exception as exc:
            raise VectorStoreError(f"Count failed: {exc}") from exc

    def info(self) -> Dict[str, Any]:
        try:
            if not self.collection_exists():
                return {"collection": self.collection, "exists": False, "points": 0}
            return {
                "collection": self.collection,
                "exists": True,
                "points": self.count(),
            }
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Info failed: {exc}") from exc

    # -- writes ---------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
        batch_size: int = 128,
    ) -> int:
        """Upsert chunks with their vectors. Payload carries all metadata."""
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}"
            )
        if not chunks:
            return 0

        points: List[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "chunk_id": chunk["chunk_id"],
                "patent_id": chunk["patent_id"],
                "section": chunk["section"],
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk["text"],
                "title": chunk.get("title", ""),
                "word_count": chunk.get("word_count", 0),
            }
            if "claim_number" in chunk:
                payload["claim_number"] = chunk["claim_number"]
            points.append(
                PointStruct(
                    id=_point_id(chunk["chunk_id"]),
                    vector=vector,
                    payload=payload,
                )
            )

        written = 0
        try:
            for start in range(0, len(points), batch_size):
                batch = points[start : start + batch_size]
                self.client.upsert(
                    collection_name=self.collection, points=batch, wait=True
                )
                written += len(batch)
                logger.info("Upserted %d/%d points", written, len(points))
        except Exception as exc:
            raise VectorStoreError(f"Upsert failed after {written} points: {exc}") from exc
        return written

    # -- reads ----------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        section: Optional[str] = None,
        patent_id: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense top-K semantic search, with optional metadata filtering.
        Returns patent_id, section, score and text per hit.
        """
        conditions = []
        if section:
            conditions.append(
                FieldCondition(key="section", match=MatchValue(value=section))
            )
        if patent_id:
            conditions.append(
                FieldCondition(key="patent_id", match=MatchValue(value=patent_id))
            )
        query_filter = Filter(must=conditions) if conditions else None

        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Search failed: {exc}") from exc

        results = []
        for point in response.points:
            payload = point.payload or {}
            hit = {
                "chunk_id": payload.get("chunk_id"),
                "patent_id": payload.get("patent_id"),
                "section": payload.get("section"),
                "score": round(float(point.score), 6),
                "text": payload.get("text", ""),
                "title": payload.get("title", ""),
            }
            if "claim_number" in payload:
                hit["claim_number"] = payload["claim_number"]
            results.append(hit)
        return results
