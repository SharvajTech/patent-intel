"""
Patent Intel Agent - Hybrid retrieval (Part 5)

Combines dense vector search (Part 4) with BM25 keyword search using
Reciprocal Rank Fusion.

RRF is used rather than score averaging because BM25 scores are unbounded
while cosine similarity is in [-1, 1]; the two are not comparable on the same
scale. RRF only consumes rank position, so no normalization is needed.

    rrf_score(d) = sum over retrievers r of  weight_r / (k + rank_r(d))
"""
import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("patent-intel-agent.hybrid")

# k dampens the influence of top ranks. 60 is the value from the original
# Cormack et al. RRF paper and is the common default.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[Dict[str, Any]]],
    weights: Optional[Sequence[float]] = None,
    k: int = DEFAULT_RRF_K,
    top_k: int = 5,
    key: str = "chunk_id",
) -> List[Dict[str, Any]]:
    """
    Fuse several ranked result lists into one.

    Each input list must be ordered best-first. Returns fused results carrying
    the RRF score plus the per-retriever ranks that produced it.
    """
    if not result_lists:
        return []
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights length must match result_lists length")
    if k <= 0:
        raise ValueError("RRF k must be positive")

    fused: Dict[Any, Dict[str, Any]] = {}

    for list_idx, (results, weight) in enumerate(zip(result_lists, weights)):
        for rank, hit in enumerate(results, start=1):
            doc_key = hit.get(key)
            if doc_key is None:
                continue
            contribution = weight / (k + rank)

            if doc_key not in fused:
                # Copy the hit, replacing the retriever-specific score.
                entry = {
                    kk: vv for kk, vv in hit.items() if kk != "score"
                }
                entry["rrf_score"] = 0.0
                entry["retriever_ranks"] = {}
                entry["retriever_scores"] = {}
                fused[doc_key] = entry

            fused[doc_key]["rrf_score"] += contribution
            fused[doc_key]["retriever_ranks"][str(list_idx)] = rank
            if "score" in hit:
                fused[doc_key]["retriever_scores"][str(list_idx)] = hit["score"]

    ranked = sorted(
        fused.values(), key=lambda e: e["rrf_score"], reverse=True
    )[:top_k]

    for entry in ranked:
        entry["rrf_score"] = round(entry["rrf_score"], 8)
    return ranked


def _label_retrievers(
    results: List[Dict[str, Any]], names: Sequence[str]
) -> List[Dict[str, Any]]:
    """Rename positional retriever keys ('0','1') to readable names."""
    for entry in results:
        entry["ranks"] = {
            names[int(i)]: r for i, r in entry.pop("retriever_ranks").items()
        }
        entry["component_scores"] = {
            names[int(i)]: s for i, s in entry.pop("retriever_scores").items()
        }
        # Present a single 'score' field for API consistency with /search.
        entry["score"] = entry["rrf_score"]
    return results


def hybrid_search(
    query: str,
    vector_store,
    embedder,
    bm25_index,
    top_k: int = 5,
    candidate_k: int = 30,
    section: Optional[str] = None,
    patent_id: Optional[str] = None,
    rrf_k: int = DEFAULT_RRF_K,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    Run both retrievers and fuse with RRF.

    `candidate_k` is how deep each retriever goes before fusion. It should be
    comfortably larger than top_k so that a document ranked highly by only one
    retriever can still surface.
    """
    candidate_k = max(candidate_k, top_k)

    # --- dense retrieval ---
    vector_results: List[Dict[str, Any]] = []
    vector_error: Optional[str] = None
    try:
        query_vector = embedder.encode_query(query)
        vector_results = vector_store.search(
            query_vector=query_vector,
            top_k=candidate_k,
            section=section,
            patent_id=patent_id,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to BM25-only
        vector_error = str(exc)
        logger.warning("Vector retrieval failed, continuing BM25-only: %s", exc)

    # --- keyword retrieval ---
    bm25_results: List[Dict[str, Any]] = []
    bm25_error: Optional[str] = None
    try:
        bm25_results = bm25_index.search(
            query, top_k=candidate_k, section=section, patent_id=patent_id
        )
    except Exception as exc:  # noqa: BLE001 - degrade to vector-only
        bm25_error = str(exc)
        logger.warning("BM25 retrieval failed, continuing vector-only: %s", exc)

    if not vector_results and not bm25_results:
        return {
            "query": query,
            "results": [],
            "result_count": 0,
            "retrievers": {
                "vector": {"count": 0, "error": vector_error},
                "bm25": {"count": 0, "error": bm25_error},
            },
        }

    fused = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        weights=[vector_weight, bm25_weight],
        k=rrf_k,
        top_k=top_k,
    )
    fused = _label_retrievers(fused, ["vector", "bm25"])

    return {
        "query": query,
        "results": fused,
        "result_count": len(fused),
        "retrievers": {
            "vector": {"count": len(vector_results), "error": vector_error},
            "bm25": {"count": len(bm25_results), "error": bm25_error},
        },
    }
