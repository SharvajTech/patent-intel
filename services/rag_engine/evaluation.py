"""
Patent Intel Agent - Retrieval evaluation (Part 8)

Compares vector, BM25 and hybrid-RRF retrieval on a labelled query set.

Metrics are computed at the PATENT level, not the chunk level: a query is
satisfied if the right patent surfaces, regardless of which of its chunks
matched. Chunk-level scoring would unfairly punish a retriever that returns
several good chunks from one correct patent.
"""
import logging
import time
from typing import Any, Callable, Dict, List, Sequence, Set

logger = logging.getLogger("patent-intel-agent.evaluation")


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    Relevant patents in the top-k, divided by k. The denominator is always k:
    if retrieval returns fewer than k distinct patents, the missing slots
    count as misses rather than silently shrinking the denominator.
    """
    if k <= 0:
        return 0.0
    top = list(dict.fromkeys(retrieved))[:k]
    return sum(1 for p in top if p in relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fraction of all relevant patents found within the top-k."""
    if not relevant:
        return 0.0
    top = set(list(dict.fromkeys(retrieved))[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    """1 / rank of the first relevant patent; 0 if none found."""
    for i, pid in enumerate(dict.fromkeys(retrieved), start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def hits_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    """1.0 if any relevant patent appears in the top-k, else 0.0."""
    top = set(list(dict.fromkeys(retrieved))[:k])
    return 1.0 if top & relevant else 0.0


def evaluate_retriever(
    name: str,
    search_fn: Callable[[str, int], List[Dict[str, Any]]],
    queries: Sequence[Dict[str, Any]],
    k_values: Sequence[int] = (1, 3, 5, 10),
    fetch_multiplier: int = 5,
) -> Dict[str, Any]:
    """
    Run one retriever over the query set and aggregate metrics.

    `search_fn(query, top_k)` must return hits carrying a `patent_id`.
    """
    max_k = max(k_values)
    # Retrieval is chunk-level; several chunks can come from one patent, so
    # fetch deeper to have enough DISTINCT patents to fill the top-k.
    fetch_depth = max_k * fetch_multiplier
    per_query: List[Dict[str, Any]] = []
    latencies: List[float] = []
    failures = 0

    for q in queries:
        query_text = q["query"]
        relevant = set(q.get("relevant_patent_ids", []))

        start = time.perf_counter()
        try:
            hits = search_fn(query_text, fetch_depth)
        except Exception as exc:  # noqa: BLE001 - one bad query must not abort
            logger.warning("[%s] query failed: %s", name, exc)
            failures += 1
            hits = []
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)

        retrieved = [h.get("patent_id", "") for h in hits]
        entry: Dict[str, Any] = {
            "query_id": q.get("id"),
            "query": query_text,
            "relevant_count": len(relevant),
            "retrieved_patents": list(dict.fromkeys(retrieved))[:max_k],
            "latency_ms": round(elapsed_ms, 2),
            "mrr": round(reciprocal_rank(retrieved, relevant), 6),
        }
        for k in k_values:
            entry[f"precision@{k}"] = round(precision_at_k(retrieved, relevant, k), 6)
            entry[f"recall@{k}"] = round(recall_at_k(retrieved, relevant, k), 6)
            entry[f"hit@{k}"] = hits_at_k(retrieved, relevant, k)
        per_query.append(entry)

    n = len(per_query) or 1
    aggregate: Dict[str, Any] = {"mrr": round(sum(e["mrr"] for e in per_query) / n, 6)}
    for k in k_values:
        for metric in ("precision", "recall", "hit"):
            key = f"{metric}@{k}"
            aggregate[key] = round(sum(e[key] for e in per_query) / n, 6)

    ordered = sorted(latencies)
    aggregate["latency_ms"] = {
        "mean": round(sum(latencies) / n, 2),
        "median": round(ordered[len(ordered) // 2], 2) if ordered else 0.0,
        "min": round(min(latencies), 2) if latencies else 0.0,
        "max": round(max(latencies), 2) if latencies else 0.0,
        "p95": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 2) if ordered else 0.0,
    }

    return {
        "retriever": name,
        "query_count": len(per_query),
        "failed_queries": failures,
        "aggregate": aggregate,
        "per_query": per_query,
    }


def format_comparison(
    results: Sequence[Dict[str, Any]], k_values: Sequence[int] = (1, 3, 5, 10)
) -> str:
    """Render a markdown comparison table across retrievers."""
    metrics = ["mrr"]
    for k in k_values:
        metrics += [f"precision@{k}", f"recall@{k}"]
    metrics.append("latency_ms_mean")

    header = "| Metric | " + " | ".join(r["retriever"] for r in results) + " |"
    sep = "|---" * (len(results) + 1) + "|"
    lines = [header, sep]

    for metric in metrics:
        cells = []
        for r in results:
            if metric == "latency_ms_mean":
                cells.append(f"{r['aggregate']['latency_ms']['mean']:.1f} ms")
            else:
                cells.append(f"{r['aggregate'].get(metric, 0.0):.4f}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
