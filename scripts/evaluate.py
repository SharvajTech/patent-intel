#!/usr/bin/env python3
"""
Patent Intel Agent - Part 8 evaluation CLI

Compares vector, BM25 and hybrid-RRF retrieval on a labelled query set and
writes metrics to data/evaluation/.

Usage
-----
  # generate a starter query set matched to the current corpus
  python scripts/evaluate.py --generate-queries

  # run the comparison
  python scripts/evaluate.py

  # BM25 only (no Qdrant / no model needed)
  python scripts/evaluate.py --retrievers bm25
"""
import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.rag_engine.bm25 import BM25Error, get_bm25_index, load_chunks  # noqa: E402
from services.rag_engine.evaluation import (  # noqa: E402
    evaluate_retriever,
    format_comparison,
)
from services.rag_engine.hybrid import hybrid_search  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evaluate")

EVAL_DIR = Path("data/evaluation")
QUERIES_PATH = EVAL_DIR / "eval_queries.json"


def generate_queries(chunks, target: int = 24) -> list:
    """
    Build a starter query set from the corpus.

    Queries are derived from patent titles: the title's distinctive words
    become the query, and every patent sharing that technical theme is
    labelled relevant. This is a REASONABLE PROXY, not human relevance
    judgement - replace it with real labels before drawing firm conclusions.
    """
    import re

    stop = set("""a an the and or of to for with in on at by system method
    apparatus device configuration using via from into""".split())

    # Group patents by their title signature (title minus the variant suffix).
    themes = defaultdict(set)
    titles = {}
    for c in chunks:
        pid = c.get("patent_id")
        title = (c.get("title") or "").strip()
        if not pid or not title:
            continue
        titles[pid] = title
        signature = re.sub(r"\(configuration \d+\)", "", title).strip().lower()
        themes[signature].add(pid)

    ranked = sorted(themes.items(), key=lambda kv: -len(kv[1]))
    queries = []
    qid = 0
    # Three phrasings per theme - full, component-focused, application-focused -
    # so the set exercises different lexical overlap with the corpus.
    for signature, pids in ranked:
        words = [w for w in re.findall(r"[a-z0-9\-]+", signature) if w not in stop]
        if len(words) < 3:
            continue
        variants = [
            " ".join(words[:10]),          # broad: whole title theme
            " ".join(words[:4]),           # narrow: leading technical terms
            " ".join(words[-5:]),          # application end of the title
        ]
        seen = set()
        for variant in variants:
            variant = variant.strip()
            if len(variant.split()) < 2 or variant in seen:
                continue
            seen.add(variant)
            qid += 1
            queries.append({
                "id": f"q{qid:03d}",
                "query": variant,
                "relevant_patent_ids": sorted(pids),
                "note": "auto-generated from title theme; replace with human labels",
            })
            if len(queries) >= target:
                break
        if len(queries) >= target:
            break

    return queries


def generate_known_item_queries(chunks, target: int = 30, seed: int = 42) -> list:
    """
    Known-item queries for corpora where every patent has a unique title
    (i.e. real data). Each query is built from ONE patent and labelled with
    that patent as the only relevant result.

    Two styles alternate so lexical overlap varies:
      - title   : the first content words of the patent title
      - keyterms: the most distinctive abstract terms (highest tf*idf), in
                  keyword form, the way a prior-art searcher might type them

    This is a PROXY for relevance, not human judgement. It measures "can the
    retriever find a patent from a description of it", and favours lexical
    methods because the query shares vocabulary with the source text.
    """
    import random
    import re

    from services.rag_engine.bm25 import get_bm25_index, tokenize

    index = get_bm25_index()
    abstracts, titles = {}, {}
    for c in chunks:
        pid = c.get("patent_id")
        if not pid:
            continue
        titles[pid] = c.get("title", "")
        if c.get("section") == "abstract":
            abstracts[pid] = c.get("text", "")

    pids = sorted(p for p in titles if abstracts.get(p) and titles[p])
    random.Random(seed).shuffle(pids)

    stop = set("a an the and or of to for with in on at by system method apparatus device".split())
    queries = []
    for pid in pids[:target]:
        n = len(queries) + 1
        if n % 2:
            words = [w for w in re.findall(r"[a-z0-9\-]+", titles[pid].lower()) if w not in stop]
            query = " ".join(words[:10])
            style = "title"
        else:
            tf = {}
            for tok in tokenize(abstracts[pid]):
                tf[tok] = tf.get(tok, 0) + 1
            ranked = sorted(tf, key=lambda t: tf[t] * index.idf.get(t, 0.0), reverse=True)
            query = " ".join(ranked[:8])
            style = "keyterms"
        if len(query.split()) < 3:
            continue
        queries.append({
            "id": f"q{len(queries) + 1:03d}",
            "query": query,
            "relevant_patent_ids": [pid],
            "style": style,
            "note": "known-item proxy label; replace with human judgements for real conclusions",
        })
    return queries


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate retrieval strategies.")
    ap.add_argument("--queries", default=str(QUERIES_PATH))
    ap.add_argument("--generate-queries", action="store_true",
                    help="Write a starter query set and exit.")
    ap.add_argument("--target-queries", type=int, default=24)
    ap.add_argument("--query-mode", choices=["auto", "theme", "known-item"], default="auto",
                    help="auto = theme labels if titles repeat, else known-item.")
    ap.add_argument("--retrievers", default="vector,bm25,hybrid",
                    help="Comma-separated subset to evaluate.")
    ap.add_argument("--k-values", default="1,3,5,10")
    ap.add_argument("--candidate-k", type=int, default=30)
    ap.add_argument("--hold-out-abstract", action="store_true",
                    help="Drop abstract chunks from results. Known-item queries are "
                         "built from abstracts, so leaving them in lets any retriever "
                         "score ~perfectly by matching the source text.")
    ap.add_argument("--output-dir", default=str(EVAL_DIR))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values = [int(k) for k in args.k_values.split(",")]

    try:
        chunks = load_chunks()
    except BM25Error as exc:
        logger.error("%s", exc)
        return 1

    if args.generate_queries:
        mode = args.query_mode
        if mode == "auto":
            titles = {c.get("patent_id"): (c.get("title") or "").lower() for c in chunks}
            distinct = len(set(titles.values()))
            mode = "known-item" if distinct >= 0.9 * len(titles) else "theme"
            logger.info("query-mode auto -> %s (%d distinct titles / %d patents)",
                        mode, distinct, len(titles))
        queries = (generate_known_item_queries(chunks, args.target_queries)
                   if mode == "known-item"
                   else generate_queries(chunks, args.target_queries))
        qpath = Path(args.queries)
        qpath.parent.mkdir(parents=True, exist_ok=True)
        with open(qpath, "w", encoding="utf-8") as f:
            json.dump({"queries": queries}, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(queries)} queries to {qpath}")
        print("Review and replace the auto-labels with real relevance "
              "judgements before trusting the metrics.")
        return 0

    qpath = Path(args.queries)
    if not qpath.exists():
        logger.error("Query set not found: %s. Run --generate-queries first.", qpath)
        return 1
    with open(qpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", data if isinstance(data, list) else [])
    if not queries:
        logger.error("No queries in %s", qpath)
        return 1
    logger.info("Loaded %d evaluation queries", len(queries))

    wanted = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    results = []
    notes = []

    bm25_index = None
    if {"bm25", "hybrid"} & set(wanted):
        bm25_index = get_bm25_index()

    store = embedder = None
    if {"vector", "hybrid"} & set(wanted):
        try:
            from services.rag_engine.embedder import get_embedder
            from services.rag_engine.vector_store import VectorStore
            embedder = get_embedder()
            store = VectorStore()
            if not store.collection_exists():
                raise RuntimeError(
                    "Qdrant collection missing - run scripts/ingest_chunks.py"
                )
            import os
            if os.getenv("EMBEDDING_BACKEND", "").lower() == "stub":
                notes.append(
                    "EMBEDDING_BACKEND=stub: vector and hybrid numbers are "
                    "NOT meaningful (hash vectors, no semantics)."
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector retrieval unavailable: %s", exc)
            notes.append(f"vector/hybrid skipped: {exc}")
            wanted = [r for r in wanted if r not in ("vector", "hybrid")]

    def _hold_out(hits):
        return [h for h in hits if h.get("section") != "abstract"] if args.hold_out_abstract else hits

    if args.hold_out_abstract:
        notes.append("--hold-out-abstract: abstract chunks excluded from results; "
                     "queries must find patents via claims/description text.")

    if "vector" in wanted:
        results.append(evaluate_retriever(
            "vector",
            lambda q, k: _hold_out(store.search(embedder.encode_query(q), top_k=k)),
            queries, k_values,
        ))

    if "bm25" in wanted:
        results.append(evaluate_retriever(
            "bm25",
            lambda q, k: _hold_out(bm25_index.search(q, top_k=k)),
            queries, k_values,
        ))

    if "hybrid" in wanted:
        results.append(evaluate_retriever(
            "hybrid_rrf",
            lambda q, k: _hold_out(hybrid_search(
                q, store, embedder, bm25_index,
                top_k=k, candidate_k=max(args.candidate_k, k),
            )["results"]),
            queries, k_values,
        ))

    if not results:
        logger.error("No retrievers could be evaluated.")
        return 1

    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": timestamp,
        "query_count": len(queries),
        "k_values": k_values,
        "notes": notes,
        "results": results,
    }

    json_path = out_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    table = format_comparison(results, k_values)
    md_path = out_dir / "evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation\n\n")
        f.write(f"Generated: {timestamp}\n\n")
        f.write(f"Queries: {len(queries)}\n\n")
        if notes:
            f.write("## Caveats\n\n")
            for n in notes:
                f.write(f"- {n}\n")
            f.write("\n")
        f.write("## Comparison\n\n")
        f.write(table + "\n")

    print("\n=== Part 8 evaluation ===")
    print(f"queries: {len(queries)}")
    for n in notes:
        print(f"CAVEAT: {n}")
    print()
    print(table)
    print(f"\nwritten to: {json_path}\n            {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
