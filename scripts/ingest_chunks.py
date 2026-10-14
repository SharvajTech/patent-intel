#!/usr/bin/env python3
"""
Patent Intel Agent - Part 4 ingest CLI

Reads patent_chunks.json, embeds each chunk with BGE-M3, and upserts the
vectors + metadata into Qdrant.

Usage
-----
  # inside the api container (Qdrant reachable as host 'qdrant')
  python scripts/ingest_chunks.py --recreate

  # smoke test without downloading the 2GB model
  EMBEDDING_BACKEND=stub python scripts/ingest_chunks.py --recreate
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.rag_engine.embedder import (  # noqa: E402
    EmbedderError,
    get_embedder,
)
from services.rag_engine.vector_store import (  # noqa: E402
    VectorStore,
    VectorStoreError,
)

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("ingest_chunks")


def load_chunks(path: Path):
    if not path.exists():
        raise SystemExit(
            f"Chunk file not found: {path}\n"
            f"Run Part 3 first:  python scripts/build_dataset.py --source sample"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("chunks", []), data.get("meta", {})
    if isinstance(data, list):
        return data, {}
    raise SystemExit(f"Unrecognized chunk file structure: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed chunks into Qdrant.")
    ap.add_argument("--input", default="data/processed/patent_chunks.json")
    ap.add_argument("--recreate", action="store_true",
                    help="Drop and recreate the collection first.")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Embedding batch size.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only ingest the first N chunks (0 = all).")
    args = ap.parse_args()

    chunks, meta = load_chunks(Path(args.input))
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        logger.error("No chunks to ingest.")
        return 1

    if meta.get("synthetic"):
        logger.warning("Corpus is SYNTHETIC sample data - not real patents.")

    logger.info("Loaded %d chunks from %s", len(chunks), args.input)

    embedder = get_embedder()
    try:
        logger.info("Loading embedding model (first run downloads ~2GB)...")
        t0 = time.time()
        embedder.warmup()
        logger.info("Model ready in %.1fs", time.time() - t0)
    except EmbedderError as exc:
        logger.error("Embedder failed: %s", exc)
        return 1

    try:
        store = VectorStore()
        store.ensure_collection(dim=embedder.dim, recreate=args.recreate)
    except VectorStoreError as exc:
        logger.error("Qdrant error: %s", exc)
        return 1

    texts = [c["text"] for c in chunks]
    t0 = time.time()
    try:
        vectors = embedder.encode(
            texts, batch_size=args.batch_size, show_progress=True
        )
    except EmbedderError as exc:
        logger.error("Encoding failed: %s", exc)
        return 1
    encode_secs = time.time() - t0
    logger.info("Encoded %d chunks in %.1fs", len(vectors), encode_secs)

    try:
        written = store.upsert_chunks(chunks, vectors)
        info = store.info()
    except VectorStoreError as exc:
        logger.error("Upsert failed: %s", exc)
        return 1

    print("\n=== Part 4 ingest ===")
    print(f"chunks embedded : {len(vectors)}")
    print(f"vector dim      : {embedder.dim}")
    print(f"encode time     : {encode_secs:.1f}s")
    print(f"points upserted : {written}")
    print(f"collection      : {info['collection']} ({info['points']} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
