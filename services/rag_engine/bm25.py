"""
Patent Intel Agent - BM25 keyword search (Part 5)

Pure-Python BM25 Okapi implementation over the patent chunk corpus.
Implemented in-project rather than pulling rank_bm25 to keep dependencies
minimal, and because we need chunk metadata alongside scores.

The index is built once from patent_chunks.json and cached process-wide.
"""
import json
import logging
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("patent-intel-agent.bm25")

CHUNKS_PATH = Path(
    os.getenv("CHUNKS_PATH", "data/processed/patent_chunks.json")
)

# Standard BM25 parameters. k1 controls term-frequency saturation,
# b controls document-length normalization.
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal stopword list. Patent prose is full of these and they add noise;
# deliberately kept short so domain terms are never dropped.
_STOPWORDS = frozenset("""
a an the and or of to in for with at by on is are be been being as that this
these those which such wherein whereby said thereof therein from into other
one two first second least such may can will would could
""".split())


class BM25Error(Exception):
    """Raised when the BM25 index cannot be built or queried."""


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and 1-char tokens."""
    if not text:
        return []
    return [
        tok for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    ]


class BM25Index:
    """In-memory BM25 Okapi index over a list of chunk dicts."""

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.doc_freqs: List[Counter] = []
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.idf: Dict[str, float] = {}
        # term -> list of document indices containing it
        self.postings: Dict[str, List[int]] = {}

    def build(self, chunks: Sequence[Dict[str, Any]]) -> "BM25Index":
        """Build the index. Indexes chunk text plus its patent title."""
        if not chunks:
            raise BM25Error("Cannot build BM25 index from an empty corpus.")

        self.chunks = list(chunks)
        self.doc_freqs = []
        self.doc_lens = []
        self.postings = {}
        doc_count_per_term: Counter = Counter()

        for idx, chunk in enumerate(self.chunks):
            # Title is included so keyword hits on the invention name count.
            text = f"{chunk.get('title', '')} {chunk.get('text', '')}"
            tokens = tokenize(text)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            self.doc_lens.append(len(tokens))
            for term in freqs:
                doc_count_per_term[term] += 1
                self.postings.setdefault(term, []).append(idx)

        n_docs = len(self.chunks)
        self.avg_doc_len = (sum(self.doc_lens) / n_docs) if n_docs else 0.0

        # Robertson/Sparck-Jones IDF with +1 smoothing to keep it non-negative.
        self.idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_count_per_term.items()
        }

        logger.info(
            "BM25 index built: %d chunks, %d unique terms, avg len %.1f",
            n_docs, len(self.idf), self.avg_doc_len,
        )
        return self

    def search(
        self,
        query: str,
        top_k: int = 5,
        section: Optional[str] = None,
        patent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the top-K chunks by BM25 score, with optional filtering."""
        if not self.chunks:
            raise BM25Error("BM25 index is empty - build it first.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Only score documents that contain at least one query term.
        candidates: set = set()
        for term in query_tokens:
            candidates.update(self.postings.get(term, ()))
        if not candidates:
            return []

        scores: Dict[int, float] = {}
        for idx in candidates:
            chunk = self.chunks[idx]
            if section and chunk.get("section") != section:
                continue
            if patent_id and chunk.get("patent_id") != patent_id:
                continue

            freqs = self.doc_freqs[idx]
            doc_len = self.doc_lens[idx]
            norm = self.k1 * (
                1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0))
            )
            score = 0.0
            for term in query_tokens:
                tf = freqs.get(term, 0)
                if not tf:
                    continue
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / (tf + norm)
            if score > 0:
                scores[idx] = score

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            chunk = self.chunks[idx]
            hit = {
                "chunk_id": chunk.get("chunk_id"),
                "patent_id": chunk.get("patent_id"),
                "section": chunk.get("section"),
                "score": round(float(score), 6),
                "text": chunk.get("text", ""),
                "title": chunk.get("title", ""),
            }
            if "claim_number" in chunk:
                hit["claim_number"] = chunk["claim_number"]
            results.append(hit)
        return results

    @property
    def size(self) -> int:
        return len(self.chunks)


def load_chunks(path: Path = CHUNKS_PATH) -> List[Dict[str, Any]]:
    """Load the chunk corpus produced by Part 3."""
    if not path.exists():
        raise BM25Error(
            f"Chunk corpus not found at {path}. "
            f"Run scripts/build_dataset.py first."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise BM25Error(f"Invalid JSON in {path}: {exc}") from exc

    chunks = data.get("chunks", []) if isinstance(data, dict) else data
    if not chunks:
        raise BM25Error(f"No chunks found in {path}.")
    return chunks


_index: Optional[BM25Index] = None
_lock = threading.Lock()


def get_bm25_index(path: Path = CHUNKS_PATH, force_reload: bool = False) -> BM25Index:
    """Process-wide lazily-built BM25 index singleton."""
    global _index
    if _index is not None and not force_reload:
        return _index
    with _lock:
        if _index is None or force_reload:
            _index = BM25Index().build(load_chunks(path))
    return _index
