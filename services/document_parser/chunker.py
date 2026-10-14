"""
Patent Intel Agent - Chunking (Part 3)

Splits cleaned patent sections into retrieval-sized chunks and attaches
metadata (patent_id, section, chunk_id). No embeddings here.
"""
import re
from typing import Any, Dict, Iterable, List

# Section names we chunk, in a stable order.
SECTIONS = ("abstract", "description", "claims")

DEFAULT_CHUNK_WORDS = 220
DEFAULT_OVERLAP_WORDS = 40
# Chunks shorter than this are dropped as noise (headings, stray fragments).
MIN_CHUNK_WORDS = 20

# "1. A system comprising..." / "12 . The method of claim 1..."
_CLAIM_START_RE = re.compile(r"(?m)^\s*(\d{1,3})\s*[\.\)]\s+(?=[A-Z(])")


def split_words(text: str) -> List[str]:
    return text.split()


def sliding_window_chunks(
    text: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> List[str]:
    """
    Split text into overlapping word windows. Overlap preserves context that
    would otherwise be cut mid-sentence at a chunk boundary.
    """
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    words = split_words(text)
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]

    step = chunk_words - overlap_words
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks


def split_claims(claims_text: str) -> List[str]:
    """
    Split a claims block into individual claims. Falls back to returning the
    whole block if no numbered claim boundaries are found.
    """
    if not claims_text.strip():
        return []

    matches = list(_CLAIM_START_RE.finditer(claims_text))
    if not matches:
        return [claims_text.strip()]

    claims: List[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(claims_text)
        claim = claims_text[start:end].strip()
        if claim:
            claims.append(claim)
    return claims


def _emit(
    patent_id: str,
    section: str,
    index: int,
    text: str,
    title: str,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    chunk_id = f"{patent_id}::{section}::{index:04d}"
    words = split_words(text)
    chunk = {
        "chunk_id": chunk_id,
        "patent_id": patent_id,
        "section": section,
        "chunk_index": index,
        "text": text,
        "title": title,
        "word_count": len(words),
        "char_count": len(text),
    }
    chunk.update(extra)
    return chunk


def chunk_patent(
    record: Dict[str, Any],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> List[Dict[str, Any]]:
    """
    Turn one cleaned patent record into a list of chunk dicts.

    Sections are chunked independently so a chunk never straddles the
    abstract/description/claims boundary. Claims are split per-claim first,
    since an individual claim is the natural retrieval unit.
    """
    patent_id = record.get("patent_id", "")
    title = record.get("title", "")
    chunks: List[Dict[str, Any]] = []

    for section in SECTIONS:
        body = (record.get(section) or "").strip()
        if not body:
            continue

        if section == "claims":
            units = split_claims(body)
        else:
            units = [body]

        index = 0
        for unit_pos, unit in enumerate(units):
            for piece in sliding_window_chunks(unit, chunk_words, overlap_words):
                if len(split_words(piece)) < min_chunk_words:
                    continue
                extra: Dict[str, Any] = {}
                if section == "claims":
                    # Record which claim this chunk came from, when detectable.
                    m = re.match(r"\s*(\d{1,3})\s*[\.\)]", unit)
                    extra["claim_number"] = int(m.group(1)) if m else unit_pos + 1
                chunks.append(_emit(patent_id, section, index, piece, title, extra))
                index += 1

    return chunks


def chunk_corpus(
    records: Iterable[Dict[str, Any]],
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    min_chunk_words: int = MIN_CHUNK_WORDS,
) -> List[Dict[str, Any]]:
    """Chunk an iterable of cleaned patent records."""
    out: List[Dict[str, Any]] = []
    for record in records:
        out.extend(
            chunk_patent(record, chunk_words, overlap_words, min_chunk_words)
        )
    return out
