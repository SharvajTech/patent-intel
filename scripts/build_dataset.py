#!/usr/bin/env python3
"""
Patent Intel Agent - Part 3 CLI

Build a clean, chunked patent corpus and write it to patent_chunks.json.

Usage examples
--------------
  # Synthetic sample corpus (no network / no API key needed) - FOR TESTING
  python scripts/build_dataset.py --source sample --limit 300

  # Your own dump (JSON / JSONL / CSV / TSV)
  python scripts/build_dataset.py --source local --input data/raw/patents.json

  # PatentsView Search API (needs a free API key)
  python scripts/build_dataset.py --source patentsview --limit 300 \
      --api-key "$PATENTSVIEW_API_KEY"

  # HUPD (Harvard USPTO Patent Dataset) via HuggingFace datasets
  python scripts/build_dataset.py --source hupd --limit 300
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Make the project root importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.document_parser.chunker import (  # noqa: E402
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    MIN_CHUNK_WORDS,
    chunk_corpus,
)
from services.document_parser.dataset import (  # noqa: E402
    DatasetError,
    build_corpus,
    load_local,
)

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("build_dataset")


def fetch_patentsview(limit: int, api_key: str):
    """
    Fetch granted patents from the PatentsView Search API.

    NOTE: title + abstract come from /g_patent/; claims come from /g_claim/.
    Full description text is only available via PatentsView bulk downloads,
    so description will usually be empty from this source.
    """
    import requests  # imported lazily so the base image stays slim

    if not api_key:
        raise DatasetError(
            "PatentsView requires an API key. Request one free at "
            "https://patentsview.org/apis/keyrequest and pass --api-key."
        )

    base = "https://search.patentsview.org/api/v1"
    headers = {"X-Api-Key": api_key}
    records, page, per_page = [], 1, min(limit, 100)

    while len(records) < limit:
        params = {
            "q": json.dumps({"_gte": {"patent_date": "2015-01-01"}}),
            "f": json.dumps([
                "patent_id", "patent_title", "patent_abstract", "patent_date",
            ]),
            "o": json.dumps({"size": per_page, "page": page}),
        }
        resp = requests.get(f"{base}/patent/", headers=headers,
                            params=params, timeout=30)
        if resp.status_code != 200:
            raise DatasetError(
                f"PatentsView returned {resp.status_code}: {resp.text[:300]}"
            )
        batch = resp.json().get("patents") or []
        if not batch:
            break
        records.extend(batch)
        page += 1

    records = records[:limit]

    # Enrich with claims, one request per patent.
    for rec in records:
        pid = rec.get("patent_id")
        if not pid:
            continue
        try:
            cparams = {
                "q": json.dumps({"patent_id": pid}),
                "f": json.dumps(["claim_sequence", "claim_text"]),
                "o": json.dumps({"size": 100}),
            }
            cresp = requests.get(f"{base}/g_claim/", headers=headers,
                                 params=cparams, timeout=30)
            if cresp.status_code == 200:
                claims = cresp.json().get("g_claims") or []
                claims.sort(key=lambda c: c.get("claim_sequence") or 0)
                rec["claims"] = "\n\n".join(
                    f"{(c.get('claim_sequence') or 0) + 1}. {c.get('claim_text','')}"
                    for c in claims if c.get("claim_text")
                )
        except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
            logger.warning("Claim fetch failed for %s: %s", pid, exc)

    return records


def fetch_hupd(limit: int):
    """
    Load records from HUPD (Harvard USPTO Patent Dataset) via HuggingFace.
    This source has abstract, claims AND full description text.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetError(
            "The 'datasets' package is required for --source hupd. "
            "Install it with: pip install datasets"
        ) from exc

    ds = load_dataset(
        "HUPD/hupd",
        name="sample",
        split="train",
        trust_remote_code=True,
    )
    records = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        records.append({
            "patent_id": row.get("patent_number") or row.get("application_number"),
            "title": row.get("title"),
            "abstract": row.get("abstract"),
            "description": row.get("description"),
            "claims": row.get("claims"),
            "date": row.get("date_published"),
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a chunked patent corpus.")
    ap.add_argument(
        "--source",
        choices=["sample", "local", "patentsview", "hupd"],
        default="sample",
        help="Where to read raw patent records from.",
    )
    ap.add_argument("--input", help="Input file path (for --source local).")
    ap.add_argument("--api-key", default="", help="API key (for --source patentsview).")
    ap.add_argument("--limit", type=int, default=300,
                    help="Max raw records to ingest (target 200-500).")
    ap.add_argument("--output", default="data/processed/patent_chunks.json",
                    help="Output path for the chunked corpus.")
    ap.add_argument("--records-output", default="data/processed/patent_records.json",
                    help="Also write the cleaned (pre-chunk) records here.")
    ap.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)
    ap.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)
    ap.add_argument("--min-chunk-words", type=int, default=MIN_CHUNK_WORDS)
    ap.add_argument("--seed", type=int, default=42, help="Seed for --source sample.")
    args = ap.parse_args()

    # ---- 1. Load raw records ------------------------------------------------
    try:
        if args.source == "sample":
            from services.document_parser.sample_source import generate_sample_records
            logger.warning(
                "Using SYNTHETIC sample data - these are NOT real patents. "
                "Use --source local/patentsview/hupd for real data."
            )
            raw = generate_sample_records(count=args.limit, seed=args.seed)
        elif args.source == "local":
            if not args.input:
                logger.error("--source local requires --input <file>")
                return 2
            raw = load_local(args.input)
        elif args.source == "patentsview":
            raw = fetch_patentsview(args.limit, args.api_key)
        else:
            raw = fetch_hupd(args.limit)
    except DatasetError as exc:
        logger.error("Failed to load source: %s", exc)
        return 1

    logger.info("Loaded %d raw records from source '%s'", len(raw), args.source)

    # ---- 2. Clean + validate + deduplicate ----------------------------------
    clean_records, stats = build_corpus(raw)

    if not clean_records:
        logger.error("No usable records after cleaning. Aborting.")
        return 1

    # ---- 3. Chunk -----------------------------------------------------------
    chunks = chunk_corpus(
        clean_records,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        min_chunk_words=args.min_chunk_words,
    )

    # ---- 4. Write -----------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    section_counts: dict = {}
    for c in chunks:
        section_counts[c["section"]] = section_counts.get(c["section"], 0) + 1

    payload = {
        "meta": {
            "source": args.source,
            "synthetic": args.source == "sample",
            "record_count": len(clean_records),
            "chunk_count": len(chunks),
            "chunk_words": args.chunk_words,
            "overlap_words": args.overlap_words,
            "sections": section_counts,
            "cleaning_stats": stats,
        },
        "chunks": chunks,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.records_output:
        rec_path = Path(args.records_output)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(clean_records, f, ensure_ascii=False, indent=2)

    # ---- 5. Report ----------------------------------------------------------
    print("\n=== Part 3 corpus build ===")
    print(f"source              : {args.source}"
          f"{'  (SYNTHETIC - not real patents)' if args.source == 'sample' else ''}")
    print(f"raw records in      : {stats['input_records']}")
    print(f"unparseable         : {stats['unparseable']}")
    print(f"rejected (quality)  : {sum(stats['rejected'].values())} {stats['rejected']}")
    print(f"duplicate IDs       : {stats['duplicate_id']}")
    print(f"duplicate content   : {stats['duplicate_content']}")
    print(f"clean records out   : {stats['clean_records']}")
    print(f"chunks produced     : {len(chunks)}")
    print(f"chunks per section  : {section_counts}")
    if chunks:
        avg = sum(c["word_count"] for c in chunks) / len(chunks)
        print(f"avg chunk words     : {avg:.1f}")
    print(f"written to          : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
