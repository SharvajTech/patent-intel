"""
Patent Intel Agent - Dataset assembly (Part 3)

Loads raw patent records from a source, validates and cleans them,
removes duplicates, and produces a normalized corpus.

Sources are pluggable so the same cleaning/dedup path serves real bulk data
and small local test files alike.
"""
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from services.document_parser.cleaning import (
    content_fingerprint,
    normalize_patent_id,
    normalize_text,
)

logger = logging.getLogger("patent-intel-agent.dataset")

# A record must have an ID, a title, and enough abstract text to be useful.
MIN_ABSTRACT_WORDS = 10
MIN_TITLE_CHARS = 5

# Field aliases seen across PatentsView / USPTO / EPO / HUPD exports.
FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "patent_id": (
        "patent_id", "patent_number", "publication_number", "doc_number",
        "id", "patent_num", "application_number",
    ),
    "title": (
        "title", "patent_title", "invention_title", "patent_abstract_title",
    ),
    "abstract": ("abstract", "patent_abstract", "abstract_text"),
    "description": (
        "description", "detailed_description", "patent_description",
        "description_text", "full_description",
    ),
    "claims": ("claims", "claim", "claims_text", "patent_claims"),
    "date": (
        "date", "patent_date", "publication_date", "grant_date", "filing_date",
    ),
    "assignee": ("assignee", "assignee_organization", "applicant", "assignees"),
}


class DatasetError(Exception):
    """Raised when a dataset source cannot be read."""


def _pick_field(raw: Dict[str, Any], canonical: str) -> Any:
    """Return the first matching alias present in the raw record."""
    for alias in FIELD_ALIASES.get(canonical, (canonical,)):
        if alias in raw and raw[alias] not in (None, ""):
            return raw[alias]
    return None


def _coerce_text(value: Any) -> str:
    """
    Flatten a field into plain text. Handles the list/dict shapes that
    APIs commonly return for claims and assignees.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                # Pull the most text-like value out of the dict.
                for key in ("text", "claim_text", "value", "name",
                            "assignee_organization"):
                    if item.get(key):
                        parts.append(str(item[key]))
                        break
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        for key in ("text", "value", "name"):
            if value.get(key):
                return str(value[key])
        return ""
    return str(value)


def normalize_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map a raw source record onto the canonical schema and clean its text.
    Returns None if the record is unusable.
    """
    patent_id = normalize_patent_id(_coerce_text(_pick_field(raw, "patent_id")))
    if not patent_id:
        return None

    record = {
        "patent_id": patent_id,
        "title": normalize_text(_coerce_text(_pick_field(raw, "title"))),
        "abstract": normalize_text(_coerce_text(_pick_field(raw, "abstract"))),
        "description": normalize_text(_coerce_text(_pick_field(raw, "description"))),
        "claims": normalize_text(_coerce_text(_pick_field(raw, "claims"))),
        "date": _coerce_text(_pick_field(raw, "date")).strip() or None,
        "assignee": normalize_text(_coerce_text(_pick_field(raw, "assignee"))) or None,
    }
    return record


def is_valid_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """Check a normalized record against minimum quality rules."""
    if not record.get("patent_id"):
        return False, "missing_patent_id"
    if len(record.get("title", "")) < MIN_TITLE_CHARS:
        return False, "title_too_short"

    abstract_words = len(record.get("abstract", "").split())
    body_words = len(record.get("description", "").split()) + len(
        record.get("claims", "").split()
    )
    # Needs either a real abstract or some body text to be worth indexing.
    if abstract_words < MIN_ABSTRACT_WORDS and body_words < MIN_ABSTRACT_WORDS:
        return False, "insufficient_text"

    return True, "ok"


def deduplicate(
    records: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Remove duplicates in two passes:
      1. exact duplicate patent_id (keep the richest version)
      2. identical title+abstract fingerprint under different IDs

    'Richest' = most total characters across abstract/description/claims,
    so a fuller record wins over a stub with the same ID.
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    stats = {"duplicate_id": 0, "duplicate_content": 0}

    def richness(rec: Dict[str, Any]) -> int:
        return sum(len(rec.get(f, "")) for f in ("abstract", "description", "claims"))

    for rec in records:
        pid = rec["patent_id"]
        if pid in by_id:
            stats["duplicate_id"] += 1
            if richness(rec) > richness(by_id[pid]):
                by_id[pid] = rec
        else:
            by_id[pid] = rec

    seen_fingerprints: Dict[str, str] = {}
    deduped: List[Dict[str, Any]] = []
    for rec in by_id.values():
        fp = content_fingerprint(rec.get("title", ""), rec.get("abstract", ""))
        # Empty abstracts all hash alike; only dedupe when there's real content.
        if rec.get("abstract"):
            if fp in seen_fingerprints:
                stats["duplicate_content"] += 1
                continue
            seen_fingerprints[fp] = rec["patent_id"]
        deduped.append(rec)

    return deduped, stats


def build_corpus(
    raw_records: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Full Part 3 cleaning pipeline: normalize -> validate -> deduplicate.
    Returns (clean_records, stats).
    """
    stats: Dict[str, Any] = {
        "input_records": 0,
        "unparseable": 0,
        "rejected": {},
        "duplicate_id": 0,
        "duplicate_content": 0,
        "clean_records": 0,
    }

    normalized: List[Dict[str, Any]] = []
    for raw in raw_records:
        stats["input_records"] += 1
        if not isinstance(raw, dict):
            stats["unparseable"] += 1
            continue
        rec = normalize_record(raw)
        if rec is None:
            stats["unparseable"] += 1
            continue
        ok, reason = is_valid_record(rec)
        if not ok:
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
            continue
        normalized.append(rec)

    deduped, dup_stats = deduplicate(normalized)
    stats["duplicate_id"] = dup_stats["duplicate_id"]
    stats["duplicate_content"] = dup_stats["duplicate_content"]
    stats["clean_records"] = len(deduped)

    return deduped, stats


# --------------------------------------------------------------------------
# Source loaders
# --------------------------------------------------------------------------

def load_json_file(path: Path) -> List[Dict[str, Any]]:
    """Load records from a .json (list or {'records': [...]}) or .jsonl file."""
    if not path.exists():
        raise DatasetError(f"File not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping bad JSONL line %s: %s", line_no, exc)
        return records

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid JSON in {path}: {exc}") from exc

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "patents", "data", "results"):
            if isinstance(data.get(key), list):
                return data[key]
    raise DatasetError(f"Unrecognized JSON structure in {path}")


def load_csv_file(path: Path) -> List[Dict[str, Any]]:
    """Load records from a CSV/TSV file."""
    if not path.exists():
        raise DatasetError(f"File not found: {path}")
    delimiter = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def load_local(path: str) -> List[Dict[str, Any]]:
    """Dispatch to the right loader based on file extension."""
    p = Path(path)
    if p.suffix.lower() in (".json", ".jsonl"):
        return load_json_file(p)
    if p.suffix.lower() in (".csv", ".tsv", ".tab"):
        return load_csv_file(p)
    raise DatasetError(
        f"Unsupported file type '{p.suffix}'. Use .json, .jsonl, .csv or .tsv."
    )
