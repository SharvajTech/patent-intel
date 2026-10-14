"""
Patent Intel Agent - Text cleaning / normalization (Part 3)

Pure functions for turning messy raw patent text into clean, comparable text.
No I/O here.
"""
import hashlib
import html
import re
import unicodedata
from typing import Optional

# Matches XML/HTML tags commonly found in USPTO/EPO full-text dumps.
_TAG_RE = re.compile(r"<[^>]+>")
# Common boilerplate that adds no retrieval value.
_BOILERPLATE_RE = re.compile(
    r"^\s*(cross[- ]reference to related applications?|"
    r"statement regarding federally sponsored research|"
    r"incorporation by reference|"
    r"brief description of the drawings?)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_WS_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
# Figure/reference-numeral noise, e.g. "(FIG. 3)" or bare "101a".
_FIG_RE = re.compile(r"\(\s*fig(?:ure)?s?\.?\s*[\d,\s\-a-z]+\)", re.IGNORECASE)


def strip_markup(text: str) -> str:
    """Remove XML/HTML tags, keeping the inner text."""
    if not text:
        return ""
    return _TAG_RE.sub(" ", text)


def normalize_text(text: Optional[str]) -> str:
    """
    Normalize unicode, collapse whitespace, strip markup and figure noise.
    Preserves paragraph breaks (double newline) so chunking can respect them.
    """
    if not text:
        return ""

    # Decode entities (&nbsp; &amp; &#8212;) before anything else, so NFKC can
    # then fold the resulting characters (e.g. NBSP -> plain space).
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = strip_markup(text)
    text = _FIG_RE.sub(" ", text)
    text = _BOILERPLATE_RE.sub("", text)

    # Normalize line endings, collapse horizontal whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    # Trim each line, then collapse runs of blank lines to a single break.
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NL_RE.sub("\n\n", text)

    return text.strip()


def normalize_patent_id(raw_id: Optional[str]) -> str:
    """
    Normalize a patent number for use as a stable key.
    'US 10,123,456 B2' -> 'US10123456B2'
    """
    if not raw_id:
        return ""
    pid = unicodedata.normalize("NFKC", str(raw_id)).upper()
    pid = re.sub(r"[^A-Z0-9]", "", pid)
    return pid


def content_fingerprint(title: str, abstract: str) -> str:
    """
    Stable hash of title+abstract, used to catch near-identical records that
    carry different patent IDs (e.g. continuations, republished families).
    """
    basis = f"{title.lower().strip()}|{abstract.lower().strip()}"
    basis = _WS_RE.sub(" ", basis)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()
