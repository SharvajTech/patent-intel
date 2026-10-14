"""
Patent Intel Agent - Document Parser (Part 2)

PDF parsing only: metadata, per-page text, and embedded image extraction.
No chunking, embeddings, RAG, BM25, or LLM logic here.
"""
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

logger = logging.getLogger("patent-intel-agent.document_parser")

# Where extracted diagram/figure images get saved.
DIAGRAMS_DIR = Path(os.getenv("DIAGRAMS_DIR", "static/diagrams"))


class PDFParseError(Exception):
    """Raised when a PDF cannot be opened or parsed."""


def _extract_metadata(doc: "fitz.Document") -> Dict[str, Any]:
    meta = doc.metadata or {}
    return {
        "title": meta.get("title") or None,
        "author": meta.get("author") or None,
        "subject": meta.get("subject") or None,
        "keywords": meta.get("keywords") or None,
        "creator": meta.get("creator") or None,
        "producer": meta.get("producer") or None,
        "creation_date": meta.get("creationDate") or None,
        "mod_date": meta.get("modDate") or None,
        "page_count": doc.page_count,
        "is_encrypted": doc.is_encrypted,
    }


def _extract_images_for_page(
    doc: "fitz.Document",
    page: "fitz.Page",
    page_number: int,
    doc_stem: str,
    output_dir: Path,
) -> List[str]:
    """
    Extract embedded raster images for a single page and save them under
    output_dir. Returns a list of saved file paths (relative, forward-slash).
    """
    saved_paths: List[str] = []
    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list, start=1):
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
        except Exception as exc:
            logger.warning(
                "Failed to extract image xref=%s on page=%s: %s",
                xref, page_number, exc,
            )
            continue

        image_bytes = base_image.get("image")
        ext = base_image.get("ext", "png")
        if not image_bytes:
            continue

        filename = f"{doc_stem}_p{page_number}_img{img_index}_{uuid.uuid4().hex[:8]}.{ext}"
        out_path = output_dir / filename
        try:
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            saved_paths.append(str(out_path).replace(os.sep, "/"))
        except OSError as exc:
            logger.warning("Failed to save image %s: %s", out_path, exc)
            continue

    return saved_paths


def parse_pdf(file_path: str, output_dir: Path = DIAGRAMS_DIR) -> Dict[str, Any]:
    """
    Parse a PDF file: extract document metadata, per-page text, and
    embedded images (saved to output_dir).

    Raises PDFParseError on any failure to open/read the file.
    """
    path = Path(file_path)
    if not path.exists():
        raise PDFParseError(f"File not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise PDFParseError(f"Not a PDF file: {file_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_stem = path.stem

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise PDFParseError(f"Failed to open PDF: {exc}") from exc

    try:
        if doc.is_encrypted:
            # Try an empty-password unlock; if it fails, we can't read content.
            if not doc.authenticate(""):
                raise PDFParseError("PDF is encrypted/password-protected; cannot parse.")

        metadata = _extract_metadata(doc)

        pages: List[Dict[str, Any]] = []
        all_image_paths: List[str] = []

        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)
            try:
                text = page.get_text("text")
            except Exception as exc:
                logger.warning("Text extraction failed on page %s: %s", page_number + 1, exc)
                text = ""

            try:
                image_paths = _extract_images_for_page(
                    doc, page, page_number + 1, doc_stem, output_dir
                )
            except Exception as exc:
                logger.warning("Image extraction failed on page %s: %s", page_number + 1, exc)
                image_paths = []

            all_image_paths.extend(image_paths)

            pages.append(
                {
                    "page_number": page_number + 1,
                    "text": text,
                    "char_count": len(text),
                    "images": image_paths,
                }
            )

        full_text = "\n\n".join(p["text"] for p in pages)

        result = {
            "filename": path.name,
            "metadata": metadata,
            "page_count": doc.page_count,
            "pages": pages,
            "text": full_text,
            "images": all_image_paths,
            "image_count": len(all_image_paths),
        }
        return result

    finally:
        doc.close()
